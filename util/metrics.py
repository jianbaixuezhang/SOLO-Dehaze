from __future__ import annotations

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Reduction = Union[str, None]

_ssim_window_cache: Dict[Tuple[int, float, int, str, str], torch.Tensor] = {}
_lpips_cache: Dict[Tuple[str, str, str], nn.Module] = {}


def _reduction_apply(x: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    if reduction == "none" or reduction is None:
        return x
    raise ValueError("reduction must be 'mean', 'sum', or 'none'")


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-8,
    reduction: Reduction = "mean",
) -> torch.Tensor:

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shapes, got {pred.shape} and {target.shape}")
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(eps)
    out = 10.0 * torch.log10((data_range ** 2) / mse)
    return _reduction_apply(out, reduction)


def _gaussian_window_2d(
    window_size: int,
    sigma: float,
    channels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (window_size, sigma, channels, str(device), str(dtype))
    if key in _ssim_window_cache:
        return _ssim_window_cache[key]
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    _ssim_window_cache[key] = window
    return window


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shapes, got {pred.shape} and {target.shape}")
    if pred.dim() != 4:
        raise ValueError("ssim requires pred/target shape (N, C, H, W)")
    _, c, h, w = pred.shape
    if h < window_size or w < window_size:
        pad_h = max(window_size - h, 0)
        pad_w = max(window_size - w, 0)
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
        pred = F.pad(pred, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        target = F.pad(target, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    window = _gaussian_window_2d(window_size, sigma, c, pred.device, pred.dtype)

    mu_x = F.conv2d(pred, window, padding=window_size // 2, groups=c)
    mu_y = F.conv2d(target, window, padding=window_size // 2, groups=c)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=c) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=c) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, window, padding=window_size // 2, groups=c) - mu_xy

    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2).clamp_min(1e-12)
    )
    ssim_per_channel = ssim_map.mean(dim=1)
    per_image = ssim_per_channel.mean(dim=(1, 2))
    return _reduction_apply(per_image, reduction)


def ciede2000(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    try:
        from skimage.color import deltaE_ciede2000, rgb2lab
    except ImportError as e:
        raise ImportError("CIEDE2000 requires scikit-image: pip install scikit-image") from e

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shapes, got {pred.shape} and {target.shape}")

    orig_device = pred.device
    pred_cpu = pred.detach().float().cpu()
    target_cpu = target.detach().float().cpu()
    scale = 1.0 / float(data_range)
    pred_np = (pred_cpu * scale).clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()
    target_np = (target_cpu * scale).clamp(0.0, 1.0).permute(0, 2, 3, 1).numpy()

    def _rgb_to_lab(arr):
        try:
            return rgb2lab(arr, channel_axis=-1)
        except TypeError:
            return rgb2lab(arr)

    deltas = []
    for i in range(pred_np.shape[0]):
        lab1 = _rgb_to_lab(pred_np[i])
        lab2 = _rgb_to_lab(target_np[i])
        d = deltaE_ciede2000(lab1, lab2)
        deltas.append(torch.tensor(d.mean(), dtype=torch.float32))
    stacked = torch.stack(deltas, dim=0).to(orig_device)
    return _reduction_apply(stacked, reduction)


def _get_lpips_model(net: str, device: torch.device) -> nn.Module:
    key = (net, str(device), "v1")
    if key in _lpips_cache:
        return _lpips_cache[key]
    try:
        import lpips as lpips_lib
    except ImportError as e:
        raise ImportError("LPIPS requires lpips: pip install lpips") from e
    model = lpips_lib.LPIPS(net=net, verbose=False)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _lpips_cache[key] = model
    return model


def lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    reduction: Reduction = "mean",
) -> torch.Tensor:

    if pred.shape != target.shape:
        raise ValueError(f"pred and target must have identical shapes, got {pred.shape} and {target.shape}")

    device = pred.device
    model = _get_lpips_model(net, device)
    x = pred.clamp(0.0, 1.0) * 2.0 - 1.0
    y = target.clamp(0.0, 1.0) * 2.0 - 1.0
    if x.dtype != torch.float32:
        x = x.float()
    if y.dtype != torch.float32:
        y = y.float()

    with torch.no_grad():
        d = model(x, y)
    d = d.view(d.size(0), -1).mean(dim=1)
    return _reduction_apply(d, reduction)
