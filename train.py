from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ROOT = Path(__file__).resolve().parent
_rs = str(_ROOT)
if _rs not in sys.path:
    sys.path.insert(0, _rs)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

from models.ColorLoss import ycrcb_chroma_direction_loss
from models.ContrastiveDomainLoss import ContrastiveDomainLoss
from models.PhaseConsistencyLoss import PhaseConsistencyLoss
from models.PerceptualLoss import UnpairedContrastivePerceptualLoss
from models.SOLODehaze import SOLO_Dehaze

from data.dataset import PairedValDataset, TestDataset, UnpairedTrainDataset, collate_test_batch
from util.metrics import ciede2000, lpips, psnr, ssim

TrainConfig = Any


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_train_transforms_hazy(cfg: TrainConfig):
    deg = float(cfg.rotate_degrees)
    return transforms.Compose(
        [
            transforms.RandomRotation(
                deg,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=(128, 128, 128),
            ),
            transforms.RandomCrop((cfg.patch_size, cfg.patch_size), pad_if_needed=True),
            transforms.ToTensor(),
        ]
    )


def _build_train_transforms_clear(cfg: TrainConfig):
    return transforms.Compose(
        [
            transforms.RandomCrop((cfg.patch_size, cfg.patch_size), pad_if_needed=True),
            transforms.ToTensor(),
        ]
    )


def _build_eval_transforms():
    return transforms.Compose([transforms.ToTensor()])


def _lr_at_epoch(cfg: TrainConfig, epoch: int) -> float:
    if epoch < cfg.warmup_epochs:
        return cfg.lr * float(epoch + 1) / float(max(cfg.warmup_epochs, 1))
    denom = float(max(cfg.num_epochs - 1 - cfg.warmup_epochs, 1))
    t = (epoch - cfg.warmup_epochs) / denom
    t = min(max(t, 0.0), 1.0)
    cos_part = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cos_part


def _apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


def _feature_map_to_tokens(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    return feat.view(b, c, h * w).transpose(1, 2).contiguous()


def _ensure_dirs(cfg: TrainConfig) -> Tuple[Path, Path, Path]:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = cfg.results_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_root = cfg.results_dir / "val_runs"
    val_root.mkdir(parents=True, exist_ok=True)
    return ckpt_dir, val_root, cfg.results_dir


def _metric_tag_for_dirname(x: float) -> str:
    return f"{x:.4f}"


def _save_val_per_image_psnr_ssim_csv(
    path: Path,
    epoch_1based: int,
    per_image: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "filename", "psnr", "ssim"])
        for row in per_image:
            w.writerow(
                [
                    epoch_1based,
                    row.get("filename", ""),
                    f"{float(row['psnr']):.8f}",
                    f"{float(row['ssim']):.8f}",
                ]
            )


def _save_test_per_image_metrics_csv(path: Path, per_image: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "has_gt", "psnr", "ssim", "ciede2000", "lpips"])
        for r in per_image:
            def _cell(key: str) -> str:
                v = r.get(key)
                if v is None:
                    return ""
                if isinstance(v, bool):
                    return "1" if v else "0"
                return f"{float(v):.8f}"

            w.writerow(
                [
                    r.get("filename", ""),
                    "1" if r.get("has_gt") else "0",
                    _cell("psnr"),
                    _cell("ssim"),
                    _cell("ciede2000"),
                    _cell("lpips"),
                ]
            )


def build_modules(cfg: TrainConfig, device: torch.device):
    net = SOLO_Dehaze(
        num_levels=cfg.dehaze_num_levels,
        hidden_dim=cfg.dehaze_hidden_dim,
        level_dims=cfg.dehaze_level_dims,
        level_heads=cfg.dehaze_level_heads,
        base_dim=cfg.dehaze_base_dim,
        base_heads=cfg.dehaze_base_heads,
    ).to(device)
    phase_consistency = PhaseConsistencyLoss(loss_type="l1").to(device)
    perceptual = UnpairedContrastivePerceptualLoss().to(device)
    contrast_token_dim = 3 * (cfg.dehaze_num_levels + 1)
    domain = ContrastiveDomainLoss(
        contrast_token_dim,
        margin=cfg.contrast_global_margin,
        ema_momentum=cfg.contrast_global_ema_momentum,
    ).to(device)
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        weight_decay=cfg.weight_decay,
    )
    return net, phase_consistency, perceptual, domain, optimizer


def _forward_loss(
    cfg: TrainConfig,
    hazy: torch.Tensor,
    clear: torch.Tensor,
    net: nn.Module,
    phase_consistency: nn.Module,
    perceptual: nn.Module,
    domain: nn.Module,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dehazed_01, feat_d, feat_h = net.forward_with_features(hazy)
    dehazed_01 = dehazed_01.clamp(0.0, 1.0)

    lw = cfg.loss_weights
    pw = float(lw.phase)
    ppw = float(lw.perceptual)
    ctw = float(lw.contrast)
    cw = float(lw.color)

    zero = torch.zeros((), device=dehazed_01.device, dtype=dehazed_01.dtype)

    if pw > 0.0:
        loss_phase = phase_consistency(dehazed_01, hazy)
    else:
        loss_phase = zero

    if ppw > 0.0:
        loss_p = perceptual(hazy, dehazed_01)
    else:
        loss_p = zero

    if ctw > 0.0:
        _, feat_c, _ = net.forward_with_features(clear)
        z_d = _feature_map_to_tokens(feat_d)
        z_c = _feature_map_to_tokens(feat_c)
        z_h = _feature_map_to_tokens(feat_h)
        d_out = domain(
            z_d,
            z_c,
            z_h,
            update_ema=True,
        )
        loss_c = d_out["global"]
    else:
        loss_c = zero

    if cw > 0.0:
        loss_color = ycrcb_chroma_direction_loss(dehazed_01, hazy)
    else:
        loss_color = zero

    total = (
        ppw * loss_p
        + ctw * loss_c
        + cw * loss_color
        + pw * loss_phase
    )
    parts = {
        "total": total.detach(),
        "phase": loss_phase.detach(),
        "perceptual": loss_p.detach(),
        "contrast": loss_c.detach(),
        "color": loss_color.detach(),
    }
    return total, parts


def train_one_epoch(
    cfg: TrainConfig,
    epoch: int,
    loader: DataLoader,
    net: nn.Module,
    phase_consistency: nn.Module,
    perceptual: nn.Module,
    domain: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lr: float,
    loss_csv_path: Path,
    iter_log_path: Path,
) -> Dict[str, float]:
    net.train()
    domain.train()
    phase_consistency.train()
    perceptual.eval()

    _apply_lr(optimizer, lr)
    sums = {
        "total": 0.0,
        "phase": 0.0,
        "perceptual": 0.0,
        "contrast": 0.0,
        "color": 0.0,
    }
    n_batches = 0
    ne = cfg.num_epochs
    log_every = max(1, int(cfg.log_interval))
    run_tot = run_phase = run_p = run_c = run_col = 0.0
    run_n = 0
    n_loader = len(loader)
    ep_disp = epoch + 1

    def _flush_window(it_hi: int, tag: str) -> None:
        nonlocal run_tot, run_phase, run_p, run_c, run_col, run_n
        if run_n == 0:
            return
        mt = run_tot / run_n
        mph = run_phase / run_n
        mp = run_p / run_n
        mc = run_c / run_n
        mcol = run_col / run_n
        print(
            f"epoch【{ep_disp}/{ne}】 {tag} "
            f"loss={mt:.5f} phase={mph:.5f} perceptual={mp:.5f} contrast={mc:.5f} "
            f"color={mcol:.5f} lr={lr:.2e}",
            flush=True,
        )
        with iter_log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    ep_disp,
                    it_hi,
                    tag,
                    f"{mt:.6f}",
                    f"{mph:.6f}",
                    f"{mp:.6f}",
                    f"{mc:.6f}",
                    f"{mcol:.6f}",
                    f"{lr:.8f}",
                ]
            )
            f.flush()
        run_tot = run_phase = run_p = run_c = run_col = 0.0
        run_n = 0

    for it, batch in enumerate(loader, start=1):
        hazy = batch["hazy"].to(device, non_blocking=True)
        clear = batch["clear"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, parts = _forward_loss(
            cfg,
            hazy,
            clear,
            net,
            phase_consistency,
            perceptual,
            domain,
        )
        loss.backward()
        optimizer.step()

        for k in sums:
            sums[k] += float(parts[k].item())
        n_batches += 1

        run_tot += float(parts["total"].item())
        run_phase += float(parts["phase"].item())
        run_p += float(parts["perceptual"].item())
        run_c += float(parts["contrast"].item())
        run_col += float(parts["color"].item())
        run_n += 1
        if it % log_every == 0:
            _flush_window(it, f"iter {it}/{n_loader}")

    if run_n > 0:
        _flush_window(n_loader, f"tail iter {n_loader - run_n + 1}-{n_loader}/{n_loader}")

    for k in sums:
        sums[k] /= max(n_batches, 1)

    print(
        f"epoch[{ep_disp}/{ne}] epoch mean "
        f"loss={sums['total']:.5f} phase={sums['phase']:.5f} "
        f"perceptual={sums['perceptual']:.5f} contrast={sums['contrast']:.5f} "
        f"color={sums['color']:.5f} lr={lr:.2e}",
        flush=True,
    )

    with loss_csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                epoch,
                f"{sums['total']:.6f}",
                f"{sums['phase']:.6f}",
                f"{sums['perceptual']:.6f}",
                f"{sums['contrast']:.6f}",
                f"{sums['color']:.6f}",
                f"{lr:.8f}",
            ]
        )
        f.flush()
    return sums


@torch.inference_mode()
def validate(
    cfg: TrainConfig,
    epoch: int,
    loader: DataLoader,
    net: nn.Module,
    device: torch.device,
    dehazed_save_dir: Optional[Path] = None,
) -> Tuple[float, float, float, List[Dict[str, Any]]]:

    net.eval()
    if dehazed_save_dir is not None:
        dehazed_save_dir = Path(dehazed_save_dir)
        dehazed_save_dir.mkdir(parents=True, exist_ok=True)
    psnrs: List[float] = []
    ssims: List[float] = []
    per_image: List[Dict[str, Any]] = []
    total_batches = len(loader)
    epoch_1based = epoch + 1
    if total_batches > 0:
        print(
            f"validation progress epoch[{epoch_1based}/{cfg.num_epochs}] 0/{total_batches}",
            end="",
            flush=True,
        )
    for bi, batch in enumerate(loader, start=1):
        hazy = batch["hazy"].to(device, non_blocking=True)
        gt = batch["clear"].to(device, non_blocking=True)
        out = net(hazy).clamp(0.0, 1.0)
        names_b = batch.get("name")
        for j in range(out.size(0)):
            oj = out[j : j + 1]
            gj = gt[j : j + 1]
            pj = float(psnr(oj, gj, data_range=1.0, reduction="mean").item())
            sj = float(ssim(oj, gj, data_range=1.0, reduction="mean").item())
            psnrs.append(pj)
            ssims.append(sj)
            fn = names_b[j] if isinstance(names_b, (list, tuple)) else names_b
            per_image.append({"filename": fn, "psnr": pj, "ssim": sj})
            if dehazed_save_dir is not None:
                to_pil_image(out[j].cpu()).save(dehazed_save_dir / fn)
        if total_batches > 0:
            print(
                f"\rvalidation progress epoch[{epoch_1based}/{cfg.num_epochs}] {bi}/{total_batches}",
                end="",
                flush=True,
            )
    if total_batches > 0:
        print("", flush=True)
    mean_p = float(np.mean(psnrs)) if psnrs else 0.0
    mean_s = float(np.mean(ssims)) if ssims else 0.0
    score = mean_p + 100.0 * mean_s
    return mean_p, mean_s, score, per_image


def _save_full_checkpoint(
    path: Path,
    epoch: int,
    best_score: float,
    net: nn.Module,
    domain: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg_dict: Dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "best_score": best_score,
            "dehaze": net.state_dict(),
            "contrastive_domain": domain.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg_dict,
        },
        path,
    )


def _load_full_checkpoint(
    path: Path,
    net: nn.Module,
    domain: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    _load_dehaze_state_dict_compat(net, ckpt["dehaze"], source=path)
    domain.load_state_dict(ckpt["contrastive_domain"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt["epoch"]), float(ckpt.get("best_score", -1e18))


def _load_dehaze_state_dict_compat(
    net: nn.Module, state_dict: Dict[str, torch.Tensor], source: Any
) -> None:
    incompatible = net.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Error(s) in loading state_dict from {source}:\n"
            f"\tMissing key(s): {incompatible.missing_keys}\n"
            f"\tUnexpected key(s): {incompatible.unexpected_keys}"
        )


def _config_to_dict(cfg: TrainConfig) -> Dict:
    def _serialize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_serialize(x) for x in obj]
        return obj

    return _serialize(asdict(cfg))


def run_training(cfg: TrainConfig) -> None:
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and cfg.device.startswith("cuda"):
        print("CUDA is unavailable, switching to CPU.")

    ckpt_dir, val_root, res_dir = _ensure_dirs(cfg)
    last_ckpt_path = ckpt_dir / "last_full.pt"
    best_w_path = ckpt_dir / "best_weights.pt"
    best_full_path = ckpt_dir / "best_full.pt"

    loss_csv = (res_dir / "loss_history.csv").resolve()
    if not loss_csv.exists():
        with loss_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "loss_total",
                    "loss_phase",
                    "loss_perceptual",
                    "loss_contrast",
                    "loss_color",
                    "lr",
                ]
            )

    iter_log_csv = (res_dir / "loss_iteration.csv").resolve()
    if not iter_log_csv.exists():
        with iter_log_csv.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    "epoch",
                    "epoch_1based",
                    "iter_end",
                    "tag",
                    "loss_total",
                    "loss_phase",
                    "loss_perceptual",
                    "loss_contrast",
                    "loss_color",
                    "lr",
                ]
            )

    tf_train_hazy = _build_train_transforms_hazy(cfg)
    tf_train_clear = _build_train_transforms_clear(cfg)
    tf_eval = _build_eval_transforms()
    train_ds = UnpairedTrainDataset(
        cfg.data_root, transform_hazy=tf_train_hazy, transform_gt=tf_train_clear
    )
    val_ds = PairedValDataset(cfg.data_root, transform_hazy=tf_eval, transform_gt=tf_eval)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    net, phase_consistency, perceptual, domain, optimizer = build_modules(cfg, device)
    start_epoch = 0
    best_score = -1e18
    cfg_dict = _config_to_dict(cfg)

    resume_file = cfg.resume_path
    if cfg.resume and resume_file.is_file():
        start_epoch, best_score = _load_full_checkpoint(
            resume_file,
            net,
            domain,
            optimizer,
        )
        net.to(device)
        domain.to(device)
        start_epoch += 1

    for epoch in range(start_epoch, cfg.num_epochs):
        lr = _lr_at_epoch(cfg, epoch)
        sums = train_one_epoch(
            cfg,
            epoch,
            train_loader,
            net,
            phase_consistency,
            perceptual,
            domain,
            optimizer,
            device,
            lr,
            loss_csv,
            iter_log_csv,
        )

        _save_full_checkpoint(
            last_ckpt_path,
            epoch,
            best_score,
            net,
            domain,
            optimizer,
            cfg_dict,
        )

        if (epoch + 1) % cfg.val_interval == 0 or epoch == cfg.num_epochs - 1:
            epoch_1based = epoch + 1
            tmp_dir = val_root / f"tmp_epoch_{epoch_1based:04d}"
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            tmp_dir.mkdir(parents=True)

            mean_p, mean_s, score, val_per_image = validate(
                cfg, epoch, val_loader, net, device, dehazed_save_dir=tmp_dir
            )
            print(
                f"validation result epoch[{epoch_1based}/{cfg.num_epochs}] "
                f"PSNR={mean_p:.4f} SSIM={mean_s:.4f} Score={score:.4f}",
                flush=True,
            )

            torch.save(net.state_dict(), tmp_dir / "weights.pt")
            metrics = {
                "epoch": epoch_1based,
                "mean_psnr": mean_p,
                "mean_ssim": mean_s,
                "score_psnr_plus_100ssim": score,
                "eval": "full_image_no_patch",
                "per_image": val_per_image,
            }
            with (tmp_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

            _save_val_per_image_psnr_ssim_csv(
                tmp_dir / "per_image_psnr_ssim.csv",
                epoch_1based,
                val_per_image,
            )

            final_name = (
                f"e{epoch_1based:04d}_psnr{_metric_tag_for_dirname(mean_p)}_ssim{_metric_tag_for_dirname(mean_s)}"
            )
            final_dir = val_root / final_name
            if final_dir.exists():
                shutil.rmtree(final_dir)
            tmp_dir.rename(final_dir)

            if score > best_score:
                best_score = score
                print(
                    f"new best on validation epoch[{epoch_1based}/{cfg.num_epochs}] "
                    f"best_score={best_score:.4f}",
                    flush=True,
                )
                torch.save(net.state_dict(), best_w_path)
                _save_full_checkpoint(
                    best_full_path,
                    epoch,
                    best_score,
                    net,
                    domain,
                    optimizer,
                    cfg_dict,
                )
                _save_full_checkpoint(
                    last_ckpt_path,
                    epoch,
                    best_score,
                    net,
                    domain,
                    optimizer,
                    cfg_dict,
                )


def _optional_ciede(pred: torch.Tensor, ref: torch.Tensor) -> Optional[float]:
    try:
        return float(ciede2000(pred, ref, data_range=1.0, reduction="mean").item())
    except Exception:
        return None


def _optional_lpips_metric(pred: torch.Tensor, ref: torch.Tensor) -> Optional[float]:
    try:
        return float(lpips(pred, ref, reduction="mean").item())
    except Exception:
        return None


def run_testing(cfg: TrainConfig) -> None:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.test_output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "__tmp_infer__"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    net = SOLO_Dehaze(
        num_levels=cfg.dehaze_num_levels,
        hidden_dim=cfg.dehaze_hidden_dim,
        level_dims=cfg.dehaze_level_dims,
        level_heads=cfg.dehaze_level_heads,
        base_dim=cfg.dehaze_base_dim,
        base_heads=cfg.dehaze_base_heads,
    ).to(device)
    if not cfg.test_weights.is_file():
        raise FileNotFoundError(f"Test weights not found: {cfg.test_weights}")
    try:
        sd = torch.load(cfg.test_weights, map_location=device, weights_only=True)
    except TypeError:
        sd = torch.load(cfg.test_weights, map_location=device)
    _load_dehaze_state_dict_compat(net, sd, source=cfg.test_weights)
    net.eval()

    tf_eval = _build_eval_transforms()
    test_ds = TestDataset(cfg.data_root, transform_hazy=tf_eval, transform_gt=tf_eval)
    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_test_batch,
    )

    per_image: List[Dict[str, Any]] = []

    with torch.inference_mode():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            names = batch["name"]
            out = net(hazy).clamp(0.0, 1.0)
            mask = batch["has_gt"]
            clears = batch["clear"]

            for i in range(out.size(0)):
                name = names[i]
                to_pil_image(out[i].cpu()).save(tmp_dir / name)

                rec: Dict[str, Any] = {"filename": name, "has_gt": bool(mask[i].item())}
                if mask[i].item() and clears[i] is not None:
                    gt = clears[i].to(device)
                    o = out[i : i + 1]
                    g = gt.unsqueeze(0) if gt.dim() == 3 else gt
                    rec["psnr"] = float(psnr(o, g, data_range=1.0, reduction="mean").item())
                    rec["ssim"] = float(ssim(o, g, data_range=1.0, reduction="mean").item())
                    c = _optional_ciede(o, g)
                    if c is not None:
                        rec["ciede2000"] = c
                    l = _optional_lpips_metric(o, g)
                    if l is not None:
                        rec["lpips"] = l
                per_image.append(rec)

    paired = [r for r in per_image if r.get("has_gt")]
    avg: Dict[str, float] = {}
    if paired:
        for key in ("psnr", "ssim", "ciede2000", "lpips"):
            vals = [float(r[key]) for r in paired if key in r and r[key] is not None]
            if vals:
                avg[f"mean_{key}"] = float(np.mean(vals))

    if paired and "mean_psnr" in avg and "mean_ssim" in avg:
        final_name = (
            "test_psnr"
            + _metric_tag_for_dirname(avg["mean_psnr"])
            + "_ssim"
            + _metric_tag_for_dirname(avg["mean_ssim"])
        )
    else:
        final_name = f"test_{time.strftime('%Y%m%d_%H%M%S')}"

    final_dir = (out_dir / final_name).resolve()
    report = {
        "output_dir": str(final_dir),
        "weights": str(cfg.test_weights.resolve()),
        "count_total": len(per_image),
        "count_with_gt": len(paired),
        "eval": "full_image_no_patch",
        "per_image": per_image,
        "average_over_paired": avg if paired else {},
    }
    report_path = tmp_dir / "test_metrics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    _save_test_per_image_metrics_csv(tmp_dir / "per_image_psnr_ssim.csv", per_image)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)

