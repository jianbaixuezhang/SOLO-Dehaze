import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from models.TransformerBlock import BasicTransformerLayer


class LevelFeatureProcessor(nn.Module):

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"LevelFeatureProcessor: dim({dim}) must be divisible by heads({heads}).")
        self.lap_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.transformer = BasicTransformerLayer(
            dim=dim,
            depth=depth,
            heads=heads,
        )
        self.unembed = nn.Conv2d(dim, in_channels, kernel_size=3, stride=1, padding=1)

        nn.init.zeros_(self.unembed.weight)
        nn.init.zeros_(self.unembed.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lap_feat = self.lap_embed(x)
        feat = self.transformer(lap_feat)
        out = self.unembed(feat)
        return x + out


class LaplacianPyramidProcessor(nn.Module):

    def __init__(self, num_levels: int = 3, channels: int = 3):
        super().__init__()
        self.num_levels = num_levels
        self.up_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.level_processors = nn.ModuleDict()
        self.base_processor = None

    def add_level_processor(self, level: int, processor: nn.Module):
        self.level_processors[str(level)] = processor

    def add_base_processor(self, processor: nn.Module):
        self.base_processor = processor

    def _downsample(self, x: torch.Tensor) -> torch.Tensor:
        h2 = max(1, x.shape[2] // 2)
        w2 = max(1, x.shape[3] // 2)
        return F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False)

    def _upsample(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        up = F.interpolate(x, size=(h * 2, w * 2), mode="bilinear", align_corners=False)
        return self.up_conv(up)

    def _align_up(self, up: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if up.shape[2] != target_hw[0] or up.shape[3] != target_hw[1]:
            up = F.interpolate(up, size=target_hw, mode="bilinear", align_corners=False)
        return up

    def decompose(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        current = x
        laplacian_levels: List[torch.Tensor] = []

        for _ in range(self.num_levels):
            down = self._downsample(current)
            up = self._upsample(down)
            up = self._align_up(up, (current.shape[2], current.shape[3]))
            lap = current - up
            laplacian_levels.append(lap)
            current = down

        return laplacian_levels, current

    def process_levels(
        self,
        laplacian_levels: List[torch.Tensor],
        gaussian_base: torch.Tensor,
    ):
        processed_levels = []
        for level, lap in enumerate(laplacian_levels):
            if str(level) in self.level_processors:
                proc = self.level_processors[str(level)]
                processed_lap = proc(lap)
            else:
                processed_lap = lap
            processed_levels.append(processed_lap)

        if self.base_processor is not None:
            processed_base = self.base_processor(gaussian_base)
        else:
            processed_base = gaussian_base

        return processed_levels, processed_base

    def reconstruct(self, laplacian_levels: List[torch.Tensor], gaussian_base: torch.Tensor) -> torch.Tensor:
        current = gaussian_base
        for level in reversed(range(len(laplacian_levels))):
            lap = laplacian_levels[level]
            up = self._upsample(current)
            up = self._align_up(up, (lap.shape[2], lap.shape[3]))
            current = up + lap
        return current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        laplacian_levels, gaussian_base = self.decompose(x)
        processed_levels, processed_base = self.process_levels(laplacian_levels, gaussian_base)
        return self.reconstruct(processed_levels, processed_base)


class SOLO_Dehaze(nn.Module):
    def __init__(
        self,
        num_levels: int = 3,
        hidden_dim: int = 64,
        level_dims: Optional[List[int]] = None,
        level_heads: Optional[List[int]] = None,
        base_dim: Optional[int] = None,
        base_heads: Optional[int] = None,
    ):
        super().__init__()
        if num_levels < 1:
            raise ValueError("num_levels must be >= 1.")

        if level_dims is None:
            level_dims = [hidden_dim] * num_levels
        elif len(level_dims) != num_levels:
            raise ValueError(f"level_dims length must equal num_levels({num_levels}).")

        if level_heads is None:
            level_heads = [4] * num_levels
        elif len(level_heads) != num_levels:
            raise ValueError(f"level_heads length must equal num_levels({num_levels}).")

        if base_dim is None:
            base_dim = hidden_dim
        if base_heads is None:
            base_heads = 4

        for i, (d, h) in enumerate(zip(level_dims, level_heads)):
            if d % h != 0:
                raise ValueError(f"level {i}: dim({d}) must be divisible by heads({h}).")
        if base_dim % base_heads != 0:
            raise ValueError(f"base: dim({base_dim}) must be divisible by heads({base_heads}).")

        self.pyramid = LaplacianPyramidProcessor(num_levels=num_levels, channels=3)
        for level in range(num_levels):
            processor = LevelFeatureProcessor(
                in_channels=3,
                dim=level_dims[level],
                depth=2,
                heads=level_heads[level],
            )
            self.pyramid.add_level_processor(level, processor)

        base_processor = LevelFeatureProcessor(
            in_channels=3,
            dim=base_dim,
            depth=4,
            heads=base_heads,
        )
        self.pyramid.add_base_processor(base_processor)

    def forward(self, hazy_img: torch.Tensor) -> torch.Tensor:
        hazy_img = hazy_img.clamp(0.0, 1.0)
        return self.pyramid(hazy_img)

    @staticmethod
    def _merge_multiscale_features(
        levels: List[torch.Tensor], base: torch.Tensor, target_hw: Tuple[int, int]
    ) -> torch.Tensor:
        feat_parts: List[torch.Tensor] = []
        for lv in levels:
            if lv.shape[2:] != target_hw:
                lv = F.interpolate(lv, size=target_hw, mode="bilinear", align_corners=False)
            feat_parts.append(lv)
        if base.shape[2:] != target_hw:
            base = F.interpolate(base, size=target_hw, mode="bilinear", align_corners=False)
        feat_parts.append(base)
        return torch.cat(feat_parts, dim=1)

    def forward_with_features(
        self, hazy_img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        hazy_img = hazy_img.clamp(0.0, 1.0)
        laplacian_levels, gaussian_base = self.pyramid.decompose(hazy_img)
        processed_levels, processed_base = self.pyramid.process_levels(laplacian_levels, gaussian_base)
        out = self.pyramid.reconstruct(processed_levels, processed_base)

        target_hw = (hazy_img.shape[2], hazy_img.shape[3])
        feat_proc = self._merge_multiscale_features(processed_levels, processed_base, target_hw)
        feat_raw = self._merge_multiscale_features(laplacian_levels, gaussian_base, target_hw)
        return out, feat_proc, feat_raw
