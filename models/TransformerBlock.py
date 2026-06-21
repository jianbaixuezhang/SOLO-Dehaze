from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .LightweightLinearAttn import LightweightLinearAttn


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        dw_kernel_size: int = 3,
    ):
        super().__init__()
        out_f = out_features or in_features
        hid = hidden_features or in_features
        k = int(dw_kernel_size)
        if k % 2 != 1 or k < 3:
            raise ValueError("dw_kernel_size must be an odd number >= 3")
        p = k // 2
        self.pw_sep = nn.Conv2d(in_features, hid, kernel_size=1, bias=True)
        self.dw = nn.Conv2d(hid, hid, kernel_size=k, stride=1, padding=p, groups=hid, bias=True)
        self.pw_direct = nn.Conv2d(in_features, hid, kernel_size=1, bias=True)
        self.proj = nn.Conv2d(hid, out_f, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.dw(self.pw_sep(x))
        b = self.pw_direct(x)
        h = F.gelu(a) * b
        return self.proj(h)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        heads=4,
        mlp_ratio=2.0,
        mlp_dw_kernel_size: int = 3,
    ):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)

        self.attn = LightweightLinearAttn(
            dim=dim,
            heads=heads,
        )

        self.norm2 = LayerNorm2d(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            dw_kernel_size=mlp_dw_kernel_size,
        )

    def forward(self, x):
        shortcut = x
        x_norm = self.norm1(x)

        out = shortcut + self.attn(x_norm)
        return out + self.mlp(self.norm2(out))


class BasicTransformerLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth=2,
        heads=4,
        mlp_ratio=2.0,
        mlp_dw_kernel_size: int = 3,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    mlp_ratio=mlp_ratio,
                    mlp_dw_kernel_size=mlp_dw_kernel_size,
                )
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x