import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class LightweightLinearAttn(nn.Module):

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        qkv_bias: bool = True,
    ):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim({dim}) must be divisible by heads({heads}).")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.to_out = nn.Linear(dim, dim)

    @staticmethod
    def phi(x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = rearrange(x, "b c h w -> b (h w) c")

        qkv = self.to_qkv(tokens)
        q, k, v = qkv.chunk(3, dim=-1)

        nh, d = self.heads, self.head_dim
        q = self.q_norm(rearrange(q, "b t (nh d) -> (b nh) t d", nh=nh, d=d)) * self.scale
        k = self.k_norm(rearrange(k, "b t (nh d) -> (b nh) t d", nh=nh, d=d)) * self.scale
        v_attn = rearrange(v, "b t (nh d) -> (b nh) t d", nh=nh, d=d)

        q, k = self.phi(q), self.phi(k)
        kv = torch.einsum("b t d, b t e -> b d e", k, v_attn)
        numerator = torch.einsum("b t d, b d e -> b t e", q, kv)
        k_sum = k.sum(dim=1)
        denominator = torch.einsum("b t d, b d -> b t", q, k_sum).unsqueeze(-1).clamp(min=1e-6)

        attn_out = numerator / denominator
        attn_out = rearrange(attn_out, "(b nh) t d -> b t (nh d)", b=b, nh=nh)

        out = self.to_out(attn_out)
        return rearrange(out, "b (h w) c -> b c h w", h=h, w=w)
