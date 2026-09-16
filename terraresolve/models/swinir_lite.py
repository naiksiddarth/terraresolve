"""SwinIR-lite: a lightweight local-window self-attention SR backbone,
standing in for the full SwinIR (Liang et al., ICCVW 2021) so it
trains on a single laptop GPU. Same constructor/call shape as EDSR --
swap `model.name: edsr` -> `model.name: swinir` in the config and
nothing else in the pipeline has to change. NOTE: this is a
simplified variant (plain windows, no shift/relative position bias) --
good enough to prove out the CNN-vs-transformer comparison story;
swap in the full official SwinIR implementation later if you want the
real thing for the final benchmark."""
from __future__ import annotations
import torch
import torch.nn as nn
from einops import rearrange

from terraresolve.models.edsr import PixelShuffleUpsampler
from terraresolve.registry import MODEL_REGISTRY


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        ws = self.window_size
        pad_h, pad_w = (ws - H % ws) % ws, (ws - W % ws) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[1], x.shape[2]
        xw = rearrange(x, "b (hh ws1) (ww ws2) c -> (b hh ww) (ws1 ws2) c", ws1=ws, ws2=ws)
        qkv = self.qkv(xw).reshape(xw.shape[0], xw.shape[1], 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(xw.shape[0], xw.shape[1], C)
        out = self.proj(out)
        out = rearrange(out, "(b hh ww) (ws1 ws2) c -> b (hh ws1) (ww ws2) c",
                         hh=Hp // ws, ww=Wp // ws, ws1=ws, ws2=ws)
        return out[:, :H, :W, :]


class SwinBlock(nn.Module):
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 4,
                 mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = rearrange(x, "b c h w -> b h w c")
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return rearrange(x, "b h w c -> b c h w")


@MODEL_REGISTRY.register("swinir")
class SwinIRLite(nn.Module):
    def __init__(self, in_channels: int = 4, n_feats: int = 60, n_blocks: int = 6,
                 window_size: int = 8, num_heads: int = 4, scale: int = 4):
        super().__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, padding=1)
        self.body = nn.Sequential(
            *[SwinBlock(n_feats, window_size, num_heads) for _ in range(n_blocks)]
        )
        self.body_conv = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.upsample = PixelShuffleUpsampler(scale, n_feats)
        self.tail = nn.Conv2d(n_feats, in_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat + self.body_conv(self.body(feat))
        feat = self.upsample(feat)
        return self.tail(feat)
