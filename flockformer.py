"""FlockFormer: point-guided flock-region segmentation.

Dependencies: torch, torchvision (required by timm), timm, einops.
Validated locally with torch 2.1.1+cu118, timm 1.0.20 and einops 0.8.1.

Usage::

    import torch
    from flockformer import FlockFormer

    model = FlockFormer().eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 4, 512, 512))
        foreground = logits.softmax(dim=1)[:, 1:2]

Input: floating-point [B, 4, H, W] images (RGB + NIR).
Output in eval mode: unnormalized group logits [B, 2, H, W].
Output in train mode: (group_logits, point_logits).
Use return_dict=True to obtain coarse masks, region priors and auxiliary outputs.
Point guidance is predicted internally; inference requires only the image.
Apply the same radiometric preprocessing used to train your checkpoint.

This standalone release preserves the paper-aligned experimental model's
probability-recursive encoder guides, module names and state_dict keys.
The public defaults follow the saved experiment configuration: use_aux=False
and detach_group_to_point=True. Set these explicitly for other experiments.
PGSFMUnetFormer remains available as an alias for existing import statements.
Only the model is included; training losses, datasets and weights are separate.

Architecture lineage: UNetFormer-style global-local decoder, ResNet backbone
provided by timm, and the project's point-guided resampling/fusion modules.
Retain applicable upstream attribution and license notices in your repository.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


def trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=0.0, std=std)


# =====================================================================================
# Basic blocks (kept compatible with original UnetFormer style)
# =====================================================================================


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer=nn.BatchNorm2d,
        bias: bool = False,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
            nn.ReLU6(inplace=True),
        )


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        norm_layer=nn.BatchNorm2d,
        bias: bool = False,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            ),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                bias=bias,
                dilation=dilation,
                stride=stride,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
            )
        )


class SeparableConvBN(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        norm_layer=nn.BatchNorm2d,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                groups=in_channels,
                bias=False,
            ),
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer=nn.ReLU6,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =====================================================================================
# Point-guided SFM modules
# =====================================================================================


class PointGuidedARS(nn.Module):
    """
    Point-guided Adaptive Resampling.

    It keeps the SFM idea (learn offsets before the next stage downsampling),
    but lets sparse point evidence guide where resampling should focus.
    """

    def __init__(self, in_channels: int, k: int = 3, compress_ratio: int = 4):
        super().__init__()
        self.k = k
        compressed_channels = max(in_channels // compress_ratio, 16)
        guide_channels = max(compressed_channels // 2, 8)

        self.compress = nn.Conv2d(in_channels, compressed_channels, kernel_size=1, bias=True)
        self.guide_proj = nn.Sequential(
            nn.Conv2d(1, guide_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU6(inplace=True),
        )
        fusion_channels = compressed_channels + guide_channels
        self.offset = nn.Conv2d(fusion_channels, k * 2, kernel_size=3, padding=1, bias=True)
        self.weight = nn.Conv2d(fusion_channels, k, kernel_size=3, padding=1, bias=True)
        self.offset_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        nn.init.constant_(self.offset.weight, 0.0)
        nn.init.constant_(self.offset.bias, 0.0)
        nn.init.constant_(self.weight.weight, 0.0)
        nn.init.constant_(self.weight.bias, 0.0)

    @staticmethod
    def _base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
        y = torch.linspace(-1 + 1 / h, 1 - 1 / h, h, device=device, dtype=dtype)
        x = torch.linspace(-1 + 1 / w, 1 - 1 / w, w, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return torch.stack([grid_x, grid_y], dim=0)

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = x.shape
        feat = self.compress(x)

        if guide is None:
            guide = x.new_zeros((b, 1, h, w))
        else:
            guide = F.interpolate(guide, size=(h, w), mode="bilinear", align_corners=False)
            guide = guide.clamp(0.0, 1.0)

        mix = torch.cat([feat, self.guide_proj(guide)], dim=1)
        offsets = self.offset(mix).view(b, self.k, 2, h, w)
        offsets = torch.tanh(offsets) * torch.sigmoid(self.offset_scale) * 2.0
        weights = F.softmax(self.weight(mix), dim=1).view(b, self.k, 1, h, w)

        base = self._base_grid(h, w, x.device, x.dtype).unsqueeze(0).unsqueeze(1)
        norm = x.new_tensor([w / 2.0, h / 2.0]).view(1, 1, 2, 1, 1)
        sample_grid = (base + offsets / norm).clamp(-1.0, 1.0)

        sampled = []
        for i in range(self.k):
            grid = sample_grid[:, i].permute(0, 2, 3, 1)
            sampled.append(
                F.grid_sample(
                    x,
                    grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
            )
        sampled = torch.stack(sampled, dim=1)
        out = (sampled * weights).sum(dim=1)
        return out


class LocalPixelRelationModule(nn.Module):
    def __init__(self, kernel_size: int = 3, upscale: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.upscale = upscale
        self.pad = kernel_size // 2

    def forward(self, mask_pred: torch.Tensor, value_feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = value_feat.shape
        k = self.kernel_size

        padded_feat = F.pad(value_feat, [self.pad] * 4, mode="replicate")
        unfolded = F.unfold(padded_feat, kernel_size=k, stride=1)
        unfolded = unfolded.view(b, c, k * k, h, w)

        weights = F.softmax(mask_pred, dim=1).unsqueeze(1)
        refined = (unfolded * weights).sum(dim=2)
        if self.upscale > 1:
            refined = F.interpolate(refined, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        return refined


class PointGuidedAdaptiveUp(nn.Module):
    """Point-guided adaptive upsampling, used in decoder instead of plain bilinear upsampling."""

    def __init__(self, low_res_channels: int, high_res_channels: int, kernel_size: int = 3, compress_ratio: int = 4):
        super().__init__()
        compressed_channels = max((low_res_channels + high_res_channels) // compress_ratio, 16)
        guide_channels = max(compressed_channels // 2, 8)

        self.compress_low = nn.Conv2d(low_res_channels, compressed_channels, 1, bias=True)
        self.compress_high = nn.Conv2d(high_res_channels, compressed_channels, 1, bias=True)
        self.guide_proj = nn.Sequential(
            nn.Conv2d(1, guide_channels, 3, padding=1, bias=True),
            nn.ReLU6(inplace=True),
        )
        self.mask_conv = nn.Conv2d(compressed_channels * 2 + guide_channels, kernel_size ** 2, kernel_size=3, padding=1, bias=True)
        self.lprm = LocalPixelRelationModule(kernel_size=kernel_size, upscale=2)

    def forward(self, x_low: torch.Tensor, x_high: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        target_h, target_w = x_high.shape[-2:]
        low_feat = self.compress_low(x_low)
        high_feat = self.compress_high(x_high)
        high_feat_low = F.interpolate(high_feat, size=x_low.shape[-2:], mode="bilinear", align_corners=False)

        if guide is None:
            guide_low = x_low.new_zeros((x_low.shape[0], 1, x_low.shape[2], x_low.shape[3]))
        else:
            guide_low = F.interpolate(guide, size=x_low.shape[-2:], mode="bilinear", align_corners=False).clamp(0.0, 1.0)

        mask = self.mask_conv(torch.cat([low_feat, high_feat_low, self.guide_proj(guide_low)], dim=1))
        x_up = self.lprm(mask, x_low)
        x_up = F.interpolate(x_up, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return x_up


# =====================================================================================
# Original UnetFormer GLTB, upgraded with point guidance
# =====================================================================================


class PointAwareGlobalLocalAttention(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        qkv_bias: bool = False,
        window_size: int = 8,
        relative_pos_embedding: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        # keep original GLTB structure
        self.qkv = Conv(dim, 3 * dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1, padding=(window_size // 2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size // 2 - 1))

        self.relative_pos_embedding = relative_pos_embedding
        self.guide_bias_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.guide_local_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
            )
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            trunc_normal_(self.relative_position_bias_table, std=.02)

    @staticmethod
    def pad(x: torch.Tensor, ps: int) -> torch.Tensor:
        _, _, h, w = x.size()
        pad_r = (ps - w % ps) % ps
        pad_b = (ps - h % ps) % ps
        if pad_r > 0 or pad_b > 0:
            mode = 'reflect'
            if (pad_r >= w and w > 0) or (pad_b >= h and h > 0):
                mode = 'replicate'
            x = F.pad(x, (0, pad_r, 0, pad_b), mode=mode)
        return x

    @staticmethod
    def pad_out(x: torch.Tensor) -> torch.Tensor:
        return F.pad(x, pad=(0, 1, 0, 1), mode='reflect')

    def _guide_attention_bias(self, guide: torch.Tensor, hp: int, wp: int) -> torch.Tensor:
        guide_windows = rearrange(
            guide,
            'b 1 (hh ws1) (ww ws2) -> (b hh ww) (ws1 ws2)',
            hh=hp // self.ws,
            ww=wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )
        guide_bias = 0.5 * (guide_windows.unsqueeze(-1) + guide_windows.unsqueeze(-2))
        guide_bias = guide_bias.unsqueeze(1)
        return guide_bias

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, c, h, w = x.shape

        local = self.local2(x) + self.local1(x)
        if guide is not None:
            guide_local = F.interpolate(guide, size=(h, w), mode='bilinear', align_corners=False).clamp(0.0, 1.0)
            local = local * (1.0 + torch.sigmoid(self.guide_local_scale) * guide_local)

        x_pad = self.pad(x, self.ws)
        _, _, hp, wp = x_pad.shape
        qkv = self.qkv(x_pad)

        q, k, v = rearrange(
            qkv,
            'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d',
            h=self.num_heads,
            d=c // self.num_heads,
            hh=hp // self.ws,
            ww=wp // self.ws,
            qkv=3,
            ws1=self.ws,
            ws2=self.ws,
        )

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws,
                self.ws * self.ws,
                -1,
            )
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots = dots + relative_position_bias.unsqueeze(0)

        if guide is not None:
            guide_pad = self.pad(guide, self.ws).clamp(0.0, 1.0)
            guide_bias = self._guide_attention_bias(guide_pad, hp, wp)
            dots = dots + torch.sigmoid(self.guide_bias_scale) * guide_bias

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(
            attn,
            '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)',
            h=self.num_heads,
            d=c // self.num_heads,
            hh=hp // self.ws,
            ww=wp // self.ws,
            ws1=self.ws,
            ws2=self.ws,
        )
        attn = attn[:, :, :h, :w]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        if guide is not None:
            out = out * (1.0 + torch.sigmoid(self.guide_local_scale) * guide_local)

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :h, :w]
        return out


class PointAwareBlock(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer=nn.ReLU6,
        norm_layer=nn.BatchNorm2d,
        window_size: int = 8,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PointAwareGlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x), guide=guide))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =====================================================================================
# UGTransformer-like weighted merge block (for encoder skip enhancement)
# =====================================================================================


class WeightedMergeBlock(nn.Module):
    def __init__(self, c_hi: int, c_lo: int, out_ch: Optional[int] = None, per_channel: bool = True):
        super().__init__()
        out_ch = out_ch or c_hi
        self.proj_hi = nn.Conv2d(c_hi, out_ch, 1, bias=False)
        self.proj_lo = nn.Conv2d(c_lo, out_ch, 1, bias=False)
        if per_channel:
            self.alpha = nn.Parameter(torch.zeros(1, out_ch, 1, 1))
        else:
            self.alpha = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.refine = ConvBNReLU(out_ch, out_ch, kernel_size=3)

    def forward(self, feat_hi: torch.Tensor, feat_lo: torch.Tensor) -> torch.Tensor:
        h, w = feat_hi.shape[-2:]
        hi = self.proj_hi(feat_hi)
        lo = self.proj_lo(feat_lo)
        lo = F.interpolate(lo, size=(h, w), mode='bilinear', align_corners=False)
        a = torch.sigmoid(self.alpha)
        fused = a * hi + (1.0 - a) * lo
        return self.refine(fused)


# =====================================================================================
# Decoder fusion blocks (still based on original UnetFormer workflow)
# =====================================================================================


class PointGuidedWF(nn.Module):
    def __init__(self, in_channels: int = 128, decode_channels: int = 128, eps: float = 1e-8, use_sfm_up: bool = True):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.use_sfm_up = use_sfm_up
        self.up = PointGuidedAdaptiveUp(decode_channels, in_channels, kernel_size=3) if use_sfm_up else None
        self.guide_gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor, res: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.up is None:
            x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)
        else:
            x = self.up(x, res, guide=guide)

        if guide is not None:
            guide = F.interpolate(guide, size=res.shape[-2:], mode='bilinear', align_corners=False).clamp(0.0, 1.0)
            gain = torch.sigmoid(self.guide_gate_scale)
            x = x * (1.0 + gain * guide)
            res = res * (1.0 + gain * guide)

        weights = F.relu(self.weights)
        fuse_weights = weights / (weights.sum(dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class PointGuidedFeatureRefinementHead(nn.Module):
    def __init__(self, in_channels: int = 64, decode_channels: int = 64, use_sfm_up: bool = True):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)
        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid(),
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(decode_channels, max(decode_channels // 16, 4), kernel_size=1),
            nn.ReLU6(inplace=True),
            Conv(max(decode_channels // 16, 4), decode_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6(inplace=True)

        self.use_sfm_up = use_sfm_up
        self.up = PointGuidedAdaptiveUp(decode_channels, in_channels, kernel_size=3) if use_sfm_up else None
        self.guide_gate_scale = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

    def forward(self, x: torch.Tensor, res: torch.Tensor, guide: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.up is None:
            x = F.interpolate(x, size=res.shape[-2:], mode='bilinear', align_corners=False)
        else:
            x = self.up(x, res, guide=guide)

        if guide is not None:
            guide = F.interpolate(guide, size=res.shape[-2:], mode='bilinear', align_corners=False).clamp(0.0, 1.0)
            gain = torch.sigmoid(self.guide_gate_scale)
            x = x * (1.0 + gain * guide)
            res = res * (1.0 + gain * guide)

        weights = F.relu(self.weights)
        fuse_weights = weights / (weights.sum(dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)

        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)
        return x


class AuxHead(nn.Module):
    def __init__(self, in_channels: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, in_channels)
        self.drop = nn.Dropout(0.1)
        self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
        return feat


# =====================================================================================
# Organic point -> group fusion head
# =====================================================================================


class PointEnvelopeGenerator(nn.Module):
    """
    Learnable point-to-group envelope generator.
    Sparse point evidence is expanded into region evidence under feature guidance.
    """

    def __init__(self, channels: int):
        super().__init__()
        point_channels = max(channels // 4, 8)
        self.point_proj = ConvBNReLU(1, point_channels, kernel_size=3)

        self.dw1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.dw2 = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False)
        self.dw3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, groups=channels, bias=False)

        self.fuse = nn.Sequential(
            ConvBNReLU(channels * 3 + point_channels + 1, channels, kernel_size=3),
            ConvBNReLU(channels, channels, kernel_size=3),
        )
        self.region_pred = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(
        self,
        point_prob: torch.Tensor,
        feat: torch.Tensor,
        coarse_group_prob: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        point_prob = point_prob.clamp(0.0, 1.0)
        coarse_group_prob = coarse_group_prob.clamp(0.0, 1.0)

        feat_point = feat * (1.0 + point_prob)
        x1 = self.dw1(feat_point)
        x2 = self.dw2(feat_point)
        x3 = self.dw3(feat_point)
        pp = self.point_proj(point_prob)

        env_feat = self.fuse(torch.cat([x1, x2, x3, pp, coarse_group_prob], dim=1))
        region_prior = torch.sigmoid(self.region_pred(env_feat))
        return env_feat, region_prior


class GroupPointFusionHead(nn.Module):
    def __init__(
        self,
        channels: int,
        num_classes: int,
        dropout: float = 0.1,
        detach_group_to_point: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.detach_group_to_point = detach_group_to_point

        self.base_feat = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
        )
        self.seg_base = nn.Conv2d(channels, num_classes, kernel_size=1)

        self.p2_proj = ConvBNReLU(channels, channels, kernel_size=3)
        self.point_fuse = nn.Sequential(
            ConvBNReLU(channels * 2 + 2, channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
            ConvBNReLU(channels, channels, kernel_size=3),
        )
        self.point_pred = nn.Conv2d(channels, 1, kernel_size=1)

        self.envelope = PointEnvelopeGenerator(channels)

        self.refine_block = nn.Sequential(
            ConvBNReLU(channels * 2 + 3, channels, kernel_size=3),
            nn.Dropout2d(p=dropout, inplace=True),
            ConvBNReLU(channels, channels, kernel_size=3),
        )
        self.seg_delta = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.refine_scale = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))

    def _foreground_prob(self, seg_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.sigmoid(seg_logits)
        return torch.softmax(seg_logits, dim=1)[:, 1:2]

    @staticmethod
    def _uncertain_mask(herd_prob: torch.Tensor) -> torch.Tensor:
        return (1.0 - (herd_prob - 0.5).abs() * 2.0).clamp(0.0, 1.0)

    def forward(self, p1: torch.Tensor, p2: torch.Tensor, point_guidance: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        p2_up = F.interpolate(self.p2_proj(p2), size=p1.shape[-2:], mode='bilinear', align_corners=False)
        base_feat = self.base_feat(p1)
        coarse_group_logits = self.seg_base(base_feat)
        coarse_group_prob = self._foreground_prob(coarse_group_logits)

        if self.detach_group_to_point and self.training:
            point_group_context = coarse_group_prob.detach()
        else:
            point_group_context = coarse_group_prob

        if point_guidance is None:
            point_guidance = torch.zeros_like(point_group_context)
        else:
            point_guidance = F.interpolate(point_guidance, size=p1.shape[-2:], mode='bilinear', align_corners=False)

        point_feat = self.point_fuse(torch.cat([p1, p2_up, point_group_context, point_guidance], dim=1))
        point_logits = self.point_pred(point_feat)
        point_prob = torch.sigmoid(point_logits)

        env_feat, region_prior = self.envelope(point_prob, base_feat, point_group_context)
        uncertain = self._uncertain_mask(coarse_group_prob)
        # Previous conservative gate:
        # refine_gate = ((point_group_context + region_prior) * 0.5) * uncertain
        # Let the learned region prior contribute even when the coarse mask is over-confident.
        refine_gate = (0.5 * uncertain + 0.5 * region_prior).clamp(0.0, 1.0)

        refine_feat = self.refine_block(
            torch.cat([base_feat, env_feat, point_prob, point_group_context, region_prior], dim=1)
        )
        seg_delta = self.seg_delta(refine_feat)
        refine_alpha = torch.sigmoid(self.refine_scale)
        final_group_logits = coarse_group_logits + refine_alpha * refine_gate * seg_delta

        return {
            "group_logits": final_group_logits,
            "point_logits": point_logits,
            "coarse_group_logits": coarse_group_logits,
            "region_prior": region_prior,
            "refine_gate": refine_gate,
            "point_prob": point_prob,
        }


# =====================================================================================
# Encoder with point-guided SFM and stage-wise point prior pyramid
# =====================================================================================


class PointGuidedResNetEncoder(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet18",
        in_channels: int = 4,
        pretrained: bool = False,
        use_pgars: bool = True,
    ):
        super().__init__()
        self.use_pgars = use_pgars
        try:
            import timm
        except Exception as e:
            raise ImportError("PGSFMUnetFormer requires timm for backbone creation. Please install timm in the training environment.") from e

        self.backbone = timm.create_model(backbone_name, features_only=False, pretrained=pretrained)

        required = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
        if not all(hasattr(self.backbone, k) for k in required):
            raise ValueError(f"Backbone {backbone_name} is not compatible with stage-wise PGARS encoder.")

        self._replace_input_conv(in_channels)
        self.channels = [
            self._stage_channels(self.backbone.layer1),
            self._stage_channels(self.backbone.layer2),
            self._stage_channels(self.backbone.layer3),
            self._stage_channels(self.backbone.layer4),
        ]

        self.seed1 = nn.Conv2d(self.channels[0], 1, kernel_size=1)
        self.seed2 = nn.Conv2d(self.channels[1], 1, kernel_size=1)
        self.seed3 = nn.Conv2d(self.channels[2], 1, kernel_size=1)
        self.seed4 = nn.Conv2d(self.channels[3], 1, kernel_size=1)

        if self.use_pgars:
            self.ars1 = PointGuidedARS(self.channels[0], k=3, compress_ratio=4)
            self.ars2 = PointGuidedARS(self.channels[1], k=3, compress_ratio=4)
            self.ars3 = PointGuidedARS(self.channels[2], k=3, compress_ratio=4)

    @staticmethod
    def _stage_channels(layer: nn.Sequential) -> int:
        block = layer[-1]
        for name in ["conv3", "conv2", "conv1"]:
            if hasattr(block, name):
                return getattr(block, name).out_channels
        raise ValueError("Cannot infer stage output channels.")

    def _replace_input_conv(self, in_channels: int) -> None:
        orig_conv = self.backbone.conv1
        if orig_conv.in_channels == in_channels:
            return

        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=orig_conv.bias is not None,
        )

        if orig_conv.weight.shape[1] >= 3 and in_channels >= 3:
            with torch.no_grad():
                new_conv.weight[:, :3] = orig_conv.weight[:, :3]
                if in_channels > 3:
                    mean_w = orig_conv.weight[:, :3].mean(dim=1, keepdim=True)
                    repeat_n = in_channels - 3
                    new_conv.weight[:, 3:3 + repeat_n] = mean_w.repeat(1, repeat_n, 1, 1)
                if new_conv.bias is not None and orig_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            if new_conv.bias is not None:
                nn.init.zeros_(new_conv.bias)

        self.backbone.conv1 = new_conv

    def _combine_guides(self, current_logit: torch.Tensor, prev_guide: Optional[torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        if prev_guide is None:
            return torch.sigmoid(current_logit)
        prev = F.interpolate(prev_guide, size=size, mode='bilinear', align_corners=False)
        return torch.sigmoid(current_logit + prev)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        if hasattr(self.backbone, "act1"):
            x = self.backbone.act1(x)
        else:
            x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        r1 = self.backbone.layer1(x)
        g1 = self._combine_guides(self.seed1(r1), None, r1.shape[-2:])
        res1 = self.ars1(r1, g1) if self.use_pgars else r1

        r2 = self.backbone.layer2(res1)
        g2 = self._combine_guides(self.seed2(r2), g1, r2.shape[-2:])
        res2 = self.ars2(r2, g2) if self.use_pgars else r2

        r3 = self.backbone.layer3(res2)
        g3 = self._combine_guides(self.seed3(r3), g2, r3.shape[-2:])
        res3 = self.ars3(r3, g3) if self.use_pgars else r3

        res4 = self.backbone.layer4(res3)
        g4 = self._combine_guides(self.seed4(res4), g3, res4.shape[-2:])

        return res1, res2, res3, res4, [g1, g2, g3, g4]


# =====================================================================================
# Decoder
# =====================================================================================


class PGSFMDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
        decode_channels: int = 64,
        dropout: float = 0.1,
        window_size: int = 8,
        num_classes: int = 2,
        use_sfm_up: bool = True,
        use_aux: bool = True,
        detach_group_to_point: bool = False,
    ):
        super().__init__()
        self.use_aux = use_aux

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = PointAwareBlock(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = PointAwareBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = PointGuidedWF(encoder_channels[-2], decode_channels, use_sfm_up=use_sfm_up)

        self.b2 = PointAwareBlock(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = PointGuidedWF(encoder_channels[-3], decode_channels, use_sfm_up=use_sfm_up)

        self.p1 = PointGuidedFeatureRefinementHead(encoder_channels[-4], decode_channels, use_sfm_up=use_sfm_up)

        self.dec_seed4 = nn.Conv2d(decode_channels, 1, kernel_size=1)
        self.dec_seed3 = nn.Conv2d(decode_channels, 1, kernel_size=1)
        self.dec_seed2 = nn.Conv2d(decode_channels, 1, kernel_size=1)

        self.head = GroupPointFusionHead(
            channels=decode_channels,
            num_classes=num_classes,
            dropout=dropout,
            detach_group_to_point=detach_group_to_point,
        )

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.aux_head = AuxHead(decode_channels, num_classes)
        self.init_weight()

    @staticmethod
    def _fuse_guides(enc_guide: Optional[torch.Tensor], dec_seed: Optional[torch.Tensor], target_size: Tuple[int, int]) -> Optional[torch.Tensor]:
        parts = []
        if enc_guide is not None:
            parts.append(F.interpolate(enc_guide, size=target_size, mode='bilinear', align_corners=False))
        if dec_seed is not None:
            parts.append(F.interpolate(torch.sigmoid(dec_seed), size=target_size, mode='bilinear', align_corners=False))
        if not parts:
            return None
        out = 0.0
        for p in parts:
            out = out + p
        return (out / len(parts)).clamp(0.0, 1.0)

    def forward(
        self,
        res1: torch.Tensor,
        res2: torch.Tensor,
        res3: torch.Tensor,
        res4: torch.Tensor,
        enc_guides: List[torch.Tensor],
        h: int,
        w: int,
        return_debug: bool = False,
    ) -> Dict[str, torch.Tensor]:
        debug: Dict[str, torch.Tensor] = {}

        g4 = enc_guides[3] if len(enc_guides) > 3 else None
        x = self.b4(self.pre_conv(res4), guide=g4)
        h4 = self.up4(x) if self.use_aux else None
        seed4 = self.dec_seed4(x)
        g3 = self._fuse_guides(enc_guides[2] if len(enc_guides) > 2 else None, seed4, res3.shape[-2:])
        debug["guide3"] = g3.detach() if g3 is not None else None

        x = self.p3(x, res3, guide=g3)
        x = self.b3(x, guide=g3)
        h3 = self.up3(x) if self.use_aux else None
        seed3 = self.dec_seed3(x)
        g2 = self._fuse_guides(enc_guides[1] if len(enc_guides) > 1 else None, seed3, res2.shape[-2:])
        debug["guide2"] = g2.detach() if g2 is not None else None

        x = self.p2(x, res2, guide=g2)
        x = self.b2(x, guide=g2)
        h2 = x if self.use_aux else None
        seed2 = self.dec_seed2(x)
        g1 = self._fuse_guides(enc_guides[0] if len(enc_guides) > 0 else None, seed2, res1.shape[-2:])
        debug["guide1"] = g1.detach() if g1 is not None else None

        p1 = self.p1(x, res1, guide=g1)
        outputs = self.head(p1, x, point_guidance=g1)

        outputs["group_logits"] = F.interpolate(outputs["group_logits"], size=(h, w), mode='bilinear', align_corners=False)
        outputs["point_logits"] = F.interpolate(outputs["point_logits"], size=(h, w), mode='bilinear', align_corners=False)
        outputs["coarse_group_logits"] = F.interpolate(outputs["coarse_group_logits"], size=(h, w), mode='bilinear', align_corners=False)
        outputs["region_prior"] = F.interpolate(outputs["region_prior"], size=(h, w), mode='bilinear', align_corners=False)
        outputs["refine_gate"] = F.interpolate(outputs["refine_gate"], size=(h, w), mode='bilinear', align_corners=False)

        if self.use_aux:
            aux = h4 + h3 + h2
            outputs["aux_logits"] = self.aux_head(aux, h, w)

        if return_debug:
            outputs["decoder_guides"] = debug
        return outputs

    def init_weight(self) -> None:
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


# =====================================================================================
# Full model
# =====================================================================================


class FlockFormer(nn.Module):
    """
    FlockFormer with internally predicted point guidance.

    Key logic:
    1) The point branch supplies internal guidance for flock-region segmentation.
    2) Sparse point evidence guides encoder resampling (PGARS) and decoder GLTB attention.
    3) Final output is still group mask logits, refined by a learnable point-to-group envelope.

    Default inference output: group logits only.
    Default training output: tuple(group_logits, point_logits)
    Optional detailed output with internal tensors: return_dict=True
    """

    def __init__(
        self,
        in_channels: int = 4,
        decode_channels: int = 64,
        dropout: float = 0.1,
        backbone_name: str = 'resnet18',
        pretrained: bool = False,
        window_size: int = 8,
        num_classes: int = 2,
        use_skip_fusion: bool = True,
        use_sfm_up: bool = True,
        use_pgars: bool = True,
        use_aux: bool = False,
        detach_group_to_point: bool = True,
    ):
        super().__init__()
        self.use_skip_fusion = use_skip_fusion
        self.num_classes = num_classes
        self.use_aux = use_aux

        self.encoder = PointGuidedResNetEncoder(
            backbone_name=backbone_name,
            in_channels=in_channels,
            pretrained=pretrained,
            use_pgars=use_pgars,
        )
        encoder_channels = tuple(self.encoder.channels)

        if use_skip_fusion:
            c1, c2, c3, c4 = encoder_channels
            self.merge12 = WeightedMergeBlock(c_hi=c1, c_lo=c2, out_ch=c1, per_channel=True)
            self.merge23 = WeightedMergeBlock(c_hi=c2, c_lo=c3, out_ch=c2, per_channel=True)
            self.merge34 = WeightedMergeBlock(c_hi=c3, c_lo=c4, out_ch=c3, per_channel=True)

        self.decoder = PGSFMDecoder(
            encoder_channels=encoder_channels,
            decode_channels=decode_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes,
            use_sfm_up=use_sfm_up,
            use_aux=use_aux,
            detach_group_to_point=detach_group_to_point,
        )

    def _foreground_prob(self, seg_logits: torch.Tensor) -> torch.Tensor:
        if self.num_classes == 1:
            return torch.sigmoid(seg_logits)
        return torch.softmax(seg_logits, dim=1)[:, 1:2]

    @torch.no_grad()
    def predict_group_prob(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        logits = self.forward(x)
        prob = self._foreground_prob(logits)
        if was_training:
            self.train()
        return prob

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
        return_point: bool = False,
        return_debug: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        h, w = x.size()[-2:]
        res1, res2, res3, res4, enc_guides = self.encoder(x)

        if self.use_skip_fusion:
            r1, r2, r3, r4 = res1, res2, res3, res4
            res3 = self.merge34(r3, r4)
            res2 = self.merge23(r2, r3)
            res1 = self.merge12(r1, r2)

        outputs = self.decoder(res1, res2, res3, res4, enc_guides, h, w, return_debug=return_debug)

        if return_dict:
            if return_debug:
                outputs["encoder_guides"] = [g.detach() for g in enc_guides]
            return outputs

        if self.training:
            return outputs["group_logits"], outputs["point_logits"]

        if return_point:
            return outputs["group_logits"], outputs["point_logits"]

        return outputs["group_logits"]


# =====================================================================================
# Backward-compatible import name; parameter names remain unchanged.
PGSFMUnetFormer = FlockFormer


def _smoke_test() -> None:
    """Run an offline CPU forward check: python flockformer.py."""
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = FlockFormer(pretrained=False)
    x = torch.randn(2, 4, 128, 128)
    model.train()
    group, point = model(x)
    assert group.shape == (2, 2, 128, 128)
    assert point.shape == (2, 1, 128, 128)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        details = model(x, return_dict=True, return_debug=True)
    assert logits.shape == group.shape
    assert torch.isfinite(logits).all()
    print("Train group/point:", tuple(group.shape), tuple(point.shape))
    print("Eval group:", tuple(logits.shape))
    print("Detailed outputs:", ", ".join(sorted(details)))
    print("Parameters:", sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    _smoke_test()
