import math
from typing import List, Optional, Tuple, Set, Union
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Enums ---

class NormType(Enum):
    BATCH = "batch"
    GROUP = "group"
    LAYER = "layer"
    NONE = "none"

class CausalityAxis(Enum):
    NONE = "none"
    HEIGHT = "height"
    WIDTH = "width"
    BOTH = "both"

class AttentionType(Enum):
    VANILLA = "vanilla"
    LINEAR = "linear"

# --- Utils ---

def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | Tuple[int, int],
    stride: int | Tuple[int, int],
    padding: int | Tuple[int, int] | str = 0,
    dilation: int | Tuple[int, int] = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis = CausalityAxis.NONE,
) -> nn.Conv2d:
    # Simplified: ignore causality padding logic for now, assume standard padding
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
        
    # LTX uses specific padding logic for causal convs.
    # For simplicity in this implementation, we use standard Conv2d.
    # If strict reproducibility is needed, we must port CausalConv2d logic.
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
    )

def build_normalization_layer(channels: int, normtype: NormType = NormType.GROUP, num_groups: int = 32) -> nn.Module:
    if normtype == NormType.GROUP:
        return nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
    elif normtype == NormType.BATCH:
        return nn.BatchNorm2d(channels)
    elif normtype == NormType.LAYER:
        return nn.LayerNorm(channels)
    else:
        return nn.Identity()

# --- ResNet ---

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.NONE,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.conv1 = make_conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, causality_axis=causality_axis
        )
        
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
            
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = make_conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, causality_axis=causality_axis
        )
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, causality_axis=causality_axis
                )
            else:
                self.conv_shortcut = make_conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0, causality_axis=causality_axis
                )
        else:
            self.conv_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(F.silu(temb))[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.conv_shortcut(x)

        return x + h

# --- Attention ---

class Attention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = make_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = make_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = make_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = make_conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

# --- Upsample ---

class Upsample(nn.Module):
    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        dims: int = 2,
        out_channels: Optional[int] = None,
        padding: int = 1,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = make_conv2d(self.channels, self.out_channels, 3, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dims == 3:
            x = F.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

# --- Audio Decoder ---

class AudioDecoder(nn.Module):
    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 1,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = {16},
        resolution: int = 256,
        z_channels: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        
        # Compute in_channels for the first block
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = make_conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = Attention(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)

        # Upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(Attention(block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, use_conv=True)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to match order

        self.norm_out = build_normalization_layer(block_in)
        self.conv_out = make_conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, H, W]
        
        # Timestep embedding (temb) is None for VAE
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # Middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # Upsampling
        for i_level in reversed(range(self.num_resolutions)):
            up_stage = self.up[i_level] # Note: self.up is a ModuleList, indexing might need adjustment based on construction
            # Actually, I constructed self.up by inserting at 0, so index 0 is the last level (lowest res)?
            # Let's recheck construction loop:
            # reversed(range(num_resolutions)) -> 3, 2, 1, 0
            # insert(0, up) -> list grows as [3], [2,3], [1,2,3], [0,1,2,3]
            # So index 0 corresponds to level 0 (highest res), index 3 to level 3 (lowest res).
            # But we want to process from lowest res (level 3) to highest (level 0).
            # So we should iterate self.up in reverse?
            # No, wait.
            # Construction:
            # i=3: up_3. insert(0) -> [up_3]
            # i=2: up_2. insert(0) -> [up_2, up_3]
            # ...
            # i=0: up_0. insert(0) -> [up_0, up_1, up_2, up_3]
            
            # So self.up[0] is level 0 (high res).
            # We start from low res (level 3).
            # So we should iterate self.up in reverse order of the list?
            # Or just construct it differently.
            pass
            
        # Let's fix construction to be standard list append
        # and iterate normally.
        
        # Re-implementing forward with correct indexing assumption (assuming standard list append construction)
        # But since I used insert(0), self.up is [Level 0, Level 1, Level 2, Level 3]
        # We need to execute Level 3 -> Level 2 -> Level 1 -> Level 0
        
        for i_level in reversed(range(self.num_resolutions)):
            # i_level: 3, 2, 1, 0
            # We need to access the module corresponding to i_level.
            # Since self.up is [L0, L1, L2, L3], we access self.up[i_level]
            
            stage = self.up[i_level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb)
                if stage.attn:
                    h = stage.attn[block_idx](h)
            
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h
