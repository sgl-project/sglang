import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.parallel.spatial_parallel import (
    SpatialParallelCausalConv3d,
    SpatialParallelConv2d,
    SpatialParallelZeroPad2d,
    chunk_height_for_parallel_decode,
    gather_and_trim_height,
    gather_height_for_global_op,
    split_for_parallel_decode,
)
from sglang.multimodal_gen.runtime.models.vaes.parallel.wan_common_utils import (
    AvgDown3D,
    DupUp3D,
    WanCausalConv3d,
    WanRMS_norm,
    WanUpsample,
    attention_block_forward,
    mid_block_forward,
    resample_forward,
    residual_block_forward,
    residual_down_block_forward,
    residual_up_block_forward,
    up_block_forward,
)


def tensor_pad(x: torch.Tensor, len_to_pad: int, dim: int = -2):
    x = torch.cat(
        [
            x,
            torch.zeros(
                *x.shape[:dim],
                len_to_pad,
                *x.shape[dim + 1 :],
                dtype=x.dtype,
                device=x.device,
            ),
        ],
        dim=dim,
    )
    return x


def tensor_chunk(x: torch.Tensor, dim: int = -2, world_size: int = 1, rank: int = 0):
    if x is None:
        return None
    if world_size <= 1:
        return x
    len_to_padding = (int(math.ceil(x.shape[dim] / world_size)) * world_size) - x.shape[
        dim
    ]
    if len_to_padding != 0:
        x = tensor_pad(x, len_to_padding, dim=dim)
    return torch.chunk(x, world_size, dim=dim)[rank]


def split_for_parallel_encode(
    x: torch.Tensor, downsample_count: int, world_size: int, rank: int
):
    orig_height = x.shape[-2]
    expected_height = orig_height // (2**downsample_count)
    factor = world_size * (2**downsample_count)
    pad_h = (factor - orig_height % factor) % factor
    if pad_h:
        x = F.pad(x, (0, 0, 0, pad_h, 0, 0))
    expected_local_height = (orig_height + pad_h) // (2**downsample_count) // world_size
    x = tensor_chunk(x, dim=-2, world_size=world_size, rank=rank)
    return x, expected_height, expected_local_height


def ensure_local_height(x: torch.Tensor, expected_local_height: int | None):
    if expected_local_height is None:
        return x
    if x.shape[-2] < expected_local_height:
        pad = expected_local_height - x.shape[-2]
        return F.pad(x, (0, 0, 0, pad, 0, 0))
    if x.shape[-2] > expected_local_height:
        return x[..., :expected_local_height, :].contiguous()
    return x


class WanDistConv2d(SpatialParallelConv2d):
    pass


class WanDistCausalConv3d(SpatialParallelCausalConv3d):
    pass


class WanDistZeroPad2d(SpatialParallelZeroPad2d):
    pass


class WanDistResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data used for parallel decoding.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        # We support parallel encode/decode; downsample uses halo exchange as well.
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanDistConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanDistConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                WanDistZeroPad2d((0, 1, 0, 0)),
                WanDistConv2d(dim, dim, 3, stride=(2, 2), height_padding=(0, 1)),
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                WanDistZeroPad2d((0, 1, 0, 0)),
                WanDistConv2d(dim, dim, 3, stride=(2, 2), height_padding=(0, 1)),
            )
            self.time_conv = WanCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x):
        return resample_forward(self, x)


class WanDistResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_act_fn(non_linearity)

        # layers
        self.norm1 = WanRMS_norm(in_dim, images=False)
        self.conv1 = WanDistCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanDistCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanDistCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x):
        return residual_block_forward(self, x)


class WanDistAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim

        # layers
        self.norm = WanRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x):
        if self.world_size > 1:
            x = gather_height_for_global_op(x).contiguous()
        x = attention_block_forward(self, x)
        if self.world_size > 1:
            x = chunk_height_for_parallel_decode(x)

        return x


class WanDistMidBlock(nn.Module):
    """
    Middle block for WanVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [WanDistResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanDistAttentionBlock(dim))
            resnets.append(WanDistResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x):
        return mid_block_forward(self, x)


class WanDistResidualDownBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout,
        num_res_blocks,
        temperal_downsample=False,
        down_flag=False,
    ):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanDistResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanDistResample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x):
        return residual_down_block_forward(self, x)


class WanDistResidualUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        temperal_upsample (bool): Whether to upsample on temporal dimension
        up_flag (bool): Whether to upsample or not
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2,
            )
        else:
            self.avg_shortcut = None

        # create residual blocks
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanDistResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        if up_flag:
            upsample_mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanDistResample(
                out_dim, mode=upsample_mode, upsample_out_dim=out_dim
            )
        else:
            self.upsampler = None

        self.gradient_checkpointing = False

    def forward(self, x):
        return residual_up_block_forward(self, x)


class WanDistUpBlock(nn.Module):
    """
    A block that handles upsampling for the WanVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanDistResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [WanDistResample(out_dim, mode=upsample_mode)]
            )

        self.gradient_checkpointing = False

    def forward(self, x):
        return up_block_forward(self, x)
