import contextvars
import math
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_decode_parallel_group_coordinator,
    get_decode_parallel_rank,
    get_decode_parallel_world_size,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

if current_platform.is_cuda():
    from sglang.kernels.ops.diffusion.causal_conv3d_cat_pad import (
        can_use_fused_causal_conv3d_cat_pad_cuda,
        fused_causal_conv3d_cat_pad_cuda,
    )
    from sglang.kernels.ops.diffusion.triton.causal_conv3d_pad import (
        fused_causal_conv3d_cat_pad as fused_causal_conv3d_cat_pad_triton,
    )
else:
    can_use_fused_causal_conv3d_cat_pad_cuda = None
    fused_causal_conv3d_cat_pad_cuda = None
    fused_causal_conv3d_cat_pad_triton = None


_causal_conv3d_cat_pad_cuda_failed = False


def fused_causal_conv3d_cat_pad(
    x: torch.Tensor,
    cache_x: torch.Tensor,
    padding: list[int],
) -> torch.Tensor:
    global _causal_conv3d_cat_pad_cuda_failed
    if (
        fused_causal_conv3d_cat_pad_cuda is not None
        and can_use_fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
        and not _causal_conv3d_cat_pad_cuda_failed
    ):
        try:
            return fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
        except Exception:
            logger.warning(
                "fused_causal_conv3d_cat_pad_cuda failed, falling back to Triton",
                exc_info=True,
            )
            _causal_conv3d_cat_pad_cuda_failed = True
    if fused_causal_conv3d_cat_pad_triton is None:
        raise RuntimeError("causal Conv3D cat/pad fusion is only available on CUDA")
    return fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)


_SPATIAL_PARALLEL_DECODE_DISABLED = contextvars.ContextVar(
    "spatial_parallel_decode_disabled", default=False
)


@contextmanager
def disable_spatial_parallel_decode():
    token = _SPATIAL_PARALLEL_DECODE_DISABLED.set(True)
    try:
        yield
    finally:
        _SPATIAL_PARALLEL_DECODE_DISABLED.reset(token)


def spatial_parallel_decode_disabled() -> bool:
    return _SPATIAL_PARALLEL_DECODE_DISABLED.get()


def _tensor_pad(x: torch.Tensor, len_to_pad: int, dim: int = -2):
    return torch.cat(
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


def _tensor_chunk(x: torch.Tensor, dim: int = -2, world_size: int = 1, rank: int = 0):
    if x is None:
        return x
    if world_size <= 1:
        return x
    return torch.tensor_split(x, world_size, dim=dim)[rank].contiguous(
        memory_format=_halo_memory_format(x)
    )


def _can_fuse_causal_conv3d_cat_pad(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: list[int],
) -> bool:
    if cache_x is None or fused_causal_conv3d_cat_pad is None:
        return False
    if not current_platform.is_cuda():
        return False
    if not x.is_cuda or not x.is_contiguous() or not cache_x.is_contiguous():
        return False
    if x.dim() != 5 or cache_x.dim() != 5 or x.dtype != cache_x.dtype:
        return False
    if x.shape[0] != cache_x.shape[0] or x.shape[1] != cache_x.shape[1]:
        return False
    if x.shape[3:] != cache_x.shape[3:]:
        return False

    width_left, width_right, height_top, height_bottom, depth_left, depth_right = (
        padding
    )
    if width_left != width_right or height_top != height_bottom or depth_right != 0:
        return False
    if depth_left < cache_x.shape[2]:
        return False
    return bool(width_left or height_top)


def causal_conv3d_cat_pad(
    x: torch.Tensor,
    cache_x: torch.Tensor | None,
    padding: list[int],
) -> torch.Tensor:
    if cache_x is not None and padding[4] > 0:
        if cache_x.device != x.device:
            cache_x = cache_x.to(x.device)
        if _can_fuse_causal_conv3d_cat_pad(x, cache_x, padding):
            return fused_causal_conv3d_cat_pad(x, cache_x, padding)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    if any(padding):
        x = F.pad(x, padding)
    return x


def split_for_parallel_decode(
    x: torch.Tensor, upsample_count: int, world_size: int, rank: int
):
    return split_height_for_parallel_decode(
        x,
        expected_height=x.shape[-2] * (2**upsample_count),
        world_size=world_size,
        rank=rank,
    )


def split_height_for_parallel_decode(
    x: torch.Tensor, expected_height: int, world_size: int, rank: int
):
    if spatial_parallel_decode_disabled():
        return x, None
    x = _tensor_chunk(x, dim=-2, world_size=world_size, rank=rank)
    return x, expected_height


def _maybe_contiguous_for_sp_gather(x: torch.Tensor) -> torch.Tensor:
    if (
        x.dim() == 5
        and hasattr(torch, "channels_last_3d")
        and x.is_contiguous(memory_format=torch.channels_last_3d)
        and not x.is_contiguous()
    ):
        return x.contiguous()
    if (
        x.dim() == 4
        and x.is_contiguous(memory_format=torch.channels_last)
        and not x.is_contiguous()
    ):
        return x.contiguous()
    return x


def gather_and_trim_height(x: torch.Tensor, expected_height: int | None):
    if spatial_parallel_decode_disabled():
        return x
    if expected_height is None:
        return x
    x, _ = gather_variable_height(x)
    if x.shape[-2] != expected_height:
        x = x[..., :expected_height, :].contiguous()
    return x


def gather_height_for_global_op(x: torch.Tensor) -> torch.Tensor:
    if spatial_parallel_decode_disabled():
        return x
    return gather_variable_height(x)[0]


def chunk_height_for_parallel_decode(x: torch.Tensor) -> torch.Tensor:
    if spatial_parallel_decode_disabled():
        return x
    return _tensor_chunk(
        x,
        dim=-2,
        world_size=get_decode_parallel_world_size(),
        rank=get_decode_parallel_rank(),
    )


def chunk_height_by_sizes(x: torch.Tensor, heights: list[int]) -> torch.Tensor:
    if spatial_parallel_decode_disabled():
        return x
    rank = get_decode_parallel_rank()
    start = sum(heights[:rank])
    return x[..., start : start + heights[rank], :].contiguous(
        memory_format=_halo_memory_format(x)
    )


def gather_height_sizes(x: torch.Tensor) -> list[int]:
    """gather heights of sharded feature_maps from peers"""
    if spatial_parallel_decode_disabled():
        return [x.shape[-2]]
    world_size = get_decode_parallel_world_size()
    if world_size <= 1:
        return [x.shape[-2]]
    local_height = torch.tensor([x.shape[-2]], device=x.device, dtype=torch.int64)
    gathered = [torch.empty_like(local_height) for _ in range(world_size)]
    dist.all_gather(
        gathered,
        local_height,
        group=get_decode_parallel_group_coordinator().device_group,
    )
    return [int(height.item()) for height in gathered]


def gather_variable_height(x: torch.Tensor) -> tuple[torch.Tensor, list[int]]:
    if spatial_parallel_decode_disabled():
        return x, [x.shape[-2]]
    world_size = get_decode_parallel_world_size()
    if world_size <= 1:
        return x, [x.shape[-2]]

    heights = gather_height_sizes(x)
    max_height = max(heights)
    if x.shape[-2] < max_height:
        x = _tensor_pad(x, max_height - x.shape[-2], dim=-2)

    gathered = get_decode_parallel_group_coordinator().all_gather(
        _maybe_contiguous_for_sp_gather(x), dim=-2
    )
    chunks = torch.split(gathered, max_height, dim=-2)
    return (
        torch.cat(
            [chunk[..., :height, :] for chunk, height in zip(chunks, heights)], dim=-2
        ),
        heights,
    )


def _halo_memory_format(reference: torch.Tensor) -> torch.memory_format:
    if reference.dim() > 1 and reference.stride(1) == 1:
        if reference.dim() == 5 and hasattr(torch, "channels_last_3d"):
            return torch.channels_last_3d
        if reference.dim() == 4:
            return torch.channels_last
    return torch.contiguous_format


def _ensure_recv_buf(
    recv_buf: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor:
    memory_format = _halo_memory_format(reference)
    if (
        recv_buf is None
        or recv_buf.shape != reference.shape
        or recv_buf.dtype != reference.dtype
        or recv_buf.device != reference.device
        or not recv_buf.is_contiguous(memory_format=memory_format)
    ):
        return torch.empty(
            reference.shape,
            dtype=reference.dtype,
            device=reference.device,
            memory_format=memory_format,
        )
    return recv_buf


def halo_exchange(
    x: torch.Tensor,
    height_halo_size: int = 1,
    recv_top_buf: torch.Tensor | None = None,
    recv_bottom_buf: torch.Tensor | None = None,
    height_pad_mode: str = "zeros",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """exchange(send and recv) top/bottom conv-input halos with adjacent spatial ranks"""
    if spatial_parallel_decode_disabled():
        return x, recv_top_buf, recv_bottom_buf
    if height_halo_size == 0:
        return x, recv_top_buf, recv_bottom_buf

    decode_group = get_decode_parallel_group_coordinator()
    rank = get_decode_parallel_rank()
    world_size = get_decode_parallel_world_size()
    group = decode_group.device_group
    group_ranks = decode_group.ranks

    top_row_ref = x[..., :height_halo_size, :]
    bottom_row_ref = x[..., -height_halo_size:, :]

    recv_top_buf = _ensure_recv_buf(recv_top_buf, top_row_ref)
    recv_bottom_buf = _ensure_recv_buf(recv_bottom_buf, bottom_row_ref)
    p2p_ops = []

    if rank > 0:
        prev_rank = group_ranks[rank - 1]
        top_row = top_row_ref.contiguous(memory_format=_halo_memory_format(top_row_ref))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_top_buf, prev_rank, group))
        p2p_ops.append(dist.P2POp(dist.isend, top_row, prev_rank, group))
    if rank < world_size - 1:
        next_rank = group_ranks[rank + 1]
        bottom_row = bottom_row_ref.contiguous(
            memory_format=_halo_memory_format(bottom_row_ref)
        )
        p2p_ops.append(dist.P2POp(dist.isend, bottom_row, next_rank, group))
        p2p_ops.append(dist.P2POp(dist.irecv, recv_bottom_buf, next_rank, group))

    if p2p_ops:
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    if rank == 0:
        recv_top_buf.copy_(
            _make_boundary_halo(
                x,
                recv_bottom_buf if world_size > 1 else None,
                height_halo_size=height_halo_size,
                is_top=True,
                mode=height_pad_mode,
            )
        )
    if rank == world_size - 1:
        recv_bottom_buf.copy_(
            _make_boundary_halo(
                x,
                recv_top_buf if world_size > 1 else None,
                height_halo_size=height_halo_size,
                is_top=False,
                mode=height_pad_mode,
            )
        )

    return (
        torch.concat([recv_top_buf, x, recv_bottom_buf], dim=-2),
        recv_top_buf,
        recv_bottom_buf,
    )


def _make_boundary_halo(
    x: torch.Tensor,
    neighbor: torch.Tensor | None,
    *,
    height_halo_size: int,
    is_top: bool,
    mode: str,
) -> torch.Tensor:
    if mode == "zeros":
        shape = list(x.shape)
        shape[-2] = height_halo_size
        return torch.zeros(shape, dtype=x.dtype, device=x.device)
    if mode == "replicate":
        edge = x[..., :1, :] if is_top else x[..., -1:, :]
        return edge.expand(*edge.shape[:-2], height_halo_size, edge.shape[-1])
    if mode == "reflect":
        source = x
        if is_top and neighbor is not None:
            source = torch.cat([x, neighbor], dim=-2)
        elif not is_top and neighbor is not None:
            source = torch.cat([neighbor, x], dim=-2)
        if is_top:
            index = torch.arange(
                height_halo_size, 0, -1, device=x.device, dtype=torch.long
            )
        else:
            index = torch.arange(
                source.shape[-2] - 2,
                source.shape[-2] - 2 - height_halo_size,
                -1,
                device=x.device,
                dtype=torch.long,
            )
        return source.index_select(-2, index)
    raise ValueError(f"Unsupported spatial padding mode for parallel decode: {mode}")


def _pad_with_mode(
    x: torch.Tensor, padding: tuple[int, ...], mode: str
) -> torch.Tensor:
    if mode == "zeros":
        return F.pad(x, padding)
    return F.pad(x, padding, mode=mode)


def _set_conv_padding(module: nn.Module, padding: tuple[int, ...]) -> None:
    module.padding = padding
    module._reversed_padding_repeated_twice = tuple(
        value for pad in reversed(padding) for value in (pad, pad)
    )


def _conv_preserves_local_height(
    *,
    height_halo_size: int,
    height_pad_top: int,
    height_pad_bottom: int,
    kernel_height: int,
    dilation_height: int,
    stride_height: int,
) -> bool:
    kernel_span = dilation_height * (kernel_height - 1)
    return (
        stride_height == 1
        and 2 * height_halo_size == kernel_span
        and height_pad_top == height_halo_size
        and height_pad_bottom == height_halo_size
    )


def _conv3d_weight_is_channels_last_3d(weight: torch.Tensor) -> bool:
    return (
        weight.dim() == 5
        and hasattr(torch, "channels_last_3d")
        and (current_platform.is_cuda() or current_platform.is_rocm())
        and weight.is_contiguous(memory_format=torch.channels_last_3d)
    )


def _match_conv3d_input_format(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5 and _conv3d_weight_is_channels_last_3d(weight):
        return x.contiguous(memory_format=torch.channels_last_3d)
    return x


def _spatial_parallel_conv_forward(
    module: nn.Module,
    x: torch.Tensor,
    conv_forward,
    *,
    height_pad_mode: str,
    match_conv3d_format: bool = False,
) -> torch.Tensor:
    # send and recv halo
    # x_padded: concatenated input
    x_padded, module._halo_recv_top_buf, module._halo_recv_bottom_buf = halo_exchange(
        x,
        height_halo_size=module.height_halo_size,
        recv_top_buf=module._halo_recv_top_buf,
        recv_bottom_buf=module._halo_recv_bottom_buf,
        height_pad_mode=height_pad_mode,
    )
    if match_conv3d_format:
        x_padded = _match_conv3d_input_format(x_padded, module.weight)
    if module.height_halo_size == 0:
        return conv_forward(x_padded)

    stride = module.stride[-2]
    if _conv_preserves_local_height(
        height_halo_size=module.height_halo_size,
        height_pad_top=module.height_pad_top,
        height_pad_bottom=module.height_pad_bottom,
        kernel_height=module.kernel_size[-2],
        dilation_height=module.dilation[-2],
        stride_height=stride,
    ):
        return conv_forward(x_padded)

    heights = gather_height_sizes(x)
    global_start = sum(heights[: module.rank])
    global_height = sum(heights)
    if stride > 1:
        shift = (
            global_start - module.height_halo_size + module.height_pad_top
        ) % stride
        if shift:
            x_padded = x_padded[..., shift:, :]
            global_start += shift
        if match_conv3d_format:
            x_padded = _match_conv3d_input_format(x_padded, module.weight)

    out = conv_forward(x_padded)

    # trim the output to original shape
    return _trim_conv_output_height(
        out,
        local_height=x.shape[-2],
        global_height=global_height,
        global_start=global_start,
        height_halo_size=module.height_halo_size,
        height_pad_top=module.height_pad_top,
        height_pad_bottom=module.height_pad_bottom,
        kernel_height=module.kernel_size[-2],
        dilation_height=module.dilation[-2],
        stride_height=stride,
    )


class SpatialParallelConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        height_padding: tuple[int, int] | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.height_halo_size = (self.dilation[-2] * (self.kernel_size[-2] - 1)) // 2
        if height_padding is None:
            height_padding = (self.padding[-2], self.padding[-2])
        self.height_pad_top, self.height_pad_bottom = height_padding

        self.padding: tuple[int, int]
        if self.height_halo_size > 0:
            self._padding = (0, 0, 0, 0)
        else:
            self._padding = (0, 0, self.padding[0], self.padding[0])

        _set_conv_padding(self, (0, self.padding[1]))
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x):
        if spatial_parallel_decode_disabled():
            return self._direct_forward(x)

        if any(self._padding):
            x = _pad_with_mode(x, self._padding, self.padding_mode)

        return _spatial_parallel_conv_forward(
            self,
            x,
            super().forward,
            height_pad_mode=self.padding_mode,
        )

    def _direct_forward(self, x):
        width_pad = self.padding[-1]
        padding = (
            width_pad,
            width_pad,
            self.height_pad_top,
            self.height_pad_bottom,
        )
        if any(padding):
            x = _pad_with_mode(x, padding, self.padding_mode)
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )


class SpatialParallelCausalConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        self.height_pad_top = self.padding[1]
        self.height_pad_bottom = self.padding[1]
        self.height_halo_size = (self.kernel_size[-2] - 1) // 2

        self.padding: tuple[int, int, int]
        if self.height_halo_size > 0:
            self._padding = (
                self.padding[2],
                self.padding[2],
                0,
                0,
                2 * self.padding[0],
                0,
            )
        else:
            self._padding = (
                self.padding[2],
                self.padding[2],
                self.padding[1],
                self.padding[1],
                2 * self.padding[0],
                0,
            )
        self.padding = (0, 0, 0)
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if spatial_parallel_decode_disabled():
            padding[2] = self.height_pad_top
            padding[3] = self.height_pad_bottom
        x = causal_conv3d_cat_pad(x, cache_x, padding)
        x = x if current_platform.is_amp_supported() else x.to(self.weight.dtype)

        if spatial_parallel_decode_disabled():
            x = _match_conv3d_input_format(x, self.weight)
            return F.conv3d(
                x,
                self.weight,
                self.bias,
                self.stride,
                (0, 0, 0),
                self.dilation,
                self.groups,
            )

        return _spatial_parallel_conv_forward(
            self,
            x,
            super().forward,
            height_pad_mode="zeros",
            match_conv3d_format=True,
        )


class SpatialParallelConv3d(nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        height_padding: tuple[int, int] | None = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.height_halo_size = (self.dilation[-2] * (self.kernel_size[-2] - 1)) // 2
        if height_padding is None:
            height_padding = (self.padding[-2], self.padding[-2])
        self.height_pad_top, self.height_pad_bottom = height_padding

        self.padding: tuple[int, int, int]
        if self.height_halo_size > 0:
            self._padding = (0, 0, 0, 0, 0, 0)
        else:
            self._padding = (
                0,
                0,
                self.padding[1],
                self.padding[1],
                0,
                0,
            )

        _set_conv_padding(self, (self.padding[0], 0, self.padding[2]))
        self._halo_recv_top_buf: torch.Tensor | None = None
        self._halo_recv_bottom_buf: torch.Tensor | None = None
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x):
        if spatial_parallel_decode_disabled():
            return self._direct_forward(x)

        if any(self._padding):
            x = _pad_with_mode(x, self._padding, self.padding_mode)

        return _spatial_parallel_conv_forward(
            self,
            x,
            super().forward,
            height_pad_mode=self.padding_mode,
            match_conv3d_format=True,
        )

    def _direct_forward(self, x):
        time_pad = self.padding[0]
        width_pad = self.padding[-1]
        padding = (
            width_pad,
            width_pad,
            self.height_pad_top,
            self.height_pad_bottom,
            time_pad,
            time_pad,
        )
        if any(padding):
            x = _pad_with_mode(x, padding, self.padding_mode)
        x = _match_conv3d_input_format(x, self.weight)
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, 0, 0),
            self.dilation,
            self.groups,
        )


class SpatialParallelZeroPad2d(nn.Module):
    def __init__(self, padding: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.padding = padding
        self.rank = get_decode_parallel_rank()
        self.world_size = get_decode_parallel_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if spatial_parallel_decode_disabled():
            return F.pad(x, self.padding)
        left, right, top, bottom = self.padding
        top = top if self.rank == 0 else 0
        bottom = bottom if self.rank == self.world_size - 1 else 0
        return F.pad(x, (left, right, top, bottom))


def _trim_conv_output_height(
    out: torch.Tensor,
    *,
    local_height: int,
    global_height: int,
    global_start: int,
    height_halo_size: int,
    height_pad_top: int,
    height_pad_bottom: int,
    kernel_height: int,
    dilation_height: int,
    stride_height: int,
) -> torch.Tensor:
    kernel_span = dilation_height * (kernel_height - 1)
    min_i = math.ceil(
        ((-height_pad_top) - (global_start - height_halo_size)) / stride_height
    )
    max_i = math.floor(
        (
            (global_height - 1 + height_pad_bottom)
            - kernel_span
            - (global_start - height_halo_size)
        )
        / stride_height
    )
    start = max(min_i, 0)
    end = min(max_i + 1, out.shape[-2])
    if start != 0 or end != out.shape[-2]:
        out = out[..., start:end, :]
    return out
