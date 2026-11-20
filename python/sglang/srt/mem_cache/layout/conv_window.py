"""Speculative conv windows of a linear verify chain, stored as one row of
D + K - 2 columns per channel: step t's window is columns t..t+K-2."""

from typing import Optional, Sequence

import torch


def deduplicated_conv_window_axis(
    cache_params, *, supported_device: bool, speculative_eagle_topk: Optional[int]
) -> Optional[int]:
    # Tree steps have different parents, so they cannot share columns. NPU and
    # CPU kernels expect contiguous steps, KDA transposes the window before conv,
    # and a shape can opt out: each needs independently written windows.
    if (
        not supported_device
        or (speculative_eagle_topk is not None and speculative_eagle_topk > 1)
        or cache_params.shape.disable_conv_window_dedup
        or cache_params.is_kda
    ):
        return None
    win = cache_params.shape.conv_kernel - 1
    if win == 0:
        return None
    axis = None
    for shape in cache_params.shape.conv:
        # A square shape matches both ends; the trailing axis (GDN's layout) wins.
        if shape[-1] == win:
            shape_axis = len(shape) - 1
        elif shape[0] == win:
            shape_axis = 0
        else:
            raise ValueError(f"conv shape {shape} has no window axis of length {win}")
        if axis is not None and axis != shape_axis:
            raise ValueError(
                f"inconsistent window axes across {cache_params.shape.conv}"
            )
        axis = shape_axis
    return axis


def conv_window_storage_shape(
    conv_shape: Sequence[int], draft_tokens: int, window_axis: Optional[int]
) -> tuple[int, ...]:
    if window_axis is None:
        return (draft_tokens, *conv_shape)
    shape = list(conv_shape)
    shape[window_axis] += draft_tokens - 1
    return tuple(shape)


def conv_window_view(
    storage: torch.Tensor, conv_shape: Sequence[int], window_axis: int
) -> torch.Tensor:
    axis = 2 + window_axis % len(conv_shape)
    draft_tokens = storage.size(axis) - conv_shape[window_axis] + 1
    # Advancing one draft step slides by one column on the window axis.
    strides = storage.stride()
    return storage.as_strided(
        (*storage.shape[:2], draft_tokens, *conv_shape),
        (*strides[:2], strides[axis], *strides[2:]),
    )


def conv_window_storage_bytes(windows: Sequence[torch.Tensor]) -> int:
    # Each leading index owns one consecutive block, so size(0) * stride(0) is
    # the physical extent; the storage bound covers empty tensors, whose strides
    # are not zero.
    return sum(
        min(t.untyped_storage().nbytes(), t.size(0) * t.stride(0) * t.element_size())
        for t in windows
    )
