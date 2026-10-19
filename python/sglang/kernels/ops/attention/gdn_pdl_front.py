# SPDX-License-Identifier: Apache-2.0
# JIT wrapper for kernels/jit/csrc/attention/gdn_pdl_front.cuh.
"""FP8 projection and BF16 convolution for seven-token linear-attention verification."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gdn_pdl_front_module() -> Module:
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError(
            "gdn_pdl_front requires CUDA compute capability 9.0 or newer"
        )
    return load_jit(
        "gdn_pdl_front",
        cuda_files=["attention/gdn_pdl_front.cuh"],
        cuda_wrappers=[("front", "gdn_pdl_front")],
        extra_cuda_cflags=["-lineinfo"],
    )


def ensure_built() -> None:
    """Compile both input-dtype specializations before model loading or graph capture."""
    _jit_gdn_pdl_front_module()


def pack(weight_nk: torch.Tensor) -> torch.Tensor:
    """Pack checkpoint FP8 weights shaped [2560,4096] before graph capture."""
    if weight_nk.dtype != torch.float8_e4m3fn or weight_nk.shape != (2560, 4096):
        raise ValueError("projection weights must be FP8 E4M3 with shape [2560,4096]")
    value = weight_nk.contiguous().view(torch.uint8)
    value = value.reshape(10, 8, 2, 2, 8, 8, 16, 2, 4, 4)
    return value.permute(0, 5, 1, 6, 2, 4, 8, 7, 3, 9).contiguous().view(-1)


def front(
    input: torch.Tensor,
    packed: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    windows: torch.Tensor,
    state_indices: torch.Tensor,
    replay_indices: torch.Tensor,
    qkvz_out: torch.Tensor,
    conv_out: torch.Tensor,
    error: torch.Tensor,
    *,
    pad_slot: int = -1,
) -> None:
    """Write preallocated projection/conv buffers and update the selected rolling state.

    Input is contiguous BF16 or FP8 E4M3 [7,4096]. Projection output is BF16
    [7,2560] in q256/k256/v1024/z1024 order; convolution output is BF16
    [7,1536]. Windows have logical shape [slots,7,1536,3] and strides
    [slot_stride,1,9,1]. Both index tensors hold one device int32/int64.
    The caller initializes the int32 error word to zero before capture and
    checks it outside the captured path. BF16 b/a projection remains separate.
    """
    _jit_gdn_pdl_front_module().front(
        input,
        packed,
        input_scale,
        weight_scale,
        conv_weight,
        conv_state,
        windows,
        state_indices,
        replay_indices,
        qkvz_out,
        conv_out,
        error,
        pad_slot,
    )
