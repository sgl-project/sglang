# SPDX-License-Identifier: Apache-2.0
# JIT wrapper for kernels/jit/csrc/attention/gdn_pdl_front_ba.cuh.
"""Preallocated FP8 projection, rolling convolution and BF16 side projection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gdn_pdl_front_ba_module() -> Module:
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError(
            "gdn_pdl_front_ba requires CUDA compute capability 9.0 or newer"
        )
    return load_jit(
        "gdn_pdl_front_ba",
        cuda_files=["attention/gdn_pdl_front_ba.cuh"],
        cuda_wrappers=[("front", "gdn_pdl_front_ba")],
        extra_cuda_cflags=["-lineinfo"],
    )


def ensure_built() -> None:
    """Compile both projection-input dtypes before model loading or graph capture."""
    _jit_gdn_pdl_front_ba_module()


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
    original_bf16: torch.Tensor,
    ba_weight: torch.Tensor,
    ba_out: torch.Tensor,
    pad_slot: int = -1,
) -> None:
    """Write projection, convolution and b/a results into caller-owned buffers.

    ``input`` is contiguous BF16 or FP8 E4M3 [7,4096]. ``original_bf16``
    independently supplies the original BF16 [7,4096] activation for b/a;
    its last dimension is contiguous and row starts are four-byte aligned.
    ``ba_weight`` is contiguous BF16 [16,4096], and ``ba_out`` is contiguous
    BF16 [7,16] in b8/a8 order. The b/a projection never reconstructs BF16
    from quantized input.

    Projection/conv outputs and state/window layouts match ``gdn_pdl_front``.
    Padding skips the selected convolution state/window/output writes; b/a
    remains computed. All writable buffers must be disjoint from operands.
    Initialize the int32 error word to zero outside capture; device failures
    trap. No output or workspace is allocated by this entry point.
    """
    _jit_gdn_pdl_front_ba_module().front(
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
        original_bf16,
        ba_weight,
        ba_out,
        pad_slot,
    )
