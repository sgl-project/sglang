# SPDX-License-Identifier: Apache-2.0
# JIT wrapper for kernels/jit/csrc/attention/gdn_pdl_front_ba_mkm.cuh.
"""``gdn_pdl_front_ba`` with the same outputs and arithmetic, plus an L2 prefetch of the recurrent operands."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gdn_pdl_front_ba_mkm_module() -> Module:
    if torch.cuda.get_device_capability()[0] < 9:
        raise RuntimeError(
            "gdn_pdl_front_ba_mkm requires CUDA compute capability 9.0 or newer"
        )
    return load_jit(
        "gdn_pdl_front_ba_mkm",
        cuda_files=["attention/gdn_pdl_front_ba_mkm.cuh"],
        cuda_wrappers=[("front", "gdn_pdl_front_ba_mkm")],
        extra_cuda_cflags=["-lineinfo"],
    )


def ensure_built() -> None:
    """Compile both projection-input dtypes before model loading or graph capture."""
    _jit_gdn_pdl_front_ba_mkm_module()


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
    checkpoint: torch.Tensor,
    history_d: torch.Tensor,
    history_k: torch.Tensor,
    history_g: torch.Tensor,
    ring_indices: torch.Tensor,
    pad_slot: int = -1,
) -> None:
    """Same contract and bytes as ``gdn_pdl_front_ba.front``; also prefetches recurrent operands into L2.

    ``checkpoint`` is indexed by ``state_indices[0]``; ``history_d``, ``history_k`` and ``history_g`` by
    ``ring_indices[0]`` (one device int32/int64). Only slot-contiguous operands are prefetched. The
    prefetch is a hint: nothing is written, and the consumer still reads after its own dependency wait.
    """
    _jit_gdn_pdl_front_ba_mkm_module().front(
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
        checkpoint,
        history_d,
        history_k,
        history_g,
        ring_indices,
        pad_slot,
    )
