# SPDX-License-Identifier: Apache-2.0
"""Lazy-built CUDA kernels used by the expert-pack MoE runtime."""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from torch.utils.cpp_extension import load

from sglang.kernels.jit.utils import KERNEL_PATH

_EXTENSION_NAME = "sglang_expert_pack_mxfp4"


@lru_cache(maxsize=1)
def _extension():
    source = KERNEL_PATH / "csrc" / "moe" / "expert_pack_mxfp4.cu"
    return load(
        name=_EXTENSION_NAME,
        sources=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=os.getenv("SGLANG_EXPERT_PACK_BUILD_VERBOSE", "0") == "1",
    )


def mxfp4_matvec(
    x: torch.Tensor,
    cache: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    role_offset: int,
    role_bytes: int,
    input_size: int,
    output_size: int,
    records_per_input: int,
) -> torch.Tensor:
    """Multiply selected raw GGUF MXFP4 matrices by BF16/FP16 rows."""

    return _extension().mxfp4_matvec(
        x,
        cache,
        slot_ids,
        role_offset,
        role_bytes,
        input_size,
        output_size,
        records_per_input,
    )


def mxfp4_matvec_dual(
    x: torch.Tensor,
    cache: torch.Tensor,
    slot_ids: torch.Tensor,
    *,
    gate_role_offset: int,
    up_role_offset: int,
    role_bytes: int,
    input_size: int,
    output_size: int,
    records_per_input: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute gate and up projections while loading each input row once."""

    return _extension().mxfp4_matvec_dual(
        x,
        cache,
        slot_ids,
        gate_role_offset,
        up_role_offset,
        role_bytes,
        input_size,
        output_size,
        records_per_input,
    )


def prewarm_mxfp4_extension() -> None:
    """Build and load the extension before the server accepts requests."""

    _extension()


def mxfp4_marlin_repack(
    raw: torch.Tensor,
    source_slots: torch.Tensor,
    target_slots: torch.Tensor,
    *,
    role_bytes: int,
    hidden_size: int,
    intermediate_size: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> None:
    """Repack raw GGUF objects into contiguous Marlin SoA cache tensors."""

    _extension().mxfp4_marlin_repack(
        raw,
        source_slots,
        target_slots,
        role_bytes,
        hidden_size,
        intermediate_size,
        w13,
        w2,
        w13_scale,
        w2_scale,
    )
