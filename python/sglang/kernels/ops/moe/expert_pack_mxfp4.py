# SPDX-License-Identifier: Apache-2.0
"""Lazy-built CUDA kernels used by the expert-pack MoE runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_CUDA_FILE = "moe/expert_pack_mxfp4.cuh"


@cache_once
def _extension() -> Module:
    """Compile and cache the expert-pack MXFP4 module.

    Both element types are instantiated in one module: a decode step calls the
    matvec kernels for whichever dtype the model runs in, and splitting the
    build per dtype would only trade one compile for two.
    """
    return load_jit(
        "expert_pack_mxfp4",
        cuda_files=[_CUDA_FILE],
        cuda_wrappers=[
            ("mxfp4_matvec", "mxfp4_matvec"),
            ("mxfp4_matvec_dual", "mxfp4_matvec_dual"),
            ("mxfp4_marlin_repack", "mxfp4_marlin_repack"),
        ],
        extra_cuda_cflags=["--use_fast_math"],
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

    out = x.new_empty((slot_ids.numel(), output_size))
    _extension().mxfp4_matvec(
        out,
        x,
        cache,
        slot_ids,
        role_offset,
        role_bytes,
        input_size,
        output_size,
        records_per_input,
    )
    return out


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

    shape = (slot_ids.numel(), output_size)
    out_gate = x.new_empty(shape)
    out_up = x.new_empty(shape)
    _extension().mxfp4_matvec_dual(
        out_gate,
        out_up,
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
    return out_gate, out_up


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
