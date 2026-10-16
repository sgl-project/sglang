# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The attention projections (wqkv_a, wq_b, wo_b) at 1-8 rows on FlashInfer's
``cutedsl_low_latency`` MXFP8 GEMM instead of the persistent CuTe DSL kernel, with
the tactic pinned per GB300 shape."""

from __future__ import annotations

import contextlib
import importlib
import logging
from functools import lru_cache
from typing import Iterator, Optional

import torch
from torch import nn

from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.models.dsv41_low_latency.switch import MXFP8_MAX_ROWS

logger = logging.getLogger(__name__)

BACKEND = "cutedsl_low_latency"
_GEMM_MODULE = "flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm"
# (cta_k, ab_stages, sfb_tmem_stages, split_k) per (SM, N, K) for 5-8 rows: the TP4
# wqkv_a, wq_b and wo_b of DeepSeek-V4.1-Flash on GB300, picked jointly in a 40-layer
# chain. FlashInfer tunes 5-8 rows as its bucket 8.
PINNED_TACTICS: dict[tuple[int, int, int], tuple[int, int, int, int]] = {
    (103, 1792, 5120): (128, 10, 1, 4),
    (103, 8192, 1280): (256, 6, 1, 1),
    (103, 5120, 2048): (256, 4, 1, 2),
}


@lru_cache(maxsize=1)
def supported() -> bool:
    from sglang.srt.layers.quantization.fp8_utils import _raw_flashinfer_mm_mxfp8
    from sglang.srt.runtime_context import get_platform

    try:
        return bool(
            _raw_flashinfer_mm_mxfp8.is_backend_supported(
                BACKEND, get_platform().device_sm
            )
        )
    except (AttributeError, ImportError, RuntimeError):
        return False


def _linear(
    layer: nn.Module, x, weight_scale: torch.Tensor, bias
) -> Optional[torch.Tensor]:
    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_mxfp8_blockscaled_linear,
    )

    data = x[0] if isinstance(x, tuple) else x
    rows = data.numel() // data.shape[-1]
    if not (
        0 < rows <= MXFP8_MAX_ROWS
        and data.shape[-1] % 128 == 0
        and data.is_contiguous()
    ):
        return None
    return flashinfer_mxfp8_blockscaled_linear(
        input=data,
        weight=layer.weight,
        weight_scale=weight_scale.reshape(-1),
        input_scale=x[1].reshape(-1) if isinstance(x, tuple) else None,
        bias=bias,
        backend=BACKEND,
    )


def install(linears: tuple[nn.Module, ...]) -> None:
    """Route each MXFP8 linear's 1-8-row calls through the low-latency GEMM."""
    if not supported():
        logger.warning(f"FlashInfer mm_mxfp8 has no {BACKEND} backend here; skipped.")
        return
    for linear in linears:
        if isinstance(linear.quant_method, Fp8LinearMethod):
            linear.quant_method.mxfp8_cutedsl_override = _linear


def _pinned_tactic(*, sm: int, n: int, k: int, rows: int):
    if not MXFP8_MAX_ROWS // 2 < rows <= MXFP8_MAX_ROWS:
        return None
    return PINNED_TACTICS.get((sm, n, k))


@contextlib.contextmanager
def pinned_tactics() -> Iterator[None]:
    """While tuning, FlashInfer's low-latency tactic list is the pinned tactic alone
    where one applies, so the autotuner caches it under its usual key on every rank."""
    from sglang.srt.runtime_context import get_platform

    sm = get_platform().device_sm
    gemm = importlib.import_module(_GEMM_MODULE)
    enumerate_tactics = gemm.autotune_tactics

    def autotune_tactics(problem_sizes_mnkl, *args, **kwargs):
        tactics = enumerate_tactics(problem_sizes_mnkl, *args, **kwargs)
        # The runner computes out.T, so the problem is (N, M, K, L).
        n, rows, k, _ = problem_sizes_mnkl
        pinned = _pinned_tactic(sm=sm, n=n, k=k, rows=rows)
        if pinned is None or pinned not in tactics:
            return tactics
        return [pinned]

    gemm.autotune_tactics = autotune_tactics
    try:
        yield
    finally:
        gemm.autotune_tactics = enumerate_tactics


def autotune(model: nn.Module, *, dtype: torch.dtype) -> None:
    """Tune buckets 1, 2, 4 and 8 with one 8-row call per weight shape; inside the
    FlashInfer autotune context and `pinned_tactics`. The DSpark draft shares these
    shapes and the process-wide tactic cache."""
    from sglang.srt.layers.linear import LinearBase

    by_shape = {}
    for module in model.modules():
        if (
            isinstance(module, LinearBase)
            and isinstance(module.quant_method, Fp8LinearMethod)
            and module.quant_method.mxfp8_cutedsl_override is not None
        ):
            by_shape.setdefault(tuple(module.weight.shape), module)
    # Sorted, so every rank of the tactic-timing reduce makes the same calls in order.
    for shape in sorted(by_shape):
        layer = by_shape[shape]
        x = torch.zeros(
            (MXFP8_MAX_ROWS, shape[1]), dtype=dtype, device=layer.weight.device
        )
        layer.quant_method.apply(layer, x)
