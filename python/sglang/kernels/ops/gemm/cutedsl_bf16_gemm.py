# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# Copyright 2026 SGLang Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
"""CuTe DSL low-latency BF16 GEMM dispatch.

Computes ``out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N])`` for bf16 inputs,
fp32 accumulation, bf16 or fp32 output. SGLang picks the tactic per shape
(``_pick_tactic``, model M ranges); FlashInfer's ``mm_bf16(backend="tgv")``
runs it.
"""

from __future__ import annotations

import bisect
from typing import Optional

import torch
from flashinfer import mm_bf16
from flashinfer.gemm.kernels.tgv_gemm_cute_ext import get_tgv_cute_ext_valid_tactics

from sglang.kernels.arch import current_arch
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_sm100_supported
from sglang.srt.utils.custom_op import direct_register_custom_op


def _run_tgv(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    out: torch.Tensor,
    pdl: bool,
    tactic: int,
) -> torch.Tensor:
    """Dispatch one GEMM: a (M, K) @ b (K, N) -> out (M, N), bias (N,).
    skip_check=True skips FlashInfer's checks, so inputs are trusted: bf16
    K-major a and b, out of shape (M, N) in bf16 or fp32, K % 8 == 0, a bf16
    bias of N elements, all on one device, and a valid tactic."""
    if bias is not None:
        bias = bias.detach()  # DLPack rejects tensors that require grad
    return mm_bf16(
        a,
        b,
        bias,
        pdl,
        out=out,
        out_dtype=out.dtype,
        backend="tgv",
        tactic=tactic,
        skip_check=True,
    )


# (tactic, cta_m, cta_n, num_mma_ctas), smallest cta_n first.
_TGV_TACTIC_LADDER = (
    (18, 64, 16, 2),
    (21, 64, 32, 2),
    (23, 64, 64, 2),
    (24, 64, 128, 2),
    (12, 128, 16, 1),
    (26, 128, 32, 2),
    (27, 128, 64, 2),
)


def _grid_ctas(m: int, n: int, cta_m: int, cta_n: int, num_mma_ctas: int) -> int:
    # Kernel axes are swapped: kernel-M = pytorch N, kernel-N = pytorch M.
    cluster_m = cta_m * num_mma_ctas
    return num_mma_ctas * -(n // -cluster_m) * -(m // -cta_n)


def _pick_tactic(m: int, n: int, k: int) -> int:
    """The tactic a model registered for this m (register_tgv_m_ranges), else
    the first ladder tactic whose grid fits one wave of the SMs."""
    tactic = _model_tgv_tactic(m, n, k)
    if tactic is not None:
        return tactic
    num_sms = current_arch().num_sms
    best, best_ctas = None, None
    for tactic, cta_m, cta_n, num_mma_ctas in _TGV_TACTIC_LADDER:
        ctas = _grid_ctas(m, n, cta_m, cta_n, num_mma_ctas)
        if ctas <= num_sms:
            return tactic
        if best_ctas is None or ctas < best_ctas:
            best, best_ctas = tactic, ctas
    return best


# Kimi-K3 per-rank dense-GEMM shapes (TP8). They sit in this heuristic's
# unmeasured region (hidden=7168 inputs fail k > 6144; o_proj-style k=1536
# fails k < 2048), but TGV wins 1.04-2.43x on every one of them on GB300
# (L2-defeating weight rotation + CUDA-graph timing; serving A/B confirmed
# e2e with GSM8K parity). kv_a (n=576) loses (0.84x) and stays out. Gated to
# small decode batches; larger m stays on the measured heuristic below.
_K3_TGV_WIN_SHAPES = frozenset(
    {
        (6144, 7168),  # KDA fused qkvg
        (6016, 7168),  # merged MoE front (gate_up | router | latent down)
        (7168, 1536),  # KDA / MLA o_proj
        (1536, 7168),  # MLA q_a / shared gate_up
        (2304, 1536),  # MLA q_b
        (3584, 7168),  # MoE latent down (unfused fallback)
        (7168, 3584),  # MoE latent up
        (7168, 768),  # shared down
        (7168, 1792),  # DSpark draft down
        (7168, 512),  # DSpark draft o_proj
        (8448, 7168),  # dense MLP gate_up
    }
)

_K3_FRONT_WARP_SPLITK = {
    8: (8, 11),
    16: (16, 9),
    32: (32, 6),
}


class TGVMRanges:
    """Sorted, disjoint m ranges of one dense-GEMM shape that take TGV, each
    with a tactic (None: the ladder pick). ``m in ranges`` says whether m takes
    TGV; ``ranges(m)`` returns its tactic."""

    def __init__(self) -> None:
        self._m_lo: list[int] = []
        self._m_hi: list[int] = []
        self._tactic: list[Optional[int]] = []

    def append(self, m_lo: int, m_hi: int, tactic: Optional[int]) -> None:
        """Insert in m order; re-adding an existing range is a no-op, any other
        overlap raises."""
        i = bisect.bisect_left(self._m_lo, m_lo)
        n = len(self._m_lo)
        if i < n and (self._m_lo[i], self._m_hi[i], self._tactic[i]) == (
            m_lo,
            m_hi,
            tactic,
        ):
            return
        if (
            not 0 < m_lo <= m_hi
            or (i > 0 and self._m_hi[i - 1] >= m_lo)
            or (i < n and self._m_lo[i] <= m_hi)
        ):
            raise ValueError(
                f"TGV m range [{m_lo}, {m_hi}] overlaps "
                f"{list(zip(self._m_lo, self._m_hi))}"
            )
        if tactic is not None and tactic not in get_tgv_cute_ext_valid_tactics(
            torch.device("cuda")
        ):
            raise ValueError(f"tactic {tactic} is not valid on this GPU")
        self._m_lo.insert(i, m_lo)
        self._m_hi.insert(i, m_hi)
        self._tactic.insert(i, tactic)

    def _index(self, m: int) -> int:
        """The index of the range holding m, or -1."""
        i = bisect.bisect_right(self._m_lo, m) - 1
        return i if i >= 0 and m <= self._m_hi[i] else -1

    def __contains__(self, m: int) -> bool:
        return self._index(m) >= 0

    def __call__(self, m: int) -> Optional[int]:
        i = self._index(m)
        return self._tactic[i] if i >= 0 else None


# (n, k) -> the m ranges a model measured; filled by register_tgv_m_ranges.
_MODEL_TGV_M_RANGES: dict[tuple[int, int], TGVMRanges] = {}


def register_tgv_m_ranges(
    shapes: dict[tuple[int, int], tuple[tuple[int, int, Optional[int]], ...]],
) -> None:
    """Route a model's dense-GEMM shapes ``(n, k)`` to TGV for the given
    ``(m_lo, m_hi, tactic)`` ranges, for every caller of the shape."""
    for shape, ranges in shapes.items():
        m_ranges = _MODEL_TGV_M_RANGES.setdefault(shape, TGVMRanges())
        for m_lo, m_hi, tactic in ranges:
            m_ranges.append(m_lo=m_lo, m_hi=m_hi, tactic=tactic)


def _model_takes_tgv(m: int, n: int, k: int) -> bool:
    m_ranges = _MODEL_TGV_M_RANGES.get((n, k))
    return m_ranges is not None and m in m_ranges


def _model_tgv_tactic(m: int, n: int, k: int) -> Optional[int]:
    m_ranges = _MODEL_TGV_M_RANGES.get((n, k))
    return None if m_ranges is None else m_ranges(m)


# MuseSpark TP4/TP8 projections using TGV on GB300; other M use cuBLAS.
_MUSE_SPARK_TGV_M_RANGES = {
    (4096, 8192): (1, 16),  # replicated pooler
    (8192, 4096): (1, 128),  # replicated unpooler
    (4608, 8192): (1, 256),  # TP4 qkvg
    (8192, 2048): (1, 128),  # TP4 o_proj
    (1024, 8192): (1, 256),  # TP4 pooler slice
    (2048, 4096): (1, 256),  # TP4 unpooler slice
    (2304, 8192): (40, 512),  # TP8 qkvg
    (8192, 1024): (1, 64),  # TP8 o_proj
    (512, 8192): (256, 512),  # TP8 pooler slice
    (1024, 4096): (1, 512),  # TP8 unpooler slice
    (14336, 8192): (24, 32),  # hidden 8192 / inter 28672 TP4 gate_up (DFLASH draft)
    (50512, 8192): (17, 32),  # DFLASH draft TP4 candidate head
    (7168, 8192): (24, 32),  # hidden 8192 / inter 28672 TP8 gate_up (DFLASH draft)
}


def use_cutedsl_bf16_gemm(m: int, n: int, k: int) -> bool:
    """TGV-vs-cuBLAS (``F.linear``) decision, CUPTI-measured on B300 under CUDA
    graph capture (cold L2). Conservative: ties and unmeasured regions fall
    back to cuBLAS."""
    if m <= 0:
        # Empty batch: DP-attention idle groups run a 0-token dummy forward to
        # keep the mlp-sync lockstep. A 0-CTA TGV grid is a driver-level
        # CUDA_ERROR_INVALID_VALUE, so leave empty inputs to cuBLAS.
        return False
    if k % 8 != 0:  # TMA requires 16B-aligned rows
        return False
    m_range = _MUSE_SPARK_TGV_M_RANGES.get((n, k))
    if m_range is not None:
        return m_range[0] <= m <= m_range[1]
    if (
        current_arch().is_sm(10, 7)
        and m in _K3_FRONT_WARP_SPLITK
        and (n, k)
        == (
            6016,
            7168,
        )
    ):
        return True
    if m <= 8 and (n, k) in _K3_TGV_WIN_SHAPES:
        return True
    if _model_takes_tgv(m, n, k):
        return True
    if n < 1024 or k < 2048 or k > 6144:
        return False
    ragged = m % 16 != 0
    if n <= 1024:
        return m <= 512 and (m >= 48 or m % 16 >= 9)
    if n <= 2624:
        return m <= 128 and k >= 4096
    if n <= 4096:
        return m <= 64
    if k < 4096:
        return k >= 3072 and 25 <= m <= 48 and ragged
    if n <= 6144:
        if m <= 64:
            return m <= 32 or k >= 6144 or ragged
        return m <= 72 and k >= 6144 and ragged
    if n <= 8192:
        return m <= 48 or (m <= 63 and ragged)
    return n <= 12288 and k >= 6144 and (m <= 32 or (m <= 48 and ragged))


def _tgv_bf16_gemm_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    tactic: Optional[int] = None,
) -> torch.Tensor:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    if x.shape[0] == 0:
        # Match cuBLAS/F.linear semantics for empty batches; a 0-CTA launch
        # would fail with CUDA_ERROR_INVALID_VALUE.
        return out
    if tactic is None:
        tactic = _pick_tactic(x.shape[0], weight.shape[0], weight.shape[1])
    return _run_tgv(x, weight.t(), bias, out, pdl=True, tactic=tactic)


def _tgv_bf16_gemm_out_run(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    if not is_sm100_supported():
        raise RuntimeError("cutedsl_bf16_gemm requires an SM10x GPU")
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16
    assert out.dtype in (torch.bfloat16, torch.float32) and out.device == x.device
    assert x.ndim == 2 and weight.ndim == 2 and out.ndim == 2
    assert x.stride(-1) == 1, "x must be K-major [M, K]"
    assert weight.stride(-1) == 1, "weight must be K-major [N, K]"
    assert out.is_contiguous() and out.shape == (x.shape[0], weight.shape[0])
    if x.shape[0] == 0:
        return None
    if (
        current_arch().is_sm(10, 7)
        and x.shape[0] in _K3_FRONT_WARP_SPLITK
        and x.shape[1] == 7168
        and weight.shape == (6016, 7168)
        and out.dtype == torch.float32
        and bias is None
    ):
        # Temporary vendored copy of FlashInfer's warp split-K kernel with an
        # FP32 output path; switch to flashinfer.gemm once it accepts FP32 out.
        from sglang.kernels.ops.gemm.dense_bf16_gemm_warp_splitk import (
            WarpSplitKTactic,
            run_warp_splitk_dense,
        )

        token_tile, stages = _K3_FRONT_WARP_SPLITK[x.shape[0]]
        run_warp_splitk_dense(
            x,
            weight.t(),
            out,
            pdl=True,
            tactic=WarpSplitKTactic(
                output_tile=32,
                token_tile=token_tile,
                k_tile=256,
                stages=stages,
                b_loader_warps=2,
            ),
        )
        return None
    _run_tgv(
        x,
        weight.t(),
        bias,
        out,
        pdl=True,
        tactic=_pick_tactic(x.shape[0], weight.shape[0], weight.shape[1]),
    )
    return None


def _tgv_bf16_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    tactic: Optional[int] = None,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]))


def _tgv_bf16_gemm_out_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    return None


direct_register_custom_op(
    op_name="cutedsl_tgv_bf16_gemm",
    op_func=_tgv_bf16_gemm_run,
    mutates_args=[],
    fake_impl=_tgv_bf16_gemm_fake,
)

direct_register_custom_op(
    op_name="cutedsl_tgv_bf16_gemm_out",
    op_func=_tgv_bf16_gemm_out_run,
    mutates_args=["out"],
    fake_impl=_tgv_bf16_gemm_out_fake,
)


@debug_kernel_api
def cutedsl_bf16_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    tactic: int | None = None,
) -> torch.Tensor:
    """out[M, N] = x[M, K] @ weight[N, K].T (+ bias[N]), all bf16, fp32 accum;
    ``tactic`` (a FlashInfer ``mm_bf16(backend="tgv")`` tactic id) overrides _pick_tactic."""
    return torch.ops.sglang.cutedsl_tgv_bf16_gemm(x, weight, bias, tactic)


@debug_kernel_api
def cutedsl_bf16_gemm_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write the BF16 GEMM directly into a caller-owned contiguous tensor."""
    torch.ops.sglang.cutedsl_tgv_bf16_gemm_out(x, weight, out, bias)
    return out
