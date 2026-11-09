from __future__ import annotations

import logging
import os

import torch

from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

_MAX_ROWS = 32
_KPAD_ALIGN = 128


def pad_lowrank(lowrank: int) -> int:
    return (lowrank + _KPAD_ALIGN - 1) // _KPAD_ALIGN * _KPAD_ALIGN


def permute_pad_up_weight(w_up: torch.Tensor, hc_count: int) -> torch.Tensor:
    k, lowrank = w_up.shape
    hidden_size = k // hc_count
    padded = torch.zeros(
        (k, pad_lowrank(lowrank)), dtype=w_up.dtype, device=w_up.device
    )
    padded[:, :lowrank] = w_up.detach()
    src = torch.arange(k, device=w_up.device)
    perm = torch.empty(k, dtype=torch.long, device=w_up.device)
    perm[(src % hidden_size) * hc_count + (src // hidden_size)] = src
    return padded[perm].contiguous()


_scratch_cache = {}
_tactic_cache = {}


def _get_scratch(lowrank: int, dtype: torch.dtype, device: torch.device):
    key = (device, lowrank, dtype)
    buf = _scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros((_MAX_ROWS, pad_lowrank(lowrank)), dtype=dtype, device=device)
        _scratch_cache[key] = buf
    return buf


# Row-aware tactics for the HC mix pair (on with the HC macro-fusion, read
# once): from 10 rows the down GEMV halves split-K (s16 -> s8, 8 stages), and
# the gated up GEMV uses 64-wide MMA tiles with 4 stages at every row count.
# Measured behind the fused HC boundary kernel, these tactics are faster at 16
# and 32 rows; rel-L2 <= 8e-5 vs before.
_TUNED_TACTICS = os.environ.get("SGLANG_HC_MACRO", "1").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)


def _tuned_tactic(m: int, n: int, k: int, SplitKTactic, validate_tactic):
    if n <= 512 and k % (128 * 16) == 0:
        tactic = (
            SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
            if m <= 8
            else SplitKTactic(mma_m=64, mma_n=8, split_k=8, ab_stages=8)
        )
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        tactic = SplitKTactic(mma_m=64, mma_n=mma_n, split_k=1, ab_stages=4)
    try:
        validate_tactic(tactic, m, n, k)
    except ValueError:
        return None
    return tactic


def _get_tactic(m: int, n: int, k: int):
    key = (m, n, k)
    tactic = _tactic_cache.get(key)
    if tactic is None:
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            SplitKTactic,
            default_tactic,
            validate_tactic,
        )

        if _TUNED_TACTICS and get_parallel().tp_size == 1:
            tactic = _tuned_tactic(m, n, k, SplitKTactic, validate_tactic)
            if tactic is not None:
                _tactic_cache[key] = tactic
                return tactic
        if n <= 512 and k % (128 * 16) == 0:
            tactic = SplitKTactic(mma_m=64, mma_n=8, split_k=16, ab_stages=6)
            try:
                validate_tactic(tactic, m, n, k)
            except ValueError:
                tactic = default_tactic(m, n, k)
        else:
            tactic = default_tactic(m, n, k)
        _tactic_cache[key] = tactic
    return tactic


# Not fused_hc_mix: fixed split-K order vs its device-scope atomics, so the bits differ.
def hc_mix(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up_permuted_padded: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        run_splitk_dense_gate,
        run_splitk_dense_silu,
    )

    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    kpad = w_up_permuted_padded.shape[1]
    out = torch.empty(
        (rows, hidden_size),
        dtype=hyper_input_normed.dtype,
        device=hyper_input_normed.device,
    )
    if rows == 0:
        return out
    inv_hc = 1.0 / hc_count
    t_pad = _get_scratch(lowrank, hyper_input_normed.dtype, hyper_input_normed.device)
    for row_start in range(0, rows, _MAX_ROWS):
        row_end = min(row_start + _MAX_ROWS, rows)
        m = row_end - row_start
        chunk = hyper_input_normed[row_start:row_end]
        run_splitk_dense_silu(
            chunk,
            w_down.T,
            t_pad[:m, :lowrank],
            True,
            _get_tactic(m, lowrank, k),
            inv_hc,
        )
        run_splitk_dense_gate(
            t_pad[:m],
            w_up_permuted_padded.T,
            chunk,
            out[row_start:row_end],
            True,
            _get_tactic(m, k, kpad),
            inv_hc,
            hc_count,
        )
    return out


# Above 32 rows (TP1, hc 4): the CuTe split-K silu down over all rows, then a Triton up GEMM with the
# gated mean fused (hc_mix_up_gate, PDL). Best per row range on VR200 (mb-h446-sweep5-1); in isolation
# the gain fades above 320 rows, so SGLANG_HC_MIX_SPLITK_MAX_ROWS picks the cap.
# (rows up to, down (mma_m, mma_n, split_k, ab_stages), up+gate (BLOCK_M, BLOCK_J, BLOCK_K, warps, stages))
_WIDE_TACTICS = (
    (64, (64, 16, 8, 8), (64, 32, 64, 8, 3)),
    (144, (64, 16, 4, 8), (64, 32, 64, 8, 3)),
    (288, (64, 32, 4, 8), (64, 32, 64, 8, 3)),
    (320, (64, 16, 2, 8), (64, 32, 64, 8, 3)),
    (448, (64, 32, 2, 8), (64, 32, 64, 8, 3)),
    (512, (64, 32, 2, 8), (128, 32, 64, 8, 3)),
)
WIDE_MAX_ROWS = _WIDE_TACTICS[-1][0]
_wide_scratch_cache = {}
_wide_tactic_cache = {}
_wide_logged = False


def _get_wide_scratch(lowrank: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    key = (device, lowrank, dtype)
    buf = _wide_scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros((WIDE_MAX_ROWS, lowrank), dtype=dtype, device=device)
        _wide_scratch_cache[key] = buf
    return buf


def _get_wide_tactics(m: int, lowrank: int, k: int):
    key = (m, lowrank, k)
    tactics = _wide_tactic_cache.get(key)
    if tactics is None:
        from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
            SplitKTactic,
            validate_tactic,
        )

        down, up = next((d, u) for rows, d, u in _WIDE_TACTICS if m <= rows)
        tactics = (SplitKTactic(*down), up)
        validate_tactic(tactics[0], m, lowrank, k, max_m=WIDE_MAX_ROWS)
        _wide_tactic_cache[key] = tactics
    return tactics


def hc_mix_wide(
    hyper_input_normed: torch.Tensor,
    w_down: torch.Tensor,
    w_up_permuted_padded: torch.Tensor,
    hc_count: int,
    hidden_size: int,
) -> torch.Tensor:
    """The HC mix for 32 < rows <= WIDE_MAX_ROWS, hc_count 4: bf16(silu(x @ w_down.T / 4)) from the cluster
    split-K down (hc_mix's arithmetic), then the fused up GEMM and gated mean."""
    from sglang.kernels.ops.gemm.dense_bf16_gemm_sm100_splitk_epilogue import (
        run_splitk_dense_silu,
    )
    from sglang.kernels.ops.gemm.hc_mix_up_gate import hc_up_gate

    global _wide_logged
    rows, k = hyper_input_normed.shape
    lowrank = w_down.shape[0]
    if hc_count != 4 or not _MAX_ROWS < rows <= WIDE_MAX_ROWS:
        raise ValueError(
            f"hc_mix_wide serves hc 4 and {_MAX_ROWS} < rows <= {WIDE_MAX_ROWS}, got hc {hc_count} rows {rows}"
        )
    t = _get_wide_scratch(lowrank, hyper_input_normed.dtype, hyper_input_normed.device)[:rows]
    down, up = _get_wide_tactics(rows, lowrank, k)
    if not _wide_logged:
        _wide_logged = True
        logger.info(
            "HC mix wide on: rows %d..%d (CuTe split-K silu down + Triton up/gate, PDL); first rows=%d down %s up %s",
            _MAX_ROWS + 1,
            WIDE_MAX_ROWS,
            rows,
            down,
            up,
        )
    run_splitk_dense_silu(
        hyper_input_normed, w_down.T, t, True, down, 1.0 / hc_count, max_m=WIDE_MAX_ROWS
    )
    return hc_up_gate(
        t, w_up_permuted_padded, hyper_input_normed, hidden_size, lowrank, config=up, pdl=True
    )
