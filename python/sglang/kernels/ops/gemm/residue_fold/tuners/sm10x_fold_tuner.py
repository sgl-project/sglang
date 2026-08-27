# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer-autotuned tactic selection for the SM100/SM103 decode fold."""

from __future__ import annotations

import os

import torch

from .fi_tuner_base import (
    bucket_of,
    make_runner_pair,
    make_tuning_config,
    tuned_call,
    weight_geometry,
)

FOLD_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128)
_FAMILY = ("row_pair", "std")
_CUSTOM_OP = "residue_fold_tactic_sm10x"
_LAYOUT_ROW_PAIR = 1

# Tactic indices that executed in this process are safe to serve without a
# separate precompile step.
_EXECUTED: set = set()
_ANNOUNCED = [False]
_DEGRADED = [False]

# Actual source and tactic of the most recent call.
_LAST_SOURCE: list = [None]
# None on the static fallback, which resolves its tile inside tactics.py.
_LAST_TILE: list = [None]


def last_tactic_source() -> str | None:
    return _LAST_SOURCE[0]


def last_tactic_tile():
    return _LAST_TILE[0]


def _record_source(src: str) -> None:
    _LAST_SOURCE[0] = src


_CANDIDATES: list | None = None
_INDEX_BY_TACTIC: dict | None = None


def _bucket_of(m: int) -> int:
    return bucket_of(FOLD_BUCKETS, m)


def _map_to_tuning_buckets(x: int) -> int:
    # Module-level named function: TuningConfig is hashed into the cache key.
    return _bucket_of(int(x))


_TUNING_CONFIG = None


def _tuning_config():
    global _TUNING_CONFIG
    if _TUNING_CONFIG is None:
        _TUNING_CONFIG = make_tuning_config(FOLD_BUCKETS, _map_to_tuning_buckets)
    return _TUNING_CONFIG


def _candidates() -> list:
    """Normalized (mode, tiler, cluster, ab) list, order = tactic index.

    The order is FAMILY_TACTICS' order, which therefore becomes part of the
    persisted cache contract -- candidates_hash() invalidates on any edit.
    """
    global _CANDIDATES, _INDEX_BY_TACTIC
    if _CANDIDATES is None:
        from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import (
            FAMILY_TACTICS,
            _norm_tactic,
        )

        _CANDIDATES = [_norm_tactic(t) for t in FAMILY_TACTICS[_FAMILY]]
        _INDEX_BY_TACTIC = {t: i for i, t in enumerate(_CANDIDATES)}
    return _CANDIDATES


def candidates_hash() -> str:
    import hashlib

    from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import FAMILY_TACTICS

    return hashlib.md5(repr(FAMILY_TACTICS[_FAMILY]).encode()).hexdigest()[:12]


def _arch() -> str | None:
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import (
        kernel_arch_for_capability,
    )

    major, minor = torch.cuda.get_device_capability()
    return kernel_arch_for_capability(int(major), int(minor))


def _legal_indices(m_tok: int) -> list[int]:
    """m-dependent legality, delegated to the ranking the dict tuner used:
    narrow tiles must cover 2*m_tok outright, smem-infeasible entries are
    already absent from the table. Ranking order is irrelevant here (the
    profiler measures everything offered); only membership matters."""
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import _rank_tactics

    cands = _candidates()
    ranked = _rank_tactics(_FAMILY, m_tok, _arch())
    return sorted(_INDEX_BY_TACTIC[t] for t in ranked if t in _INDEX_BY_TACTIC)


def _quant_row_pair(inputs):
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
        scaled_fp4_quant_mext_r1,
    )

    x, _, gs_inv, _, _ = inputs
    x_fp4, x_sf = scaled_fp4_quant_mext_r1(x, gs_inv, layout_mode=_LAYOUT_ROW_PAIR)
    return x_fp4, x_sf.view(torch.float8_e4m3fn)


def _run_tactic(inputs, idx, out_dtype) -> torch.Tensor:
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.host import mext_fold_gemm_sm103

    mode, tiler, cluster, ab = _candidates()[idx]
    _LAST_TILE[0] = (mode, tiler, cluster, ab)
    x_fp4, x_sf = _quant_row_pair(inputs)
    _, weight, _, wsb, alpha = inputs
    out = mext_fold_gemm_sm103(
        weight,
        x_fp4,
        wsb,
        x_sf,
        alpha,
        out_dtype,
        mma_tiler_mn=tiler,
        cluster_shape_mn=cluster,
        store_mode=mode,
        kernel_arch=_arch(),
        ab_stage_override=ab,
    )
    _EXECUTED.add(idx)
    return out


def _run_fallback(inputs, out_dtype) -> torch.Tensor:
    """Run the static row-pair fallback without entering the tuner again."""
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.tactics import run_fold_default

    _LAST_TILE[0] = None
    x_fp4, x_sf = _quant_row_pair(inputs)
    _, weight, _, wsb, alpha = inputs
    major, minor = torch.cuda.get_device_capability()
    return run_fold_default(
        int(major), int(minor), weight, x_fp4, wsb, x_sf, alpha, out_dtype
    )


def precompile_row_pair() -> bool:
    """Report unavailable until every row-pair candidate is precompiled."""
    # The static fallback warmup does not compile the full autotuner candidate
    # set, so advertising readiness would expose uncompiled tactics at serve time.
    return False


def _make_runners(out_dtype):
    def _fallback(inputs):
        return _run_fallback(inputs, out_dtype)

    def _valid(inputs, tuning):
        legal = _legal_indices(int(inputs[0].shape[0]))
        if tuning:
            return legal
        return [i for i in legal if i in _EXECUTED]

    def _run(inputs, i):
        return _run_tactic(inputs, i, out_dtype)

    def _extras(inputs):
        _, _, a_ld = weight_geometry(inputs[1])
        return ("row_pair", _arch(), candidates_hash(), a_ld, str(out_dtype))

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=_fallback,
        fallback_extras=("static_row_pair", 2),
        valid_tactics=_valid,
        run_tactic=_run,
        candidate_extras=_extras,
        degraded_flag=_DEGRADED,
        on_degrade=lambda: _record_source("fallback"),
    )


_RUNNERS_BY_DTYPE: dict = {}


def _announce(runner, tactic, runners) -> str:
    if runner is runners[1] and tactic >= 0:
        return f"tactic source: autotuned {_candidates()[tactic]}"
    if tactic >= 0:
        return "tactic source: autotuned (the static fallback won the profile)"
    return "tactic source: static row-pair (no tuned entry -- fallback)"


def tuner_enabled() -> bool:
    return os.environ.get("SGLANG_RESIDUE_SM10X_FOLD_TUNER", "1") == "1"


def tuned_sm10x_fold(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Route one sm10x decode fold through FlashInfer's tuned tactic choice.

    `weight` may be an ext-K base-K prefix VIEW; its stride carries a_ld.
    Returns D[x.shape[0], n_w].
    """
    inputs = [x, weight, input_global_scale_inv, weight_scale_base, alpha]

    def _runners():
        runners = _RUNNERS_BY_DTYPE.get(output_dtype)
        if runners is None:
            runners = _RUNNERS_BY_DTYPE[output_dtype] = _make_runners(output_dtype)
        return runners

    return tuned_call(
        _CUSTOM_OP,
        runners_getter=_runners,
        config_getter=_tuning_config,
        inputs=inputs,
        fallback_forward=lambda ins: _run_fallback(ins, output_dtype),
        kill_env_value=("1" if tuner_enabled() else "0"),
        announce_flag=_ANNOUNCED,
        announce=_announce,
        record=_record_source,
    )
