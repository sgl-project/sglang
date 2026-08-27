# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Autotuned dispatch for the mext_r1 prefill GEMM.

FlashInfer selects between a two-GEMM implementation and a wrapped-K kernel
per layer shape and token-count bucket. The two-GEMM path is the cache-miss
fallback; wrapped-K tactics are exposed only after their kernels are compiled.
"""

from __future__ import annotations

import torch

# FlashInfer profiles each token-count bucket once; these are tuning keys,
# not compile keys, because the SM100/103 kernels use symbolic shapes.
KLOOP_BUCKETS = (128, 256, 512, 1024, 2048)
# SM100/103 K-loop (tile, cluster) configs. One symbolic-shape compile
# per config covers every shape and m -- no buckets, no padding, whitelist
# keyed on the config alone. (256,128) is a 2-SM MMA and needs cluster (2,1).
KLOOP_SM100_CFGS = (((128, 128), (1, 1)), ((256, 128), (2, 1)))

_CUSTOM_OP = "residue_mext_prefill_fp4_gemm"

# ("sm100", tile, cluster) entries compiled during warmup.
_PRECOMPILED: set = set()


def _bucket_of(m: int) -> int:
    for b in KLOOP_BUCKETS:
        if m <= b:
            return b
    return KLOOP_BUCKETS[-1]


def _map_to_tuning_buckets(x: int) -> int:
    # Module-level named function: TuningConfig is hashed into the cache key,
    # so this must be ONE stable object, not a fresh lambda per call.
    return _bucket_of(int(x))


_TUNING_CONFIG = None


def _tuning_config():
    global _TUNING_CONFIG
    if _TUNING_CONFIG is None:
        from flashinfer.autotuner import DynamicTensorSpec, TuningConfig

        _TUNING_CONFIG = TuningConfig(
            dynamic_tensor_specs=(
                DynamicTensorSpec(
                    # tuples, not lists: the spec is hashed into the cache key
                    # (its docstring says "list", its __hash__ says otherwise)
                    input_idx=(0,),
                    dim_idx=(0,),
                    gen_tuning_buckets=KLOOP_BUCKETS,
                    map_to_tuning_buckets=_map_to_tuning_buckets,
                ),
            ),
        )
    return _TUNING_CONFIG


def _run_two_gemm(inputs):
    from sglang.kernels.ops.gemm.residue_nvfp4_linear import (
        _run_mext_r1_two_gemm,
    )

    x, w, gs_inv, wsb, alpha = inputs
    return _run_mext_r1_two_gemm(x, w, gs_inv, wsb, alpha, x.dtype)


def _run_kloop(inputs, tactic_idx) -> torch.Tensor:
    from sglang.kernels.ops.quantization.residue_nvfp4_quant import (
        scaled_fp4_quant_mext_r1,
    )

    x, w, gs_inv, wsb, alpha = inputs
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.host import (
        kext_kloop_gemm_sm100,
    )

    tile, cluster = KLOOP_SM100_CFGS[tactic_idx]
    f, s = scaled_fp4_quant_mext_r1(x, gs_inv, layout_mode="concat_k")
    return kext_kloop_gemm_sm100(
        w,
        f,
        wsb,
        s,
        alpha,
        x.dtype,
        mma_tiler_mn=tile,
        cluster_shape_mn=cluster,
    )


def precompile_kloop_sm100(out_dtype=torch.bfloat16) -> int:
    """Compile the (few) symbolic sm100 kloop kernels; whitelist per config."""
    from sglang.kernels.ops.gemm.residue_fold.cute_fold.host import _compile_fold_gemm

    # A real allocation, not torch.cuda.init(): init() alone leaves no CURRENT
    # context, and cutlass HardwareInfo then dies with
    # CUDA_ERROR_INVALID_CONTEXT. PWAL callers always have one; offline
    # tooling (prewarm scripts, probes) is exactly who hits this.
    torch.zeros(1, device="cuda")
    done = 0
    for tile, cluster in KLOOP_SM100_CFGS:
        try:
            _compile_fold_gemm(
                tile, cluster, out_dtype, mode="kloop_tma", kernel_arch="sm100"
            )
            _PRECOMPILED.add(("sm100", tile, cluster))
            done += 1
        except Exception as e:  # noqa: BLE001
            print(
                f"[residue] sm100 kloop precompile failed tile={tile} "
                f"cluster={cluster}: {type(e).__name__}: {str(e)[:70]}",
                flush=True,
            )
    return done


# The fold kernel is reserved for decode; prefill compares two_gemm and k_loop.

_DEGRADED = [False]
_ANNOUNCED = [False]


def _kloop_valid_tactics(inputs, tuning):
    if tuning:
        return list(range(len(KLOOP_SM100_CFGS)))
    return [
        i
        for i, (tile, cluster) in enumerate(KLOOP_SM100_CFGS)
        if ("sm100", tile, cluster) in _PRECOMPILED
    ]


def _make_runners():
    from .fi_tuner_base import make_runner_pair

    return make_runner_pair(
        _CUSTOM_OP,
        fallback_forward=_run_two_gemm,
        fallback_extras=("two_gemm", 1),
        valid_tactics=_kloop_valid_tactics,
        run_tactic=_run_kloop,
        candidate_extras=lambda inputs: ("k_loop", 1),
        degraded_flag=_DEGRADED,
    )


_RUNNERS = None


def tuned_mext_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    input_global_scale_inv: torch.Tensor,
    weight_scale_base: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    force_kloop: bool = False,
) -> torch.Tensor:
    """Route one mext_r1 prefill GEMM through FlashInfer's tuned choice."""
    global _RUNNERS
    inputs = [x, weight, input_global_scale_inv, weight_scale_base, alpha]

    if force_kloop:
        for i, (tile, cluster) in enumerate(KLOOP_SM100_CFGS):
            if ("sm100", tile, cluster) in _PRECOMPILED:
                return _run_kloop(inputs, i)
        return _run_two_gemm(inputs)

    from .fi_tuner_base import tuned_call

    def _runners():
        global _RUNNERS
        if _RUNNERS is None:
            _RUNNERS = _make_runners()
        return _RUNNERS

    return tuned_call(
        _CUSTOM_OP,
        runners_getter=_runners,
        config_getter=_tuning_config,
        inputs=inputs,
        fallback_forward=_run_two_gemm,
        kill_env_value=None,  # no kill switch: two_gemm IS the fallback
        announce_flag=_ANNOUNCED,
        announce=None,
    )
