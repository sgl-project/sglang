"""Benchmark the complete GLM-5.3 mHC post -> pre boundary on gfx950.

This compares the production unfused AITER decomposition with AITER's existing
fused GEMM+sqrsum kernel. The benchmark intentionally keeps mHC finalization and
RMSNorm identical between both arms.
"""

import argparse
import itertools
import json
import statistics

import torch

DEFAULT_M = (
    1,
    8,
    16,
    24,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
)
HIDDEN_SIZE = 4096
HC_MULT = 4
MIX_SIZE = HC_MULT * (2 + HC_MULT)
RMS_EPS = 1e-6
HC_EPS = 1e-6
SINKHORN_ITERS = 20


def _time_ms(fn, warmup: int, repeats: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), samples


def _inputs(m: int):
    torch.manual_seed(20261002 + m)
    device = torch.device("cuda")
    layer_input = (
        torch.randn(m, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.02
    )
    residual = (
        torch.randn(m, HC_MULT, HIDDEN_SIZE, device=device, dtype=torch.bfloat16) * 0.02
    )
    post = torch.sigmoid(torch.randn(m, HC_MULT, device=device))
    comb = torch.softmax(torch.randn(m, HC_MULT, HC_MULT, device=device), dim=-1)
    fn = (
        torch.randn(
            MIX_SIZE,
            HC_MULT * HIDDEN_SIZE,
            device=device,
            dtype=torch.float32,
        )
        * 0.01
    )
    scale = torch.tensor([0.5, 0.25, 0.25], device=device, dtype=torch.float32)
    base = torch.zeros(MIX_SIZE, device=device, dtype=torch.float32)
    norm_weight = torch.linspace(
        0.75, 1.25, HIDDEN_SIZE, device=device, dtype=torch.bfloat16
    )
    return layer_input, residual, post, comb, fn, scale, base, norm_weight


def _run_cell(m: int, warmup: int, repeats: int):
    from aiter.ops import mhc

    layer_input, residual, post, comb, fn, scale, base, norm_weight = _inputs(m)
    kwargs = {
        "rms_eps": RMS_EPS,
        "hc_pre_eps": HC_EPS,
        "hc_sinkhorn_eps": HC_EPS,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": SINKHORN_ITERS,
        "norm_weight": norm_weight,
        "norm_eps": RMS_EPS,
    }

    def unfused():
        next_residual = torch.empty_like(residual)
        mhc.mhc_post(next_residual, layer_input, residual, post, comb)
        post_out, comb_out, layer_out = mhc.mhc_pre(
            next_residual, fn, scale, base, **kwargs
        )
        return post_out, comb_out, layer_out, next_residual

    def production_auto():
        return mhc.mhc_fused_post_pre(
            layer_input,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            **kwargs,
        )

    def forced_fused():
        arch = "gfx950"
        old_bound = mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND.get(arch)
        mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = 1 << 30
        try:
            return mhc.mhc_fused_post_pre(
                layer_input,
                residual,
                post,
                comb,
                fn,
                scale,
                base,
                force_fused=True,
                **kwargs,
            )
        finally:
            if old_bound is None:
                del mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch]
            else:
                mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = old_bound

    reference = unfused()
    auto = production_auto()
    fused = forced_fused()
    torch.cuda.synchronize()

    def errors(actual):
        return {
            "post_max_abs": (actual[0].float() - reference[0].float())
            .abs()
            .max()
            .item(),
            "comb_max_abs": (actual[1].float() - reference[1].float())
            .abs()
            .max()
            .item(),
            "layer_max_abs": (actual[2].float() - reference[2].float())
            .abs()
            .max()
            .item(),
            "residual_max_abs": (actual[3].float() - reference[3].float())
            .abs()
            .max()
            .item(),
            "finite": all(bool(torch.isfinite(tensor).all()) for tensor in actual),
        }

    baseline_ms, baseline_samples = _time_ms(unfused, warmup, repeats)
    auto_ms, auto_samples = _time_ms(production_auto, warmup, repeats)
    fused_ms, fused_samples = _time_ms(forced_fused, warmup, repeats)
    return {
        "m": m,
        "hidden_size": HIDDEN_SIZE,
        "hc_mult": HC_MULT,
        "baseline_ms": baseline_ms,
        "auto_ms": auto_ms,
        "auto_delta_pct": (auto_ms / baseline_ms - 1) * 100,
        "forced_fused_ms": fused_ms,
        "forced_fused_delta_pct": (fused_ms / baseline_ms - 1) * 100,
        "baseline_samples": baseline_samples,
        "auto_samples": auto_samples,
        "forced_fused_samples": fused_samples,
        "auto_error": errors(auto),
        "forced_fused_error": errors(fused),
    }


def _sweep_cell(m: int, warmup: int, repeats: int, packed: bool = False):
    from aiter.ops import mhc

    layer_input, residual, post, comb, fn, scale, base, norm_weight = _inputs(m)
    candidate_fn = mhc.mhc_shuffle_fn(fn) if packed else fn
    kwargs = {
        "rms_eps": RMS_EPS,
        "hc_pre_eps": HC_EPS,
        "hc_sinkhorn_eps": HC_EPS,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": SINKHORN_ITERS,
        "norm_weight": norm_weight,
        "norm_eps": RMS_EPS,
    }

    def unfused():
        next_residual = torch.empty_like(residual)
        mhc.mhc_post(next_residual, layer_input, residual, post, comb)
        post_out, comb_out, layer_out = mhc.mhc_pre(
            next_residual, fn, scale, base, **kwargs
        )
        return post_out, comb_out, layer_out, next_residual

    reference = unfused()
    baseline_ms, _ = _time_ms(unfused, warmup, repeats)
    arch = "gfx950"
    old_bound = mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND.get(arch)
    old_config = mhc.get_mhc_fused_post_pre_config
    for split_k, tile_m, tile_n, tile_k in itertools.product(
        (1, 2, 4, 8), (16, 32, 64), (16, 32), (32, 64)
    ):
        if (
            HIDDEN_SIZE % (split_k * tile_k)
            or HIDDEN_SIZE // split_k < 2 * tile_k
            or (tile_m == 64 and tile_k == 64)
        ):
            continue
        config = (split_k, tile_m, tile_n, tile_k)

        def candidate():
            mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = 1 << 30
            mhc.get_mhc_fused_post_pre_config = lambda *_args, **_kwargs: config
            try:
                return mhc.mhc_fused_post_pre(
                    layer_input,
                    residual,
                    post,
                    comb,
                    candidate_fn,
                    scale,
                    base,
                    force_fused=True,
                    w_preshuffle_bf16=packed,
                    **kwargs,
                )
            finally:
                mhc.get_mhc_fused_post_pre_config = old_config
                if old_bound is None:
                    del mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch]
                else:
                    mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = old_bound

        try:
            actual = candidate()
            torch.cuda.synchronize()
            candidate_ms, samples = _time_ms(candidate, warmup, repeats)
            row = {
                "mode": "sweep",
                "packed": packed,
                "m": m,
                "config": config,
                "baseline_ms": baseline_ms,
                "candidate_ms": candidate_ms,
                "delta_pct": (candidate_ms / baseline_ms - 1) * 100,
                "samples": samples,
                "post_max_abs": (actual[0].float() - reference[0].float())
                .abs()
                .max()
                .item(),
                "comb_max_abs": (actual[1].float() - reference[1].float())
                .abs()
                .max()
                .item(),
                "layer_max_abs": (actual[2].float() - reference[2].float())
                .abs()
                .max()
                .item(),
                "residual_max_abs": (actual[3].float() - reference[3].float())
                .abs()
                .max()
                .item(),
            }
        except Exception as err:
            row = {
                "mode": "sweep",
                "packed": packed,
                "m": m,
                "config": config,
                "error": repr(err),
            }
        print(json.dumps(row), flush=True)


def _packed_cell(m: int, warmup: int, repeats: int):
    from aiter.ops import mhc

    layer_input, residual, post, comb, fn, scale, base, norm_weight = _inputs(m)
    packed_fn = mhc.mhc_shuffle_fn(fn)
    kwargs = {
        "rms_eps": RMS_EPS,
        "hc_pre_eps": HC_EPS,
        "hc_sinkhorn_eps": HC_EPS,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": SINKHORN_ITERS,
        "norm_weight": norm_weight,
        "norm_eps": RMS_EPS,
    }

    def fp32_unfused():
        next_residual = torch.empty_like(residual)
        mhc.mhc_post(next_residual, layer_input, residual, post, comb)
        post_out, comb_out, layer_out = mhc.mhc_pre(
            next_residual, fn, scale, base, **kwargs
        )
        return post_out, comb_out, layer_out, next_residual

    def packed_unfused():
        next_residual = torch.empty_like(residual)
        mhc.mhc_post(next_residual, layer_input, residual, post, comb)
        post_out, comb_out, layer_out = mhc.mhc_pre(
            next_residual,
            packed_fn,
            scale,
            base,
            w_preshuffle_bf16=1,
            **kwargs,
        )
        return post_out, comb_out, layer_out, next_residual

    def packed_fused():
        arch = "gfx950"
        old_bound = mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND.get(arch)
        mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = 1 << 30
        try:
            return mhc.mhc_fused_post_pre(
                layer_input,
                residual,
                post,
                comb,
                packed_fn,
                scale,
                base,
                force_fused=True,
                w_preshuffle_bf16=True,
                **kwargs,
            )
        finally:
            if old_bound is None:
                del mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch]
            else:
                mhc.MHC_FUSED_POST_PRE_M_UPPER_BOUND[arch] = old_bound

    reference = fp32_unfused()
    packed_reference = packed_unfused()
    fused = packed_fused()
    torch.cuda.synchronize()

    def errors(actual):
        return {
            "post_max_abs": (actual[0].float() - reference[0].float())
            .abs()
            .max()
            .item(),
            "comb_max_abs": (actual[1].float() - reference[1].float())
            .abs()
            .max()
            .item(),
            "layer_max_abs": (actual[2].float() - reference[2].float())
            .abs()
            .max()
            .item(),
            "residual_max_abs": (actual[3].float() - reference[3].float())
            .abs()
            .max()
            .item(),
        }

    baseline_ms, _ = _time_ms(fp32_unfused, warmup, repeats)
    packed_ms, _ = _time_ms(packed_unfused, warmup, repeats)
    fused_ms, _ = _time_ms(packed_fused, warmup, repeats)
    return {
        "mode": "packed",
        "m": m,
        "baseline_ms": baseline_ms,
        "packed_unfused_ms": packed_ms,
        "packed_unfused_delta_pct": (packed_ms / baseline_ms - 1) * 100,
        "packed_fused_ms": fused_ms,
        "packed_fused_delta_pct": (fused_ms / baseline_ms - 1) * 100,
        "packed_unfused_error": errors(packed_reference),
        "packed_fused_error": errors(fused),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, nargs="+", default=DEFAULT_M)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-packed", action="store_true")
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()

    for m in args.m:
        if args.sweep:
            _sweep_cell(m, args.warmup, args.repeats)
        elif args.sweep_packed:
            _sweep_cell(m, args.warmup, args.repeats, packed=True)
        elif args.packed:
            print(json.dumps(_packed_cell(m, args.warmup, args.repeats)), flush=True)
        else:
            print(json.dumps(_run_cell(m, args.warmup, args.repeats)), flush=True)


if __name__ == "__main__":
    main()
