#!/usr/bin/env python3
"""mori's mxfp8 GEMM against the native linear it would replace.

Single process: this path has no collective. What it replaces is
`mxfp8_native_blockscaled_linear`, whose own route changes with M --
`native_route_plan` picks a GEMV shape at small M, `dot_scaled` in the middle
and `hipblaslt_bf16` at the top -- so the baseline is not one kernel and the
curve is not smooth. That is the curve `_MIN_GRID` has to be read off.

One M per process. That is not fussiness: in the fused path's threshold work a
multi-M sweep reported 697us and 1083us for two M values that pad to the same
size and must therefore cost the same, and isolated runs gave 697 for both.

    python bench_mori_mxfp8_gemm.py --shape wq_b -m 1024
    python bench_mori_mxfp8_gemm.py --shape wq_a_tp4 -m 1024
    python bench_mori_mxfp8_gemm.py -n 4096 -k 1280 -m 1024     # any shape
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

MXFP8_BK = 32

#: Every fp8 linear DeepSeek-V4.1-Flash actually has, read off the checkpoint
#: (`layers.N.attn.*`, `layers.N.ffn.shared_experts.*`) and split by the
#: parallelism each one is declared with in `models/deepseek_v4.py`. The routed
#: experts are absent on purpose -- they are fp4 (`expert_dtype`), so they never
#: reach this path.
#:
#: `wq_b` and `wo_b` keep their bare names because the rest of this branch's
#: measurements are recorded under them.
SHAPES = {
    # ---- TP4, the deployed configuration -------------------------------
    "wq_b": (8192, 1280),  # ColumnParallel 32768x1280 / 4
    "wo_b": (5120, 2048),  # RowParallel    5120x8192 / 4
    "wq_a_tp4": (1280, 5120),  # Replicated, not split
    "wkv_tp4": (512, 5120),  # Replicated, not split
    "wqkv_a_tp4": (1792, 5120),  # Replicated, the fused wq_a + wkv
    "wo_a_tp4": (2048, 4096),  # ColumnParallel 8192x4096 / 4
    "shared_gate_up_tp4": (1152, 5120),  # MergedColumn 2*2304x5120 / 4
    "shared_down_tp4": (5120, 576),  # RowParallel 5120x2304 / 4
    # ---- TP8, to separate "the shape" from "the split" -----------------
    "wq_b_tp8": (4096, 1280),
    "wo_b_tp8": (5120, 1024),
    "wo_a_tp8": (1024, 4096),
    # ---- TP1, the unsplit weight ---------------------------------------
    "wq_b_tp1": (32768, 1280),
    "wo_b_tp1": (5120, 8192),
}


class _Layer:
    def __init__(self, weight, weight_scale_mx_e8m0, weight_bf16):
        self.weight = weight
        self.weight_scale_mx_e8m0 = weight_scale_mx_e8m0
        self.weight_bf16 = weight_bf16
        self.mxfp8_native_ready = True


def build(n, k, seed=1234):
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        prepare_mxfp8_native_weight,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    eb = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    shuffled, scale_e8m0, weight_bf16 = prepare_mxfp8_native_weight(
        w, torch.exp2(eb.float() - 127.0), (32, 32)
    )
    return _Layer(shuffled.view(torch.float8_e4m3fn), scale_e8m0, weight_bf16)


def _timing():
    """mori's shared timer, from its benchmark tree.

    Imported rather than reimplemented: it carries two corrections that this
    file got wrong for a whole round of thresholds -- a single-call graph
    capture has a 13.4us floor on this box, and a repeated call reads the
    weight out of LLC at 1.7x the bandwidth a forward pass gets. Both are
    invisible at M=16384 and dominate at M=64.
    """
    sys.path.insert(0, os.environ.get("MORI_BENCH_DIR", _DEFAULT_MORI_BENCH))
    import timing

    return timing


_DEFAULT_MORI_BENCH = "/workspace/dsv41/mori/benchmark/cco/flydsl/gemm_ar"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", choices=sorted(SHAPES), default=None)
    p.add_argument("-n", type=int, default=None, help="with -k, any shape")
    p.add_argument("-k", type=int, default=None)
    p.add_argument("-m", type=int, required=True)
    p.add_argument("--floor", type=int, default=0, help="_MIN_GRID override")
    p.add_argument("--reps", type=int, default=32)
    args = p.parse_args()

    if args.shape is not None:
        n, k = SHAPES[args.shape]
    elif args.n and args.k:
        n, k = args.n, args.k
        args.shape = f"{n}x{k}"
    else:
        p.error("pass --shape, or -n and -k")
    vram_before = _timing().vram_used()
    import sglang.srt.layers.mori_mxfp8_gemm as mori_gemm
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        native_route_plan,
    )
    from sglang.srt.environ import envs

    # Measure the whole curve, not just the part the gate serves.
    mori_gemm._MIN_GRID = args.floor
    layer = build(n, k)
    m = args.m
    x = (torch.randn(m, k, device="cuda") / 8).to(torch.bfloat16)

    def base():
        return mxfp8_native_blockscaled_linear(
            x,
            layer.weight.view(torch.uint8),
            layer.weight_scale_mx_e8m0,
            weight_bf16=layer.weight_bf16,
        )

    timing = _timing()

    with envs.SGLANG_OPT_MORI_MXFP8_GEMM.override(True):

        def mori(_picked=None):
            return mori_gemm.mori_mxfp8_linear(layer, x, None, None, False)

        served = mori() is not None
        t_base = timing.cold_hot_us(lambda _p: base(), [layer.weight], reps=args.reps)
        t_mori = None
        if served:
            # mori reads its *own* copy of the weight, converted once and cached
            # on the layer, so rotating `layer.weight` would leave the mori
            # column hot while the baseline went cold. Rotate the converted one
            # and rebind the cache each call.
            mori_w, mori_s = mori_gemm.mori_weight(layer)

            def mori_rotated(picked):
                layer._mori_b = (picked[0], mori_s)
                return mori()

            t_mori = timing.cold_hot_us(mori_rotated, [mori_w], reps=args.reps)
            layer._mori_b = (mori_w, mori_s)

    print(
        "RESULT_JSON "
        + json.dumps(
            {
                "shape": args.shape,
                "n": n,
                "k": k,
                "m": m,
                "served": served,
                "ref_route": native_route_plan(
                    m, n, k, layer.weight_bf16 is not None, False
                ),
                "base": t_base,
                "mori": t_mori,
                "vram_before": vram_before,
                "vram_after": timing.vram_used(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
