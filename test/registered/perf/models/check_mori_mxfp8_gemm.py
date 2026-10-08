#!/usr/bin/env python3
"""Does mori's mxfp8 GEMM agree with the native linear it replaces?

Single process -- this path has no collective, which is the point of it. Builds
a layer the way sglang's loader leaves it after `prepare_mxfp8_native_weight`,
runs both routes on the same inputs, and compares.

What is really under test is the operand contract, because every way of getting
it wrong returns a finite, plausible number rather than an error: the weight
permutation between sglang's shuffle and mori's, the B scale's transpose and
widen, the packed A scale, and the zero-padding when M is not a multiple of 64.

The comparison is against `mxfp8_native_blockscaled_linear` rather than an fp32
reference on purpose: both quantise the same activation the same way, so what is
left is the GEMM, and a disagreement points at the layouts rather than at fp8.

**relL2 of exactly 0 is the expected result where the reference takes
`dot_scaled`**, and a stronger check than any tolerance: that route computes the
same mxfp8 product from the same operands, and ue8m0 scales are exact powers of
two, so the two agree bit for bit. Where the reference takes `hipblaslt_bf16`
instead -- a bf16 GEMM over a dequantised weight -- it is a different arithmetic
path and ~2e-4 is the whole difference between them. The route is reported so
the zero is readable rather than alarming.

    python check_mori_mxfp8_gemm.py
"""

from __future__ import annotations

import argparse
import json

import torch

MXFP8_BK = 32
#: wq_b is column-parallel and has no all-reduce, so only this path can serve it.
SHAPES = {"wq_b": (8192, 1280), "wo_b": (5120, 2048)}


class _Layer:
    """Just enough of the linear for both routes to read."""

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


def check_gemv(mori_gemm, label, n, k, m):
    """The skinny-GEMM leg, which is a different kernel and a different contract.

    Only the fp8-input form is checked because only it is served -- a bf16
    activation is declined, since quantising it costs more than the kernel
    saves. The reference is `mxfp8_gemv` on the same fp8 bytes: same
    instruction, same operands, so **bit-identical is the expected result** and
    anything else means the layouts disagree.
    """
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import mxfp8_gemv

    layer = build(n, k)
    xq, xs = mxfp8_e4m3_quantize(
        (torch.randn(m, k, device="cuda") / 8).to(torch.bfloat16)
    )
    got = mori_gemm.mori_mxfp8_linear(layer, xq, None, xs, True)
    if got is None:
        return {"shape": label, "m": m, "leg": "gemv", "served": False}
    ref = mxfp8_gemv(
        xq, layer.weight.view(torch.uint8), layer.weight_scale_mx_e8m0, x_scale=xs
    )
    rel = (
        torch.linalg.vector_norm(got.float() - ref.float())
        / torch.linalg.vector_norm(ref.float())
    ).item()
    return {
        "shape": label,
        "n": n,
        "k": k,
        "m": m,
        "leg": "gemv",
        "served": True,
        "bit_identical": bool(torch.equal(got, ref)),
        "rel_l2": rel,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-m", default="512,1024,4096,16384,1000")
    p.add_argument("--gemv-m", default="1,2,7,16,32")
    p.add_argument("--floor", type=int, default=0, help="_MIN_GRID override")
    args = p.parse_args()

    import sglang.srt.layers.mori_mxfp8_gemm as mori_gemm
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
        native_route_plan,
    )
    from sglang.srt.environ import envs

    mori_gemm._MIN_GRID = args.floor
    with envs.SGLANG_OPT_MORI_MXFP8_GEMM.override(True):
        for label, (n, k) in SHAPES.items():
            for m in (int(v) for v in args.gemv_m.split(",")):
                print(
                    "RESULT_JSON " + json.dumps(check_gemv(mori_gemm, label, n, k, m)),
                    flush=True,
                )
        for label, (n, k) in SHAPES.items():
            layer = build(n, k)
            for m in (int(v) for v in args.m.split(",")):
                x = (torch.randn(m, k, device="cuda") / 8).to(torch.bfloat16)
                got = mori_gemm.mori_mxfp8_linear(layer, x, None, None, False)
                if got is None:
                    print(
                        "RESULT_JSON "
                        + json.dumps({"shape": label, "m": m, "served": False}),
                        flush=True,
                    )
                    continue
                ref = mxfp8_native_blockscaled_linear(
                    x,
                    layer.weight.view(torch.uint8),
                    layer.weight_scale_mx_e8m0,
                    weight_bf16=layer.weight_bf16,
                )
                rel = (
                    torch.linalg.vector_norm(got.float() - ref.float())
                    / torch.linalg.vector_norm(ref.float())
                ).item()
                print(
                    "RESULT_JSON "
                    + json.dumps(
                        {
                            "shape": label,
                            "n": n,
                            "k": k,
                            "m": m,
                            "served": True,
                            "ref_route": native_route_plan(
                                m, n, k, layer.weight_bf16 is not None, False
                            ),
                            "bit_identical": bool(torch.equal(got, ref)),
                            "rel_l2": rel,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
