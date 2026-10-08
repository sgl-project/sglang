#!/usr/bin/env python3
"""What the fused wo_b is worth at DeepSeek-V4.1-Flash's shapes, against what runs today.

The thresholds in ``mori_gemm_ar.py`` decide when to fuse, and they have to be
measured against the path they replace rather than against mori's own split
baseline -- those are different questions and they give different answers.

Today's path is ``mxfp8_native_blockscaled_linear`` followed by an NCCL
all-reduce. At V4.1-Flash's TP4 shape ``native_route_plan`` sends it to
``hipblaslt_bf16``, so the baseline is a bf16 GEMM over a dequantised weight
copy plus a collective -- not an mxfp8 GEMM. That is most of why fusing looks
different here than on V4-Pro.

One variant per process: ``_state`` caches the op and its symmetric window, and
the gather dtype is fixed at construction, so switching it in-process would
either leak a window or measure a stale one.

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=4 \
        bench_fused_wo_b_dsv41.py --variant fused-fp8 -m 1024,2048,4096,8192,16384
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics

import torch
import torch.distributed as dist

MXFP8_BK = 32


class _Layer:
    """Just enough of wo_b's RowParallelLinear for both paths to read.

    Fields are the ones ``_process_mxfp8_linear_weight_scale`` leaves behind on
    the real layer, so both the baseline and the fused path see what they would
    see in the server.
    """

    def __init__(self, weight, weight_scale_mx_e8m0, weight_bf16):
        self.weight = weight
        self.weight_scale_mx_e8m0 = weight_scale_mx_e8m0
        self.weight_bf16 = weight_bf16
        self.mxfp8_native_ready = True


def build(rank, n, k, seed=1234):
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        prepare_mxfp8_native_weight,
    )

    g = torch.Generator(device="cuda").manual_seed(seed + 1009 * rank)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    eb = torch.randint(
        120,
        123,
        (n // MXFP8_BK, k // MXFP8_BK),
        generator=g,
        device="cuda",
        dtype=torch.int32,
    )
    wsf = torch.exp2(eb.float() - 127.0)
    shuffled, scale_e8m0, weight_bf16 = prepare_mxfp8_native_weight(w, wsf, (32, 32))
    return _Layer(shuffled.view(torch.float8_e4m3fn), scale_e8m0, weight_bf16)


def median_us(fn, warmup=5, iters=21):
    """Median over iters, then the max across ranks.

    Median because a collective's first calls are startup artifacts -- two of
    162 once accounted for 67% of the all-reduce time and made it read 3x below
    roofline. Max across ranks because the layer is not done until the slowest
    one is.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    local = torch.tensor([statistics.median(ts)], device="cuda")
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return local.item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("base", "fused"), required=True)
    p.add_argument("-n", type=int, default=5120)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("-m", default="1024,2048,4096,8192,16384")
    p.add_argument("--iters", type=int, default=21)
    p.add_argument(
        "--floor",
        type=int,
        default=0,
        help="lower the eligibility floor for measurement only. The floors are "
        "what this tool exists to derive, so it has to be able to measure below "
        "the ones currently set -- otherwise every run just confirms them.",
    )
    p.add_argument(
        "--pad-fill",
        type=float,
        default=0.0,
        help="same, for the minimum fill ratio: a ragged M pads up, and what "
        "the ratio has to clear is the gain at the padded size.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_reduce,
    )

    init_distributed_environment(local_rank=local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()
    initialize_model_parallel(tensor_model_parallel_size=world)

    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    ms = [int(v) for v in args.m.split(",")]
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", chunked_prefill_size=max(ms))
    )

    import sglang.srt.layers.mori_gemm_ar as mori_wo_b
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        mxfp8_native_blockscaled_linear,
    )
    from sglang.srt.environ import envs
    from sglang.srt.layers.mori_gemm_ar import fused_wo_b

    if args.floor:
        mori_wo_b._MIN_FUSED_M = args.floor
        mori_wo_b._MIN_FUSED_M_FP8_GATHER = args.floor
    if args.pad_fill:
        mori_wo_b._MIN_PAD_FILL = args.pad_fill

    layer = build(rank, args.n, args.k)

    with envs.SGLANG_OPT_FUSED_WO_B_AR.override(args.variant == "fused"):
        for m in ms:
            x = (torch.randn(m, args.k, device="cuda") / 8).to(torch.bfloat16)

            if args.variant == "base":

                def call(x=x):
                    # uint8, as fp8_hip._apply_native passes it: the shuffled
                    # GEMM's weight load is typed uint8 and refuses an fp8 view
                    out = mxfp8_native_blockscaled_linear(
                        x,
                        layer.weight.view(torch.uint8),
                        layer.weight_scale_mx_e8m0,
                        weight_bf16=layer.weight_bf16,
                    )
                    return tensor_model_parallel_all_reduce(out)

                fused = True  # the baseline always applies
            else:
                probe = fused_wo_b(layer, x)
                fused = probe is not None
                if not fused:
                    if rank == 0:
                        print(
                            "RESULT_JSON "
                            + json.dumps({"variant": args.variant, "m": m, "ran": False}),
                            flush=True,
                        )
                    continue

                def call(x=x):
                    return fused_wo_b(layer, x)

            t = median_us(call, iters=args.iters)
            if rank == 0:
                print(
                    "RESULT_JSON "
                    + json.dumps(
                        {
                            "variant": args.variant,
                            "gather": os.environ.get(
                                "SGLANG_OPT_FUSED_WO_B_AR_FP8_GATHER", "0"
                            ),
                            "m": m,
                            "ran": fused,
                            "us": t,
                        }
                    ),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
