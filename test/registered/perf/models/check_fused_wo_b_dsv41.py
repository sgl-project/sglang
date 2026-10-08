#!/usr/bin/env python3
"""Does the fused wo_b agree with the GEMM it replaces, at V4.1-Flash's shapes?

Stands in for the layer rather than the server: builds a RowParallelLinear-shaped
weight the way sglang's loader leaves it after ``prepare_mxfp8_native_weight``,
hands it to ``fused_wo_b``, and compares against the reference every rank
computes in fp32 from the same bytes.

What this is really testing is the operand contract, because every way of
getting it wrong returns a finite, plausible number rather than an error:

* the weight permutation between sglang's shuffle and mori's
* the B scale's transpose and widen
* the packed A scale the quantiser writes
* the row padding, when M is not a multiple of tp_size * 256

    MORI_ENABLE_SDMA=1 torchrun --standalone --nproc_per_node=4 \
        check_fused_wo_b.py -n 5120 -k 2048 -m 4096,8192,3000
"""

from __future__ import annotations

import argparse
import json
import logging
import os

import torch
import torch.distributed as dist

MXFP8_BK = 32


class _Layer:
    """Just enough of a RowParallelLinear for the helper to read."""

    def __init__(self, weight, weight_scale_mx_e8m0):
        self.weight = weight
        self.weight_scale_mx_e8m0 = weight_scale_mx_e8m0
        self.mxfp8_native_ready = True


def build(rank, n, k, seed=1234):
    """The weight as the loader leaves it, plus the fp32 pieces for a reference."""
    from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
        prepare_mxfp8_native_weight,
    )

    g = torch.Generator(device="cuda").manual_seed(seed + 1009 * rank)
    w = (torch.randn(n, k, generator=g, device="cuda") / 8).to(torch.float8_e4m3fn)
    eb = torch.randint(
        120, 123, (n // MXFP8_BK, k // MXFP8_BK), generator=g, device="cuda",
        dtype=torch.int32,
    )
    wsf = torch.exp2(eb.float() - 127.0)
    shuffled, scale_e8m0, _ = prepare_mxfp8_native_weight(w, wsf, (32, 32))
    layer = _Layer(shuffled.view(torch.float8_e4m3fn), scale_e8m0)
    return layer, w, wsf


def reference(world, m, n, k, x_all):
    """All-reduced fp32, computed from the unquantised bytes on every rank."""
    from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import mxfp8_e4m3_quantize

    acc = torch.zeros(m, n, device="cuda", dtype=torch.float32)
    for r in range(world):
        _, w, wsf = build(r, n, k)
        xq, xs = mxfp8_e4m3_quantize(x_all[r])
        sav = torch.exp2(xs.to(torch.int32).float() - 127.0)
        af, bf = xq.float(), w.float()
        for j in range(k // MXFP8_BK):
            sl = slice(j * MXFP8_BK, (j + 1) * MXFP8_BK)
            acc += (
                (af[:, sl] @ bf[:, sl].t())
                * sav[:, j : j + 1]
                * wsf[:, j].repeat_interleave(MXFP8_BK)[None, :n]
            )
    return acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-n", type=int, default=5120)
    p.add_argument("-k", type=int, default=2048)
    p.add_argument("-m", default="4096,8192,3000")
    p.add_argument(
        "--floor",
        type=int,
        default=0,
        help="lower the eligibility floor. The floors are a profitability "
        "judgement; correctness has to be checkable below them too.",
    )
    args = p.parse_args()

    # the module reports every rejection through logger.warning, once each
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    # sglang's TP group, not just torch.distributed: fused_wo_b reads
    # get_tp_group() for the world size and to build mori's communicator on.
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(local_rank=local_rank)
    rank, world = dist.get_rank(), dist.get_world_size()
    initialize_model_parallel(tensor_model_parallel_size=world)

    # _window_m_max sizes the symmetric window from chunked_prefill_size, so the
    # helper needs server args published. Without them it raises ValueError, and
    # that is a *per-call* rejection -- the path falls back silently, which is
    # correct in a server but makes this harness measure nothing.
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", chunked_prefill_size=max(
            int(v) for v in args.m.split(",")
        ))
    )

    import sglang.srt.layers.mori_gemm_ar as mori_wo_b
    from sglang.srt.environ import envs
    from sglang.srt.layers.mori_gemm_ar import fused_wo_b

    if args.floor:
        mori_wo_b._MIN_FUSED_M = mori_wo_b._MIN_FUSED_M_FP8_GATHER = args.floor

    layer, _, _ = build(rank, args.n, args.k)

    with envs.SGLANG_OPT_FUSED_WO_B_AR.override(True):
        for m in (int(v) for v in args.m.split(",")):
            # every rank's activation, so the reference can sum the same bytes
            x_all = [
                (
                    torch.randn(
                        m,
                        args.k,
                        generator=torch.Generator(device="cuda").manual_seed(77 + r),
                        device="cuda",
                    )
                    / 8
                ).to(torch.bfloat16)
                for r in range(world)
            ]
            out = fused_wo_b(layer, x_all[rank])
            if out is None:
                if rank == 0:
                    print(
                        f"RESULT_JSON {json.dumps({'m': m, 'fused': False})}",
                        flush=True,
                    )
                continue
            ref = reference(world, m, args.n, args.k, x_all)
            rel = (
                torch.linalg.vector_norm(out.float() - ref)
                / torch.linalg.vector_norm(ref)
            ).item()
            if rank == 0:
                print(
                    "RESULT_JSON "
                    + json.dumps({"m": m, "fused": True, "rel_l2": rel}),
                    flush=True,
                )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
