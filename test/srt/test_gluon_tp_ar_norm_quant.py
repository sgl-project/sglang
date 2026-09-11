"""Standalone 4-rank correctness + microbenchmark for the in-tree Gluon
TP all-reduce + residual add + Gemma RMSNorm + per-1x128 FP8 quant kernel.

Runs outside the serving stack so it is fast to iterate and, crucially, gives a
per-call cost that does not depend on the torch profiler -- collectives absorb
rank skew under profiling, so trace durations are not trustworthy for this
kernel.

    torchrun --nproc_per_node=4 test_intree_collective.py [--bench]
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, "/sgl-workspace/sglang/python")

from sglang.srt.distributed.device_communicators import gluon_tp_ar_norm_quant as G

HIDDEN = G.HIDDEN_SIZE
EPS = G.EPS
FP8_MAX = 448.0


def reference(x_local, residual, weight, group_size=128):
    """Eager reference: all-reduce -> +residual -> Gemma RMSNorm -> per-group FP8.

    The reduction is done by gathering every rank's contribution and summing in
    fp32 **in ascending rank order**, then rounding once to bf16 -- exactly what
    the kernel does (``(((x0+x1)+x2)+x3)`` over the global-rank-ordered peer
    table). Using ``dist.all_reduce`` here instead would introduce a different
    accumulation order/precision and show up as a spurious mismatch.

    The round-to-bf16 before the residual add is deliberate and matches both
    aiter and Artemis, so the fused path equals the unfused one.
    """
    world = dist.get_world_size()
    gathered = [torch.empty_like(x_local) for _ in range(world)]
    dist.all_gather(gathered, x_local)
    acc = gathered[0].to(torch.float32)
    for i in range(1, world):
        acc = acc + gathered[i].to(torch.float32)
    reduced = acc.to(torch.bfloat16)

    value = reduced.to(torch.float32) + residual.to(torch.float32)
    residual_out = value.to(torch.bfloat16)

    var = (value * value).mean(dim=-1, keepdim=True)
    normed = value * torch.rsqrt(var + EPS) * weight.to(torch.float32)
    normed_bf16 = normed.to(torch.bfloat16)

    v = normed_bf16.to(torch.float32).view(-1, HIDDEN // group_size, group_size)
    amax = v.abs().amax(dim=-1).clamp_min(1.0e-10)
    scales = amax / FP8_MAX
    q = torch.clamp(v / scales[..., None], -FP8_MAX, FP8_MAX)
    # The kernel stores fp8_e4m3; round the reference the same way, otherwise
    # every comparison is off by up to one fp8 ULP (32 near the 448 ceiling).
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    return q.view(-1, HIDDEN), scales, residual_out, normed_bf16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", rank)
    torch.manual_seed(1234 + rank)

    weight = (torch.randn(HIDDEN, device=device) * 0.02 + 1.0).to(torch.bfloat16)
    dist.broadcast(weight, src=0)

    state = G.GluonTpArNormQuantState(
        group=dist.group.WORLD, device=device, max_rows=max(G.SUPPORTED_M)
    )
    if rank == 0:
        print(f"rendezvous OK; peer input table = {state.input_ptr_table.tolist()}")

    ok = True
    for m in G.SUPPORTED_M:
        x = (torch.randn(m, HIDDEN, device=device) * 0.5).to(torch.bfloat16)
        residual = (torch.randn(m, HIDDEN, device=device) * 0.5).to(torch.bfloat16)
        dist.broadcast(residual, src=0)  # residual is rank-identical in serving

        ref_q, ref_s, ref_r, ref_n = reference(x, residual, weight)
        dist.barrier()
        q, s, r, n = G.fused_tp_ar_add_gemma_rmsnorm_group_fp8_quant(
            state, x, residual, weight
        )
        dist.barrier()

        dq = (q.to(torch.float32) - ref_q).abs().max().item()
        ds = (s.to(torch.float32) - ref_s).abs().max().item()
        dr = (r.to(torch.float32) - ref_r.to(torch.float32)).abs().max().item()
        dn = (n.to(torch.float32) - ref_n.to(torch.float32)).abs().max().item()
        # fp8 values are integers in [-448,448]; scales are ~1e-3. Tolerances are
        # loose on q (1 ulp of fp8) and tight on the bf16 outputs.
        good = (
            dq <= 1.001
            and ds <= 2e-3 * max(1.0, ref_s.abs().max().item())
            and dr <= 3e-2
            and dn <= 3e-2
        )
        ok &= good
        if rank == 0:
            print(
                f"M={m:3d}  dq={dq:8.4f} ds={ds:10.3e} dres={dr:8.4f} dnorm={dn:8.4f}"
                f"  {'OK' if good else 'MISMATCH'}"
            )

    if args.bench and ok:
        if rank == 0:
            print("\nper-call cost (no profiler):")
        for m in G.SUPPORTED_M:
            x = (torch.randn(m, HIDDEN, device=device) * 0.5).to(torch.bfloat16)
            residual = (torch.randn(m, HIDDEN, device=device) * 0.5).to(torch.bfloat16)
            for _ in range(20):
                G.fused_tp_ar_add_gemma_rmsnorm_group_fp8_quant(
                    state, x, residual, weight
                )
            dist.barrier()
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                G.fused_tp_ar_add_gemma_rmsnorm_group_fp8_quant(
                    state, x, residual, weight
                )
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000.0 / args.iters
            t = torch.tensor([us], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(f"  M={m:3d}  {t.item():7.2f} us/call (max over ranks)")

    state.close()
    dist.barrier()
    if rank == 0:
        print("\nRESULT:", "PASS" if ok else "FAIL")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
