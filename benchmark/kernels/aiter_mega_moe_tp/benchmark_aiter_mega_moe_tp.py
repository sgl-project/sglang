"""One MoE layer, one forward: stock AITER fused MoE + TP all-reduce vs AITER MegaMoE-TP.

    torchrun --nproc_per_node=2 benchmark/kernels/aiter_mega_moe_tp/benchmark_aiter_mega_moe_tp.py
    torchrun --nproc_per_node=4 benchmark/kernels/aiter_mega_moe_tp/benchmark_aiter_mega_moe_tp.py \
        --tokens 1024 2048 --comm-dtype bf16

Needs ROCm gfx950, AITER with ``aiter.ops.flydsl.mega_moe_tp`` and at least 2 GPUs.

Shape: hidden 4096, MoE intermediate 2048 split over the TP ranks, 288 routed experts plus
one shared expert (expert 288, weight 1) so 289 experts and top-k 8 + 1, MXFP4 weights
(``shuffle_weight`` (16, 16) and ``e8m0_shuffle``). Both arms run the same weights, routing
and inputs, in the 289-expert layout:

  stock      ``aiter.fused_moe.fused_moe`` as the Quark MXFP4 MoE scheme calls it on gfx950
             (per_1x32, separated gate/up rows), then ``dist.all_reduce`` (RCCL) of the output.
             The all-reduce is not SGLang's custom all-reduce, and the shared expert is fused
             into ``fused_moe`` here, while a TP 2 server with ``--disable-shared-experts-fusion``
             runs it separately. The stock timings therefore differ from a serving baseline.
  MegaMoE-TP one ``MegaMoeTP`` call in ``comm_mode="ar"``: routed experts, shared expert and
             the all-reduce in one kernel. ``AITER_MEGAMOE_TP_LB_*`` tune its large-batch
             schedule and are read by AITER; the values in effect are printed.

Token counts that are not a multiple of the TP size are rounded up for both arms (the kernel
takes a multiple of tp rows). Each iteration is timed with device events after a barrier, so
all ranks start together. The reported time is the slowest rank's median.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

HIDDEN = 4096
MOE_INTER = 2048  # global; each rank holds MOE_INTER // tp
ROUTED = 288
TOPK = 8  # routed; the shared expert is a ninth column


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1024, 1404, 2048, 4096]
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--comm-dtype", choices=["fp8", "bf16"], default="fp8")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if args.warmup < 10:
        parser.error("--warmup must be at least 10")
    return args


def build_weights(rank, device, inter, seed):
    """Random MXFP4 weights of this rank's intermediate slice, in the kernels' layout."""
    from aiter.ops.shuffle import shuffle_weight
    from aiter.utility.fp4_utils import e8m0_shuffle

    experts = ROUTED + 1
    gen = torch.Generator(device=device).manual_seed(seed + rank)

    def stack(rows, cols, exponent):
        codes = torch.randint(
            0,
            256,
            (experts, rows, cols // 2),
            dtype=torch.uint8,
            device=device,
            generator=gen,
        )
        scales = torch.randint(
            exponent,
            exponent + 3,
            (experts, rows, cols // 32),
            dtype=torch.uint8,
            device=device,
            generator=gen,
        )
        shuffled_scales = e8m0_shuffle(scales.view(experts * rows, -1))
        return (
            shuffle_weight(codes, layout=(16, 16)),
            shuffled_scales.view(experts, rows, -1),
        )

    w1, w1_scale = stack(2 * inter, HIDDEN, 119)
    w2, w2_scale = stack(HIDDEN, inter, 120)
    return w1, w1_scale, w2, w2_scale


def make_inputs(tokens, device, seed):
    """Replicated activations and routing: 8 distinct routed experts plus the shared one."""
    gen = torch.Generator(device=device).manual_seed(seed + tokens)
    x = torch.randn(tokens, HIDDEN, device=device, dtype=torch.bfloat16, generator=gen)
    scores = torch.rand(tokens, ROUTED, device=device, generator=gen)
    ids = scores.topk(TOPK, dim=-1).indices
    weights = torch.softmax(torch.randn(tokens, TOPK, device=device, generator=gen), -1)
    ids = torch.cat([ids, torch.full_like(ids[:, :1], ROUTED)], dim=1).to(torch.int32)
    weights = torch.cat([weights, torch.ones_like(weights[:, :1])], dim=1)
    for tensor in (x, ids, weights):
        dist.broadcast(tensor, src=0)
    return x, ids.contiguous(), weights.float().contiguous()


def median_us(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3)
    slowest = torch.tensor([statistics.median(times)], device="cuda")
    dist.all_reduce(slowest, op=dist.ReduceOp.MAX)
    return float(slowest)


def main():
    args = parse_args()
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.mega_moe_tp import MegaMoeTP, MegaMoeTPConfig
    from aiter.ops.flydsl.moe_common import GateMode

    inter = MOE_INTER // world
    rows = sorted({-(-tokens // world) * world for tokens in args.tokens})
    w1, w1_scale, w2, w2_scale = build_weights(rank, device, inter, args.seed)

    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    stock_w1 = w1.view(fp4) if fp4 is not None else w1
    stock_w2 = w2.view(fp4) if fp4 is not None else w2
    stock_w1.is_shuffled = stock_w2.is_shuffled = True

    def stock(x, ids, weights):
        out = fused_moe(
            x,
            stock_w1,
            stock_w2,
            weights,
            ids,
            quant_type=QuantType.per_1x32,
            activation=ActivationType.Silu,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            gate_mode=GateMode.SEPARATED.value,
        )
        dist.all_reduce(out)
        return out

    comm_dtype, ar_gather = (
        ("fp8", "auto") if args.comm_dtype == "fp8" else ("bf16", "bf16")
    )
    kernel = MegaMoeTP(
        MegaMoeTPConfig(
            rank=rank,
            world_size=world,
            model_dim=HIDDEN,
            inter_dim=inter,
            experts=ROUTED + 1,
            topk=TOPK + 1,
            max_local_tokens=max(rows) // world,
            activation=ActivationType.Silu,
            comm_mode="ar",
            comm_dtype=comm_dtype,
            ar_gather=ar_gather,
        ),
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        group=cpu_group,
        device=device,
    )
    kernel.prepare(sorted({m // world for m in rows}))

    if rank == 0:
        lb_env = {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith("AITER_MEGAMOE_TP")
        }
        print(
            f"device {torch.cuda.get_device_properties(device).gcnArchName}, TP {world}, "
            f"hidden {HIDDEN}, intermediate/rank {inter}, experts {ROUTED}+1, top-k {TOPK}+1, "
            f"comm {args.comm_dtype}, warmup {args.warmup}, iters {args.iters}, "
            f"AITER_MEGAMOE_TP env {lb_env or 'unset'}"
        )
        print(
            "| tokens | stock us | MegaMoE-TP us | speedup % | stock tok/s | MegaMoE-TP tok/s | rel L2 |"
        )
        print("|---|---|---|---|---|---|---|")

    for tokens in rows:
        x, ids, weights = make_inputs(tokens, device, args.seed)
        out = torch.empty_like(x)

        def fused():
            kernel(x, weights, ids, out=out)

        reference = stock(x, ids, weights)
        fused()
        error = float(
            (out.float() - reference.float()).norm() / reference.float().norm()
        )
        stock_us = median_us(lambda: stock(x, ids, weights), args.warmup, args.iters)
        fused_us = median_us(fused, args.warmup, args.iters)
        if rank == 0:
            print(
                f"| {tokens} | {stock_us:.0f} | {fused_us:.0f} | "
                f"{(stock_us - fused_us) / stock_us * 100:+.1f} | "
                f"{tokens / stock_us * 1e6:.0f} | {tokens / fused_us * 1e6:.0f} | {error:.4f} |"
            )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
