#!/usr/bin/env python3
"""Kernel-level tuning for K3 SP collectives around attention residual.

This benchmark deliberately excludes the model engine and A2A. It compares
the two mathematically equivalent attention-side orderings:

  baseline: AG(raw token shard) -> full-batch attn-res aggregation
  carry:    local attn-res aggregation -> AG(normalized token shard)

It also reports the already-fused RS+local-residual followed by the MLP-side
attn-res aggregation, which is the upper bound for a future monolithic
RS+residual+attn-res kernel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import torch
import torch.distributed as dist

from sglang.kernels.ops.kimi_k3 import attn_res, sp_collective
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.layers.k3_ar_fusion import symm_alloc

_HIDDEN_SIZE = 7168
_NVB_LAYER_COUNTS = {1: 12, 2: 12, 3: 12, 4: 12, 5: 12, 6: 12, 7: 12, 8: 8}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--nvbs", type=int, nargs="+", default=list(range(1, 9)))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--fused-blocks",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 96, 128],
    )
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--archive-md5", required=True)
    parser.add_argument(
        "--skip-rs",
        action="store_true",
        help="Tune AG fusion only; do not repeat RS candidate timings.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Capture/replay correctness only; collect no timing samples.",
    )
    parser.add_argument(
        "--o-proj-direct",
        action="store_true",
        help=(
            "Generate the fused RS input with the K3 BF16 o_proj GEMM writing "
            "directly into symmetric storage, including it in graph capture."
        ),
    )
    parser.add_argument(
        "--o-proj-padding",
        type=int,
        default=0,
        help="Zero-tail rows after direct o_proj output (padded-extend test).",
    )
    parser.add_argument("--graph-replays", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def time_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    dist.barrier()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iters)
    dist.barrier()
    return median(samples)


def assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    try:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)
    except AssertionError as exc:
        diff = (actual.float() - expected.float()).abs()
        raise AssertionError(
            f"{label}: max_abs={float(diff.max()):.6g}; {exc}"
        ) from exc


def graph_replay(fn, replays: int) -> None:
    dist.barrier()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    dist.barrier()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()


def symmetric_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    cpu_group,
) -> tuple[torch.Tensor, int]:
    import torch.distributed._symmetric_memory as torch_symm_mem

    with symm_alloc():
        tensor = torch.empty(shape, dtype=dtype, device="cuda")
    handle = torch_symm_mem.rendezvous(tensor, cpu_group.group_name)
    if handle is None or handle.multicast_ptr == 0:
        raise RuntimeError("multicast symmetric tensor allocation failed")
    rank = dist.get_rank()
    mc_ptr = handle.multicast_ptr + tensor.data_ptr() - handle.buffer_ptrs[rank]
    return tensor, mc_ptr


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world not in (4, 8):
        raise RuntimeError(
            f"K3 SP attention-residual tuning requires TP4 or TP8, got {world}"
        )
    device = torch.device("cuda", local_rank)
    cpu_group = dist.new_group(backend="gloo")
    comm = CustomAllReduceV2(group=cpu_group, device=device)
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("CustomAllReduceV2 with multicast is required")
    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    attn_res.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    if args.o_proj_direct:
        from sglang.srt.layers.quantization import unquant

        unquant.initialize_bf16_gemm_config(
            SimpleNamespace(bf16_gemm_backend="cutedsl")
        )
        o_proj_method = unquant.UnquantizedLinearMethod()

    cw = torch.linspace(-0.01, 0.01, _HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    ow = torch.linspace(1.25, 0.75, _HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    eps = 1e-6
    result = {
        "source_id": args.source_id,
        "archive_md5": args.archive_md5,
        "skip_rs": args.skip_rs,
        "graph_only": args.graph_only,
        "o_proj_direct": args.o_proj_direct,
        "o_proj_padding": args.o_proj_padding,
        "graph_replays": args.graph_replays,
        "device": torch.cuda.get_device_name(),
        "world_size": world,
        "hidden_size": _HIDDEN_SIZE,
        "nvb_layer_counts": _NVB_LAYER_COUNTS,
        "measurements_us": {},
        "selection": {},
    }

    for tokens in args.tokens:
        if tokens % world:
            raise ValueError(f"tokens={tokens} must be divisible by world={world}")
        local_tokens = tokens // world
        gen = torch.Generator(device=device)
        gen.manual_seed(12000 + rank)
        local_head = torch.randn(
            local_tokens,
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        full_head = torch.empty(
            (tokens, _HIDDEN_SIZE), dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(full_head, local_head)

        bank_gen = torch.Generator(device=device)
        bank_gen.manual_seed(13000)
        bank = torch.randn(
            tokens,
            max(args.nvbs) + 1,
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=bank_gen,
        )
        lo = rank * local_tokens
        local_bank = bank[lo : lo + local_tokens]
        local_normed = torch.empty_like(local_head)
        full_normed = torch.empty_like(full_head)
        fused_ag_normed, fused_ag_normed_mc = symmetric_tensor(
            (tokens, _HIDDEN_SIZE), torch.bfloat16, cpu_group
        )

        ag_dispatch = sp_collective.get_dispatch(
            "all_gather", world, _HIDDEN_SIZE, tokens, device
        )
        rs_dispatch = sp_collective.get_dispatch(
            "reduce_scatter", world, _HIDDEN_SIZE, tokens, device
        )
        if ag_dispatch is None or rs_dispatch is None:
            if rank == 0:
                print(f"T={tokens}: table selects NCCL; skip composite tuning")
            continue

        if ag_dispatch.strategy == "direct":
            ag_raw, ag_raw_mc = symmetric_tensor(
                (tokens, _HIDDEN_SIZE), torch.bfloat16, cpu_group
            )
            ag_normed, ag_normed_mc = symmetric_tensor(
                (tokens, _HIDDEN_SIZE), torch.bfloat16, cpu_group
            )

            def gather_raw():
                sp_collective.all_gather_direct(
                    world,
                    local_head,
                    ag_raw,
                    output_mc_ptr=ag_raw_mc,
                    tuning=ag_dispatch.tuning,
                )

            def gather_normed():
                sp_collective.all_gather_direct(
                    world,
                    local_normed,
                    ag_normed,
                    output_mc_ptr=ag_normed_mc,
                    tuning=ag_dispatch.tuning,
                )

        elif ag_dispatch.strategy == "push":
            ag_raw = torch.empty_like(full_head)
            ag_normed = torch.empty_like(full_head)

            def gather_raw():
                sp_collective.all_gather(
                    world,
                    local_head,
                    ag_raw,
                    ws_mc_base=comm.mc_base_ptr,
                    tuning=ag_dispatch.tuning,
                )

            def gather_normed():
                sp_collective.all_gather(
                    world,
                    local_normed,
                    ag_normed,
                    ws_mc_base=comm.mc_base_ptr,
                    tuning=ag_dispatch.tuning,
                )

        else:
            raise AssertionError(f"unknown AG strategy {ag_dispatch.strategy}")

        if args.o_proj_direct:
            if tokens > 512:
                raise ValueError("--o-proj-direct is limited to T<=512")
            real_tokens = tokens - args.o_proj_padding
            if real_tokens <= 0:
                raise ValueError("--o-proj-padding must be smaller than tokens")
            o_proj_input = torch.randn(
                real_tokens,
                1536,
                dtype=torch.bfloat16,
                device=device,
                generator=gen,
            )
            o_proj_weight = (
                torch.randn(
                    _HIDDEN_SIZE,
                    1536,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=gen,
                )
                * 0.01
            )
            rs_input = torch.zeros(
                tokens,
                _HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
            )
            rs_input[:real_tokens] = torch.mm(o_proj_input, o_proj_weight.t())
            o_proj_layer = SimpleNamespace(weight=o_proj_weight)
        else:
            rs_input = torch.randn(
                tokens,
                _HIDDEN_SIZE,
                dtype=torch.bfloat16,
                device=device,
                generator=gen,
            )
        residual = torch.randn(
            local_tokens,
            _HIDDEN_SIZE,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        rs_prefix = torch.empty_like(residual)
        rs_pull_input, rs_input_mc = symmetric_tensor(
            (tokens, _HIDDEN_SIZE), torch.bfloat16, cpu_group
        )
        if args.o_proj_direct:

            def o_proj_into_rs():
                o_proj_method.apply_into(
                    o_proj_layer,
                    o_proj_input,
                    rs_pull_input[:real_tokens],
                )
                rs_pull_input[real_tokens:].zero_()

            o_proj_into_rs()
            torch.cuda.synchronize()
            assert_close(rs_pull_input, rs_input, f"direct o_proj T={tokens}")
        else:
            rs_pull_input.copy_(rs_input)
        if rs_dispatch.strategy == "pull":

            def reduce_scatter():
                sp_collective.reduce_scatter_pull(
                    world,
                    rs_pull_input,
                    rs_prefix,
                    residual,
                    input_mc_ptr=rs_input_mc,
                    tuning=rs_dispatch.tuning,
                )

        elif rs_dispatch.strategy == "push":

            def reduce_scatter():
                sp_collective.reduce_scatter_res(
                    world,
                    rs_input,
                    rs_prefix,
                    residual,
                    tuning=rs_dispatch.tuning,
                )

        else:
            raise AssertionError(f"unknown RS strategy {rs_dispatch.strategy}")

        fused_prefix = torch.empty_like(residual)
        fused_normed = torch.empty_like(residual)
        token_result = {
            "ag_strategy": ag_dispatch.strategy,
            "rs_strategy": rs_dispatch.strategy,
            "by_nvb": {},
        }
        for nvb in args.nvbs:

            def baseline():
                gather_raw()
                attn_res.attn_res_fused_tma(ag_raw, bank, cw, ow, full_normed, nvb, eps)

            def carry():
                attn_res.attn_res_fused_tma(
                    local_head,
                    local_bank,
                    cw,
                    ow,
                    local_normed,
                    nvb,
                    eps,
                )
                gather_normed()

            def fused_direct_ag(max_blocks: int, write_prefix: bool = False):
                attn_res.attn_res_fused_direct_ag(
                    world,
                    local_head,
                    local_bank,
                    cw,
                    ow,
                    fused_ag_normed,
                    nvb,
                    eps,
                    output_mc_ptr=fused_ag_normed_mc,
                    max_blocks=max_blocks,
                    write_prefix=write_prefix,
                )

            def rs_attn_res():
                reduce_scatter()
                attn_res.attn_res_fused_tma(
                    rs_prefix,
                    local_bank,
                    cw,
                    ow,
                    local_normed,
                    nvb,
                    eps,
                )

            def fused_pull_rs(max_blocks: int):
                attn_res.attn_res_fused_pull_rs(
                    world,
                    rs_pull_input,
                    residual,
                    local_bank,
                    cw,
                    ow,
                    fused_normed,
                    fused_prefix,
                    nvb,
                    eps,
                    input_mc_ptr=rs_input_mc,
                    max_blocks=max_blocks,
                )

            baseline()
            carry()
            fused_direct_ag(max(args.fused_blocks))
            if not args.skip_rs:
                rs_attn_res()
                fused_pull_rs(max(args.fused_blocks))
            torch.cuda.synchronize()
            assert_close(ag_normed, full_normed, f"T={tokens} nvb={nvb}")
            assert_close(
                fused_ag_normed,
                full_normed,
                f"fused direct AG T={tokens} nvb={nvb}",
            )
            if not args.skip_rs:
                assert_close(
                    fused_prefix, rs_prefix, f"fused prefix T={tokens} nvb={nvb}"
                )
                assert_close(
                    fused_normed,
                    local_normed,
                    f"fused normed T={tokens} nvb={nvb}",
                )

            if args.graph_only:
                bank_row_before_write = local_bank[:, nvb].clone()
                graph_replay(
                    lambda: fused_direct_ag(64, write_prefix=True),
                    args.graph_replays,
                )
                assert_close(
                    fused_ag_normed,
                    full_normed,
                    f"graph fused direct AG T={tokens} nvb={nvb}",
                )
                assert_close(
                    local_bank[:, nvb],
                    local_head,
                    f"graph fused bank write T={tokens} nvb={nvb}",
                )
                local_bank[:, nvb].copy_(bank_row_before_write)
                if not args.skip_rs:
                    if args.o_proj_direct:

                        def graph_rs():
                            o_proj_into_rs()
                            fused_pull_rs(64)

                    else:

                        def graph_rs():
                            fused_pull_rs(64)

                    graph_replay(
                        graph_rs,
                        args.graph_replays,
                    )
                    assert_close(
                        fused_prefix,
                        rs_prefix,
                        f"graph fused prefix T={tokens} nvb={nvb}",
                    )
                    assert_close(
                        fused_normed,
                        local_normed,
                        f"graph fused normed T={tokens} nvb={nvb}",
                    )
                token_result["by_nvb"][str(nvb)] = {"graph_valid": True}
                if rank == 0:
                    print(
                        f"T={tokens:4d} nvb={nvb} graph replay valid "
                        f"(AG{'' if args.skip_rs else ' + RS'}, "
                        f"{args.graph_replays} replays)",
                        flush=True,
                    )
                continue

            baseline_us = time_us(baseline, args.warmup, args.iters)
            carry_us = time_us(carry, args.warmup, args.iters)
            fused_ag_candidates = {
                blocks: time_us(
                    lambda blocks=blocks: fused_direct_ag(blocks),
                    args.warmup,
                    args.iters,
                )
                for blocks in args.fused_blocks
            }
            fused_ag_blocks = min(fused_ag_candidates, key=fused_ag_candidates.get)
            fused_ag_us = fused_ag_candidates[fused_ag_blocks]
            if not args.skip_rs:
                rs_us = time_us(reduce_scatter, args.warmup, args.iters)
                local_attn_us = time_us(
                    lambda: attn_res.attn_res_fused_tma(
                        rs_prefix,
                        local_bank,
                        cw,
                        ow,
                        local_normed,
                        nvb,
                        eps,
                    ),
                    args.warmup,
                    args.iters,
                )
                rs_attn_us = time_us(rs_attn_res, args.warmup, args.iters)
                fused_candidates = {
                    blocks: time_us(
                        lambda blocks=blocks: fused_pull_rs(blocks),
                        args.warmup,
                        args.iters,
                    )
                    for blocks in args.fused_blocks
                }
                fused_blocks = min(fused_candidates, key=fused_candidates.get)
                fused_us = fused_candidates[fused_blocks]
            measurement = {
                "ag_then_full_attn_res": baseline_us,
                "local_attn_res_then_ag": carry_us,
                "fused_direct_ag_attn_res": fused_ag_us,
                "fused_direct_ag_max_blocks": fused_ag_blocks,
                "fused_direct_ag_candidates": fused_ag_candidates,
            }
            if not args.skip_rs:
                measurement.update(
                    {
                        "rs_residual": rs_us,
                        "local_attn_res": local_attn_us,
                        "rs_residual_then_local_attn_res": rs_attn_us,
                        "fused_pull_rs_attn_res": fused_us,
                        "fused_pull_rs_max_blocks": fused_blocks,
                        "fused_pull_rs_candidates": fused_candidates,
                    }
                )
            token_result["by_nvb"][str(nvb)] = measurement
            if rank == 0:
                rs_summary = (
                    f"RS+AR {rs_attn_us:7.2f}->{fused_us:7.2f} us "
                    f"({rs_attn_us / fused_us:5.2f}x, blocks={fused_blocks}) "
                    f"(RS {rs_us:6.2f}, AR {local_attn_us:6.2f})"
                    if not args.skip_rs
                    else "RS skipped"
                )
                print(
                    f"T={tokens:4d} nvb={nvb} "
                    f"AG->AR {baseline_us:7.2f} us, "
                    f"AR->AG {carry_us:7.2f} us, "
                    f"fused AG {fused_ag_us:7.2f} us "
                    f"({carry_us / fused_ag_us:5.2f}x, "
                    f"blocks={fused_ag_blocks}), "
                    f"{rs_summary}",
                    flush=True,
                )

        if rank == 0:
            if args.graph_only:
                result["measurements_us"][str(tokens)] = token_result
                result["selection"][str(tokens)] = {"graph_valid": True}
                continue
            weighted_baseline = sum(
                token_result["by_nvb"][str(nvb)]["ag_then_full_attn_res"] * count
                for nvb, count in _NVB_LAYER_COUNTS.items()
                if nvb in args.nvbs
            )
            weighted_carry = sum(
                token_result["by_nvb"][str(nvb)]["local_attn_res_then_ag"] * count
                for nvb, count in _NVB_LAYER_COUNTS.items()
                if nvb in args.nvbs
            )
            token_result["weighted_92_layers_us"] = {
                "ag_then_full_attn_res": weighted_baseline,
                "local_attn_res_then_ag": weighted_carry,
            }
            result["measurements_us"][str(tokens)] = token_result
            result["selection"][str(tokens)] = {
                "strategy": (
                    "carry" if weighted_carry < weighted_baseline else "gather_raw"
                ),
                "speedup": weighted_baseline / weighted_carry,
            }
            print(
                f"T={tokens:4d} weighted selector="
                f"{result['selection'][str(tokens)]['strategy']} "
                f"({weighted_baseline / weighted_carry:.3f}x)",
                flush=True,
            )

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}", flush=True)
    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
