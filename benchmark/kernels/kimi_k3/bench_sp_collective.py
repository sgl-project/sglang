#!/usr/bin/env python3
"""Correctness benchmark and tuner for K3 SP-MoE AG/RS.

Launch with one process per GPU. Rank 0 writes a JSON file containing the best
launch config per global-token size; every candidate is checked against NCCL
before it is timed.

Tables are keyed on torch.cuda.get_device_name(), so a table measured on one
fabric does not carry to another even at equal world size: GB300 world=8 crosses
MNNVL between two 4-GPU nodes, B300 world=8 is one node over NVSwitch. Pass
--output-auto to write to the exact path the runtime looks up.

GB300, 2x4 MNNVL:

  SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE=1 \
  SGLANG_FORCE_CUSTOM_ALL_REDUCE_V2_PUSH_SIZE_KB=32768 \
  torchrun --nnodes=2 --nproc-per-node=4 ... bench_sp_collective.py --tune ...

B300, 1x8 single node -- no multinode env vars, and leave the push size
unforced so the sweep can pick it:

  torchrun --nnodes=1 --nproc-per-node=8 \
    benchmark/kernels/kimi_k3/bench_sp_collective.py --tune --output-auto
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import torch.distributed as dist

from sglang.kernels.ops.kimi_k3 import sp_collective
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
        help="Global token counts; each must be divisible by world size.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument(
        "--validate-graph",
        action="store_true",
        help="Capture and replay the table-selected RS+AG pair for every token size.",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Run table-selected CUDA graph validation without timing candidates.",
    )
    parser.add_argument(
        "--local-residual",
        action="store_true",
        help="Pass only this rank's destination shard to the fused RS epilogue.",
    )
    parser.add_argument(
        "--validate-attn-res-carry",
        action="store_true",
        help="Compare local attn-res aggregation/bank writes with full-batch execution.",
    )
    parser.add_argument(
        "--num-blocks", type=int, nargs="+", default=[2, 4, 8, 16, 32, 64, 96]
    )
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument(
        "--direct-block-sizes", type=int, nargs="+", default=[128, 256, 512, 1024]
    )
    parser.add_argument(
        "--ag-strategies",
        nargs="+",
        choices=["push", "direct"],
        default=["push", "direct"],
    )
    parser.add_argument(
        "--rs-strategies",
        nargs="+",
        choices=["push", "pull"],
        default=["push", "pull"],
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-auto",
        action="store_true",
        help="Write to the exact path the runtime looks up for this device and "
        "world size, under sglang/kernels/ops/kimi_k3/configs/sp_collective. "
        "Overrides --output.",
    )
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
        flat_idx = int(diff.argmax())
        raise AssertionError(
            f"{label}: max_abs={float(diff.max()):.6g} at flat={flat_idx}, "
            f"actual={float(actual.view(-1)[flat_idx]):.6g}, "
            f"expected={float(expected.view(-1)[flat_idx]):.6g}; {exc}"
        ) from exc


def validate_attn_res_carry(
    rank: int, world: int, device: torch.device, hidden_size: int
) -> None:
    """Check local aggregation and snapshot writes against the full stream."""
    from sglang.srt.layers.attn_residual import AttnResidual
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.linear import ReplicatedLinear

    tokens = world * 8
    gen = torch.Generator(device=device)
    gen.manual_seed(9012)
    initial = torch.randn(
        tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
        generator=gen,
    )
    score_norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    out_norm = RMSNorm(hidden_size, eps=1e-6).to(device=device, dtype=torch.bfloat16)
    score_proj = ReplicatedLinear(
        hidden_size,
        1,
        bias=False,
        quant_config=None,
        prefix="attn_res_carry_check",
    ).to(device=device, dtype=torch.bfloat16)
    score_norm.weight.data.copy_(torch.linspace(0.75, 1.25, hidden_size, device=device))
    out_norm.weight.data.copy_(torch.linspace(1.25, 0.75, hidden_size, device=device))
    score_proj.weight.data.copy_(
        torch.linspace(-0.01, 0.01, hidden_size, device=device).view(1, -1)
    )

    full = AttnResidual(initial, 8)
    carried = AttnResidual(initial, 8)
    full_normed, _ = full.forward(
        initial, None, score_proj, score_norm, out_norm, write=True
    )
    carried_normed, _ = carried.forward(
        initial, None, score_proj, score_norm, out_norm, write=True
    )
    assert_close(carried_normed, full_normed, "attn-res initial write")

    local_tokens = tokens // world
    lo = rank * local_tokens
    hi = lo + local_tokens
    rows = slice(lo, hi)
    gathered = torch.empty_like(initial)
    for layer in range(1, 25):
        head = torch.randn(
            tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        write = layer % 12 == 0
        full_normed, full_prefix = full.forward(
            head, None, score_proj, score_norm, out_norm, write=write
        )
        local_normed, local_prefix = carried.forward(
            head[rows].contiguous(),
            None,
            score_proj,
            score_norm,
            out_norm,
            rows=rows,
            write=write,
        )
        dist.all_gather_into_tensor(gathered, local_normed)
        assert_close(gathered, full_normed, f"attn-res layer={layer}")
        assert_close(local_prefix, full_prefix[rows], f"prefix layer={layer}")
        assert_close(
            carried.block_residual[rows, : full.num_valid_blocks],
            full.block_residual[rows, : full.num_valid_blocks],
            f"bank layer={layer}",
        )
    if rank == 0:
        print(
            "attention-residual shard carry validated through two local bank writes",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    device = torch.device("cuda", local_rank)
    cpu_group = dist.new_group(backend="gloo")

    # GroupCoordinator constructs custom-AR over its CPU group as well: the
    # topology probe and symmetric-memory rendezvous need a non-NCCL group,
    # while the NCCL world remains the reference collective below.
    comm = CustomAllReduceV2(group=cpu_group, device=device)
    if comm.disabled or comm.mc_base_ptr == 0:
        raise RuntimeError("CustomAllReduceV2 with multicast is required")
    sp_collective.register_comm(comm.obj, pull_sem_mc_ptr=comm.pull_sem_mc_ptr)
    if args.validate_attn_res_carry:
        validate_attn_res_carry(rank, world, device, args.hidden_size)
        dist.barrier()

    candidates = [
        sp_collective.Tuning(blocks, threads)
        for blocks in args.num_blocks
        for threads in args.block_sizes
        if blocks < comm.config.num_push_blocks
    ]
    if not args.tune:
        candidates = [sp_collective.DEFAULT_TUNING]
    direct_candidates = [
        sp_collective.Tuning(blocks, threads)
        for blocks in args.num_blocks
        for threads in args.direct_block_sizes
        if blocks <= comm.config.num_pull_blocks
    ]
    if not args.tune:
        direct_candidates = [sp_collective.DEFAULT_TUNING]

    result = {
        "device": torch.cuda.get_device_name(),
        "world_size": world,
        "hidden_size": args.hidden_size,
        "push_slot_bytes": comm.max_push_size,
        "num_push_counters": comm.config.num_push_blocks,
        "local_residual": args.local_residual,
        "configs": {"reduce_scatter": {}, "all_gather": {}},
        "measurements_us": {"reduce_scatter": {}, "all_gather": {}},
    }

    for tokens in args.tokens:
        if tokens % world:
            raise ValueError(f"tokens={tokens} must be divisible by world={world}")
        local_tokens = tokens // world
        local_bytes = local_tokens * args.hidden_size * 2
        if local_bytes > comm.max_push_size:
            if rank == 0:
                print(
                    f"skip T={tokens}: local shard {local_bytes} B exceeds "
                    f"push slot {comm.max_push_size} B",
                    flush=True,
                )
            continue

        gen = torch.Generator(device=device)
        gen.manual_seed(1234 + rank)
        rs_input = torch.randn(
            tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        residual_gen = torch.Generator(device=device)
        residual_gen.manual_seed(5678)
        residual_full = torch.randn(
            tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=residual_gen,
        )
        lo = rank * local_tokens
        residual = (
            residual_full[lo : lo + local_tokens].contiguous()
            if args.local_residual
            else residual_full
        )
        # The fused epilogue accumulates the rank contributions and residual
        # in fp32, then rounds once to bf16.  NCCL RS followed by bf16 add
        # rounds twice and can differ by one bf16 ulp, so use the fused
        # mathematical contract for correctness and time the legacy order
        # separately below.
        rs_sum = rs_input.float()
        dist.all_reduce(rs_sum)
        rs_ref = (
            rs_sum[lo : lo + local_tokens]
            + (
                residual.float()
                if args.local_residual
                else residual[lo : lo + local_tokens].float()
            )
        ).to(torch.bfloat16)

        ag_input = torch.randn(
            local_tokens,
            args.hidden_size,
            dtype=torch.bfloat16,
            device=device,
            generator=gen,
        )
        ag_ref = torch.empty(
            tokens, args.hidden_size, dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(ag_ref, ag_input)

        rs_out = torch.empty_like(rs_ref)
        rs_pull_input = None
        rs_pull_out = None
        rs_pull_mc_ptr = 0
        if "pull" in args.rs_strategies:
            import torch.distributed._symmetric_memory as torch_symm_mem

            from sglang.srt.layers.k3_ar_fusion import symm_alloc

            with symm_alloc():
                rs_pull_input = torch.empty_like(rs_input)
            rs_pull_input.copy_(rs_input)
            rs_pull_out = torch.empty_like(rs_ref)
            rs_pull_handle = torch_symm_mem.rendezvous(
                rs_pull_input, cpu_group.group_name
            )
            if rs_pull_handle is None or rs_pull_handle.multicast_ptr == 0:
                raise RuntimeError("pull RS requires multicast symmetric input")
            rs_pull_mc_ptr = (
                rs_pull_handle.multicast_ptr
                + rs_pull_input.data_ptr()
                - rs_pull_handle.buffer_ptrs[rank]
            )
        ag_out = torch.empty_like(ag_ref)
        ag_direct_out = None
        ag_direct_mc_ptr = 0
        if "direct" in args.ag_strategies:
            import torch.distributed._symmetric_memory as torch_symm_mem

            from sglang.srt.layers.k3_ar_fusion import symm_alloc

            with symm_alloc():
                ag_direct_out = torch.empty_like(ag_ref)
            ag_direct_handle = torch_symm_mem.rendezvous(
                ag_direct_out, cpu_group.group_name
            )
            if ag_direct_handle is None or ag_direct_handle.multicast_ptr == 0:
                raise RuntimeError("direct AG requires multicast symmetric output")
            ag_direct_mc_ptr = (
                ag_direct_handle.multicast_ptr
                + ag_direct_out.data_ptr()
                - ag_direct_handle.buffer_ptrs[rank]
            )

        def validate_graph_pair() -> None:
            rs_dispatch = sp_collective.get_dispatch(
                "reduce_scatter", world, args.hidden_size, tokens, device
            )
            ag_dispatch = sp_collective.get_dispatch(
                "all_gather", world, args.hidden_size, tokens, device
            )
            if rs_dispatch is None or ag_dispatch is None:
                if rank == 0:
                    print(
                        f"T={tokens:5d} table selects NCCL; graph validation skipped",
                        flush=True,
                    )
                return
            if rs_dispatch.strategy == "pull":
                if rs_pull_input is None or rs_pull_out is None:
                    raise RuntimeError("pull RS input was not initialized")
                graph_rs_out = rs_pull_out
                graph_rs = lambda: sp_collective.reduce_scatter_pull(
                    world,
                    rs_pull_input,
                    graph_rs_out,
                    residual,
                    input_mc_ptr=rs_pull_mc_ptr,
                    tuning=rs_dispatch.tuning,
                )
            else:
                graph_rs_out = torch.empty_like(rs_ref)
                graph_rs = lambda: sp_collective.reduce_scatter_res(
                    world,
                    rs_input,
                    graph_rs_out,
                    residual,
                    tuning=rs_dispatch.tuning,
                )
            if ag_dispatch.strategy == "direct":
                if ag_direct_out is None:
                    raise RuntimeError("direct AG output was not initialized")
                graph_ag_out = ag_direct_out
                graph_ag = lambda: sp_collective.all_gather_direct(
                    world,
                    ag_input,
                    graph_ag_out,
                    output_mc_ptr=ag_direct_mc_ptr,
                    tuning=ag_dispatch.tuning,
                )
            else:
                graph_ag_out = torch.empty_like(ag_ref)
                graph_ag = lambda: sp_collective.all_gather(
                    world,
                    ag_input,
                    graph_ag_out,
                    ws_mc_base=comm.mc_base_ptr,
                    tuning=ag_dispatch.tuning,
                )

            def graph_pair():
                graph_rs()
                graph_ag()

            for _ in range(3):
                graph_pair()
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                graph_pair()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            assert_close(graph_rs_out, rs_ref, f"graph RS T={tokens}")
            assert_close(graph_ag_out, ag_ref, f"graph AG T={tokens}")
            if rank == 0:
                print(f"T={tokens:5d} graph replay validated", flush=True)

        if args.graph_only:
            if not args.validate_graph:
                raise ValueError("--graph-only requires --validate-graph")
            validate_graph_pair()
            continue

        rs_measurements = {}
        ag_measurements = {}

        for tuning in candidates:
            key = f"blocks={tuning.num_blocks},threads={tuning.block_size}"
            if "push" in args.rs_strategies:
                sp_collective.reduce_scatter_res(
                    world, rs_input, rs_out, residual, tuning=tuning
                )
            if "push" in args.ag_strategies:
                sp_collective.all_gather(
                    world,
                    ag_input,
                    ag_out,
                    ws_mc_base=comm.mc_base_ptr,
                    tuning=tuning,
                )
            torch.cuda.synchronize()
            if "push" in args.rs_strategies:
                assert_close(rs_out, rs_ref, f"RS push T={tokens} {key}")
            if "push" in args.ag_strategies:
                assert_close(ag_out, ag_ref, f"AG push T={tokens} {key}")

            if "push" in args.rs_strategies:
                rs_measurements[f"push,{key}"] = time_us(
                    lambda: sp_collective.reduce_scatter_res(
                        world, rs_input, rs_out, residual, tuning=tuning
                    ),
                    args.warmup,
                    args.iters,
                )
            if "push" in args.ag_strategies:
                ag_measurements[f"push,{key}"] = time_us(
                    lambda: sp_collective.all_gather(
                        world,
                        ag_input,
                        ag_out,
                        ws_mc_base=comm.mc_base_ptr,
                        tuning=tuning,
                    ),
                    args.warmup,
                    args.iters,
                )

        if rs_pull_input is not None and rs_pull_out is not None:
            for tuning in direct_candidates:
                key = f"pull,blocks={tuning.num_blocks}," f"threads={tuning.block_size}"
                sp_collective.reduce_scatter_pull(
                    world,
                    rs_pull_input,
                    rs_pull_out,
                    residual,
                    input_mc_ptr=rs_pull_mc_ptr,
                    tuning=tuning,
                )
                torch.cuda.synchronize()
                assert_close(rs_pull_out, rs_ref, f"RS T={tokens} {key}")
                rs_measurements[key] = time_us(
                    lambda: sp_collective.reduce_scatter_pull(
                        world,
                        rs_pull_input,
                        rs_pull_out,
                        residual,
                        input_mc_ptr=rs_pull_mc_ptr,
                        tuning=tuning,
                    ),
                    args.warmup,
                    args.iters,
                )

        if ag_direct_out is not None:
            for tuning in direct_candidates:
                key = (
                    f"direct,blocks={tuning.num_blocks}," f"threads={tuning.block_size}"
                )
                sp_collective.all_gather_direct(
                    world,
                    ag_input,
                    ag_direct_out,
                    output_mc_ptr=ag_direct_mc_ptr,
                    tuning=tuning,
                )
                torch.cuda.synchronize()
                assert_close(ag_direct_out, ag_ref, f"AG T={tokens} {key}")
                ag_measurements[key] = time_us(
                    lambda: sp_collective.all_gather_direct(
                        world,
                        ag_input,
                        ag_direct_out,
                        output_mc_ptr=ag_direct_mc_ptr,
                        tuning=tuning,
                    ),
                    args.warmup,
                    args.iters,
                )

        nccl_rs = torch.empty_like(rs_ref)
        nccl_ag = torch.empty_like(ag_ref)

        def nccl_rs_fn():
            dist.reduce_scatter_tensor(nccl_rs, rs_input)
            nccl_rs.add_(
                residual if args.local_residual else residual[lo : lo + local_tokens]
            )

        nccl_rs_us = time_us(nccl_rs_fn, args.warmup, args.iters)
        nccl_ag_us = time_us(
            lambda: dist.all_gather_into_tensor(nccl_ag, ag_input),
            args.warmup,
            args.iters,
        )
        best_rs = min(rs_measurements, key=rs_measurements.get)
        best_ag = min(ag_measurements, key=ag_measurements.get)

        if args.validate_graph:
            validate_graph_pair()

        if rank == 0:
            rs_us = rs_measurements[best_rs]
            ag_us = ag_measurements[best_ag]
            print(
                f"T={tokens:5d} RS {nccl_rs_us:8.2f}->{rs_us:8.2f} us "
                f"({nccl_rs_us / rs_us:5.2f}x) {best_rs}; "
                f"AG {nccl_ag_us:8.2f}->{ag_us:8.2f} us "
                f"({nccl_ag_us / ag_us:5.2f}x) {best_ag}",
                flush=True,
            )
            result["configs"]["reduce_scatter"][str(tokens)] = {
                "strategy": best_rs.split(",")[0],
                "num_blocks": int(best_rs.split(",")[1].split("=")[1]),
                "block_size": int(best_rs.split(",")[2].split("=")[1]),
            }
            result["configs"]["all_gather"][str(tokens)] = {
                "strategy": best_ag.split(",")[0],
                "num_blocks": int(best_ag.split(",")[1].split("=")[1]),
                "block_size": int(best_ag.split(",")[2].split("=")[1]),
            }
            result["measurements_us"]["reduce_scatter"][str(tokens)] = {
                "nccl": nccl_rs_us,
                "candidates": rs_measurements,
            }
            result["measurements_us"]["all_gather"][str(tokens)] = {
                "nccl": nccl_ag_us,
                "candidates": ag_measurements,
            }

    out_path = args.output
    if args.output_auto:
        # Must match sp_collective._table()'s filename exactly, including its
        # space/slash normalization, or the runtime silently misses the table.
        device_name = torch.cuda.get_device_name(device).replace(" ", "_").replace("/", "_")
        out_path = (
            Path(sp_collective.__file__).parent
            / "configs"
            / "sp_collective"
            / f"world={world},H={args.hidden_size},device_name={device_name}.json"
        )

    if rank == 0 and out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path}", flush=True)

    comm.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
