"""Full eager CP lookup A/B, using a real-sized per-rank pinned host table.

PYTHONPATH=python torchrun --standalone --nproc-per-node=4 \
    benchmark/engram/bench_cp_lookup.py --output cp-lookup.json
The table is synthetic FP8 with the production DSV4.1-Flash dimensions.
No model is loaded. Includes hash-ID gather, host lookup, exchange and reorder.
"""

import argparse
import json
import os
import statistics
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist

from sglang.srt.arg_groups.engram_config import EngramConfig
from sglang.srt.distributed import init_distributed_environment
from sglang.srt.distributed.parallel_state import initialize_model_parallel
from sglang.srt.layers.engram import EngramEmbedding, EngramLayout
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import publish_build_topology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows-per-column", type=int, default=16000000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--trace-prefix", help="Optional trace AFTER all timed cases")
    args = parser.parse_args()
    rank, world, local = (
        int(os.environ[x]) for x in ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    )
    torch.cuda.set_device(local)
    init_distributed_environment(
        world_size=world, rank=rank, local_rank=local, distributed_init_method="env://"
    )
    publish_build_topology(
        world_rank=rank,
        tp_size=world,
        attn_cp_size=world,
        disable_custom_all_reduce=True,
    )
    initialize_model_parallel()
    get_context().override(
        "cp-benchmark", engram_config=EngramConfig(placement="cpu", lookup="all_reduce")
    )
    cfg = SimpleNamespace(
        engram_layer_ids=[1],
        engram_num_embeddings=[0],
        engram_max_ngram_size=4,
        engram_n_heads=8,
        engram_vocab_size=args.rows_per_column,
        engram_head_dim=256,
    )
    bounds = EngramLayout.from_config(cfg).column_bounds(0)
    t0 = time.perf_counter()
    embed = EngramEmbedding(bounds[-1], 256, 1, bounds)
    rng = np.random.default_rng(42 + rank)
    for tensor, low, high in [(embed.weight, 0, 127), (embed.scale, 124, 130)]:
        array = tensor.view(torch.uint8).numpy().reshape(-1)
        for i in range(0, array.size, 16 << 20):
            view = array[i : i + (16 << 20)]
            view[:] = rng.integers(low, high, view.size, dtype=np.uint8)
    embed.host_table.dirty = True
    embed.finish_load("cp-benchmark")
    initialization_s = time.perf_counter() - t0
    device = torch.device("cuda", local)
    lo = torch.tensor(bounds[:-1], device=device)
    widths = torch.tensor(np.diff(bounds), device=device)
    result = dict(
        base="4f8b6867534c8a6632845353997555fca857dd63",
        world=world,
        table_rows=bounds[-1],
        columns=24,
        dim=256,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        initialization_s=initialization_s,
        cases=[],
    )
    for n, tail in [
        (1, False),
        (32, False),
        (64, False),
        (256, False),
        (1024, False),
        (2048, False),
        (2048, True),
    ]:
        torch.manual_seed(420 + n + rank)
        # Rotate between four independent uniform ID sets; do not mix first
        # access with the warmed steady-state measurements.
        inputs = [
            lo + (torch.rand(n, 24, device=device) * widths).long() for _ in range(4)
        ]
        counts = [n] * world
        if tail:
            counts[-1] -= 1
            if rank == world - 1:
                for ids in inputs:
                    ids[-1].zero_()
        total = sum(counts)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            extend_seq_lens_cpu=[total],
            attn_cp_metadata=SimpleNamespace(
                per_rank_actual_token=counts,
                max_rank_len=[n] * world,
                total_seq_lens=total,
            ),
        )

        def run(i=0):
            return embed(inputs[i % len(inputs)], batch, cp_all_tokens=True)

        first = {}
        for mode in ["all_reduce", "all_to_all"]:
            embed.lookup = mode
            dist.barrier()
            torch.cuda.synchronize()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            out = run()
            end.record()
            end.synchronize()
            first[mode] = start.elapsed_time(end) * 1000
            if mode == "all_reduce":
                reference = out
            else:
                assert torch.equal(reference.view(torch.int16), out.view(torch.int16))
            del out
        del reference
        for mode in ["all_reduce", "all_to_all"]:
            embed.lookup = mode
            for i in range(12):
                run(i)
        torch.cuda.synchronize()
        timings = {"all_reduce": [], "all_to_all": []}
        memory = {}
        for trial in range(args.rounds):
            modes = (
                ["all_reduce", "all_to_all"]
                if trial % 2 == 0
                else ["all_to_all", "all_reduce"]
            )
            for mode in modes:
                embed.lookup = mode
                dist.barrier()
                torch.cuda.synchronize()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for i in range(args.calls):
                    run(i)
                end.record()
                end.synchronize()
                timings[mode].append(start.elapsed_time(end) * 1000 / args.calls)
        for mode in timings:
            embed.lookup = mode
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            out = run()
            torch.cuda.synchronize()
            memory[mode] = torch.cuda.max_memory_allocated() - before
            del out
        records = [None] * world
        dist.all_gather_object(
            records, dict(rank=rank, us=timings, first_us=first, temporary_bytes=memory)
        )
        case = dict(rows_per_rank=n, padded_tail=tail, ranks=records)
        if rank == 0:
            summary = {}
            for mode in timings:
                slowest = [
                    max(r["us"][mode][i] for r in records) for i in range(args.rounds)
                ]
                summary[mode] = dict(
                    median_us=statistics.median(slowest),
                    min_us=min(slowest),
                    max_us=max(slowest),
                )
            case["slowest_rank"] = summary
            print(json.dumps(dict(rows=n, tail=tail, **summary)), flush=True)
        result["cases"].append(case)
    if args.trace_prefix:
        # Use an unpadded full batch; profiling is outside all timing windows.
        batch.attn_cp_metadata.per_rank_actual_token = [n] * world
        batch.attn_cp_metadata.total_seq_lens = n * world
        batch.extend_seq_lens_cpu = [n * world]
        inputs[0][-1].copy_(lo)
        for mode in ["all_reduce", "all_to_all"]:
            embed.lookup = mode
            dist.barrier()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                run()
            prof.export_chrome_trace(f"{args.trace_prefix}-{mode}-rank{rank}.json")
    if rank == 0:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
