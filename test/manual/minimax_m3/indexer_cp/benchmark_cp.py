"""Four-rank native SGLang indexer CP parity and full-chain graph timing.

Run with torchrun --standalone --nproc_per_node=4 inside the pinned ROCm env.
This measures the indexer, including both gathers; it is not model throughput.
"""

import argparse
import hashlib
import json
import os
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist


def setup():
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=10))
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.runtime_context import get_parallel

    get_parallel().override_permanently(
        tp_size=4,
        attn_tp_size=4,
        attn_dp_size=1,
        attn_cp_size=1,
        enable_dp_attention=False,
    )
    group = GroupCoordinator(
        [list(range(4))],
        rank,
        "nccl",
        use_pynccl=True,
        use_mscclpp=False,
        use_custom_allreduce=True,
        use_torch_symm_mem_all_reduce=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_npu_communicator=False,
        group_name="m3_cp_benchmark",
    )
    return rank, group


def inputs(batch, max_len, dtype, rank, kind="random"):
    torch.manual_seed(20260928)
    device = torch.device("cuda", rank)
    padded_len = (max_len + 127) // 128 * 128
    nslots = batch * padded_len
    # Shuffle physical pages independently of logical block ownership.
    pages = torch.randperm(nslots // 16, device=device, dtype=torch.int32)
    table = (
        (pages[:, None] * 16 + torch.arange(16, device=device))
        .reshape(batch, padded_len)
        .to(torch.int32)
    )
    cache = torch.randn((nslots, 1, 128), device=device, dtype=torch.bfloat16).to(dtype)
    all_q = torch.randn((batch, 4, 128), device=device, dtype=torch.bfloat16)
    if kind in ("ties", "one_shard"):
        cache.zero_()
        all_q.zero_()
        all_q[:, :, 0] = 1
        if kind == "one_shard":
            values = ((torch.arange(padded_len, device=device) // 128) % 4) == 3
            values = values.to(torch.float32).repeat(batch) * 8
            # Assignment to FP8 via a BF16 staging tensor is supported on ROCm.
            staging = cache.to(torch.bfloat16)
            staging[table.flatten().long(), 0, 0] = values.to(torch.bfloat16)
            cache = staging.to(dtype)
    lengths = torch.full((batch,), max_len, dtype=torch.int64, device=device)
    if kind == "mixed" and batch > 1:
        lengths[1:] = 1024
    slots = torch.arange(batch, dtype=torch.int64, device=device)
    return all_q[:, rank : rank + 1], cache, table, slots, lengths


def functions(cp, data, max_len, scale=1.0):
    from sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx import (
        flash_decode_with_topk_idx,
    )

    q, cache, table, slots, lengths = data

    def native():
        return flash_decode_with_topk_idx(
            q=q,
            sink=None,
            k_cache=cache,
            v_cache=None,
            req_to_token=table,
            seq_lens=lengths,
            max_seqlen=max_len,
            slot_ids=slots,
            block_size=128,
            topk=16,
            init_blocks=1,
            local_blocks=2,
            disable_index_value=True,
            k_scale=scale,
        )[1]

    def candidate():
        return cp(q, cache, table, slots, lengths, max_len, 1, 2, k_scale=scale)

    return {"native_tp": native, "indexer_cp": candidate}


def exact(a, b, label):
    equal = torch.equal(a, b)
    ok = torch.tensor(int(equal), device=a.device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if not ok.item():
        mismatch = int((a != b).sum().item())
        raise AssertionError(
            f"{label}: rank {dist.get_rank()} differs at {mismatch} IDs"
        )


def oracle(data, scale):
    q, cache, table, slots, lengths = data
    out = torch.full((1, q.shape[0], 16), -1, dtype=torch.int32, device=q.device)
    for row, length in enumerate(lengths.tolist()):
        if length == 0:
            continue
        k = cache[table[row, :length].long(), 0].float()
        score = (k @ q[row, 0].float()) * (128**-0.5 * 1.4426950409 * scale)
        nblocks = (length + 127) // 128
        padded = torch.full((nblocks * 128,), -torch.inf, device=q.device)
        padded[:length] = score
        score = padded.view(nblocks, 128).max(dim=1).values
        score[:1] = 1e30
        score[max(0, nblocks - 2) :] = 1e29
        chosen = torch.argsort(score, descending=True, stable=True)[:16].sort().values
        out[0, row, : chosen.numel()] = chosen.to(torch.int32)
    return out


def capture(group, fns, calls):
    graphs, outputs = {}, {}
    for name, fn in fns.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with group.graph_capture() as context:
            # Exercise SGLang's registered-buffer warmup semantics.
            fn()
            with torch.cuda.graph(graph, stream=context.stream):
                for _ in range(calls):
                    output = fn()
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        graphs[name], outputs[name] = graph, output
    return graphs, outputs


def timing(graphs, calls, rounds=7, replays=100):
    samples = {name: [] for name in graphs}
    for repeat in range(rounds):
        names = list(graphs)
        if repeat % 2:
            names.reverse()
        for name in names:
            dist.barrier()
            graph = graphs[name]
            for _ in range(5):
                graph.replay()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            us = torch.tensor(
                start.elapsed_time(end) * 1000 / (replays * calls), device="cuda"
            )
            # TP step completion is bounded by the slowest rank.
            dist.all_reduce(us, op=dist.ReduceOp.MAX)
            samples[name].append(us.item())
    return {
        name: {"median_us": statistics.median(x), "rounds_us": x}
        for name, x in samples.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    rank, group = setup()
    from sglang.srt.layers.attention.minimax_sparse_ops.indexer_cp import (
        MiniMaxIndexerCP,
    )

    cp = MiniMaxIndexerCP(group)
    cp.warmup()
    report = {
        "scope": "four-rank indexer chain, both gathers included; not model throughput",
        "torch": torch.__version__,
        "gpu": str(torch.cuda.get_device_properties(rank)),
        "custom_gather_available": group._has_aiter_custom_all_gather(),
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timing": "7 alternating rounds, 100 graph replays, 8 calls per graph; max rank latency; warm repeated tensors",
        "checks": [],
        "benchmarks": [],
        "completed": False,
    }

    def save():
        if rank == 0:
            args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            for kind in ("random", "ties", "one_shard", "mixed"):
                data = inputs(3, 8193, dtype, rank, kind)
                data[-1][-1] = 0
                fns = functions(cp, data, 8193, scale=0.5)
                native, candidate = fns["native_tp"](), fns["indexer_cp"]()
                exact(native, candidate, f"eager {dtype} {kind}")
                exact(candidate, oracle(data, 0.5), f"oracle {dtype} {kind}")
                graphs, outputs = capture(group, fns, 1)
                data[-1].copy_(torch.tensor([127, 17 * 128 + 1, 0], device="cuda"))
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                exact(
                    outputs["native_tp"], outputs["indexer_cp"], "changed-length replay"
                )
                exact(outputs["indexer_cp"], oracle(data, 0.5), "changed-length oracle")
                report["checks"].append(
                    {"dtype": str(dtype), "kind": kind, "passed": True}
                )
                save()
                if rank == 0:
                    print("PASS", dtype, kind, flush=True)
                del graphs, outputs, fns, data
        cases = [
            (b, length, "uniform")
            for length in (8192, 32768, 131072)
            for b in ((1, 16) if args.quick else (1, 2, 8, 16, 32))
        ]
        if not args.quick:
            cases += [(16, 131072, "mixed"), (32, 131072, "mixed")]
        for batch, length, kind in cases:
            data = inputs(batch, length, torch.float8_e4m3fn, rank, kind)
            fns = functions(cp, data, length)
            exact(fns["native_tp"](), fns["indexer_cp"](), "benchmark eager")
            graphs, outputs = capture(group, fns, 8)
            exact(outputs["native_tp"], outputs["indexer_cp"], "benchmark graph")
            times = timing(graphs, 8)
            baseline = times["native_tp"]["median_us"]
            candidate = times["indexer_cp"]["median_us"]
            row = {
                "batch": batch,
                "context": length,
                "distribution": kind,
                "times": times,
                "latency_reduction_pct": (1 - candidate / baseline) * 100,
                "speedup": baseline / candidate,
            }
            report["benchmarks"].append(row)
            save()
            if rank == 0:
                print(json.dumps(row), flush=True)
            del graphs, outputs, fns, data
        report["completed"] = True
        save()
    except Exception as exc:
        report["error"] = repr(exc)
        save()
        raise
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
