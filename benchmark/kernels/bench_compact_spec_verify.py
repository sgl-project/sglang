"""Serving-path operator comparison against SGLang's multimem gather."""

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
import torch.distributed as dist

from sglang.srt.distributed import get_tp_group
from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    MultimemAllGatherer,
)
from sglang.srt.distributed.parallel_state import cleanup_dist_env_and_memory
from sglang.srt.speculative.compact_verify.core import Verify
from sglang.srt.speculative.compact_verify.engine import (
    ServingVerifier,
    runtime_supported,
)


def fixture(b, k, v, rank, tp, kind):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1847)
    # Identical full fixture on all ranks; release target full logits before timing.
    target = torch.randn((b, k + 1, v), generator=gen, device=device).to(torch.bfloat16)
    q = torch.softmax(
        target[:, :k].float()
        + 0.5 * torch.randn((b, k, v), generator=gen, device=device),
        -1,
    )
    candidates = torch.randint(
        v, (b, k + 1), generator=gen, device=device, dtype=torch.int32
    )
    coins = torch.rand((b, k + 1), generator=gen, device=device)
    final = torch.rand(b, generator=gen, device=device)
    if kind == "identical":
        q = torch.softmax(target[:, :k].float(), -1)
    candidates[:, 1:] = (
        torch.multinomial(q.reshape(-1, v), 1, generator=gen).view(b, k).int()
    )
    if kind == "all_accept":
        coins.zero_()
    elif kind in ("first_reject", "last_reject"):
        step = 0 if kind == "first_reject" else k - 1
        coins.zero_()
        coins[:, step] = 0.999
        for i in range(b):
            target[i, step, candidates[i, step + 1]] = -20
    local = target[..., rank * (v // tp) : (rank + 1) * (v // tp)].contiguous()
    return Verify(local, q, candidates, coins, final)


def _init_distributed():
    from sglang.srt.distributed.parallel_state import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from sglang.srt.runtime_context import publish
    from sglang.srt.server_args import ServerArgs

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    publish(ServerArgs(model_path="dummy", tp_size=world_size), role="scheduler")
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        backend="nccl",
    )
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    return rank, world_size


def measure(fn):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(30):
        dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        value = torch.tensor(
            start.elapsed_time(end) * 1000, device="cuda", dtype=torch.float64
        )
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        values.append(value.item())
        del output
    return {
        "samples_us": values,
        "p50_us": statistics.median(values),
        "p90_us": float(torch.tensor(values, dtype=torch.float64).quantile(0.9)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank, tp = _init_distributed()
    assert tp == 4 and runtime_supported()
    root = Path(__file__).resolve().parents[2]
    assert not subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    obj = fixture(128, 7, 154880, rank, tp, "random")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    initial_alloc = torch.cuda.memory_allocated()
    initial_free = torch.cuda.mem_get_info()[0]
    gather = MultimemAllGatherer(max_tokens=1024, enabled=True, skip_entry_sync=True)
    group = get_tp_group()

    def baseline():
        full = gather(obj.local.flatten(0, 1)).float().view(128, 8, 154880)
        p = torch.softmax(full, -1)
        output = obj.sample(p, obj.q, obj.candidates, obj.idx, obj.coins)
        for value in output:
            group.broadcast(value, 0)
        return output

    for _ in range(3):
        expected = baseline()
    assert gather._state is not None and gather._state.symm_mem_hdl.multicast_ptr != 0
    del expected
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph_a = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph_a):
            output_a = baseline()
    stream.synchronize()
    a_memory = {
        "extra_peak_allocated": torch.cuda.max_memory_allocated() - initial_alloc,
        "extra_live_device": initial_free - torch.cuda.mem_get_info()[0],
    }
    graph_a.replay()
    torch.cuda.synchronize()
    before_c_alloc = torch.cuda.memory_allocated()
    before_c_free = torch.cuda.mem_get_info()[0]
    torch.cuda.reset_peak_memory_stats()
    runner = ServingVerifier(
        obj.local, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx
    )
    output_c = runner.run()
    torch.cuda.synchronize()
    for actual, expected in zip(output_c[:3], output_a):
        torch.testing.assert_close(actual.flatten(), expected.flatten(), rtol=0, atol=0)
    c_memory = {
        "extra_peak_allocated": torch.cuda.max_memory_allocated() - before_c_alloc,
        "extra_live_device": before_c_free - torch.cuda.mem_get_info()[0],
    }

    def candidate():
        runner.bind(
            obj.local, obj.q, obj.candidates, obj.coins, obj.final_coins, obj.idx
        )
        return runner.run()

    reports = []
    for repairs in (0, 5):
        runner.force_flags.zero_()
        runner.force_flags[:repairs] = True
        reports.append(
            {
                "forced_repairs": repairs,
                "blocks": [
                    measure(fn)
                    for fn in (graph_a.replay, candidate, candidate, graph_a.replay)
                ],
            }
        )
    memory = [None] * 4
    dist.all_gather_object(memory, {"baseline": a_memory, "candidate": c_memory})
    graph_a.reset()
    runner.close()
    cleanup_dist_env_and_memory()
    if rank == 0:
        result = {
            "source": sha,
            "status": "PASS",
            "baseline": "SGLang multimem + original sampler + TP broadcast, captured",
            "candidate": "serving graph including input/pointer binding",
            "batch": 128,
            "drafts": 7,
            "cells": reports,
            "memory_per_rank": memory,
            "normal_shutdown": True,
        }
        args.output.write_text(json.dumps(result, indent=2))
        print(
            json.dumps(
                {
                    "source": sha,
                    "status": "PASS",
                    "p50_blocks": [[x["p50_us"] for x in r["blocks"]] for r in reports],
                    "memory": memory,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
