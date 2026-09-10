"""Compute-only gate, runnable on SM89 before renting an SM120 pair.

PYTHONPATH=python:test python -m nccl_ep_test.single_gpu --report /tmp/compute.json
"""

import argparse
import json
from pathlib import Path

import torch

from .triton_compute import check_expert_output, configure_compute, make_compute_fixture


def run(*, experts=4, capacity=8, replays=1000, require_sm=None):
    from sglang.srt.layers.moe.moe_runner.nccl_ep_triton import run_nccl_ep_triton

    sm = torch.cuda.get_device_capability()
    if require_sm is not None and sm[0] * 10 + sm[1] != require_sm:
        raise RuntimeError(f"Required SM{require_sm}, found SM{sm[0]}{sm[1]}")
    configure_compute()
    dispatched, quant, config = make_compute_fixture(
        hidden=2048, intermediate=1408, experts=experts, capacity=capacity
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_nccl_ep_triton(dispatched, quant, config)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run_nccl_ep_triton(dispatched, quant, config)
    patterns = [
        torch.zeros(experts, dtype=torch.int32, device="cuda"),
        torch.full((experts,), capacity, dtype=torch.int32, device="cuda"),
        torch.arange(experts, dtype=torch.int32, device="cuda") % (capacity + 1),
    ]
    for counts in patterns:
        dispatched.masked_m.copy_(counts)
        expected = run_nccl_ep_triton(dispatched, quant, config)
        graph.replay()
        torch.cuda.synchronize()
        for expert, count in enumerate(counts.cpu().tolist()):
            torch.testing.assert_close(
                output.hidden_states[expert, :count],
                expected.hidden_states[expert, :count],
                rtol=0,
                atol=0,
            )
        check_expert_output(output.hidden_states, dispatched, quant)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for step in range(replays):
        dispatched.masked_m.copy_(patterns[step % len(patterns)])
        graph.replay()
    end.record()
    end.synchronize()
    check_expert_output(output.hidden_states, dispatched, quant)
    return {
        "gate": "single_gpu_compute",
        "passed": True,
        "device": torch.cuda.get_device_name(),
        "sm": list(sm),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "experts": experts,
        "capacity": capacity,
        "replays": replays,
        "replay_with_count_copy_ms": start.elapsed_time(end) / replays,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "native_ep_tested": False,
        "cpu_reference_tolerance": {"rtol": 0.02, "atol": 0.02},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=8)
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--require-sm", type=int)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if min(args.experts, args.capacity, args.replays) <= 0:
        parser.error("experts, capacity and replays must be positive")
    result = run(
        experts=args.experts,
        capacity=args.capacity,
        replays=args.replays,
        require_sm=args.require_sm,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
