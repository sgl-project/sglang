"""Compare repeated CUDA graph forks with main-first and side-first capture.

This synthetic diagnostic needs only PyTorch and one CUDA GPU. It measures
unprofiled replay latency separately from the trace, checks changed inputs,
and reports replay streams. It does not replace the K3 end-to-end benchmark.

Run in separate processes to compare connection settings::

    CUDA_DEVICE_MAX_CONNECTIONS=8 python benchmark/kernels/bench_cuda_graph_stream_order.py --output-dir /tmp/streams-8
    CUDA_DEVICE_MAX_CONNECTIONS=128 python benchmark/kernels/bench_cuda_graph_stream_order.py --output-dir /tmp/streams-128
"""

import argparse
import collections
import json
import os
import statistics
from pathlib import Path


def summarize_trace(path):
    data = json.loads(path.read_text())
    graphs = collections.defaultdict(list)
    for event in data["traceEvents"]:
        if event.get("cat") == "kernel":
            graph_id = event.get("args", {}).get("graph id", 0)
            if graph_id:
                graphs[graph_id].append(event)
    return {
        "cuda_driver_version": data.get("cuda_driver_version"),
        "cuda_runtime_version": data.get("cuda_runtime_version"),
        "graphs": {
            str(graph_id): {
                "streams": len({e["args"]["stream"] for e in events}),
                "kernels": len(events),
                "replays": len({e["args"]["correlation"] for e in events}),
            }
            for graph_id, events in graphs.items()
        },
    }


def run_case(torch, args, order):
    main = torch.cuda.Stream()
    side = torch.cuda.Stream()
    x = torch.empty(args.elements, device="cuda")
    # Keep every buffer alive; allocator reuse must not change the graph DAG.
    main_buffers = [torch.empty_like(x) for _ in range(args.layers)]
    side_buffers = [torch.empty_like(x[:128]) for _ in range(args.layers)]
    outputs = [torch.empty_like(x) for _ in range(args.layers)]

    def forward():
        value = x
        for full, small, output in zip(main_buffers, side_buffers, outputs):
            if order != "serial":
                # The fork precedes BOTH branches in either capture order.
                side.wait_stream(main)
            if order == "side-first":
                with torch.cuda.stream(side):
                    torch.mul(value[:128], 0.25, out=small)
                    small.add_(0.125)
            torch.mul(value, 0.75, out=full)
            full.add_(0.125)
            if order == "main-first":
                with torch.cuda.stream(side):
                    torch.mul(value[:128], 0.25, out=small)
                    small.add_(0.125)
            elif order == "serial":
                torch.mul(value[:128], 0.25, out=small)
                small.add_(0.125)
            if order != "serial":
                main.wait_stream(side)
            torch.add(full.view(-1, 128), small, out=output.view(-1, 128))
            value = output
        return value

    x.fill_(1)
    main.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(main):
        for _ in range(3):
            forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=main):
        result = forward()

    # Both branches have an analytical result for uniform inputs. Mutating x
    # between replays exposes missing dependencies and stale input reads.
    for initial in (1.0, -2.0, 3.0):
        x.fill_(initial)
        graph.replay()
        expected = initial
        for _ in range(args.layers):
            expected += 0.25
        torch.testing.assert_close(result, torch.full_like(result, expected))

    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(7):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / args.replays)

    path = args.output_dir / f"{order}.trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(str(path))
    summary = summarize_trace(path)
    summary.update(order=order, median_replay_ms=statistics.median(samples))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=92)
    parser.add_argument("--elements", type=int, default=7168)
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    if (
        args.layers < 1
        or args.replays < 1
        or args.elements < 128
        or args.elements % 128
    ):
        parser.error(
            "layers/replays must be positive; elements must be a multiple of 128"
        )

    import torch

    if not torch.cuda.is_available():
        parser.error("a CUDA GPU is required")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "connections": os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "unset"),
        "layers": args.layers,
        "elements": args.elements,
        "cases": [],
    }
    for order in ("side-first", "main-first", "serial"):
        result = run_case(torch, args, order)
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
