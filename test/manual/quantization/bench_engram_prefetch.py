"""Fixed H20-friendly host-table A/B, no model download or parameter sweep.

Run: python test/manual/quantization/bench_engram_prefetch.py --output /tmp/engram.json
The two 264 MiB tables exceed L2. A fixed GEMM between Engram layers stands
in for backbone compute; these are synthetic timings, not model throughput.
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.layers.engram_prefetch import EngramPrefetcher
from sglang.srt.runtime_context import reset_context
from sglang.test.engram_utils import make_engram_layers
from sglang.test.test_utils import publish_build_topology


@torch.inference_mode()
def benchmark(output, execution="eager"):
    torch.manual_seed(73)
    publish_build_topology(tp_size=1, disable_custom_all_reduce=True)
    with tempfile.NamedTemporaryFile() as rendezvous:
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="file://" + rendezvous.name,
        )
        initialize_model_parallel()
        try:
            layers = make_engram_layers(
                rows=1 << 20, dim=256, layer_ids=(0, 2), hidden_size=256
            )
            prefetch = EngramPrefetcher(layers, 512, 1)
            gap_weight = torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda")
            results = {
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "chunk_size": 512,
                "prefetch_layers": 1,
                "execution": execution,
                "host_bytes": sum(layer.embed.host_table.nbytes for layer in layers),
                "prefetch_buffer_bytes": prefetch.buffer_bytes,
                "workloads": {},
            }
            for name, tokens in (("decode", 8), ("prefill", 4096)):
                hashes = torch.randint(0, 1 << 20, (tokens, 2, 4), device="cuda")
                source = torch.randn(
                    tokens, 2, 256, device="cuda", dtype=torch.bfloat16
                )
                gap_input = torch.randn(
                    tokens, 2048, device="cuda", dtype=torch.bfloat16
                )

                def run(enabled):
                    if enabled:
                        prefetch.begin(hashes, None)
                    value = layers[0](
                        source, hashes[:, 0], prefetch=prefetch if enabled else None
                    )
                    # Identical compute in both paths; output is retained until the second layer finishes.
                    gap = gap_input @ gap_weight
                    value = layers[1](
                        value, hashes[:, 1], prefetch=prefetch if enabled else None
                    )
                    return value, gap

                entries = {}
                for enabled in (False, True):
                    for _ in range(5):
                        hashes.random_(0, 1 << 20)
                        run(enabled)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    base_bytes = torch.cuda.memory_allocated()
                    graph, captured_output = None, None

                    def execute():
                        return run(enabled)

                    if execution == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured_output = run(enabled)
                        torch.cuda.synchronize()
                        execute = graph.replay
                    capture_bytes = torch.cuda.memory_allocated() - base_bytes
                    start = torch.cuda.Event(enable_timing=True)
                    stop = torch.cuda.Event(enable_timing=True)
                    samples = []
                    wall = time.perf_counter()
                    for _ in range(20):
                        # Fresh uniformly distributed rows prevent the small
                        # per-batch working set from staying resident in L2.
                        # Index generation precedes the timed GPU region.
                        hashes.random_(0, 1 << 20)
                        start.record()
                        execute()
                        stop.record()
                        stop.synchronize()
                        samples.append(start.elapsed_time(stop))
                    entries["prefetch" if enabled else "synchronous"] = {
                        "median_ms": statistics.median(samples),
                        "total_wall_s": time.perf_counter() - wall,
                        "peak_increment_bytes": torch.cuda.max_memory_allocated()
                        - base_bytes,
                        "allocated_after_bytes": torch.cuda.memory_allocated(),
                        "capture_increment_bytes": capture_bytes,
                    }
                    # Drop one graph before measuring the other path's memory.
                    del execute, graph, captured_output
                entries["speedup"] = (
                    entries["synchronous"]["median_ms"]
                    / entries["prefetch"]["median_ms"]
                )
                results["workloads"][name] = entries
                if (
                    execution == "eager"
                    and name == "prefill"
                    and entries["speedup"] <= 1
                ):
                    trace_path = str(Path(output).with_suffix(".trace.json"))
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        for enabled in (False, True):
                            hashes.random_(0, 1 << 20)
                            with torch.profiler.record_function(
                                "prefetch" if enabled else "synchronous"
                            ):
                                run(enabled)
                        torch.cuda.synchronize()
                    prof.export_chrome_trace(trace_path)
                    results["trace"] = trace_path
            Path(output).write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(results, indent=2))
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
            reset_context()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/engram-prefetch.json")
    parser.add_argument("--execution", choices=("eager", "graph"), default="eager")
    args = parser.parse_args()
    benchmark(args.output, args.execution)
