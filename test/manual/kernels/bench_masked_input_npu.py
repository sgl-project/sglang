"""Ascend token mapping benchmark (no service); run after correctness tests.

python test/manual/kernels/bench_masked_input_npu.py --output /tmp/masked-input
"""

import argparse
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
import torch_npu

from sglang.srt.layers import vocab_parallel_embedding as embedding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = (124160, 248320, 0, 248320, 248320)
    results = []
    for dtype in (torch.int32, torch.int64):
        for size in (32, 4096, 65536):
            x = torch.randint(0, 248320, (size,), dtype=dtype, device="npu")
            for fused in (False, True):
                with patch.object(embedding, "_is_npu", fused):
                    fn = lambda: embedding.get_masked_input_and_mask(x, *cfg)
                    for _ in range(20):
                        fn()
                    torch.npu.synchronize()
                    samples = []
                    for _ in range(args.repeats):
                        start_event = torch.npu.Event(enable_timing=True)
                        end_event = torch.npu.Event(enable_timing=True)
                        start_event.record()
                        start_cpu = time.perf_counter_ns()
                        for _ in range(args.iterations):
                            fn()
                        submit_ns = time.perf_counter_ns() - start_cpu
                        end_event.record()
                        end_event.synchronize()
                        samples.append(
                            dict(
                                cpu_submit_us=submit_ns / args.iterations / 1000,
                                npu_event_us=start_event.elapsed_time(end_event)
                                * 1000
                                / args.iterations,
                            )
                        )
                    row = dict(
                        dtype=str(dtype),
                        size=size,
                        variant="fused" if fused else "reference",
                        iterations=args.iterations,
                        samples=samples,
                        median_cpu_submit_us=statistics.median(
                            s["cpu_submit_us"] for s in samples
                        ),
                        median_npu_event_us=statistics.median(
                            s["npu_event_us"] for s in samples
                        ),
                    )
                    results.append(row)
                    print(json.dumps(row), flush=True)
                    (args.output / "benchmark.json").write_text(
                        json.dumps(results, indent=2)
                    )

    x = torch.arange(32, device="npu", dtype=torch.int64)
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        record_shapes=True,
        profile_memory=True,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
            str(args.output / "profile")
        ),
        experimental_config=torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            l2_cache=False,
            data_simplification=False,
        ),
    ):
        for fused in (False, True):
            with patch.object(embedding, "_is_npu", fused):
                with torch.profiler.record_function(
                    "mask_fused" if fused else "mask_reference"
                ):
                    outputs = embedding.get_masked_input_and_mask(x, *cfg)
        torch.npu.synchronize()
    # Events bracket queued batches and include host-induced queue gaps. They
    # are stream elapsed times; the trace supplies individual kernel durations.
    assert outputs[0].shape == outputs[1].shape == x.shape


if __name__ == "__main__":
    main()
