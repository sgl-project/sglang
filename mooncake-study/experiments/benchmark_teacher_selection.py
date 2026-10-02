"""Compare teacher extraction including host-known row-index preparation."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from sglang.srt.training_capture.teacher import capture_teacher, warmup_teacher_capture


def tensor_indices(raw, vocab, indices):
    return capture_teacher(
        raw, vocab, torch.tensor(indices, dtype=torch.long, device=raw.device)
    )


def measure(function, arguments, iterations):
    for _ in range(5):
        function(*arguments)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        function(*arguments)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1e6 / iterations


def benchmark(args):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    rows = []
    probes = []
    functions = {
        "tensor_indices": tensor_indices,
        "host_indices": capture_teacher,
    }
    for vocab in (32768, 151936):
        warmup_teacher_capture(vocab, "cuda")
        for dtype in (torch.float32, torch.bfloat16):
            for indices in ([3], [2, 3, 4], list(range(8)), [0, 3, 7]):
                raw = torch.randn(8, vocab + 64, device="cuda", dtype=dtype)
                raw[:, vocab:] = 10000
                arguments = raw, vocab, indices
                expected = tensor_indices(*arguments)
                actual = capture_teacher(*arguments)
                for field in ("token_ids", "logits", "logsumexp"):
                    torch.testing.assert_close(
                        getattr(actual, field), getattr(expected, field), rtol=0, atol=0
                    )
                samples = {name: [] for name in functions}
                for repetition in range(args.repeats):
                    order = list(functions)
                    if repetition % 2:
                        order.reverse()
                    for name in order:
                        samples[name].append(
                            measure(functions[name], arguments, args.iterations)
                        )
                result = {
                    "vocabulary": vocab,
                    "dtype": str(dtype),
                    "batch_rows": 8,
                    "indices": indices,
                    "eager_wall_us": samples,
                    "median_us": {
                        name: statistics.median(values)
                        for name, values in samples.items()
                    },
                }
                rows.append(result)
                probes.append((vocab, dtype, indices, result))
                print(json.dumps(result), flush=True)
    # CUDA profiling can leave callbacks installed after stop; initialize it
    # only after every uninstrumented timing measurement has finished.
    for vocab, dtype, indices, result in probes:
        raw = torch.randn(8, vocab + 64, device="cuda", dtype=dtype)
        counts = {}
        for name, function in functions.items():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                function(raw, vocab, indices)
                torch.cuda.synchronize()
            counts[name] = {
                event.key: event.count
                for event in profiler.key_averages()
                if event.key
                in ("aten::index_select", "cudaStreamSynchronize", "cudaMemcpyAsync")
            }
        result["profile_calls"] = counts
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.repeats) < 1:
        parser.error("iterations and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        root = Path(__file__).resolve().parents[2]
        paths = [
            "python/sglang/srt/training_capture/teacher.py",
            "mooncake-study/experiments/benchmark_teacher_selection.py",
        ]
        report = {
            "status": "running",
            "source_revision": args.source_revision,
            "source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in paths
            },
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "config": vars(args) | {"output": str(args.output)},
            "scope": "Complete eager teacher extraction, including CPU row selection and any index upload/gather. Alternating order, final synchronization; no serving, D2H, Store or Catalog. Profiler counts are collected separately from timing.",
        }
        try:
            report["cases"] = benchmark(args)
            report["status"] = "completed"
        except Exception as error:
            report["status"], report["error"] = "failed", repr(error)
            raise
        finally:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    main()
