"""Compare full teacher extraction with its original Torch implementation."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.teacher import (
    TeacherRows,
    capture_teacher,
    warmup_teacher_capture,
)


@torch.no_grad()
def reference_capture(raw_logits, vocab_size, row_indices=None):
    """Frozen algorithm from 323df1356; keep validation and owned output types."""
    if raw_logits.ndim != 2 or not raw_logits.is_floating_point():
        raise ContractError(
            "teacher logits must be a floating [rows, vocabulary] tensor"
        )
    if not 128 <= vocab_size <= raw_logits.shape[1]:
        raise ContractError("teacher capture requires the complete unpadded vocabulary")
    scores = raw_logits[:, :vocab_size]
    if row_indices is not None:
        scores = scores.index_select(0, row_indices)
    values, ids = torch.topk(scores, k=128, dim=-1, sorted=True)
    return TeacherRows(
        token_ids=ids.to(torch.int32),
        logits=values.float(),
        logsumexp=torch.logsumexp(scores.float(), dim=-1),
    )


def graph_call(fn, arguments, iterations):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            fn(*arguments)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(iterations):
            result = fn(*arguments)
    return graph, result


def measure(fn, arguments, graph, iterations):
    for _ in range(5):
        fn(*arguments)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*arguments)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - start) * 1e6 / iterations
    for _ in range(3):
        graph.replay()
    start_event, end_event = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start_event.record()
    for _ in range(10):
        graph.replay()
    end_event.record()
    end_event.synchronize()
    return {
        "eager_wall_us": eager_us,
        "batched_graph_us": start_event.elapsed_time(end_event)
        * 1000
        / (10 * iterations),
    }


def benchmark(args):
    torch.manual_seed(71)
    torch.set_num_threads(1)
    # Production warms FP32 scores before admission; retain the observed cost.
    start = time.perf_counter()
    for vocab in args.vocabularies:
        warmup_teacher_capture(vocab, "cuda")
    warmup_seconds = time.perf_counter() - start
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        for vocab in args.vocabularies:
            for count in args.rows:
                for selected in (False, True):
                    raw = torch.randn(count, vocab + 64, device="cuda", dtype=dtype)
                    raw[:, vocab:] = 10000
                    indices = (
                        torch.arange(count - 1, -1, -1, device="cuda")
                        if selected
                        else None
                    )
                    arguments = raw, vocab, indices
                    expected = reference_capture(*arguments)
                    actual = capture_teacher(*arguments)
                    torch.testing.assert_close(
                        actual.token_ids, expected.token_ids, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual.logits, expected.logits, rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual.logsumexp, expected.logsumexp, rtol=1e-6, atol=1e-6
                    )
                    error = (actual.logsumexp - expected.logsumexp).abs().max().item()
                    functions = {
                        "reference": reference_capture,
                        "current": capture_teacher,
                    }
                    graphs = {
                        name: graph_call(fn, arguments, args.iterations)
                        for name, fn in functions.items()
                    }
                    samples = {name: [] for name in functions}
                    for repetition in range(args.repeats):
                        order = list(functions)
                        if repetition % 2:
                            order.reverse()
                        for name in order:
                            samples[name].append(
                                measure(
                                    functions[name],
                                    arguments,
                                    graphs[name][0],
                                    args.iterations,
                                )
                            )
                    medians = {
                        name: {
                            field: statistics.median(row[field] for row in values)
                            for field in values[0]
                        }
                        for name, values in samples.items()
                    }
                    result = {
                        "dtype": str(dtype),
                        "vocabulary": vocab,
                        "rows": count,
                        "selected_rows": selected,
                        "max_lse_abs_error": error,
                        "median": medians,
                        "samples": samples,
                    }
                    rows.append(result)
                    print(json.dumps(result), flush=True)
                    del graphs
    return warmup_seconds, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--vocabularies", nargs="+", type=int, default=[32768, 151936])
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 4, 32, 128])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if (
        min(args.vocabularies) < 128
        or min(args.rows + [args.iterations, args.repeats]) < 1
    ):
        parser.error("positive counts and vocabulary >= 128 are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        root = Path(__file__).resolve().parents[2]
        sources = [
            "python/sglang/srt/training_capture/teacher.py",
            "python/sglang/srt/training_capture/coordinator.py",
            "python/sglang/srt/layers/logsumexp.py",
            "mooncake-study/experiments/benchmark_teacher_capture.py",
        ]
        report = {
            "status": "running",
            "source_revision": args.source_revision,
            "config": vars(args) | {"output": str(args.output)},
            "source_sha256": {
                path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                for path in sources
            },
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "scope": "Complete teacher extraction including top-k, LSE, conversions and optional row selection. Eager host wall time includes dispatch and final synchronization. Batched CUDA-graph event time amortizes host submission; neither is serving latency. No Store, D2H or Catalog work is included.",
        }
        try:
            report["warmup_seconds"], report["cases"] = benchmark(args)
            report["status"] = "completed"
        except Exception as error:
            report["status"], report["error"] = "failed", repr(error)
            raise
        finally:
            json.dump(report, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()
