"""Benchmark Qwen4-Exp QSA MQA scoring backends."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT / "python"))

from sglang.srt.utils import is_sm120_supported  # noqa: E402

HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 16
COMPRESS_RATIO = 4
DECODE_SOURCE_TOKENS = 32768


def _imports():
    from sglang.kernels.ops.attention.qsa.mqa import (
        triton_qsa_mqa_decode,
        triton_qsa_mqa_prefill,
    )
    from sglang.srt.layers.attention.qsa.mqa import (
        tilelang_qsa_mqa_decode,
        tilelang_qsa_mqa_prefill,
        torch_qsa_mqa_decode,
        torch_qsa_mqa_prefill,
    )

    return {
        "triton": (triton_qsa_mqa_prefill, triton_qsa_mqa_decode),
        "tilelang": (tilelang_qsa_mqa_prefill, tilelang_qsa_mqa_decode),
        "torch": (torch_qsa_mqa_prefill, torch_qsa_mqa_decode),
    }


def _time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def _prefill_case(source_tokens: int):
    torch.manual_seed(100 + source_tokens)
    keys = source_tokens // COMPRESS_RATIO
    q = torch.randn(source_tokens, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(keys, 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    starts = torch.zeros(source_tokens, device="cuda", dtype=torch.int32)
    ends = torch.div(
        torch.arange(source_tokens, device="cuda", dtype=torch.int32) + 1,
        COMPRESS_RATIO,
        rounding_mode="floor",
    ).clamp_max_(keys)
    return q, k, starts, ends


def _decode_case(batch: int, backend: str):
    torch.manual_seed(200 + batch)
    compressed_len = DECODE_SOURCE_TOKENS // COMPRESS_RATIO
    max_pages = (compressed_len + PAGE_SIZE - 1) // PAGE_SIZE
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q, torch.zeros_like(q)], dim=1)
    cache = torch.randn(
        batch * max_pages,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    page_table = torch.arange(
        batch * max_pages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_pages)
    lengths = torch.full((batch,), compressed_len, device="cuda", dtype=torch.int32)
    return q, cache, page_table, lengths, compressed_len


def _score_stats(actual, expected, starts, ends):
    finite = torch.isfinite(expected)
    error = (actual[finite] - expected[finite]).abs().float()
    max_error = error.max()
    exact_rows = 0
    tied_rows = 0
    unexplained_rows = 0
    for row in range(expected.shape[0]):
        start = int(starts[row])
        end = int(ends[row])
        width = min(512, end - start)
        if width <= 0:
            exact_rows += 1
            continue
        actual_idx = torch.topk(actual[row, start:end], width).indices
        expected_values, expected_idx = torch.topk(expected[row, start:end], width)
        if set(actual_idx.tolist()) == set(expected_idx.tolist()):
            exact_rows += 1
            continue
        cutoff = expected_values[-1]
        cutoff_ties = (expected[row, start:end] - cutoff).abs() <= 2 * max_error
        stable = set(
            torch.nonzero((expected[row, start:end] > cutoff) & ~cutoff_ties)
            .flatten()
            .tolist()
        )
        if stable.issubset(set(actual_idx.tolist())):
            tied_rows += 1
        else:
            unexplained_rows += 1
    return {
        "finite_values": int(error.numel()),
        "max_abs": float(max_error),
        "p50_abs": float(torch.quantile(error, 0.5)),
        "p99_abs": float(torch.quantile(error, 0.99)),
        "p999_abs": float(torch.quantile(error, 0.999)),
        "topk_exact_rows": exact_rows,
        "topk_cutoff_tie_rows": tied_rows,
        "topk_unexplained_rows": unexplained_rows,
    }


def _verify():
    from sglang.kernels.ops.attention.qsa.mqa import (
        triton_qsa_mqa_decode,
        triton_qsa_mqa_prefill,
    )
    from sglang.srt.layers.attention.qsa.mqa import (
        torch_qsa_mqa_decode,
        torch_qsa_mqa_prefill,
    )

    torch.manual_seed(301)
    lengths = torch.tensor([1, 127, 513, 2049, 8192], dtype=torch.int32)
    offsets = torch.cat([torch.zeros(1, dtype=torch.int32), lengths.cumsum(0)])
    sequence_ids = torch.tensor([0, 1, 1, 2, 3, 3, 4], dtype=torch.long)
    starts = offsets[:-1].index_select(0, sequence_ids).cuda()
    ends = offsets[1:].index_select(0, sequence_ids).cuda()
    q = torch.randn(
        sequence_ids.numel(), HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(int(offsets[-1]), 1, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    expected = torch_qsa_mqa_prefill(q, k, starts, ends)
    actual = triton_qsa_mqa_prefill(q, k, starts, ends)
    prefill = _score_stats(actual, expected, starts, ends)

    torch.manual_seed(302)
    decode_lengths = torch.tensor(
        [1, 513, 2047, 8193, 32768], device="cuda", dtype=torch.int32
    )
    max_len = int(decode_lengths.max())
    max_pages = (max_len + PAGE_SIZE - 1) // PAGE_SIZE
    batch = decode_lengths.numel()
    page_table = torch.arange(
        batch * max_pages, device="cuda", dtype=torch.int32
    ).reshape(batch, max_pages)
    cache = torch.randn(
        batch * max_pages,
        PAGE_SIZE,
        1,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q = torch.randn(batch, HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q = torch.cat([q, torch.zeros_like(q)], dim=1)
    case = (q, cache, page_table, decode_lengths, max_len)
    expected = torch_qsa_mqa_decode(*case)
    actual = triton_qsa_mqa_decode(*case)
    decode = _score_stats(
        actual, expected, torch.zeros_like(decode_lengths), decode_lengths
    )
    return {"ragged_prefill": prefill, "ragged_decode": decode}


def _run_backend(args):
    prefill, decode = _imports()[args.backend]
    results = []
    for batch in (1, 16, 64):
        case = _decode_case(batch, args.backend)
        latency = _time_us(
            lambda case=case: decode(*case), args.warmup, args.decode_iterations
        )
        results.append(
            {
                "point": f"decode-b{batch}",
                "backend": args.backend,
                "source_tokens": DECODE_SOURCE_TOKENS,
                "query_rows": batch,
                "compressed_keys": DECODE_SOURCE_TOKENS // COMPRESS_RATIO,
                "latency_us": latency,
            }
        )
        del case
        torch.cuda.empty_cache()
    for source_tokens in (8192, 32768):
        case = _prefill_case(source_tokens)
        latency = _time_us(
            lambda case=case: prefill(*case), args.warmup, args.prefill_iterations
        )
        results.append(
            {
                "point": f"prefill-{source_tokens // 1024}k",
                "backend": args.backend,
                "source_tokens": source_tokens,
                "query_rows": source_tokens,
                "compressed_keys": source_tokens // COMPRESS_RATIO,
                "latency_us": latency,
            }
        )
        del case
        torch.cuda.empty_cache()
    print(json.dumps(results))


def _environment_line():
    import triton

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()[0]
    return (
        f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
        f"driver={driver} cuda={torch.version.cuda} torch={torch.__version__} "
        f"triton={triton.__version__} sglang_commit={commit} model=Qwen3.8-Flash-Next "
        "quant=FP8/NVFP4-indexer-BF16 heads=4 head_dim=128 compress_ratio=4 "
        "page_size=16 decode_source_tokens=32768"
    )


def _run_all(args):
    results = []
    failures = {}
    for backend in ("torch", "triton", "tilelang"):
        env = os.environ.copy()
        env["SGLANG_QSA_MQA_BACKEND"] = backend
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--backend",
            backend,
            "--warmup",
            str(args.warmup),
            "--decode-iterations",
            str(args.decode_iterations),
            "--prefill-iterations",
            str(args.prefill_iterations),
        ]
        try:
            completed = subprocess.run(
                command,
                env=env,
                check=True,
                text=True,
                capture_output=True,
                timeout=args.tilelang_timeout if backend == "tilelang" else None,
            )
            results.extend(json.loads(completed.stdout.strip().splitlines()[-1]))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            failures[backend] = str(exc)

    verification = _verify()
    print(_environment_line())
    print(
        "| point | rows | compressed keys | torch us | TileLang us | "
        "Triton us | Triton speedup |"
    )
    print("|:--|--:|--:|--:|--:|--:|--:|")
    by_point = {}
    for result in results:
        by_point.setdefault(result["point"], {})[result["backend"]] = result
    for point in ("decode-b1", "decode-b16", "decode-b64", "prefill-8k", "prefill-32k"):
        row = by_point.get(point, {})
        torch_us = row.get("torch", {}).get("latency_us")
        triton_us = row.get("triton", {}).get("latency_us")
        tilelang_us = row.get("tilelang", {}).get("latency_us")
        sample = next(iter(row.values()), {})
        speedup = torch_us / triton_us if torch_us and triton_us else None
        value = lambda item: "n/a" if item is None else f"{item:.1f}"
        print(
            f"| {point} | {sample.get('query_rows', 'n/a')} | "
            f"{sample.get('compressed_keys', 'n/a')} | {value(torch_us)} | "
            f"{value(tilelang_us)} | {value(triton_us)} | {value(speedup)}x |"
        )
    for backend, failure in failures.items():
        print(f"{backend}_status={failure}")
    print(
        "| verification | finite values | max abs | p50 abs | p99 abs | "
        "p99.9 abs | exact top-k rows | cutoff-tie rows | unexplained rows |"
    )
    print("|:--|--:|--:|--:|--:|--:|--:|--:|--:|")
    for name, stats in verification.items():
        print(
            f"| {name} | {stats['finite_values']} | {stats['max_abs']:.6g} | "
            f"{stats['p50_abs']:.6g} | {stats['p99_abs']:.6g} | "
            f"{stats['p999_abs']:.6g} | {stats['topk_exact_rows']} | "
            f"{stats['topk_cutoff_tie_rows']} | {stats['topk_unexplained_rows']} |"
        )
    if args.json:
        print(
            json.dumps(
                {
                    "environment": _environment_line(),
                    "results": results,
                    "failures": failures,
                    "verification": verification,
                }
            )
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("all", "torch", "triton", "tilelang"),
        default="all",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--decode-iterations", type=int, default=20)
    parser.add_argument("--prefill-iterations", type=int, default=3)
    parser.add_argument("--tilelang-timeout", type=int, default=600)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.backend == "all":
        _run_all(args)
    else:
        _run_backend(args)


if __name__ == "__main__":
    if not is_sm120_supported():
        print("[skip] QSA MQA benchmark requires SM120 CUDA.")
        sys.exit(0)
    main()
