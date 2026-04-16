"""
Micro-benchmark: PyTorch ``normal_decode_set_metadata`` vs Triton
``fused_normal_decode_set_metadata``.

Run from repo root after editable install::

    cd python && pip install -e .
    cd .. && python benchmark/kernels/attention/bench_fused_decode_metadata.py

Or with PYTHONPATH::

    PYTHONPATH=python python benchmark/kernels/attention/bench_fused_decode_metadata.py

Colab: clone your branch, then ``pip install -e /content/sglang/python`` and run the same.
"""

from __future__ import annotations

import argparse
import os

import torch
import triton

from sglang.srt.layers.attention.fused_decode_metadata import (
    fused_normal_decode_set_metadata,
)


def _reference_normal_decode_set_metadata(
    cache_seqlens_int32: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    strided_indices: torch.Tensor,
    max_seq_pages: int,
    seq_lens: torch.Tensor,
    seq_len_delta: int,
    page_size: int,
) -> None:
    """Same logic as ``flashattention_backend.normal_decode_set_metadata`` (no SWA)."""
    cache_seqlens_int32.copy_(seq_lens + seq_len_delta)
    cu_seqlens_k[1:].copy_(
        torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32)
    )
    page_indices = req_to_token[
        req_pool_indices[:, None],
        strided_indices[:max_seq_pages][None, :],
    ]
    page_table[:, :max_seq_pages].copy_(page_indices // page_size)


def _make_case(
    *,
    bs: int,
    max_batch: int,
    max_context_len: int,
    page_size: int,
    seq_len_delta: int,
    device: torch.device,
):
    req_to_token = torch.randint(
        0,
        max_batch * max_context_len,
        (max_batch, max_context_len),
        dtype=torch.int64,
        device=device,
    )
    req_pool_indices = torch.randperm(max_batch, device=device)[:bs].to(torch.int64)

    upper = max(max_context_len // 2, 2)
    seq_lens = torch.randint(1, upper, (bs,), dtype=torch.int32, device=device)

    strided_indices = torch.arange(0, max_context_len, page_size, device=device)
    max_num_pages = (max_context_len + page_size - 1) // page_size
    max_len = int(seq_lens.max().item()) + seq_len_delta
    max_seq_pages = (max_len + page_size - 1) // page_size

    return {
        "req_to_token": req_to_token,
        "req_pool_indices": req_pool_indices,
        "seq_lens": seq_lens,
        "strided_indices": strided_indices,
        "max_num_pages": max_num_pages,
        "max_seq_pages": max_seq_pages,
    }


def _assert_close(case, *, page_size: int, seq_len_delta: int, device: torch.device):
    bs = case["seq_lens"].shape[0]
    inp = case
    max_seq_pages = inp["max_seq_pages"]

    ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
    ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    ref_pt = torch.zeros(bs, inp["max_num_pages"], dtype=torch.int32, device=device)
    fused_cache = torch.zeros_like(ref_cache)
    fused_cu = torch.zeros_like(ref_cu)
    fused_pt = torch.zeros_like(ref_pt)

    _reference_normal_decode_set_metadata(
        ref_cache,
        ref_cu,
        ref_pt,
        inp["req_to_token"],
        inp["req_pool_indices"],
        inp["strided_indices"],
        max_seq_pages,
        inp["seq_lens"],
        seq_len_delta,
        page_size,
    )
    fused_normal_decode_set_metadata(
        fused_cache,
        fused_cu,
        fused_pt,
        inp["req_to_token"],
        inp["req_pool_indices"],
        inp["strided_indices"],
        max_seq_pages,
        inp["seq_lens"],
        seq_len_delta,
        page_size,
    )
    torch.testing.assert_close(ref_cache, fused_cache)
    torch.testing.assert_close(ref_cu, fused_cu)
    torch.testing.assert_close(ref_pt, fused_pt)


def _warmup_fused(case, *, page_size: int, seq_len_delta: int, device: torch.device):
    """Autotune benchmarks on first page_size; warm up before ``do_bench``."""
    bs = case["seq_lens"].shape[0]
    inp = case
    max_seq_pages = inp["max_seq_pages"]
    for _ in range(5):
        cache = torch.zeros(bs, dtype=torch.int32, device=device)
        cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        pt = torch.zeros(bs, inp["max_num_pages"], dtype=torch.int32, device=device)
        fused_normal_decode_set_metadata(
            cache,
            cu,
            pt,
            inp["req_to_token"],
            inp["req_pool_indices"],
            inp["strided_indices"],
            max_seq_pages,
            inp["seq_lens"],
            seq_len_delta,
            page_size,
        )
    torch.cuda.synchronize()


def run_correctness(device: torch.device) -> None:
    for page_size in (1, 16, 64):
        case = _make_case(
            bs=128,
            max_batch=512,
            max_context_len=4096,
            page_size=page_size,
            seq_len_delta=1,
            device=device,
        )
        _assert_close(case, page_size=page_size, seq_len_delta=1, device=device)
    print("Correctness check passed.")


def run_benchmark(
    *,
    batch_sizes: list[int],
    max_batch: int,
    max_context_len: int,
    page_size: int,
    seq_len_delta: int,
    device: torch.device,
) -> None:
    quantiles = [0.5, 0.2, 0.8]
    print(
        f"max_batch={max_batch} max_context_len={max_context_len} "
        f"page_size={page_size} seq_len_delta={seq_len_delta}"
    )
    print(f"{'bs':>6}  {'pytorch_ms':>12}  {'fused_ms':>12}  {'speedup':>8}")

    for bs in batch_sizes:
        case = _make_case(
            bs=bs,
            max_batch=max_batch,
            max_context_len=max_context_len,
            page_size=page_size,
            seq_len_delta=seq_len_delta,
            device=device,
        )
        max_seq_pages = case["max_seq_pages"]
        max_num_pages = case["max_num_pages"]

        ref_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        ref_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        ref_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        fused_cache = torch.zeros(bs, dtype=torch.int32, device=device)
        fused_cu = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        fused_pt = torch.zeros(bs, max_num_pages, dtype=torch.int32, device=device)

        def run_ref():
            _reference_normal_decode_set_metadata(
                ref_cache,
                ref_cu,
                ref_pt,
                case["req_to_token"],
                case["req_pool_indices"],
                case["strided_indices"],
                max_seq_pages,
                case["seq_lens"],
                seq_len_delta,
                page_size,
            )

        def run_fused():
            fused_normal_decode_set_metadata(
                fused_cache,
                fused_cu,
                fused_pt,
                case["req_to_token"],
                case["req_pool_indices"],
                case["strided_indices"],
                max_seq_pages,
                case["seq_lens"],
                seq_len_delta,
                page_size,
            )

        _warmup_fused(
            case, page_size=page_size, seq_len_delta=seq_len_delta, device=device
        )

        ms_ref, _, _ = triton.testing.do_bench(run_ref, quantiles=quantiles)
        ms_fused, _, _ = triton.testing.do_bench(run_fused, quantiles=quantiles)
        speedup = ms_ref / ms_fused if ms_fused > 0 else float("inf")
        print(
            f"{bs:6d}  {1000 * ms_ref:12.4f}  {1000 * ms_fused:12.4f}  {speedup:8.2f}x"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="",
        help="If set, mkdir and print path (optional; for parity with other bench scripts).",
    )
    parser.add_argument(
        "--max_batch", type=int, default=2048, help="req_to_token first dim cap"
    )
    parser.add_argument(
        "--max_context_len", type=int, default=16384, help="req_to_token second dim"
    )
    parser.add_argument("--page_size", type=int, default=16)
    parser.add_argument("--seq_len_delta", type=int, default=0)
    parser.add_argument(
        "--batch_sizes",
        type=str,
        default="1,2,4,8,16,32,64,128,256,512,1024",
        help="Comma-separated batch sizes",
    )
    parser.add_argument("--skip_correctness", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    if args.save_path:
        os.makedirs(args.save_path, exist_ok=True)
        print(f"save_path: {args.save_path}")

    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    if not args.skip_correctness:
        run_correctness(device)

    run_benchmark(
        batch_sizes=batch_sizes,
        max_batch=args.max_batch,
        max_context_len=args.max_context_len,
        page_size=args.page_size,
        seq_len_delta=args.seq_len_delta,
        device=device,
    )


if __name__ == "__main__":
    main()
