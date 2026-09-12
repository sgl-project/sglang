from __future__ import annotations

import argparse
import statistics
import time
from typing import Callable

import torch

from sglang.kernels.ops.attention.minimax_sparse.prefill.sgl_native_q8kv8 import (
    sgl_native_q8kv8_sparse_prefill,
)
from sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse import (
    flash_prefill_with_gqa_share_sparse,
)


def make_case(total_q: int, seq_len: int, topk: int):
    device = "cuda"
    torch.manual_seed(20260826 + total_q + seq_len + topk)
    q_bf16 = (torch.randn(total_q, 4, 128, device=device) * 0.2).to(torch.bfloat16)
    k_cache = (torch.randn(seq_len, 1, 128, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )
    v_cache = (torch.randn(seq_len, 1, 128, device=device) * 0.2).to(
        torch.float8_e4m3fn
    )

    page_size = 128
    pages = seq_len // page_size
    page_perm = torch.randperm(pages, device=device)
    req_to_token = torch.empty(1, seq_len, dtype=torch.int32, device=device)
    offsets = torch.arange(page_size, device=device, dtype=torch.int32)
    for logical_page in range(pages):
        physical_page = page_perm[logical_page]
        req_to_token[0, logical_page * page_size : (logical_page + 1) * page_size] = (
            physical_page * page_size + offsets
        )

    slot_ids = torch.zeros(1, dtype=torch.int64, device=device)
    selected = torch.arange(pages - topk, pages, dtype=torch.int32, device=device)
    topk_idx = selected.view(1, 1, topk).expand(1, total_q, topk).contiguous()
    cu_seqlens = torch.tensor([0, total_q], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    prefix_lens = torch.tensor([seq_len - total_q], dtype=torch.int32, device=device)
    return (
        q_bf16,
        k_cache,
        v_cache,
        req_to_token,
        slot_ids,
        topk_idx,
        cu_seqlens,
        seq_lens,
        prefix_lens,
    )


def quant_q(q_bf16: torch.Tensor, q_scale: float | None) -> torch.Tensor:
    if q_scale is not None:
        q_bf16 = q_bf16 / q_scale
    return q_bf16.to(torch.float8_e4m3fn)


def measure(
    fn: Callable[[], torch.Tensor], warmup: int, repeats: int
) -> tuple[float, torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples), output


def diff_stats(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[float, float, float]:
    diff = (lhs.float() - rhs.float()).abs().flatten()
    return (
        diff.max().item(),
        diff.mean().item(),
        torch.quantile(diff, 0.99).item(),
    )


def run_shape(
    round_idx: int,
    total_q: int,
    seq_len: int,
    topk: int,
    warmup: int,
    repeats: int,
) -> None:
    args = make_case(total_q=total_q, seq_len=seq_len, topk=topk)
    q_bf16, k, v, req, slots, topk_idx, cu, seq, prefix = args
    q_fp8 = quant_q(q_bf16, q_scale=None)

    def triton_baseline():
        return flash_prefill_with_gqa_share_sparse(
            q=q_bf16,
            k_cache=k,
            v_cache=v,
            sink=None,
            req_to_token=req,
            slot_ids=slots,
            topk_idx=topk_idx,
            block_size_q=1,
            block_size_k=128,
            cu_seqlens=cu,
            seq_lens=seq,
            prefix_lens=prefix,
            max_seqlen_q=total_q,
        )

    def native_kernel_only():
        return sgl_native_q8kv8_sparse_prefill(
            q=q_fp8,
            k_cache=k,
            v_cache=v,
            req_to_token=req,
            slot_ids=slots,
            topk_idx=topk_idx,
            cu_seqlens=cu,
            seq_lens=seq,
            prefix_lens=prefix,
            block_size_k=128,
        )

    def native_with_q_quant():
        return sgl_native_q8kv8_sparse_prefill(
            q=quant_q(q_bf16, q_scale=None),
            k_cache=k,
            v_cache=v,
            req_to_token=req,
            slot_ids=slots,
            topk_idx=topk_idx,
            cu_seqlens=cu,
            seq_lens=seq,
            prefix_lens=prefix,
            block_size_k=128,
        )

    triton_ms, triton_out = measure(triton_baseline, warmup=warmup, repeats=repeats)
    native_ms, native_out = measure(native_kernel_only, warmup=warmup, repeats=repeats)
    native_with_quant_ms, native_with_quant_out = measure(
        native_with_q_quant, warmup=warmup, repeats=repeats
    )
    max_abs, mean_abs, p99_abs = diff_stats(native_out, triton_out)
    max_abs_q, mean_abs_q, p99_abs_q = diff_stats(native_with_quant_out, triton_out)

    print(
        "| "
        f"{round_idx} | {seq_len:,} | {topk} | {total_q} | "
        f"{triton_ms:.4f} | {native_ms:.4f} | "
        f"{triton_ms / native_ms:.3f}x | {(1.0 - native_ms / triton_ms) * 100.0:.1f}% | "
        f"{native_with_quant_ms:.4f} | {triton_ms / native_with_quant_ms:.3f}x | "
        f"{(1.0 - native_with_quant_ms / triton_ms) * 100.0:.1f}% | "
        f"{max_abs:.6f} | {mean_abs:.6f} | {p99_abs:.6f} | "
        f"{max_abs_q:.6f} | {mean_abs_q:.6f} | {p99_abs_q:.6f} |"
    )


def parse_csv_ints(value: str) -> list[int]:
    return [int(x) for x in value.split(",") if x]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", default="8192")
    parser.add_argument("--total-qs", default="32,128")
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()

    print(
        "| round | seq_len | topk | total_q | triton_bf16q_fp8kv_ms | "
        "native_q8kv8_kernel_ms | kernel_speedup | kernel_latency_reduction | "
        "native_with_q_quant_ms | with_quant_speedup | with_quant_latency_reduction | "
        "kernel_max_abs | kernel_mean_abs | kernel_p99_abs | "
        "with_quant_max_abs | with_quant_mean_abs | with_quant_p99_abs |"
    )
    print(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for round_idx in range(1, args.rounds + 1):
        for seq_len in parse_csv_ints(args.seq_lens):
            for total_q in parse_csv_ints(args.total_qs):
                run_shape(
                    round_idx=round_idx,
                    total_q=total_q,
                    seq_len=seq_len,
                    topk=args.topk,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )


if __name__ == "__main__":
    started = time.time()
    main()
    print(f"\nelapsed_s={time.time() - started:.2f}")
