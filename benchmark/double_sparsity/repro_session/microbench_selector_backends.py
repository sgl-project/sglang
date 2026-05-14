"""Microbench: torch vs FlashInfer selector backend per-call cost.

Times the `select(...)` call at the shapes that drive `try_native_sparse
_decode` on Llama-3.1-70B/TP=8/128K (h_kv=1 per TP rank, max_ctx=131072)
across several `(bs, top_k)` combinations. Per-step cost across 80
layers is reported so the absolute saving relative to TBT is directly
visible.

Usage:
  PYTHONPATH=python python3 \\
      benchmark/double_sparsity/repro_session/microbench_selector_backends.py
"""

from __future__ import annotations

import time

import torch

from sglang.srt.mem_cache.sparsity.algorithms.selector_backends import (
    FLASHINFER_TOPK_MAX,
    make_selector,
)


def _make_inputs(bs: int, h_kv: int, max_ctx: int, seq_len: int, top_k: int):
    device = torch.device("cuda:0")
    att_out = torch.randn(bs, h_kv, max_ctx, dtype=torch.float32, device=device)
    # Mask sink / recent / oob to -inf the way the score kernel does.
    att_out[..., :4] = float("-inf")
    att_out[..., seq_len - 64 : seq_len] = float("-inf")
    att_out[..., seq_len - 1 :] = float("-inf")
    req_to_token = (
        torch.arange(max_ctx, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(bs, max_ctx)
        .contiguous()
    )
    seq_lens = torch.full((bs,), seq_len, dtype=torch.int64, device=device)
    out = torch.zeros((bs, h_kv, top_k + 4 + 64), dtype=torch.int32, device=device)
    return att_out, req_to_token, seq_lens, out


def bench_backend(
    backend: str,
    bs: int,
    h_kv: int,
    max_ctx: int,
    seq_len: int,
    top_k: int,
    n_warmup: int = 30,
    n_iters: int = 200,
) -> float:
    att_out, r2t, seq_lens, out = _make_inputs(bs, h_kv, max_ctx, seq_len, top_k)
    sel = make_selector(backend)
    for _ in range(n_warmup):
        sel.select(
            att_out_approx=att_out,
            req_to_token_indexed=r2t,
            seq_lens=seq_lens,
            top_k=top_k,
            sink_tokens=4,
            recent_tokens=64,
            out=out,
        )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        sel.select(
            att_out_approx=att_out,
            req_to_token_indexed=r2t,
            seq_lens=seq_lens,
            top_k=top_k,
            sink_tokens=4,
            recent_tokens=64,
            out=out,
        )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e6  # µs/call


def main() -> None:
    max_ctx = 131072
    seq_len = 131000  # ~near-full history
    h_kv = 1  # Llama-3.1 70B with TP=8: 1 KV head per rank

    print(
        f"{'shape':>12}  {'torch (µs)':>10}  {'flashinfer (µs)':>15}  "
        f"{'speedup':>8}  {'per-step saving (80L) ms':>26}"
    )
    print("-" * 80)
    for bs in (1, 4, 8, 16, 32):
        for top_k in (1024, 2048, 4096, 8192):
            t_torch = bench_backend("torch", bs, h_kv, max_ctx, seq_len, top_k)
            if top_k > FLASHINFER_TOPK_MAX:
                fi_str = f"{'n/a (cap)':>15}"
                speedup_str = f"{'-':>8}"
                saving_str = f"{'':>26}"
            else:
                try:
                    t_fi = bench_backend(
                        "flashinfer_topk_page_table",
                        bs,
                        h_kv,
                        max_ctx,
                        seq_len,
                        top_k,
                    )
                    speedup = t_torch / t_fi
                    saving = (t_torch - t_fi) * 80 / 1000
                    fi_str = f"{t_fi:15.1f}"
                    speedup_str = f"{speedup:8.2f}x"
                    saving_str = f"{saving:26.2f}"
                except Exception as e:
                    fi_str = f"{'ERR':>15}"
                    speedup_str = f"{'-':>8}"
                    saving_str = f"  {str(e)[:23]:>24}"

            print(
                f"  bs={bs:>2} tb={top_k:>4}  {t_torch:10.1f}  {fi_str}  "
                f"{speedup_str}  {saving_str}"
            )


if __name__ == "__main__":
    main()
