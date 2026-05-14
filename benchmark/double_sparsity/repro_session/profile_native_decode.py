"""Profile the new native DS sparse-decode pipeline at 70B/TP=8 shapes.

Counterpart to `profile_ds_selection.py` (which profiles the legacy
3-stage selection + union path). This script times each phase of the
native path in isolation:

    score (Triton)
    torch.topk
    _build_selected_physical (torch)
    sparse attention stage2 (Triton)
    sparse attention stage3 (Triton)

80 iterations simulates one full decode step (1 layer x 80 layers). The
total per-decode-step number is the headline: subtract dense TBT (~8.5
ms at 32K from prior session) and the remainder must be ≤ a few ms for
DS-on to win.

Usage:
  PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/profile_native_decode.py
  CTX=131072 PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/profile_native_decode.py
"""

from __future__ import annotations

import argparse
import os

import torch

from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
    _build_selected_physical,
    _launch_score,
    _launch_sparse_attn,
    ds_native_sparse_decode,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--h-kv", type=int, default=1, help="local KV heads (TP=8 → 1)")
    p.add_argument("--h-q", type=int, default=8, help="local Q heads (TP=8 → 8)")
    p.add_argument("--d", type=int, default=128, help="head_dim")
    p.add_argument("--s", type=int, default=32, help="heavy channels")
    p.add_argument(
        "--ctx", type=int, default=int(os.environ.get("CTX", "30720")),
        help="seq_len for the synthetic prompt",
    )
    p.add_argument("--max-ctx", type=int, default=None, help="capacity")
    p.add_argument("--top-k", type=int, default=512)
    p.add_argument("--sink", type=int, default=4)
    p.add_argument("--recent", type=int, default=64)
    p.add_argument("--block-t", type=int, default=128)
    p.add_argument("--block-seq", type=int, default=128)
    p.add_argument("--block-n", type=int, default=16)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--layers", type=int, default=80, help="simulated decode-step depth")
    args = p.parse_args()
    max_ctx = args.max_ctx or args.ctx + 1024

    device = "cuda"
    torch.manual_seed(0)
    bs, h_kv, h_q, d, s = args.bs, args.h_kv, args.h_q, args.d, args.s
    total = args.top_k + args.sink + args.recent
    max_blocks = (total + args.block_seq - 1) // args.block_seq

    # Inputs
    T_pool = max_ctx
    q = torch.randn(bs, h_q, d, device=device, dtype=torch.bfloat16)
    q_label = torch.randn(bs, h_kv, s, device=device, dtype=torch.bfloat16)
    k_buffer = torch.randn(T_pool, h_kv, d, device=device, dtype=torch.bfloat16)
    v_buffer = torch.randn(T_pool, h_kv, d, device=device, dtype=torch.bfloat16)
    k_label = torch.randn(T_pool, h_kv, s, device=device, dtype=torch.bfloat16)
    req_to_token = (
        torch.arange(max_ctx, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(bs, max_ctx)
        .contiguous()
    )
    seq_lens = torch.full((bs,), args.ctx, device=device, dtype=torch.int64)

    # Scratch — [bs, H_kv, max_ctx] layout so torch.topk yields [bs, H_kv, top_k].
    att_out = torch.full(
        (bs, h_kv, max_ctx), float("-inf"), dtype=torch.float32, device=device
    )
    sel_phys = torch.zeros((bs, h_kv, total), dtype=torch.int32, device=device)
    mid_out = torch.zeros((bs, h_q, max_blocks, d), dtype=torch.float32, device=device)
    mid_log = torch.full(
        (bs, h_q, max_blocks), float("-inf"), dtype=torch.float32, device=device
    )
    output = torch.zeros(bs, h_q, d, dtype=torch.bfloat16, device=device)
    sm_scale = 1.0 / (d**0.5)

    def f_score():
        _launch_score(
            q_label=q_label,
            k_label_layer=k_label,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            att_out=att_out,
            sm_scale=sm_scale,
            sink_tokens=args.sink,
            recent_tokens=args.recent,
            block_t=args.block_t,
        )

    # `torch.topk` allocates the index tensor per call (CUB workspace + output).
    # That's part of the cost we want to measure.
    def f_topk():
        torch.topk(att_out, args.top_k, dim=-1, sorted=False)

    def f_build():
        topk_logical = torch.zeros(
            bs, h_kv, args.top_k, device=device, dtype=torch.int32
        )
        _build_selected_physical(
            topk_logical=topk_logical,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            sink_tokens=args.sink,
            recent_tokens=args.recent,
            out=sel_phys,
        )

    def f_attn():
        _launch_sparse_attn(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            selected_physical=sel_phys,
            mid_out=mid_out,
            mid_o_logexpsum=mid_log,
            output=output,
            sm_scale=sm_scale,
            block_seq=args.block_seq,
            block_n=args.block_n,
        )

    def bench(name, fn, warmup, iters):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        return ms

    print(
        f"Native DS sparse-decode profile  bs={bs} h_kv={h_kv} h_q={h_q} d={d} s={s}"
    )
    print(
        f"  ctx={args.ctx}  max_ctx={max_ctx}  top_k={args.top_k}  sink={args.sink}  "
        f"recent={args.recent}  total_selected={total}"
    )
    print(
        f"  block_t={args.block_t}  block_seq={args.block_seq}  block_n={args.block_n}"
    )
    print()
    print(f"  {'phase':38s} {'ms/call':>10s} {'µs/call':>10s} {'ms × {} layers'.format(args.layers):>22s}")
    print("  " + "-" * 80)
    # End-to-end: ds_native_sparse_decode in one call. This is the real
    # path cost (subset of per-op times because the topk output flows
    # into _build_selected_physical without re-allocation).
    def f_end_to_end():
        ds_native_sparse_decode(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            k_label_layer=k_label,
            q_label=q_label,
            req_to_token_indexed=req_to_token,
            seq_lens=seq_lens,
            top_k=args.top_k,
            sink_tokens=args.sink,
            recent_tokens=args.recent,
            sm_scale=sm_scale,
            att_out_approx=att_out,
            selected_physical=sel_phys,
            mid_out=mid_out,
            mid_o_logexpsum=mid_log,
            output=output,
            score_block_t=args.block_t,
            attn_block_seq=args.block_seq,
            attn_block_n=args.block_n,
        )

    t_score = bench("score (Triton)", f_score, args.warmup, args.iters)
    t_topk = bench("torch.topk", f_topk, args.warmup, args.iters)
    t_build = bench("build_selected_physical", f_build, args.warmup, args.iters)
    t_attn = bench("sparse attn stage2+3 (Triton)", f_attn, args.warmup, args.iters)
    t_e2e = bench("END-TO-END ds_native_sparse_decode", f_end_to_end, args.warmup, args.iters)
    for name, t in [
        ("score (Triton)", t_score),
        ("torch.topk", t_topk),
        ("build_selected_physical", t_build),
        ("sparse attn stage2+3 (Triton)", t_attn),
        ("END-TO-END ds_native_sparse_decode", t_e2e),
    ]:
        print(f"  {name:38s} {t:10.3f} {t*1000:10.1f} {t*args.layers:18.2f}")
    total_ms = t_e2e * args.layers
    print()
    print(f"  END-TO-END per layer:        {t_e2e*1000:.1f} µs")
    print(f"  END-TO-END per decode step:  {total_ms:.2f} ms  ({args.layers} layers)")
    print()
    print(f"  Compare: legacy DS-on TBT at 32K was 100.14 ms (this branch, prior session)")
    print(f"           DS-off TBT at 32K was 8.52 ms")
    print(f"           Target for visible win: TBT(on) <= 0.90 * TBT(off) → ≤ 7.7 ms")


if __name__ == "__main__":
    main()
