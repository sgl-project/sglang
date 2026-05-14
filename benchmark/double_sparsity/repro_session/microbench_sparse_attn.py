"""Sparse-attention microbench — verify the headline DS property.

Per the v2 plan §"Performance validation": "Run sparse attention
microbench with fixed selected counts 512, 1024, 2048 and seq_len 32K,
64K, 128K; attention time should be nearly constant across seq_len for
fixed selected count."

If the sparse attention kernel correctly bounds its runtime by
`total_selected` (not `seq_len`), the rows of the printed table should
be flat. A non-flat row would indicate the kernel is still doing
seq_len-bound work — e.g. iterating over all token positions, not just
the selected set.

Usage:
  PYTHONPATH=python python3 benchmark/double_sparsity/repro_session/microbench_sparse_attn.py
"""

from __future__ import annotations

import torch

from sglang.srt.layers.attention.triton_ops.double_sparsity_native_decode import (
    _launch_sparse_attn,
)


def bench(fn, warmup=20, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    device = "cuda"
    bs, h_q, h_kv, d = 1, 8, 1, 128
    sm_scale = 1.0 / (d**0.5)

    SEQS = [32 * 1024, 64 * 1024, 128 * 1024]
    SELECTEDS = [512, 1024, 2048]

    # Use enough pool slots for the largest seq.
    T_pool = max(SEQS) * 2
    k_buf = torch.randn(T_pool, h_kv, d, device=device, dtype=torch.bfloat16)
    v_buf = torch.randn(T_pool, h_kv, d, device=device, dtype=torch.bfloat16)
    q = torch.randn(bs, h_q, d, device=device, dtype=torch.bfloat16)

    print(
        "Sparse-attn microbench  bs=1 H_q=8 H_kv=1 d=128 (70B/TP=8 shape)"
    )
    print(
        "Headline: per-cell time should be ~constant across rows (i.e. attention "
        "is bounded by `total_selected`, not by seq_len). Per-row jitter > 5% "
        "would be suspicious."
    )
    print()
    header = "  selected\\seq_len " + "".join(f"{s//1024:>10d}K" for s in SEQS)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for total in SELECTEDS:
        # selected_physical: [bs, h_kv, total] pointing at random valid positions
        # in [0, seq_len). For the microbench, just pick the first `total` ids
        # in [0, seq_len) — sparse-attn cost depends only on count, not on
        # which positions are chosen (loads are O(total) regardless).
        cells = []
        for seq in SEQS:
            sel = (
                torch.randperm(seq, device=device, dtype=torch.int32)[:total]
                .reshape(bs, h_kv, total)
                .contiguous()
            )
            block_seq = 128
            num_blocks = (total + block_seq - 1) // block_seq
            mid_out = torch.zeros(
                bs, h_q, num_blocks, d, dtype=torch.float32, device=device
            )
            mid_log = torch.full(
                (bs, h_q, num_blocks),
                float("-inf"),
                dtype=torch.float32,
                device=device,
            )
            output = torch.zeros(bs, h_q, d, dtype=torch.bfloat16, device=device)

            def f():
                _launch_sparse_attn(
                    q=q, k_buffer=k_buf, v_buffer=v_buf,
                    selected_physical=sel,
                    mid_out=mid_out, mid_o_logexpsum=mid_log,
                    output=output, sm_scale=sm_scale,
                    block_seq=block_seq, block_n=16,
                )

            ms = bench(f)
            cells.append(ms)
        row = "  {:>16d}".format(total) + "".join(
            f"{c*1000:>9.1f}µs" for c in cells
        )
        print(row)
    print()
    print(
        "Interpretation: a flat row (e.g. 'selected=512: 35us, 35us, 36us') "
        "means the kernel correctly loads only selected K/V — the headline "
        "DS property holds. Rising values across the row would mean the "
        "kernel is doing seq_len-bound work."
    )


if __name__ == "__main__":
    main()
