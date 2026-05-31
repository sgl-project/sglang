#!/usr/bin/env python3
"""Verify the _logical_score_kernel early-exit is numerically identical + faster.

Correctness: with seq_len=4096, the selection (out_indices, out_lengths) must be
IDENTICAL whether the scan width is 4608 or the full 163840 — the early-exit only
skips blocks that were already masked to -inf. Perf: time the 61-layer selection
at width 163840 (was ~32 ms/step before the change).
"""
import time, torch
from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    retrieve_topk_graph_safe,
)

DEV = "cuda"
BS, L, H, LABEL_DIM, HEAD_DIM = 16, 61, 16, 16, 128
SEQ_LEN, MAX_TOP_K, T_PHYS = 4096, 2048, 8192

torch.manual_seed(0)
q = torch.randn(BS, H, HEAD_DIM, dtype=torch.float32, device=DEV)
ch_sel = torch.randint(0, HEAD_DIM, (L, H, LABEL_DIM), dtype=torch.int32, device=DEV)
ch_w = torch.randn(L, H, LABEL_DIM, dtype=torch.float32, device=DEV)
sig = torch.randint(-127, 127, (L, T_PHYS, H, LABEL_DIM), dtype=torch.int8, device=DEV)
scales = torch.rand(L, T_PHYS, H, dtype=torch.float16, device=DEV) * 0.1 + 0.01
written = torch.zeros(L, T_PHYS, dtype=torch.bool, device=DEV)
written[:, :SEQ_LEN] = True


def run(width, layer_id):
    rpi = torch.arange(BS, dtype=torch.int32, device=DEV)
    r2t = torch.arange(width, dtype=torch.int32, device=DEV).clamp_max(T_PHYS - 1)
    r2t = r2t.unsqueeze(0).repeat(BS, 1).contiguous()
    seq = torch.full((BS,), SEQ_LEN, dtype=torch.int32, device=DEV)
    out_idx = torch.full((BS, MAX_TOP_K), -1, dtype=torch.int32, device=DEV)
    out_len = torch.zeros(BS, dtype=torch.int32, device=DEV)
    sc = dict(
        scratch_scores=torch.zeros(BS, width, dtype=torch.float32, device=DEV),
        scratch_topk_values=torch.zeros(BS, MAX_TOP_K, dtype=torch.float32, device=DEV),
        scratch_topk_indices=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        scratch_invalid_mask=torch.zeros(BS, MAX_TOP_K, dtype=torch.bool, device=DEV),
        scratch_sorted_vals=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        scratch_boundary=torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        scratch_valid_i64=torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        scratch_throwaway_idx=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
    )
    retrieve_topk_graph_safe(
        queries=q, token_signatures=sig, written=written, channel_selection=ch_sel,
        channel_weights=ch_w, layer_id=layer_id, req_pool_indices=rpi, req_to_token=r2t,
        seq_lens=seq, max_seq_len=width, max_top_k=MAX_TOP_K, out_indices=out_idx,
        out_lengths=out_len, token_scales=scales, **sc)
    return out_idx.clone(), out_len.clone()

# Correctness: identical selection at tight vs full width, for several layers.
mism = 0
for lid in (0, 7, 30, 60):
    a_idx, a_len = run(4608, lid)
    b_idx, b_len = run(163840, lid)
    if not (torch.equal(a_idx, b_idx) and torch.equal(a_len, b_len)):
        mism += 1
        print(f"  layer {lid}: MISMATCH idx_equal={torch.equal(a_idx,b_idx)} len_equal={torch.equal(a_len,b_len)}")
    else:
        print(f"  layer {lid}: identical (valid_lengths[0]={int(a_len[0])})")

# Perf: time 61-layer selection at the full 163840 width with the early-exit.
def time_full(width, iters=30, warmup=8):
    rpi = torch.arange(BS, dtype=torch.int32, device=DEV)
    r2t = torch.arange(width, dtype=torch.int32, device=DEV).clamp_max(T_PHYS - 1).unsqueeze(0).repeat(BS, 1).contiguous()
    seq = torch.full((BS,), SEQ_LEN, dtype=torch.int32, device=DEV)
    out_idx = torch.full((BS, MAX_TOP_K), -1, dtype=torch.int32, device=DEV)
    out_len = torch.zeros(BS, dtype=torch.int32, device=DEV)
    sc = dict(
        scratch_scores=torch.zeros(BS, width, dtype=torch.float32, device=DEV),
        scratch_topk_values=torch.zeros(BS, MAX_TOP_K, dtype=torch.float32, device=DEV),
        scratch_topk_indices=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        scratch_invalid_mask=torch.zeros(BS, MAX_TOP_K, dtype=torch.bool, device=DEV),
        scratch_sorted_vals=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
        scratch_boundary=torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        scratch_valid_i64=torch.zeros(BS, 1, dtype=torch.int64, device=DEV),
        scratch_throwaway_idx=torch.zeros(BS, MAX_TOP_K, dtype=torch.int64, device=DEV),
    )
    def step():
        for lid in range(L):
            retrieve_topk_graph_safe(queries=q, token_signatures=sig, written=written,
                channel_selection=ch_sel, channel_weights=ch_w, layer_id=lid, req_pool_indices=rpi,
                req_to_token=r2t, seq_lens=seq, max_seq_len=width, max_top_k=MAX_TOP_K,
                out_indices=out_idx, out_lengths=out_len, token_scales=scales, **sc)
    for _ in range(warmup): step()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.time(); step(); torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    ts.sort(); return ts[len(ts)//2]

t = time_full(163840)
print(f"\nselection @ width=163840 (61 layers, bs=16, seq=4096), WITH early-exit: {t:.2f} ms/step")
print("CORRECTNESS:", "PASS (identical at all tested layers)" if mism == 0 else f"FAIL ({mism} mismatches)")
