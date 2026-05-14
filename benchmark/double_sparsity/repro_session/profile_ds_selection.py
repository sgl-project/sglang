"""Profile DS selection pipeline at 70B/TP=8 32K decode shape.

Synthesizes shapes that match the actual 32K DS-on bench and times each
phase via CUDA events so we measure pure GPU work (no server boot / no
prefill / no FA3). 80 calls simulates 80 transformer layers per decode step.
"""

import torch

from sglang.srt.mem_cache.sparsity.triton_ops.select_triton import (
    ds_select_stage1_block_topk,
    ds_select_stage2_merge,
    ds_union_per_batch,
)
from sglang.srt.mem_cache.sparsity.triton_ops.k_label_kernels import (
    ds_compute_k_label_write,
)

device = "cuda"
# Match the actual 32K bench DS-on config:
#   CTX=30720, server_ctx=32000, num_blocks ≈ 31-32 with block_t=1024
#   bs=1 (concurrency=1), H_kv_local=1 (8 KV heads / TP=8)
#   H_q_local=8 (64 Q heads / TP=8), head_dim=128, S=32 channels
#   token_budget=512, max_selected=8192, k_block=64
BS = 1
H_KV = 1
H_Q = 8
HEAD_DIM = 128
S = 32
NUM_TOKENS_POOL = 64 * 1024  # roomy
BLOCK_T = 1024
K_BLOCK = 64
NUM_BLOCKS_MAX = 32  # at 32K + padding
TOKEN_BUDGET = 512
EFFECTIVE_BUDGET = min(TOKEN_BUDGET, NUM_BLOCKS_MAX * K_BLOCK)
MAX_SELECTED = 8192
SINK_TOKENS = 4
RECENT_TOKENS = 64
MIN_SEQ_LEN = 4096
SEQ_LEN = 30720  # the bench prompt length

NUM_LAYERS = 80  # iterations per decode step

# Tensors matching DS algorithm preallocations
torch.manual_seed(0)
queries = torch.randn(BS, H_Q, HEAD_DIM, device=device, dtype=torch.bfloat16)
channel_idx = torch.arange(S, device=device, dtype=torch.int32).expand(H_KV, S).contiguous()
k_label_layer = torch.randn(NUM_TOKENS_POOL, H_KV, S, device=device, dtype=torch.bfloat16)
req_to_token = torch.arange(NUM_TOKENS_POOL, device=device, dtype=torch.int32).unsqueeze(0).expand(1, -1).contiguous()
req_pool_indices = torch.zeros(BS, device=device, dtype=torch.int32)
seq_lens = torch.full((BS,), SEQ_LEN, device=device, dtype=torch.int64)

block_topk_logical = torch.full((BS, H_KV, NUM_BLOCKS_MAX, K_BLOCK), -1, device=device, dtype=torch.int32)
block_topk_scores = torch.full((BS, H_KV, NUM_BLOCKS_MAX, K_BLOCK), float("-inf"), device=device, dtype=torch.float32)
merged_logical = torch.full((BS, H_KV, EFFECTIVE_BUDGET), -1, device=device, dtype=torch.int32)
merged_scores = torch.full((BS, H_KV, EFFECTIVE_BUDGET), float("-inf"), device=device, dtype=torch.float32)
selected_logical = torch.full((BS, MAX_SELECTED), -1, device=device, dtype=torch.int32)
valid_lengths = torch.zeros(BS, device=device, dtype=torch.int32)

# K_label write inputs (matching update_representations for decode)
k_new = torch.randn(BS, H_KV, HEAD_DIM, device=device, dtype=torch.bfloat16)
out_cache_loc = torch.arange(SEQ_LEN, SEQ_LEN + BS, device=device, dtype=torch.int64)
k_label_buf = torch.zeros(NUM_TOKENS_POOL, H_KV, S, device=device, dtype=torch.bfloat16)


def bench(name, fn, warmup=10, iters=200):
    """Time `fn` averaged over `iters` calls using CUDA events."""
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
    ms_total = start.elapsed_time(end)
    ms_per_call = ms_total / iters
    print(f"  {name:40s} {ms_per_call*1000:7.1f} µs/call   {ms_per_call*NUM_LAYERS:7.2f} ms/decode_step (80 layers)")
    return ms_per_call


def stage1():
    ds_select_stage1_block_topk(
        queries=queries,
        channel_idx=channel_idx,
        k_label=k_label_layer,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        num_kv_heads=H_KV,
        block_t=BLOCK_T,
        k_block=K_BLOCK,
        gqa_reduction_id=0,
        block_topk_logical=block_topk_logical,
        block_topk_scores=block_topk_scores,
    )


def stage2():
    ds_select_stage2_merge(
        block_topk_logical=block_topk_logical,
        block_topk_scores=block_topk_scores,
        effective_budget=EFFECTIVE_BUDGET,
        merged_logical=merged_logical,
        merged_scores=merged_scores,
    )


def union():
    ds_union_per_batch(
        merged_logical=merged_logical,
        merged_scores=merged_scores,
        seq_lens=seq_lens,
        sink_tokens=SINK_TOKENS,
        recent_tokens=RECENT_TOKENS,
        min_seq_len=MIN_SEQ_LEN,
        max_selected_per_request=MAX_SELECTED,
        selected_logical=selected_logical,
        valid_lengths=valid_lengths,
    )


def k_label_write():
    ds_compute_k_label_write(
        k=k_new,
        channel_idx=channel_idx,
        out_cache_loc=out_cache_loc,
        k_label=k_label_buf,
    )


print(f"Profiling DS selection pipeline @ 32K shape (bs={BS}, H_kv={H_KV}, S={S}, seq_len={SEQ_LEN})")
print(f"  block_t={BLOCK_T}, k_block={K_BLOCK}, num_blocks={NUM_BLOCKS_MAX}, eff_budget={EFFECTIVE_BUDGET}, max_sel={MAX_SELECTED}")
print()
print(f"  {'phase':40s} {'µs/call':>7s}             ms × 80 layers")
print("  " + "-" * 90)

t_k_label = bench("K_label write (Triton)", k_label_write)
t_stage1 = bench("Stage-1 block-topk (Triton)", stage1)
t_stage2 = bench("Stage-2 merge (Triton)", stage2)
t_union = bench("ds_union_per_batch (torch-on-CUDA)", union)

print()
total_per_layer_us = (t_k_label + t_stage1 + t_stage2 + t_union) * 1000
total_per_step_ms = total_per_layer_us * NUM_LAYERS / 1000
print(f"  TOTAL per layer: {total_per_layer_us:.1f} µs")
print(f"  TOTAL per decode step ({NUM_LAYERS} layers): {total_per_step_ms:.2f} ms")
print(f"  Observed DS-on TBT p50 = 100.14 ms   DS-off TBT p50 = 8.52 ms")
print(f"  DS-on overhead vs dense: {100.14 - 8.52:.1f} ms")
print()
print(f"  Union share of selection cost: {t_union / (t_k_label + t_stage1 + t_stage2 + t_union) * 100:.1f}%")
