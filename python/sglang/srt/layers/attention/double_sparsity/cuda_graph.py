"""CUDA-graph replay-stable buffers for the Double Sparsity decode path.

The DS selection pipeline must be replay-stable: every shape, every device-side
branch, and every allocation must be deterministic across capture and replay.
:class:`DSGraphState` owns the pre-allocated output buffers (``selected_indices``,
``valid_lengths``) and all scoring / radix-top-k scratch, sized to the worst-case
batch before capture. The captured region then performs zero new allocations and
contains no host-reading ``if`` statements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DSGraphState:
    """Static buffers + scratch that survive CUDA graph capture / replay."""

    selected_indices: torch.Tensor  # int32 [max_bs, max_top_k], -1 padded
    valid_lengths: torch.Tensor  # int32 [max_bs]
    max_seq_len: int = 0  # static sequence width for graph-safe logical scoring

    # Allocation-free graph-safe path scratch (sized at allocate_graph_state time
    # when max_seq_len > 0). The paged absorbed kernel scores into scratch_scores
    # in place; the radix selector reads it (or its bf16 reduce view) directly.
    scratch_scores: Optional[torch.Tensor] = None  # fp32 [max_bs, max_seq_len]
    scratch_pv_mask: Optional[torch.Tensor] = None  # bool [max_bs, max_seq_len]
    # bf16 transport scratch for the cross-TP score reduce (score_reduce_dtype
    # == "bf16"): halves the reduce bytes over the static score width.
    scratch_scores_bf16: Optional[torch.Tensor] = None  # bf16 [max_bs, max_seq_len]
    # Absorbed-latent selection scratch: the query latent projection v_h, the
    # weighted channel-gathered query, the int64 layer mask (so the gather does no
    # per-step .long()), the fp32-cast query, and the per-(batch,head) query norm
    # for the cosine denominator. All built in place — no per-step allocation.
    scratch_absorbed_v: Optional[torch.Tensor] = None  # fp32 [max_bs, H, kv_lora_rank]
    scratch_absorbed_qsel: Optional[torch.Tensor] = None  # fp32 [max_bs, H, label_dim]
    scratch_absorbed_sel_i64: Optional[torch.Tensor] = None  # int64 [H, label_dim]
    scratch_absorbed_q: Optional[torch.Tensor] = None  # fp32 [max_bs, H, nope_dim]
    scratch_qnorm: Optional[torch.Tensor] = None  # fp32 [max_bs, H]
    # Scratch bundle for the sequence-aware deterministic radix top-k
    # (topk_kernel.select_topk_sequence_order_triton).
    scratch_topk_hist: Optional[torch.Tensor] = None  # int32 [max_bs, 256]
    scratch_topk_key_prefix: Optional[torch.Tensor] = None  # int64 [max_bs]
    scratch_topk_quota: Optional[torch.Tensor] = None  # int32 [max_bs]
    scratch_topk_block_above: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_block_tie: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_above_pref: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    scratch_topk_tie_pref: Optional[torch.Tensor] = None  # int32 [max_bs, nblocks]
    topk_block: int = 1024
    # Production input scratch — `forward_batch.req_pool_indices` is int64 but the
    # captured selector region requires int32. `_select_topk_indices` does an
    # in-place copy_() into these views before calling retrieve_topk_graph_safe.
    scratch_req_pool_indices: Optional[torch.Tensor] = None  # int32 [max_bs]
    scratch_seq_lens: Optional[torch.Tensor] = None  # int32 [max_bs]


def allocate_graph_state(
    *,
    max_bs: int,
    max_top_k: int,
    max_seq_len: int = 0,
    num_local_heads: int = 0,
    label_dim: int = 0,
    score_reduce_bf16: bool = False,
    topk_block: int = 1024,
    kv_lora_rank: int = 0,
    qk_nope_head_dim: int = 0,
    device: Optional[torch.device] = None,
) -> DSGraphState:
    """Pre-allocate replay-stable buffers for the DS decode path.

    ``max_bs`` is the worst-case decode batch size. When ``max_seq_len > 0`` the
    score scratch, absorbed-latent scratch, and radix top-k bundle are allocated
    to the worst-case ``(max_bs, max_seq_len)`` / ``(max_bs, max_top_k)``.
    ``num_local_heads``/``label_dim`` size the absorbed scratch; ``kv_lora_rank``
    is the latent width; ``qk_nope_head_dim`` is the served query width.
    """
    if max_bs <= 0:
        raise ValueError(f"max_bs must be positive, got {max_bs}.")
    if max_top_k <= 0:
        raise ValueError(f"max_top_k must be positive, got {max_top_k}.")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = torch.full((max_bs, max_top_k), -1, dtype=torch.int32, device=device)
    valid = torch.zeros((max_bs,), dtype=torch.int32, device=device)

    scratch_scores = None
    scratch_pv_mask = None
    scratch_scores_bf16 = None
    scratch_absorbed_v = None
    scratch_absorbed_qsel = None
    scratch_absorbed_sel_i64 = None
    scratch_absorbed_q = None
    scratch_qnorm = None
    scratch_req_pool_indices = None
    scratch_seq_lens = None
    scratch_topk_hist = None
    scratch_topk_key_prefix = None
    scratch_topk_quota = None
    scratch_topk_block_above = None
    scratch_topk_block_tie = None
    scratch_topk_above_pref = None
    scratch_topk_tie_pref = None
    if max_seq_len > 0:
        scratch_scores = torch.zeros(
            (max_bs, max_seq_len), dtype=torch.float32, device=device
        )
        scratch_pv_mask = torch.zeros(
            (max_bs, max_seq_len), dtype=torch.bool, device=device
        )
        scratch_req_pool_indices = torch.zeros(
            (max_bs,), dtype=torch.int32, device=device
        )
        scratch_seq_lens = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        if score_reduce_bf16:
            scratch_scores_bf16 = torch.zeros(
                (max_bs, max_seq_len), dtype=torch.bfloat16, device=device
            )
        if num_local_heads > 0 and kv_lora_rank > 0 and label_dim > 0:
            scratch_absorbed_v = torch.zeros(
                (max_bs, num_local_heads, kv_lora_rank),
                dtype=torch.float32,
                device=device,
            )
            scratch_absorbed_qsel = torch.zeros(
                (max_bs, num_local_heads, label_dim),
                dtype=torch.float32,
                device=device,
            )
            scratch_absorbed_sel_i64 = torch.zeros(
                (num_local_heads, label_dim), dtype=torch.int64, device=device
            )
            # The served query is bf16/fp16 [bs, H, qk_nope_head_dim]; cast it into
            # this fp32 scratch in place so the v_h build never allocates. Fall back
            # to label_dim width when qk_nope_head_dim is absent (CPU unit fixtures).
            _q_width = qk_nope_head_dim if qk_nope_head_dim > 0 else label_dim
            scratch_absorbed_q = torch.zeros(
                (max_bs, num_local_heads, _q_width),
                dtype=torch.float32,
                device=device,
            )
            scratch_qnorm = torch.zeros(
                (max_bs, num_local_heads), dtype=torch.float32, device=device
            )
        topk_nblocks = (max_seq_len + topk_block - 1) // topk_block
        scratch_topk_hist = torch.zeros((max_bs, 256), dtype=torch.int32, device=device)
        scratch_topk_key_prefix = torch.zeros(
            (max_bs,), dtype=torch.int64, device=device
        )
        scratch_topk_quota = torch.zeros((max_bs,), dtype=torch.int32, device=device)
        scratch_topk_block_above = torch.zeros(
            (max_bs, topk_nblocks), dtype=torch.int32, device=device
        )
        scratch_topk_block_tie = torch.zeros(
            (max_bs, topk_nblocks), dtype=torch.int32, device=device
        )
        scratch_topk_above_pref = torch.zeros(
            (max_bs, topk_nblocks), dtype=torch.int32, device=device
        )
        scratch_topk_tie_pref = torch.zeros(
            (max_bs, topk_nblocks), dtype=torch.int32, device=device
        )

    return DSGraphState(
        selected_indices=selected,
        valid_lengths=valid,
        max_seq_len=max_seq_len,
        scratch_scores=scratch_scores,
        scratch_pv_mask=scratch_pv_mask,
        scratch_scores_bf16=scratch_scores_bf16,
        scratch_absorbed_v=scratch_absorbed_v,
        scratch_absorbed_qsel=scratch_absorbed_qsel,
        scratch_absorbed_sel_i64=scratch_absorbed_sel_i64,
        scratch_absorbed_q=scratch_absorbed_q,
        scratch_qnorm=scratch_qnorm,
        scratch_topk_hist=scratch_topk_hist,
        scratch_topk_key_prefix=scratch_topk_key_prefix,
        scratch_topk_quota=scratch_topk_quota,
        scratch_topk_block_above=scratch_topk_block_above,
        scratch_topk_block_tie=scratch_topk_block_tie,
        scratch_topk_above_pref=scratch_topk_above_pref,
        scratch_topk_tie_pref=scratch_topk_tie_pref,
        topk_block=topk_block,
        scratch_req_pool_indices=scratch_req_pool_indices,
        scratch_seq_lens=scratch_seq_lens,
    )


def radix_topk_scratch(state: Optional[DSGraphState]) -> Optional[dict]:
    """The radix top-k scratch bundle of a graph state, as kwargs for
    ``topk_kernel.select_topk_sequence_order_triton`` — or None when the state
    has no bundle (CPU / no-scratch unit-test path)."""
    if state is None or state.scratch_topk_hist is None:
        return None
    return {
        "scratch_hist": state.scratch_topk_hist,
        "scratch_key_prefix": state.scratch_topk_key_prefix,
        "scratch_quota": state.scratch_topk_quota,
        "scratch_block_above": state.scratch_topk_block_above,
        "scratch_block_tie": state.scratch_topk_block_tie,
        "scratch_above_pref": state.scratch_topk_above_pref,
        "scratch_tie_pref": state.scratch_topk_tie_pref,
    }
