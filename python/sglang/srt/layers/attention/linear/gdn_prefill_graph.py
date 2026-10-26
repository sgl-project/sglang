"""GDN prefill on static bucket tables, using the existing attention kernels."""

from __future__ import annotations

import torch
import triton

from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.kernels.ops.attention.linear.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
    fused_qkv_split_gdn_prefill,
)
from sglang.kernels.ops.mamba.causal_conv1d_triton import causal_conv1d_fn
from sglang.kernels.ops.mamba.kda_prefill_track import (
    kda_track_conv_window,
    kda_track_ssm_state,
)
from sglang.srt.layers.attention.linear.kda_prefill_graph import KDAPrefillGraphMetadata
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention

# Delta-rule chunk size of chunk_gated_delta_rule (fla/chunk.py).
GDN_CHUNK_SIZE = 64


def chunk_o_block_size(num_tokens: int) -> int:
    """chunk_fwd_o's chunk size for a varlen batch of ``num_tokens`` rows."""
    return min(GDN_CHUNK_SIZE, max(16, triton.next_power_of_2(num_tokens)))


def buckets_keep_chunk_o_block_size(capture_num_tokens: list[int]) -> bool:
    """Whether every batch replays with the chunk_fwd_o chunk size its eager
    break uses: the break runs on the real token count, the graph on the bucket."""
    buckets = sorted(capture_num_tokens)
    for n in range(1, GDN_CHUNK_SIZE + 1):
        bucket = next((b for b in buckets if b >= n), None)
        if bucket is not None and chunk_o_block_size(bucket) != chunk_o_block_size(n):
            return False
    return True


MAX_FUSED_QKV_SPLIT_DIM = 8192


def gdn_extend_in_graph(
    *,
    layer: RadixLinearAttention,
    metadata: KDAPrefillGraphMetadata,
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    if metadata.track:
        kda_track_conv_window(
            conv_pool=conv_states.transpose(-1, -2),
            x=mixed_qkv,
            track_mode=metadata.track_mode,
            track_conv_start=metadata.track_conv_start,
            track_dst=metadata.track_dst,
            num_tokens=metadata.num_logical_tokens,
        )
    else:
        pass
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        conv_states=conv_states,
        has_initial_state=metadata.has_initial_state,
        cache_indices=metadata.cache_indices,
        query_start_loc=metadata.cu_seqlens,
        seq_lens_cpu=None,
        block_table=metadata.conv_block_table,
    ).transpose(0, 1)
    if layer.q_dim + layer.k_dim + layer.v_dim <= MAX_FUSED_QKV_SPLIT_DIM:
        query, key, value = fused_qkv_split_gdn_prefill(
            mixed_qkv,
            layer.num_q_heads,
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_q_dim,
            layer.head_k_dim,
            layer.head_v_dim,
        )
    else:
        query, key, value = mixed_qkv.split(
            [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        query = query.view(1, metadata.num_tokens, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, metadata.num_tokens, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, metadata.num_tokens, layer.num_v_heads, layer.head_v_dim)
    g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
    snapshot = (
        torch.empty(
            (metadata.max_seqs, *ssm_states.shape[1:]),
            dtype=torch.float32,
            device=ssm_states.device,
        )
        if metadata.track
        else None
    )
    output, _, _ = chunk_gated_delta_rule(
        q=query,
        k=key,
        v=value,
        g=g,
        beta=beta,
        initial_state=ssm_states,
        initial_state_indices=metadata.cache_indices,
        cu_seqlens=metadata.cu_seqlens,
        use_qk_l2norm_in_kernel=True,
        track_state=snapshot,
        track_chunk_idx=metadata.track_chunk_idx if metadata.track else None,
        chunk_indices=metadata.chunk_indices,
        chunk_offsets=metadata.chunk_offsets,
    )
    if metadata.track:
        kda_track_ssm_state(
            ssm_pool=ssm_states,
            h_track=snapshot,
            track_mode=metadata.track_mode,
            track_dst=metadata.track_dst,
            cache_indices=metadata.cache_indices,
        )
    else:
        pass
    return output
