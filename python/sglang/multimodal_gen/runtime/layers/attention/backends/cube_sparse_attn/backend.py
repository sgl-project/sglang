# SPDX-License-Identifier: Apache-2.0
"""Cube sparse attention backend (correctness-first FlexAttention kernel).

The mask/metadata layer is kernel-agnostic (see ``mask.py``); the kernel call
is confined to ``_run_block_sparse_attention`` so faster block-sparse kernels
can be swapped in without touching mask semantics. A semantic cube label can
occupy multiple physical FlexAttention blocks when an embedded keyframe and
target frame share coordinates.
"""

import functools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.cube_sparse_attn.mask import (
    CubePrecomputed,
    PackedStreams,
    cube_topk_block_indices,
    precompute_cube_attention,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn import (
    FlashAttentionImpl,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


class CubeSparseAttentionBackend(AttentionBackend):

    @staticmethod
    def get_enum() -> AttentionBackendEnum:
        return AttentionBackendEnum.CUBE_SPARSE_ATTN

    @staticmethod
    def get_impl_cls() -> type["CubeSparseAttentionImpl"]:
        return CubeSparseAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["CubeSparseAttentionMetadata"]:
        return CubeSparseAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["CubeSparseAttentionMetadataBuilder"]:
        return CubeSparseAttentionMetadataBuilder


@dataclass
class CubeSparseAttentionMetadata(AttentionMetadata):
    # Per-denoise-step top-k keep ratios; indexed by current_timestep.
    topk_ratio_list: list[float]
    # precompute_cube_attention(...) result for the positive packed layout.
    precomputed: CubePrecomputed


class CubeSparseAttentionMetadataBuilder(AttentionMetadataBuilder):

    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(  # type: ignore[override]
        self,
        *,
        packed: dict[str, Any],
        local_cube_size: list[int] | tuple[int, ...],
        topk_ratio_list: list[float],
        num_steps: int,
        device: torch.device,
        **kwargs: dict[str, Any],
    ) -> CubeSparseAttentionMetadata:
        """Build cube metadata from a minimax_h3_packed_sequence(...) layout.

        The five per-stream token index tensors are derived from the packed
        dict; segment shapes come from its ``stream_layout`` entry.
        """
        topk_ratio_list = [float(ratio) for ratio in topk_ratio_list]
        if len(topk_ratio_list) != num_steps:
            raise ValueError(
                f"topk_ratio_list has {len(topk_ratio_list)} entries for "
                f"{num_steps} denoise steps"
            )
        for ratio in topk_ratio_list:
            if not 0.0 < ratio <= 1.0:
                raise ValueError(
                    f"topk_ratio_list entries must be in (0, 1], got {ratio}"
                )
        if "stream_layout" not in packed:
            raise ValueError(
                "packed layout has no stream_layout entry; cube sparse "
                "attention requires the packed_sequence stream_layout export"
            )
        layout = packed["stream_layout"]

        img_pos = packed["img_pos"].view(-1).to(torch.long)
        update_mask = packed["update_mask"].view(-1).to(torch.bool)
        audio_pos = packed["audio_pos"].view(-1).to(torch.long)
        if "audio_update_mask" in packed:
            audio_update_mask = packed["audio_update_mask"].view(-1).to(torch.bool)
        else:
            audio_update_mask = torch.ones(audio_pos.shape[0], dtype=torch.bool)
        text_index = packed["text_pos"].view(-1).to(torch.long)
        cond_image_index = img_pos[~update_mask]
        latent_index = img_pos[update_mask]
        cond_audio_index = audio_pos[~audio_update_mask]
        audio_index = audio_pos[audio_update_mask]

        seq_len = int(packed["seq_len"])
        used = int(packed["cu_seqlens"].view(-1)[1])
        sparse_ratios = [ratio for ratio in topk_ratio_list if ratio < 1.0]
        precomputed = precompute_cube_attention(
            [tuple(layout["target_shape"])],
            torch.tensor([0, used], dtype=torch.long),
            seq_len,
            tuple(local_cube_size),
            device,
            PackedStreams(
                text=text_index,
                cond_image=cond_image_index,
                latent=latent_index,
                cond_audio=cond_audio_index,
                audio=audio_index,
                cond_image_shapes=[tuple(layout["cond_image_shapes"])],
                cond_image_roles=[tuple(layout["cond_image_roles"])],
                cond_event_orders=[tuple(layout["cond_event_orders"])],
                cond_audio_stream_lens=[tuple(layout["cond_audio_stream_lens"])],
            ),
            packed["img_position_ids"],
            max(sparse_ratios, default=0.0),
        )
        precomputed.runtime.pad_score_mod = _make_pad_score_mod(
            precomputed.layout.is_real
        )
        return CubeSparseAttentionMetadata(
            current_timestep=0,
            topk_ratio_list=topk_ratio_list,
            precomputed=precomputed,
        )


@functools.cache
def _compiled_flex_attention():
    return torch.compile(flex_attention, mode="max-autotune-no-cudagraphs")


def _make_pad_score_mod(is_real: torch.Tensor):
    is_real_bool = is_real.to(torch.bool)

    def _pad_score_mod(score, b, h, q_idx, kv_idx):
        valid = is_real_bool[q_idx] & is_real_bool[kv_idx]
        return torch.where(valid, score, float("-inf"))

    return _pad_score_mod


def _run_block_sparse_attention(
    padded_q: torch.Tensor,
    padded_k: torch.Tensor,
    padded_v: torch.Tensor,
    block_layout: dict[str, torch.Tensor | None],
    precomputed: CubePrecomputed,
    softmax_scale: float,
) -> torch.Tensor:
    """Run block-sparse attention on the cube-padded layout.

    ``padded_q/k/v`` are ``[padded_seqlen, heads, dim]``. ``block_layout``
    contains physical KV rows produced directly by semantic TopK.
    Per-head sparse buffers are reused. Sparse steps must omit the full-KV
    tensors entirely rather than pass zero-count ones: on CUDA a present but
    empty full-KV pair still steers FlexAttention into its slower mixed-layout
    specialization, while omitting the pair selects the partial-only kernel.
    Single kernel entry point; swap here for faster kernels.
    """
    num_heads = padded_q.shape[1]

    def expand_heads(value):
        if value.shape[1] == num_heads:
            return value
        if value.shape[1] != 1:
            raise ValueError(
                f"cube BlockMask has {value.shape[1]} heads for {num_heads} Q heads"
            )
        return value.expand(1, num_heads, *value.shape[2:])

    kv_num_blocks = block_layout["kv_num_blocks"]
    kv_indices = block_layout["kv_indices"]
    if kv_num_blocks is None or kv_indices is None:
        raise ValueError("cube BlockMask requires compact KV block tensors")
    kv_num_blocks = expand_heads(kv_num_blocks)
    kv_indices = expand_heads(kv_indices)
    full_kv_num_blocks = block_layout.get("full_kv_num_blocks")
    full_kv_indices = block_layout.get("full_kv_indices")
    if (full_kv_num_blocks is None) != (full_kv_indices is None):
        raise ValueError(
            "cube BlockMask full_kv_num_blocks and full_kv_indices must "
            "both be present or both be omitted"
        )
    if full_kv_num_blocks is not None:
        full_kv_num_blocks = expand_heads(full_kv_num_blocks)
        full_kv_indices = expand_heads(full_kv_indices)

    block_mask = BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=precomputed.layout.cube_token_size,
        seq_lengths=(padded_q.shape[0], padded_k.shape[0]),
        compute_q_blocks=False,
    )
    out = _compiled_flex_attention()(
        padded_q.permute(1, 0, 2)[None],
        padded_k.permute(1, 0, 2)[None],
        padded_v.permute(1, 0, 2)[None],
        score_mod=precomputed.runtime.pad_score_mod,
        block_mask=block_mask,
        scale=softmax_scale,
    )
    return out[0].permute(1, 0, 2)


def cube_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: CubeSparseAttentionMetadata,
    softmax_scale: float,
) -> torch.Tensor:
    """Cube sparse attention over packed [total, heads, dim] q/k/v.

    Builds a per-step label top-k mask from pooled q/k, applies the
    cube-contiguous reorder and padding, runs the block-sparse kernel, then
    scatters the result back to packed-token order.
    Rows past real_total_len (packing pad) get zero output.
    """
    precomputed = attn_metadata.precomputed
    topk_ratio = attn_metadata.topk_ratio_list[attn_metadata.current_timestep]
    layout = precomputed.layout
    real_total_len = layout.real_total_len
    gather_idx = layout.gather_indices

    block_layout = cube_topk_block_indices(
        query[:real_total_len], key[:real_total_len], precomputed, topk_ratio
    )

    padded_q = query[:real_total_len].index_select(0, gather_idx)
    padded_k = key[:real_total_len].index_select(0, gather_idx)
    padded_v = value[:real_total_len].index_select(0, gather_idx)
    pad_indices = layout.pad_indices
    if pad_indices.numel() > 0:
        padded_q.index_fill_(0, pad_indices, 0)
        padded_k.index_fill_(0, pad_indices, 0)
        padded_v.index_fill_(0, pad_indices, 0)

    out = _run_block_sparse_attention(
        padded_q,
        padded_k,
        padded_v,
        block_layout,
        precomputed,
        softmax_scale,
    )

    output = out.index_select(0, layout.expand_indices)
    if real_total_len < query.shape[0]:
        output = F.pad(output, (0, 0, 0, 0, 0, query.shape[0] - real_total_len))
    return output


class CubeSparseAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        if causal:
            raise ValueError("cube sparse attention is non-causal only")
        self.softmax_scale = softmax_scale
        # Preserve H3's exact dense baseline on schedule entries that disable sparsity
        self._dense_impl = FlashAttentionImpl(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=prefix,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: CubeSparseAttentionMetadata,
    ) -> torch.Tensor:
        return cube_sparse_attention(
            query, key, value, attn_metadata, self.softmax_scale
        )

    def forward_varlen(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run the packed H3 layout carried by the active forward context."""

        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, CubeSparseAttentionMetadata):
            raise ValueError(
                "cube sparse attention requires CubeSparseAttentionMetadata "
                "in the active forward context"
            )
        if metadata.topk_ratio_list[metadata.current_timestep] == 1.0:
            return self._dense_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                cu_seqlens_host=cu_seqlens_host,
            )
        return cube_sparse_attention(query, key, value, metadata, self.softmax_scale)
