# Copyright 2024 The Qwen Team and The Hugging Face Inc. team.
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""BAGEL's Qwen2 mixture-of-transformers model for image generation.

This module implements BAGEL's Apache-2.0 image-generation paths, including
request-local text planning. In particular, it does not copy
``modeling/bagel/modeling_utils.py`` because that file is CC BY-NC 4.0.
Timestep embeddings reuse SGLang's Apache-2.0 layer and the positional table
is loaded directly from the checkpoint.

Sources:
  - https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/bagel.py
  - https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/qwen2_navit.py
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.kernels.ops.layernorm.rmsnorm_hf import (
    can_use_rmsnorm_hf,
    is_supported_rmsnorm_hf_hidden_size,
    rmsnorm_hf,
)
from sglang.multimodal_gen.configs.models.dits.bagel import (
    BagelDiTArchConfig,
    BagelDiTConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_tp_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.layernorm import apply_qk_norm
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import TimestepEmbedder
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.models.dits.bagel_taylorseer import (
    BagelTaylorSeerContext,
    TaylorSeerState,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

logger = logging.getLogger(__name__)
_USE_FUSED_FP32_QK_NORM: bool | None = None
_AttentionStateKey = tuple[torch.device, torch.dtype, int, int, int, int]
_BAGEL_FLASH_ATTENTION_STATE: dict[_AttentionStateKey, bool] = {}
_BAGEL_FLASHINFER_ATTENTION_STATE: dict[_AttentionStateKey, bool] = {}


class _BagelColumnParallelLinear(ColumnParallelLinear):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states)[0]


class _BagelRowParallelLinear(RowParallelLinear):
    def forward(self, hidden_states: Tensor) -> Tensor:
        return super().forward(hidden_states)[0]


def _tp_size() -> int:
    """Return one before distributed initialization and the live TP size after it."""
    return get_tp_world_size() if model_parallel_is_initialized() else 1


def _column_parallel_linear(
    input_size: int,
    output_size: int,
    *,
    bias: bool,
    gather_output: bool = False,
) -> nn.Module:
    """Shard an output dimension only when the runtime initialized TP."""
    tp_size = _tp_size()
    if output_size % tp_size != 0:
        raise ValueError(
            f"BAGEL column dimension {output_size} must be divisible by TP size "
            f"{tp_size}"
        )
    if tp_size == 1:
        return nn.Linear(input_size, output_size, bias=bias)
    return _BagelColumnParallelLinear(
        input_size,
        output_size,
        bias=bias,
        gather_output=gather_output,
    )


def _row_parallel_linear(input_size: int, output_size: int, *, bias: bool) -> nn.Module:
    """Shard an input dimension and all-reduce its output when TP is active."""
    tp_size = _tp_size()
    if input_size % tp_size != 0:
        raise ValueError(
            f"BAGEL row dimension {input_size} must be divisible by TP size {tp_size}"
        )
    if tp_size == 1:
        return nn.Linear(input_size, output_size, bias=bias)
    return _BagelRowParallelLinear(
        input_size,
        output_size,
        bias=bias,
        input_is_parallel=True,
    )


def _vocab_parallel_embedding(vocab_size: int, hidden_size: int) -> nn.Module:
    """Shard the token table while keeping the TP=1 module unchanged."""
    if _tp_size() == 1:
        return nn.Embedding(vocab_size, hidden_size)
    return VocabParallelEmbedding(vocab_size, hidden_size)


@dataclass
class BagelKVCache:
    """Per-request, per-layer prefix key/value tensors."""

    key_cache: list[Tensor | None]
    value_cache: list[Tensor | None]

    @classmethod
    def empty(cls, num_layers: int) -> BagelKVCache:
        """Create an empty cache for ``num_layers`` transformer blocks."""
        return cls([None] * num_layers, [None] * num_layers)

    @property
    def sequence_length(self) -> int:
        first_key = self.key_cache[0] if self.key_cache else None
        return 0 if first_key is None else int(first_key.shape[0])

    @classmethod
    def concatenate(cls, caches: list[BagelKVCache]) -> BagelKVCache:
        """Pack request-local caches by concatenating them in request order.

        Args:
            caches: Per-request caches with an identical layer count.

        Returns:
            One cache whose tensors are concatenated in request order.

        Raises:
            ValueError: If no caches are provided or layer layouts differ.
        """
        if not caches:
            raise ValueError("BAGEL cache packing requires at least one request")
        num_layers = len(caches[0].key_cache)
        if any(
            len(cache.key_cache) != num_layers or len(cache.value_cache) != num_layers
            for cache in caches
        ):
            raise ValueError("BAGEL cache layer counts must match when batching")

        packed_keys: list[Tensor | None] = []
        packed_values: list[Tensor | None] = []
        for layer_index in range(num_layers):
            keys = [cache.key_cache[layer_index] for cache in caches]
            values = [cache.value_cache[layer_index] for cache in caches]
            if any(
                (key is None) != (value is None) for key, value in zip(keys, values)
            ):
                raise ValueError("BAGEL key/value cache entries must be paired")
            present_keys = [key for key in keys if key is not None]
            present_values = [value for value in values if value is not None]
            if present_keys and len(present_keys) != len(caches):
                raise ValueError(
                    "BAGEL cache layers must be present for every batched request"
                )
            packed_keys.append(
                None if not present_keys else torch.cat(present_keys, dim=0)
            )
            packed_values.append(
                None if not present_values else torch.cat(present_values, dim=0)
            )
        return cls(packed_keys, packed_values)


@dataclass(frozen=True)
class BagelPrefixContext:
    """One immutable view of a request-owned text prefix."""

    kv_cache: BagelKVCache
    kv_lens: Tensor
    rope_offset: int


@dataclass(frozen=True)
class BagelContext:
    """All request-local state consumed by BAGEL denoising.

    For T2I, ``conditional`` is text and ``unconditional`` is empty. For
    Editing, they are ``image + text`` and ``image-only`` respectively, while
    ``secondary_unconditional`` is the text-only baseline used by image CFG.
    Thinking uses ``system + user + thought``, ``system``, and
    ``system + user`` for those same three branches.
    """

    conditional_kv: BagelKVCache
    unconditional_kv: BagelKVCache
    conditional_kv_lens: Tensor
    unconditional_kv_lens: Tensor
    conditional_rope_offset: int | Tensor
    unconditional_rope_offset: int | Tensor
    height: int
    width: int
    start_of_image_token_id: int
    end_of_image_token_id: int
    secondary_unconditional_kv: BagelKVCache | None = None
    secondary_unconditional_kv_lens: Tensor | None = None
    secondary_unconditional_rope_offset: int | Tensor | None = None
    three_way_cfg_kind: str | None = None

    @property
    def has_three_way_cfg(self) -> bool:
        return self.secondary_unconditional_kv is not None

    @property
    def is_editing(self) -> bool:
        return self.three_way_cfg_kind == "editing"

    @property
    def is_thinking(self) -> bool:
        return self.three_way_cfg_kind == "thinking"

    @property
    def batch_size(self) -> int:
        return int(self.conditional_kv_lens.numel())

    @classmethod
    def from_prefixes(
        cls,
        conditional: BagelPrefixContext,
        unconditional: BagelPrefixContext,
        *,
        height: int,
        width: int,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
        secondary_unconditional: BagelPrefixContext | None = None,
        three_way_cfg_kind: str | None = None,
    ) -> BagelContext:
        """Build a denoising context from separately prepared prefixes."""
        if (secondary_unconditional is None) != (three_way_cfg_kind is None):
            raise ValueError(
                "BAGEL third CFG prefix and its semantic kind must be set together"
            )
        if three_way_cfg_kind not in {None, "editing", "thinking"}:
            raise ValueError(
                f"unsupported BAGEL three-way CFG kind: {three_way_cfg_kind}"
            )
        return cls(
            conditional_kv=conditional.kv_cache,
            unconditional_kv=unconditional.kv_cache,
            conditional_kv_lens=conditional.kv_lens,
            unconditional_kv_lens=unconditional.kv_lens,
            conditional_rope_offset=conditional.rope_offset,
            unconditional_rope_offset=unconditional.rope_offset,
            height=height,
            width=width,
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
            secondary_unconditional_kv=(
                None
                if secondary_unconditional is None
                else secondary_unconditional.kv_cache
            ),
            secondary_unconditional_kv_lens=(
                None
                if secondary_unconditional is None
                else secondary_unconditional.kv_lens
            ),
            secondary_unconditional_rope_offset=(
                None
                if secondary_unconditional is None
                else secondary_unconditional.rope_offset
            ),
            three_way_cfg_kind=three_way_cfg_kind,
        )

    @classmethod
    def pack(cls, contexts: list[BagelContext]) -> BagelContext:
        """Pack compatible two-way T2I contexts for one denoising forward.

        Args:
            contexts: Request-owned contexts in scheduler merge order.

        Returns:
            A context with concatenated KV tensors and per-request lengths and
            RoPE offsets.

        Raises:
            ValueError: If contexts are empty, already batched, three-way, or
                use different image geometry or special tokens.
        """
        if not contexts:
            raise ValueError("BAGEL context packing requires at least one request")
        reference = contexts[0]
        signature = (
            reference.height,
            reference.width,
            reference.start_of_image_token_id,
            reference.end_of_image_token_id,
        )
        for context in contexts:
            if context.batch_size != 1:
                raise ValueError("BAGEL can only pack request-local contexts")
            if context.has_three_way_cfg:
                raise ValueError("BAGEL dynamic batching supports pure T2I only")
            if (
                context.height,
                context.width,
                context.start_of_image_token_id,
                context.end_of_image_token_id,
            ) != signature:
                raise ValueError("BAGEL batched contexts must use identical geometry")

        def pack_branch(
            cache_name: str, lengths_name: str, offset_name: str
        ) -> tuple[BagelKVCache, Tensor, Tensor]:
            caches = [getattr(context, cache_name) for context in contexts]
            lengths = [getattr(context, lengths_name) for context in contexts]
            offsets = [getattr(context, offset_name) for context in contexts]
            for cache, branch_lengths, offset in zip(caches, lengths, offsets):
                if branch_lengths.numel() != 1:
                    raise ValueError("BAGEL request context must contain one KV length")
                if cache.sequence_length != int(branch_lengths.item()):
                    raise ValueError("BAGEL context KV length does not match its cache")
                if not isinstance(offset, int):
                    raise ValueError(
                        "BAGEL request context must contain one RoPE offset"
                    )
            packed_lengths = torch.cat(lengths).to(dtype=torch.int32)
            packed_offsets = packed_lengths.new_tensor(offsets, dtype=torch.long)
            return BagelKVCache.concatenate(caches), packed_lengths, packed_offsets

        conditional_kv, conditional_lens, conditional_offsets = pack_branch(
            "conditional_kv", "conditional_kv_lens", "conditional_rope_offset"
        )
        unconditional_kv, unconditional_lens, unconditional_offsets = pack_branch(
            "unconditional_kv", "unconditional_kv_lens", "unconditional_rope_offset"
        )
        return cls(
            conditional_kv=conditional_kv,
            unconditional_kv=unconditional_kv,
            conditional_kv_lens=conditional_lens,
            unconditional_kv_lens=unconditional_lens,
            conditional_rope_offset=conditional_offsets,
            unconditional_rope_offset=unconditional_offsets,
            height=reference.height,
            width=reference.width,
            start_of_image_token_id=reference.start_of_image_token_id,
            end_of_image_token_id=reference.end_of_image_token_id,
        )


class _LoadedPositionTable(nn.Module):
    """Checkpoint-owned 2D position table without generated initialization."""

    def __init__(self, rows: int, hidden_size: int) -> None:
        super().__init__()
        self.pos_embed = nn.Parameter(
            torch.empty(rows, hidden_size), requires_grad=False
        )

    def forward(self, position_ids: Tensor) -> Tensor:
        return F.embedding(position_ids, self.pos_embed)


class _TimestepMLP(nn.Module):
    """Single-rank MLP with SGLang TimestepEmbedder checkpoint names."""

    def __init__(self, frequency_size: int, hidden_size: int) -> None:
        super().__init__()
        self.fc_in = nn.Linear(frequency_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc_out(F.silu(self.fc_in(hidden_states)))


class _BagelTimestepEmbedder(TimestepEmbedder):
    """Reuse SGLang's timestep logic without requiring an initialized TP group."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int) -> None:
        nn.Module.__init__(self)
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = 10000
        self.mlp = _TimestepMLP(frequency_embedding_size, hidden_size)
        self.freq_dtype = torch.float32


class _Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rope_theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device="cpu")
                / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, hidden_states: Tensor, position_ids: Tensor
    ) -> tuple[Tensor, Tensor]:
        frequencies = torch.outer(
            position_ids.float(), self.inv_freq.to(position_ids.device)
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embeddings.cos().to(hidden_states.dtype),
            embeddings.sin().to(hidden_states.dtype),
        )


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: Tensor, key: Tensor, cosine: Tensor, sine: Tensor
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    return (
        query * cosine + _rotate_half(query) * sine,
        key * cosine + _rotate_half(key) * sine,
    )


def _sdpa_attention(query: Tensor, key: Tensor, value: Tensor, causal: bool) -> Tensor:
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    if num_query_heads != num_kv_heads:
        repeats = num_query_heads // num_kv_heads
        key = key.repeat_interleave(repeats, dim=1)
        value = value.repeat_interleave(repeats, dim=1)
    query = query.transpose(0, 1).unsqueeze(0)
    key = key.transpose(0, 1).unsqueeze(0)
    value = value.transpose(0, 1).unsqueeze(0)
    attention_mask = None
    if causal and query.shape[-2] != key.shape[-2]:
        # SDPA's built-in non-square causal mask is upper-left aligned. Prefix
        # decoding needs a bottom-right aligned mask so every new query can see
        # the complete immutable prefix plus preceding queries in this block.
        query_length = query.shape[-2]
        key_length = key.shape[-2]
        prefix_length = key_length - query_length
        query_positions = torch.arange(query_length, device=query.device).unsqueeze(1)
        key_positions = torch.arange(key_length, device=query.device).unsqueeze(0)
        attention_mask = key_positions <= prefix_length + query_positions
        causal = False
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=causal,
    )
    return output.squeeze(0).transpose(0, 1)


def _interleave_prefix_and_query(
    prefix: Tensor,
    query: Tensor,
    prefix_lens: Tensor,
    query_lens: Tensor,
) -> Tensor:
    """Pack ``[prefix_i, query_i]`` sequences in request order.

    Args:
        prefix: Concatenated request prefixes.
        query: Concatenated request query blocks.
        prefix_lens: Prefix length for each request.
        query_lens: Query length for each request.

    Returns:
        A varlen-attention key/value layout ordered as
        ``[prefix_0, query_0, prefix_1, query_1, ...]``.

    Raises:
        ValueError: If the length vectors or packed tensor sizes disagree.
    """
    prefix_sizes = [int(length) for length in prefix_lens.tolist()]
    query_sizes = [int(length) for length in query_lens.tolist()]
    if len(prefix_sizes) != len(query_sizes) or not prefix_sizes:
        raise ValueError("BAGEL prefix and query length vectors must align")
    if any(length < 0 for length in prefix_sizes) or any(
        length <= 0 for length in query_sizes
    ):
        raise ValueError(
            "BAGEL prefix lengths must be non-negative and queries positive"
        )
    if sum(prefix_sizes) != prefix.shape[0] or sum(query_sizes) != query.shape[0]:
        raise ValueError("BAGEL packed tensors do not match their length vectors")

    prefix_chunks = torch.split(prefix, prefix_sizes, dim=0)
    query_chunks = torch.split(query, query_sizes, dim=0)
    sequences = [
        torch.cat((prefix_chunk, query_chunk), dim=0)
        for prefix_chunk, query_chunk in zip(prefix_chunks, query_chunks)
    ]
    return torch.cat(sequences, dim=0)


def _can_use_bagel_cuda_attention(
    query: Tensor, attention_backend: AttentionBackendEnum
) -> bool:
    """Return whether BAGEL may use CUDA dense-attention kernels."""
    return (
        attention_backend == AttentionBackendEnum.FA
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
    )


def _bagel_flash_attention_version(query: Tensor) -> int:
    """Select FA4 on Blackwell-or-newer GPUs and FA3 otherwise."""
    major, _ = torch.cuda.get_device_capability(query.device)
    return 4 if major >= 10 else 3


def _attention_state_key(
    query: Tensor, key: Tensor, version: int
) -> _AttentionStateKey:
    """Build a process-local capability key for one attention configuration."""
    return (
        query.device,
        query.dtype,
        version,
        int(query.shape[1]),
        int(key.shape[1]),
        int(query.shape[-1]),
    )


def _run_bagel_flash_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_lens: Tensor,
    key_lens: Tensor,
    *,
    causal: bool,
    version: int,
) -> Tensor:
    """Run SGLang's packed FA3/FA4 implementation."""
    from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func

    cu_query = F.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
    cu_key = F.pad(torch.cumsum(key_lens, dim=0), (1, 0)).to(torch.int32)
    return flash_attn_varlen_func(
        query,
        key,
        value,
        cu_query,
        cu_key,
        max_seqlen_q=int(query_lens.max().item()),
        max_seqlen_k=int(key_lens.max().item()),
        causal=causal,
        ver=version,
    )


def _run_single_flashinfer_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    causal: bool,
) -> Tensor:
    """Run FlashInfer's NHD prefill kernel for one request."""
    from flashinfer import single_prefill_with_kv_cache

    return single_prefill_with_kv_cache(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        causal=causal,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
    )


def _run_flashinfer_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_sizes: list[int],
    key_sizes: list[int],
    *,
    causal: bool,
) -> Tensor:
    """Run FlashInfer per request so packed batches cannot attend across samples."""
    query_chunks = torch.split(query, query_sizes, dim=0)
    key_chunks = torch.split(key, key_sizes, dim=0)
    value_chunks = torch.split(value, key_sizes, dim=0)
    return torch.cat(
        [
            _run_single_flashinfer_attention(
                query_chunk,
                key_chunk,
                value_chunk,
                causal=causal,
            )
            for query_chunk, key_chunk, value_chunk in zip(
                query_chunks, key_chunks, value_chunks
            )
        ],
        dim=0,
    )


def _run_varlen_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_lens: Tensor,
    key_lens: Tensor,
    *,
    causal: bool,
    attention_backend: AttentionBackendEnum,
) -> Tensor:
    """Use FA, then FlashInfer on Blackwell, with a CPU-safe SDPA fallback."""
    query_sizes = [int(length) for length in query_lens.tolist()]
    key_sizes = [int(length) for length in key_lens.tolist()]
    if len(query_sizes) != len(key_sizes) or not query_sizes:
        raise ValueError("BAGEL query and key length vectors must align")
    if any(length <= 0 for length in query_sizes) or any(
        key_length < query_length
        for query_length, key_length in zip(query_sizes, key_sizes)
    ):
        raise ValueError("BAGEL varlen attention received invalid sequence lengths")
    if sum(query_sizes) != query.shape[0] or sum(key_sizes) != key.shape[0]:
        raise ValueError("BAGEL attention tensors do not match their length vectors")
    if value.shape[0] != key.shape[0]:
        raise ValueError("BAGEL attention key and value lengths must match")

    if _can_use_bagel_cuda_attention(query, attention_backend):
        version = _bagel_flash_attention_version(query)
        state_key = _attention_state_key(query, key, version)
        flash_attention_error: Exception | None = None
        flash_attention_state = _BAGEL_FLASH_ATTENTION_STATE.get(state_key)
        if flash_attention_state is not False:
            try:
                output = _run_bagel_flash_attention(
                    query,
                    key,
                    value,
                    query_lens,
                    key_lens,
                    causal=causal,
                    version=version,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (ImportError, NotImplementedError, RuntimeError, TypeError) as error:
                if flash_attention_state is True:
                    raise
                _BAGEL_FLASH_ATTENTION_STATE[state_key] = False
                flash_attention_error = error
            else:
                _BAGEL_FLASH_ATTENTION_STATE[state_key] = True
                return output

        # The old BAGEL integration used FlashInfer only as a Blackwell
        # fallback. Keep FA as the public backend while preserving that
        # resilience, and split packed batches to avoid cross-request attention.
        if (
            version == 4
            and _BAGEL_FLASHINFER_ATTENTION_STATE.get(state_key) is not False
        ):
            flashinfer_state = _BAGEL_FLASHINFER_ATTENTION_STATE.get(state_key)
            try:
                output = _run_flashinfer_varlen_attention(
                    query,
                    key,
                    value,
                    query_sizes,
                    key_sizes,
                    causal=causal,
                )
            except torch.cuda.OutOfMemoryError:
                raise
            except (ImportError, NotImplementedError, RuntimeError, TypeError) as error:
                if flashinfer_state is True:
                    raise
                _BAGEL_FLASHINFER_ATTENTION_STATE[state_key] = False
                logger.warning(
                    "BAGEL FlashAttention and FlashInfer unavailable; using SDPA "
                    "(FlashAttention: %s; FlashInfer: %s)",
                    flash_attention_error or "cached unavailable",
                    error,
                )
                flash_attention_error = None
            else:
                _BAGEL_FLASHINFER_ATTENTION_STATE[state_key] = True
                if flash_attention_error is not None:
                    logger.warning(
                        "BAGEL FlashAttention unavailable; using FlashInfer: %s",
                        flash_attention_error,
                    )
                elif flashinfer_state is None:
                    logger.info(
                        "BAGEL is using FlashInfer after a cached FlashAttention failure"
                    )
                return output

        if flash_attention_error is not None:
            logger.warning(
                "BAGEL FlashAttention unavailable; falling back to SDPA: %s",
                flash_attention_error,
            )

    if len(query_sizes) == 1:
        return _sdpa_attention(query, key, value, causal)
    query_chunks = torch.split(query, query_sizes, dim=0)
    key_chunks = torch.split(key, key_sizes, dim=0)
    value_chunks = torch.split(value, key_sizes, dim=0)
    return torch.cat(
        [
            _sdpa_attention(query_chunk, key_chunk, value_chunk, causal)
            for query_chunk, key_chunk, value_chunk in zip(
                query_chunks, key_chunks, value_chunks
            )
        ],
        dim=0,
    )


class _BagelRMSNorm(nn.Module):
    """Qwen2 RMSNorm with the official cast and multiply ordering."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        if _can_use_bagel_rmsnorm_jit(hidden_states, self.weight):
            hidden_size = hidden_states.shape[-1]
            original_shape = hidden_states.shape
            flat_hidden_states = hidden_states.contiguous().reshape(-1, hidden_size)
            return rmsnorm_hf(flat_hidden_states, self.weight, self.eps).reshape(
                original_shape
            )

        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)

    @property
    def variance_epsilon(self) -> float:
        """Expose the epsilon name used by SGLang's fused Q/K helper."""
        return self.eps


def _can_use_bagel_rmsnorm_jit(hidden_states: Tensor, weight: Tensor) -> bool:
    """Return whether BAGEL can preserve its HF RMSNorm semantics with JIT."""
    return (
        current_platform.is_cuda()
        and hidden_states.is_cuda
        and hidden_states.layout == torch.strided
        and hidden_states.dim() > 0
        and hidden_states.numel() > 0
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and weight.dtype == hidden_states.dtype
        and weight.device == hidden_states.device
        and weight.numel() == hidden_states.shape[-1]
        and is_supported_rmsnorm_hf_hidden_size(hidden_states.shape[-1])
        and can_use_rmsnorm_hf(hidden_states.shape[-1], hidden_states.dtype)
    )


def _can_use_fused_fp32_qk_norm(
    query: Tensor,
    key: Tensor,
    query_norm: _BagelRMSNorm,
    key_norm: _BagelRMSNorm,
) -> bool:
    """Return whether the Triton pair kernel preserves BAGEL FP32 semantics."""
    return (
        _USE_FUSED_FP32_QK_NORM is not False
        and current_platform.is_cuda()
        and query.is_cuda
        and query.dtype == torch.float32
        and key.dtype == query.dtype
        and query.device == key.device
        and query_norm.weight.device == query.device
        and key_norm.weight.device == key.device
        and query_norm.eps == key_norm.eps
        and query.shape[0] == key.shape[0]
        and query.shape[-1] == key.shape[-1]
        and query.numel() > 0
        and key.numel() > 0
        and query.is_contiguous()
        and key.is_contiguous()
    )


def _fused_fp32_qk_norm(
    query: Tensor,
    key: Tensor,
    query_norm: _BagelRMSNorm,
    key_norm: _BagelRMSNorm,
) -> tuple[Tensor, Tensor]:
    """Run the one-launch Triton Q/K RMSNorm used by FP32 generation."""
    from sglang.srt.layers.fused_qk_norm import fused_qk_norm

    return fused_qk_norm(
        query,
        key,
        query_norm.weight,
        key_norm.weight,
        query_norm.eps,
    )


def _apply_bagel_qk_norm(
    query: Tensor,
    key: Tensor,
    query_norm: _BagelRMSNorm,
    key_norm: _BagelRMSNorm,
    head_dim: int,
) -> tuple[Tensor, Tensor]:
    """Apply one-launch Q/K RMSNorm while preserving BAGEL cast ordering."""
    if query.shape[0] != key.shape[0]:
        raise ValueError("BAGEL Q/K normalization requires matching token counts")
    if query.shape[-1] != head_dim or key.shape[-1] != head_dim:
        raise ValueError(
            f"BAGEL Q/K head dimensions must both equal {head_dim}, got "
            f"{query.shape[-1]} and {key.shape[-1]}"
        )
    if _can_use_fused_fp32_qk_norm(query, key, query_norm, key_norm):
        global _USE_FUSED_FP32_QK_NORM
        try:
            output = _fused_fp32_qk_norm(query, key, query_norm, key_norm)
        except torch.cuda.OutOfMemoryError:
            raise
        except (ImportError, NotImplementedError, RuntimeError, TypeError) as error:
            # Backend import/JIT failures are stable for this process. Disable
            # the optional optimization after its first failed probe, but do
            # not hide a later runtime error after the kernel has succeeded.
            if _USE_FUSED_FP32_QK_NORM is True:
                raise
            _USE_FUSED_FP32_QK_NORM = False
            logger.warning(
                "BAGEL fused FP32 Q/K RMSNorm unavailable; using eager RMSNorm: %s",
                error,
            )
        else:
            _USE_FUSED_FP32_QK_NORM = True
            return output

    # The JIT kernel supports FP16/BF16. Request HF ordering so its output
    # matches `_BagelRMSNorm`; FP32 and unsupported layouts use the exact eager
    # fallback in `apply_qk_norm`.
    return apply_qk_norm(
        q=query,
        k=key,
        q_norm=query_norm,
        k_norm=key_norm,
        head_dim=head_dim,
        allow_inplace=True,
        cast_x_before_out_mul=True,
    )


class _BagelMoTAttention(nn.Module):
    def __init__(
        self,
        config: BagelDiTArchConfig,
        layer_index: int,
        attention_backend: AttentionBackendEnum,
        *,
        load_generation_expert: bool,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        query_size = config.num_attention_heads * config.attention_head_dim
        kv_size = config.num_key_value_heads * config.attention_head_dim
        tp_size = _tp_size()
        if config.num_attention_heads % tp_size != 0:
            raise ValueError(
                f"BAGEL attention heads {config.num_attention_heads} must be "
                f"divisible by TP size {tp_size}"
            )
        if config.num_key_value_heads % tp_size != 0:
            raise ValueError(
                f"BAGEL KV heads {config.num_key_value_heads} must be divisible "
                f"by TP size {tp_size}"
            )
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.attention_head_dim
        self.layer_index = layer_index
        self.attention_backend = attention_backend
        # DenoisingStage discovers the selected backend by scanning child
        # modules for this conventional attribute.
        self.backend = attention_backend
        self.generation_enabled = load_generation_expert

        self.und_q_proj = _column_parallel_linear(hidden_size, query_size, bias=True)
        self.und_k_proj = _column_parallel_linear(hidden_size, kv_size, bias=True)
        self.und_v_proj = _column_parallel_linear(hidden_size, kv_size, bias=True)
        self.und_o_proj = _row_parallel_linear(query_size, hidden_size, bias=False)
        self.und_q_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.und_k_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.gen_q_proj = (
            _column_parallel_linear(hidden_size, query_size, bias=True)
            if load_generation_expert
            else None
        )
        self.gen_k_proj = (
            _column_parallel_linear(hidden_size, kv_size, bias=True)
            if load_generation_expert
            else None
        )
        self.gen_v_proj = (
            _column_parallel_linear(hidden_size, kv_size, bias=True)
            if load_generation_expert
            else None
        )
        self.gen_o_proj = (
            _row_parallel_linear(query_size, hidden_size, bias=False)
            if load_generation_expert
            else None
        )
        self.gen_q_norm = (
            _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if load_generation_expert
            else None
        )
        self.gen_k_norm = (
            _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if load_generation_expert
            else None
        )

    def _project_understanding(
        self, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        sequence_length = hidden_states.shape[0]
        query = self.und_q_proj(hidden_states).view(
            sequence_length, self.num_heads, self.head_dim
        )
        key = self.und_k_proj(hidden_states).view(
            sequence_length, self.num_kv_heads, self.head_dim
        )
        value = self.und_v_proj(hidden_states).view(
            sequence_length, self.num_kv_heads, self.head_dim
        )
        query, key = _apply_bagel_qk_norm(
            query,
            key,
            self.und_q_norm,
            self.und_k_norm,
            self.head_dim,
        )
        return query, key, value

    def _project_generation(
        self, hidden_states: Tensor, text_indexes: Tensor, latent_indexes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if not self.generation_enabled:
            raise RuntimeError(
                "BAGEL generation expert is not loaded for this transformer"
            )
        assert self.gen_q_proj is not None
        assert self.gen_k_proj is not None
        assert self.gen_v_proj is not None
        assert self.gen_q_norm is not None
        assert self.gen_k_norm is not None
        sequence_length = hidden_states.shape[0]
        # Official BAGEL keeps generation Q/K in FP32 through QK-norm and RoPE,
        # then casts to BF16 immediately before FlashAttention. Preserve that
        # ordering because casting before RoPE causes measurable parity drift.
        query = torch.zeros(
            sequence_length,
            self.num_heads,
            self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        key = torch.zeros(
            sequence_length,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        value = hidden_states.new_zeros(
            sequence_length, self.num_kv_heads, self.head_dim
        )

        if text_indexes.numel() > 0:
            text_states = hidden_states[text_indexes]
            text_query = (
                self.und_q_proj(text_states)
                .view(-1, self.num_heads, self.head_dim)
                .float()
            )
            text_key = (
                self.und_k_proj(text_states)
                .view(-1, self.num_kv_heads, self.head_dim)
                .float()
            )
            text_query, text_key = _apply_bagel_qk_norm(
                text_query,
                text_key,
                self.und_q_norm,
                self.und_k_norm,
                self.head_dim,
            )
            query[text_indexes] = text_query
            key[text_indexes] = text_key
            value[text_indexes] = self.und_v_proj(text_states).view(
                -1, self.num_kv_heads, self.head_dim
            )

        latent_states = hidden_states[latent_indexes]
        latent_query = (
            self.gen_q_proj(latent_states)
            .view(-1, self.num_heads, self.head_dim)
            .float()
        )
        latent_key = (
            self.gen_k_proj(latent_states)
            .view(-1, self.num_kv_heads, self.head_dim)
            .float()
        )
        latent_query, latent_key = _apply_bagel_qk_norm(
            latent_query,
            latent_key,
            self.gen_q_norm,
            self.gen_k_norm,
            self.head_dim,
        )
        query[latent_indexes] = latent_query
        key[latent_indexes] = latent_key
        value[latent_indexes] = self.gen_v_proj(latent_states).view(
            -1, self.num_kv_heads, self.head_dim
        )
        return query, key, value

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor,
        sine: Tensor,
        text_indexes: Tensor,
        latent_indexes: Tensor,
        *,
        mode: str,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        query_lens: Tensor | None = None,
        update_cache: bool,
        causal: bool,
    ) -> tuple[Tensor, BagelKVCache]:
        if mode == "und":
            query, key, value = self._project_understanding(hidden_states)
        elif mode == "gen":
            query, key, value = self._project_generation(
                hidden_states, text_indexes, latent_indexes
            )
        else:
            raise ValueError(f"unsupported BAGEL attention mode: {mode}")

        query, key = _apply_rope(query, key, cosine, sine)
        query = query.to(value.dtype)
        key = key.to(value.dtype)
        query_length = hidden_states.shape[0]
        is_single_request = query_lens is None
        if is_single_request:
            if prefix_lens.numel() != 1:
                raise ValueError("BAGEL singleton attention requires one prefix")
            query_lens = torch.tensor(
                [query_length], dtype=torch.int32, device=hidden_states.device
            )
            prefix_length = int(prefix_lens[0].item())
        else:
            query_lens = query_lens.to(
                device=hidden_states.device, dtype=torch.int32
            ).reshape(-1)
            prefix_lens = prefix_lens.to(
                device=hidden_states.device, dtype=torch.int32
            ).reshape(-1)
            if query_lens.numel() != prefix_lens.numel():
                raise ValueError("BAGEL prefix and query request counts must match")
            if int(query_lens.sum().item()) != query_length:
                raise ValueError("BAGEL query lengths do not match the packed query")
            prefix_length = int(prefix_lens.sum().item())
        prefix_key = prefix_cache.key_cache[self.layer_index]
        prefix_value = prefix_cache.value_cache[self.layer_index]

        if prefix_key is None:
            if prefix_value is not None or prefix_length != 0:
                raise ValueError("inconsistent BAGEL empty prefix KV cache")
            merged_key = key
            merged_value = value
        else:
            if (
                prefix_value is None
                or prefix_key.shape[0] != prefix_length
                or prefix_value.shape[0] != prefix_length
            ):
                raise ValueError("inconsistent BAGEL prefix KV cache")
            if is_single_request:
                merged_key = torch.cat((prefix_key, key), dim=0)
                merged_value = torch.cat((prefix_value, value), dim=0)
            else:
                merged_key = _interleave_prefix_and_query(
                    prefix_key, key, prefix_lens, query_lens
                )
                merged_value = _interleave_prefix_and_query(
                    prefix_value, value, prefix_lens, query_lens
                )
        merged_lens = prefix_lens + query_lens

        if update_cache:
            prefix_cache.key_cache[self.layer_index] = merged_key
            prefix_cache.value_cache[self.layer_index] = merged_value

        attended = _run_varlen_attention(
            query,
            merged_key,
            merged_value,
            query_lens,
            merged_lens,
            causal=causal,
            attention_backend=self.attention_backend,
        ).reshape(query_length, -1)

        if mode == "und":
            return self.und_o_proj(attended), prefix_cache
        assert self.gen_o_proj is not None
        output = hidden_states.new_zeros(hidden_states.shape)
        if text_indexes.numel() > 0:
            output[text_indexes] = self.und_o_proj(attended[text_indexes])
        output[latent_indexes] = self.gen_o_proj(attended[latent_indexes])
        return output, prefix_cache


class _BagelMoTMLP(nn.Module):
    def __init__(
        self, config: BagelDiTArchConfig, *, load_generation_expert: bool
    ) -> None:
        super().__init__()
        self.generation_enabled = load_generation_expert
        self.und_gate = _column_parallel_linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.und_up = _column_parallel_linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.und_down = _row_parallel_linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.gen_gate = (
            _column_parallel_linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            if load_generation_expert
            else None
        )
        self.gen_up = (
            _column_parallel_linear(
                config.hidden_size, config.intermediate_size, bias=False
            )
            if load_generation_expert
            else None
        )
        self.gen_down = (
            _row_parallel_linear(
                config.intermediate_size, config.hidden_size, bias=False
            )
            if load_generation_expert
            else None
        )

    @staticmethod
    def _expert(
        hidden_states: Tensor, gate: nn.Module, up: nn.Module, down: nn.Module
    ) -> Tensor:
        return down(F.silu(gate(hidden_states)) * up(hidden_states))

    def forward(
        self,
        hidden_states: Tensor,
        text_indexes: Tensor,
        latent_indexes: Tensor,
        *,
        mode: str,
    ) -> Tensor:
        if mode == "und":
            return self._expert(
                hidden_states, self.und_gate, self.und_up, self.und_down
            )
        if not self.generation_enabled:
            raise RuntimeError(
                "BAGEL generation expert is not loaded for this transformer"
            )
        assert self.gen_gate is not None
        assert self.gen_up is not None
        assert self.gen_down is not None
        output = torch.zeros_like(hidden_states)
        if text_indexes.numel() > 0:
            output[text_indexes] = self._expert(
                hidden_states[text_indexes],
                self.und_gate,
                self.und_up,
                self.und_down,
            )
        output[latent_indexes] = self._expert(
            hidden_states[latent_indexes], self.gen_gate, self.gen_up, self.gen_down
        )
        return output


class _BagelMoTLayer(nn.Module):
    def __init__(
        self,
        config: BagelDiTArchConfig,
        layer_index: int,
        attention_backend: AttentionBackendEnum,
        *,
        load_generation_expert: bool,
    ) -> None:
        super().__init__()
        self.und_in_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gen_in_norm = (
            _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if load_generation_expert
            else None
        )
        self.attn = _BagelMoTAttention(
            config,
            layer_index,
            attention_backend,
            load_generation_expert=load_generation_expert,
        )
        self.und_post_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gen_post_norm = (
            _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if load_generation_expert
            else None
        )
        self.mlp = _BagelMoTMLP(config, load_generation_expert=load_generation_expert)

    @staticmethod
    def _route_norm(
        hidden_states: Tensor,
        text_indexes: Tensor,
        latent_indexes: Tensor,
        *,
        mode: str,
        und_norm: _BagelRMSNorm,
        gen_norm: _BagelRMSNorm | None,
    ) -> Tensor:
        if mode == "und":
            return und_norm(hidden_states)
        if gen_norm is None:
            raise RuntimeError(
                "BAGEL generation expert is not loaded for this transformer"
            )
        output = torch.zeros_like(hidden_states)
        if text_indexes.numel() > 0:
            output[text_indexes] = und_norm(hidden_states[text_indexes])
        output[latent_indexes] = gen_norm(hidden_states[latent_indexes])
        return output

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor,
        sine: Tensor,
        text_indexes: Tensor,
        latent_indexes: Tensor,
        *,
        mode: str,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        query_lens: Tensor | None = None,
        update_cache: bool,
        causal: bool,
    ) -> tuple[Tensor, BagelKVCache]:
        normalized = self._route_norm(
            hidden_states,
            text_indexes,
            latent_indexes,
            mode=mode,
            und_norm=self.und_in_norm,
            gen_norm=self.gen_in_norm,
        )
        attention_output, prefix_cache = self.attn(
            normalized,
            cosine,
            sine,
            text_indexes,
            latent_indexes,
            mode=mode,
            prefix_cache=prefix_cache,
            prefix_lens=prefix_lens,
            query_lens=query_lens,
            update_cache=update_cache,
            causal=causal,
        )
        hidden_states = hidden_states + attention_output
        normalized = self._route_norm(
            hidden_states,
            text_indexes,
            latent_indexes,
            mode=mode,
            und_norm=self.und_post_norm,
            gen_norm=self.gen_post_norm,
        )
        return (
            hidden_states
            + self.mlp(normalized, text_indexes, latent_indexes, mode=mode),
            prefix_cache,
        )


class BagelTransformer(BaseDiT):
    """Stateless BAGEL image generator with request-owned prefix caches.

    Args:
        config: SGLang BAGEL DiT configuration.
        hf_config: Raw checkpoint configuration retained for loader parity.
        quant_config: Reserved for SGLang quantization; BAGEL supports BF16 only.
        attention_backend: Dense attention backend (``fa`` or ``torch_sdpa``).
    """

    _fsdp_shard_conditions: list = []
    _compile_conditions: list = []
    param_names_mapping = BagelDiTArchConfig().param_names_mapping
    reverse_param_names_mapping: dict = {}
    lora_param_names_mapping: dict = {}
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self,
        config: BagelDiTConfig | None = None,
        hf_config: dict | None = None,
        quant_config=None,
        attention_backend: str | AttentionBackendEnum | None = None,
        **_: object,
    ) -> None:
        config = config or BagelDiTConfig()
        if quant_config is not None:
            raise ValueError("BAGEL image generation supports unquantized BF16 only")
        super().__init__(config, hf_config or {})
        arch = config.arch_config
        selected_attention_backend = self._resolve_attention_backend(attention_backend)
        self.tp_size = _tp_size()

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.num_layers = arch.num_hidden_layers
        self.latent_patch_size = arch.latent_patch_size
        self.latent_channel = arch.latent_channel
        self.latent_downsample = arch.latent_downsample
        self.max_latent_size = arch.max_latent_size
        self.generation_enabled = bool(config.load_generation_expert)

        self.embed_tokens = _vocab_parallel_embedding(arch.vocab_size, arch.hidden_size)
        self.rotary_emb = _Qwen2RotaryEmbedding(
            arch.attention_head_dim, arch.rope_theta
        )
        self.layers = nn.ModuleList(
            [
                _BagelMoTLayer(
                    arch,
                    index,
                    selected_attention_backend,
                    load_generation_expert=self.generation_enabled,
                )
                for index in range(arch.num_hidden_layers)
            ]
        )
        self.und_final_norm = _BagelRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.gen_final_norm = (
            _BagelRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
            if self.generation_enabled
            else None
        )
        self.lm_head = (
            _column_parallel_linear(
                arch.hidden_size,
                arch.vocab_size,
                bias=False,
                gather_output=True,
            )
            if config.load_lm_head
            else None
        )

        patch_dim = arch.latent_patch_size**2 * arch.latent_channel
        self.vae2llm = (
            nn.Linear(patch_dim, arch.hidden_size) if self.generation_enabled else None
        )
        self.llm2vae = (
            nn.Linear(arch.hidden_size, patch_dim) if self.generation_enabled else None
        )
        self.time_embedder = (
            _BagelTimestepEmbedder(
                arch.hidden_size,
                frequency_embedding_size=arch.timestep_frequency_embedding_size,
            )
            if self.generation_enabled
            else None
        )
        self.latent_pos_embed = (
            _LoadedPositionTable(arch.latent_position_embedding_rows, arch.hidden_size)
            if self.generation_enabled
            else None
        )

        self.requires_grad_(False)
        self.__post_init__()

    @classmethod
    def _resolve_attention_backend(
        cls, attention_backend: str | AttentionBackendEnum | None
    ) -> AttentionBackendEnum:
        """Resolve the requested dense attention backend for BAGEL."""
        if attention_backend is None:
            return AttentionBackendEnum.FA
        if isinstance(attention_backend, AttentionBackendEnum):
            resolved = attention_backend
        else:
            normalized = attention_backend.strip().lower()
            if normalized in {"fa3", "fa4"}:
                normalized = "fa"
            try:
                resolved = AttentionBackendEnum[normalized.upper()]
            except KeyError as error:
                raise ValueError(
                    f"Unsupported BAGEL attention backend: {attention_backend}"
                ) from error
        if resolved not in cls._supported_attention_backends:
            raise ValueError(f"Unsupported BAGEL attention backend: {resolved.name}")
        return resolved

    @torch.no_grad()
    def prefill_context(self, input_ids: Tensor) -> BagelPrefixContext:
        """Prefill one text prefix into a new request-owned KV cache.

        Args:
            input_ids: One-dimensional token IDs, including prompt boundary
                tokens when required by the tokenizer contract.

        Returns:
            A prefix context that can be attached to one request.

        Raises:
            ValueError: If the input is not one-dimensional.
        """
        if input_ids.ndim != 1:
            raise ValueError(
                "BAGEL prefill expects one token sequence, "
                f"got shape {tuple(input_ids.shape)}"
            )
        device = self.device
        empty = BagelPrefixContext(
            BagelKVCache.empty(self.num_layers),
            torch.zeros(1, dtype=torch.int32, device=device),
            0,
        )
        return self._append_text_prefix(empty, input_ids)

    @staticmethod
    def _fork_prefix(prefix: BagelPrefixContext) -> BagelPrefixContext:
        """Create a cache object that shares immutable per-layer KV tensors.

        Appending replaces each list entry with a newly concatenated tensor, so
        the source prefix remains unchanged without a costly deep KV copy.
        """
        cache = BagelKVCache(
            list(prefix.kv_cache.key_cache),
            list(prefix.kv_cache.value_cache),
        )
        return BagelPrefixContext(cache, prefix.kv_lens.clone(), prefix.rope_offset)

    @staticmethod
    def _prefix_view(
        prefix: BagelPrefixContext,
        *,
        sequence_length: int,
        rope_offset: int,
    ) -> BagelPrefixContext:
        """Create an immutable leading-prefix view without copying KV tensors."""
        if sequence_length < 0 or sequence_length > prefix.kv_cache.sequence_length:
            raise ValueError("BAGEL prefix view length is outside the source cache")
        key_cache = [
            None if tensor is None else tensor[:sequence_length]
            for tensor in prefix.kv_cache.key_cache
        ]
        value_cache = [
            None if tensor is None else tensor[:sequence_length]
            for tensor in prefix.kv_cache.value_cache
        ]
        lengths = prefix.kv_lens.new_tensor([sequence_length])
        return BagelPrefixContext(
            BagelKVCache(key_cache, value_cache), lengths, rope_offset
        )

    def _append_text_prefix(
        self, prefix: BagelPrefixContext, input_ids: Tensor
    ) -> BagelPrefixContext:
        """Append a causal text block to one request-owned prefix."""
        updated_prefix, _ = self._append_text_block(prefix, input_ids)
        return updated_prefix

    def _append_text_block(
        self, prefix: BagelPrefixContext, input_ids: Tensor
    ) -> tuple[BagelPrefixContext, Tensor]:
        """Append text and return both the updated prefix and final hidden states."""
        device = self.device
        input_ids = input_ids.to(device=device, dtype=torch.long)
        if input_ids.ndim != 1:
            raise ValueError(
                "BAGEL text prefix expects one token sequence, "
                f"got shape {tuple(input_ids.shape)}"
            )
        if input_ids.numel() == 0:
            return prefix, self.embed_tokens.weight.new_empty((0, self.hidden_size))

        hidden_states = self.embed_tokens(input_ids)
        sequence_length = int(input_ids.numel())
        position_ids = torch.arange(
            prefix.rope_offset,
            prefix.rope_offset + sequence_length,
            device=device,
        )
        cosine, sine = self.rotary_emb(hidden_states, position_ids)
        text_indexes = torch.arange(sequence_length, device=device)
        latent_indexes = torch.empty(0, dtype=torch.long, device=device)
        cache = prefix.kv_cache

        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states,
                cosine,
                sine,
                text_indexes,
                latent_indexes,
                mode="und",
                prefix_cache=cache,
                prefix_lens=prefix.kv_lens,
                update_cache=True,
                causal=True,
            )
        updated_lens = prefix.kv_lens + sequence_length
        return (
            BagelPrefixContext(
                cache,
                updated_lens,
                prefix.rope_offset + sequence_length,
            ),
            hidden_states,
        )

    @torch.no_grad()
    def prepare_thinking_prefixes(
        self, system_input_ids: Tensor, user_input_ids: Tensor
    ) -> tuple[BagelPrefixContext, BagelPrefixContext]:
        """Prefill official Thinking system and user messages.

        Args:
            system_input_ids: Wrapped system message token IDs.
            user_input_ids: Wrapped user prompt token IDs.

        Returns:
            ``(system_prefix, user_prefix)`` where the latter contains both
            messages and owns only newly appended cache-list entries.

        Raises:
            RuntimeError: If this transformer was created without ``lm_head``.
        """
        self._require_lm_head()
        system_prefix = self.prefill_context(system_input_ids)
        user_prefix = self._append_text_prefix(
            self._fork_prefix(system_prefix), user_input_ids
        )
        return system_prefix, user_prefix

    @torch.no_grad()
    def build_understanding_prefix(
        self,
        vision_embeddings: Tensor,
        user_input_ids: Tensor,
        *,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
        system_input_ids: Tensor | None = None,
    ) -> BagelPrefixContext:
        """Build the official system-optional, image, then user text prefix.

        Args:
            vision_embeddings: ViT semantic tokens shaped ``[tokens, hidden_size]``.
            user_input_ids: Wrapped user message token IDs.
            start_of_image_token_id: Tokenizer-validated vision-start ID.
            end_of_image_token_id: Tokenizer-validated vision-end ID.
            system_input_ids: Optional wrapped VLM reasoning system message.

        Returns:
            Request-owned prefix ready for autoregressive text decoding.

        Raises:
            RuntimeError: If the language head is not loaded.
            ValueError: If image embeddings or token IDs are invalid.
        """
        self._require_lm_head()
        device = self.device
        prefix = BagelPrefixContext(
            BagelKVCache.empty(self.num_layers),
            torch.zeros(1, dtype=torch.int32, device=device),
            0,
        )
        if system_input_ids is not None:
            prefix = self._append_text_prefix(prefix, system_input_ids)
        prefix = self._append_image_prefix(
            prefix,
            vision_embeddings,
            mode="und",
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
        )
        return self._append_text_prefix(prefix, user_input_ids)

    @torch.no_grad()
    def generate_text(
        self,
        prefix: BagelPrefixContext,
        *,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 0.3,
        seed: int = 0,
        return_finish_reason: bool = False,
    ) -> Tensor | tuple[Tensor, str]:
        """Generate one request-local BAGEL planning message.

        The returned sequence matches official BAGEL: its first token is BOS,
        a predicted EOS is excluded, and ``max_length`` counts the BOS decode
        iteration. Text sampling uses an isolated device generator so it does
        not consume either global RNG state or the CPU diffusion-noise stream.

        Args:
            prefix: Immutable ``system + user`` request prefix.
            bos_token_id: Message-start token ID used for the first iteration.
            eos_token_id: Predicted message-end token that stops decoding.
            max_length: Maximum decode iterations, including BOS.
            do_sample: Sample from logits instead of greedy argmax.
            temperature: Positive sampling temperature when sampling is enabled.
            seed: Seed for the request-local text generator.
            return_finish_reason: Return ``(tokens, reason)`` for text response
                transports while preserving the tensor-only Thinking contract.

        Returns:
            One-dimensional generated token IDs, including BOS and excluding
            the predicted EOS. When requested, also returns ``"stop"`` after
            EOS or ``"length"`` after exhausting ``max_length``.

        Raises:
            RuntimeError: If the optional language head was not loaded.
            ValueError: If token IDs or decoding parameters are invalid.
        """
        lm_head = self._require_lm_head()
        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length <= 0
        ):
            raise ValueError("BAGEL max_length must be a positive integer")
        if (
            prefix.rope_offset + max_length
            > self.config.arch_config.max_position_embeddings
        ):
            raise ValueError(
                "BAGEL text generation would exceed max_position_embeddings"
            )
        if not isinstance(do_sample, bool):
            raise ValueError("BAGEL do_sample must be a boolean")
        if do_sample and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not math.isfinite(float(temperature))
            or temperature <= 0
        ):
            raise ValueError("BAGEL sampling temperature must be finite and positive")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("BAGEL text seed must be a non-negative integer")
        vocab_size = self.config.arch_config.vocab_size
        if not 0 <= int(bos_token_id) < vocab_size:
            raise ValueError("BAGEL BOS token ID is outside the vocabulary")
        if not 0 <= int(eos_token_id) < vocab_size:
            raise ValueError("BAGEL EOS token ID is outside the vocabulary")

        device = self.device
        decode_prefix = self._fork_prefix(prefix)
        current = torch.tensor([int(bos_token_id)], dtype=torch.long, device=device)
        generated: list[Tensor] = []
        text_generator = None
        hit_eos = False
        if do_sample:
            text_generator = torch.Generator(device=device)
            text_generator.manual_seed(int(seed))

        for _ in range(max_length):
            generated.append(current)
            decode_prefix, hidden_states = self._append_text_block(
                decode_prefix, current
            )
            logits = lm_head(self.und_final_norm(hidden_states[-1]))
            if do_sample:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=text_generator,
                )
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if int(next_token.item()) == int(eos_token_id):
                hit_eos = True
                break
            current = next_token.to(dtype=torch.long)
        generated_ids = torch.cat(generated)
        if return_finish_reason:
            return generated_ids, "stop" if hit_eos else "length"
        return generated_ids

    @torch.no_grad()
    def build_thinking_context(
        self,
        system_prefix: BagelPrefixContext,
        user_prefix: BagelPrefixContext,
        thought_input_ids: Tensor,
        *,
        height: int,
        width: int,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
    ) -> BagelContext:
        """Build official three-way CFG prefixes after planning.

        Args:
            system_prefix: Prefix containing only the wrapped system message.
            user_prefix: Prefix containing system and wrapped user messages.
            thought_input_ids: Clean generated thought rewrapped with BOS/EOS.
            height: Requested output image height.
            width: Requested output image width.
            start_of_image_token_id: Tokenizer-validated vision-start ID.
            end_of_image_token_id: Tokenizer-validated vision-end ID.

        Returns:
            Thinking context whose branches are ``system + user + thought``,
            ``system``, and ``system + user``.

        Raises:
            ValueError: If geometry, token IDs, or thought input is invalid.
        """
        self._validate_image_size(height, width)
        self._validate_special_token_ids(
            int(start_of_image_token_id), int(end_of_image_token_id)
        )
        if thought_input_ids.ndim != 1 or thought_input_ids.numel() < 2:
            raise ValueError("BAGEL Thinking requires one wrapped thought message")
        conditional = self._append_text_prefix(
            self._fork_prefix(user_prefix), thought_input_ids
        )
        return BagelContext.from_prefixes(
            conditional,
            system_prefix,
            height=height,
            width=width,
            start_of_image_token_id=int(start_of_image_token_id),
            end_of_image_token_id=int(end_of_image_token_id),
            secondary_unconditional=user_prefix,
            three_way_cfg_kind="thinking",
        )

    def _require_lm_head(self) -> nn.Linear:
        """Return the loaded language head or reject non-Thinking pipelines."""
        if self.lm_head is None:
            raise RuntimeError(
                "BAGEL text generation requires a transformer with load_lm_head=True"
            )
        return self.lm_head

    def _require_generation_expert(self) -> None:
        """Reject image-generation calls on the slim UND-only transformer."""
        if not self.generation_enabled:
            raise RuntimeError(
                "BAGEL image generation requires load_generation_expert=True"
            )

    def _append_image_prefix(
        self,
        prefix: BagelPrefixContext,
        image_embeddings: Tensor,
        *,
        mode: str,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
    ) -> BagelPrefixContext:
        """Append one VAE-generation or ViT-understanding image block."""
        if image_embeddings.ndim != 2:
            raise ValueError(
                "BAGEL image embeddings must be a 2D token matrix, "
                f"got shape {tuple(image_embeddings.shape)}"
            )
        if image_embeddings.shape[1] != self.hidden_size:
            raise ValueError(
                "BAGEL image embedding width must equal hidden_size; "
                f"got {image_embeddings.shape[1]} and {self.hidden_size}"
            )
        if mode not in {"gen", "und"}:
            raise ValueError(f"unsupported BAGEL image-prefix mode: {mode}")

        device = self.device
        boundary_ids = torch.tensor(
            [start_of_image_token_id, end_of_image_token_id],
            dtype=torch.long,
            device=device,
        )
        boundary_embeddings = self.embed_tokens(boundary_ids)
        image_embeddings = image_embeddings.to(
            device=device, dtype=boundary_embeddings.dtype
        )
        sequence_length = image_embeddings.shape[0] + 2
        hidden_states = boundary_embeddings.new_zeros(sequence_length, self.hidden_size)
        boundary_indexes = torch.tensor(
            [0, sequence_length - 1], dtype=torch.long, device=device
        )
        image_indexes = torch.arange(1, sequence_length - 1, device=device)
        hidden_states[boundary_indexes] = boundary_embeddings
        hidden_states[image_indexes] = image_embeddings

        if mode == "gen":
            text_indexes = boundary_indexes
            latent_indexes = image_indexes
        else:
            text_indexes = torch.arange(sequence_length, device=device)
            latent_indexes = torch.empty(0, dtype=torch.long, device=device)
        position_ids = torch.full(
            (sequence_length,),
            prefix.rope_offset,
            dtype=torch.long,
            device=device,
        )
        cosine, sine = self.rotary_emb(hidden_states, position_ids)
        cache = prefix.kv_cache
        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states,
                cosine,
                sine,
                text_indexes,
                latent_indexes,
                mode=mode,
                prefix_cache=cache,
                prefix_lens=prefix.kv_lens,
                update_cache=True,
                causal=False,
            )
        return BagelPrefixContext(
            cache,
            prefix.kv_lens + sequence_length,
            prefix.rope_offset + 1,
        )

    @staticmethod
    def pack_contexts(contexts: list[BagelContext]) -> BagelContext:
        """Pack compatible T2I contexts for true dynamic batching.

        Args:
            contexts: Request-local contexts in scheduler merge order.

        Returns:
            One context containing packed KV caches and per-request metadata.

        Raises:
            ValueError: If the contexts are not compatible pure T2I requests.
        """
        return BagelContext.pack(contexts)

    @torch.no_grad()
    def build_context(
        self,
        conditional_input_ids: Tensor,
        unconditional_input_ids: Tensor | None = None,
        *,
        height: int,
        width: int,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
    ) -> BagelContext:
        """Build conditional and unconditional context for one T2I request.

        Args:
            conditional_input_ids: Prompt token IDs.
            unconditional_input_ids: Optional negative/unconditional token IDs;
                ``None`` selects the official empty CFG prefix.
            height: Requested output image height.
            width: Requested output image width.
            start_of_image_token_id: Tokenizer-validated vision start ID.
            end_of_image_token_id: Tokenizer-validated vision end ID.

        Returns:
            Request-owned context passed explicitly to :meth:`forward`.

        Raises:
            ValueError: If image dimensions cannot be represented by the
                checkpoint's positional table.
        """
        self._validate_image_size(height, width)
        start_of_image_token_id = int(start_of_image_token_id)
        end_of_image_token_id = int(end_of_image_token_id)
        self._validate_special_token_ids(start_of_image_token_id, end_of_image_token_id)
        conditional = self.prefill_context(conditional_input_ids)
        if unconditional_input_ids is None:
            unconditional_input_ids = conditional_input_ids.new_empty((0,))
        unconditional = self.prefill_context(unconditional_input_ids)
        return BagelContext.from_prefixes(
            conditional,
            unconditional,
            height=height,
            width=width,
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
        )

    @torch.no_grad()
    def build_editing_context(
        self,
        vae_patches: Tensor,
        vae_position_ids: Tensor,
        vision_embeddings: Tensor,
        text_input_ids: Tensor,
        *,
        height: int,
        width: int,
        start_of_image_token_id: int,
        end_of_image_token_id: int,
    ) -> BagelContext:
        """Build main, image-only, and text-only prefixes for one edit.

        Args:
            vae_patches: Sampled source-image latents patchified to
                ``[tokens, latent_patch_size**2 * latent_channel]``.
            vae_position_ids: Source-image latent position IDs.
            vision_embeddings: ViT + connector + position embeddings shaped
                ``[tokens, hidden_size]``.
            text_input_ids: Editing instruction wrapped in message boundaries.
            height: Output image height.
            width: Output image width.
            start_of_image_token_id: Tokenizer-validated vision-start ID.
            end_of_image_token_id: Tokenizer-validated vision-end ID.

        Returns:
            A request-owned three-way denoising context.

        Raises:
            RuntimeError: If generation-only experts were not loaded.
            ValueError: If image-token shapes, positions, or output geometry do
                not match checkpoint constraints.
        """
        self._require_generation_expert()
        self._validate_image_size(height, width)
        start_of_image_token_id = int(start_of_image_token_id)
        end_of_image_token_id = int(end_of_image_token_id)
        self._validate_special_token_ids(start_of_image_token_id, end_of_image_token_id)

        expected_patch_width = self.latent_patch_size**2 * self.latent_channel
        if vae_patches.ndim != 2 or vae_patches.shape[1] != expected_patch_width:
            raise ValueError(
                "BAGEL Editing VAE patches must have shape [tokens, "
                f"{expected_patch_width}], got {tuple(vae_patches.shape)}"
            )
        if (
            vae_position_ids.ndim != 1
            or vae_position_ids.shape[0] != vae_patches.shape[0]
        ):
            raise ValueError("BAGEL Editing VAE patch and position counts must match")
        if vae_position_ids.numel() and (
            int(vae_position_ids.min()) < 0
            or int(vae_position_ids.max()) >= self.max_latent_size**2
        ):
            raise ValueError(
                "BAGEL Editing VAE position ID is outside the checkpoint table"
            )
        if text_input_ids.ndim != 1 or text_input_ids.numel() == 0:
            raise ValueError("BAGEL Editing requires one non-empty text token sequence")

        device = self.device
        vae_patches = vae_patches.to(device=device, dtype=self.vae2llm.weight.dtype)
        vae_position_ids = vae_position_ids.to(device=device, dtype=torch.long)
        clean_timestep = torch.zeros(
            vae_patches.shape[0], device=device, dtype=torch.float32
        )
        vae_embeddings = (
            self.vae2llm(vae_patches)
            + self.time_embedder(clean_timestep)
            + self.latent_pos_embed(vae_position_ids)
        )

        empty = BagelPrefixContext(
            BagelKVCache.empty(self.num_layers),
            torch.zeros(1, dtype=torch.int32, device=device),
            0,
        )
        image_prefix = self._append_image_prefix(
            empty,
            vae_embeddings,
            mode="gen",
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
        )
        image_prefix = self._append_image_prefix(
            image_prefix,
            vision_embeddings,
            mode="und",
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
        )

        image_prefix_length = image_prefix.kv_cache.sequence_length
        image_rope_offset = image_prefix.rope_offset
        main_prefix = self._append_text_prefix(
            self._fork_prefix(image_prefix), text_input_ids
        )
        # Slice image-only views from the completed main cache. These views
        # share storage with main instead of retaining a second full image KV.
        image_only_prefix = self._prefix_view(
            main_prefix,
            sequence_length=image_prefix_length,
            rope_offset=image_rope_offset,
        )
        text_only_prefix = self.prefill_context(text_input_ids)
        return BagelContext.from_prefixes(
            main_prefix,
            image_only_prefix,
            height=height,
            width=width,
            start_of_image_token_id=start_of_image_token_id,
            end_of_image_token_id=end_of_image_token_id,
            secondary_unconditional=text_only_prefix,
            three_way_cfg_kind="editing",
        )

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor | list[Tensor] | None = None,
        *,
        bagel_context: BagelContext,
        guidance_scale: float | Tensor = 4.0,
        image_guidance_scale: float | Tensor = 1.0,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        taylorseer_context: BagelTaylorSeerContext | None = None,
        guidance: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        """Predict flow velocity with BAGEL's internal two- or three-way CFG.

        Args:
            hidden_states: Patchified latents shaped ``[tokens, patch_dim]`` or
                ``[batch, tokens, patch_dim]``.
            timestep: Scalar, per-request, or per-token raw flow sigma.
            encoder_hidden_states: Unused standard DiT compatibility argument.
            bagel_context: Explicit request-owned prefix and image state.
            guidance_scale: Text classifier-free guidance strength.
            image_guidance_scale: Secondary three-way CFG strength for Editing
                or Thinking.
            cfg_interval: Open/closed sigma interval ``(low, high]``.
            cfg_renorm_min: Lower clamp for BAGEL's CFG norm correction.
            cfg_renorm_type: ``global`` or ``channel``.
            taylorseer_context: Optional request-owned layer forecast state.
            guidance: Unused standard embedded-guidance argument.

        Returns:
            Flow prediction with the same rank and shape as ``hidden_states``.

        Raises:
            RuntimeError: If generation-only experts were not loaded.
            ValueError: If batched/dynamic input or context geometry is invalid.
        """
        del encoder_hidden_states, guidance
        self._require_generation_expert()
        had_batch_dimension = hidden_states.ndim == 3
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        elif hidden_states.ndim != 3:
            raise ValueError(
                "BAGEL expects [tokens, patch_dim] or [batch, tokens, patch_dim] "
                "latents, "
                f"got shape {tuple(hidden_states.shape)}"
            )
        self._validate_hidden_states(hidden_states, bagel_context)

        timestep = self._normalize_timestep(
            timestep,
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.device,
        )
        if hidden_states.shape[0] > 1:
            sample_sigmas = timestep.reshape(hidden_states.shape[:2])[:, 0]
            if not torch.allclose(
                sample_sigmas, sample_sigmas[:1].expand_as(sample_sigmas)
            ):
                raise ValueError("BAGEL batched requests must use the same timestep")
        if taylorseer_context is not None:
            taylorseer_context.validate_branch_count(
                has_secondary=bagel_context.has_three_way_cfg
            )
        conditional = self._generation_branch(
            hidden_states,
            timestep,
            bagel_context.conditional_kv,
            bagel_context.conditional_kv_lens,
            bagel_context.conditional_rope_offset,
            bagel_context,
            taylorseer_context.conditional if taylorseer_context else None,
        )

        scale = self._as_float(guidance_scale)
        image_scale = self._as_float(image_guidance_scale)
        sigma = float(timestep[0].item())
        cfg_enabled = cfg_interval[0] < sigma <= cfg_interval[1] and scale > 1.0
        if cfg_enabled:
            unconditional = self._generation_branch(
                hidden_states,
                timestep,
                bagel_context.unconditional_kv,
                bagel_context.unconditional_kv_lens,
                bagel_context.unconditional_rope_offset,
                bagel_context,
                taylorseer_context.unconditional if taylorseer_context else None,
            )
            if bagel_context.has_three_way_cfg:
                secondary_unconditional_kv = bagel_context.secondary_unconditional_kv
                secondary_unconditional_lens = (
                    bagel_context.secondary_unconditional_kv_lens
                )
                secondary_unconditional_offset = (
                    bagel_context.secondary_unconditional_rope_offset
                )
                if (
                    secondary_unconditional_kv is None
                    or secondary_unconditional_lens is None
                    or secondary_unconditional_offset is None
                ):
                    raise ValueError("incomplete BAGEL three-way CFG context")
                secondary_unconditional = None
                if image_scale > 1.0:
                    secondary_unconditional = self._generation_branch(
                        hidden_states,
                        timestep,
                        secondary_unconditional_kv,
                        secondary_unconditional_lens,
                        secondary_unconditional_offset,
                        bagel_context,
                        (
                            taylorseer_context.secondary_unconditional
                            if taylorseer_context
                            else None
                        ),
                    )
                conditional = self._apply_cfg_three_way(
                    conditional,
                    unconditional,
                    secondary_unconditional,
                    scale,
                    image_scale,
                    renorm_min=cfg_renorm_min,
                    renorm_type=cfg_renorm_type,
                )
            else:
                conditional = self._apply_cfg(
                    conditional,
                    unconditional,
                    scale,
                    renorm_min=cfg_renorm_min,
                    renorm_type=cfg_renorm_type,
                )
        # Preserve official BF16 velocity rounding. The BAGEL scheduler keeps
        # the persistent Euler state in FP32 without upcasting this delta first.
        if hidden_states.shape[0] == 1:
            return conditional.unsqueeze(0) if had_batch_dimension else conditional
        return conditional

    def _generation_branch(
        self,
        latents: Tensor,
        timestep: Tensor,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        rope_offset: int | Tensor,
        context: BagelContext,
        taylorseer_state: TaylorSeerState | None,
    ) -> Tensor:
        """Run one CFG branch and advance only that branch's Taylor state.

        Args:
            latents: Current request or packed-request latent tokens.
            timestep: Per-token raw flow sigma values.
            prefix_cache: Immutable request prefix KV cache.
            prefix_lens: Prefix length for each packed request.
            rope_offset: RoPE offset for each packed request.
            context: Request-owned BAGEL geometry and token IDs.
            taylorseer_state: Optional cache for this exact CFG branch.

        Returns:
            Predicted flow tokens for this CFG branch.

        Raises:
            ValueError: If state, context, or packed geometry is inconsistent.
            RuntimeError: If Taylor state lifecycle or model execution fails.
        """
        if taylorseer_state is None:
            return self._generation_step(
                latents,
                timestep,
                prefix_cache,
                prefix_lens,
                rope_offset,
                context,
                None,
            )

        # Official BAGEL advances each cache on actual branch forwards. In
        # particular, CFG-disabled steps must not age unconditional caches.
        taylorseer_state.begin_next_step()
        try:
            prediction = self._generation_step(
                latents,
                timestep,
                prefix_cache,
                prefix_lens,
                rope_offset,
                context,
                taylorseer_state,
            )
            taylorseer_state.end_step()
        except Exception:
            # Per-layer updates are not transactional. Poison every CFG branch
            # in the shared request context and release cached tensors instead
            # of permitting a retry with partially advanced derivatives.
            taylorseer_state.poison()
            raise
        return prediction

    def _generation_step(
        self,
        latents: Tensor,
        timestep: Tensor,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        rope_offset: int | Tensor,
        context: BagelContext,
        taylorseer_state: TaylorSeerState | None,
    ) -> Tensor:
        device = latents.device
        batch_size, token_count, patch_width = latents.shape
        if prefix_lens.numel() != batch_size:
            raise ValueError("BAGEL context request count does not match latent batch")
        if batch_size == 1:
            # Preserve the original singleton operation order exactly. Besides
            # avoiding varlen packing overhead, this keeps fixed-seed B=1 output
            # bit-identical to the pre-batching implementation.
            return self._generation_step_single(
                latents[0],
                timestep,
                prefix_cache,
                prefix_lens,
                rope_offset,
                context,
                taylorseer_state,
            )

        rope_offsets = torch.as_tensor(
            rope_offset, dtype=torch.long, device=device
        ).reshape(-1)
        if rope_offsets.numel() == 1:
            rope_offsets = rope_offsets.expand(batch_size)
        elif rope_offsets.numel() != batch_size:
            raise ValueError("BAGEL RoPE offsets must match latent batch size")

        position_ids = self._latent_position_ids(
            context.height, context.width, device
        ).repeat(batch_size)
        boundary_ids = torch.tensor(
            [context.start_of_image_token_id, context.end_of_image_token_id],
            dtype=torch.long,
            device=device,
        ).repeat(batch_size)
        boundary_embeddings = self.embed_tokens(boundary_ids)
        flat_latents = latents.reshape(batch_size * token_count, patch_width)
        latent_embeddings = (
            self.vae2llm(flat_latents)
            + self.time_embedder(timestep)
            + self.latent_pos_embed(position_ids)
        )
        latent_embeddings = latent_embeddings.to(boundary_embeddings.dtype)

        sequence_length = token_count + 2
        query_lens = torch.full(
            (batch_size,), sequence_length, dtype=torch.int32, device=device
        )
        sequence_offsets = torch.arange(batch_size, device=device) * sequence_length
        text_indexes = (
            sequence_offsets[:, None]
            + torch.tensor([0, sequence_length - 1], device=device)
        ).reshape(-1)
        latent_indexes = (
            sequence_offsets[:, None]
            + torch.arange(1, sequence_length - 1, device=device)
        ).reshape(-1)
        packed = boundary_embeddings.new_zeros(
            batch_size * sequence_length, self.hidden_size
        )
        packed[text_indexes] = boundary_embeddings
        packed[latent_indexes] = latent_embeddings

        rope_positions = rope_offsets.repeat_interleave(sequence_length)
        cosine, sine = self.rotary_emb(packed, rope_positions)
        for layer_index, layer in enumerate(self.layers):
            if taylorseer_state is not None and not taylorseer_state.should_compute(
                layer_index
            ):
                packed = taylorseer_state.approximate(layer_index)
                continue
            packed, _ = layer(
                packed,
                cosine,
                sine,
                text_indexes,
                latent_indexes,
                mode="gen",
                prefix_cache=prefix_cache,
                prefix_lens=prefix_lens,
                query_lens=query_lens,
                update_cache=False,
                causal=False,
            )
            if taylorseer_state is not None:
                taylorseer_state.update_cache(layer_index, packed)

        normalized = torch.zeros_like(packed)
        normalized[text_indexes] = self.und_final_norm(packed[text_indexes])
        normalized[latent_indexes] = self.gen_final_norm(packed[latent_indexes])
        return self.llm2vae(normalized[latent_indexes]).reshape(
            batch_size, token_count, patch_width
        )

    def _generation_step_single(
        self,
        latents: Tensor,
        timestep: Tensor,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        rope_offset: int | Tensor,
        context: BagelContext,
        taylorseer_state: TaylorSeerState | None,
    ) -> Tensor:
        """Run the original singleton image-token path without batch packing."""
        device = latents.device
        position_ids = self._latent_position_ids(context.height, context.width, device)
        boundary_ids = torch.tensor(
            [context.start_of_image_token_id, context.end_of_image_token_id],
            dtype=torch.long,
            device=device,
        )
        boundary_embeddings = self.embed_tokens(boundary_ids)
        latent_embeddings = (
            self.vae2llm(latents)
            + self.time_embedder(timestep)
            + self.latent_pos_embed(position_ids)
        )
        latent_embeddings = latent_embeddings.to(boundary_embeddings.dtype)

        sequence_length = latents.shape[0] + 2
        packed = boundary_embeddings.new_zeros(sequence_length, self.hidden_size)
        text_indexes = torch.tensor(
            [0, sequence_length - 1], dtype=torch.long, device=device
        )
        latent_indexes = torch.arange(1, sequence_length - 1, device=device)
        packed[text_indexes] = boundary_embeddings
        packed[latent_indexes] = latent_embeddings

        singleton_rope_offset = int(torch.as_tensor(rope_offset).reshape(-1)[0].item())
        rope_positions = torch.full(
            (sequence_length,), singleton_rope_offset, dtype=torch.long, device=device
        )
        cosine, sine = self.rotary_emb(packed, rope_positions)
        for layer_index, layer in enumerate(self.layers):
            if taylorseer_state is not None and not taylorseer_state.should_compute(
                layer_index
            ):
                packed = taylorseer_state.approximate(layer_index)
                continue
            packed, _ = layer(
                packed,
                cosine,
                sine,
                text_indexes,
                latent_indexes,
                mode="gen",
                prefix_cache=prefix_cache,
                prefix_lens=prefix_lens,
                update_cache=False,
                causal=False,
            )
            if taylorseer_state is not None:
                taylorseer_state.update_cache(layer_index, packed)

        normalized = torch.zeros_like(packed)
        normalized[text_indexes] = self.und_final_norm(packed[text_indexes])
        normalized[latent_indexes] = self.gen_final_norm(packed[latent_indexes])
        return self.llm2vae(normalized[latent_indexes])

    def _validate_image_size(self, height: int, width: int) -> None:
        if height <= 0 or width <= 0:
            raise ValueError("BAGEL image dimensions must be positive")
        if height % self.latent_downsample or width % self.latent_downsample:
            raise ValueError(
                f"BAGEL image dimensions must be divisible by {self.latent_downsample}"
            )
        latent_height = height // self.latent_downsample
        latent_width = width // self.latent_downsample
        if latent_height > self.max_latent_size or latent_width > self.max_latent_size:
            raise ValueError(
                "BAGEL image exceeds the checkpoint position table: "
                f"{latent_height}x{latent_width} latent patches, maximum "
                f"{self.max_latent_size}x{self.max_latent_size}"
            )

    def _validate_special_token_ids(self, start: int, end: int) -> None:
        arch = self.config.arch_config
        vocab_size = arch.vocab_size
        if not 0 <= start < vocab_size or not 0 <= end < vocab_size:
            raise ValueError("BAGEL vision token IDs must be inside the vocabulary")
        expected = (arch.start_of_image_token_id, arch.end_of_image_token_id)
        if (start, end) != expected:
            raise ValueError(
                "BAGEL vision token IDs do not match the checkpoint: expected "
                f"{expected}, got {(start, end)}"
            )

    def _validate_hidden_states(
        self, hidden_states: Tensor, context: BagelContext
    ) -> None:
        self._validate_image_size(context.height, context.width)
        if hidden_states.ndim != 3:
            raise ValueError("BAGEL internal latents must have rank three")
        batch_size = hidden_states.shape[0]
        if context.batch_size != batch_size:
            raise ValueError("BAGEL context request count does not match latent batch")
        if context.unconditional_kv_lens.numel() != batch_size:
            raise ValueError("BAGEL unconditional context count does not match batch")
        if batch_size > 1 and context.has_three_way_cfg:
            raise ValueError("BAGEL dynamic batching supports pure T2I only")
        expected_tokens = (context.height // self.latent_downsample) * (
            context.width // self.latent_downsample
        )
        expected_channels = self.latent_patch_size**2 * self.latent_channel
        expected_shape = (batch_size, expected_tokens, expected_channels)
        if hidden_states.shape != expected_shape:
            raise ValueError(
                "BAGEL latent shape does not match request geometry: expected "
                f"{expected_shape}, got "
                f"{tuple(hidden_states.shape)}"
            )

    def _latent_position_ids(
        self, height: int, width: int, device: torch.device
    ) -> Tensor:
        latent_height = height // self.latent_downsample
        latent_width = width // self.latent_downsample
        rows = torch.arange(latent_height, device=device).unsqueeze(1)
        columns = torch.arange(latent_width, device=device).unsqueeze(0)
        return (rows * self.max_latent_size + columns).reshape(-1)

    @staticmethod
    def _normalize_timestep(
        timestep: Tensor,
        batch_size: int,
        token_count: int,
        device: torch.device,
    ) -> Tensor:
        timestep = torch.as_tensor(timestep, device=device).reshape(-1)
        if timestep.numel() == 1:
            return timestep.expand(batch_size * token_count)
        if timestep.numel() == batch_size:
            return timestep.repeat_interleave(token_count)
        if timestep.numel() == batch_size * token_count:
            return timestep
        raise ValueError(
            "BAGEL timestep must be scalar, per-request, or per-token; "
            f"got {timestep.numel()} values for {batch_size}x{token_count} tokens"
        )

    @staticmethod
    def _as_float(value: float | Tensor) -> float:
        if isinstance(value, Tensor):
            if value.numel() == 0:
                raise ValueError("guidance_scale tensor cannot be empty")
            return float(value.reshape(-1)[0].item())
        return float(value)

    @staticmethod
    def _apply_cfg(
        conditional: Tensor,
        unconditional: Tensor,
        guidance_scale: float,
        *,
        renorm_min: float,
        renorm_type: str,
    ) -> Tensor:
        guided = unconditional + guidance_scale * (conditional - unconditional)
        if renorm_type == "global":
            if conditional.ndim == 3:
                original_norm = torch.linalg.vector_norm(
                    conditional, dim=(1, 2), keepdim=True
                )
                guided_norm = torch.linalg.vector_norm(guided, dim=(1, 2), keepdim=True)
            else:
                original_norm = torch.linalg.vector_norm(conditional)
                guided_norm = torch.linalg.vector_norm(guided)
        elif renorm_type == "channel":
            original_norm = torch.linalg.vector_norm(conditional, dim=-1, keepdim=True)
            guided_norm = torch.linalg.vector_norm(guided, dim=-1, keepdim=True)
        else:
            raise ValueError(f"unsupported BAGEL CFG renorm type: {renorm_type}")
        correction = (original_norm / (guided_norm + 1e-8)).clamp(
            min=renorm_min, max=1.0
        )
        output = guided * correction
        if renorm_type == "global" and conditional.ndim == 3:
            # CUDA autocast computes norms in FP32. A batched global correction
            # has shape [B, 1, 1] and would otherwise promote BF16 velocity to
            # FP32, unlike each equivalent singleton's scalar correction.
            output = output.to(guided.dtype)
        return output

    @staticmethod
    def _apply_cfg_three_way(
        main: Tensor,
        image_only: Tensor,
        text_only: Tensor | None,
        text_scale: float,
        image_scale: float,
        *,
        renorm_min: float,
        renorm_type: str,
    ) -> Tensor:
        """Combine BAGEL's main and two baseline predictions.

        Args:
            main: Fully conditioned prediction.
            image_only: First CFG baseline prediction.
            text_only: Second CFG baseline prediction, or ``None`` when the
                secondary CFG scale is disabled.
            text_scale: Text CFG scale.
            image_scale: Image CFG scale.
            renorm_min: Lower clamp for norm correction.
            renorm_type: ``text_channel`` (official Editing default), ``global``,
                or ``channel``.

        Returns:
            Three-way guided flow prediction.

        Raises:
            ValueError: If ``renorm_type`` is unsupported or image CFG is
                enabled without a text-only prediction.
        """
        text_guided = image_only + text_scale * (main - image_only)
        if renorm_type == "text_channel":
            original_norm = torch.linalg.vector_norm(main, dim=-1, keepdim=True)
            guided_norm = torch.linalg.vector_norm(text_guided, dim=-1, keepdim=True)
            correction = (original_norm / (guided_norm + 1e-8)).clamp(
                min=renorm_min, max=1.0
            )
            text_guided = text_guided * correction
            if image_scale <= 1.0:
                return text_guided
            if text_only is None:
                raise ValueError("BAGEL image CFG requires a text-only prediction")
            return text_only + image_scale * (text_guided - text_only)

        if image_scale > 1.0:
            if text_only is None:
                raise ValueError("BAGEL image CFG requires a text-only prediction")
            guided = text_only + image_scale * (text_guided - text_only)
        else:
            guided = text_guided
        if renorm_type == "global":
            original_norm = torch.linalg.vector_norm(main)
            guided_norm = torch.linalg.vector_norm(guided)
        elif renorm_type == "channel":
            original_norm = torch.linalg.vector_norm(main, dim=-1, keepdim=True)
            guided_norm = torch.linalg.vector_norm(guided, dim=-1, keepdim=True)
        else:
            raise ValueError(f"unsupported BAGEL CFG renorm type: {renorm_type}")
        correction = (original_norm / (guided_norm + 1e-8)).clamp(
            min=renorm_min, max=1.0
        )
        return guided * correction

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
        *,
        strict: bool = True,
    ) -> set[str]:
        """Stream the selected image-generation weights into this model.

        Args:
            weights: Iterator of ``(checkpoint_name, tensor)`` pairs.
            strict: Enforce complete target coverage and classified source keys.

        Returns:
            Target parameter names populated from the iterator.

        Raises:
            ValueError: If required weights are missing, shapes mismatch, or a
                checkpoint key is outside the explicit component allowlist.
        """
        params = dict(self.named_parameters())
        required = set(params)
        loaded: set[str] = set()
        unexpected: list[str] = []
        for source_name, tensor in weights:
            if not self.accepts_checkpoint_weight(source_name):
                continue
            target_name = self._map_checkpoint_name(source_name)
            parameter = params.get(target_name)
            if parameter is None:
                unexpected.append(source_name)
                continue
            self._load_parameter(target_name, parameter, tensor)
            loaded.add(target_name)

        missing = sorted(required - loaded)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing BAGEL weights: {missing}")
            if unexpected:
                details.append(f"unclassified checkpoint weights: {sorted(unexpected)}")
            raise ValueError("; ".join(details))
        return loaded

    def accepts_checkpoint_weight(self, source_name: str) -> bool:
        """Return whether this configured transformer owns a checkpoint tensor.

        Args:
            source_name: Tensor name from BAGEL's ``ema.safetensors``.

        Returns:
            ``True`` when the tensor must be streamed into this transformer.
        """
        if source_name.startswith(("connector.", "vit_model.", "vit_pos_embed.")):
            return False
        if self.lm_head is None and source_name.startswith("language_model.lm_head."):
            return False
        target_name = self._map_checkpoint_name(source_name)
        if not self.generation_enabled and self._is_generation_target(target_name):
            return False
        return True

    @staticmethod
    def _is_generation_target(target_name: str) -> bool:
        """Classify parameters used only by image generation."""
        if target_name.startswith(
            (
                "gen_final_norm.",
                "vae2llm.",
                "llm2vae.",
                "time_embedder.",
                "latent_pos_embed.",
            )
        ):
            return True
        return bool(
            re.match(r"^layers\.\d+\.(?:gen_|attn\.gen_|mlp\.gen_)", target_name)
        )

    def _map_checkpoint_name(self, source_name: str) -> str:
        name = source_name
        for pattern, replacement in self.param_names_mapping.items():
            if re.match(pattern, name):
                name = re.sub(pattern, replacement, name)
                break
        return name

    def _load_parameter(
        self, name: str, parameter: nn.Parameter, tensor: Tensor
    ) -> None:
        weight_loader = getattr(parameter, "weight_loader", None)
        if parameter.is_meta:
            parent: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            if weight_loader is None:
                if tuple(parameter.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"BAGEL weight shape mismatch for {name}: expected "
                        f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
                    )
                materialized = nn.Parameter(
                    tensor.to(dtype=parameter.dtype), requires_grad=False
                )
            else:
                materialized = nn.Parameter(
                    torch.empty(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=tensor.device,
                    ),
                    requires_grad=False,
                )
                materialized.__dict__.update(parameter.__dict__)
                weight_loader(materialized, tensor)
            setattr(parent, parts[-1], materialized)
            return

        if weight_loader is not None:
            weight_loader(parameter, tensor)
            return
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"BAGEL weight shape mismatch for {name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
            )
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


EntryClass = BagelTransformer
