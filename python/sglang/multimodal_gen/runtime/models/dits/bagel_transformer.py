# Copyright 2024 The Qwen Team and The Hugging Face Inc. team.
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""BAGEL's Qwen2 mixture-of-transformers model for text-to-image denoising.

This module implements only the Apache-2.0 T2I path from BAGEL.  In
particular, it does not copy ``modeling/bagel/modeling_utils.py`` because that
file is CC BY-NC 4.0.  Timestep embeddings reuse SGLang's Apache-2.0 layer and
the positional table is loaded directly from the checkpoint.

Sources:
  - https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/bagel.py
  - https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/bagel/qwen2_navit.py
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.multimodal_gen.configs.models.dits.bagel import (
    BagelDiTArchConfig,
    BagelDiTConfig,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import TimestepEmbedder
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum

logger = logging.getLogger(__name__)


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
        """Return the prefix length represented by this cache."""
        first_key = self.key_cache[0] if self.key_cache else None
        return 0 if first_key is None else int(first_key.shape[0])


@dataclass(frozen=True)
class BagelPrefixContext:
    """One immutable view of a request-owned text prefix."""

    kv_cache: BagelKVCache
    kv_lens: Tensor
    rope_offset: int


@dataclass(frozen=True)
class BagelContext:
    """All request-local state consumed by BAGEL denoising."""

    conditional_kv: BagelKVCache
    unconditional_kv: BagelKVCache
    conditional_kv_lens: Tensor
    unconditional_kv_lens: Tensor
    conditional_rope_offset: int
    unconditional_rope_offset: int
    height: int
    width: int
    start_of_image_token_id: int
    end_of_image_token_id: int

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
    ) -> BagelContext:
        """Build a denoising context from separately prepared prefixes."""
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
    output = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=causal
    )
    return output.squeeze(0).transpose(0, 1)


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
    """Use SGLang FlashAttention on CUDA and a CPU-safe SDPA fallback."""
    if query_lens.numel() != 1 or key_lens.numel() != 1:
        raise ValueError("BAGEL T2I supports exactly one request per forward")

    if (
        attention_backend == AttentionBackendEnum.FA
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
    ):
        try:
            from sglang.jit_kernel.flash_attention import flash_attn_varlen_func

            cu_query = F.pad(torch.cumsum(query_lens, dim=0), (1, 0)).to(torch.int32)
            cu_key = F.pad(torch.cumsum(key_lens, dim=0), (1, 0)).to(torch.int32)
            major, _ = torch.cuda.get_device_capability(query.device)
            return flash_attn_varlen_func(
                query,
                key,
                value,
                cu_query,
                cu_key,
                max_seqlen_q=int(query_lens.max().item()),
                max_seqlen_k=int(key_lens.max().item()),
                causal=causal,
                ver=4 if major >= 10 else 3,
            )
        except (ImportError, NotImplementedError, RuntimeError, TypeError) as error:
            logger.warning(
                "BAGEL FlashAttention unavailable; falling back to SDPA: %s", error
            )

    return _sdpa_attention(query, key, value, causal)


class _BagelRMSNorm(nn.Module):
    """Qwen2 RMSNorm with the official cast and multiply ordering."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        normalized = hidden_states.float()
        variance = normalized.pow(2).mean(-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class _BagelMoTAttention(nn.Module):
    def __init__(
        self,
        config: BagelDiTArchConfig,
        layer_index: int,
        attention_backend: AttentionBackendEnum,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        query_size = config.num_attention_heads * config.attention_head_dim
        kv_size = config.num_key_value_heads * config.attention_head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.attention_head_dim
        self.layer_index = layer_index
        self.attention_backend = attention_backend
        # DenoisingStage discovers the selected backend by scanning child
        # modules for this conventional attribute.
        self.backend = attention_backend

        self.und_q_proj = nn.Linear(hidden_size, query_size, bias=True)
        self.und_k_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.und_v_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.und_o_proj = nn.Linear(query_size, hidden_size, bias=False)
        self.und_q_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.und_k_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.gen_q_proj = nn.Linear(hidden_size, query_size, bias=True)
        self.gen_k_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.gen_v_proj = nn.Linear(hidden_size, kv_size, bias=True)
        self.gen_o_proj = nn.Linear(query_size, hidden_size, bias=False)
        self.gen_q_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.gen_k_norm = _BagelRMSNorm(self.head_dim, eps=config.rms_norm_eps)

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
        return (
            self.und_q_norm(query),
            self.und_k_norm(key),
            value,
        )

    def _project_generation(
        self, hidden_states: Tensor, text_indexes: Tensor, latent_indexes: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
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
            query[text_indexes] = self.und_q_norm(text_query)
            key[text_indexes] = self.und_k_norm(text_key)
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
        query[latent_indexes] = self.gen_q_norm(latent_query)
        key[latent_indexes] = self.gen_k_norm(latent_key)
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
        query_lens = torch.tensor(
            [query_length], dtype=torch.int32, device=hidden_states.device
        )
        prefix_length = int(prefix_lens[0].item())
        prefix_key = prefix_cache.key_cache[self.layer_index]
        prefix_value = prefix_cache.value_cache[self.layer_index]

        if prefix_key is None:
            merged_key = key
            merged_value = value
        else:
            if prefix_value is None or prefix_key.shape[0] != prefix_length:
                raise ValueError("inconsistent BAGEL prefix KV cache")
            merged_key = torch.cat((prefix_key, key), dim=0)
            merged_value = torch.cat((prefix_value, value), dim=0)
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
        output = hidden_states.new_zeros(hidden_states.shape)
        if text_indexes.numel() > 0:
            output[text_indexes] = self.und_o_proj(attended[text_indexes])
        output[latent_indexes] = self.gen_o_proj(attended[latent_indexes])
        return output, prefix_cache


class _BagelMoTMLP(nn.Module):
    def __init__(self, config: BagelDiTArchConfig) -> None:
        super().__init__()
        self.und_gate = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.und_up = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.und_down = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.gen_gate = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gen_up = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.gen_down = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    @staticmethod
    def _expert(
        hidden_states: Tensor, gate: nn.Linear, up: nn.Linear, down: nn.Linear
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
    ) -> None:
        super().__init__()
        self.und_in_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gen_in_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = _BagelMoTAttention(config, layer_index, attention_backend)
        self.und_post_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gen_post_norm = _BagelRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = _BagelMoTMLP(config)

    @staticmethod
    def _route_norm(
        hidden_states: Tensor,
        text_indexes: Tensor,
        latent_indexes: Tensor,
        *,
        mode: str,
        und_norm: _BagelRMSNorm,
        gen_norm: _BagelRMSNorm,
    ) -> Tensor:
        if mode == "und":
            return und_norm(hidden_states)
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
    """Stateless BAGEL T2I denoiser with request-owned prefix caches.

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
            raise ValueError("BAGEL T2I supports unquantized BF16 weights only")
        super().__init__(config, hf_config or {})
        arch = config.arch_config
        selected_attention_backend = self._resolve_attention_backend(attention_backend)

        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.num_layers = arch.num_hidden_layers
        self.latent_patch_size = arch.latent_patch_size
        self.latent_channel = arch.latent_channel
        self.latent_downsample = arch.latent_downsample
        self.max_latent_size = arch.max_latent_size

        self.embed_tokens = nn.Embedding(arch.vocab_size, arch.hidden_size)
        self.rotary_emb = _Qwen2RotaryEmbedding(
            arch.attention_head_dim, arch.rope_theta
        )
        self.layers = nn.ModuleList(
            [
                _BagelMoTLayer(arch, index, selected_attention_backend)
                for index in range(arch.num_hidden_layers)
            ]
        )
        self.und_final_norm = _BagelRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.gen_final_norm = _BagelRMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)

        patch_dim = arch.latent_patch_size**2 * arch.latent_channel
        self.vae2llm = nn.Linear(patch_dim, arch.hidden_size)
        self.llm2vae = nn.Linear(arch.hidden_size, patch_dim)
        self.time_embedder = _BagelTimestepEmbedder(
            arch.hidden_size,
            frequency_embedding_size=arch.timestep_frequency_embedding_size,
        )
        self.latent_pos_embed = _LoadedPositionTable(
            arch.latent_position_embedding_rows, arch.hidden_size
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
        input_ids = input_ids.to(device=device, dtype=torch.long)
        cache = BagelKVCache.empty(self.num_layers)
        prefix_lens = torch.zeros(1, dtype=torch.int32, device=device)
        if input_ids.numel() == 0:
            return BagelPrefixContext(cache, prefix_lens, 0)

        hidden_states = self.embed_tokens(input_ids)
        sequence_length = int(input_ids.numel())
        position_ids = torch.arange(sequence_length, device=device)
        cosine, sine = self.rotary_emb(hidden_states, position_ids)
        text_indexes = torch.arange(sequence_length, device=device)
        latent_indexes = torch.empty(0, dtype=torch.long, device=device)

        for layer in self.layers:
            hidden_states, cache = layer(
                hidden_states,
                cosine,
                sine,
                text_indexes,
                latent_indexes,
                mode="und",
                prefix_cache=cache,
                prefix_lens=prefix_lens,
                update_cache=True,
                causal=True,
            )
        prefix_lens = torch.tensor([sequence_length], dtype=torch.int32, device=device)
        return BagelPrefixContext(cache, prefix_lens, sequence_length)

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

    def forward(
        self,
        hidden_states: Tensor,
        timestep: Tensor,
        encoder_hidden_states: Tensor | list[Tensor] | None = None,
        *,
        bagel_context: BagelContext,
        guidance_scale: float | Tensor = 4.0,
        cfg_interval: tuple[float, float] = (0.4, 1.0),
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        guidance: Tensor | None = None,
        **_: object,
    ) -> Tensor:
        """Predict flow velocity with BAGEL's internal two-way CFG.

        Args:
            hidden_states: Patchified latents shaped ``[tokens, patch_dim]`` or
                ``[1, tokens, patch_dim]``.
            timestep: Scalar, length-one, or per-token raw flow sigma.
            encoder_hidden_states: Unused standard DiT compatibility argument.
            bagel_context: Explicit request-owned prefix and image state.
            guidance_scale: Text classifier-free guidance strength.
            cfg_interval: Open/closed sigma interval ``(low, high]``.
            cfg_renorm_min: Lower clamp for BAGEL's CFG norm correction.
            cfg_renorm_type: ``global`` or ``channel``.
            guidance: Unused standard embedded-guidance argument.

        Returns:
            Flow prediction with the same rank and shape as ``hidden_states``.

        Raises:
            ValueError: If batched/dynamic input or context geometry is invalid.
        """
        del encoder_hidden_states, guidance
        had_batch_dimension = hidden_states.ndim == 3
        if had_batch_dimension:
            if hidden_states.shape[0] != 1:
                raise ValueError("BAGEL T2I does not support dynamic batching")
            hidden_states = hidden_states[0]
        if hidden_states.ndim != 2:
            raise ValueError(
                "BAGEL expects [tokens, patch_dim] latents, "
                f"got shape {tuple(hidden_states.shape)}"
            )
        self._validate_hidden_states(hidden_states, bagel_context)

        timestep = self._normalize_timestep(
            timestep, hidden_states.shape[0], hidden_states.device
        )
        conditional = self._generation_step(
            hidden_states,
            timestep,
            bagel_context.conditional_kv,
            bagel_context.conditional_kv_lens,
            bagel_context.conditional_rope_offset,
            bagel_context,
        )

        scale = self._as_float(guidance_scale)
        sigma = float(timestep[0].item())
        cfg_enabled = cfg_interval[0] < sigma <= cfg_interval[1] and scale > 1.0
        if cfg_enabled:
            unconditional = self._generation_step(
                hidden_states,
                timestep,
                bagel_context.unconditional_kv,
                bagel_context.unconditional_kv_lens,
                bagel_context.unconditional_rope_offset,
                bagel_context,
            )
            conditional = self._apply_cfg(
                conditional,
                unconditional,
                scale,
                renorm_min=cfg_renorm_min,
                renorm_type=cfg_renorm_type,
            )
        # Official BAGEL keeps the Euler state in FP32 even though each model
        # evaluation runs in BF16. The standard FlowMatch scheduler preserves
        # the model-output dtype, so return FP32 velocity to retain that state.
        conditional = conditional.float()
        return conditional.unsqueeze(0) if had_batch_dimension else conditional

    def _generation_step(
        self,
        latents: Tensor,
        timestep: Tensor,
        prefix_cache: BagelKVCache,
        prefix_lens: Tensor,
        rope_offset: int,
        context: BagelContext,
    ) -> Tensor:
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

        rope_positions = torch.full(
            (sequence_length,), rope_offset, dtype=torch.long, device=device
        )
        cosine, sine = self.rotary_emb(packed, rope_positions)
        for layer in self.layers:
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
        expected_tokens = (context.height // self.latent_downsample) * (
            context.width // self.latent_downsample
        )
        expected_channels = self.latent_patch_size**2 * self.latent_channel
        if hidden_states.shape != (expected_tokens, expected_channels):
            raise ValueError(
                "BAGEL latent shape does not match request geometry: expected "
                f"{(expected_tokens, expected_channels)}, got "
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
        timestep: Tensor, token_count: int, device: torch.device
    ) -> Tensor:
        timestep = torch.as_tensor(timestep, device=device).reshape(-1)
        if timestep.numel() == 1:
            return timestep.expand(token_count)
        if timestep.numel() != token_count:
            raise ValueError(
                "BAGEL timestep must be scalar or per-token; "
                f"got {timestep.numel()} values for {token_count} tokens"
            )
        return timestep

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
        return guided * correction

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
        *,
        strict: bool = True,
    ) -> set[str]:
        """Stream the T2I subset of ``ema.safetensors`` into this model.

        Args:
            weights: Iterator of ``(checkpoint_name, tensor)`` pairs.
            strict: Enforce complete target coverage and classified source keys.

        Returns:
            Target parameter names populated from the iterator.

        Raises:
            ValueError: If required weights are missing, shapes mismatch, or a
                checkpoint key is outside the explicit non-T2I allowlist.
        """
        params = dict(self.named_parameters())
        required = set(params)
        loaded: set[str] = set()
        unexpected: list[str] = []
        skipped_prefixes = (
            "connector.",
            "vit_model.",
            "vit_pos_embed.",
            "language_model.lm_head.",
        )

        for source_name, tensor in weights:
            if source_name.startswith(skipped_prefixes):
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
                details.append(f"missing T2I weights: {missing}")
            if unexpected:
                details.append(f"unclassified checkpoint weights: {sorted(unexpected)}")
            raise ValueError("; ".join(details))
        return loaded

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
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"BAGEL weight shape mismatch for {name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
            )
        if parameter.is_meta:
            parent: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(tensor.to(dtype=parameter.dtype), requires_grad=False),
            )
            return

        weight_loader = getattr(parameter, "weight_loader", None)
        if weight_loader is not None:
            weight_loader(parameter, tensor)
        else:
            parameter.data.copy_(
                tensor.to(device=parameter.device, dtype=parameter.dtype)
            )


EntryClass = BagelTransformer
