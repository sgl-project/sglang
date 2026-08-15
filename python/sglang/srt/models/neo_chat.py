# SPDX-License-Identifier: Apache-2.0
"""Inference-only native SenseNova U1 language backbone."""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.neo_chat_mask import build_u1_hybrid_backend_mask
from sglang.srt.models.neo_chat_vision import NEOVisionModel
from sglang.srt.utils import add_prefix
from torch import nn

logger = logging.getLogger(__name__)

_U1_ROPE_CACHE: dict[tuple[int, int, int], nn.Module] = {}


class _U1RotaryCache(nn.Module):
    """CPU-built fp32 RoPE cache matching the public U1 implementation."""

    def __init__(self, head_size: int, max_position: int, base: int) -> None:
        super().__init__()
        frequencies = torch.arange(0, head_size, 2, dtype=torch.float32)
        inverse_frequencies = 1.0 / (base ** (frequencies / head_size))
        positions = torch.arange(max_position, dtype=torch.float32)
        angles = torch.einsum("i,j->ij", positions, inverse_frequencies)
        cache = torch.cat([angles.cos(), angles.sin()], dim=-1)
        cache = cache.to(torch.get_default_device())
        self.register_buffer("cos_sin_cache", cache, persistent=False)


def _get_u1_rope(
    head_size: int,
    *,
    max_position: int,
    base: int,
) -> nn.Module:
    key = (head_size, max_position, base)
    rotary_cache = _U1_ROPE_CACHE.get(key)
    default_device = torch.get_default_device()
    if (
        rotary_cache is None
        or rotary_cache.cos_sin_cache.device != default_device
        or rotary_cache.cos_sin_cache.device.type == "meta"
    ):
        rotary_cache = _U1RotaryCache(head_size, max_position, base)
        _U1_ROPE_CACHE[key] = rotary_cache
    return rotary_cache


def _rms_norm(norm: RMSNorm, hidden_states: torch.Tensor) -> torch.Tensor:
    """Match the public U1 RMSNorm cast order."""

    original_dtype = hidden_states.dtype
    states_fp32 = hidden_states.float()
    variance = states_fp32.pow(2).mean(dim=-1, keepdim=True)
    states_fp32 = states_fp32 * torch.rsqrt(variance + norm.variance_epsilon)
    return norm.weight.to(dtype=original_dtype) * states_fp32.to(original_dtype)


def _stacked_weight_target(name: str) -> tuple[str, str | None]:
    mappings = (
        (".q_proj_mot_gen.", ".qkv_proj_mot_gen.", "q"),
        (".k_proj_mot_gen.", ".qkv_proj_mot_gen.", "k"),
        (".v_proj_mot_gen.", ".qkv_proj_mot_gen.", "v"),
        (".q_proj.", ".qkv_proj.", "q"),
        (".k_proj.", ".qkv_proj.", "k"),
        (".v_proj.", ".qkv_proj.", "v"),
    )
    for source, target, shard_id in mappings:
        if source in name:
            return name.replace(source, target), shard_id
    return name, None


def _apply_u1_rope(
    rotary_embedding,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.flatten().to(rotary_embedding.cos_sin_cache.device)
    cache = rotary_embedding.cos_sin_cache.index_select(0, positions)
    cos, sin = cache.chunk(2, dim=-1)

    def apply(hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0],
            -1,
            head_size,
        )
        local_cos = cos.unsqueeze(1).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        local_sin = sin.unsqueeze(1).to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        first_half, second_half = hidden_states.chunk(2, dim=-1)
        hidden_states = torch.cat(
            [
                first_half * local_cos - second_half * local_sin,
                second_half * local_cos + first_half * local_sin,
            ],
            dim=-1,
        )
        return hidden_states.reshape(original_shape)

    return apply(query), apply(key)


class NEOChatMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_proj", prefix),
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(hidden_states)
        up, _ = self.up_proj(hidden_states)
        hidden_states = F.silu(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class NEOChatAttention(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.t_dim = self.head_dim // 2
        self.hw_dim = self.head_dim // 4
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.qkv_proj_mot_gen = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj_mot_gen", prefix),
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.o_proj_mot_gen = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj_mot_gen", prefix),
        )

        self.q_norm = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_hw = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.q_norm_hw_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_hw = RMSNorm(self.t_dim, eps=config.rms_norm_eps)
        self.k_norm_hw_mot_gen = RMSNorm(self.t_dim, eps=config.rms_norm_eps)

        self.rotary_emb_t = _get_u1_rope(
            self.t_dim,
            max_position=config.max_position_embeddings,
            base=int(config.rope_theta),
        )
        self.rotary_emb_hw = _get_u1_rope(
            self.hw_dim,
            max_position=config.max_position_embeddings_hw,
            base=int(config.rope_theta_hw),
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projection = self.qkv_proj_mot_gen if use_mot_gen else self.qkv_proj
        # Separate GEMMs preserve the public U1 eager arithmetic more closely than
        # a fused QKV GEMM while retaining SGLang's packed weight layout.
        q_weight, k_weight, v_weight = projection.weight.split(
            [self.q_size, self.kv_size, self.kv_size],
            dim=0,
        )
        q = F.linear(hidden_states, q_weight)
        k = F.linear(hidden_states, k_weight)
        v = F.linear(hidden_states, v_weight)

        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q_t, q_hw = q.split([self.t_dim, self.t_dim], dim=-1)
        k_t, k_hw = k.split([self.t_dim, self.t_dim], dim=-1)
        if use_mot_gen:
            q_t = _rms_norm(self.q_norm_mot_gen, q_t)
            q_hw = _rms_norm(self.q_norm_hw_mot_gen, q_hw)
            k_t = _rms_norm(self.k_norm_mot_gen, k_t)
            k_hw = _rms_norm(self.k_norm_hw_mot_gen, k_hw)
        else:
            q_t = _rms_norm(self.q_norm, q_t)
            q_hw = _rms_norm(self.q_norm_hw, q_hw)
            k_t = _rms_norm(self.k_norm, k_t)
            k_hw = _rms_norm(self.k_norm_hw, k_hw)

        q_h, q_w = q_hw.split([self.hw_dim, self.hw_dim], dim=-1)
        k_h, k_w = k_hw.split([self.hw_dim, self.hw_dim], dim=-1)
        return (
            torch.cat([q_t, q_h, q_w], dim=-1),
            torch.cat([k_t, k_h, k_w], dim=-1),
            v,
        )

    def _apply_split_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_t, q_h, q_w = q.split([self.t_dim, self.hw_dim, self.hw_dim], dim=-1)
        k_t, k_h, k_w = k.split([self.t_dim, self.hw_dim, self.hw_dim], dim=-1)
        q_t, k_t = _apply_u1_rope(
            self.rotary_emb_t,
            indexes[0],
            q_t,
            k_t,
            head_size=self.t_dim,
        )
        q_h, k_h = _apply_u1_rope(
            self.rotary_emb_hw,
            indexes[1],
            q_h,
            k_h,
            head_size=self.hw_dim,
        )
        q_w, k_w = _apply_u1_rope(
            self.rotary_emb_hw,
            indexes[2],
            q_w,
            k_w,
            head_size=self.hw_dim,
        )
        q = torch.cat([q_t, q_h, q_w], dim=-1).reshape(-1, self.q_size)
        k = torch.cat([k_t, k_h, k_w], dim=-1).reshape(-1, self.kv_size)
        return q, k

    def _qkv(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = self._project_qkv(
            hidden_states,
            use_mot_gen=use_mot_gen,
        )
        q, k = self._apply_split_rope(q, k, indexes)
        return q, k, v.reshape(-1, self.kv_size)

    def _output_projection(
        self,
        hidden_states: torch.Tensor,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        projection = self.o_proj_mot_gen if use_mot_gen else self.o_proj
        hidden_states, _ = projection(hidden_states)
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch,
        image_gen_indicators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_gen_indicators is None or not image_gen_indicators.any():
            q, k, v = self._qkv(
                hidden_states,
                indexes,
                use_mot_gen=False,
            )
            return self._output_projection(
                self.attn(q, k, v, forward_batch),
                use_mot_gen=False,
            )

        if image_gen_indicators.all():
            q, k, v = self._qkv(
                hidden_states,
                indexes,
                use_mot_gen=True,
            )
            return self._output_projection(
                self.attn(q, k, v, forward_batch),
                use_mot_gen=True,
            )

        q_understanding, k_understanding, v_understanding = self._qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        q_generation, k_generation, v_generation = self._qkv(
            hidden_states,
            indexes,
            use_mot_gen=True,
        )
        token_mask = image_gen_indicators.reshape(-1, 1)
        q = torch.where(token_mask, q_generation, q_understanding)
        k = torch.where(token_mask, k_generation, k_understanding)
        v = torch.where(token_mask, v_generation, v_understanding)
        attention_output = self.attn(q, k, v, forward_batch)
        understanding_output = self._output_projection(
            attention_output,
            use_mot_gen=False,
        )
        generation_output = self._output_projection(
            attention_output,
            use_mot_gen=True,
        )
        return torch.where(
            token_mask,
            generation_output,
            understanding_output,
        )


class NEOChatDecoderLayer(nn.Module):
    def __init__(
        self,
        config,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = NEOChatAttention(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = NEOChatMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.mlp_mot_gen = NEOChatMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp_mot_gen", prefix),
        )
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.input_layernorm_mot_gen = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm_mot_gen = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def _forward_one_path(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        input_norm = (
            self.input_layernorm_mot_gen if use_mot_gen else self.input_layernorm
        )
        hidden_states = _rms_norm(input_norm, hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            indexes,
            forward_batch,
            image_gen_indicators=(
                torch.ones(
                    hidden_states.shape[0],
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
                if use_mot_gen
                else None
            ),
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        post_norm = (
            self.post_attention_layernorm_mot_gen
            if use_mot_gen
            else self.post_attention_layernorm
        )
        mlp = self.mlp_mot_gen if use_mot_gen else self.mlp
        hidden_states = mlp(_rms_norm(post_norm, hidden_states))
        return residual + hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch,
        image_gen_indicators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image_gen_indicators is None or not image_gen_indicators.any():
            return self._forward_one_path(
                hidden_states,
                indexes,
                forward_batch,
                use_mot_gen=False,
            )
        if image_gen_indicators.all():
            return self._forward_one_path(
                hidden_states,
                indexes,
                forward_batch,
                use_mot_gen=True,
            )

        token_mask = image_gen_indicators.reshape(-1, 1)
        residual = hidden_states
        understanding_states = _rms_norm(self.input_layernorm, hidden_states)
        generation_states = _rms_norm(
            self.input_layernorm_mot_gen,
            hidden_states,
        )
        hidden_states = torch.where(
            token_mask,
            generation_states,
            understanding_states,
        )
        hidden_states = self.self_attn(
            hidden_states,
            indexes,
            forward_batch,
            image_gen_indicators=image_gen_indicators,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        understanding_states = _rms_norm(
            self.post_attention_layernorm,
            hidden_states,
        )
        generation_states = _rms_norm(
            self.post_attention_layernorm_mot_gen,
            hidden_states,
        )
        hidden_states = torch.where(
            token_mask,
            generation_states,
            understanding_states,
        )
        understanding_output = self.mlp(hidden_states)
        generation_output = self.mlp_mot_gen(hidden_states)
        hidden_states = torch.where(
            token_mask,
            generation_output,
            understanding_output,
        )
        return residual + hidden_states


class NEOChatTextModel(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if get_pp_group().world_size != 1:
            raise NotImplementedError(
                "NEOChatModel pipeline parallelism is not yet supported."
            )
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.layers = nn.ModuleList(
            [
                NEOChatDecoderLayer(
                    config,
                    layer_id=layer_id,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{layer_id}", prefix),
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_mot_gen = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.is_mrope_enabled = True

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    @staticmethod
    def _indexes(
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.mrope_positions is not None:
            return forward_batch.mrope_positions
        flat_positions = positions.flatten()
        zeros = torch.zeros_like(flat_positions)
        return torch.stack([flat_positions, zeros, zeros], dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        indexes = self._indexes(positions, forward_batch)
        if image_gen_indicators is not None:
            image_gen_indicators = image_gen_indicators.flatten().bool()

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                indexes,
                forward_batch,
                image_gen_indicators=image_gen_indicators,
            )

        if image_gen_indicators is None or not image_gen_indicators.any():
            return _rms_norm(self.norm, hidden_states)
        if image_gen_indicators.all():
            return _rms_norm(self.norm_mot_gen, hidden_states)
        token_mask = image_gen_indicators.reshape(-1, 1)
        return torch.where(
            token_mask,
            _rms_norm(self.norm_mot_gen, hidden_states),
            _rms_norm(self.norm, hidden_states),
        )


class NEOChatForCausalLM(nn.Module):
    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.model = NEOChatTextModel(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        self.logits_processor = LogitsProcessor(config)
        self.is_mrope_enabled = True

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            image_gen_indicators=image_gen_indicators,
        )
        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )


class NEOChatModel(nn.Module):
    """Native SRT entry class for SenseNova U1."""

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = NEOChatForCausalLM(
            config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_model = NEOVisionModel(config.vision_config)
        self.is_mrope_enabled = True
        self.last_load_report: dict[str, object] | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def pad_input_ids(
        self,
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ) -> list[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(
        self,
        items: list[MultimodalDataItem],
    ) -> torch.Tensor:
        pixel_values = torch.cat([item.feature for item in items], dim=0)
        grid_hw = torch.cat([item.grid_hw for item in items], dim=0)
        return self.vision_model(pixel_values, grid_hw)

    @staticmethod
    def _request_image_tags(
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        image_tags = []
        for request_index, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            prefix_len = forward_batch.extend_prefix_lens_cpu[request_index]
            request_tag = torch.zeros(
                extend_len,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            mm_inputs = (
                None
                if forward_batch.mm_inputs is None
                else forward_batch.mm_inputs[request_index]
            )
            if mm_inputs is not None:
                for item in mm_inputs.mm_items:
                    if not item.is_image():
                        continue
                    for span_start, span_end in item.offsets:
                        overlap_start = max(span_start, prefix_len)
                        overlap_end = min(
                            span_end + 1,
                            prefix_len + extend_len,
                        )
                        if overlap_start < overlap_end:
                            request_tag[
                                overlap_start - prefix_len : overlap_end - prefix_len
                            ] = True
            image_tags.append(request_tag)
        return torch.cat(image_tags)

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        forward_batch.model_specific_states = None

        indexes = forward_batch.mrope_positions
        if indexes is None:
            positions = forward_batch.positions.flatten()
            zeros = torch.zeros_like(positions)
            indexes = torch.stack([positions, zeros, zeros], dim=0)
        else:
            indexes = indexes.clone()

        if forward_batch.forward_mode.is_decode():
            for request_index in range(forward_batch.batch_size):
                mm_inputs = (
                    None
                    if forward_batch.mm_inputs is None
                    else forward_batch.mm_inputs[request_index]
                )
                if mm_inputs is None:
                    indexes[1:, request_index] = 0
            forward_batch.mrope_positions = indexes
            forward_batch.model_specific_states = {
                "indexes": indexes,
                "image_gen_indicators": torch.zeros(
                    indexes.shape[1],
                    dtype=torch.bool,
                    device=indexes.device,
                ),
                "custom_mask": None,
                "mask_indptr": None,
            }
            return
        if not forward_batch.forward_mode.is_extend():
            return

        token_offset = 0
        for request_index, extend_len in enumerate(forward_batch.extend_seq_lens_cpu):
            mm_inputs = (
                None
                if forward_batch.mm_inputs is None
                else forward_batch.mm_inputs[request_index]
            )
            if mm_inputs is None:
                indexes[1:, token_offset : token_offset + extend_len] = 0
            token_offset += extend_len
        forward_batch.mrope_positions = indexes
        image_token_tag = self._request_image_tags(forward_batch)
        image_gen_indicators = torch.zeros_like(image_token_tag)
        custom_mask, mask_indptr = build_u1_hybrid_backend_mask(
            indexes,
            image_token_tag,
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_prefix_lens_cpu,
            force_custom_mask=forward_batch.contains_mm_inputs(),
        )
        forward_batch.model_specific_states = {
            "indexes": indexes,
            "image_token_tag": image_token_tag,
            "image_gen_indicators": image_gen_indicators,
            "custom_mask": custom_mask,
            "mask_indptr": mask_indptr,
        }

    @staticmethod
    def _install_hybrid_mask(forward_batch: ForwardBatch) -> None:
        states = forward_batch.model_specific_states
        if states is None or states["custom_mask"] is None:
            return
        metadata = get_attn_backend().forward_metadata
        metadata.custom_mask = states["custom_mask"]
        metadata.mask_indptr = states["mask_indptr"]

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        image_gen_indicators = None
        if forward_batch.model_specific_states is not None:
            self._install_hybrid_mask(forward_batch)
            image_gen_indicators = forward_batch.model_specific_states[
                "image_gen_indicators"
            ]

        if (
            not forward_batch.forward_mode.is_decode()
            and forward_batch.contains_mm_inputs()
        ):
            return general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.language_model,
                multimodal_model=self,
                positions=positions,
                image_gen_indicators=image_gen_indicators,
            )
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            image_gen_indicators=image_gen_indicators,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        params = dict(self.named_parameters())
        expected_language_params = {
            name for name in params if name.startswith("language_model.")
        }
        expected_vision_params = {
            name for name in params if name.startswith("vision_model.")
        }
        loaded_params: set[str] = set()
        loaded_checkpoint_tensors = 0
        skipped_non_language_tensors = 0
        unknown_language_weights: list[str] = []
        unknown_vision_weights: list[str] = []
        for name, loaded_weight in weights:
            if name.startswith("vision_model.embeddings."):
                loaded_checkpoint_tensors += 1
                target_name = name.replace(
                    "vision_model.embeddings.",
                    "vision_model.",
                    1,
                )
                if target_name not in params:
                    unknown_vision_weights.append(name)
                    continue
                param = params[target_name]
                weight_loader = getattr(
                    param,
                    "weight_loader",
                    default_weight_loader,
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(target_name)
                continue
            if not name.startswith("language_model."):
                skipped_non_language_tensors += 1
                continue
            loaded_checkpoint_tensors += 1
            target_name, shard_id = _stacked_weight_target(name)
            if target_name not in params:
                if target_name.endswith(".bias"):
                    continue
                unknown_language_weights.append(name)
                continue
            param = params[target_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            if shard_id is None:
                weight_loader(param, loaded_weight)
            else:
                weight_loader(param, loaded_weight, shard_id)
            loaded_params.add(target_name)

        missing_language_params = sorted(expected_language_params - loaded_params)
        missing_vision_params = sorted(expected_vision_params - loaded_params)
        self.last_load_report = {
            "loaded_checkpoint_tensors": loaded_checkpoint_tensors,
            "loaded_native_parameters": len(loaded_params),
            "skipped_non_language_tensors": skipped_non_language_tensors,
            "missing_language_parameters": missing_language_params,
            "missing_vision_parameters": missing_vision_params,
            "unknown_language_weights": sorted(unknown_language_weights),
            "unknown_vision_weights": sorted(unknown_vision_weights),
        }
        if (
            missing_language_params
            or missing_vision_params
            or unknown_language_weights
            or unknown_vision_weights
        ):
            raise RuntimeError(
                f"NEOChatModel weight load is incomplete: {self.last_load_report}"
            )
        logger.info("NEOChatModel weight load report: %s", self.last_load_report)
        return loaded_params


EntryClass = NEOChatModel


__all__ = [
    "EntryClass",
    "NEOChatModel",
    "_stacked_weight_target",
]
