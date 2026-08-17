# SPDX-License-Identifier: Apache-2.0
"""Inference-only native SenseNova U1 language backbone."""

from __future__ import annotations

import base64
import logging
import os
import time
from collections import OrderedDict
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F
from sglang.srt.distributed import get_pp_group
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    embed_mm_inputs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.neo_chat_flow import (
    NEOChatFlowModules,
    apply_u1_time_schedule,
    build_u1_flow_batch_layout,
    compute_u1_noise_scale,
    patchify_images,
    unpatchify_images,
)
from sglang.srt.models.neo_chat_limits import (
    U1_EXACT_TEXT_CUSTOM_PARAM,
    U1_FLOW_CUSTOM_PARAM,
    validate_u1_flow_steps,
    validate_u1_image_size,
)
from sglang.srt.models.neo_chat_mask import (
    build_u1_hybrid_allowed_matrix,
    build_u1_hybrid_backend_mask,
)
from sglang.srt.models.neo_chat_exact_text import U1ExactTextRuntime
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


def _compiled_u1_rms_norm_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    original_dtype = hidden_states.dtype
    states_fp32 = hidden_states.float()
    variance = states_fp32.pow(2).mean(dim=-1, keepdim=True)
    states_fp32 = states_fp32 * torch.rsqrt(variance + epsilon)
    return weight.to(dtype=original_dtype) * states_fp32.to(original_dtype)


def _compiled_u1_add_rms_norm_impl(
    residual: torch.Tensor,
    update: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden_states = residual + update
    return hidden_states, _compiled_u1_rms_norm_impl(
        hidden_states,
        weight,
        epsilon,
    )


def _compiled_u1_rope_impl(
    cache: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positions = positions.flatten().to(cache.device)
    local_cache = cache.index_select(0, positions)
    cos, sin = local_cache.chunk(2, dim=-1)

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


def _compiled_u1_silu_mul_impl(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    return F.silu(gate) * up


_compiled_u1_rms_norm = torch.compile(
    _compiled_u1_rms_norm_impl,
    fullgraph=True,
    dynamic=False,
)
_compiled_u1_add_rms_norm = torch.compile(
    _compiled_u1_add_rms_norm_impl,
    fullgraph=True,
    dynamic=False,
)
_compiled_u1_rope = torch.compile(
    _compiled_u1_rope_impl,
    fullgraph=True,
    dynamic=False,
)
_compiled_u1_silu_mul = torch.compile(
    _compiled_u1_silu_mul_impl,
    fullgraph=True,
    dynamic=False,
)
_USE_COMPILED_MOT_GEN_NORM = os.environ.get(
    "SENSENOVA_U1_COMPILED_MOT_GEN_NORM",
    "",
).lower() in {"1", "true", "yes", "on"}
_COMPILED_MOT_GEN_ACTIVE: ContextVar[bool] = ContextVar(
    "_COMPILED_MOT_GEN_ACTIVE",
    default=False,
)


def _use_compiled_mot_gen() -> bool:
    return _USE_COMPILED_MOT_GEN_NORM and _COMPILED_MOT_GEN_ACTIVE.get()


@contextmanager
def _compiled_mot_gen_scope(enabled: bool):
    token = _COMPILED_MOT_GEN_ACTIVE.set(enabled)
    try:
        yield
    finally:
        _COMPILED_MOT_GEN_ACTIVE.reset(token)


def _mot_gen_rms_norm(
    norm: RMSNorm,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if not _use_compiled_mot_gen():
        return _rms_norm(norm, hidden_states)
    return _compiled_u1_rms_norm(
        hidden_states,
        norm.weight,
        norm.variance_epsilon,
    )


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


def _flow_weight_target(name: str) -> str:
    if name.startswith("fm_modules.vision_model_mot_gen.embeddings."):
        return name.replace(
            "fm_modules.vision_model_mot_gen.embeddings.",
            "fm_modules.vision_model_mot_gen.",
            1,
        )
    return name


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
        compiled_mot_gen: bool = False,
    ) -> None:
        super().__init__()
        self.compiled_mot_gen = compiled_mot_gen
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
        hidden_states = (
            _compiled_u1_silu_mul(gate, up)
            if self.compiled_mot_gen and _use_compiled_mot_gen()
            else F.silu(gate) * up
        )
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
            q_t = _mot_gen_rms_norm(self.q_norm_mot_gen, q_t)
            q_hw = _mot_gen_rms_norm(self.q_norm_hw_mot_gen, q_hw)
            k_t = _mot_gen_rms_norm(self.k_norm_mot_gen, k_t)
            k_hw = _mot_gen_rms_norm(self.k_norm_hw_mot_gen, k_hw)
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
        *,
        use_mot_gen: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_t, q_h, q_w = q.split([self.t_dim, self.hw_dim, self.hw_dim], dim=-1)
        k_t, k_h, k_w = k.split([self.t_dim, self.hw_dim, self.hw_dim], dim=-1)
        if use_mot_gen and _use_compiled_mot_gen():
            q_t, k_t = _compiled_u1_rope(
                self.rotary_emb_t.cos_sin_cache,
                indexes[0],
                q_t,
                k_t,
                self.t_dim,
            )
            q_h, k_h = _compiled_u1_rope(
                self.rotary_emb_hw.cos_sin_cache,
                indexes[1],
                q_h,
                k_h,
                self.hw_dim,
            )
            q_w, k_w = _compiled_u1_rope(
                self.rotary_emb_hw.cos_sin_cache,
                indexes[2],
                q_w,
                k_w,
                self.hw_dim,
            )
        else:
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
        q, k = self._apply_split_rope(
            q,
            k,
            indexes,
            use_mot_gen=use_mot_gen,
        )
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

    def _official_eager_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool,
        allowed: torch.Tensor | None = None,
        kv_already_repeated: bool = False,
    ) -> torch.Tensor:
        q_len = q.shape[0]
        k_len = k.shape[0]
        q = q.view(q_len, self.total_num_heads, self.head_dim)
        cache_heads = (
            self.total_num_heads if kv_already_repeated else self.total_num_kv_heads
        )
        k = k.view(k_len, cache_heads, self.head_dim)
        v = v.view(k_len, cache_heads, self.head_dim)
        if not kv_already_repeated:
            repeat = self.total_num_heads // self.total_num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        q_bhsd = q.transpose(0, 1).unsqueeze(0)
        k_bhsd = k.transpose(0, 1).unsqueeze(0)
        v_bhsd = v.transpose(0, 1).unsqueeze(0)
        attention_weights = torch.matmul(q_bhsd, k_bhsd.transpose(2, 3)) * self.scaling

        mask = None
        if allowed is not None:
            allowed = allowed.to(device=attention_weights.device, dtype=torch.bool)
            mask = torch.where(
                allowed.view(1, 1, q_len, k_len),
                torch.zeros(
                    (),
                    device=attention_weights.device,
                    dtype=attention_weights.dtype,
                ),
                torch.full(
                    (),
                    float("-inf"),
                    device=attention_weights.device,
                    dtype=attention_weights.dtype,
                ),
            )
        elif causal:
            if q_len != k_len:
                raise ValueError(
                    "causal eager attention requires equal query and key lengths"
                )
            allowed_causal = torch.ones(
                (q_len, k_len),
                device=attention_weights.device,
                dtype=torch.bool,
            ).tril()
            mask = torch.where(
                allowed_causal.view(1, 1, q_len, k_len),
                torch.zeros(
                    (),
                    device=attention_weights.device,
                    dtype=attention_weights.dtype,
                ),
                torch.full(
                    (),
                    float("-inf"),
                    device=attention_weights.device,
                    dtype=attention_weights.dtype,
                ),
            )
        if mask is not None:
            attention_weights = attention_weights + mask

        attention_weights = F.softmax(
            attention_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(q_bhsd.dtype)
        output = torch.matmul(attention_weights, v_bhsd)
        return output.transpose(1, 2).contiguous().reshape(q_len, self.q_size)

    def repeat_eager_kv_cache(self, states: torch.Tensor) -> torch.Tensor:
        seq_len = int(states.shape[0])
        states = states.view(
            seq_len,
            self.total_num_kv_heads,
            self.head_dim,
        )
        repeat = self.total_num_heads // self.total_num_kv_heads
        return states.repeat_interleave(repeat, dim=1).reshape(
            seq_len,
            self.q_size,
        )

    def _forward_one_path(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        use_mot_gen: bool,
    ) -> torch.Tensor:
        q, k, v = self._qkv(
            hidden_states,
            indexes,
            use_mot_gen=use_mot_gen,
        )
        return self._output_projection(
            self.attn(q, k, v, forward_batch),
            use_mot_gen=use_mot_gen,
        )

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
        self.layer_id = int(layer_id)
        self.exact_compiled_add_rms = False
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
            compiled_mot_gen=True,
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

    def _eager_post_attention_norm(
        self,
        residual: torch.Tensor,
        update: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.exact_compiled_add_rms and 30 <= self.layer_id <= 41:
            return _compiled_u1_add_rms_norm(
                residual,
                update,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.variance_epsilon,
            )
        hidden_states = residual + update
        return hidden_states, _rms_norm(
            self.post_attention_layernorm,
            hidden_states,
        )

    def eager_text_prefill_with_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        *,
        allowed: torch.Tensor | None,
        repeat_kv_cache: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k, v = self.self_attn._qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        attention_output = self.self_attn._official_eager_attention(
            q,
            k,
            v,
            causal=allowed is None,
            allowed=allowed,
        )
        attention_update = self.self_attn._output_projection(
            attention_output,
            use_mot_gen=False,
        )
        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        hidden_states = residual + hidden_states
        if repeat_kv_cache:
            k = self.self_attn.repeat_eager_kv_cache(k)
            v = self.self_attn.repeat_eager_kv_cache(v)
        return hidden_states, k.detach(), v.detach()

    def eager_text_decode_with_static_cache(
        self,
        hidden_states: torch.Tensor,
        indexes: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        *,
        cache_position: int,
        repeat_kv_cache: bool,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = _rms_norm(self.input_layernorm, hidden_states)
        q, k_current, v_current = self.self_attn._qkv(
            hidden_states,
            indexes,
            use_mot_gen=False,
        )
        if repeat_kv_cache:
            k_current = self.self_attn.repeat_eager_kv_cache(k_current)
            v_current = self.self_attn.repeat_eager_kv_cache(v_current)
        cache_k[cache_position : cache_position + 1].copy_(k_current)
        cache_v[cache_position : cache_position + 1].copy_(v_current)
        attention_output = self.self_attn._official_eager_attention(
            q,
            cache_k[: cache_position + 1],
            cache_v[: cache_position + 1],
            causal=False,
            kv_already_repeated=repeat_kv_cache,
        )
        attention_update = self.self_attn._output_projection(
            attention_output,
            use_mot_gen=False,
        )
        hidden_states, mlp_input = self._eager_post_attention_norm(
            residual,
            attention_update,
        )
        residual = hidden_states
        hidden_states = self.mlp(mlp_input)
        return residual + hidden_states

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
        hidden_states = (
            _mot_gen_rms_norm(input_norm, hidden_states)
            if use_mot_gen
            else _rms_norm(input_norm, hidden_states)
        )
        hidden_states = self.self_attn._forward_one_path(
            hidden_states,
            indexes,
            forward_batch,
            use_mot_gen=use_mot_gen,
        )
        post_norm = (
            self.post_attention_layernorm_mot_gen
            if use_mot_gen
            else self.post_attention_layernorm
        )
        mlp = self.mlp_mot_gen if use_mot_gen else self.mlp
        if use_mot_gen and _use_compiled_mot_gen():
            hidden_states, mlp_input = _compiled_u1_add_rms_norm(
                residual,
                hidden_states,
                post_norm.weight,
                post_norm.variance_epsilon,
            )
        else:
            hidden_states = residual + hidden_states
            mlp_input = _rms_norm(post_norm, hidden_states)
        residual = hidden_states
        hidden_states = mlp(mlp_input)
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
        generation_states = _mot_gen_rms_norm(
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
        generation_states = _mot_gen_rms_norm(
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
        self.prefill_cuda_graph_capture_variant = "sensenova_u1_flow"
        self.prefill_cuda_graph_capture_flag = "force_mot_gen_for_prefill_graph_capture"
        self.force_mot_gen_for_prefill_graph_capture = False

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

    def eager_text_prefill_with_cache(
        self,
        input_ids: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None,
        indexes: torch.Tensor,
        image_token_tag: torch.Tensor,
        repeat_kv_cache: bool,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = (
            self.embed_tokens(input_ids) if input_embeds is None else input_embeds
        )
        allowed = None
        if bool(image_token_tag.any().item()):
            allowed = build_u1_hybrid_allowed_matrix(
                indexes[0],
                image_token_tag.reshape(-1).to(
                    device=indexes.device,
                    dtype=torch.bool,
                ),
            )

        caches = []
        for layer in self.layers:
            hidden_states, k, v = layer.eager_text_prefill_with_cache(
                hidden_states,
                indexes,
                allowed=allowed,
                repeat_kv_cache=repeat_kv_cache,
            )
            caches.append((k, v))
        return _rms_norm(self.norm, hidden_states), caches

    def eager_text_decode_with_static_cache(
        self,
        input_ids: torch.Tensor,
        *,
        indexes: torch.Tensor,
        caches: list[tuple[torch.Tensor, torch.Tensor]],
        cache_position: int,
        repeat_kv_cache: bool,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        if len(caches) != len(self.layers):
            raise ValueError("eager text cache layer count mismatch")
        for layer, (cache_k, cache_v) in zip(
            self.layers,
            caches,
            strict=True,
        ):
            hidden_states = layer.eager_text_decode_with_static_cache(
                hidden_states,
                indexes,
                cache_k,
                cache_v,
                cache_position=cache_position,
                repeat_kv_cache=repeat_kv_cache,
            )
        return _rms_norm(self.norm, hidden_states)

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
        if (
            self.force_mot_gen_for_prefill_graph_capture
            and image_gen_indicators is None
        ):
            with _compiled_mot_gen_scope(True):
                for layer in self.layers:
                    hidden_states = layer._forward_one_path(
                        hidden_states,
                        indexes,
                        forward_batch,
                        use_mot_gen=True,
                    )
                return _mot_gen_rms_norm(self.norm_mot_gen, hidden_states)
        if image_gen_indicators is not None:
            image_gen_indicators = image_gen_indicators.flatten().bool()

        all_generation = bool(
            image_gen_indicators is not None and image_gen_indicators.all()
        )
        with _compiled_mot_gen_scope(all_generation):
            for layer in self.layers:
                hidden_states = layer(
                    hidden_states,
                    indexes,
                    forward_batch,
                    image_gen_indicators=image_gen_indicators,
                )

            if image_gen_indicators is None or not image_gen_indicators.any():
                return _rms_norm(self.norm, hidden_states)
            if all_generation:
                return _mot_gen_rms_norm(self.norm_mot_gen, hidden_states)
            token_mask = image_gen_indicators.reshape(-1, 1)
            return torch.where(
                token_mask,
                _mot_gen_rms_norm(self.norm_mot_gen, hidden_states),
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
        self.exact_lm_head_linear = False

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def eager_text_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        last_hidden = hidden_states[-1:].to(self.lm_head.weight.dtype)
        if self.exact_lm_head_linear:
            return F.linear(last_hidden, self.lm_head.weight).to(hidden_states.dtype)
        return torch.matmul(last_hidden, self.lm_head.weight.T).to(hidden_states.dtype)

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

    decode_text_mrope_temporal_only = True
    _FLOW_TIMESTEP_CACHE_MAX_ENTRIES = 8

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
        self.fm_modules = NEOChatFlowModules(config)
        self.exact_text_runtime = U1ExactTextRuntime(self)
        self._flow_timestep_embed_cache: OrderedDict[
            tuple[object, ...],
            tuple[torch.Tensor, ...],
        ] = OrderedDict()
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
    def _flow_specs(forward_batch: ForwardBatch) -> list[dict] | None:
        sampling_info = forward_batch.sampling_info
        custom_params = None if sampling_info is None else sampling_info.custom_params
        if custom_params is None:
            return None

        specs = []
        for params in custom_params:
            spec = (
                None
                if not isinstance(params, dict)
                else params.get(U1_FLOW_CUSTOM_PARAM)
            )
            specs.append(spec)
        if not any(spec is not None for spec in specs):
            return None
        if any(spec is None for spec in specs):
            raise NotImplementedError(
                "SenseNova U1 flow requests cannot share a batch with text requests"
            )
        return specs

    @staticmethod
    def _exact_text_specs(
        forward_batch: ForwardBatch,
    ) -> list[dict] | None:
        sampling_info = forward_batch.sampling_info
        custom_params = None if sampling_info is None else sampling_info.custom_params
        if custom_params is None:
            return None
        specs = [
            (
                params.get(U1_EXACT_TEXT_CUSTOM_PARAM)
                if isinstance(params, dict)
                else None
            )
            for params in custom_params
        ]
        if not any(spec is not None for spec in specs):
            return None
        if any(not isinstance(spec, dict) for spec in specs):
            raise NotImplementedError(
                "SenseNova U1 exact text requests cannot share a batch with "
                "other request types"
            )
        return specs

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
        forward_batch.cross_attention_custom_mask = None

        flow_specs = (
            self._flow_specs(forward_batch)
            if forward_batch.forward_mode.is_extend()
            else None
        )
        if flow_specs is not None:
            indexes, image_gen_indicators = build_u1_flow_batch_layout(
                forward_batch.positions,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
                flow_specs,
            )
            image_token_tag = (
                self._request_image_tags(forward_batch) | image_gen_indicators
            )
            custom_mask, mask_indptr = build_u1_hybrid_backend_mask(
                indexes,
                image_token_tag,
                forward_batch.extend_seq_lens_cpu,
                forward_batch.extend_prefix_lens_cpu,
                force_custom_mask=True,
            )
            forward_batch.mrope_positions = indexes
            forward_batch.cross_attention_custom_mask = custom_mask
            forward_batch.model_specific_states = {
                "indexes": indexes,
                "image_token_tag": image_token_tag,
                "image_gen_indicators": image_gen_indicators,
                "custom_mask": custom_mask,
                "mask_indptr": mask_indptr,
                "flow_specs": flow_specs,
            }
            return

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
        forward_batch.cross_attention_custom_mask = custom_mask
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

    def _flow_timestep_embeds(
        self,
        *,
        timesteps: torch.Tensor,
        image_token_count: int,
        noise_scale: float,
    ) -> tuple[tuple[torch.Tensor, ...], bool]:
        parameter = next(self.fm_modules.parameters())
        cache_key = (
            parameter.device.type,
            parameter.device.index,
            str(parameter.dtype),
            image_token_count,
            tuple(float(value) for value in timesteps.detach().cpu().tolist()),
            float(noise_scale),
            bool(self.fm_modules.add_noise_scale_embedding),
        )
        cached = self._flow_timestep_embed_cache.pop(cache_key, None)
        if cached is not None:
            self._flow_timestep_embed_cache[cache_key] = cached
            return cached, True

        normalized_noise_scale = noise_scale / float(
            getattr(self.config, "noise_scale_max_value", 1.0)
        )
        values = []
        for timestep in timesteps[:-1]:
            expanded_timestep = timestep.expand(image_token_count)
            timestep_embeds = self.fm_modules.timestep_embedder(expanded_timestep)
            if self.fm_modules.add_noise_scale_embedding:
                timestep_embeds = (
                    timestep_embeds
                    + self.fm_modules.noise_scale_embedder(
                        torch.full_like(
                            expanded_timestep,
                            normalized_noise_scale,
                        )
                    )
                )
            values.append(timestep_embeds.detach())
        cached = tuple(values)
        self._flow_timestep_embed_cache[cache_key] = cached
        while (
            len(self._flow_timestep_embed_cache) > self._FLOW_TIMESTEP_CACHE_MAX_ENTRIES
        ):
            self._flow_timestep_embed_cache.popitem(last=False)
        return cached, False

    def _forward_flow(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        states = forward_batch.model_specific_states
        flow_specs = states["flow_specs"]
        if forward_batch.batch_size != 1:
            raise NotImplementedError(
                "SenseNova U1 bounded flow currently supports batch_size=1"
            )
        spec = flow_specs[0]
        image_gen_indicators = states["image_gen_indicators"]
        image_token_count = int(image_gen_indicators.sum().item())
        metadata = get_attn_backend().forward_metadata
        metadata_custom_mask = getattr(metadata, "custom_mask", None)
        expected_custom_mask = states["custom_mask"]
        if (
            metadata_custom_mask is None
            or expected_custom_mask is None
            or metadata_custom_mask.numel() != expected_custom_mask.numel()
        ):
            raise RuntimeError(
                "SenseNova U1 flow hybrid attention mask was not attached "
                "to backend metadata"
            )

        width, height = validate_u1_image_size(
            spec["width"],
            spec["height"],
        )
        num_steps = validate_u1_flow_steps(spec.get("num_steps", 2))
        seed = int(spec.get("seed", 0))
        logger.info(
            "SenseNova U1 bounded flow start: prefix_tokens=%d image_tokens=%d "
            "steps=%d size=%dx%d",
            forward_batch.extend_prefix_lens_cpu[0],
            image_token_count,
            num_steps,
            width,
            height,
        )

        patch_size = int(self.config.patch_size)
        merge_size = int(1 / float(self.config.downsample_ratio))
        token_height = height // (patch_size * merge_size)
        token_width = width // (patch_size * merge_size)
        grid_height = height // patch_size
        grid_width = width // patch_size
        if image_token_count != token_height * token_width:
            raise ValueError("SenseNova U1 flow image token count is inconsistent")

        parameter = next(self.fm_modules.parameters())
        device = parameter.device
        dtype = parameter.dtype
        noise_scale = compute_u1_noise_scale(
            grid_height=grid_height,
            grid_width=grid_width,
            merge_size=merge_size,
            noise_scale=float(getattr(self.config, "noise_scale", 1.0)),
            noise_scale_mode=str(getattr(self.config, "noise_scale_mode", "constant")),
            base_image_seq_len=int(
                getattr(self.config, "noise_scale_base_image_seq_len", 64)
            ),
            max_value=float(getattr(self.config, "noise_scale_max_value", 1.0)),
        )
        generator = torch.Generator(device=device).manual_seed(seed)
        image_prediction = noise_scale * torch.randn(
            (1, 3, height, width),
            device=device,
            dtype=dtype,
            generator=generator,
        )
        initial_prediction = image_prediction.clone()
        timesteps = torch.linspace(
            0.0,
            1.0,
            num_steps + 1,
            device=device,
        )
        if bool(spec.get("enable_timestep_shift", True)):
            timesteps = apply_u1_time_schedule(
                timesteps,
                image_seq_len=image_token_count,
                timestep_shift=float(spec.get("timestep_shift", 1.0)),
                time_schedule=str(getattr(self.config, "time_schedule", "standard")),
                time_shift_type=str(
                    getattr(self.config, "time_shift_type", "exponential")
                ),
                base_shift=float(getattr(self.config, "base_shift", 0.5)),
                max_shift=float(getattr(self.config, "max_shift", 1.15)),
                base_image_seq_len=int(getattr(self.config, "base_image_seq_len", 64)),
                max_image_seq_len=int(getattr(self.config, "max_image_seq_len", 4096)),
            )
        timestep_embeds_by_step, timestep_cache_hit = self._flow_timestep_embeds(
            timesteps=timesteps,
            image_token_count=image_token_count,
            noise_scale=noise_scale,
        )

        grid_hw = torch.tensor(
            [[grid_height, grid_width]],
            dtype=torch.long,
            device=device,
        )
        all_image_tokens = image_token_count == input_ids.numel()
        base_input_embeds = None
        if not all_image_tokens:
            if forward_batch.contains_mm_inputs():
                mm_inputs_list = [
                    mm_input
                    for mm_input in forward_batch.mm_inputs
                    if mm_input is not None
                ]
                base_input_embeds, _ = embed_mm_inputs(
                    mm_inputs_list=mm_inputs_list,
                    extend_prefix_lens=forward_batch.extend_prefix_lens_cpu,
                    extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                    input_ids=input_ids,
                    input_embedding=self.get_input_embeddings(),
                    multimodal_model=self,
                )
            else:
                base_input_embeds = self.get_input_embeddings()(input_ids)
        final_hidden_states = None
        step_delta_l2_tensors = []
        profile_flow_stages = device.type == "cuda" and os.environ.get(
            "SENSENOVA_U1_PROFILE_FLOW_STAGES",
            "",
        ).lower() in {"1", "true", "yes", "on"}
        flow_stage_events = []
        flow_started = time.perf_counter()
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        for step_index in range(num_steps):
            if profile_flow_stages:
                step_events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
                step_events[0].record()
            timestep = timesteps[step_index]
            next_timestep = timesteps[step_index + 1]
            latent_patches = patchify_images(
                image_prediction,
                patch_size * merge_size,
            )
            image_patches = patchify_images(
                image_prediction,
                patch_size,
                channel_first=True,
            )
            image_embeds = self.fm_modules.vision_model_mot_gen(
                image_patches.reshape(grid_height * grid_width, -1),
                grid_hw,
            )
            timestep_embeds = timestep_embeds_by_step[step_index]
            image_embeds = image_embeds + timestep_embeds.to(image_embeds.dtype)
            if profile_flow_stages:
                step_events[1].record()

            if base_input_embeds is None:
                input_embeds = image_embeds
            else:
                input_embeds = base_input_embeds.clone()
                input_embeds[image_gen_indicators] = image_embeds
            final_hidden_states = self.language_model.model(
                input_ids,
                positions,
                forward_batch,
                input_embeds=input_embeds,
                image_gen_indicators=image_gen_indicators,
            )
            if profile_flow_stages:
                step_events[2].record()
            image_hidden_states = final_hidden_states[image_gen_indicators].reshape(
                1,
                image_token_count,
                -1,
            )
            image_prediction_target = self.fm_modules.fm_head(
                image_hidden_states
            ).reshape_as(latent_patches)
            velocity = (image_prediction_target - latent_patches) / (
                1 - timestep
            ).clamp_min(float(getattr(self.config, "t_eps", 0.05)))
            updated_patches = latent_patches + (next_timestep - timestep) * velocity
            updated_prediction = unpatchify_images(
                updated_patches,
                patch_size * merge_size,
                height,
                width,
            )
            step_delta_l2_tensors.append(
                (updated_prediction - image_prediction).float().norm()
            )
            image_prediction = updated_prediction
            if profile_flow_stages:
                step_events[3].record()
                flow_stage_events.append(step_events)

        assert final_hidden_states is not None
        total_delta_l2_tensor = (image_prediction - initial_prediction).float().norm()
        if end_event is not None:
            end_event.record()
            end_event.synchronize()
            flow_compute_seconds = start_event.elapsed_time(end_event) / 1000.0
        else:
            flow_compute_seconds = time.perf_counter() - flow_started
        step_delta_l2 = [
            float(value)
            for value in torch.stack(step_delta_l2_tensors).detach().cpu().tolist()
        ]
        total_delta_l2 = float(total_delta_l2_tensor.detach().cpu().item())
        flow_stage_seconds = []
        for stage_events in flow_stage_events:
            flow_stage_seconds.append(
                {
                    "vision_timestep": stage_events[0].elapsed_time(stage_events[1])
                    / 1000.0,
                    "transformer": stage_events[1].elapsed_time(stage_events[2])
                    / 1000.0,
                    "flow_head_update": stage_events[2].elapsed_time(stage_events[3])
                    / 1000.0,
                }
            )
        next_token_logits = torch.full(
            (1, self.config.llm_config.vocab_size),
            -torch.inf,
            dtype=torch.float32,
            device=device,
        )
        next_token_logits[0, 0] = 0
        output = LogitsProcessorOutput(next_token_logits=next_token_logits)
        final_image = image_prediction.detach().to(torch.float16).cpu().contiguous()
        image_b64 = (
            base64.b64encode(final_image.numpy().tobytes()).decode("ascii")
            if bool(spec.get("return_image_tensor", False))
            else None
        )
        output.customized_info = {
            "sensenova_u1_flow_steps": [num_steps],
            "sensenova_u1_flow_image_shape": [list(final_image.shape)],
            "sensenova_u1_flow_image_dtype": ["float16"],
            "sensenova_u1_flow_image_b64": [image_b64],
            "sensenova_u1_flow_noise_scale": [noise_scale],
            "sensenova_u1_flow_compute_seconds": [flow_compute_seconds],
            "sensenova_u1_flow_custom_mask_numel": [int(metadata_custom_mask.numel())],
            "sensenova_u1_flow_step_delta_l2": [step_delta_l2],
            "sensenova_u1_flow_total_delta_l2": [total_delta_l2],
            "sensenova_u1_flow_timestep_cache_hit": [timestep_cache_hit],
            "sensenova_u1_flow_timesteps": [
                [float(value) for value in timesteps.detach().cpu().tolist()]
            ],
        }
        if bool(spec.get("return_image_tensor_raw", False)):
            output.customized_info["sensenova_u1_flow_image_tensor"] = [final_image]
        if flow_stage_seconds:
            output.customized_info["sensenova_u1_flow_stage_seconds"] = [
                flow_stage_seconds
            ]
        logger.info(
            "SenseNova U1 bounded flow complete: steps=%d total_delta_l2=%.6f "
            "timestep_cache_hit=%s",
            num_steps,
            output.customized_info["sensenova_u1_flow_total_delta_l2"][0],
            timestep_cache_hit,
        )
        return output

    def _forward_exact_text(
        self,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        specs = self._exact_text_specs(forward_batch)
        if specs is None or forward_batch.batch_size != 1:
            raise NotImplementedError(
                "SenseNova U1 exact text currently requires batch_size=1"
            )
        spec = specs[0]
        decode_steps = int(spec["decode_steps"])
        if decode_steps <= 0:
            raise ValueError("SenseNova U1 exact text decode_steps must be positive")
        states = forward_batch.model_specific_states
        if states is None:
            raise RuntimeError("SenseNova U1 exact text metadata was not prepared")
        indexes = states["indexes"]
        image_token_tag = states["image_token_tag"]

        if forward_batch.contains_mm_inputs():
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            input_embeds, _ = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=forward_batch.extend_prefix_lens_cpu,
                extend_seq_lens=forward_batch.extend_seq_lens_cpu,
                input_ids=input_ids,
                input_embedding=self.get_input_embeddings(),
                multimodal_model=self,
            )
        else:
            input_embeds = None

        result = self.exact_text_runtime.generate(
            input_ids=input_ids,
            indexes=indexes,
            image_token_tag=image_token_tag,
            input_embeds=input_embeds,
            decode_steps=decode_steps,
            compiled_add_rms=bool(spec.get("compiled_add_rms", False)),
            lm_head_linear=bool(spec.get("lm_head_linear", False)),
        )
        terminal_ids = {
            int(spec["img_start_token_id"]),
            *[int(token_id) for token_id in spec.get("eos_token_ids", [])],
        }
        token_ids = []
        for token_id in result.token_ids:
            token_ids.append(int(token_id))
            if int(token_id) in terminal_ids:
                break
        if not token_ids:
            raise RuntimeError("SenseNova U1 exact text generated no token")

        logits = torch.full(
            (1, self.config.llm_config.vocab_size),
            -torch.inf,
            dtype=torch.float32,
            device=input_ids.device,
        )
        logits[0, token_ids[0]] = 0
        output = LogitsProcessorOutput(next_token_logits=logits)
        output.customized_info = {
            "sensenova_u1_exact_text_tail": [token_ids[1:]],
            "sensenova_u1_exact_text_stats": [
                {
                    "generated_tokens": len(token_ids),
                    "prefill_elapsed_s": result.prefill_elapsed_s,
                    "decode_elapsed_s": result.decode_elapsed_s,
                    "total_elapsed_s": result.total_elapsed_s,
                    "prefix_cache_hit": result.prefix_cache_hit,
                    "graph_created": result.graph_created,
                    "graph_replayed": result.graph_replayed,
                }
            ],
        }
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ):
        if (
            forward_batch.model_specific_states is None
            and forward_batch.forward_mode.is_extend()
            and (
                self._flow_specs(forward_batch) is not None
                or self._exact_text_specs(forward_batch) is not None
            )
        ):
            self.prepare_forward_batch(forward_batch)
        if (
            forward_batch.forward_mode.is_extend()
            and self._exact_text_specs(forward_batch) is not None
            and forward_batch.batch_size == 1
        ):
            return self._forward_exact_text(
                input_ids,
                forward_batch,
            )
        image_gen_indicators = None
        if forward_batch.model_specific_states is not None:
            self._install_hybrid_mask(forward_batch)
            image_gen_indicators = forward_batch.model_specific_states[
                "image_gen_indicators"
            ]
            if "flow_specs" in forward_batch.model_specific_states:
                return self._forward_flow(
                    input_ids,
                    positions,
                    forward_batch,
                )

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
        self.exact_text_runtime.clear()
        self._flow_timestep_embed_cache.clear()
        params = dict(self.named_parameters())
        expected_language_params = {
            name for name in params if name.startswith("language_model.")
        }
        expected_vision_params = {
            name for name in params if name.startswith("vision_model.")
        }
        expected_flow_params = {
            name for name in params if name.startswith("fm_modules.")
        }
        loaded_params: set[str] = set()
        loaded_checkpoint_tensors = 0
        skipped_non_language_tensors = 0
        unknown_language_weights: list[str] = []
        unknown_vision_weights: list[str] = []
        unknown_flow_weights: list[str] = []
        for name, loaded_weight in weights:
            if name.startswith("fm_modules."):
                loaded_checkpoint_tensors += 1
                target_name = _flow_weight_target(name)
                if target_name not in params:
                    unknown_flow_weights.append(name)
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
        missing_flow_params = sorted(expected_flow_params - loaded_params)
        self.last_load_report = {
            "loaded_checkpoint_tensors": loaded_checkpoint_tensors,
            "loaded_native_parameters": len(loaded_params),
            "skipped_non_language_tensors": skipped_non_language_tensors,
            "missing_language_parameters": missing_language_params,
            "missing_vision_parameters": missing_vision_params,
            "missing_flow_parameters": missing_flow_params,
            "unknown_language_weights": sorted(unknown_language_weights),
            "unknown_vision_weights": sorted(unknown_vision_weights),
            "unknown_flow_weights": sorted(unknown_flow_weights),
        }
        if (
            missing_language_params
            or missing_vision_params
            or missing_flow_params
            or unknown_language_weights
            or unknown_vision_weights
            or unknown_flow_weights
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
    "_flow_weight_target",
    "_stacked_weight_target",
]
