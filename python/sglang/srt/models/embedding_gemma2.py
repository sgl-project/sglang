# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""EmbeddingGemma2 model (EmbeddingGemma v2) for SGLang.

Inference-only multimodal bidirectional embedding model compatible with HuggingFace.
Supports text, image, audio, and video inputs with mean pooling and unit L2 normalization.
"""

import logging
from collections.abc import Iterable
from typing import Any, Callable, cast

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.activation import get_act_fn
from sglang.srt.layers.attention.vision import VisionSdpaAttention
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    ForwardBatch,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma4_audio import Gemma4AudioEncoder
from sglang.srt.models.gemma4_causal import Gemma4MLP
from sglang.srt.models.gemma4_mm import (
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
)
from sglang.srt.models.gemma4_vision import Gemma4VisionEncoder
from sglang.srt.runtime_context import get_mm, get_parallel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class EmbeddingGemma2RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization with optional learnable scale.

    Used across EmbeddingGemma2 attention norms, projection norms, and decoder layer norms.
    Supports forward_native binding for 3D/4D tensors and with_scale=False operations
    (e.g., v_norm in self-attention layers where scaling is omitted).
    """

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)

    def forward_native(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.forward(hidden_states)


class EmbeddingGemma2Attention(nn.Module):
    """Attention layer for EmbeddingGemma2.

    Implements sliding window attention with an inclusive bidirectional window (<= 512)
    and full global attention depending on config.layer_types. In sliding layers,
    each token attends to all tokens within distance <= 512 both backward and forward,
    unlike causal SWA where attention is restricted to past tokens.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"

        # Dims depend on layer type
        if self.is_sliding:
            self.head_dim = getattr(config, "swa_head_dim", 256)
            self.num_kv_heads = getattr(config, "swa_num_key_value_heads", 2)
            rope_theta = config.rope_parameters["sliding_attention"].get(
                "rope_theta"
            ) or config.rope_parameters["sliding_attention"].get("base", 10000.0)
            self.sliding_window = 512
        else:
            self.head_dim = getattr(config, "head_dim", 512)
            self.num_kv_heads = getattr(config, "num_key_value_heads", 1)
            rope_theta = config.rope_parameters["full_attention"].get(
                "rope_theta"
            ) or config.rope_parameters["full_attention"].get("base", 1000000.0)
            self.sliding_window = -1

        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = self.num_kv_heads
        tp_size = get_parallel().tp_size
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.q_norm = EmbeddingGemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = EmbeddingGemma2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = EmbeddingGemma2RMSNorm(
            self.head_dim, eps=config.rms_norm_eps, with_scale=False
        )

        max_pos = min(
            config.max_position_embeddings,
            getattr(config, "context_len", config.max_position_embeddings),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_pos,
            base=int(rope_theta),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_idx,
            sliding_window_size=self.sliding_window,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            attn_type=AttentionType.ENCODER_ONLY,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q).flatten(-2, -1)

        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k).flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))

        v = v.unflatten(-1, (self.num_kv_heads, self.head_dim))
        v = self.v_norm(v)

        attn_out = self.attn(q, k, v, forward_batch=forward_batch)
        if attn_out.dim() == 3:
            attn_out = attn_out.flatten(-2, -1)
        out, _ = self.o_proj(attn_out)
        return out


class EmbeddingGemma2PLEBlock(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.per_layer_input_gate = ReplicatedLinear(
            config.hidden_size,
            config.hidden_size_per_layer_input,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_input_gate", prefix),
        )
        self.per_layer_projection = ReplicatedLinear(
            config.hidden_size_per_layer_input,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_projection", prefix),
        )
        self.post_per_layer_input_norm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.act_fn = get_act_fn(config.hidden_activation)

    def forward(
        self, hidden_states: torch.Tensor, per_layer_input: torch.Tensor
    ) -> torch.Tensor:
        gate = self.act_fn(self.per_layer_input_gate(hidden_states)[0])
        gated = gate * per_layer_input
        proj = self.per_layer_projection(gated)[0]
        return hidden_states + self.post_per_layer_input_norm(proj)


class EmbeddingGemma2DecoderLayer(nn.Module):
    """Decoder layer for EmbeddingGemma2.

    Combines self-attention (with sliding or full attention depending on layer index),
    feed-forward MLP, and Per-Layer Embedding (PLE) block, followed by scaling by a
    persistent scalar parameter (`layer_scalar`).
    """

    def __init__(
        self,
        layer_idx: int,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = EmbeddingGemma2Attention(
            config, layer_idx, quant_config, prefix=add_prefix("self_attn", prefix)
        )
        self.mlp = Gemma4MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.ple_block = EmbeddingGemma2PLEBlock(
            config, quant_config=quant_config, prefix=add_prefix("ple_block", prefix)
        )
        self.input_layernorm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = EmbeddingGemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)
        self.layer_scalar: torch.Tensor

    def forward(
        self,
        positions: torch.Tensor,
        h: torch.Tensor,
        forward_batch: ForwardBatch,
        per_layer_input: torch.Tensor,
    ) -> torch.Tensor:
        r = h
        h = (
            self.post_attention_layernorm(
                self.self_attn(
                    positions, self.input_layernorm(h), forward_batch=forward_batch
                )
            )
            + r
        )
        r = h
        h = (
            self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(h)))
            + r
        )
        h = self.ple_block(h, per_layer_input)
        return h * self.layer_scalar


class EmbeddingGemma2TextPLE(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.num_layers = config.num_hidden_layers
        self.hidden_dim = config.hidden_size_per_layer_input
        self.per_layer_model_projection = ColumnParallelLinear(
            config.hidden_size,
            self.num_layers * self.hidden_dim,
            bias=False,
            gather_output=True,
            quant_config=quant_config,
            prefix=add_prefix("per_layer_model_projection", prefix),
        )
        self.per_layer_model_projection_scale = config.hidden_size**-0.5
        self.per_layer_projection_norm = EmbeddingGemma2RMSNorm(
            self.hidden_dim, eps=config.rms_norm_eps
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        proj, _ = self.per_layer_model_projection(inputs_embeds)
        proj = proj * self.per_layer_model_projection_scale
        proj = proj.reshape(*inputs_embeds.shape[:-1], self.num_layers, self.hidden_dim)
        return self.per_layer_projection_norm(proj)


class EmbeddingGemma2VocabParallelEmbedding(VocabParallelEmbedding):
    """Embedding that replaces multimodal placeholder tokens with pad_token_id before lookup."""

    def __init__(
        self,
        *args,
        pad_token_id: int = 0,
        mm_token_ids: tuple[int, ...] = (),
        embed_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.mm_token_ids = mm_token_ids
        self.embed_scale_val = embed_scale
        self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        self.embed_scale: torch.Tensor

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mm_token_ids:
            for mm_id in self.mm_token_ids:
                if (input_ids == mm_id).any():
                    input_ids = torch.where(
                        input_ids == mm_id, self.pad_token_id, input_ids
                    )
        return super().forward(input_ids) * self.embed_scale.to(
            cast(torch.dtype, self.weight.dtype)
        )


class EmbeddingGemma2TextModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        mm_ids = tuple(
            tid
            for tid in (
                getattr(config, "image_token_id", 258880),
                getattr(config, "audio_token_id", 258881),
                getattr(config, "video_token_id", 258884),
            )
            if tid is not None
        )
        self.embed_tokens = EmbeddingGemma2VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            pad_token_id=config.pad_token_id,
            mm_token_ids=mm_ids,
            embed_scale=config.hidden_size**0.5,
            quant_config=quant_config,
            prefix=add_prefix("embed_tokens", prefix),
        )
        self.ple = EmbeddingGemma2TextPLE(
            config, quant_config=quant_config, prefix=add_prefix("ple", prefix)
        )
        self.layers = nn.ModuleList(
            [
                EmbeddingGemma2DecoderLayer(
                    i,
                    config,
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = EmbeddingGemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_projection = ReplicatedLinear(
            config.hidden_size,
            getattr(config, "embedding_dim", 768),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("embedding_projection", prefix),
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def dtype(self) -> torch.dtype:
        return cast(torch.dtype, self.embed_tokens.weight.dtype)

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self.embed_tokens.weight.device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        forward_batch: ForwardBatch | None = None,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if positions is None and forward_batch is not None:
            positions = getattr(forward_batch, "positions", None)
        if positions is None:
            positions = kwargs.get("positions")
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        per_layer = self.ple(input_embeds)
        h = input_embeds
        for i, layer in enumerate(self.layers):
            h = layer(positions, h, forward_batch, per_layer[..., i, :])
        h = self.norm(h)
        h, _ = self.embedding_projection(h)
        return h


class EmbeddingGemma2VisionPooler(nn.Module):
    """Vision pooler for EmbeddingGemma2.

    Isolated from Gemma4VisionPooler to prevent regressions in generative vision pipelines.
    Performs 2D spatial average pooling over vision patch tokens into soft visual tokens
    scaled by root_hidden_size (hidden_size ** 0.5) according to output_length.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.root_hidden_size = self.hidden_size**0.5

    def _avg_pool_by_positions(
        self, x: torch.Tensor, patch_positions: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_seq_len = x.shape[1]
        k = int((input_seq_len // length) ** 0.5)
        k_squared = k**2
        if k_squared * length != input_seq_len:
            raise ValueError(
                f"Cannot pool {x.shape} to {length}: {k=}^2 times {length=} must be {input_seq_len}."
            )
        clamped_positions = patch_positions.clamp(min=0)
        max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]

        weights = F.one_hot(kernel_idxs.long(), length).float() / k_squared
        output = weights.transpose(1, 2) @ x.float()
        mask = torch.logical_not((weights == 0).all(dim=1))
        return output, mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_positions: torch.Tensor,
        padding_positions: torch.Tensor,
        output_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if output_length is None:
            raise ValueError(
                "output_length is required for EmbeddingGemma2VisionPooler"
            )
        if output_length > hidden_states.shape[1]:
            raise ValueError(
                f"Cannot output more soft tokens (requested {output_length}) than there are patches"
                f" ({hidden_states.shape[1]}). Change the value of `num_soft_tokens` when processing."
            )
        length = output_length
        if isinstance(length, (list, tuple)):
            length = length[0]
        orig_dtype = hidden_states.dtype
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        if hidden_states.shape[1] == length:
            pooler_mask = ~padding_positions
        else:
            hidden_states, pooler_mask = self._avg_pool_by_positions(
                hidden_states, patch_positions, length
            )
        hidden_states = (hidden_states.float() * self.root_hidden_size).to(orig_dtype)
        return hidden_states, pooler_mask


class EmbeddingGemma2Model(nn.Module):
    """EmbeddingGemma2 multimodal embedding model.

    Supports text, image, audio, and video inputs with mean pooling and L2 normalization
    over a 768-dim output embedding space.

    Architecture and design choices:
      - Selective modality loading: vision and audio towers are conditionally loaded
        based on server_args.limit_mm_data_per_request to save VRAM when only text,
        image, audio, or video modalities are served.
      - EmbeddingGemma2RMSNorm / forward_native rebinding: rebinds forward to forward_native
        across Gemma4RMSNorm and RMSNorm modules to handle 3D/4D tensors and with_scale=False.
      - EmbeddingGemma2VisionPooler: dedicated 2D spatial average pooler isolated from
        generative Gemma4VisionPooler to prevent regressions in generation paths.
      - Bidirectional sliding window: attention uses inclusive <= 512 bidirectional SWA
        enabling full bidirectional cross-token representations for embedding tasks.
      - Non-meta buffer materialization: materializes persistent buffers such as embed_scale
        while preserving non-meta buffers (including scalar/tensor buffers with inf/-inf).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        text_config = config.text_config
        if hasattr(text_config, "allow_global_per_layer_attribute_access"):
            text_config.allow_global_per_layer_attribute_access = True
        self.text_config = text_config

        try:
            limit_mm = get_mm().limit_mm_data_per_request or {}
        except Exception:  # noqa: BLE001
            limit_mm = {}

        # Gated Vision Tower
        has_vision = getattr(config, "vision_config", None) is not None and (
            limit_mm.get("image", 1) > 0 or limit_mm.get("video", 1) > 0
        )
        self.vision_tower: Gemma4VisionEncoder | None
        self.embed_vision: Gemma4MultimodalEmbedder | None
        self.audio_tower: Gemma4AudioEncoder | None
        self.embed_audio: Gemma4MultimodalEmbedder | None

        if has_vision:
            self.vision_tower = Gemma4VisionEncoder(
                config.vision_config,
                quant_config=quant_config,
                prefix=add_prefix("vision_tower", prefix),
            )
            self.vision_tower.pooler = EmbeddingGemma2VisionPooler(config.vision_config)  # type: ignore[assignment]
            for layer in self.vision_tower.encoder.layers:  # type: ignore[union-attr]
                layer.self_attn.qkv_backend = VisionSdpaAttention(  # type: ignore[attr-defined,union-attr]
                    head_dim=config.vision_config.head_dim,
                    num_heads=layer.self_attn.num_heads_per_partition,  # type: ignore[attr-defined,arg-type,union-attr]
                    num_kv_heads=layer.self_attn.num_kv_heads_per_partition,  # type: ignore[attr-defined,arg-type,union-attr]
                    dropout=0.0,
                    flatten_batch=False,
                    softmax_in_single_precision=False,
                    softmax_scale=1.0,
                )
            self.embed_vision = Gemma4MultimodalEmbedder(
                config.vision_config,
                text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_vision", prefix),
            )
        else:
            self.vision_tower = None
            self.embed_vision = None

        # Gated Audio Tower
        has_audio = (
            getattr(config, "audio_config", None) is not None
            and limit_mm.get("audio", 1) > 0
        )
        if has_audio:
            self.audio_tower = Gemma4AudioEncoder(
                config.audio_config,
                quant_config=quant_config,
                prefix=add_prefix("audio_tower", prefix),
            )
            self.embed_audio = Gemma4MultimodalEmbedder(
                config.audio_config,
                text_config,
                quant_config=quant_config,
                prefix=add_prefix("embed_audio", prefix),
            )
        else:
            self.audio_tower = None
            self.embed_audio = None

        self.language_model = EmbeddingGemma2TextModel(
            text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pooler = Pooler(PoolingType.MEAN, normalize=True)

        for m in self.modules():
            if isinstance(m, (Gemma4RMSNorm, RMSNorm)):
                m.forward = m.forward_native  # type: ignore[method-assign,assignment]

    def get_attention_sliding_window_size(self) -> int:
        return 512

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError(
                "Image inputs provided but vision tower is not loaded. "
                "Ensure limit_mm_data_per_request allows image > 0."
            )
        vt = self.vision_tower
        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list(
                [getattr(item, "image_position_ids", None)]
            )
            for pv_idx, pv in enumerate(all_pixel_values):
                if pv.dim() in (2, 3) and pv.shape[-1] == self.text_config.hidden_size:
                    all_embeds.append(pv.to(self.language_model.device))
                    continue
                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values[{pv_idx}] has no matching image_position_ids."
                    )
                pp = all_position_ids[pv_idx]
                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)
                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())
                pp = pp.to(device=vt.device)

                pooled, pooler_mask = vt(pv, pp)
                for hs, mask in zip(pooled, pooler_mask):
                    real_tokens = hs[mask]
                    all_embeds.append(
                        self.embed_vision(
                            inputs_embeds=real_tokens.unsqueeze(0)
                        ).squeeze(0)
                    )
        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.text_config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.vision_tower is None or self.embed_vision is None:
            raise ValueError(
                "Video inputs provided but vision tower is not loaded. "
                "Ensure limit_mm_data_per_request allows video > 0."
            )
        vt = self.vision_tower
        all_embeds = []
        for item in items:
            all_pixel_values = flatten_nested_list([item.feature])
            all_position_ids = flatten_nested_list(
                [getattr(item, "video_position_ids", None)]
            )
            for pv_idx, pv in enumerate(all_pixel_values):
                if pv.dim() in (2, 3) and pv.shape[-1] == self.text_config.hidden_size:
                    all_embeds.append(pv.to(self.language_model.device))
                    continue
                if pv_idx >= len(all_position_ids) or all_position_ids[pv_idx] is None:
                    raise ValueError(
                        f"pixel_values_videos[{pv_idx}] has no matching video_position_ids."
                    )
                pp = all_position_ids[pv_idx]
                if pv.dim() == 4:
                    pv = pv.reshape(-1, pv.shape[-2], pv.shape[-1])
                if pp.dim() == 4:
                    pp = pp.reshape(-1, pp.shape[-2], pp.shape[-1])
                if pv.dim() == 2:
                    pv = pv.unsqueeze(0)
                if pp.dim() == 2:
                    pp = pp.unsqueeze(0)
                pv = pv.to(device=vt.device, dtype=self.language_model.dtype())
                pp = pp.to(device=vt.device)

                pooled, pooler_mask = vt(pv, pp)
                for hs, mask in zip(pooled, pooler_mask):
                    real_tokens = hs[mask]
                    all_embeds.append(
                        self.embed_vision(
                            inputs_embeds=real_tokens.unsqueeze(0)
                        ).squeeze(0)
                    )
        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.text_config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        if self.audio_tower is None or self.embed_audio is None:
            raise ValueError(
                "Audio inputs provided but audio tower is not loaded. "
                "Ensure limit_mm_data_per_request allows audio > 0."
            )
        all_input_features = flatten_nested_list([item.feature for item in items])
        all_input_features_mask = flatten_nested_list(
            [~item.input_features_mask for item in items]  # type: ignore[operator]
        )
        all_embeds = []
        for input_features, input_features_mask in zip(
            all_input_features, all_input_features_mask
        ):
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if input_features_mask.dim() == 1:
                input_features_mask = input_features_mask.unsqueeze(0)

            input_features = input_features.to(
                device=self.audio_tower.device,
                dtype=self.language_model.dtype(),
            )
            input_features_mask = input_features_mask.to(device=input_features.device)

            audio_encodings, audio_mask = self.audio_tower(
                input_features, input_features_mask
            )
            audio_features = self.embed_audio(inputs_embeds=audio_encodings)
            for enc, mask in zip(audio_features, audio_mask):
                all_embeds.append(enc[~mask])
        if all_embeds:
            return torch.cat(all_embeds, dim=0)
        else:
            return torch.empty(
                0,
                self.text_config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        get_embedding: bool = True,
        **kwargs,
    ):
        data_embedding_funcs: dict[
            Modality, Callable[[list[MultimodalDataItem]], Any]
        ] = {}
        if self.vision_tower is not None:
            data_embedding_funcs[Modality.IMAGE] = self.get_image_feature
            data_embedding_funcs[Modality.VIDEO] = self.get_video_feature
        if self.audio_tower is not None:
            data_embedding_funcs[Modality.AUDIO] = self.get_audio_feature

        if forward_batch.contains_mm_inputs():
            h = general_mm_embed_routine(
                input_ids=input_ids,
                forward_batch=forward_batch,
                language_model=self.language_model,
                data_embedding_funcs=data_embedding_funcs,
                placeholder_tokens={
                    Modality.IMAGE: [getattr(self.config, "image_token_id", 258880)],
                    Modality.AUDIO: [getattr(self.config, "audio_token_id", 258881)],
                    Modality.VIDEO: [getattr(self.config, "video_token_id", 258884)],
                },
                positions=positions,
            )
        else:
            h = self.language_model(
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
            )

        # Pooler expects fp32 tensor, mean pool, L2 normalize
        return self.pooler(h.float(), forward_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> list[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded_params = set()

        for name, loaded_weight in weights:
            # Tower skipping
            if self.audio_tower is None and (
                name.startswith(("audio_tower.", "embed_audio."))
            ):
                continue
            if self.vision_tower is None and (
                name.startswith(("vision_tower.", "embed_vision."))
            ):
                continue

            # Tower remapping using Gemma4 static methods
            if name.startswith("audio_tower."):
                name = Gemma4ForConditionalGeneration._remap_audio_tower_name(name)
            if name.startswith(
                ("vision_tower.", "audio_tower.", "embed_vision.", "embed_audio.")
            ):
                name = Gemma4ForConditionalGeneration._remap_tower_name(
                    name, params_dict
                )

            # Persistent buffer loading (e.g. layer_scalar)
            if name in buffers_dict:
                buffers_dict[name].copy_(loaded_weight)
                loaded_params.add(name)
                continue

            # Stacked projection routing
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name in name:
                    mapped_name = name.replace(shard_name, param_name)
                    if mapped_name in params_dict:
                        param = params_dict[mapped_name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        try:
                            weight_loader(param, loaded_weight, shard_id)  # type: ignore[call-arg]
                        except TypeError:
                            weight_loader(param, loaded_weight)
                        loaded_params.add(mapped_name)
                        break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)
        # Audit unloaded parameters
        expected_params = set(params_dict.keys())
        if self.vision_tower is None:
            expected_params = {
                k
                for k in expected_params
                if not k.startswith(("vision_tower.", "embed_vision."))
            }
        if self.audio_tower is None:
            expected_params = {
                k
                for k in expected_params
                if not k.startswith(("audio_tower.", "embed_audio."))
            }
        unloaded = expected_params - loaded_params
        if unloaded:
            logger.warning(
                f"EmbeddingGemma2Model: {len(unloaded)} expected parameters were not loaded: {sorted(unloaded)[:10]}"
            )

        # Materialize non-persistent / meta buffers
        target_device = self.language_model.device
        target_dtype = self.language_model.dtype()
        for module in self.modules():
            for name, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and (buf.is_meta or buf.device != target_device):
                    if name == "embed_scale":
                        scale_val = getattr(
                            module,
                            "embed_scale_val",
                            getattr(self.config, "hidden_size", 512) ** 0.5,
                        )
                        module._buffers[name] = torch.tensor(
                            scale_val, dtype=target_dtype, device=target_device
                        )
                    elif not buf.is_meta:
                        module._buffers[name] = buf.to(
                            device=target_device, dtype=target_dtype
                        )
                    else:
                        module._buffers[name] = torch.ones(
                            buf.shape, dtype=target_dtype, device=target_device
                        )

        for m in self.modules():
            if isinstance(m, (Gemma4RMSNorm, RMSNorm)):
                m.forward = m.forward_native  # type: ignore[method-assign,assignment]

        return list(loaded_params)

    def get_weights_by_name(
        self, name: str, truncate_size: int = 100, tp_size: int = 1
    ) -> torch.Tensor | None:
        params_dict = dict(self.named_parameters())
        if name in params_dict:
            return (
                params_dict[name]
                .data.cpu()
                .to(torch.float32)
                .numpy()
                .tolist()[:truncate_size]
            )
        return None


EntryClass = [EmbeddingGemma2Model]
