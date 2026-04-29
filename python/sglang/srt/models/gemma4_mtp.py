# Copyright 2026 SGLang Team
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
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from sglang.srt.distributed import get_tensor_model_parallel_world_size
from sglang.srt.layers.layernorm import Gemma4RMSNorm, RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.gemma3_causal import Gemma3MLP
from sglang.srt.utils import add_prefix, make_layers

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenKVMTPContext:
    """Target KV pool + assistant-logical → target-physical layer map."""

    target_token_to_kv_pool: KVCache
    physical_layer_ids: Dict[int, int]

    def get_physical_layer_id(self, idx: int) -> int:
        if idx not in self.physical_layer_ids:
            raise KeyError(
                f"FrozenKVMTPContext has no physical layer id for assistant "
                f"logical index {idx}; available: {sorted(self.physical_layer_ids)}"
            )
        return self.physical_layer_ids[idx]


def _get_text_config(model_or_config) -> PretrainedConfig:
    """Normalize either a model or a (possibly wrapped) config to ``Gemma4TextConfig``."""
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", cfg)


def _resolve_target_text_model(target_model):
    for attr in ("language_model", "model"):
        candidate = getattr(target_model, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate
    raise AttributeError(
        f"Frozen-KV MTP cannot locate the target trunk on "
        f"{type(target_model).__name__}; expected ``.language_model`` "
        "(multimodal) or ``.model`` (text-only) with a ``.layers`` attribute."
    )


def build_frozen_kv_context(
    assistant_model: "Gemma4AssistantForCausalLM",
    target_model,
    target_token_to_kv_pool: KVCache,
) -> FrozenKVMTPContext:
    """Map each assistant layer to the target physical layer that owns its K/V.

    HF Gemma 4 ties each typed (sliding/full) assistant layer to the target's
    last layer of the same type; that layer is itself KV-shared with an
    earlier non-shared layer (via ``kv_shared_layer_index``). We collapse
    those two hops once so attention can hand a direct ``layer_id`` to
    ``RadixAttention`` at bind time.
    """
    target_text = _get_text_config(target_model)
    assistant_text = _get_text_config(assistant_model)
    layers = target_model.model.layers

    def kv_owner(idx: int) -> int:
        attn = layers[idx].self_attn
        owner = (
            getattr(attn, "kv_shared_layer_index", None)
            if getattr(attn, "is_kv_shared_layer", False)
            else idx
        )
        if owner is None or getattr(
            layers[owner].self_attn, "is_kv_shared_layer", False
        ):
            raise RuntimeError(
                f"Frozen-KV MTP: target layer {idx} resolved to physical "
                f"{owner!r}, which is missing or itself KV-shared (HF invariant changed?)."
            )
        return owner

    L = target_text.num_hidden_layers
    by_type = {target_text.layer_types[i]: kv_owner(i) for i in (L - 2, L - 1)}

    physical: Dict[int, int] = {}
    for i, t in enumerate(assistant_text.layer_types):
        if t not in by_type:
            raise ValueError(
                f"Frozen-KV MTP assistant layer {i} has type {t!r}, "
                f"expected one of {sorted(by_type)}."
            )
        physical[i] = by_type[t]

    return FrozenKVMTPContext(
        target_token_to_kv_pool=target_token_to_kv_pool,
        physical_layer_ids=physical,
    )


class Gemma4MTPAttention(nn.Module):
    """Q-only path; K/V read from the target pool at the bound ``layer_id``."""

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        head_dim: int,
        max_position_embeddings: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        layer_type = config.layer_types[layer_id]
        is_swa = layer_type == "sliding_attention"

        total_num_heads = config.num_attention_heads
        total_num_kv_heads = (
            getattr(config, "swa_num_key_value_heads", config.num_key_value_heads)
            if is_swa
            else config.num_key_value_heads
        )
        assert total_num_heads % tp_size == 0
        assert max(total_num_kv_heads, tp_size) % min(total_num_kv_heads, tp_size) == 0
        self.num_heads = total_num_heads // tp_size
        self.num_kv_heads = max(1, total_num_kv_heads // tp_size)

        hidden_size = config.hidden_size
        self.q_proj = ColumnParallelLinear(
            hidden_size,
            total_num_heads * head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            total_num_heads * head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = Gemma4RMSNorm(head_dim, eps=config.rms_norm_eps)

        rope = config.rope_parameters.get(
            layer_type, {"rope_type": "default", "rope_theta": 10000.0}
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=rope.get("rope_theta", 10000.0),
            rope_scaling={"rope_type": rope.get("rope_type", "default")},
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=config.sliding_window if is_swa else None,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        q, _ = self.q_proj(hidden_states)
        q = self.q_norm(q.unflatten(-1, (self.num_heads, self.head_dim))).flatten(
            -2, -1
        )
        kv_size = self.num_kv_heads * self.head_dim
        q, _ = self.rotary_emb(positions, q, torch.zeros_like(q[..., :kv_size]))
        q = q.unflatten(-1, (self.num_heads, self.head_dim))

        attn_output = self.attn(
            q, None, None, forward_batch=forward_batch, save_kv_cache=False
        )
        if attn_output.dim() == 3:
            attn_output = attn_output.flatten(-2, -1)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    """Gemma 4 MTP decoder layer (Q-only attention; no MoE/PLE per HF assistant config)."""

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_type = config.layer_types[layer_id]
        head_dim = (
            config.head_dim
            if layer_type == "full_attention"
            else getattr(config, "swa_head_dim", config.head_dim)
        )

        self.self_attn = Gemma4MTPAttention(
            layer_id=layer_id,
            config=config,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = Gemma3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        for name in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            setattr(self, name, RMSNorm(config.hidden_size, eps=config.rms_norm_eps))
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(
            self.self_attn(
                positions=positions,
                hidden_states=self.input_layernorm(hidden_states),
                forward_batch=forward_batch,
            )
        )
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.post_feedforward_layernorm(self.mlp(hidden_states))
        return (hidden_states + residual) * self.layer_scalar


class Gemma4MTPTextModel(nn.Module):
    """Trunk over ``input_embeds``; no embed table (the assistant owns it)."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: Gemma4MTPDecoderLayer(
                layer_id=idx,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=add_prefix("layers", prefix),
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )
        return self.norm(hidden_states)


class Gemma4AssistantForCausalLM(PreTrainedModel):
    """Gemma 4 MTP assistant: target embed + recurrent hidden through pre/post projection; own ``lm_head``."""

    base_model_prefix = "model"

    packed_modules_mapping: Dict[str, list] = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    embedding_modules: Dict[str, str] = {}
    embedding_padding_modules: list = []
    supports_lora = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        text_config = _get_text_config(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = config.backbone_hidden_size
        self.use_ordered_embeddings = bool(
            getattr(config, "use_ordered_embeddings", False)
        )
        self.centroid_intermediate_top_k = int(
            getattr(config, "centroid_intermediate_top_k", 32)
        )

        self.target_embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            self.backbone_hidden_size,
            prefix=add_prefix("target_embed_tokens", prefix),
        )
        self.pre_projection = ReplicatedLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("pre_projection", prefix),
        )
        self.model = Gemma4MTPTextModel(
            config=text_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.post_projection = ReplicatedLinear(
            self.hidden_size,
            self.backbone_hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("post_projection", prefix),
        )

        # Full-vocab logits per rank → ``skip_all_gather=True`` for both heads.
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.logits_processor = LogitsProcessor(text_config, skip_all_gather=True)

        if self.use_ordered_embeddings:
            self.num_centroids = int(config.num_centroids)
            self.vocab_size_per_centroid, rem = divmod(
                self.vocab_size, self.num_centroids
            )
            if rem:
                raise ValueError(
                    "Frozen-KV MTP centroid head requires vocab_size to be a "
                    f"multiple of num_centroids (vocab={self.vocab_size}, "
                    f"num_centroids={self.num_centroids})."
                )
            self.centroids = nn.Linear(self.hidden_size, self.num_centroids, bias=False)
            self.register_buffer(
                "token_ordering",
                torch.zeros(self.vocab_size, dtype=torch.long),
                persistent=True,
            )
        else:
            self.num_centroids = self.vocab_size_per_centroid = self.centroids = None
            self.register_buffer("token_ordering", None, persistent=False)

        self.kv_context: Optional[FrozenKVMTPContext] = None
        self.post_init()

    def bind_frozen_kv_context(self, ctx: FrozenKVMTPContext) -> None:
        """Set each layer's ``RadixAttention.layer_id`` to the target physical id."""
        for assistant_logical, layer in enumerate(self.model.layers):
            target_phys = ctx.get_physical_layer_id(assistant_logical)
            layer.self_attn.attn.layer_id = target_phys
            layer.self_attn.layer_id = assistant_logical
        self.kv_context = ctx

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.target_embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Rebind target embedding; ``head`` ignored (assistant keeps ``lm_head``)."""
        del head
        del self.target_embed_tokens.weight
        self.target_embed_tokens.weight = embed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        token_embed = (
            self.target_embed_tokens(input_ids)
            if input_embeds is None
            else input_embeds
        )

        if forward_batch.spec_info is None or not hasattr(
            forward_batch.spec_info, "hidden_states"
        ):
            raise RuntimeError(
                "Frozen-KV MTP forward requires forward_batch.spec_info."
                "hidden_states to carry the recurrent state. The worker's "
                "_frozen_kv_target_view context manager must be exited "
                "before model forward, leaving spec_info populated."
            )
        prev_hidden = forward_batch.spec_info.hidden_states
        if token_embed.shape != prev_hidden.shape:
            raise ValueError(
                "Frozen-KV MTP forward: token_embed and prev_hidden must have "
                f"the same shape (got {token_embed.shape} vs {prev_hidden.shape})."
            )

        z, _ = self.pre_projection(torch.cat([token_embed, prev_hidden], dim=-1))
        hidden_states = self.model(
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=z,
            **kwargs,
        )
        projected_states, _ = self.post_projection(hidden_states)

        if self.use_ordered_embeddings:
            return self._centroid_logits_processor(
                input_ids, hidden_states, projected_states, forward_batch
            )

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            hidden_states_before_norm=projected_states,
        )

    def _apply_centroid_masking(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """HF-style dense-scatter centroid logits (E2B/E4B)."""
        if self.centroids is None or self.token_ordering is None:
            raise RuntimeError(
                "Frozen-KV MTP centroid head invoked but centroid weights "
                "are not initialized."
            )
        prefix_shape = hidden_states.shape[:-1]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])

        centroid_logits = self.centroids(flat_hidden)
        _, top_k_indices = torch.topk(
            centroid_logits, k=self.centroid_intermediate_top_k, dim=-1
        )
        canonical_positions_per_cluster = self.token_ordering.long().view(
            self.num_centroids, self.vocab_size_per_centroid
        )
        selected_canonical = canonical_positions_per_cluster[top_k_indices]
        selected_flat = selected_canonical.reshape(-1)
        selected_embeddings = self.lm_head.weight[selected_flat].view(
            flat_hidden.shape[0],
            self.centroid_intermediate_top_k * self.vocab_size_per_centroid,
            self.hidden_size,
        )
        selected_logits = (
            flat_hidden.unsqueeze(1) @ selected_embeddings.transpose(-1, -2)
        ).squeeze(1)
        mask_value = selected_logits.min() - 1.0
        output = torch.full(
            (flat_hidden.shape[0], self.vocab_size),
            fill_value=mask_value.item(),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output.scatter_(
            dim=-1,
            index=selected_canonical.view(flat_hidden.shape[0], -1),
            src=selected_logits,
        )
        return output.view(*prefix_shape, self.vocab_size)

    def _centroid_logits_processor(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        projected_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        logits_metadata = LogitsMetadata.from_forward_batch(forward_batch)
        if logits_metadata.extend_return_logprob:
            raise NotImplementedError(
                "Frozen-KV MTP centroid head does not support input logprobs yet."
            )

        (
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            *_,
        ) = self.logits_processor._get_pruned_states(
            hidden_states, projected_states, None, logits_metadata
        )
        hidden_states_to_store = self.logits_processor._get_hidden_states_to_store(
            hidden_states,
            projected_states,
            None,
            pruned_states,
            pruned_states_before_norm,
            aux_pruned_states,
            sample_indices,
            logits_metadata,
        )
        del input_ids, hidden_states, projected_states

        logits = self._apply_centroid_masking(pruned_states)
        sampled_logits = (
            logits[sample_indices] if sample_indices is not None else logits
        )
        return LogitsProcessorOutput(
            next_token_logits=sampled_logits,
            hidden_states=hidden_states_to_store,
            mm_input_embeds=logits_metadata.mm_input_embeds,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        # The HF assistant checkpoint loads by exact name except for two transforms:
        # (1) ``model.embed_tokens.weight`` (tied via ``tie_word_embeddings``) is the
        #     only source of ``lm_head.weight`` since this model has no embed table
        #     of its own (``target_embed_tokens`` is rebound from the target at
        #     runtime, so it is expected to be missing from any checkpoint),
        # (2) per-layer ``mlp.{gate,up}_proj.weight`` are fused into ``gate_up_proj``.
        gate_up_mapping = (("gate_proj", 0), ("up_proj", 1))
        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if name == "model.embed_tokens.weight" and getattr(
                self.config, "tie_word_embeddings", False
            ):
                default_weight_loader(self.lm_head.weight, loaded_weight)
                loaded_params.add("lm_head.weight")
                continue

            for shard_name, shard_id in gate_up_mapping:
                if shard_name not in name:
                    continue
                mapped = name.replace(shard_name, "gate_up_proj")
                if mapped not in params_dict:
                    continue
                param = params_dict[mapped]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped)
                break
            else:
                if name not in params_dict:
                    continue
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(
                    param, loaded_weight
                )
                loaded_params.add(name)

        missing = sorted(
            n
            for n, _ in self.named_parameters()
            if n not in loaded_params and not n.startswith("target_embed_tokens.")
        )
        if missing:
            logger.warning(
                "Some weights are not initialized from checkpoints: %s", missing
            )
        return loaded_params


EntryClass = Gemma4AssistantForCausalLM
