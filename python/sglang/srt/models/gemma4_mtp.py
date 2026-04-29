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
"""Gemma 4 MTP assistant: Q-only layers read the target's frozen KV
(:class:`FrozenKVMTPContext` maps assistant logical → target physical layer id).

Composes (1) assistant logical → target logical from HF's last-two-target-layer
rule with ``layer_types``, and (2) target logical → physical via the target
layer's ``kv_shared_layer_index`` (one hop; owner is non-shared).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Optional, Set, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from sglang.srt.distributed import (
    get_tensor_model_parallel_world_size,
)
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
from sglang.srt.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3_causal import Gemma3MLP
from sglang.srt.utils import add_prefix, make_layers

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrozenKVMTPContext:
    """Target KV pool + assistant logical → target physical layer map."""

    target_token_to_kv_pool: "KVCache"
    physical_layer_ids: Dict[int, int]

    def get_physical_layer_id(self, assistant_logical_idx: int) -> int:
        try:
            return self.physical_layer_ids[assistant_logical_idx]
        except KeyError as e:
            raise KeyError(
                "FrozenKVMTPContext has no physical layer id for assistant "
                f"logical index {assistant_logical_idx}; available indices: "
                f"{sorted(self.physical_layer_ids)}"
            ) from e


def _get_text_config(model_or_config) -> PretrainedConfig:
    """Return the inner Gemma 4 text config from either the model or its config."""
    cfg = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(cfg, "text_config", None)
    return text_config if text_config is not None else cfg


def resolve_target_physical(target_model, logical_layer_id: int) -> int:
    """Target logical layer → KV-owning physical layer (follow ``kv_shared_layer_index``)."""
    layers = target_model.model.layers
    if logical_layer_id < 0 or logical_layer_id >= len(layers):
        raise ValueError(
            f"Frozen-KV MTP: target logical layer id {logical_layer_id} out of "
            f"range [0, {len(layers) - 1}]."
        )
    self_attn = layers[logical_layer_id].self_attn
    is_shared = getattr(self_attn, "is_kv_shared_layer", False)
    shared_idx = getattr(self_attn, "kv_shared_layer_index", None)
    if is_shared and shared_idx is not None:
        return shared_idx
    return logical_layer_id


def _build_assistant_to_target_logical(
    assistant_text_config: PretrainedConfig,
    target_text_config: PretrainedConfig,
) -> Dict[int, int]:
    """Assistant logical → target logical: HF ties sliding/full assistant layers to target ``L-2`` / ``L-1``."""
    target_num_layers = target_text_config.num_hidden_layers
    target_layer_types = target_text_config.layer_types
    if target_num_layers < 2:
        raise ValueError(
            "Frozen-KV MTP requires the target to have at least two layers, "
            f"got num_hidden_layers={target_num_layers}."
        )

    second_last_type = target_layer_types[target_num_layers - 2]
    last_type = target_layer_types[target_num_layers - 1]
    if {second_last_type, last_type} != {"sliding_attention", "full_attention"}:
        raise ValueError(
            "Frozen-KV MTP expects the target's last two layers to be "
            "{sliding_attention, full_attention}, got "
            f"({second_last_type}, {last_type})."
        )

    type_to_target_logical = {
        target_layer_types[target_num_layers - 2]: target_num_layers - 2,
        target_layer_types[target_num_layers - 1]: target_num_layers - 1,
    }

    assistant_layer_types = assistant_text_config.layer_types
    a2t: Dict[int, int] = {}
    for assistant_logical, layer_type in enumerate(assistant_layer_types):
        if layer_type not in type_to_target_logical:
            raise ValueError(
                f"Frozen-KV MTP assistant layer {assistant_logical} has type "
                f"'{layer_type}', which is not in the target's last two layer "
                f"types {sorted(type_to_target_logical)}."
            )
        a2t[assistant_logical] = type_to_target_logical[layer_type]

    return a2t


def build_physical_layer_ids(
    assistant_model: "Gemma4AssistantForCausalLM",
    target_model,
) -> Dict[int, int]:
    """Compose A (assistant→target logical) and B (target logical→physical)."""
    assistant_text_config = _get_text_config(assistant_model)
    target_text_config = _get_text_config(target_model)
    a2t = _build_assistant_to_target_logical(assistant_text_config, target_text_config)
    physical: Dict[int, int] = {}
    for assistant_logical, target_logical in a2t.items():
        physical_id = resolve_target_physical(target_model, target_logical)
        target_self_attn = target_model.model.layers[physical_id].self_attn
        if getattr(target_self_attn, "is_kv_shared_layer", False):
            raise RuntimeError(
                "Frozen-KV MTP: target physical layer "
                f"{physical_id} for assistant layer {assistant_logical} is "
                "still KV-shared after one-hop resolution. The HF invariant "
                "that ``kv_shared_layer_index`` points at a non-shared layer "
                "appears to have changed."
            )
        physical[assistant_logical] = physical_id
    return physical


def build_frozen_kv_context(
    assistant_model: "Gemma4AssistantForCausalLM",
    target_model,
    target_token_to_kv_pool,
) -> FrozenKVMTPContext:
    """Build the per-worker :class:`FrozenKVMTPContext`."""
    physical_layer_ids = build_physical_layer_ids(assistant_model, target_model)
    return FrozenKVMTPContext(
        target_token_to_kv_pool=target_token_to_kv_pool,
        physical_layer_ids=physical_layer_ids,
    )


class Gemma4MTPAttention(nn.Module):
    """Q-only path; K/V read from target pool at bound ``layer_id``."""

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
        self.config = config
        tp_size = get_tensor_model_parallel_world_size()

        layer_type = config.layer_types[layer_id]
        self.layer_type = layer_type
        self.sliding_window = (
            config.sliding_window if layer_type == "sliding_attention" else None
        )

        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        if layer_type == "sliding_attention":
            self.total_num_kv_heads = getattr(
                config, "swa_num_key_value_heads", config.num_key_value_heads
            )
        else:
            self.total_num_kv_heads = config.num_key_value_heads

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        hidden_size = config.hidden_size
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim

        self.q_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("q_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.q_norm = Gemma4RMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
        )

        if layer_type in config.rope_parameters:
            rope_parameters = dict(config.rope_parameters[layer_type])
        else:
            rope_parameters = dict(rope_type="default", rope_theta=10000.0)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_parameters.get("rope_theta", 10000.0),
            rope_scaling={"rope_type": rope_parameters.get("rope_type", "default")},
            partial_rotary_factor=rope_parameters.get("partial_rotary_factor", 1.0),
            is_neox_style=True,
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            scaling=1.0,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            sliding_window_size=self.sliding_window,
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
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)

        kv_size = self.num_kv_heads * self.head_dim
        dummy_k = torch.zeros_like(q[..., :kv_size])
        q, _ = self.rotary_emb(positions, q, dummy_k)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        attn_output = self.attn(
            q, None, None, forward_batch=forward_batch, save_kv_cache=False
        )
        if attn_output.dim() == 3:
            attn_output = attn_output.flatten(-2, -1)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma4MTPDecoderLayer(nn.Module):
    """Like ``Gemma4DecoderLayer`` but attention is :class:`Gemma4MTPAttention`; no MoE/PLE per HF assistant config."""

    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_id = layer_id

        layer_type = config.layer_types[layer_id]
        if layer_type == "full_attention":
            head_dim = config.head_dim
        else:
            head_dim = getattr(config, "swa_head_dim", config.head_dim)

        self.self_attn = Gemma4MTPAttention(
            layer_id=layer_id,
            config=config,
            head_dim=head_dim,
            max_position_embeddings=config.max_position_embeddings,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1), persistent=True)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


class Gemma4MTPTextModel(PreTrainedModel):
    """Trunk over ``input_embeds`` from ``pre_projection``; no owned ``embed_tokens``."""

    config_class = PretrainedConfig

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        self.hidden_size = config.hidden_size

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
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        del input_ids
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


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
        self.text_config = text_config
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

        # Full-vocab logits per rank → ``skip_all_gather=True`` for dense and centroid paths.
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.logits_processor = LogitsProcessor(text_config, skip_all_gather=True)

        if self.use_ordered_embeddings:
            self.num_centroids = int(config.num_centroids)
            self.vocab_size_per_centroid = self.vocab_size // self.num_centroids
            if self.num_centroids * self.vocab_size_per_centroid != self.vocab_size:
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
            self.num_centroids = None
            self.vocab_size_per_centroid = None
            self.centroids = None
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

    def _run_trunk(
        self,
        token_embed: torch.Tensor,
        prev_hidden: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Trunk path for tests; returns norm-space hidden and post_projection state."""
        if token_embed.shape != prev_hidden.shape:
            raise ValueError(
                "Frozen-KV MTP forward: token_embed and prev_hidden must have "
                f"the same shape (got {token_embed.shape} vs {prev_hidden.shape})."
            )
        z, _ = self.pre_projection(torch.cat([token_embed, prev_hidden], dim=-1))
        hidden_states = self.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=z,
            **kwargs,
        )
        projected_states, _ = self.post_projection(hidden_states)
        return hidden_states, projected_states

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            token_embed = self.target_embed_tokens(input_ids)
        else:
            token_embed = input_embeds

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

        hidden_states, projected_states = self._run_trunk(
            token_embed, prev_hidden, positions, forward_batch, **kwargs
        )

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
            _,
            _,
        ) = self.logits_processor._get_pruned_states(
            hidden_states,
            projected_states,
            None,
            logits_metadata,
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
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        params_dict.update(dict(self.named_buffers()))

        non_persistent_buffers: Set[str] = set()
        for mod_name, mod in self.named_modules():
            for buf_name in getattr(mod, "_non_persistent_buffers_set", set()):
                full = f"{mod_name}.{buf_name}" if mod_name else buf_name
                non_persistent_buffers.add(full)

        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            if any(
                tag in name
                for tag in (
                    ".k_proj.",
                    ".v_proj.",
                    ".k_norm.",
                    ".v_norm.",
                )
            ):
                continue
            if (
                "rotary_emb.inv_freq" in name
                or "rotary_emb.cos_cached" in name
                or "rotary_emb.sin_cached" in name
            ):
                continue
            if name.startswith("target_embed_tokens."):
                continue

            name = name.replace("model.language_model.", "model.")
            orig_name = name

            if (
                orig_name.endswith("embed_tokens.weight")
                and getattr(self.config, "tie_word_embeddings", False)
                and self.lm_head.weight.shape == loaded_weight.shape
            ):
                default_weight_loader(self.lm_head.weight, loaded_weight)
                loaded_params.add("lm_head.weight")

            stacked = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                name = orig_name
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                stacked = True
                break

            if stacked:
                loaded_params.add(name)
                continue

            name = orig_name
            if name.endswith(".bias") and name not in params_dict:
                continue
            mapped = maybe_remap_kv_scale_name(name, params_dict)
            if mapped is None or mapped not in params_dict:
                continue
            param = params_dict[mapped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(mapped)

        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            param_names = set(dict(self.named_parameters()).keys())
            buckets = {
                logging.WARNING: (
                    "Some weights are not initialized from checkpoints",
                    lambda p: p in param_names
                    and not p.startswith("target_embed_tokens."),
                ),
                logging.INFO: (
                    "Persistent buffers not in checkpoint (using default init)",
                    lambda p: p not in param_names and p not in non_persistent_buffers,
                ),
                logging.DEBUG: (
                    "Non-persistent buffers not in checkpoint (expected)",
                    lambda p: p in non_persistent_buffers,
                ),
            }
            for level, (msg, pred) in buckets.items():
                names = sorted(p for p in unloaded_params if pred(p))
                if names:
                    logger.log(level, "%s: %s", msg, names)
        return loaded_params


class Gemma4MTPForCausalLM(Gemma4AssistantForCausalLM):
    pass


EntryClass = [Gemma4AssistantForCausalLM, Gemma4MTPForCausalLM]
