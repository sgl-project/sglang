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

import copy
import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.mem_cache.memory_pool import KVCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.gemma4_causal import Gemma4ForCausalLM, Gemma4TextModel
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPContext
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


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


class Gemma4AssistantForCausalLM(Gemma4ForCausalLM):
    """Gemma 4 MTP assistant: target embed + recurrent hidden through pre/post projection; own ``lm_head``."""

    base_model_prefix = "model"

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        text_config = copy.deepcopy(_get_text_config(config))
        text_config.num_kv_shared_layers = 0
        PreTrainedModel.__init__(self, config=text_config)
        self.assistant_config = config
        self.config = text_config
        self.quant_config = quant_config

        self.vocab_size = text_config.vocab_size
        self.hidden_size = text_config.hidden_size
        self.backbone_hidden_size = config.backbone_hidden_size
        self.target_embed_scale = self.backbone_hidden_size**0.5
        self.use_ordered_embeddings = bool(
            getattr(config, "use_ordered_embeddings", False)
        )
        self.centroid_intermediate_top_k = int(
            getattr(config, "centroid_intermediate_top_k", 32)
        )

        self.target_embed_weight: Optional[torch.Tensor] = None
        self.pre_projection = ReplicatedLinear(
            2 * self.backbone_hidden_size,
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix=add_prefix("pre_projection", prefix),
        )
        self.model = Gemma4TextModel(
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

        if text_config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
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
        """Bind assistant attention to target-owned KV and suppress assistant KV writes."""
        for assistant_logical, layer in enumerate(self.model.layers):
            target_phys = ctx.get_physical_layer_id(assistant_logical)
            layer.self_attn.is_kv_shared_layer = True
            layer.self_attn.kv_shared_layer_index = target_phys
            layer.self_attn.attn.layer_id = target_phys
            layer.self_attn.layer_id = assistant_logical
        self.kv_context = ctx

    def build_frozen_kv_mtp_context(
        self,
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
        assistant_text = _get_text_config(self)
        layers = _resolve_target_text_model(target_model).layers

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
                    f"{owner!r}, which is missing or itself KV-shared "
                    "(HF invariant changed?)."
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

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.target_embed_weight is None:
            raise RuntimeError(
                "Gemma4AssistantForCausalLM target embedding is not bound yet."
            )
        return self.target_embed_weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        """Rebind target embedding; ``head`` ignored (assistant keeps ``lm_head``)."""
        del head
        self.target_embed_weight = embed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_attention_sliding_window_size(self) -> int:
        # Gemma 4 config treats the bound as inclusive; SGLang attention metadata
        # uses an exclusive window size, matching the target Gemma 4 models.
        return self.config.sliding_window - 1

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
            if self.target_embed_weight is None:
                raise RuntimeError(
                    "Gemma4AssistantForCausalLM requires set_embed_and_head() "
                    "before token-id forward."
                )
            token_embed = (
                torch.nn.functional.embedding(input_ids, self.target_embed_weight)
                * self.target_embed_scale
            )
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
            per_layer_inputs=None,
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
        """Centroid-masked logits for E2B/E4B assistant heads."""
        if self.centroids is None or self.token_ordering is None:
            raise RuntimeError(
                "Frozen-KV MTP centroid head invoked but centroid weights "
                "are not initialized."
            )
        prefix_shape = hidden_states.shape[:-1]
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
        num_tokens = flat_hidden.shape[0]

        _, top_k_indices = torch.topk(
            self.centroids(flat_hidden),
            k=self.centroid_intermediate_top_k,
            dim=-1,
        )

        # Contiguous gather: [C, vpc, H] indexed by centroid IDs.
        num_selected = self.centroid_intermediate_top_k * self.vocab_size_per_centroid
        selected_embeddings = self.lm_head.weight.view(
            self.num_centroids,
            self.vocab_size_per_centroid,
            self.hidden_size,
        )[top_k_indices].reshape(num_tokens, num_selected, self.hidden_size)

        selected_logits = torch.bmm(
            flat_hidden.unsqueeze(1),
            selected_embeddings.transpose(1, 2),
        ).squeeze(1)

        # Scatter to real vocab positions via token_ordering.
        centroid_vocab_indices = (
            self.token_ordering.long()
            .view(self.num_centroids, self.vocab_size_per_centroid)[top_k_indices]
            .view(num_tokens, -1)
        )
        mask_value = torch.finfo(selected_logits.dtype).min / 2
        output = torch.full(
            (num_tokens, self.vocab_size),
            mask_value,
            dtype=selected_logits.dtype,
            device=selected_logits.device,
        )
        output.scatter_(dim=-1, index=centroid_vocab_indices, src=selected_logits)
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

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        def remap_assistant_weights():
            for name, weight in weights:
                if name.startswith("masked_embedding."):
                    name = name.removeprefix("masked_embedding.")
                yield name, weight

        result = super().load_weights(remap_assistant_weights())
        if self.use_ordered_embeddings:
            self._reorder_embedding_to_centroid_order()
        return result

    @torch.no_grad()
    def _reorder_embedding_to_centroid_order(self) -> None:
        """Reorder lm_head.weight from natural vocab order to centroid order."""
        if self.token_ordering is None:
            return
        ordering = self.token_ordering.long()
        lm_head_w = self.lm_head.weight
        reordered = lm_head_w.data[ordering]
        lm_head_w.data.copy_(reordered)
        logger.info(
            "Reordered lm_head/embed_tokens (%s) to centroid order "
            "for contiguous centroid masking.",
            list(lm_head_w.shape),
        )


EntryClass = Gemma4AssistantForCausalLM
