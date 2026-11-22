# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
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

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
"""Joint Qwen3-VL and action-expert transformer of LingBot-VLA 2.0: embeddings, MRoPE positions,
shared attention over both streams and the prefix KV cache."""

from __future__ import annotations

import logging

import torch
from torch import Tensor
from transformers import PretrainedConfig, PreTrainedModel
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb

from .qwen2_action_expert import (
    Qwen2ActionExpert,
    Qwen2TokenMoeBlock,
    replace_lnorm_with_adanorm,
)
from .qwen3vl_in_vla import Qwen3VLForConditionalGeneration
from .utils import our_eager_attention_forward, our_sdpa_attention_forward

logger = logging.getLogger(__name__)


class QwenvlWithExpertV2Config(PretrainedConfig):
    model_type = "QwenvlWithExpertV2Model"

    def __init__(
        self,
        vocab_size: int = 0,
        attention_implementation: str = "sdpa",
        use_cache: bool = False,
        expert_hidden_size: int = 768,
        expert_intermediate_size: int = 2752,
        action_num_attention_heads: int = 32,
        action_num_key_value_heads: int = 8,
        action_head_dim: int = 128,
        num_layers: int = 36,
        **kwargs,
    ):
        self.attention_implementation = attention_implementation
        self.vocab_size = vocab_size
        self.action_num_attention_heads = action_num_attention_heads
        self.action_num_key_value_heads = action_num_key_value_heads
        self.action_head_dim = action_head_dim

        self.qwen_expert_config = CONFIG_MAPPING["qwen2"](
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151645,
            hidden_act="silu",
            hidden_size=expert_hidden_size,
            head_dim=action_head_dim,
            initializer_range=0.02,
            intermediate_size=expert_intermediate_size,
            max_position_embeddings=32768,
            max_window_layers=21,
            model_type="qwen2",
            num_attention_heads=action_num_attention_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=action_num_key_value_heads,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=32768,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            transformers_version="4.57.3",
            use_cache=use_cache,
            use_sliding_window=False,
            vocab_size=151936,
        )
        super().__init__(**kwargs)


class QwenvlWithExpertV2Model(PreTrainedModel):
    config_class = QwenvlWithExpertV2Config

    def __init__(self, config: QwenvlWithExpertV2Config, vlm_config: PretrainedConfig):
        super().__init__(config=config)
        self.config = config
        # note (0xtoward): the dual-stream forward selects text attention separately from the vision tower.
        hf_attn = "eager"
        hf_vit_attn = self.config.vit_attn_implementation
        if self.config.vocab_size:
            vlm_config.text_config.vocab_size = self.config.vocab_size
        vlm_config._attn_implementation = hf_attn
        vlm_config.text_config._attn_implementation = hf_attn
        vlm_config.vision_config._attn_implementation = hf_vit_attn
        self.qwenvl = Qwen3VLForConditionalGeneration._from_config(vlm_config)

        self.config.qwen_expert_config._attn_implementation = hf_attn
        self.qwen_expert = Qwen2ActionExpert._from_config(
            self.config.qwen_expert_config
        )

        if self.config.adanorm_time:
            replace_lnorm_with_adanorm(
                self.qwen_expert,
                self.config.qwen_expert_config.hidden_size,
                self.config.qwen_expert_config.hidden_size,
                config.final_norm_adanorm,
            )

        self._install_moe_blocks()
        self.attention_interface = self.get_attention_interface()

    def _install_moe_blocks(self):
        if not self.config.use_moe:
            return
        layers = self.qwen_expert.model.layers
        token_moe_layers = [
            idx for idx in self.config.token_moe_layers if idx < len(layers)
        ]

        if token_moe_layers:
            token_config = CONFIG_MAPPING["qwen2_moe"](
                num_experts=self.config.token_num_experts,
                num_experts_per_tok=self.config.token_top_k,
                norm_topk_prob=True,
                hidden_size=self.config.qwen_expert_config.hidden_size,
                moe_intermediate_size=self.config.token_moe_intermediate_size,
                shared_expert_intermediate_size=self.config.token_shared_intermediate_size,
                output_router_logits=False,
            )
            token_config.router_activation = self.config.router_activation
            token_config.routed_scaling_factor = self.config.routed_scaling_factor
            token_config.use_shared_expert_gate = self.config.use_shared_expert_gate
            for idx in token_moe_layers:
                layers[idx].mlp = Qwen2TokenMoeBlock(token_config)

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.LongTensor,
    ):
        image_embeds, deepstack_image_embeds = self.qwenvl.model.visual(
            pixel_values, grid_thw=image_grid_thw
        )
        split_sizes = (
            image_grid_thw.prod(-1) // self.qwenvl.model.visual.spatial_merge_size**2
        ).tolist()
        image_chunks = list(torch.split(image_embeds, split_sizes))
        deepstack_chunks = [
            list(torch.split(deepstack_embeds, split_sizes))
            for deepstack_embeds in deepstack_image_embeds
        ]
        image_embeds = torch.stack(image_chunks, dim=0)
        deepstack_image_embeds = [
            torch.stack(chunks, dim=0) for chunks in deepstack_chunks
        ]
        return image_embeds, deepstack_image_embeds

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.qwenvl.model.language_model.embed_tokens(tokens)

    def embed_special_token(self, token_id: int, batch: int, count: int, device, dtype):
        token = torch.tensor([token_id], device=device, dtype=torch.long)
        emb = self.embed_language_tokens(token).to(dtype=dtype)
        return emb.view(1, 1, 1, -1).expand(batch, count, 1, -1)

    def build_prefix_position_ids(
        self, input_ids, attention_mask, image_grid_thw=None, video_grid_thw=None
    ):
        # note (0xtoward): get_rope_index needs explicit modality IDs in Transformers 5.
        vlm_cfg = self.qwenvl.config
        image_token_id = getattr(vlm_cfg, "image_token_id", None)
        video_token_id = getattr(vlm_cfg, "video_token_id", None)
        mm_token_type_ids = torch.zeros_like(input_ids)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        position_ids, _ = self.qwenvl.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        # note (0xtoward): the checkpoint expects the padding positions used by Transformers 4.57.
        if attention_mask is not None:
            pad = attention_mask == 0
            while pad.dim() < position_ids.dim():
                pad = pad.unsqueeze(0)
            position_ids = position_ids.masked_fill(pad.expand_as(position_ids), 1)
        return position_ids

    def apply_mrope(
        self, query_states, key_states, position_ids=None, position_embeddings=None
    ):
        if position_embeddings is None:
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(
                query_states, position_ids
            )
        return apply_rotary_pos_emb(
            query_states, key_states, *position_embeddings, unsqueeze_dim=2
        )

    def handle_kv_cache(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        if use_cache:
            if past_key_values is None:
                past_key_values = {}
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                key_states = torch.cat(
                    [past_key_values[layer_idx]["key_states"], key_states], dim=1
                )
                value_states = torch.cat(
                    [past_key_values[layer_idx]["value_states"], value_states], dim=1
                )
        return key_states, value_states, past_key_values

    def _apply_deepstack(
        self, hidden_states, layer_idx, visual_pos_masks, deepstack_visual_embeds
    ):
        """Add the level's dense deepstack delta."""
        if (
            deepstack_visual_embeds is not None
            and visual_pos_masks is not None
            and layer_idx < len(deepstack_visual_embeds)
        ):
            hidden_states = hidden_states + deepstack_visual_embeds[layer_idx].to(
                hidden_states.dtype
            )
        return hidden_states

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: dict[int, dict[str, Tensor]] | None = None,
        inputs_embeds: list[Tensor | None] | None = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        ada_cond: Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        assert inputs_embeds is not None
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        num_layers = self.qwenvl.config.text_config.num_hidden_layers

        # note (0xtoward): all layers share the same rotary embeddings.
        if position_embeddings is None:
            rep = next(h for h in inputs_embeds if h is not None)
            position_embeddings = self.qwenvl.model.language_model.rotary_emb(
                rep, position_ids
            )

        for layer_idx in range(num_layers):
            inputs_embeds, past_key_values = self._layer_forward(
                layer_idx,
                inputs_embeds,
                attention_mask,
                position_embeddings,
                past_key_values,
                use_cache,
                fill_kv_cache,
                ada_cond,
                visual_pos_masks,
                deepstack_visual_embeds,
            )

        outputs_embeds: list[torch.FloatTensor | None] = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
            elif self.config.final_norm_adanorm and i == 1:
                out_emb = models[i].norm(hidden_states, ada_cond)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(models[i].norm(hidden_states))
        return outputs_embeds, past_key_values

    def _layer_forward(
        self,
        layer_idx,
        inputs_embeds,
        attention_mask,
        position_embeddings,
        past_key_values,
        use_cache,
        fill_kv_cache,
        ada_cond,
        visual_pos_masks,
        deepstack_visual_embeds,
    ):
        """One dual-stream layer: per-stream QKV -> joint attention -> per-stream out/MLP."""
        models = [self.qwenvl.model.language_model, self.qwen_expert.model]
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                continue
            if i == 1:
                q, k, v = models[i].layers[layer_idx](
                    hidden_states, compute_kqv=True, ada_cond=ada_cond
                )
            else:
                q, k, v = models[i].layers[layer_idx](hidden_states, compute_kqv=True)
            query_states.append(q)
            key_states.append(k)
            value_states.append(v)

        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        query_states, key_states = self.apply_mrope(
            query_states, key_states, position_embeddings=position_embeddings
        )
        key_states, value_states, past_key_values = self.handle_kv_cache(
            key_states,
            value_states,
            layer_idx,
            past_key_values=past_key_values,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        att_output = self.attention_interface(
            query_states, key_states, value_states, attention_mask
        )

        outputs_embeds = []
        start = 0
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is None:
                outputs_embeds.append(None)
                continue
            end = start + hidden_states.shape[1]
            if i == 1:
                out_emb = models[i].layers[layer_idx](
                    hidden_states,
                    att_output,
                    start,
                    end,
                    output_atten=True,
                    ada_cond=ada_cond,
                )
            else:
                out_emb = models[i].layers[layer_idx](
                    hidden_states, att_output, start, end, output_atten=True
                )
                out_emb = self._apply_deepstack(
                    out_emb, layer_idx, visual_pos_masks, deepstack_visual_embeds
                )
            outputs_embeds.append(out_emb)
            start = end
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "sdpa":
            return our_sdpa_attention_forward
        if self.config.attention_implementation == "eager":
            logger.debug("Using Eager attention")
            return our_eager_attention_forward
        raise ValueError(
            f"Invalid attention implementation: {self.config.attention_implementation}"
        )
