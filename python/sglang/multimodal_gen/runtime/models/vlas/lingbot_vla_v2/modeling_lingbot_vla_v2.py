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
"""LingBot-VLA 2.0 policy: assembles the dual-stream model, embeds the observation prefix and loads
checkpoint weights."""

from __future__ import annotations

from collections.abc import Iterable

import einops
import torch
from torch import Tensor, nn
from transformers import AutoConfig, PretrainedConfig

from .align_queries import AlignQueries
from .dual_stream import QwenvlWithExpertV2Config, QwenvlWithExpertV2Model
from .flow_matching import FlowMatching
from .utils import make_att_2d_masks


class LingBotVLAV2ForActionPrediction(AlignQueries, FlowMatching):
    def __init__(self, config, vlm_config: PretrainedConfig | None = None):
        super().__init__()
        self.config = config
        if vlm_config is None:
            vlm_config = AutoConfig.from_pretrained(
                config.tokenizer_path, revision=config.tokenizer_revision
            )
        qwenvl_with_export_config = QwenvlWithExpertV2Config(
            vocab_size=config.vocab_size,
            attention_implementation=config.attention_implementation,
            use_cache=config.use_cache,
            expert_hidden_size=config.expert_hidden_size,
            expert_intermediate_size=config.expert_intermediate_size,
            action_num_attention_heads=config.action_num_attention_heads,
            action_num_key_value_heads=config.action_num_key_value_heads,
            action_head_dim=config.action_head_dim,
            num_layers=vlm_config.text_config.num_hidden_layers,
        )
        for name in [
            "adanorm_time",
            "final_norm_adanorm",
            "vit_attn_implementation",
            "use_moe",
            "token_moe_layers",
            "token_num_experts",
            "token_top_k",
            "token_moe_intermediate_size",
            "token_shared_intermediate_size",
            "router_activation",
            "routed_scaling_factor",
            "use_shared_expert_gate",
        ]:
            setattr(qwenvl_with_export_config, name, getattr(config, name))
        self.qwenvl_with_expert = QwenvlWithExpertV2Model(
            qwenvl_with_export_config, vlm_config
        )
        self.proj_width = width = config.expert_hidden_size

        self.state_proj = nn.Linear(config.max_state_dim, width)
        self.action_in_proj = nn.Linear(config.max_action_dim, width)
        self.action_out_proj = nn.Linear(width, config.max_action_dim)
        self.action_time_mlp_in = nn.Linear(width * 2, width)
        self.action_time_mlp_out = nn.Linear(width, width)

        self._init_align(config.align_params, vlm_config.text_config.hidden_size)

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        image_grid_thw=None,
    ):
        if image_grid_thw is None:
            raise ValueError(
                "LingbotVLAV2Policy requires image_grid_thw from the Qwen3-VL image processor."
            )
        bsize = images.shape[0]
        device = images.device
        if images.ndim == 3:
            bsize = 1
            num_images = images.shape[0]
        else:
            num_images = images.shape[1] if images.ndim >= 4 else 1
        if images.ndim == 4:
            images = einops.rearrange(images, "b n l d -> (b n) l d")
        elif images.ndim == 5:
            images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        if image_grid_thw.ndim == 3:
            flat_grid_thw = einops.rearrange(image_grid_thw, "b n d -> (b n) d")
        else:
            flat_grid_thw = image_grid_thw

        img_emb, deepstack_embs = self.qwenvl_with_expert.get_image_features(
            images, flat_grid_thw
        )
        embed_dtype = img_emb.dtype
        num_patch = img_emb.shape[1]
        img_emb = einops.rearrange(
            img_emb, "(b n) l d -> b n l d", b=bsize, n=num_images
        )
        deepstack_embs = [
            einops.rearrange(x, "(b n) l d -> b n l d", b=bsize, n=num_images)
            for x in deepstack_embs
        ]
        if img_masks.ndim == 1:
            img_masks = img_masks.unsqueeze(0)

        cfg = self.qwenvl_with_expert.qwenvl.config
        visual_token_id = cfg.image_token_id

        start_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_start_token_id, bsize, num_images, device, embed_dtype
        )
        end_emb = self.qwenvl_with_expert.embed_special_token(
            cfg.vision_end_token_id, bsize, num_images, device, embed_dtype
        )
        img_chunks = torch.cat([start_emb, img_emb, end_emb], dim=2)
        image_token_len = num_patch + 2
        image_pad_masks = einops.repeat(img_masks, "b n -> b n l", l=image_token_len)
        image_visual_masks = torch.zeros_like(image_pad_masks)
        image_visual_masks[:, :, 1 : 1 + num_patch] = einops.repeat(
            img_masks, "b n -> b n l", l=num_patch
        )
        fake_image_ids = torch.full(
            (bsize, num_images, image_token_len),
            visual_token_id,
            dtype=torch.long,
            device=device,
        )
        fake_image_ids[:, :, 0] = cfg.vision_start_token_id
        fake_image_ids[:, :, -1] = cfg.vision_end_token_id
        # note (0xtoward): missing cameras have no grid entry for MRoPE.
        fake_image_ids = fake_image_ids.masked_fill(
            ~img_masks[..., None], cfg.text_config.eos_token_id
        )

        img_emb = einops.rearrange(img_chunks, "b n l d -> b (n l) d")
        image_pad_masks = einops.rearrange(image_pad_masks, "b n l -> b (n l)")
        visual_pos_masks = einops.rearrange(image_visual_masks, "b n l -> b (n l)")
        fake_image_ids = einops.rearrange(fake_image_ids, "b n l -> b (n l)")

        lang_emb = self.qwenvl_with_expert.embed_language_tokens(lang_tokens).to(
            dtype=embed_dtype
        )

        embs = [img_emb, lang_emb]
        pad_masks = [image_pad_masks, lang_masks]
        prefix_input_ids = [fake_image_ids, lang_tokens.to(device)]
        full_visual_pos_masks = [visual_pos_masks, torch.zeros_like(lang_masks)]
        if self.use_align:
            queries = self._align_query_tokens().to(embed_dtype).expand(bsize, -1, -1)
            query_masks = lang_masks.new_ones(bsize, queries.shape[1])
            embs.append(queries)
            pad_masks.append(query_masks)
            prefix_input_ids.append(
                torch.full_like(
                    query_masks, cfg.text_config.eos_token_id, dtype=torch.long
                )
            )
            full_visual_pos_masks.append(torch.zeros_like(query_masks))
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        prefix_input_ids = torch.cat(prefix_input_ids, dim=1)
        full_visual_pos_masks = torch.cat(full_visual_pos_masks, dim=1)

        if self.config.vlm_causal:
            att_masks = torch.ones(
                (bsize, embs.shape[1]), device=device, dtype=torch.bool
            )
        else:
            att_masks = torch.zeros(
                (bsize, embs.shape[1]), device=device, dtype=torch.bool
            )

        rope_grid_thw = flat_grid_thw[einops.rearrange(img_masks, "b n -> (b n)")]
        prefix_position_ids = self.qwenvl_with_expert.build_prefix_position_ids(
            prefix_input_ids,
            pad_masks.long(),
            image_grid_thw=rope_grid_thw,
            video_grid_thw=None,
        )
        # note (0xtoward): zero-padding lets DeepStack add features without indexing a variable number of tokens.
        img_visual_only = einops.repeat(img_masks, "b n -> b n l", l=num_patch)
        img_len = img_emb.shape[1]
        tail_len = embs.shape[1] - img_len
        dense_deepstack = []
        for deepstack in deepstack_embs:
            level = deepstack * img_visual_only.unsqueeze(-1).to(deepstack.dtype)
            wrapped = level.new_zeros(
                bsize, num_images, image_token_len, level.shape[-1]
            )
            wrapped[:, :, 1 : 1 + num_patch] = (
                level  # patches sit between the vision boundaries
            )
            level = einops.rearrange(wrapped, "b n l d -> b (n l) d")
            if tail_len > 0:  # language (and query) tail stays zero
                level = torch.cat(
                    [level, level.new_zeros(bsize, tail_len, level.shape[-1])], dim=1
                )
            dense_deepstack.append(level.to(embed_dtype))

        result = (
            embs,
            pad_masks,
            att_masks,
            prefix_position_ids,
            full_visual_pos_masks,
            dense_deepstack,
        )
        return result

    def _embed_and_fill_prefix(
        self, images, img_masks, lang_tokens, lang_masks, image_grid_thw
    ):
        """Prefix half of sample_actions: embed_prefix
        (vision tower + language embedding + mrope position ids) followed by the
        36-layer KV fill. Returns exactly what the denoise loop consumes."""
        (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            prefix_position_ids,
            visual_pos_masks,
            deepstack_visual_embeds,
        ) = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            image_grid_thw=image_grid_thw,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        _, past_key_values = self.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return prefix_pad_masks, prefix_position_ids, past_key_values

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        expected = self.state_dict()
        loaded = set()
        skipped_heads = (
            "depth_align_head.",
            "future_depth_align_head.",
            "future_video_align_head.",
            "current_video_align_head.",
            "future_video_cls_head.",
        )
        for original_name, value in weights:
            name = original_name.removeprefix("model.")
            if name.startswith(skipped_heads):
                continue
            if name not in expected:
                raise ValueError(
                    f"Unexpected LingBot inference weight: {original_name}"
                )
            if name in loaded or expected[name].shape != value.shape:
                raise ValueError(
                    f"Duplicate or incompatible LingBot weight: {original_name}"
                )
            with torch.no_grad():
                expected[name].copy_(
                    value.to(expected[name].device, expected[name].dtype)
                )
            loaded.add(name)
        missing = set(expected) - loaded
        if missing:
            raise ValueError(
                f"Missing LingBot inference weights: {sorted(missing)[:20]}"
            )
        return loaded
