# Copyright 2023-2024 SGLang Team
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

# Modeling from:
# ./llama.py and
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/GlmOcr/modular_GlmOcr.py
"""Inference-only GLM-OCR model compatible with HuggingFace weights."""

import logging
from functools import lru_cache
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.models.glm_ocr.configuration_glm_ocr import (
    GlmOcrConfig,
    GlmOcrTextConfig,
    GlmOcrVisionConfig,
)

from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.attention import vision_utils
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader, WeightsMapper
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.glm4 import Glm4Model
from sglang.srt.models.glm4v import (
    Glm4vForConditionalGeneration,
    Glm4vPatchMerger,
    Glm4vRMSNorm,
    Glm4vVisionMLP,
    Glm4vVisionModel,
    Glm4vVisionPatchEmbed,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix
from sglang.srt.utils.hf_transformers_utils import get_processor

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class GlmOcrRMSNorm(Glm4vRMSNorm):
    pass


class GlmOcrVisionMLP(Glm4vVisionMLP):
    pass


class GlmOcrVisionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        num_heads: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        attn_qkv_bias: bool = True,
        num_dummy_heads: int = 0,
        rms_norm_eps: float = 1e-5,
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=rms_norm_eps)
        self.norm2 = RMSNorm(dim, eps=rms_norm_eps)
        self.attn = VisionAttention(
            embed_dim=dim,
            num_heads=num_heads,
            projection_size=dim,
            use_qkv_parallel=True,
            qkv_bias=attn_qkv_bias,
            proj_bias=True,
            qk_normalization_by_head_size=True,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
            num_dummy_heads=num_dummy_heads,
            use_data_parallel=use_data_parallel,
        )
        self.mlp = GlmOcrVisionMLP(
            dim,
            intermediate_dim,
            bias=True,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
            use_data_parallel=use_data_parallel,
        )

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        S, B, H = x.shape
        # norm1: flatten to 2D -> [S*B, H], then reshape back
        x2d = x.reshape(-1, H)
        hidden_states = self.norm1(x2d).reshape(S, B, H)

        # Attention expects [B, S, H]
        hidden_states = rearrange(hidden_states, "s b h -> b s h")
        attn = self.attn(
            hidden_states,
            cu_seqlens=cu_seqlens,
            rotary_pos_emb_cos=rotary_pos_emb_cos,
            rotary_pos_emb_sin=rotary_pos_emb_sin,
        )
        attn = rearrange(attn, "b s h -> s b h")

        # norm2 with fused residual-add: also 2D
        attn2d = attn.reshape(-1, H)
        x_norm_2d, x_after_add_2d = self.norm2(x2d, residual=attn2d)
        x_norm = x_norm_2d.reshape(S, B, H)
        x_after_add = x_after_add_2d.reshape(S, B, H)

        # MLP and final residual
        mlp_out = self.mlp(x_norm)
        x = x_after_add + mlp_out
        return x


class GlmOcrVisionPatchEmbed(Glm4vVisionPatchEmbed):
    pass


class GlmOcrVisionPatchMerger(Glm4vPatchMerger):
    pass


class GlmOcrVisionModel(Glm4vVisionModel):
    def __init__(
        self,
        vision_config: GlmOcrVisionConfig,
        text_config: GlmOcrTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        use_data_parallel: bool = False,
    ) -> None:
        super().__init__(vision_config, quant_config, prefix, use_data_parallel)

        patch_size = vision_config.patch_size
        temporal_patch_size = vision_config.temporal_patch_size
        in_channels = vision_config.in_channels
        depth = vision_config.depth
        self.hidden_size = vision_config.hidden_size
        self.num_heads = vision_config.num_heads

        self.patch_size = vision_config.patch_size
        self.spatial_merge_size = vision_config.spatial_merge_size
        self.out_hidden_size = vision_config.out_hidden_size
        self.intermediate_size = vision_config.intermediate_size
        self.use_data_parallel = use_data_parallel

        self.patch_embed = GlmOcrVisionPatchEmbed(
            patch_size=patch_size,
            temporal_patch_size=temporal_patch_size,
            in_channels=in_channels,
            hidden_size=self.hidden_size,
        )

        head_dim = self.hidden_size // self.num_heads
        self.rotary_pos_emb = get_rope(
            head_size=head_dim,
            rotary_dim=head_dim // 2,
            max_position=8192,
            base=10000.0,
            is_neox_style=True,
        )

        self.blocks = nn.ModuleList(
            [
                GlmOcrVisionBlock(
                    dim=self.hidden_size,
                    intermediate_dim=self.intermediate_size,
                    num_heads=self.num_heads,
                    quant_config=quant_config,
                    prefix=add_prefix(f"blocks.{layer_idx}", prefix),
                    rms_norm_eps=vision_config.rms_norm_eps,
                    attn_qkv_bias=vision_config.attention_bias,
                    use_data_parallel=use_data_parallel,
                )
                for layer_idx in range(depth)
            ]
        )
        self.merger = GlmOcrVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=text_config.intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=add_prefix("merger", prefix),
            use_data_parallel=use_data_parallel,
        )

        self.downsample = nn.Conv2d(
            in_channels=vision_config.hidden_size,
            out_channels=vision_config.out_hidden_size,
            kernel_size=vision_config.spatial_merge_size,
            stride=vision_config.spatial_merge_size,
        )
        self.post_layernorm = GlmOcrRMSNorm(
            vision_config.hidden_size, eps=vision_config.rms_norm_eps
        )

    def forward(self, x: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        # patchify
        x = x.to(device=self.device, dtype=self.dtype)
        x = self.patch_embed(x)

        # compute position embedding
        rotary_pos_emb_cos, rotary_pos_emb_sin, image_type_ids = self.rot_pos_emb(
            grid_thw
        )
        # compute cu_seqlens
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        rotary_pos_emb_cos = torch.cat([rotary_pos_emb_cos, rotary_pos_emb_cos], dim=-1)
        rotary_pos_emb_sin = torch.cat([rotary_pos_emb_sin, rotary_pos_emb_sin], dim=-1)

        # x.shape: (s, b, d) where b=1 for vision processing
        # transformers
        x = x.unsqueeze(1)
        for blk in self.blocks:
            x = blk(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )

        # adapter
        x = self.post_layernorm(x)
        x = x.view(-1, self.spatial_merge_size, self.spatial_merge_size, x.shape[-1])
        x = x.permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        x = self.merger(x)

        return x


class GlmOcrForConditionalGeneration(Glm4vForConditionalGeneration):
    def __init__(
        self,
        config: GlmOcrConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)

        self.pp_group = get_pp_group()
        self.config = config
        self.use_data_parallel = get_global_server_args().mm_enable_dp_encoder
        self.visual = GlmOcrVisionModel(
            vision_config=config.vision_config,
            text_config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("visual", prefix),
            use_data_parallel=self.use_data_parallel,
        )

        vision_utils.update_vit_attn_dummy_heads_config(self.config)

        self.model = Glm4Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )

        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and self.config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    self.config.vocab_size,
                    self.config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # ranks other than the last rank will have a placeholder layer
            self.lm_head = PPMissingLayer()

        self.is_mrope_enabled = "mrope_section" in self.config.rope_scaling

        self.logits_processor = LogitsProcessor(config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    def load_weights(
        self,
        weights: Iterable[Tuple[str, torch.Tensor]],
        is_nextn: bool = False,
    ) -> set[str]:
        if is_nextn:
            return self._load_nextn_weights(weights)

        # Drop next-token-prediction (NextN) layers from the main weight stream
        # — they live at layer indices >= num_hidden_layers and only get loaded
        # when ``is_nextn=True``.
        num_nextn_layers = getattr(self.config, "num_nextn_predict_layers", 0)
        num_hidden_layers = self.config.num_hidden_layers

        def _filter_nextn(stream):
            for name, w in stream:
                if num_nextn_layers > 0 and name.startswith("model.layers"):
                    parts = name.split(".")
                    if len(parts) >= 3 and int(parts[2]) >= num_hidden_layers:
                        continue
                yield name, w

        # Pad vision-tower attention weights for dummy heads at the stream
        # boundary (no-op when num_dummy_heads == 0). Pad uses the post-rename
        # name (visual.* + qkv_proj).
        def _pad_visual(stream):
            for name, w in stream:
                if "visual" in name or "model.visual" in name:
                    renamed = name.replace("model.visual.", "visual.").replace(
                        "attn.qkv.", "attn.qkv_proj."
                    )
                    yield name, vision_utils.pad_vit_attn_dummy_heads(
                        self.config, renamed, w
                    )
                else:
                    yield name, w

        # Checkpoint conventions:
        #   - transformers>=4.52: ``model.language_model.<...>`` / ``model.visual.<...>``
        #   - original release:    ``model.<...>``                / ``visual.<...>``
        # Map both onto runtime python attributes ``self.model`` / ``self.visual``.
        # Vision attention uses ``attn.qkv``; runtime ``VisionAttention``
        # exposes ``attn.qkv_proj``.
        mapper = WeightsMapper(
            orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
            orig_to_new_prefix={
                "model.language_model.": "model.",
                "model.visual.": "visual.",
            },
        )
        return AutoWeightsLoader(self).load_weights(
            _pad_visual(_filter_nextn(weights)), mapper=mapper
        )

    def _load_nextn_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> set[str]:
        """Legacy hand-rolled loader for the next-token-prediction shard.

        NextN weights live at ``model.layers.{nextn_layer_id}.*`` in the
        checkpoint and need to be remapped onto ``model.<...>`` (specific
        names) and ``model.decoder.<...>`` (everything else). This path is
        rare and architecture-specific, so we preserve the explicit loop.
        """
        if not hasattr(self.config, "num_nextn_predict_layers"):
            raise ValueError("num_nextn_predict_layers is not in the config")
        num_nextn_layers = self.config.num_nextn_predict_layers
        assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
        nextn_layer_id = (
            0 if self.config.num_hidden_layers == 1 else self.config.num_hidden_layers
        )
        nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
        nextn_spec_weight_names = [
            "shared_head.norm",
            "eh_proj",
            "enorm",
            "hnorm",
        ]
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if not name.startswith(nextn_layer_prefix):
                continue
            if "shared_head.head" in name or "embed_tokens" in name:
                continue
            is_decoder = True
            for weight_name in nextn_spec_weight_names:
                if weight_name in name:
                    name = name.replace(nextn_layer_prefix, "model")
                    is_decoder = False
                    break
            if is_decoder:
                name = name.replace(nextn_layer_prefix, "model.decoder")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded.add(name)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)
        return loaded


EntryClass = [GlmOcrForConditionalGeneration]
