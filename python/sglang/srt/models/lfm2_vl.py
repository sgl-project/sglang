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
"""Inference-only LFM2-VL model compatible with HuggingFace weights."""

import json
import logging
import time
from typing import Iterable, List, Optional, Tuple

import torch

# #region agent log
_DEBUG_LOG_PATH = "/sgl-workspace/sglang/.cursor/debug.log"
_DEBUG_SESSION = "lfm2vl-debug"
_DEBUG_CALL_COUNT = 0


def _dbg(hypothesis_id, location, message, data=None):
    global _DEBUG_CALL_COUNT
    _DEBUG_CALL_COUNT += 1
    entry = {
        "sessionId": _DEBUG_SESSION,
        "runId": "initial",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": time.time(),
        "callNum": _DEBUG_CALL_COUNT,
    }
    try:
        import os

        os.makedirs(os.path.dirname(_DEBUG_LOG_PATH), exist_ok=True)
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


# #endregion
from torch import nn

from sglang.srt.configs.lfm2_vl import Lfm2VlConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.lfm2 import Lfm2ForCausalLM
from sglang.srt.utils import add_prefix
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.auto.modeling_auto import AutoModel

logger = logging.getLogger(__name__)


class Lfm2VlMultiModalProjector(nn.Module):
    """Multimodal projector with pixel unshuffle downsampling."""

    def __init__(self, config: Lfm2VlConfig):
        super().__init__()
        in_channels = config.vision_config.hidden_size * (config.downsample_factor**2)
        self.factor = config.downsample_factor
        self.use_layer_norm = config.projector_use_layernorm
        self.layer_norm = (
            nn.LayerNorm(in_channels) if config.projector_use_layernorm else None
        )
        self.linear_1 = nn.Linear(
            in_channels,
            config.projector_hidden_size,
            bias=config.projector_bias,
        )
        self.act = ACT2FN[config.projector_hidden_act]
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text_config.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, image_features: torch.Tensor):
        image_features = self.pixel_unshuffle(image_features)
        if self.use_layer_norm:
            image_features = self.layer_norm(image_features)
        hidden_states = self.linear_1(image_features)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

    def pixel_unshuffle(self, hidden_states: torch.Tensor):
        batch_size, width, height, channels = hidden_states.size()
        hidden_states = hidden_states.reshape(
            batch_size, width, height // self.factor, channels * self.factor
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(
            batch_size,
            height // self.factor,
            width // self.factor,
            channels * self.factor**2,
        )
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        return hidden_states


class Lfm2VlForConditionalGeneration(PreTrainedModel):
    config_class = Lfm2VlConfig

    def __init__(
        self,
        config: Lfm2VlConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        # Vision tower: SigLip2 via HF AutoModel
        self.vision_tower = AutoModel.from_config(config=config.vision_config)

        # Multimodal projector
        self.multi_modal_projector = Lfm2VlMultiModalProjector(config)

        # Language model: reuse sglang's LFM2 implementation
        self.language_model = Lfm2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        self.logits_processor = LogitsProcessor(config.text_config)
        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        # #region agent log
        _dbg(
            "H5",
            "pad_input_ids:entry",
            "Padding input_ids for mm tokens",
            {
                "input_ids_len_before": len(input_ids),
                "image_token_id": self.config.image_token_id,
                "image_token_count_before": sum(
                    1 for t in input_ids if t == self.config.image_token_id
                ),
                "mm_items_count": (
                    len(mm_inputs.mm_items) if hasattr(mm_inputs, "mm_items") else 0
                ),
            },
        )
        # #endregion
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        result = pattern.pad_input_tokens(input_ids, mm_inputs)
        # #region agent log
        _dbg(
            "H5",
            "pad_input_ids:exit",
            "After padding",
            {
                "input_ids_len_after": len(result),
                "image_token_count_after": sum(
                    1 for t in result if t == self.config.image_token_id
                ),
            },
        )
        # #endregion
        return result

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Process images through vision tower and projector.

        Handles SigLip2's NaFlex variable-resolution output by unpadding
        features using the attention mask and reshaping per spatial_shapes.
        """
        # #region agent log
        _dbg(
            "H3",
            "get_image_feature:entry",
            "Called with items",
            {
                "num_items": len(items),
                "item_modalities": [str(item.modality) for item in items],
                "item_has_feature": [item.feature is not None for item in items],
                "item_has_attn_mask": [
                    hasattr(item, "pixel_attention_mask")
                    and item.pixel_attention_mask is not None
                    for item in items
                ],
                "item_has_spatial": [
                    hasattr(item, "spatial_shapes") and item.spatial_shapes is not None
                    for item in items
                ],
            },
        )
        # #endregion
        all_pixel_values = flatten_nested_list([item.feature for item in items])
        all_pixel_attention_masks = flatten_nested_list(
            [item.pixel_attention_mask for item in items]
        )
        all_spatial_shapes = flatten_nested_list(
            [item.spatial_shapes for item in items]
        )

        image_features_list = []

        for pixel_values_batch, attn_mask_batch, shapes_batch in zip(
            all_pixel_values, all_pixel_attention_masks, all_spatial_shapes
        ):
            # Normalize shapes
            if pixel_values_batch.dim() == 2:
                pixel_values_batch = pixel_values_batch.unsqueeze(0)
            if attn_mask_batch.dim() == 1:
                attn_mask_batch = attn_mask_batch.unsqueeze(0)
            if shapes_batch.dim() == 1:
                shapes_batch = shapes_batch.unsqueeze(0)

            # #region agent log
            _dbg(
                "H4",
                "get_image_feature:pre_vit",
                "Tensors before vision tower",
                {
                    "pv_shape": list(pixel_values_batch.shape),
                    "pv_dtype": str(pixel_values_batch.dtype),
                    "pv_device": str(pixel_values_batch.device),
                    "pv_mean": float(pixel_values_batch.float().mean()),
                    "pv_std": float(pixel_values_batch.float().std()),
                    "attn_shape": list(attn_mask_batch.shape),
                    "attn_sum_per_img": attn_mask_batch.sum(dim=1).tolist(),
                    "shapes_batch": shapes_batch.tolist(),
                    "vt_device": str(self.vision_tower.device),
                    "vt_dtype": str(self.vision_tower.dtype),
                },
            )
            # #endregion

            pixel_values_batch = pixel_values_batch.to(
                device=self.vision_tower.device,
                dtype=self.vision_tower.dtype,
            )
            attn_mask_batch = attn_mask_batch.to(device=self.vision_tower.device)
            shapes_batch = shapes_batch.to(device=self.vision_tower.device)

            # Forward through SigLip2 vision tower
            vision_outputs = self.vision_tower(
                pixel_values=pixel_values_batch,
                spatial_shapes=shapes_batch,
                pixel_attention_mask=attn_mask_batch,
                return_dict=True,
            )
            last_hidden_state = vision_outputs.last_hidden_state

            # #region agent log
            _dbg(
                "H1",
                "get_image_feature:post_vit",
                "Vision tower output",
                {
                    "lhs_shape": list(last_hidden_state.shape),
                    "lhs_dtype": str(last_hidden_state.dtype),
                    "lhs_mean": float(last_hidden_state.float().mean()),
                    "lhs_std": float(last_hidden_state.float().std()),
                    "lhs_has_nan": bool(last_hidden_state.isnan().any()),
                    "lhs_has_inf": bool(last_hidden_state.isinf().any()),
                    "lhs_abs_max": float(last_hidden_state.float().abs().max()),
                },
            )
            # #endregion

            # Unpad and project each image
            img_feature_lengths = attn_mask_batch.sum(dim=1)
            batch_size = last_hidden_state.size(0)

            for img_idx in range(batch_size):
                feature = last_hidden_state[img_idx]
                # Unpad: keep only non-padded tokens
                feat_len = img_feature_lengths[img_idx].item()
                feature = feature[:feat_len, :].unsqueeze(0)

                # Reshape to spatial dimensions (1, H, W, C)
                h, w = shapes_batch[img_idx].tolist()
                feature = feature.reshape(1, int(h), int(w), -1)

                # Project through multimodal projector
                img_embedding = self.multi_modal_projector(feature)

                # Flatten to (num_tokens, hidden_size)
                img_embedding = img_embedding.reshape(-1, img_embedding.size(-1))
                image_features_list.append(img_embedding)

                # #region agent log
                if img_idx == 0:
                    _dbg(
                        "H5",
                        "get_image_feature:per_image",
                        "First image embedding",
                        {
                            "feat_len": feat_len,
                            "h": int(h),
                            "w": int(w),
                            "reshape_to": [1, int(h), int(w), -1],
                            "proj_output_shape": list(img_embedding.shape),
                            "proj_mean": float(img_embedding.float().mean()),
                            "proj_std": float(img_embedding.float().std()),
                            "proj_has_nan": bool(img_embedding.isnan().any()),
                        },
                    )
                # #endregion

        # #region agent log
        if image_features_list:
            final = torch.cat(image_features_list, dim=0)
            _dbg(
                "H5",
                "get_image_feature:exit",
                "Final concatenated features",
                {
                    "final_shape": list(final.shape),
                    "final_dtype": str(final.dtype),
                    "final_mean": float(final.float().mean()),
                    "final_std": float(final.float().std()),
                },
            )
            return final
        # #endregion
        return torch.tensor(
            [], device=self.vision_tower.device, dtype=self.vision_tower.dtype
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs,
    ):
        # #region agent log
        _dbg(
            "H2",
            "forward:entry",
            "Forward called",
            {
                "input_ids_shape": list(input_ids.shape),
                "forward_mode": str(forward_batch.forward_mode),
                "has_mm_inputs": forward_batch.mm_inputs is not None,
                "mm_inputs_count": (
                    len([m for m in forward_batch.mm_inputs if m is not None])
                    if forward_batch.mm_inputs
                    else 0
                ),
            },
        )
        # #endregion
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # Collect weights by destination
        vision_weights = []
        projector_weights = []
        lm_weights = []

        for name, loaded_weight in weights:
            if name.startswith("model.vision_tower."):
                # model.vision_tower.* → vision_tower.*
                new_name = name.replace("model.vision_tower.", "vision_tower.", 1)
                vision_weights.append((new_name, loaded_weight))
            elif name.startswith("model.multi_modal_projector."):
                # model.multi_modal_projector.* → multi_modal_projector.*
                new_name = name.replace(
                    "model.multi_modal_projector.", "multi_modal_projector.", 1
                )
                projector_weights.append((new_name, loaded_weight))
            elif name.startswith("model.language_model."):
                # model.language_model.* → language_model.model.*
                new_name = name.replace(
                    "model.language_model.", "language_model.model.", 1
                )
                lm_weights.append((new_name, loaded_weight))
            elif name.startswith("lm_head."):
                # lm_head.* → language_model.lm_head.*
                new_name = name.replace("lm_head.", "language_model.lm_head.", 1)
                lm_weights.append((new_name, loaded_weight))
            else:
                # Try direct mapping
                lm_weights.append((name, loaded_weight))

        params_dict = dict(self.named_parameters())

        # Load vision tower weights
        # #region agent log
        _vt_loaded = 0
        _vt_skipped = []
        # #endregion
        for name, loaded_weight in vision_weights:
            if name not in params_dict:
                # #region agent log
                _vt_skipped.append(name)
                # #endregion
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            # #region agent log
            _vt_loaded += 1
            # #endregion
        # #region agent log
        _dbg(
            "H1",
            "load_weights:vision",
            "Vision tower weight loading complete",
            {
                "loaded": _vt_loaded,
                "skipped": _vt_skipped[:10],
                "total_vision_weights": len(vision_weights),
            },
        )
        # #endregion

        # Load projector weights
        # #region agent log
        _pj_loaded = 0
        _pj_skipped = []
        # #endregion
        for name, loaded_weight in projector_weights:
            if name not in params_dict:
                # #region agent log
                _pj_skipped.append(name)
                # #endregion
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            # #region agent log
            _pj_loaded += 1
            # #endregion
        # #region agent log
        _dbg(
            "H1",
            "load_weights:projector",
            "Projector weight loading complete",
            {
                "loaded": _pj_loaded,
                "skipped": _pj_skipped,
                "total_projector_weights": len(projector_weights),
            },
        )
        # #endregion

        # Load language model weights via Lfm2ForCausalLM.load_weights
        # Strip the "language_model." prefix since Lfm2ForCausalLM expects
        # names like "model.layers.0..." and "lm_head.weight"
        lm_weights_stripped = []
        for name, loaded_weight in lm_weights:
            if name.startswith("language_model."):
                name = name[len("language_model.") :]
            lm_weights_stripped.append((name, loaded_weight))
        self.language_model.load_weights(lm_weights_stripped)


EntryClass = Lfm2VlForConditionalGeneration
