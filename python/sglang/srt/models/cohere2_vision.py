# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
# Adapted from:
#  - vllm/model_executor/models/cohere2_vision.py
#  - sglang/srt/models/jet_vlm.py
"""Inference-only Cohere2Vision (Command-A-Vision) multimodal model."""

import math
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.siglip import SiglipVisionModel

import torch.nn.functional as F

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

from sglang.srt.models.cohere2_moe import Cohere2MoeForCausalLM


class Cohere2VisionMultiModalProjector(nn.Module):
    """Pixel-shuffle downsample -> SwiGLU MLP -> text hidden dim.

    Mirrors transformers.models.cohere2_vision.Cohere2VisionMultiModalProjector.
    """

    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.downsample_factor = config.downsample_factor
        input_dim = config.vision_config.hidden_size * (config.downsample_factor**2)
        # The HF projector stores a single ``linear_1`` whose output is split
        # halves for the SwiGLU gate / value. We mirror that with a merged
        # column-parallel linear of two equal-size shards.
        self.intermediate_size = config.alignment_intermediate_size // 2
        self.linear_1 = MergedColumnParallelLinear(
            input_dim,
            [self.intermediate_size] * 2,
            bias=True,
        )
        self.linear_2 = RowParallelLinear(
            self.intermediate_size,
            config.text_config.hidden_size,
            bias=True,
        )

    def pixel_shuffle(self, image_features: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = image_features.shape
        height = width = int(math.isqrt(seq_len))
        image_features = image_features.reshape(batch_size, width, height, -1)
        channels = image_features.shape[-1]
        image_features = image_features.reshape(
            batch_size,
            width,
            int(height / self.downsample_factor),
            int(channels * self.downsample_factor),
        )
        image_features = image_features.permute(0, 2, 1, 3)
        image_features = image_features.reshape(
            batch_size,
            int(height / self.downsample_factor),
            int(width / self.downsample_factor),
            -1,
        )
        image_features = image_features.permute(0, 2, 1, 3)
        return image_features

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        image_features = self.pixel_shuffle(image_features)
        # Flatten (B, H, W, D) -> (B, H*W, D) for the linear layers.
        b, h, w, d = image_features.shape
        image_features = image_features.reshape(b, h * w, d)
        gate_up, _ = self.linear_1(image_features)
        # HF Cohere2Vision SwiGLU: chunks (x, gate), output = x * silu(gate).
        # SGLang's SiluAndMul swaps the halves, so we do the chunk inline.
        x, gate = gate_up.chunk(2, dim=-1)
        hidden_states = x * F.silu(gate)
        hidden_states, _ = self.linear_2(hidden_states)
        return hidden_states


def _remap_quant_config_for_sglang(quant_config):
    """The HF checkpoint stores quantization metadata using HF module names
    (e.g. ``model.language_model.layers.X.self_attn.q_proj``).  Our SGLang
    module hierarchy uses ``language_model.model.layers.X.self_attn.q_proj``
    for the same parameter.  We rewrite the ``ignore`` list and any
    target-scheme keys so that ``should_ignore_layer`` matches our prefixes."""
    if quant_config is None or not hasattr(quant_config, "ignore"):
        return

    def _rewrite(name: str) -> str:
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model."):]
        if name.startswith("model.vision_tower."):
            return "vision_tower." + name[len("model.vision_tower."):]
        if name.startswith("model.multi_modal_projector."):
            return "multi_modal_projector." + name[len("model.multi_modal_projector."):]
        return name

    quant_config.ignore = [_rewrite(n) for n in quant_config.ignore]
    if hasattr(quant_config, "target_scheme_map") and isinstance(
        quant_config.target_scheme_map, dict
    ):
        quant_config.target_scheme_map = {
            _rewrite(k): v for k, v in quant_config.target_scheme_map.items()
        }


class Cohere2VisionForConditionalGeneration(nn.Module):
    # Used by the model loader to fan out fused Linear modules
    # (e.g. ``qkv_proj`` -> ``[q_proj, k_proj, v_proj]``) when matching
    # against the quantization config's ignore list.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config

        # Rewrite the quant config's ignore list to match the SGLang module
        # prefix layout.  Must happen before any Linear is instantiated.
        _remap_quant_config_for_sglang(quant_config)

        # The SiglipVisionModel is loaded via the standard transformers
        # implementation. It does not need TP since the vision tower is small
        # relative to the LM. Disable LM-style quant for the vision tower.
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = Cohere2VisionMultiModalProjector(config)
        self.language_model = Cohere2MoeForCausalLM(
            config=config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, mm_input: List[MultimodalDataItem]) -> torch.Tensor:
        # Concatenate pixel patches from all images.
        pixel_values = torch.cat(
            [
                torch.as_tensor(item.feature, device=self.vision_tower.device)
                for item in mm_input
            ],
            dim=0,
        )
        pixel_values = pixel_values.to(self.vision_tower.dtype)

        vision_outputs: BaseModelOutputWithPooling = self.vision_tower(
            pixel_values=pixel_values, return_dict=True
        )
        image_features = vision_outputs.last_hidden_state
        image_features = self.multi_modal_projector(image_features)

        # Flatten patches: (np, tokens_per_patch, dim) -> (np*tokens, dim)
        return image_features.reshape(-1, image_features.shape[-1])

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
        **kwargs,
    ) -> LogitsProcessorOutput:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
            },
            positions=positions,
            get_embedding=get_embedding,
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # The HF / quantized checkpoint stores tensors under the
        # `model.language_model.`, `model.vision_tower.`, and
        # `model.multi_modal_projector.` prefixes.  Re-map them to our
        # SGLang module names, then dispatch.
        lm_weights: List[Tuple[str, torch.Tensor]] = []
        vision_weights: List[Tuple[str, torch.Tensor]] = []
        projector_weights: List[Tuple[str, torch.Tensor]] = []

        for name, w in weights:
            if name.startswith("model.language_model."):
                # Drop the leading "model." so the LM sees the "language_model."
                # prefix already stripped; the LM expects "model.layers..." names.
                stripped = name[len("model.language_model."):]
                lm_weights.append((f"model.{stripped}", w))
            elif name.startswith("language_model."):
                stripped = name[len("language_model."):]
                lm_weights.append((f"model.{stripped}", w))
            elif name.startswith("model.vision_tower."):
                vision_weights.append((name[len("model."):], w))
            elif name.startswith("vision_tower."):
                vision_weights.append((name, w))
            elif name.startswith("model.multi_modal_projector."):
                projector_weights.append((name[len("model."):], w))
            elif name.startswith("multi_modal_projector."):
                projector_weights.append((name, w))
            elif name.startswith("lm_head."):
                # Tied with embed_tokens; ignore.
                continue
            else:
                # Unknown top-level keys; pass through to LM as a fallback.
                lm_weights.append((name, w))

        self.language_model.load_weights(lm_weights)

        # Load vision tower weights. transformers >=5 SiglipVisionModel
        # exposes the inner encoder directly, so its parameters are at
        # ``embeddings.*`` / ``encoder.layers.*`` / ``post_layernorm.*`` (no
        # leading ``vision_model.``).  The checkpoint stores them as
        # ``vision_tower.vision_model.<...>``.
        vt_params = dict(self.vision_tower.named_parameters())
        for name, w in vision_weights:
            assert name.startswith("vision_tower.")
            stripped = name[len("vision_tower."):]
            # Some HF versions still keep the ``vision_model.`` middle prefix.
            if stripped not in vt_params and stripped.startswith("vision_model."):
                stripped = stripped[len("vision_model."):]
            if stripped not in vt_params:
                # Helpful diagnostic: show the available keys' style.
                sample = sorted(vt_params.keys())[:3]
                raise ValueError(
                    f"Unexpected vision tower weight: {name} (looked for "
                    f"{stripped!r}, sample params: {sample})"
                )
            vt_params[stripped].data.copy_(w)

        # Load projector. Names look like multi_modal_projector.linear_1.weight
        # — we use the merged-column linear, which has its own weight_loader.
        # The HF checkpoint stores the merged ``linear_1`` weight as one tensor
        # of shape [2*N, in], matching MergedColumnParallelLinear's combined
        # storage. So a normal copy_ via default_weight_loader works.
        proj_params = dict(self.multi_modal_projector.named_parameters())
        for name, w in projector_weights:
            assert name.startswith("multi_modal_projector.")
            stripped = name[len("multi_modal_projector."):]
            if stripped not in proj_params:
                raise ValueError(f"Unexpected projector weight: {name}")
            param = proj_params[stripped]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w)


EntryClass = Cohere2VisionForConditionalGeneration
