# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Inference-only LLaVA-OneVision model compatible with HuggingFace weights.

Wires HF's SiglipVisionModel and LlavaOnevisionMultiModalProjector to
SGLang's native Qwen2ForCausalLM, and routes the forward pass through
``general_mm_embed_routine``. Handles the ``model.*`` checkpoint prefix
that transformers ≥ 4.52 introduced for LLaVA-OneVision.

Coordinate convention: this file uses HuggingFace's (height, width)
ordering everywhere (matches the config, HF helpers, and the checkpoint
math). PIL/SGLang store image sizes as (width, height); we swap at the
boundary.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlavaOnevisionConfig, SiglipVisionModel
from transformers.image_processing_utils import select_best_resolution
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionMultiModalProjector,
    unpad_image,
)

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import general_mm_embed_routine
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.models.utils import AutoWeightsLoader, WeightsMapper
from sglang.srt.utils import add_prefix


class LlavaOnevisionForConditionalGeneration(nn.Module):
    """LLaVA-OneVision: SigLIP vision tower + Qwen2 LM, image and video support."""

    # Two HF-side layout drifts we absorb here:
    #   1. transformers ≥ 4.52 nests LLaVA-OneVision weights under an extra
    #      top-level `model.` prefix (handled by orig_to_new_prefix).
    #   2. transformers ≥ 5.6 flattened SiglipVisionModel — the `.vision_model`
    #      intermediate wrapper is gone, but existing checkpoints still name
    #      keys with it (handled by orig_to_new_substr; runs before prefix
    #      so both drifts compose cleanly).
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            "vision_tower.vision_model.": "vision_tower.",
        },
        orig_to_new_prefix={
            "model.language_model.": "language_model.model.",
            "model.vision_tower.": "vision_tower.",
            "model.multi_modal_projector.": "multi_modal_projector.",
            "model.image_newline": "image_newline",
            "lm_head.": "language_model.lm_head.",
        },
    )

    def __init__(
        self,
        config: LlavaOnevisionConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vision_config = config.vision_config
        self.text_config = config.text_config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        self.language_model = Qwen2ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.image_newline = nn.Parameter(
            torch.empty(config.text_config.hidden_size)
        )

        # Init-static values (see general-code-style.md).
        self._patches_per_side = self.vision_config.image_size // self.vision_config.patch_size
        self._base_num_tokens = self._patches_per_side * self._patches_per_side
        self._max_num_patches = int(
            self.config.vision_aspect_ratio.removeprefix("anyres_max_")
        )
        self._image_token_id = self.config.image_token_index
        self._video_token_id = self.config.video_token_index

    # ------------------------------------------------------------------ #
    # Multimodal glue (dispatched by general_mm_embed_routine)
    # ------------------------------------------------------------------ #

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """Expand each single ``<image>`` / ``<video>`` marker in ``input_ids``
        into the exact number of feature tokens the model will emit, and set
        ``item.offsets = [(start, end)]`` per item so the scheduler can splice
        visual features in at the right positions.

        HuggingFace's chat template inserts one marker per media; the model
        side is responsible for the expansion (matches classic LLaVA behavior).
        """
        stream = list(input_ids)
        for item in mm_inputs.mm_items:
            if item.modality == Modality.IMAGE:
                marker_id = self._image_token_id
                num_tokens = self._num_image_tokens(item)
            elif item.modality == Modality.VIDEO:
                marker_id = self._video_token_id
                num_tokens = self._num_video_tokens(item)
            else:
                continue
            try:
                pos = stream.index(marker_id)
            except ValueError:
                continue
            stream[pos : pos + 1] = [item.pad_value] * num_tokens
            item.offsets = [(pos, pos + num_tokens - 1)]
        return stream

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return torch.cat([self._encode_one_image(item) for item in items], dim=0)

    def get_video_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        return torch.cat([self._encode_one_video(item) for item in items], dim=0)

    # ------------------------------------------------------------------ #
    # Placeholder-count math (mirrors _merge_image_patch_embeddings /
    # _encode_one_video output length; ported from HF's
    # LlavaOnevisionProcessor._get_number_of_features).
    # ------------------------------------------------------------------ #

    def _num_image_tokens(self, item: MultimodalDataItem) -> int:
        # PIL stores (width, height); HF math uses (height, width). Swap here.
        image_sizes = item.model_specific_data.get("image_sizes") or []
        num_patches = item.feature.shape[0]
        if num_patches <= 1 or not image_sizes:
            # Flat (pad) mode: just the base patch + one newline row cell.
            return self._flat_image_token_count()
        orig_w, orig_h = image_sizes[0]
        unpadded, newline = self._unpadded_features(orig_h=orig_h, orig_w=orig_w)
        total = self._base_num_tokens + unpadded + newline
        if self.config.vision_feature_select_strategy == "default":
            total -= 1  # CLS drop
        return total

    def _flat_image_token_count(self) -> int:
        total = self._base_num_tokens + 1  # +1 for image_newline stitch
        if self.config.vision_feature_select_strategy == "default":
            total -= 1
        return total

    def _unpadded_features(self, *, orig_h: int, orig_w: int) -> Tuple[int, int]:
        best_h, best_w = select_best_resolution(
            (orig_h, orig_w), self.config.image_grid_pinpoints
        )
        image_size = self.vision_config.image_size
        scale_h = best_h // image_size
        scale_w = best_w // image_size
        curr_h = self._patches_per_side * scale_h
        curr_w = self._patches_per_side * scale_w
        orig_aspect = orig_w / orig_h
        curr_aspect = curr_w / curr_h
        if orig_aspect > curr_aspect:
            new_h = int(round(orig_h * (curr_w / orig_w), 7))
            padding = (curr_h - new_h) // 2
            curr_h -= padding * 2
        else:
            new_w = int(round(orig_w * (curr_h / orig_h), 7))
            padding = (curr_w - new_w) // 2
            curr_w -= padding * 2
        unpadded = curr_h * curr_w
        newline = curr_h
        ratio = math.sqrt(
            curr_h * curr_w / (self._max_num_patches * self._patches_per_side ** 2)
        )
        if ratio > 1.1:
            unpadded = int(curr_h // ratio) * int(curr_w // ratio)
            newline = int(curr_h // ratio)
        return unpadded, newline

    def _num_video_tokens(self, item: MultimodalDataItem) -> int:
        # Video: pool each frame 2x2, then stack + append one newline vector.
        num_frames = item.feature.shape[0]
        pooled = math.ceil(self._patches_per_side / 2) ** 2
        return num_frames * pooled + 1

    # ------------------------------------------------------------------ #
    # Vision encode helpers
    # ------------------------------------------------------------------ #

    def _as_vision_input(self, feature) -> torch.Tensor:
        """Bring a MultimodalDataItem.feature to the vision tower's device+dtype.

        The LLaVA image processor stores features as ``np.ndarray(float16)``;
        the framework's per-item mover only relocates tensors, so numpy stays
        numpy until it reaches the model. ``torch.as_tensor`` is a zero-copy
        view when possible.
        """
        tensor = feature if isinstance(feature, torch.Tensor) else torch.as_tensor(feature)
        params = self.vision_tower.parameters()
        first = next(params)
        return tensor.to(device=first.device, dtype=first.dtype, non_blocking=True)

    def _run_vision_tower(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        features = outputs.hidden_states[self.config.vision_feature_layer]
        strategy = self.config.vision_feature_select_strategy
        if strategy == "default":
            return features[:, 1:]
        if strategy == "full":
            return features
        raise ValueError(f"Unexpected vision_feature_select_strategy: {strategy}")

    def _encode_one_image(self, item: MultimodalDataItem) -> torch.Tensor:
        pixel_values = self._as_vision_input(item.feature)
        assert pixel_values.dim() == 4, pixel_values.dim()
        patch_features = self._run_vision_tower(pixel_values)
        patch_embeds = self.multi_modal_projector(patch_features)

        image_sizes = item.model_specific_data.get("image_sizes") or []
        if not image_sizes:
            return patch_embeds.flatten(0, 1)
        orig_w, orig_h = image_sizes[0]
        return self._merge_image_patch_embeddings(
            orig_h=orig_h,
            orig_w=orig_w,
            patch_embeddings=patch_embeds,
        )

    def _encode_one_video(self, item: MultimodalDataItem) -> torch.Tensor:
        pixel_values = self._as_vision_input(item.feature)
        assert pixel_values.dim() == 4, pixel_values.dim()  # (num_frames, C, H, W)
        features = self._run_vision_tower(pixel_values)
        features = self.multi_modal_projector(features)
        features = self._pool_video_features(features)
        num_frames = features.shape[0]
        flat = features.reshape(1, num_frames * features.shape[1], -1)
        newline = self.image_newline[None, None, :]
        return torch.cat((flat, newline), dim=1).squeeze(0)

    def _pool_video_features(
        self, features: torch.Tensor, stride: int = 2
    ) -> torch.Tensor:
        s = self._patches_per_side
        f, _, d = features.shape
        features = features.view(f, s, s, d).permute(0, 3, 1, 2)
        scaled = [math.ceil(s / stride), math.ceil(s / stride)]
        features = nn.functional.interpolate(features, size=scaled, mode="bilinear")
        return features.permute(0, 2, 3, 1).contiguous().view(f, -1, d)

    def _merge_image_patch_embeddings(
        self,
        *,
        orig_h: int,
        orig_w: int,
        patch_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Anyres merge with image_newline stitching.

        Uses HF's (height, width) convention throughout — matches
        ``select_best_resolution`` and ``unpad_image`` imported from HF.
        """
        s = self._patches_per_side
        base = patch_embeddings[0]
        if s * s != base.shape[0]:
            raise ValueError("Patch count mismatch vs. vision config.")

        if patch_embeddings.shape[0] == 1:
            return torch.cat(
                (base, self.image_newline[None].to(base.device)), dim=0
            )

        others = patch_embeddings[1:]
        best_h, best_w = select_best_resolution(
            (orig_h, orig_w), self.config.image_grid_pinpoints
        )
        image_size = self.vision_config.image_size
        n_h = best_h // image_size
        n_w = best_w // image_size
        others = others[: n_h * n_w].view(n_h, n_w, s, s, -1)
        others = (
            others.permute(4, 0, 2, 1, 3).contiguous().flatten(1, 2).flatten(2, 3)
        )
        others = unpad_image(others, (orig_h, orig_w))
        _, ch, cw = others.shape
        ratio = math.sqrt(ch * cw / (self._max_num_patches * s ** 2))
        if ratio > 1.1:
            others = nn.functional.interpolate(
                others[None],
                [int(ch // ratio), int(cw // ratio)],
                mode="bilinear",
            )[0]
        _, ch, cw = others.shape
        newline = self.image_newline[:, None, None].expand(-1, ch, 1).to(others.device)
        others = torch.cat((others, newline), dim=-1)
        others = others.flatten(1, 2).transpose(0, 1)
        return torch.cat((base, others), dim=0)

    # ------------------------------------------------------------------ #
    # Top-level forward + weight loading
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds=None,
        get_embedding: bool = False,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def load_weights(
        self, weights: Iterable[Tuple[str, torch.Tensor]]
    ) -> set:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_sglang_mapper)


EntryClass = [LlavaOnevisionForConditionalGeneration]
