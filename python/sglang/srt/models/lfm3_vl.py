# Copyright 2026 Liquid AI. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LFM3-VL with shared image/video features and vision weight loading for Omni."""

from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from sglang.srt.configs.lfm3_vl import Lfm3VlConfig
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.lfm3 import Lfm3ForCausalLM
from sglang.srt.models.lfm3_vl_vision import Lfm3VlProjector, Lfm3VlVisionEncoder
from sglang.srt.utils import add_prefix


class Lfm3VisionFeaturesMixin:
    """Shared vision features and weight loading for LFM3-VL and Omni."""

    def _visual_features(
        self,
        pixel_values: list[torch.Tensor],
        grids: list[torch.Tensor],
    ) -> torch.Tensor:
        """Encode + project one modality's items into a flat token tensor.

        Returns `(total_visual_tokens, text_hidden_size)`, ordered by item, then
        temporal patch, then raster position -- matching the order in which the
        processor expanded the placeholder tokens.
        """
        flat_pixels = torch.cat(pixel_values, dim=0).to(
            device=self.vision_encoder.device, dtype=self.vision_encoder.dtype
        )
        # grid_thw stays on CPU: the tower iterates over it in Python.
        grid_thw = torch.cat(grids, dim=0).cpu()

        encoded = self.vision_encoder(flat_pixels, grid_thw)
        projected = [
            self.vision_projector(hidden, height=height, width=width)
            for hidden, height, width in encoded
        ]
        return torch.cat(projected, dim=0)

    @staticmethod
    def _as_tensor(value) -> torch.Tensor:
        return torch.from_numpy(value) if isinstance(value, np.ndarray) else value

    def get_image_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._visual_features(
            [self._as_tensor(item.feature) for item in items],
            # grid_thw is (N, 3); a single item may carry several images/videos.
            [self._as_tensor(item.image_grid_thw).reshape(-1, 3) for item in items],
        )

    def get_video_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        return self._visual_features(
            [self._as_tensor(item.feature) for item in items],
            [self._as_tensor(item.video_grid_thw).reshape(-1, 3) for item in items],
        )

    def _load_vision_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        vision_weights = []
        projector_weights = []
        remaining = []
        for name, weight in weights:
            if name.startswith("model.vision_tower."):
                vision_weights.append(
                    (name.removeprefix("model.vision_tower."), weight)
                )
            elif name.startswith("model.multi_modal_projector."):
                projector_weights.append(
                    (name.removeprefix("model.multi_modal_projector."), weight)
                )
            else:
                remaining.append((name, weight))

        vision_loaded = self.vision_encoder.load_weights(vision_weights)
        missing = dict(self.vision_encoder.named_parameters()).keys() - vision_loaded
        if missing:
            raise ValueError(f"Missing LFM3 vision weights: {sorted(missing)}")
        params = dict(self.vision_projector.named_parameters())
        projector_loaded = set()
        for name, weight in projector_weights:
            parameter = params[name]
            getattr(parameter, "weight_loader", default_weight_loader)(
                parameter, weight
            )
            projector_loaded.add(name)
        missing = params.keys() - projector_loaded
        if missing:
            raise ValueError(f"Missing LFM3 projector weights: {sorted(missing)}")
        loaded = {f"vision_encoder.{name}" for name in vision_loaded} | {
            f"vision_projector.{name}" for name in projector_loaded
        }
        return remaining, loaded


class Lfm3VlForConditionalGeneration(Lfm3VisionFeaturesMixin, nn.Module):
    def __init__(
        self,
        config: Lfm3VlConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.vision_encoder = Lfm3VlVisionEncoder(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_encoder", prefix),
        )
        self.vision_projector = Lfm3VlProjector(
            config,
            quant_config=quant_config,
            prefix=add_prefix("vision_projector", prefix),
        )
        # The LM half is the unmodified LFM3 text model, driven by the
        # nested Lfm3Config provided by HF.
        self.language_model = Lfm3ForCausalLM(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )

    def get_num_kv_cache_layers(self) -> int:
        return self.language_model.get_num_kv_cache_layers()

    def get_attention_sliding_window_size(self) -> int | None:
        return self.language_model.get_attention_sliding_window_size()

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.model.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            positions=positions,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        remaining, vision_loaded = self._load_vision_weights(weights)

        def language_weights():
            for name, weight in remaining:
                if name.startswith("model.language_model."):
                    yield "model." + name.removeprefix("model.language_model."), weight
                elif name.startswith("lm_head."):
                    yield name, weight
                else:
                    raise KeyError(f"unexpected LFM3-VL weight: {name}")

        loaded = self.language_model.load_weights(language_weights())
        return vision_loaded | {f"language_model.{name}" for name in loaded}


EntryClass = Lfm3VlForConditionalGeneration
