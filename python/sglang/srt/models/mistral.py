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
"""Inference-only Mistral model."""

import logging
from collections.abc import Iterable
from typing import List

import regex as re
import torch
from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

from sglang.srt.managers.schedule_batch import MultimodalDataItem
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)


class MistralForCausalLM(LlamaForCausalLM):
    pass


class MistralForCausalLMMistralFormat(MistralForCausalLM):
    """Mistral GQA model loaded from mistral native format (params.json).

    Handles weight name remapping from mistral native format to HF/Llama
    format. This is the GQA counterpart to MistralLarge3ForCausalLM which
    handles MLA models in mistral native format.
    """

    # fmt: off
    remapping = {
        r"layers\.(\d+)\.attention_norm\.weight": r"model.layers.\1.input_layernorm.weight",
        r"layers\.(\d+)\.attention\.wq\.(\w+)": r"model.layers.\1.self_attn.q_proj.\2",
        r"layers\.(\d+)\.attention\.wk\.(\w+)": r"model.layers.\1.self_attn.k_proj.\2",
        r"layers\.(\d+)\.attention\.wv\.(\w+)": r"model.layers.\1.self_attn.v_proj.\2",
        r"layers\.(\d+)\.attention\.wo\.(\w+)": r"model.layers.\1.self_attn.o_proj.\2",
        r"layers\.(\d+)\.ffn_norm\.weight": r"model.layers.\1.post_attention_layernorm.weight",
        r"layers\.(\d+)\.feed_forward\.w1\.(\w+)": r"model.layers.\1.mlp.gate_proj.\2",
        r"layers\.(\d+)\.feed_forward\.w2\.(\w+)": r"model.layers.\1.mlp.down_proj.\2",
        r"layers\.(\d+)\.feed_forward\.w3\.(\w+)": r"model.layers.\1.mlp.up_proj.\2",
        r"norm\.weight": "model.norm.weight",
        r"tok_embeddings\.weight": "model.embed_tokens.weight",
        r"output\.weight": "lm_head.weight",
    }
    # fmt: on

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return super().load_weights(self._remap_mistral_to_llama(weights))

    def _remap_mistral_to_llama(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        """Remap Mistral native format weight names to HF/Llama format."""
        for name, loaded_weight in weights:
            # Pass through weights already in HF/Llama layout so this loader
            # tolerates mixed-format checkpoints (e.g. native body + HF-style
            # multi_modal_projector weights spliced in by a parent class).
            if name.startswith("model.") or name.startswith("lm_head."):
                yield name, loaded_weight
                continue

            for k, v in self.remapping.items():
                match = re.fullmatch(k, name)
                if match:
                    name = match.expand(v)
                    break
            else:
                logger.warning(f"Unrecognized weight: {name}. Skipping.")
                continue

            if name.endswith(".qscale_act"):
                name = re.sub(r"\.qscale_act$", ".input_scale", name)
            elif name.endswith(".qscale_weight"):
                name = re.sub(r"\.qscale_weight$", ".weight_scale", name)

            yield name, loaded_weight


class Mistral3ForConditionalGeneration:
    MULTIMODAL_PROJECTOR_TYPE = Mistral3MultiModalProjector

    def __init__(self, **kwargs):
        # lazy load inner class
        # to bypass circular import
        from sglang.srt.models.llava import LlavaForConditionalGeneration

        # override config: mistral's projector adds patchmerger that doesn't require padding
        kwargs["config"].vision_config.pad_image_border = False

        self.inner = LlavaForConditionalGeneration(**kwargs)
        self.inner.multi_modal_projector = self.MULTIMODAL_PROJECTOR_TYPE(
            kwargs["config"]
        )
        self.inner.get_image_feature = self.get_image_feature

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """Extract features from image inputs.

        Args:
            items: List of MultimodalDataItem objects containing image data
                Note that an item can be either "image" or "multi-images"

        Returns:
            torch.Tensor: features from image inputs, concatenated
        """
        features = []
        for item in items:
            # in each item, we assume pixel_values is always batched
            pixel_values, image_sizes = item.feature, item.image_sizes
            image_outputs = self.vision_tower(
                pixel_values, image_sizes, output_hidden_states=True
            )
            selected_image_feature = image_outputs.hidden_states[
                self.vision_feature_layer
            ]

            if self.vision_feature_select_strategy in ["default", "patch"]:
                selected_image_feature = selected_image_feature[:, 1:]
            elif self.vision_feature_select_strategy == "full":
                selected_image_feature = selected_image_feature
            else:
                raise ValueError(
                    f"Unexpected select feature: {self.vision_feature_select_strategy}"
                )
            features.append(
                self.multi_modal_projector(
                    selected_image_feature.squeeze(0), image_sizes
                )
            )
        ret = torch.cat(features, dim=0)
        return ret

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __hasattr__(self, name):
        return hasattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        return self.inner(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Normalize transformers v5 Mistral3 weight names for
        LlavaForConditionalGeneration.load_weights.

        v5 checkpoints lay out Mistral3 weights as:
          model.language_model.{embed_tokens,layers.*,norm}.*
          model.vision_tower.*
          model.multi_modal_projector.*
          lm_head.*

        The Llava loader routes by top-level `language_model.` /
        `vision_tower.` prefixes, stripping one segment before forwarding to
        the sub-module.  The sub-module's own `load_weights` expects the
        standard HF layout: `model.layers.*`, `model.embed_tokens.weight`,
        `lm_head.weight` for Llama, and `vision_tower` internals at their
        top level.  So we rewrite:
          model.language_model.X   -> language_model.model.X
          model.vision_tower.X     -> vision_tower.X
          model.multi_modal_projector.X -> multi_modal_projector.X
          lm_head.X                -> language_model.lm_head.X
        """

        def normalize(ws):
            for name, w in ws:
                if name.startswith("model.language_model."):
                    rest = name[len("model.language_model.") :]
                    name = "language_model.model." + rest
                elif name.startswith("model.vision_tower."):
                    name = "vision_tower." + name[len("model.vision_tower.") :]
                elif name.startswith("model.multi_modal_projector."):
                    name = (
                        "multi_modal_projector."
                        + name[len("model.multi_modal_projector.") :]
                    )
                elif name.startswith("lm_head."):
                    name = "language_model." + name
                yield name, w

        return self.inner.load_weights(normalize(weights))


EntryClass = [MistralForCausalLM, Mistral3ForConditionalGeneration]
