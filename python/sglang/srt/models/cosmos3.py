# Copyright 2023-2025 SGLang Team
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
"""Inference-only Cosmos3 Reasoner (understanding tower) model.

Cosmos3 ships a unified diffusers-layout checkpoint that stores a Qwen3-VL
understanding tower alongside a generation (diffusion) tower. The Reasoner
serves only the understanding tower, so it reuses the Qwen3-VL inference stack
and drops the generation-tower weights at load time.

The checkpoint keeps the LLM weights under ``transformer/`` and the vision
encoder weights under ``vision_encoder/``, so the two are loaded from separate
subfolders via ``allow_patterns_overrides`` / ``secondary_weights``.
"""

from typing import Iterable, Optional, Tuple

import torch

from sglang.srt.configs.cosmos3 import Cosmos3Config
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.utils import WeightsMapper
from sglang.srt.server_args import get_global_server_args


class Cosmos3ForConditionalGeneration(Qwen3VLForConditionalGeneration):
    # Cosmos3 unified checkpoints store a Qwen3-VL understanding tower alongside
    # a generation tower in a flat key layout. This mapper drops the generation
    # tower weights and rewrites the understanding tower keys into the nested
    # Qwen3-VL checkpoint form consumed by the parent ``load_weights``.
    hf_to_sglang_mapper = WeightsMapper(
        orig_to_new_substr={
            # Drop the generation (diffusion) tower.
            "_moe_gen": None,
            ".add_q_proj.": None,
            ".add_k_proj.": None,
            ".add_v_proj.": None,
            ".to_add_out.": None,
            ".norm_added_q.": None,
            ".norm_added_k.": None,
            # Understanding-tower attention projections -> Qwen3 names.
            ".to_q.": ".q_proj.",
            ".to_k.": ".k_proj.",
            ".to_v.": ".v_proj.",
            ".to_out.": ".o_proj.",
            ".norm_q.": ".q_norm.",
            ".norm_k.": ".k_norm.",
        },
        orig_to_new_prefix={
            # Understanding-tower (LLM) keys -> nested language-model namespace.
            "layers.": "model.language_model.layers.",
            "embed_tokens.": "model.language_model.embed_tokens.",
            "norm.": "model.language_model.norm.",
            # Vision-encoder keys -> visual namespace.
            "blocks.": "model.visual.blocks.",
            "merger.": "model.visual.merger.",
            "patch_embed.": "model.visual.patch_embed.",
            "pos_embed.": "model.visual.pos_embed.",
            "deepstack_merger_list.": "model.visual.deepstack_merger_list.",
            # Diffusion-only latent/timestep/modality heads -> dropped.
            "proj_in.": None,
            "proj_out.": None,
            "time_embedder.": None,
            "audio_": None,
            "action_": None,
        },
    )

    # The understanding-tower LLM weights live in the ``transformer/`` subfolder
    # of the diffusers-layout checkpoint.
    allow_patterns_overrides = ["transformer/*.safetensors"]

    def __init__(
        self,
        config: Cosmos3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

        # The vision encoder weights live in a separate ``vision_encoder/``
        # subfolder, so load them as a secondary weight source.
        server_args = get_global_server_args()
        self.secondary_weights = [
            DefaultModelLoader.Source(
                model_or_path=server_args.model_path,
                revision=getattr(server_args, "revision", None),
                prefix="",
                allow_patterns_overrides=["vision_encoder/*.safetensors"],
            )
        ]

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        return super().load_weights(self.hf_to_sglang_mapper.apply(weights))


EntryClass = Cosmos3ForConditionalGeneration
