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
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/nano_nemotron_vl.py

import logging
from typing import Iterable

import torch
import torch.nn as nn

from sglang.srt.configs.nano_nemotron_vl import NemotronH_Nano_VL_V2_Config
from sglang.srt.layers.activation import ReLU2
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.models.radio import RadioModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class NemotronH_Nano_VL_V2(nn.Module):
    def __init__(
        self,
        config: NemotronH_Nano_VL_V2_Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.downsample_ratio = config.downsample_ratio
        self.language_model = NemotronHForCausalLM(
            config=config.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.vision_model = RadioModel(config=config.create_radio_config()).to(
            self.language_model.config.dtype
        )

        vit_hidden_size = config.vit_hidden_size
        self.rmsnorm_hidden_size = vit_hidden_size * int(1 / self.downsample_ratio) ** 2
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            RMSNorm(
                hidden_size=self.rmsnorm_hidden_size,
                eps=1e-5,
            ),
            nn.Linear(
                self.rmsnorm_hidden_size,
                vision_projection_hidden_size,
                bias=False,
            ),
            ReLU2(),
            nn.Linear(vision_projection_hidden_size, llm_hidden_size, bias=False),
        ).to(self.language_model.config.torch_dtype)
        self.config = config

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        # Get all special token IDs
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        helper = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)

        return helper.pad_input_tokens(input_ids, mm_inputs)

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: float = 0.5) -> torch.Tensor:
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(
            n,
            w,
            int(h * scale_factor),
            int(c / scale_factor),
        )
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale -->
        # N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.config.ps_version != "v1":
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def extract_feature(self, pixel_values):
        # Process images in a micro-batch of at most 128 frames per call
        # This is done on purpose to ensure peak GPU ram usage of huge batch
        # (namely for really long videos with EVS ON) won't cause any problems
        # as we don't support chunked prefill for video media
        micro_batch_size = 128
        n = pixel_values.shape[0]
        vit_embeds_list = []
        for i in range(0, n, micro_batch_size):
            vit_embeds = self.vision_model(pixel_values[i : i + micro_batch_size])
            vit_embeds = vit_embeds.to(dtype=torch.bfloat16)
            h = w = int(vit_embeds.shape[1] ** 0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio
            )
            vit_embeds = vit_embeds.view(-1, self.rmsnorm_hidden_size)
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds = vit_embeds.view(n, -1, self.rmsnorm_hidden_size)
            vit_embeds_list.append(vit_embeds)
        vit_embeds = torch.cat(vit_embeds_list, dim=0)
        return vit_embeds

    def get_image_feature(self, items: list[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        image_features = self.extract_feature(pixel_values)
        return image_features

    def get_video_feature(self, items: list[MultimodalDataItem]):
        """
        Projects the last hidden state from the video model into language model space.

        Returns:
            video_features (`torch.Tensor`): Video feature tensor of shape `(num_videos, video_length, embed_dim)`).
        """
        pixel_values = torch.cat([item.feature for item in items])
        video_features = self.extract_feature(pixel_values)
        return video_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs={
                Modality.IMAGE: self.get_image_feature,
                Modality.VIDEO: self.get_video_feature,
            },
            positions=positions,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        adapter_dict = dict(self.mlp1.named_parameters())

        def is_llm(name: str) -> bool:
            return name.startswith("language_model")

        def is_adapter_weights(weight: tuple[str, torch.Tensor]):
            return weight[0].startswith("mlp1")

        def is_vision_weights(name: str) -> bool:
            return name.startswith("vision_model.radio_model.")

        # Separate weights by component
        llm_weights = []
        vision_weights = []

        for name, w in weights:
            if is_llm(name):
                # Strip 'language_model.' prefix for LLM weights
                llm_weights.append((".".join(name.split(".")[1:]), w))
            elif is_adapter_weights((name, w)):
                # Load vision-language adapter weights directly
                trimmed_name = ".".join(name.split(".")[1:])
                param = adapter_dict[trimmed_name]
                with torch.no_grad():
                    default_weight_loader(param, w)
            elif is_vision_weights(name):
                # Convert: vision_model.radio_model.* → radio_model.*
                hf_key = name[len("vision_model.") :]  # Remove "vision_model." prefix
                vision_weights.append((hf_key, w))
        self.language_model.load_weights(llm_weights)
        self.vision_model.load_weights(vision_weights)


EntryClass = [NemotronH_Nano_VL_V2]
