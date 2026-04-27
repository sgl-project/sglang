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
from sglang.srt.models.parakeet import ProjectedParakeet
from sglang.srt.models.radio import RadioModel
from sglang.srt.multimodal.evs import EVS, EVSConfig
from sglang.srt.multimodal.evs.evs_module import VideoEVSDataItem
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class NemotronH_Nano_VL_V2(EVS):
    @staticmethod
    def create_evs_config(config: NemotronH_Nano_VL_V2_Config):
        return EVSConfig(video_pruning_rate=config.video_pruning_rate)

    def __init__(
        self,
        config: NemotronH_Nano_VL_V2_Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)

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
        self.rmsnorm_hidden_size = (
            vit_hidden_size * int(round(1 / self.downsample_ratio)) ** 2
        )
        vision_projection_hidden_size = config.projector_hidden_size
        llm_hidden_size = config.llm_config.hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.model_dtype = self.language_model.config.torch_dtype

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
        ).to(self.model_dtype)

        self.sound_encoder: ProjectedParakeet | None = None
        if getattr(config, "sound_config", None) is not None:
            logger.info(
                "Found sound config, initializing sound encoder for Nemotron AVLM"
            )
            self.sound_encoder = ProjectedParakeet(
                config.sound_config,
                dtype=self.language_model.config.torch_dtype,
                llm_hidden_size=llm_hidden_size,
                max_model_len=getattr(config, "max_model_len", 8192),
            )

        self.config = config

    def pad_input_ids(self, input_ids: list[int], mm_inputs: MultimodalInputs):
        im_start_id: int = mm_inputs.im_start_id
        im_end_id: int = mm_inputs.im_end_id

        visual_items = [item for item in mm_inputs.mm_items if not item.is_audio()]
        audio_items = [item for item in mm_inputs.mm_items if item.is_audio()]

        all_data_offsets = []

        if visual_items:
            mm_inputs.mm_items = visual_items
            helper = MultiModalityDataPaddingPatternTokenPairs(
                [(im_start_id, im_end_id)]
            )
            input_ids = helper.pad_input_tokens(input_ids, mm_inputs)
            all_data_offsets.extend(mm_inputs.data_offsets)

        audio_start_id = getattr(mm_inputs, "audio_start_id", None)
        audio_end_id = getattr(mm_inputs, "audio_end_id", None)
        if audio_items and audio_start_id is not None and audio_end_id is not None:
            mm_inputs.mm_items = audio_items
            helper = MultiModalityDataPaddingPatternTokenPairs(
                [(audio_start_id, audio_end_id)]
            )
            input_ids = helper.pad_input_tokens(input_ids, mm_inputs)
            all_data_offsets.extend(mm_inputs.data_offsets)

        mm_inputs.mm_items = visual_items + audio_items
        mm_inputs.data_offsets = all_data_offsets

        if audio_items:
            for item in visual_items:
                if isinstance(item, VideoEVSDataItem):
                    item.pre_chunked_input_ids = input_ids

        return input_ids

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

    def extract_feature_dynamic(self, pixel_values_list: list[torch.Tensor]):
        """Extract features from variable-size images (dynamic resolution).

        Each image has different spatial dimensions. They are passed as a list
        to RADIO which handles ragged packing with cu_seqlens internally.
        """
        features, num_patches_list = self.vision_model(pixel_values_list)
        patch_size = self.config.patch_size
        results = []
        offset = 0
        for i, num_patches in enumerate(num_patches_list):
            img_feats = features[0, offset : offset + num_patches]
            h_patches = pixel_values_list[i].shape[-2] // patch_size
            w_patches = pixel_values_list[i].shape[-1] // patch_size
            img_feats = img_feats.reshape(1, h_patches, w_patches, -1)
            img_feats = self.pixel_shuffle(img_feats, self.downsample_ratio)
            img_feats = img_feats.view(-1, self.rmsnorm_hidden_size)
            img_feats = self.mlp1(img_feats)
            results.append(img_feats)
            offset += num_patches
        return torch.cat(results, dim=0)

    def extract_video_feature_temporal(self, pixel_values, num_frames):
        """Extract video features with temporal compression (tubelet grouping)."""
        vit_embeds = self.vision_model(pixel_values, num_frames=num_frames)
        num_tubelets = vit_embeds.shape[0]
        patch_size = self.config.patch_size
        h_patches = pixel_values.shape[-2] // patch_size
        w_patches = pixel_values.shape[-1] // patch_size
        vit_embeds = vit_embeds.reshape(num_tubelets, h_patches, w_patches, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, self.downsample_ratio)
        vit_embeds = vit_embeds.view(-1, self.rmsnorm_hidden_size)
        vit_embeds = self.mlp1(vit_embeds)
        vit_embeds = vit_embeds.view(num_tubelets, -1, self.llm_hidden_size)
        return vit_embeds

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def extract_feature(self, pixel_values):
        micro_batch_size = 128
        n = pixel_values.shape[0]
        patch_size = self.config.patch_size
        h_patches = pixel_values.shape[-2] // patch_size
        w_patches = pixel_values.shape[-1] // patch_size
        vit_embeds_list = []
        for i in range(0, n, micro_batch_size):
            chunk = pixel_values[i : i + micro_batch_size]
            batch_size = chunk.shape[0]
            vit_embeds = self.vision_model(chunk)
            vit_embeds = vit_embeds.to(dtype=self.model_dtype)
            vit_embeds = vit_embeds.reshape(batch_size, h_patches, w_patches, -1)
            vit_embeds = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio
            )
            vit_embeds = vit_embeds.view(-1, self.rmsnorm_hidden_size)
            vit_embeds = self.mlp1(vit_embeds)
            vit_embeds = vit_embeds.view(batch_size, -1, self.llm_hidden_size)
            vit_embeds_list.append(vit_embeds)
        vit_embeds = torch.cat(vit_embeds_list, dim=0)
        return vit_embeds

    def get_image_feature(self, items: list[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        is_dynamic = any(getattr(item, "is_dynamic", False) for item in items)
        if is_dynamic:
            pixel_values_list = [item.feature for item in items]
            return self.extract_feature_dynamic(pixel_values_list)

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
        if getattr(self.config, "video_temporal_patch_size", 1) > 1:
            num_frames = pixel_values.shape[0]
            return self.extract_video_feature_temporal(pixel_values, num_frames)
        video_features = self.extract_feature(pixel_values)
        return video_features

    def get_audio_feature(self, items: list[MultimodalDataItem]):
        """
        Encode audio features through the Parakeet sound encoder.

        Each item carries mel spectrogram features, an attention mask, and a
        clip count. Multiple clips per audio item are grouped and concatenated
        (trimmed to valid output lengths) to form a single embedding per item.
        """
        assert self.sound_encoder is not None

        all_features = []
        all_masks = []
        all_num_clips = []
        for item in items:
            all_features.append(item.feature)
            all_masks.append(item.feature_attention_mask)
            all_num_clips.append(item.audio_num_clips)

        input_audio_features = torch.cat(all_features, dim=0)
        feature_attention_mask = torch.cat(all_masks, dim=0)

        target_device = next(self.sound_encoder.parameters()).device
        input_audio_features = input_audio_features.to(
            dtype=self.language_model.config.torch_dtype, device=target_device
        )
        feature_attention_mask = feature_attention_mask.to(device=target_device)

        sound_embeds = self.sound_encoder(input_audio_features, feature_attention_mask)

        valid_input_lens = feature_attention_mask.sum(dim=1)
        valid_output_lens = (
            self.sound_encoder.encoder._get_subsampling_output_length(valid_input_lens)
            .long()
            .tolist()
        )

        grouped_embeds = []
        clip_offset = 0
        for num_clips in all_num_clips:
            embeds = []
            for clip_idx in range(clip_offset, clip_offset + num_clips):
                valid_len = valid_output_lens[clip_idx]
                embeds.append(sound_embeds[clip_idx, :valid_len])
            grouped_embeds.append(torch.cat(embeds, dim=0))
            clip_offset += num_clips

        return torch.cat(grouped_embeds, dim=0)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        get_embedding: bool = False,
    ):
        data_embedding_funcs = {
            Modality.IMAGE: self.get_image_feature,
            Modality.VIDEO: self.get_video_feature,
        }
        if self.sound_encoder is not None:
            data_embedding_funcs[Modality.AUDIO] = self.get_audio_feature

        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            multimodal_model=self,
            data_embedding_funcs=data_embedding_funcs,
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

        def is_sound_weights(name: str) -> bool:
            return name.startswith("sound")

        # Separate weights by component
        llm_weights = []
        vision_weights = []
        sound_weights = []

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
                hf_key = name[len("vision_model.") :]
                vision_weights.append((hf_key, w))
            elif is_sound_weights(name):
                sound_weights.append((name, w))

        self.language_model.load_weights(llm_weights)
        self.vision_model.load_weights(vision_weights)
        if self.sound_encoder is not None and len(sound_weights) > 0:
            self.sound_encoder.load_weights(sound_weights)


EntryClass = [NemotronH_Nano_VL_V2]
