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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3n_mm.py

import logging
import re
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypedDict, Union

import torch
from torch import nn
from transformers import (
    Gemma3nAudioConfig,
    Gemma3nConfig,
    Gemma3nTextConfig,
    Gemma3nVisionConfig,
    PreTrainedModel,
)
from transformers.models.auto.modeling_auto import AutoModel

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternTokenPairs,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    flatten_nested_list,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.models.gemma3n_audio import Gemma3nAudioEncoder
from sglang.srt.models.gemma3n_causal import Gemma3nRMSNorm, Gemma3nTextModel
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Gemma3nImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class Gemma3nAudioInputs(TypedDict):
    input_features: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length, num_features)`"""
    input_features_mask: torch.Tensor
    """Shape: `(batch_size * num_audio, seq_length)`"""


class Gemma3nMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Union[Gemma3nAudioConfig, Gemma3nVisionConfig],
        text_config: Gemma3nTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()

        self.multimodal_hidden_size = multimodal_config.hidden_size
        self.eps = multimodal_config.rms_norm_eps
        self.vocab_offset = multimodal_config.vocab_offset
        self.vocab_size = multimodal_config.vocab_size
        self.text_hidden_size = text_config.hidden_size

        self.embedding = VocabParallelEmbedding(
            self.vocab_size,
            self.multimodal_hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("embedding", prefix),
        )

        self.hard_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.soft_embedding_norm = Gemma3nRMSNorm(
            self.multimodal_hidden_size,
            eps=self.eps,
        )

        self.embedding_projection = RowParallelLinear(
            self.multimodal_hidden_size,
            self.text_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("embedding_projection", prefix),
        )

        self.embedding_post_projection_norm = Gemma3nRMSNorm(
            self.text_hidden_size,
            eps=self.eps,
            with_scale=False,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embeds token ids or soft tokens for multimodal content into language model space.

        Args:
            input_ids: A torch.LongTensor containing the token ids to embed. Values should be in the range
                `[vocab_offset, vocab_offset + vocab_size)`.
            inputs_embeds: A torch.Tensor containing the soft tokens to embed.

        Returns:
            A torch.Tensor of embeddings with  shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is not None:
            emb_norm = self.soft_embedding_norm(inputs_embeds)
        else:
            # Handle out of vocab ids to prevent CUDA assertion failures
            out_of_vocab_id = self.vocab_size - 1
            adjusted_ids = input_ids - self.vocab_offset
            adjusted_ids = torch.where(adjusted_ids < 0, out_of_vocab_id, adjusted_ids)
            adjusted_ids = torch.where(
                adjusted_ids >= self.vocab_size, out_of_vocab_id, adjusted_ids
            )
            hard_emb = self.embedding(adjusted_ids)
            emb_norm = self.hard_embedding_norm(hard_emb)

        emb_norm_proj, _ = self.embedding_projection(emb_norm)
        return self.embedding_post_projection_norm(emb_norm_proj)


class Gemma3nForConditionalGeneration(PreTrainedModel):
    config_class = Gemma3nConfig
    """Gemma3n multimodal model for conditional generation."""

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
        ".out_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
        "out_proj": ("proj", 0),
    }

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]
    # Gemma does not apply LoRA to the embedding layer
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma3nConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config

        prefix = add_prefix("model", prefix)

        # Vision components
        # TODO: Use sglang's vision model
        self.vision_tower = AutoModel.from_config(config=config.vision_config)

        self.embed_vision = Gemma3nMultimodalEmbedder(
            config.vision_config,
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("embed_vision", prefix),
        )

        # Audio components
        self.embed_audio = Gemma3nMultimodalEmbedder(
            config.audio_config,
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("embed_audio", prefix),
        )

        self.audio_tower = Gemma3nAudioEncoder(
            config.audio_config,
            quant_config=quant_config,
            prefix=add_prefix("audio_tower", prefix),
        )

        self.vocab_size = config.text_config.vocab_size
        self.vocab_size_per_layer_input = config.text_config.vocab_size_per_layer_input

        # Text model
        self.language_model = Gemma3nTextModel(
            config.text_config,
            quant_config,
            prefix=add_prefix("language_model", prefix),
        )

        # Create logits processor for the multimodal model
        self.logits_processor = LogitsProcessor(config.text_config)

        # if self.language_model.logits_processor.logit_scale:
        #     logit_scale = getattr(config, "logit_scale", 1.0)
        #     self.language_model.logits_processor.logit_scale *= logit_scale

        self.post_init()

    def pad_input_ids(
        self,
        input_ids: List[int],
        image_inputs: MultimodalInputs,
        audio_inputs: Optional[MultimodalInputs] = None,
    ) -> List[int]:
        """Pad input IDs with image and audio tokens."""
        # Get special token IDs
        media_token_pairs = []

        if image_inputs is not None and hasattr(image_inputs, "im_start_id"):
            im_start_id: int = image_inputs.im_start_id
            im_end_id: int = image_inputs.im_end_id
            media_token_pairs.append((im_start_id, im_end_id))

        if audio_inputs is not None and hasattr(audio_inputs, "audio_start_id"):
            audio_start_id: int = audio_inputs.audio_start_id
            audio_end_id: int = audio_inputs.audio_end_id
            media_token_pairs.append((audio_start_id, audio_end_id))

        if media_token_pairs:
            pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)
            # Combine both image and audio inputs for padding
            all_inputs = image_inputs
            if audio_inputs is not None:
                # Merge audio inputs into all_inputs if needed
                pass
            ids = pattern.pad_input_tokens(input_ids, all_inputs)
            return ids

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        return self.config.text_config.sliding_window - 1

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        # Process images one by one to handle flatten_batch=True constraint in vision_tower
        all_pixel_values = flatten_nested_list([item.pixel_values for item in items])
        vision_outputs_list = []

        for pixel_values_batch in all_pixel_values:
            # Normalize input shape to [batch_size, channels, height, width]
            if pixel_values_batch.dim() == 5:
                pixel_values_batch = pixel_values_batch.squeeze(0)
            elif pixel_values_batch.dim() == 3:
                pixel_values_batch = pixel_values_batch.unsqueeze(0)
            elif pixel_values_batch.dim() != 4:
                raise ValueError(
                    f"Unexpected pixel_values shape: {pixel_values_batch.shape}"
                )

            # Process each image in the batch
            batch_size = pixel_values_batch.shape[0]
            for i in range(batch_size):
                pixel_value = pixel_values_batch[i : i + 1]  # Keep batch dimension as 1
                pixel_value = pixel_value.to(
                    device=self.vision_tower.device, dtype=self.language_model.dtype()
                )
                vision_outputs = self.vision_tower(
                    pixel_values=pixel_value, do_pooling=False, return_dict=True
                ).last_hidden_state
                vision_outputs_list.append(vision_outputs)

        # Concatenate all vision outputs
        vision_outputs = torch.cat(vision_outputs_list, dim=0)

        # Convert from (batch, channels, height, width) to (batch, height * width, channels)
        vision_outputs = vision_outputs.reshape(
            vision_outputs.shape[0],
            self.config.vision_config.hidden_size,
            self.config.vision_soft_tokens_per_image,
        ).permute(0, 2, 1)

        # Normalize and embed the soft tokens into language model space
        vision_outputs *= self.config.vision_config.hidden_size**0.5
        return self.embed_vision(inputs_embeds=vision_outputs)

    def get_audio_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        """
        Projects the last hidden state from the audio encoder into language model space.

        Args:
            items: List of multimodal data items containing audio data.

        Returns:
            audio_features (`torch.Tensor`): Audio feature tensor of shape `(num_audios, audio_length, embed_dim)`).
        """
        # Extract audio features and masks from items
        all_input_features = flatten_nested_list(
            [item.input_features for item in items]
        )
        all_input_features_mask = flatten_nested_list(
            [item.input_features_mask for item in items]
        )

        # Process audio features one by one
        audio_features_list = []

        for input_features, input_features_mask in zip(
            all_input_features, all_input_features_mask
        ):
            # Ensure proper tensor format
            if input_features.dim() == 2:
                input_features = input_features.unsqueeze(0)
            if input_features_mask.dim() == 1:
                input_features_mask = input_features_mask.unsqueeze(0)

            # Move to device and dtype
            input_features = input_features.to(
                device=next(self.audio_tower.parameters()).device,
                dtype=self.language_model.dtype(),
            )
            input_features_mask = input_features_mask.to(device=input_features.device)

            # Process through audio tower
            audio_outputs, audio_mask = self.audio_tower(
                input_features, input_features_mask
            )

            # Embed the audio outputs
            audio_embeds = self.embed_audio(inputs_embeds=audio_outputs)
            audio_features_list.append(audio_embeds)

        # Concatenate all audio features
        if audio_features_list:
            return torch.cat(audio_features_list, dim=0)
        else:
            # Return empty tensor with correct dimensions if no audio
            return torch.empty(
                0,
                0,
                self.language_model.config.hidden_size,
                device=next(self.parameters()).device,
                dtype=self.language_model.dtype(),
            )

    def get_per_layer_inputs(
        self, input_ids: torch.LongTensor
    ) -> Optional[torch.Tensor]:
        return self.language_model.get_per_layer_inputs(input_ids)

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.language_model.project_per_layer_inputs(
            inputs_embeds, per_layer_inputs
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs: object,
    ) -> LogitsProcessor:
        """Forward pass for multimodal Gemma3n."""
        if (input_ids is None) ^ (input_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        # if input_ids is not None:
        #     # replace the ids that are not in the vocab with the image token id
        #     input_ids = torch.where(input_ids >= self.vocab_size, self.config.image_token_id, input_ids)
        #     # print(f"DEBUG: input_ids: {input_ids}")
        #     input_embeds = self.get_input_embeddings()(input_ids)

        #     # Prepare per-layer inputs from inputs_ids
        #     per_layer_inputs_mask = torch.logical_and(input_ids >= 0, input_ids < self.vocab_size_per_layer_input)
        #     per_layer_inputs_tokens = torch.where(per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids))
        #     per_layer_inputs = self.language_model.get_per_layer_inputs(per_layer_inputs_tokens)
        #     # Ensure no gaps between text, vision, and audio embeddings, in that order
        #     assert self.embed_audio.vocab_offset == self.vocab_size - self.embed_audio.vocab_size
        #     assert self.embed_vision.vocab_offset == (
        #         self.vocab_size - self.embed_audio.vocab_size - self.embed_vision.vocab_size
        #     )

        #     # Handle vision tokens (>= embed_vision.vocab_offset and < embed_audio.vocab_offset)
        #     vision_mask = torch.logical_and(
        #         input_ids >= self.embed_vision.vocab_offset,
        #         input_ids < self.embed_audio.vocab_offset
        #     )
        #     vision_indices = torch.where(vision_mask)
        #     vision_tokens = input_ids[vision_indices]
        #     vision_embeds_flat = self.embed_vision(input_ids=vision_tokens)
        #     input_embeds[vision_indices] = vision_embeds_flat

        #     # Handle audio tokens (>= embed_audio.vocab_offset)
        #     audio_mask = input_ids >= self.embed_audio.vocab_offset
        #     audio_indices = torch.where(audio_mask)
        #     audio_tokens = input_ids[audio_indices]
        #     audio_embeds_flat = self.embed_audio(input_ids=audio_tokens)
        #     input_embeds[audio_indices] = audio_embeds_flat

        if input_ids is not None:
            # Prepare per-layer inputs from inputs_ids
            per_layer_inputs_mask = torch.logical_and(
                input_ids >= 0, input_ids < self.vocab_size_per_layer_input
            )
            per_layer_inputs_tokens = torch.where(
                per_layer_inputs_mask, input_ids, torch.zeros_like(input_ids)
            )
            per_layer_inputs = self.language_model.get_per_layer_inputs(
                per_layer_inputs_tokens
            )

        print(f"{forward_batch.mm_inputs=}, {forward_batch.contains_mm_inputs()=}")
        # Use general_mm_embed_routine for handling multimodal data
        # This will automatically handle text, image, and audio embeddings
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            image_data_embedding_func=self.get_image_feature,
            audio_data_embedding_func=self.get_audio_feature,
            positions=positions,
            per_layer_inputs=per_layer_inputs,
        )

        # Process hidden states through logits processor
        return self.logits_processor(
            input_ids, hidden_states, self.language_model.embed_tokens, forward_batch
        )

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".up_proj", 1),
            (".gate_up_proj", ".gate_proj", 0),
        ]
        """Load weights for the model."""
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            name = re.sub(r"^model\.", "", name)
            # if "language_model" in name:
            #     causal_loaded_params = Gemma3nForCausalLM.load_weights(
            #         self, [(name, loaded_weight)]
            #     )
            #     loaded_params.update(causal_loaded_params)
            #     continue
            # else:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "vision_model" in name:
                    # adapt to VisionAttention
                    name = name.replace(".self_attn.out_proj", ".self_attn.proj")
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


EntryClass = Gemma3nForConditionalGeneration
