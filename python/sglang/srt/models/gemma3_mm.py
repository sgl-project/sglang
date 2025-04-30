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
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/gemma3_mm.py

import logging
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set, Tuple, TypedDict

import torch
from torch import nn
from transformers import AutoModel, Gemma3Config, PreTrainedModel

from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
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
from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)

cached_get_processor = lru_cache(get_processor)


class Gemma3ImagePixelInputs(TypedDict):
    pixel_values: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class Gemma3MultiModalProjector(nn.Module):
    """Projector for Gemma3 multimodal."""

    def __init__(self, config: Gemma3Config):
        super().__init__()

        self.mm_input_projection_weight = nn.Parameter(
            torch.zeros(
                config.vision_config.hidden_size, config.text_config.hidden_size
            )
        )

        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )

        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side
        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.kernel_size, stride=self.kernel_size
        )

    def forward(self, vision_outputs: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length, hidden_size = vision_outputs.shape

        # Reshape for pooling
        reshaped_vision_outputs = vision_outputs.transpose(1, 2)
        reshaped_vision_outputs = reshaped_vision_outputs.reshape(
            batch_size, hidden_size, self.patches_per_image, self.patches_per_image
        )
        reshaped_vision_outputs = reshaped_vision_outputs.contiguous()

        # Apply pooling
        pooled_vision_outputs = self.avg_pool(reshaped_vision_outputs)
        pooled_vision_outputs = pooled_vision_outputs.flatten(2)
        pooled_vision_outputs = pooled_vision_outputs.transpose(1, 2)

        # Apply normalization
        normed_vision_outputs = self.mm_soft_emb_norm(pooled_vision_outputs)

        # Project to text embedding space
        projected_vision_outputs = torch.matmul(
            normed_vision_outputs, self.mm_input_projection_weight
        )

        return projected_vision_outputs.type_as(vision_outputs)


class Gemma3ForConditionalGeneration(PreTrainedModel):
    config_class = Gemma3Config
    """Gemma3 multimodal model for conditional generation."""

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
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
    # Gemma does not apply LoRA to the embedding layer.
    embedding_modules = {}
    embedding_padding_modules = []
    supports_lora = True

    def __init__(
        self,
        config: Gemma3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config)
        self.config = config
        self.quant_config = quant_config
        # Vision components
        # TODO: replace with vision attention
        # self.vision_tower = SiglipVisionModel(
        #     config.vision_config,
        #     quant_config,
        #     prefix=add_prefix("vision_tower", prefix),
        # )
        self.vision_tower = AutoModel.from_config(config=config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size

        # Text model
        self.language_model = Gemma3ForCausalLM(
            config.text_config, quant_config, prefix=add_prefix("model", prefix)
        )
        if self.language_model.logits_processor.logit_scale:
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.language_model.logits_processor.logit_scale *= logit_scale
        self.post_init()

    def pad_input_ids(
        self, input_ids: List[int], image_inputs: MultimodalInputs
    ) -> List[int]:
        """Pad input IDs with image tokens."""
        # Get special token IDs
        im_start_id: int = image_inputs.im_start_id
        im_end_id: int = image_inputs.im_end_id

        media_token_pairs = [(im_start_id, im_end_id)]
        pattern = MultiModalityDataPaddingPatternTokenPairs(media_token_pairs)
        ids = pattern.pad_input_tokens(input_ids, image_inputs)
        return ids

    def prepare_attn_masks(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        mask_dtype: torch.dtype,
        **kwargs,
    ) -> Dict:
        """Prepare attention masks for multimodal inputs."""
        kwargs["has_images"] = True

        # Distinguish sequences by position id 0
        start_indices = (positions == 0).cpu().nonzero()
        num_seqs = len(start_indices)
        seq_lens = []

        for i in range(num_seqs):
            start_idx = start_indices[i].item()
            if i < num_seqs - 1:
                end_idx = start_indices[i + 1].item()
            else:
                end_idx = len(input_ids)
            seq_lens.append(end_idx - start_idx)

        kwargs["seq_lens"] = seq_lens

        # Create attention masks
        global_attn_masks = []
        local_attn_masks = []
        sliding_window = self.config.text_config.interleaved_sliding_window

        start_idx = 0
        for seq_len in seq_lens:
            end_idx = start_idx + seq_len
            input_token_ids = input_ids[start_idx:end_idx]
            start_idx = end_idx

            # Create global causal mask
            global_attn_mask = torch.empty(
                1,
                1,
                seq_len,
                seq_len,
                dtype=mask_dtype,
                device=input_ids.device,
            )
            global_attn_mask.fill_(float("-inf"))
            global_attn_mask = global_attn_mask.triu(diagonal=1)

            # Consider bidirectional attention between image tokens
            img_mask = torch.zeros_like(global_attn_mask)
            img_pos = input_token_ids == self.config.image_token_index
            img_mask[:, :, :, img_pos] += 1
            img_mask[:, :, img_pos, :] += 1
            global_attn_mask = torch.where(img_mask == 2, 0, global_attn_mask)
            global_attn_masks.append(global_attn_mask)

            # Create local causal mask with sliding window
            local_attn_mask = torch.ones_like(global_attn_mask)
            local_attn_mask = torch.tril(local_attn_mask, diagonal=-sliding_window)
            local_attn_mask = torch.where(
                local_attn_mask == 0, global_attn_mask, float("-inf")
            )
            local_attn_masks.append(local_attn_mask)

        kwargs["global_attn_masks"] = global_attn_masks
        kwargs["local_attn_masks"] = local_attn_masks
        return kwargs

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    def get_attention_sliding_window_size(self):
        """
        This value is used to initialize attention backends in `ForwardBatch`.
        """
        return self.language_model.get_attention_sliding_window_size()

    def get_image_feature(self, items: List[MultimodalDataItem]):
        """
        Projects the last hidden state from the vision model into language model space.

        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
        pixel_values = torch.stack(
            flatten_nested_list([item.pixel_values for item in items]), dim=0
        )
        pixel_values = pixel_values.to("cuda")
        pixel_values = pixel_values.to(dtype=self.language_model.dtype())

        vision_outputs = self.vision_tower(pixel_values=pixel_values).last_hidden_state
        image_features = self.multi_modal_projector(vision_outputs)
        return image_features

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        **kwargs: object,
    ) -> LogitsProcessor:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.text_config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.text_config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        >>> model = Gemma3ForConditionalGeneration.from_pretrained("google/Gemma3-test-224px-hf")
        >>> processor = AutoProcessor.from_pretrained("google/Gemma3-test-224px-hf")

        >>> prompt = "answer en Where is the cow standing?"
        >>> url = "https://huggingface.co/gv-hf/Gemma3-test-224px-hf/resolve/main/cow_beach_1.png"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt,  return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "answer en Where is the cow standing?\nbeach"
        ```"""

        # Important: position_ids in Gemma3 are 1-indexed
        # This really does cost me sometime
        positions += 1

        # Replace image id with PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        hs = general_mm_embed_routine(
            input_ids=llm_input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            image_data_embedding_func=self.get_image_feature,
            positions=positions,
        )

        return hs

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model."""
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            if "language_model" in name:
                # Gemma3ForCausalLM.load_weights(self, [(name.replace("language_model.", ""), loaded_weight)])
                causal_loaded_params = Gemma3ForCausalLM.load_weights(
                    self, [(name, loaded_weight)]
                )
                loaded_params.update(causal_loaded_params)
                continue
            else:
                # Skip lm_head.weight as it's tied with embed_tokens
                if "lm_head.weight" in name:
                    continue

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
        unloaded_params = params_dict.keys() - loaded_params
        if unloaded_params:
            pass
            # raise RuntimeError(
            #     f"Some weights are not initialized from checkpoints: {unloaded_params}")
        return loaded_params


EntryClass = Gemma3ForConditionalGeneration

AutoModel.register(Gemma3Config, Gemma3ForConditionalGeneration, exist_ok=True)
