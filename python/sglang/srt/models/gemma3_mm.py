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
from transformers import AutoModel, PretrainedConfig, PreTrainedModel
from transformers.utils import is_torchdynamo_compiling

from sglang.srt.configs import Gemma3Config
from sglang.srt.hf_transformers_utils import get_processor
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import Gemma3RMSNorm
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.multi_modality_padding import (
    MultiModalityDataPaddingPatternTokenPairs,
)
from sglang.srt.managers.schedule_batch import ImageInputs
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


#
#
# class SiglipVisionModel(nn.Module):
#     """Vision model for Gemma3 multimodal."""
#
#     def __init__(
#         self,
#         vision_config: PretrainedConfig,
#         quant_config: Optional[QuantizationConfig] = None,
#         prefix: str = "",
#     ):
#         # self.model = SiglipModel(vision_config)
#         super().__init__()
#         self.config = vision_config
#         self.embed_dim = vision_config.hidden_size
#         self.image_size = vision_config.image_size
#         self.patch_size = vision_config.patch_size
#         self.num_patches = (self.image_size // self.patch_size) ** 2
#
#         self.patch_embed = nn.Conv2d(
#             in_channels=3,
#             out_channels=self.embed_dim,
#             kernel_size=self.patch_size,
#             stride=self.patch_size,
#             bias=False,
#         )
#
#         self.class_embedding = nn.Parameter(torch.randn(1, 1, self.embed_dim))
#         self.positional_embedding = nn.Parameter(
#             torch.randn(1, self.num_patches + 1, self.embed_dim)
#         )
#
#         self.pre_layernorm = nn.LayerNorm(self.embed_dim)
#
#         self.layers = nn.ModuleList(
#             [
#                 SiglipVisionLayer(
#                     config=vision_config,
#                     quant_config=quant_config,
#                     prefix=add_prefix(f"layers.{i}", prefix),
#                 )
#                 for i in range(vision_config.num_hidden_layers)
#             ]
#         )
#
#         self.post_layernorm = nn.LayerNorm(self.embed_dim)
#
#     def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
#         batch_size = pixel_values.shape[0]
#
#         # Extract patches
#         x = self.patch_embed(pixel_values)
#         x = x.flatten(2).transpose(1, 2)  # B, N, C
#
#         # Add class token
#         class_embedding = self.class_embedding.expand(batch_size, -1, -1)
#         x = torch.cat([class_embedding, x], dim=1)
#
#         # Add positional embedding
#         x = x + self.positional_embedding
#
#         # Apply pre-layernorm
#         x = self.pre_layernorm(x)
#
#         # Apply transformer layers
#         for layer in self.layers:
#             x = layer(x)
#
#         # Apply post-layernorm
#         x = self.post_layernorm(x)
#
#         return x


class SiglipVisionLayer(nn.Module):
    """Vision transformer layer for Gemma3 multimodal."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.attention = VisionAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            projection_size=config.hidden_size,
            use_qkv_parallel=False,
            use_context_forward=False,
            softmax_in_single_precision=False,
            flatten_batch=True,
            quant_config=quant_config,
            prefix=add_prefix("attention", prefix),
        )
        self.attention_layernorm = nn.LayerNorm(config.hidden_size)

        self.mlp = nn.Sequential(
            ColumnParallelLinear(
                config.hidden_size,
                config.intermediate_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp.0", prefix),
            ),
            nn.GELU(),
            RowParallelLinear(
                config.intermediate_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("mlp.2", prefix),
            ),
        )
        self.mlp_layernorm = nn.LayerNorm(config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.attention_layernorm(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.mlp_layernorm(hidden_states)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        hidden_states_parallel, _ = mlp_fc1(hidden_states)
        hidden_states_parallel = mlp_act(hidden_states_parallel)
        hidden_states, _ = mlp_fc2(hidden_states_parallel)

        hidden_states = residual + hidden_states
        return hidden_states


class Gemma3MultiModalProjector(nn.Module):
    """Projector for Gemma3 multimodal."""

    def __init__(self, config: PretrainedConfig):
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
        self, input_ids: List[int], image_inputs: ImageInputs
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

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def get_image_features(self, pixel_values: torch.Tensor):
        """
        Projects the last hidden state from the vision model into language model space.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_images, image_length, embed_dim)`).
        """
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

        image_inputs = []
        if forward_batch.image_inputs is not None:
            image_inputs = [
                img for img in forward_batch.image_inputs if img is not None
            ]

        if not forward_batch.forward_mode.is_decode() and len(image_inputs) != 0:
            image_input: ImageInputs = image_inputs[0]
            image_features = self.get_image_features(image_input.pixel_values)
            if input_ids is None:
                raise ValueError("Unimplemented")
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(
                        self.config.image_token_index,
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                )
            else:
                # boolean-masking image tokens
                special_image_mask = torch.isin(
                    input_ids,
                    torch.tensor(image_input.pad_values, device=input_ids.device),
                ).unsqueeze(-1)
                # Important: clamp after extracting original image boundaries
                llm_input_ids.clamp_(min=0, max=self.vocab_size - 1)
                inputs_embeds = self.get_input_embeddings()(llm_input_ids)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
                    inputs_embeds.device
                )
            if (
                not is_torchdynamo_compiling()
                and inputs_embeds[special_image_mask].numel() != image_features.numel()
            ):
                pass
                # image_tokens_in_text = special_image_mask.sum(dim=1).sum(dim=0)[0]
                # raise ValueError(
                #     f"Number of images does not match number of special image tokens in the input text. "
                #     f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                #     "tokens from image embeddings."
                # )
            else:
                image_features = image_features.to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features
                )
        else:
            llm_input_ids.clamp_(min=0, max=self.vocab_size - 1)
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        outputs = self.language_model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=inputs_embeds,
            **kwargs,
        )

        return outputs

    def tie_weights(self):
        return self.language_model.tie_weights()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Load weights for the model."""
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                if "vision" in name:
                    continue
                else:
                    # TODO: vision attention not adapted for now, use default siglip
                    name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
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


EntryClass = Gemma3ForConditionalGeneration

AutoModel.register(Gemma3Config, Gemma3ForConditionalGeneration, exist_ok=True)
