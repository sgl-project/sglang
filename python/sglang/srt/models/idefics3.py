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

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/idefics3.py
"""Inference-only Idefics3 model compatible with HuggingFace weights."""

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import Idefics3Config

from sglang.srt.layers.linear import RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.idefics2 import (
    Idefics2VisionTransformer as Idefics3VisionTransformer,
)
from sglang.srt.models.llama import LlamaModel
from sglang.srt.utils import add_prefix


class Idefics3ImagePixelInputs:
    """Container for image pixel inputs."""

    def __init__(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
        num_patches: torch.Tensor,
    ):
        self.pixel_values = pixel_values
        self.pixel_attention_mask = pixel_attention_mask
        self.num_patches = num_patches

    def __getitem__(self, key):
        if key == "pixel_values":
            return self.pixel_values
        elif key == "pixel_attention_mask":
            return self.pixel_attention_mask
        elif key == "num_patches":
            return self.num_patches
        elif key == "type":
            return "pixel_values"
        else:
            raise KeyError(f"Unknown key: {key}")


class Idefics3ImageEmbeddingInputs:
    """Container for image embedding inputs."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __getitem__(self, key):
        if key == "data":
            return self.data
        elif key == "type":
            return "image_embeds"
        else:
            raise KeyError(f"Unknown key: {key}")


class Idefics3SimpleMLP(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        input_size = config.vision_config.hidden_size * (config.scale_factor**2)
        output_size = config.text_config.hidden_size
        self.proj = RowParallelLinear(
            input_size,
            output_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("proj", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.proj(x)
        return out


class Idefics3Connector(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = Idefics3SimpleMLP(
            config,
            quant_config,
            prefix=add_prefix("modality_projection", prefix),
        )

    def pixel_shuffle(self, x: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(
            bsz,
            int(width / scale_factor),
            int(height / scale_factor),
            embed_dim * (scale_factor**2),
        )
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class Idefics3Model(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = self.config.text_config.vocab_size
        self.vision_model = Idefics3VisionTransformer(
            config.vision_config,
            quant_config=quant_config,
            prefix=add_prefix("vision_model", prefix),
        )
        self.connector = Idefics3Connector(
            config,
            quant_config,
            prefix=add_prefix("connector", prefix),
        )
        self.text_model = LlamaModel(
            config.text_config,
            quant_config=quant_config,
            prefix=add_prefix("text_model", prefix),
        )

        self.image_seq_len = int(
            ((config.vision_config.image_size // config.vision_config.patch_size) ** 2)
            / (config.scale_factor**2)
        )
        self.image_token_id = self.config.image_token_id

    def image_pixels_to_features(
        self,
        pixel_values: torch.Tensor,
        pixel_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: we skip the step to select the vision feature layer since
        # this is already done inside the vision tower
        pixel_values = pixel_values.to(
            dtype=self.vision_model.embeddings.patch_embedding.weight.dtype
        )  # fp16 compatibility

        # Remove padding images - padding images are full 0.
        nb_values_per_image = pixel_values.shape[1:].numel()
        real_images_inds = (pixel_values == 0.0).sum(
            dim=(-1, -2, -3)
        ) != nb_values_per_image
        pixel_values = pixel_values[real_images_inds].contiguous()

        # Handle the vision attention mask
        # Remove padding images from the mask
        pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

        patch_size = self.config.vision_config.patch_size
        patches_subgrid = pixel_attention_mask.unfold(
            dimension=1, size=patch_size, step=patch_size
        )
        patches_subgrid = patches_subgrid.unfold(
            dimension=2, size=patch_size, step=patch_size
        )
        patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

        # Get sequence from the vision encoder
        image_hidden_states = self.vision_model(
            pixel_values=pixel_values,
            patch_attention_mask=patch_attention_mask,
        )

        return image_hidden_states

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.text_model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
    ) -> torch.Tensor:
        hidden_states = self.text_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
        )
        return hidden_states


class Idefics3ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: Idefics3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.model = Idefics3Model(
            config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.image_token_id = self.config.image_token_id

        self.lm_head = ParallelLMHead(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )
        if self.config.text_config.tie_word_embeddings:
            self.lm_head.weight = self.model.text_model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(config.text_config)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

    def _parse_and_validate_image_input(
        self, **kwargs: Any
    ) -> Optional[Union[Idefics3ImagePixelInputs, Idefics3ImageEmbeddingInputs]]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if image_embeds is not None:
            if not isinstance(image_embeds, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of image embeddings. "
                    f"Got type: {type(image_embeds)}"
                )

            return Idefics3ImageEmbeddingInputs(data=image_embeds)

        if pixel_values is not None:
            if not isinstance(pixel_values, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                )

            pixel_attention_mask = kwargs.pop("pixel_attention_mask")
            if not isinstance(pixel_attention_mask, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of pixel_attention_mask. "
                    f"Got type: {type(pixel_attention_mask)}"
                )

            num_patches = kwargs.pop("num_patches")
            if not isinstance(num_patches, (torch.Tensor, list)):
                raise ValueError(
                    "Incorrect type of num_patches. " f"Got type: {type(num_patches)}"
                )

            return Idefics3ImagePixelInputs(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                num_patches=num_patches,
            )

        raise AssertionError("This line should be unreachable.")

    def _process_image_pixels(self, inputs: Idefics3ImagePixelInputs) -> torch.Tensor:
        pixel_values = inputs["pixel_values"]
        pixel_attention_mask = inputs["pixel_attention_mask"]

        return self.model.image_pixels_to_features(
            pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def _process_image_input(
        self,
        image_input: Union[Idefics3ImagePixelInputs, Idefics3ImageEmbeddingInputs],
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if image_input["type"] == "image_embeds":
            return image_input["data"]

        image_features = self._process_image_pixels(image_input)
        image_features = self.model.connector(image_features)

        num_patches = image_input["num_patches"]
        return [e.flatten(0, 1) for e in image_features.split(num_patches.tolist())]

    def get_multimodal_embeddings(self, **kwargs: Any) -> List[torch.Tensor]:
        image_input = self._parse_and_validate_image_input(**kwargs)
        if image_input is None:
            return []

        return self._process_image_input(image_input)

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None and len(multimodal_embeddings) != 0:
            inputs_embeds = self._merge_multimodal_embeddings(
                input_ids,
                inputs_embeds,
                multimodal_embeddings,
                self.config.image_token_id,
            )
        return inputs_embeds

    def _merge_multimodal_embeddings(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        multimodal_embeddings: List[torch.Tensor],
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Merge multimodal embeddings into text embeddings."""
        mask = input_ids == placeholder_token_id
        num_expected_tokens = mask.sum().item()

        # Flatten multimodal embeddings
        flattened_embeddings = []
        for emb in multimodal_embeddings:
            if emb.dim() == 2:
                flattened_embeddings.append(emb)
            else:
                flattened_embeddings.append(emb.flatten(0, -2))

        if flattened_embeddings:
            mm_embeddings = torch.cat(flattened_embeddings, dim=0)
            if mm_embeddings.shape[0] != num_expected_tokens:
                raise ValueError(
                    f"Number of multimodal tokens ({mm_embeddings.shape[0]}) "
                    f"does not match number of placeholders ({num_expected_tokens})"
                )
            inputs_embeds[mask] = mm_embeddings

        return inputs_embeds

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        if input_embeds is None:
            vision_embeddings = self.get_multimodal_embeddings(**kwargs)
            input_embeds = self.get_input_embeddings(input_ids, vision_embeddings)
            input_ids = None

        hidden_states = self.model(input_ids, positions, forward_batch, input_embeds)

        if not get_embedding:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch
            )
        else:
            return self.pooler(hidden_states, forward_batch)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if self.config.text_config.tie_word_embeddings and "lm_head.weight" in name:
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)


EntryClass = Idefics3ForConditionalGeneration
