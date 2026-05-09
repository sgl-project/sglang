# coding=utf-8
# Adapted from Qwen2.5-VL SGLang implementation

import logging
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.configs import DotsOCRConfig
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import MultimodalDataItem, MultimodalInputs
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.auto_loader import AutoWeightsLoader, WeightsMapper
from sglang.srt.models.dots_vlm_vit import DotsVisionTransformer
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


class DotsOCRForCausalLM(nn.Module):
    def __init__(
        self,
        config: DotsOCRConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Initialize vision transformer
        self.visual = DotsVisionTransformer(
            config.vision_config,
        )

        # Initialize language model
        self.model = Qwen2ForCausalLM(config, quant_config)

        # Initialize LM head
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=add_prefix("lm_head", prefix),
            )

        self.logits_processor = LogitsProcessor(config)

    def pad_input_ids(self, input_ids: List[int], mm_inputs: MultimodalInputs):
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        # Extract pixel values and grid information (following reference pattern)
        pixel_values = torch.cat([item.feature for item in items], dim=0).type(
            self.visual.dtype
        )
        image_grid_thw = torch.concat(
            [item.image_grid_thw for item in items], dim=0
        ).to(self.visual.device)

        # Add dimension checks like in reference code
        assert pixel_values.dim() == 2, f"{pixel_values.dim()=}"
        assert image_grid_thw.dim() == 2, f"{image_grid_thw.dim()=}"

        # Process through vision tower
        image_embeds = self.visual(pixel_values, image_grid_thw)

        # Ensure consistent dtype for FlashInfer compatibility
        # Force bfloat16 to match model's expected dtype
        if hasattr(self.model, "embed_tokens"):
            target_dtype = self.model.embed_tokens.weight.dtype
            if image_embeds.dtype != target_dtype:
                image_embeds = image_embeds.to(target_dtype)

        return image_embeds

    def _pad_vit_attn_dummy_heads(self, name: str, loaded_weight: torch.Tensor):
        """pad attn qkv weights for dummy heads"""
        num_dummy_heads = self.config.vision_config.num_dummy_heads
        if num_dummy_heads == 0:
            return loaded_weight
        head_dim = self.config.vision_config.head_dim

        if "attn.qkv_proj" in name:
            wq, wk, wv = loaded_weight.chunk(3, dim=0)
            if name.endswith(".weight"):
                dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]
            elif name.endswith(".bias"):
                dummy_shape = [num_dummy_heads, head_dim]
            else:
                raise RuntimeError(f"Unsupported weight with name={name}")
            pad_func = lambda x: torch.cat(
                [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
            ).flatten(0, 1)
            wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
            loaded_weight = torch.cat([wq, wk, wv], dim=0)
        if "attn.proj.weight" in name:
            padded_weight = loaded_weight.new_zeros(
                loaded_weight.shape[0], head_dim * num_dummy_heads
            )
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
        if "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
            padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
            loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
        return loaded_weight

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> torch.Tensor:
        hidden_states = general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            multimodal_model=self,
            language_model=self.model,
        )
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for the model using the AutoWeightsLoader walker."""

        def _pad_vision(stream):
            for name, w in stream:
                if name.startswith("vision_tower."):
                    # Pad uses the post-rename name (visual.* + qkv_proj).
                    renamed = name.replace("vision_tower.", "visual.").replace(
                        "attn.qkv.", "attn.qkv_proj."
                    )
                    yield name, self._pad_vit_attn_dummy_heads(renamed, w)
                else:
                    yield name, w

        # Checkpoint conventions:
        #   * ``vision_tower.<...>`` → runtime ``self.visual``
        #   * ``attn.qkv`` → runtime ``attn.qkv_proj`` (VisionAttention).
        # Top-level ``model.*`` and ``lm_head.*`` are wrapped under the
        # runtime ``self.model = Qwen2ForCausalLM(...)`` python attribute,
        # so route them through ``model.<...>``.
        mapper = WeightsMapper(
            orig_to_new_substr={"attn.qkv.": "attn.qkv_proj."},
            orig_to_new_prefix={
                "vision_tower.": "visual.",
                "model.": "model.model.",
                "lm_head.": "model.lm_head.",
            },
        )
        # ``self.lm_head`` is either a tied alias of ``self.model.embed_tokens``
        # (handled by the inner ``Qwen2ForCausalLM.load_weights``) or a
        # never-loaded duplicate; either way nothing in the checkpoint targets
        # the outer attribute after the mapper rewrites ``lm_head.`` →
        # ``model.lm_head.``.
        return AutoWeightsLoader(self).load_weights(_pad_vision(weights), mapper=mapper)

    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight


EntryClass = [DotsOCRForCausalLM]
