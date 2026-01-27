# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Deepseek-OCR model compatible with HuggingFace weights."""

from typing import Optional, TypeAlias

import torch
from torch import Tensor, nn

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.models.deepseek import DeepseekForCausalLM
from sglang.srt.models.deepseek_ocr import DeepseekOCRForCausalLM
from sglang.srt.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM
from sglang.srt.models.transformers import maybe_prefix

MultiModalEmbeddings: TypeAlias = list[Tensor] | Tensor | tuple[Tensor, ...]

from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.models.deepencoder2 import (
    build_qwen2_decoder_as_encoder,
    build_sam_vit_b,
)
from sglang.srt.models.deepseek_ocr import MlpProjector


class NoRepeatNGramLogitsProcessor:
    def __init__(
        self,
        ngram_size: int,
        window_size: int,
        whitelist_token_ids: set[int] | None = None,
    ):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids or set()

    def __call__(
        self,
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        if len(output_ids) < self.ngram_size:
            return logits

        current_prefix = tuple(output_ids[-(self.ngram_size - 1) :])

        search_start = max(0, len(output_ids) - self.window_size)
        search_end = len(output_ids) - self.ngram_size + 1

        banned_tokens = set()
        for i in range(search_start, search_end):
            ngram = tuple(output_ids[i : i + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned_tokens.add(ngram[-1])

        banned_tokens = banned_tokens - self.whitelist_token_ids

        if banned_tokens:
            logits[list(banned_tokens)] = -float("inf")

        return logits


class DeepseekOCR2ForCausalLM(DeepseekOCRForCausalLM):

    def __init__(
        self,
        *,
        config: DeepseekVLV2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        # super().__init__()

        self.config = config

        self.vision_config = config.vision_config
        self.projector_config = config.projector_config
        self.text_config = config.text_config

        n_embed = 1280

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special token for image token sequence format
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # <|view_separator|>, <|\n|>
            self.image_newline = nn.Parameter(torch.randn(n_embed) * embed_std)
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )

        if self.text_config.topk_method == "noaux_tc":
            self.model = DeepseekV3ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        elif not self.text_config.use_mla:
            self.model = DeepseekForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )
        else:
            self.model = DeepseekV2ForCausalLM(
                config=config.text_config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "language"),
            )

        self.sam_model = build_sam_vit_b()
        self.vision_model = build_qwen2_decoder_as_encoder()
        n_embed = 1280
        self.projector = MlpProjector(
            projector_type="linear",
            input_dim=2048,
            n_embed=n_embed,
        )

        self.tile_tag = config.tile_tag
        self.global_view_pos = config.global_view_pos

        # special token for image token sequence format
        n_embed = self.projector_config.n_embed
        embed_std = 1 / torch.sqrt(torch.tensor(n_embed, dtype=torch.float32))
        if self.tile_tag == "2D":
            # This is a typo in original implementation
            self.view_seperator = nn.Parameter(torch.randn(n_embed) * embed_std)
        else:
            raise ValueError(
                f"Only 2D tile_tag is supported currently, got: {self.tile_tag}"
            )


EntryClass = [DeepseekOCR2ForCausalLM]
