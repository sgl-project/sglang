# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""StableDiffusion3 pipeline implementation."""

import torch

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    InputValidationStage,
    PipelineStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SD3ConditioningStage(PipelineStage):
    """Merge CLIP-T, CLIP-G and T5 embeddings into unified prompt/pooled tensors."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        batch.prompt_embeds, batch.pooled_embeds = self._merge(
            batch.prompt_embeds, batch.pooled_embeds
        )
        if batch.do_classifier_free_guidance:
            batch.negative_prompt_embeds, batch.neg_pooled_embeds = self._merge(
                batch.negative_prompt_embeds, batch.neg_pooled_embeds
            )
        return batch

    @staticmethod
    def _merge(
        embeds_list: list[torch.Tensor],
        pooled_list: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Merge 3 encoder outputs into unified prompt/pooled tensors.

        SD3-medium uses exactly 3 text encoders (CLIP-L, CLIP-G, T5).
        Returns single-element lists to match the batch field format expected
        by downstream stages (get_pos_prompt_embeds accesses index [0]).
        """
        if len(embeds_list) != 3:
            raise ValueError(
                f"SD3 requires exactly 3 prompt embedding tensors, got {len(embeds_list)}."
            )
        if len(pooled_list) < 2:
            raise ValueError(
                f"SD3 requires at least 2 pooled embedding tensors, got {len(pooled_list)}."
            )

        clipt, clipg, t5 = embeds_list
        clip_merged = torch.cat([clipt, clipg], dim=-1)
        clip_merged = torch.nn.functional.pad(
            clip_merged, (0, t5.shape[-1] - clip_merged.shape[-1])
        )
        merged_embeds = [torch.cat([clip_merged, t5], dim=-2)]
        merged_pooled = [torch.cat([pooled_list[0], pooled_list[1]], dim=-1)]
        return merged_embeds, merged_pooled


class StableDiffusion3Pipeline(ComposedPipelineBase):
    """StableDiffusion3 pipeline implementation."""

    pipeline_name = "StableDiffusion3Pipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        self.add_stage(
            TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                    self.get_module("text_encoder_3"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                    self.get_module("tokenizer_3"),
                ],
            ),
            "prompt_encoding_stage_primary",
        )

        self.add_stage(SD3ConditioningStage())

        self.add_standard_timestep_preparation_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = StableDiffusion3Pipeline
