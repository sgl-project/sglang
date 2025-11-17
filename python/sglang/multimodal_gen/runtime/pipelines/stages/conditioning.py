# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Conditioning stage for diffusion pipelines.
"""

import torch

from sglang.multimodal_gen.configs.pipelines import StableDiffusion3PipelineConfig
from sglang.multimodal_gen.runtime.pipelines.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines.stages.base import PipelineStage
from sglang.multimodal_gen.runtime.pipelines.stages.validators import (
    StageValidators as V,
)
from sglang.multimodal_gen.runtime.pipelines.stages.validators import VerificationResult
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ConditioningStage(PipelineStage):
    """
    Stage for applying conditioning to the diffusion process.

    This stage handles the application of conditioning, such as classifier-free guidance,
    to the diffusion process.
    """

    @torch.no_grad()
    def forward(
        self,
        batch: Req,
        server_args: ServerArgs,
    ) -> Req:
        """
        Apply conditioning to the diffusion process.

        Args:
            batch: The current batch information.
            server_args: The inference arguments.

        Returns:
            The batch with applied conditioning.
        """
        is_stable_diffusion3 = isinstance(
            server_args.pipeline_config, StableDiffusion3PipelineConfig
        )
        if is_stable_diffusion3:
            prompt_embeds_list = batch.prompt_embeds
            negative_prompt_embeds_list = batch.negative_prompt_embeds
            pooled_embeds_list = batch.pooled_embeds
            neg_pooled_embeds_list = batch.neg_pooled_embeds
            # prompt deal
            clipt_encoder_embeds = prompt_embeds_list[0]
            clipg_encoder_embeds = prompt_embeds_list[1]
            t5_encoder_embeds = prompt_embeds_list[2]
            clipt_pool_embeds = pooled_embeds_list[0]
            clipg_pool_embeds = pooled_embeds_list[1]
            clip_prompt_embeds = torch.cat(
                [clipt_encoder_embeds, clipg_encoder_embeds], dim=-1
            )
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_encoder_embeds.shape[-1] - clip_prompt_embeds.shape[-1]),
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_encoder_embeds], dim=-2)
            pooled_prompt_embeds = torch.cat(
                [clipt_pool_embeds, clipg_pool_embeds], dim=-1
            )
            batch.prompt_embeds = [prompt_embeds]
            batch.pooled_embeds = [pooled_prompt_embeds]

            # negative prompt deal
            if batch.do_classifier_free_guidance:
                negative_clipt_encoder_embeds = negative_prompt_embeds_list[0]
                negative_clipg_encoder_embeds = negative_prompt_embeds_list[1]
                negative_t5_encoder_embeds = negative_prompt_embeds_list[2]
                negative_clipt_pool_embeds = neg_pooled_embeds_list[0]
                negative_clipg_pool_embeds = neg_pooled_embeds_list[1]
                negative_clip_prompt_embeds = torch.cat(
                    [negative_clipt_encoder_embeds, negative_clipg_encoder_embeds],
                    dim=-1,
                )
                negative_clip_prompt_embeds = torch.nn.functional.pad(
                    negative_clip_prompt_embeds,
                    (
                        0,
                        negative_t5_encoder_embeds.shape[-1]
                        - negative_clip_prompt_embeds.shape[-1],
                    ),
                )
                negative_prompt_embeds = torch.cat(
                    [negative_clip_prompt_embeds, negative_t5_encoder_embeds], dim=-2
                )
                negative_pooled_prompt_embeds = torch.cat(
                    [negative_clipt_pool_embeds, negative_clipg_pool_embeds], dim=-1
                )
                batch.negative_prompt_embeds = [negative_prompt_embeds]
                batch.neg_pooled_embeds = [negative_pooled_prompt_embeds]


            return batch
        # TODO!!
        if not batch.do_classifier_free_guidance:
            return batch
        else:
            return batch

        logger.info("batch.negative_prompt_embeds: %s", batch.negative_prompt_embeds)
        logger.info(
            "do_classifier_free_guidance: %s", batch.do_classifier_free_guidance
        )
        logger.info("cfg_scale: %s", batch.guidance_scale)

        # Ensure negative prompt embeddings are available
        assert (
            batch.negative_prompt_embeds is not None
        ), "Negative prompt embeddings are required for classifier-free guidance"

        # Concatenate primary embeddings and masks
        batch.prompt_embeds = torch.cat(
            [batch.negative_prompt_embeds, batch.prompt_embeds]
        )
        if batch.attention_mask is not None:
            batch.attention_mask = torch.cat(
                [batch.negative_attention_mask, batch.attention_mask]
            )

        # Concatenate secondary embeddings and masks if present
        if batch.prompt_embeds_2 is not None:
            batch.prompt_embeds_2 = torch.cat(
                [batch.negative_prompt_embeds_2, batch.prompt_embeds_2]
            )
        if batch.attention_mask_2 is not None:
            batch.attention_mask_2 = torch.cat(
                [batch.negative_attention_mask_2, batch.attention_mask_2]
            )

        return batch

    def verify_input(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify conditioning stage inputs."""
        result = VerificationResult()
        result.add_check(
            "do_classifier_free_guidance",
            batch.do_classifier_free_guidance,
            V.bool_value,
        )
        result.add_check("guidance_scale", batch.guidance_scale, V.positive_float)
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        result.add_check(
            "negative_prompt_embeds",
            batch.negative_prompt_embeds,
            lambda x: not batch.do_classifier_free_guidance or V.list_not_empty(x),
        )
        return result

    def verify_output(self, batch: Req, server_args: ServerArgs) -> VerificationResult:
        """Verify conditioning stage outputs."""
        result = VerificationResult()
        result.add_check("prompt_embeds", batch.prompt_embeds, V.list_not_empty)
        return result
