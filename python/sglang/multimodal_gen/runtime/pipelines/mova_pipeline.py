# SPDX-License-Identifier: Apache-2.0
"""
MoVA pipeline integration (native SGLang pipeline).
"""

from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.mova import MovaPipelineConfig
from sglang.multimodal_gen.configs.sample.mova import MovaSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ConditioningStage,
    ImageVAEEncodingStage,
    InputValidationStage,
    MovaDecodingStage,
    MovaDenoisingStage,
    MovaLatentPreparationStage,
    MovaTimestepPreparationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class MovaPipeline(ComposedPipelineBase):
    """MoVA pipeline with SGLang stage orchestration."""

    pipeline_name = "MoVA"
    is_video_pipeline = True
    _required_config_modules = [
        "video_vae",
        "audio_vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "video_dit",
        "video_dit2",
        "audio_dit",
        "dual_tower_bridge",
    ]
    pipeline_config_cls = MovaPipelineConfig
    sampling_params_cls = MovaSamplingParams

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        """
        Initialize the pipeline.

        MoVA supports Context Parallel (sequence parallel) through USPAttention,
        which uses Ulysses-style all-to-all communication for distributed attention.
        """
        if server_args.sp_degree > 1:
            logger.info(
                "MoVA Context Parallel enabled with sp_degree=%d. "
                "Using USPAttention for distributed self-attention.",
                server_args.sp_degree,
            )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )
        self.add_stage(
            stage_name="prompt_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage(stage_name="conditioning_stage", stage=ConditioningStage())
        if getattr(self.get_module("video_dit"), "require_vae_embedding", True):
            self.add_stage(
                stage_name="image_latent_preparation_stage",
                stage=ImageVAEEncodingStage(vae=self.get_module("video_vae")),
            )
        self.add_stage(
            stage_name="mova_latent_preparation_stage",
            stage=MovaLatentPreparationStage(
                audio_vae=self.get_module("audio_vae"),
                require_vae_embedding=getattr(
                    self.get_module("video_dit"), "require_vae_embedding", True
                ),
            ),
        )
        self.add_stage(
            stage_name="mova_timestep_preparation_stage",
            stage=MovaTimestepPreparationStage(
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            stage_name="mova_denoising_stage",
            stage=MovaDenoisingStage(
                video_dit=self.get_module("video_dit"),
                video_dit2=self.get_module("video_dit2"),
                audio_dit=self.get_module("audio_dit"),
                dual_tower_bridge=self.get_module("dual_tower_bridge"),
                scheduler=self.get_module("scheduler"),
            ),
        )
        self.add_stage(
            stage_name="mova_decoding_stage",
            stage=MovaDecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            ),
        )


class MoVAPipelineAlias(MovaPipeline):
    pipeline_name = "MoVAPipeline"


EntryClass = [MovaPipeline, MoVAPipelineAlias]
