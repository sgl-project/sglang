# SPDX-License-Identifier: Apache-2.0
"""
LongCat video diffusion pipeline implementation (Phase 1: Wrapper).

This module contains a wrapper implementation of the LongCat video diffusion pipeline
using FastVideo's modular pipeline architecture with the original LongCat modules.
"""

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
    InputValidationStage,
    LatentPreparationStage,
    TextEncodingStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.LongCatRefineInitStage import (
    LongCatRefineInitStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.LongCatRefineTimestepStage import (
    LongCatRefineTimestepStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LongCatPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LongCatPipeline"

    """
    LongCat video diffusion pipeline with LoRA support.

    Phase 1 implementation using wrapper modules from third_party/longcat_video.
    This validates the pipeline infrastructure before full FastVideo integration.
    """

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        """Initialize LongCat-specific components."""
        # LongCat uses FlowMatchEulerDiscreteScheduler which is already loaded
        # from the model_index.json, so no need to override

        # Enable BSA (Block Sparse Attention) if configured
        pipeline_config = server_args.pipeline_config
        transformer = self.get_module("transformer", None)
        if transformer is None:
            return
        # If user toggles BSA via CLI/config
        if pipeline_config.enable_bsa:
            # Build effective BSA params:
            # 1) from explicit CLI overrides if provided
            # 2) else from pipeline_config.bsa_params
            # 3) else fall back to reasonable defaults
            bsa_params_cfg = getattr(pipeline_config, "bsa_params", None) or {}
            sparsity = getattr(pipeline_config, "bsa_sparsity", None)
            cdf_threshold = getattr(pipeline_config, "bsa_cdf_threshold", None)
            chunk_q = getattr(pipeline_config, "bsa_chunk_q", None)
            chunk_k = getattr(pipeline_config, "bsa_chunk_k", None)

            effective_bsa_params = (
                dict(bsa_params_cfg) if isinstance(bsa_params_cfg, dict) else {}
            )
            if sparsity is not None:
                effective_bsa_params["sparsity"] = sparsity
            if cdf_threshold is not None:
                effective_bsa_params["cdf_threshold"] = cdf_threshold
            if chunk_q is not None:
                effective_bsa_params["chunk_3d_shape_q"] = chunk_q
            if chunk_k is not None:
                effective_bsa_params["chunk_3d_shape_k"] = chunk_k
            # Provide defaults if still missing
            effective_bsa_params.setdefault("sparsity", 0.9375)
            effective_bsa_params.setdefault("chunk_3d_shape_q", [4, 4, 4])
            effective_bsa_params.setdefault("chunk_3d_shape_k", [4, 4, 4])

            if hasattr(transformer, "enable_bsa"):
                logger.info(
                    "Enabling Block Sparse Attention (BSA) for LongCat transformer"
                )
                transformer.enable_bsa()
                # Propagate params to all attention modules
                if hasattr(transformer, "blocks"):
                    try:
                        for blk in transformer.blocks:
                            if hasattr(blk, "self_attn"):
                                blk.self_attn.bsa_params = effective_bsa_params
                    except Exception as e:
                        logger.warning("Failed to set BSA params on all blocks: %s", e)
                logger.info("BSA parameters in effect: %s", effective_bsa_params)
            else:
                logger.warning(
                    "BSA is enabled in config but transformer does not support it"
                )
        else:
            # Explicitly disable if present
            if hasattr(transformer, "disable_bsa"):
                transformer.disable_bsa()

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Set up pipeline stages with proper dependency injection."""

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

        # Add refine initialization stage (will be skipped if not refining)
        self.add_stage(
            stage_name="longcat_refine_init_stage",
            stage=LongCatRefineInitStage(vae=self.get_module("vae")),
        )

        # First prepare generic timesteps (for non-refine paths)
        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        # Then override timesteps for refinement (will be a no-op if not refining),
        # matching LongCat's generate_refine schedule.
        self.add_stage(
            stage_name="longcat_refine_timestep_stage",
            stage=LongCatRefineTimestepStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=LatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer", None),
            ),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                transformer_2=self.get_module("transformer_2", None),
                vae=self.get_module("vae"),
            ),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = LongCatPipeline
