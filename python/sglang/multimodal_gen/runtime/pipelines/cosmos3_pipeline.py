# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 video diffusion pipeline.

Cosmos3 has no separate text encoder — the transformer embeds text directly
via its Understanding (UND) pathway, and the Generation (GEN) pathway
cross-attends to the cached UND K/V at each denoising step.
"""

import os

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    Cosmos3DecodingStage,
    Cosmos3DenoisingStage,
    Cosmos3ImagePreprocessStage,
    Cosmos3LatentPreparationStage,
    Cosmos3TimestepPreparationStage,
    Cosmos3TokenizationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Cosmos3Pipeline(ComposedPipelineBase):
    """Cosmos3 diffusion pipeline shared by T2V, I2V, and T2I.

    Text is tokenized and embedded directly inside the transformer; there is
    no separate text encoder. Modality is dispatched per-request inside the
    stages from ``batch.data_type`` and ``batch.preprocessed_image``.
    """

    pipeline_name = "Cosmos3OmniDiffusersPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_tokenizer",
        "vae",
        "transformer",
        "scheduler",
        "sound_tokenizer",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        """Create Cosmos3 pipeline stages.

        Stage order:
        1. Cosmos3ImagePreprocessStage - Load + aspect-resize the I2V image (no-op otherwise)
        2. Cosmos3TokenizationStage - Tokenize with Qwen2 chat template
        3. Cosmos3LatentPreparationStage - Noise latent (or image-conditioned for I2V)
        4. Cosmos3TimestepPreparationStage - Set up scheduler timesteps
        5. Cosmos3DenoisingStage - Dual-pathway denoising (UND once, GEN per step)
        6. Cosmos3DecodingStage - VAE decode to video, or to a single image for T2I
        """
        text_tokenizer = self.get_module("text_tokenizer")
        vae = self.get_module("vae")
        transformer = self.get_module("transformer")
        scheduler = self.get_module("scheduler")
        sound_tokenizer = self.get_module("sound_tokenizer")

        guardrails_disabled = (
            os.environ.get("SGLANG_DISABLE_COSMOS3_GUARDRAILS", "0") == "1"
        )
        guardrails_on = False
        if not guardrails_disabled:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
                is_cosmos_guardrail_available,
            )

            guardrails_on = is_cosmos_guardrail_available()
            if not guardrails_on:
                logger.warning(
                    "Cosmos3 guardrails disabled because cosmos-guardrail is not "
                    "installed. Install it with: pip install cosmos-guardrail==0.3.1"
                )

        self.add_stage(Cosmos3ImagePreprocessStage())
        self.add_stage(Cosmos3TokenizationStage(tokenizer=text_tokenizer))
        if guardrails_on:
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_guardrails import (
                Cosmos3TextGuardrailStage,
            )

            self.add_stage(Cosmos3TextGuardrailStage())
        self.add_stage(Cosmos3LatentPreparationStage(vae, transformer))
        self.add_stage(Cosmos3TimestepPreparationStage(scheduler))
        self.add_stage(Cosmos3DenoisingStage(transformer, scheduler, server_args))
        self.add_stage(
            Cosmos3DecodingStage(
                vae, guardrails=guardrails_on, sound_tokenizer=sound_tokenizer
            )
        )

        logger.info(
            "Cosmos3 pipeline stages created successfully (guardrails=%s)",
            guardrails_on,
        )


EntryClass = Cosmos3Pipeline
