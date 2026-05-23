# SPDX-License-Identifier: Apache-2.0
#
# SANA-WM TI2V Pipeline.
#
# Stage ordering (Hybrid style):
#   InputValidationStage
#   → TextEncodingStage (Gemma-2, same as SANA T2I)
#   → SanaWMBeforeDenoisingStage (first-frame VAE, Plücker, noise latents, timesteps)
#   → DenoisingStage (standard)
#   → DecodingStage (standard, uses LTX-2 VAE decoder)
#
# Optional two-stage pipeline (SanaWMTwoStagePipeline):
#   Same as above + LTX2RefinementStage for high-quality up-sampling.
#
# pipeline_name must match _class_name in SANA-WM model_index.json.
# Default expected: "SanaWMPipeline" (update if the HF checkpoint differs).

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class SanaWMPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    SANA-WM TI2V pipeline.

    Expects the following modules from the HuggingFace model directory:
    - text_encoder: Gemma-2-2b-it
    - tokenizer: Gemma-2 tokenizer
    - vae: LTX-2 VAE
    - transformer: SanaWMTransformer3DModel
    - scheduler: FlowMatchEulerDiscreteScheduler
    """

    # Must match `_class_name` in model_index.json of the checkpoint.
    pipeline_name = "SanaWMPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # 1. Input validation (standard)
        self.add_stage(InputValidationStage())

        # 2. Text encoding via Gemma-2 (standard — handles deduplication, masking, CFG)
        self.add_stage(
            TextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
            "prompt_encoding_stage",
        )

        # 3. Video-specific pre-processing (first-frame VAE, Plücker, noise init, timesteps)
        self.add_stage(
            SanaWMBeforeDenoisingStage(
                vae=self.get_module("vae"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline_config=server_args.pipeline_config,
            ),
            "sana_wm_before_denoising",
        )

        # 4. Standard denoising loop (calls transformer.forward at each step)
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 5. Standard VAE decoding (LTX-2 VAE decoder)
        self.add_standard_decoding_stage()


class SanaWMTwoStagePipeline(SanaWMPipeline):
    """
    Optional two-stage SANA-WM pipeline with LTX-2 refinement.

    Stage-1: SanaWM DiT (generates coarse 720p video)
    Stage-2: LTX-2 Refiner (σ_start=0.9, ~3 Euler steps, sharpens detail)

    Requires `ltx2_transformer` module in the model directory (or component_paths).
    Falls back gracefully to single-stage if LTX-2 modules are not found.
    """

    pipeline_name = "SanaWMTwoStagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    _optional_config_modules = [
        "transformer_2",  # LTX-2 refiner transformer (optional)
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # Stages 1-5: same as SanaWMPipeline
        super().create_pipeline_stages(server_args)

        # Stage 6 (optional): LTX-2 refinement if refiner transformer is loaded
        transformer_2 = self.get_module("transformer_2", required=False)
        if transformer_2 is not None:
            try:
                from sglang.multimodal_gen.runtime.pipelines_core.stages import (
                    LTX2RefinementStage,
                )
                self.add_stage(
                    LTX2RefinementStage(
                        transformer=transformer_2,
                        scheduler=self.get_module("scheduler"),
                    ),
                    "ltx2_refinement",
                )
                logger.info("LTX-2 refinement stage added (two-stage mode).")
            except ImportError:
                logger.warning(
                    "LTX2RefinementStage not available; running single-stage mode."
                )
        else:
            logger.info(
                "No refiner transformer found; running SanaWM in single-stage mode."
            )


# REQUIRED: entry point for the pipeline registry auto-discovery
EntryClass = [SanaWMPipeline, SanaWMTwoStagePipeline]
