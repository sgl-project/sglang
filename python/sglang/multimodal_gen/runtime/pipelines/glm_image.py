from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DecodingStage,
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageAR,
    GlmImageBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class GlmImageDenoiserDecodingStage(DecodingStage):
    """Reuse standard VAE decoding, but assign it to DENOISER instead of DECODER."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER


class GlmImageDenoiserPreparationStage(GlmImageBeforeDenoisingStage):
    """Reuse GLM preparation, but assign it to DENOISER instead of ENCODER."""

    @property
    def role_affinity(self) -> RoleType:
        return RoleType.DENOISER


class GlmImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "GlmImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "vision_language_encoder",
        "processor",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        is_glm_distributed_denoiser = (
            self._disagg_role == RoleType.DENOISER
            and server_args.srt_encoder_url is not None
        )
        self.add_stage(
            GlmImageAR(
                processor=self.get_module("processor"),
                vision_language_encoder=self.get_module("vision_language_encoder"),
            ),
            "glm_image_ar",
        )

        before_denoising_stage_cls = (
            GlmImageDenoiserPreparationStage
            if is_glm_distributed_denoiser
            else GlmImageBeforeDenoisingStage
        )
        self.add_stage(
            before_denoising_stage_cls(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "glm_image_before_denoising_stage",
        )

        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        if is_glm_distributed_denoiser:
            self.add_stage(
                GlmImageDenoiserDecodingStage(
                    vae=self.get_module("vae"), pipeline=self
                ),
                "decoding_stage",
            )
        else:
            self.add_standard_decoding_stage()


EntryClass = [GlmImagePipeline]
