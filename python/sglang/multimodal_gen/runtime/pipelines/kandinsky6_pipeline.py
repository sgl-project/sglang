# SPDX-License-Identifier: Apache-2.0
"""Kandinsky6 TI2VA: text (optionally plus a conditioning image) to
synchronized video + audio.

There is deliberately no separate T2VA/IT2VA pipeline class -- one stage
chain handles both: ``Kandinsky6ImageEncodingStage`` is a no-op when the
request carries no conditioning image and injects it as an extra reference
frame when one is supplied, mirroring how the diffusers reference's
``Kandinsky6I2VAPipeline`` is itself just ``Kandinsky6T2VAPipeline`` plus an
optional image branch on the same call path, not a materially different
pipeline.

This model jointly denoises two coupled modalities with a non-standard
denoise loop, so -- like MOVA (this codebase's closest precedent for a
dual-modality video+audio diffusion pipeline) and unlike a standard
single-tensor pipeline -- it owns a model-specific stage chain
(``runtime/pipelines_core/stages/model_specific_stages/kandinsky6/``) rather
than using the framework's composite ``add_standard_*`` stage builders.
"""

from __future__ import annotations

from sglang.multimodal_gen.configs.pipeline_configs.kandinsky6 import (
    Kandinsky6TI2VAPipelineConfig,
)
from sglang.multimodal_gen.configs.sample.kandinsky6 import (
    Kandinsky6TI2VASamplingParams,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6 import (
    Kandinsky6AudioDecodingStage,
    Kandinsky6DecodingStage,
    Kandinsky6DenoisingStage,
    Kandinsky6ImageEncodingStage,
    Kandinsky6LatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.text_encoding import (
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.timestep_preparation import (
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class Kandinsky6TI2VAPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "Kandinsky6TI2VAPipeline"
    is_video_pipeline = True
    pipeline_config_cls = Kandinsky6TI2VAPipelineConfig
    sampling_params_cls = Kandinsky6TI2VASamplingParams

    # Kandinsky6AudioVAE bundles the mel-VAE decoder *and* the BigVGAN-v2
    # vocoder in one checkpoint component -- there is no separate "vocoder"
    # module, unlike LTX-2's separately-loaded audio_vae/vocoder pair.
    _required_config_modules = [
        "scheduler",
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "transformer",
        "vae",
        "audio_vae",
    ]

    def validate_disagg_role(self, role: RoleType) -> None:
        # Simplest v1: the coupled video+audio denoising loop and the
        # tail-cond re-pinning state aren't (yet) split across disaggregated
        # roles. Matches MiniMaxH3Pipeline's precedent for a dual-modality
        # pipeline that hasn't been made disagg-aware.
        if role != RoleType.MONOLITHIC:
            raise ValueError(
                "Kandinsky6TI2VAPipeline only supports monolithic deployment; "
                f"disaggregation role {role.value!r} is not supported"
            )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            stage_name="input_validation_stage", stage=InputValidationStage()
        )

        self.add_stage(
            stage_name="text_encoding_stage",
            stage=TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
        )

        self.add_stage(
            stage_name="timestep_preparation_stage",
            stage=TimestepPreparationStage(scheduler=self.get_module("scheduler")),
        )

        self.add_stage(
            stage_name="latent_preparation_stage",
            stage=Kandinsky6LatentPreparationStage(),
        )

        # No-op unless the request supplies a conditioning image; see
        # Kandinsky6ImageEncodingStage's docstring.
        self.add_stage(
            stage_name="image_encoding_stage",
            stage=Kandinsky6ImageEncodingStage(vae=self.get_module("vae")),
        )

        self.add_stage(
            stage_name="denoising_stage",
            stage=Kandinsky6DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
            ),
        )

        # Audio decodes before video, matching the diffusers reference and
        # FastVideo's own stage order -- do not reorder.
        self.add_stage(
            stage_name="audio_decoding_stage",
            stage=Kandinsky6AudioDecodingStage(audio_vae=self.get_module("audio_vae")),
        )

        self.add_stage(
            stage_name="decoding_stage",
            stage=Kandinsky6DecodingStage(vae=self.get_module("vae"), pipeline=self),
        )


EntryClass = [Kandinsky6TI2VAPipeline]
