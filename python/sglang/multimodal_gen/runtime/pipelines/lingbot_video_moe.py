# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoDenoisingStage,
    LingBotVideoImageConditioningStage,
    LingBotVideoRefinementStage,
    LingBotVideoRefinerUpscaleStage,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.auto_negative import (
    LingBotVideoAutoNegativeStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter import (
    LingBotVideoPromptRewriteStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe.rewriter_backends import (
    build_rewriter_backend,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _flow_shift_kwarg(batch, server_args: ServerArgs) -> tuple[str, float | None]:
    shift = (
        batch.flow_shift
        if batch.flow_shift is not None
        else server_args.pipeline_config.flow_shift
    )
    return ("shift", shift)


class LingBotVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "LingBotVideoPipeline"
    is_video_pipeline = True

    _required_config_modules = [
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def _text_encoding_stage(
        self, *, encode_text_only: bool = False
    ) -> LingBotVideoTextEncodingStage:
        return LingBotVideoTextEncodingStage(
            text_encoders=[self.get_module("text_encoder")],
            tokenizers=[self.get_module("processor")],
            transformer=self.get_module("transformer"),
            encode_text_only=encode_text_only,
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        config = server_args.pipeline_config
        # One backend for both stages, so an in-process rewriter loads once.
        rewriter_backend = build_rewriter_backend(
            config, trust_remote_code=server_args.trust_remote_code
        )
        if rewriter_backend is not None:
            self.add_stage(LingBotVideoPromptRewriteStage(backend=rewriter_backend))
            if config.rewriter_auto_negative:
                self.add_stage(LingBotVideoAutoNegativeStage(backend=rewriter_backend))
        self.add_stage(
            LingBotVideoImageConditioningStage(vae=self.get_module("vae")),
        )
        self.add_stage(self._text_encoding_stage())
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_flow_shift_kwarg],
        )
        self._add_denoising_stages()
        self.add_standard_decoding_stage()

    def _add_denoising_stages(self) -> None:
        self.add_stage(
            LingBotVideoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )


class LingBotVideoRefinerPipeline(LingBotVideoPipeline):
    """Base denoise, then the refiner weights at a higher resolution.

    The refiner DiT doubles the resident weights, so the single-pass pipeline stays
    the default.
    """

    pipeline_name = "LingBotVideoRefinerPipeline"

    _required_config_modules = [
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "transformer_2",
        "scheduler",
    ]

    def _load_config(self) -> dict[str, Any]:
        # The refiner DiT is absent from model_index.json but shares the base architecture.
        config = super()._load_config()
        config["transformer_2"] = config["transformer"]
        return config

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        # The name drives loader selection, so keep it and redirect only the path.
        if (
            module_name == "transformer_2"
            and server_args.component_paths.get(module_name) is None
        ):
            return os.path.join(self.model_path, "refiner")
        return super()._resolve_component_path(
            server_args, module_name, load_module_name
        )

    def _add_denoising_stages(self) -> None:
        super()._add_denoising_stages()
        self.add_stage(
            self._text_encoding_stage(encode_text_only=True),
            "refiner_text_encoding_stage",
        )
        self.add_stage(
            LingBotVideoRefinerUpscaleStage(vae=self.get_module("vae"), pipeline=self),
            "refiner_upscale_stage",
        )
        self.add_stage(
            LingBotVideoRefinementStage(
                transformer=self.get_module("transformer_2"),
                scheduler=self.get_module("scheduler"),
            ),
            "refinement_stage",
        )


EntryClass = [LingBotVideoPipeline, LingBotVideoRefinerPipeline]
