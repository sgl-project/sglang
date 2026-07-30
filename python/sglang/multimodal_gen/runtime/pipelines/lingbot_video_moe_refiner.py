# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from sglang.multimodal_gen.runtime.pipelines.lingbot_video_moe import (
    LingBotVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoRefinementStage,
    LingBotVideoRefinerUpscaleStage,
    LingBotVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotVideoRefinerPipeline(LingBotVideoPipeline):
    """Two-pass LingBot-Video: base denoise, then the refiner weights at a higher resolution.

    Opt in with ``--pipeline-class-name LingBotVideoRefinerPipeline``; the refiner DiT
    doubles the resident weights, so the single-pass pipeline stays the default.
    """

    pipeline_name = "LingBotVideoRefinerPipeline"

    _required_config_modules = (
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "transformer_2",
        "scheduler",
    )

    _refiner_subfolder = "refiner"

    def _load_config(self) -> dict[str, Any]:
        # The refiner DiT ships as a bare subfolder, absent from model_index.json,
        # and is the same architecture as the base DiT.
        config = super()._load_config()
        config["transformer_2"] = config["transformer"]
        return config

    def _resolve_component_path(
        self, server_args: ServerArgs, module_name: str, load_module_name: str
    ) -> str:
        # Keep the transformer_2 name so the loader registry still picks the
        # transformer loader, and only redirect where the weights come from.
        if module_name == "transformer_2":
            return os.path.join(self.model_path, self._refiner_subfolder)
        return super()._resolve_component_path(
            server_args, module_name, load_module_name
        )

    def _maybe_add_refiner_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(
            LingBotVideoTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("processor")],
                transformer=self.get_module("transformer"),
                encode_text_only=True,
            ),
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


EntryClass = [LingBotVideoRefinerPipeline]
