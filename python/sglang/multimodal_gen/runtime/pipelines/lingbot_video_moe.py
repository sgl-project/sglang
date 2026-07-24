# SPDX-License-Identifier: Apache-2.0
# Ported from the LingBot-Video reference pipeline:
#   /vllm-workspace/lingbot-video/lingbot_video/pipeline_lingbot_video.py
#
# MVP scope: single-GPU, T2V base, B=1, structured-JSON captions.

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.lingbot_video_moe import (
    LingBotVideoBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class LingBotVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """LingBot-Video MoE text-to-video pipeline (base, single-GPU MVP).

    ``pipeline_name`` matches the diffusers ``model_index.json`` ``_class_name``
    so the registry resolves this class with no model detector. The Qwen3-VL
    text front-end is exposed as the ``processor`` module (not ``tokenizer``).
    """

    pipeline_name = "LingBotVideoPipeline"
    is_video_pipeline = True

    # Note: "processor" (Qwen3-VL), not "tokenizer".
    _required_config_modules = (
        "text_encoder",
        "processor",
        "vae",
        "transformer",
        "scheduler",
    )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        # 1. Monolithic pre-processing (Qwen3-VL encode, latents, timesteps).
        self.add_stage(
            LingBotVideoBeforeDenoisingStage(
                vae=self.get_module("vae"),
                text_encoder=self.get_module("text_encoder"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 2. Standard denoising loop (two-call CFG: cond + uncond).
        self.add_stage(
            DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
        )

        # 3. Standard VAE decoding.
        self.add_standard_decoding_stage()


EntryClass = [LingBotVideoPipeline]
