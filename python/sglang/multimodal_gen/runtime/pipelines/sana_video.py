# SPDX-License-Identifier: Apache-2.0
"""SANA-Video text-to-video pipeline."""

from typing import ClassVar

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class SanaVideoPipeline(LoRAPipeline, ComposedPipelineBase):
    """Standard staged pipeline for the official SANA-Video checkpoint."""

    pipeline_name = "SanaVideoPipeline"
    is_video_pipeline = True

    _required_config_modules: ClassVar[list[str]] = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_standard_t2i_stages()


EntryClass = SanaVideoPipeline
