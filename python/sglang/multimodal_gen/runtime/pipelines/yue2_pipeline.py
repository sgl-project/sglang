# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path

import torch

from sglang.multimodal_gen.configs.pipeline_configs.yue2 import Yue2PipelineConfig
from sglang.multimodal_gen.configs.sample.yue2 import Yue2SamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Yue2Pipeline(ComposedPipelineBase):
    """YuE2 lyrics -> ABC -> codec tokens -> NAR latents -> 48 kHz audio."""

    pipeline_name = "YuE2Pipeline"
    is_video_pipeline = False
    _required_config_modules: list[str] = []
    pipeline_config_cls = Yue2PipelineConfig
    sampling_params_cls = Yue2SamplingParams

    def load_modules(self, server_args, loaded_modules=None):
        from sglang.srt.model_loader.yue2_loader import Yue2Runtime

        vae_dir = Path(
            os.environ.get(
                "SGLANG_YUE2_VAE_DIR",
                Path(self.model_path).parent / "YuE2-Vae",
            )
        )
        device = torch.device("cuda", torch.cuda.current_device())
        runtime = Yue2Runtime.from_paths(
            model_dir=self.model_path,
            vae_dir=vae_dir,
            device=device,
        )
        return {
            "mot": runtime.model,
            "vae": runtime.vae,
            "tokenizer": runtime.tokenizer,
            "runtime": runtime,
        }

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.yue2 import (
            Yue2ARStage,
            Yue2ArtifactExportStage,
            Yue2NARStage,
            Yue2PrepareRequestStage,
            Yue2VAEDecodeStage,
        )

        self.add_stage(Yue2PrepareRequestStage(self.get_module("tokenizer")))
        self.add_stage(
            Yue2ARStage(self.get_module("mot"), self.get_module("tokenizer"))
        )
        self.add_stage(Yue2NARStage(self.get_module("mot")))
        self.add_stage(Yue2ArtifactExportStage())
        self.add_stage(Yue2VAEDecodeStage(self.get_module("vae")))


EntryClass = [Yue2Pipeline]
