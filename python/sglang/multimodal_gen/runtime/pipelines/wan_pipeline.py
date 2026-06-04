# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""
Wan video diffusion pipeline implementation.

This module contains an implementation of the Wan video diffusion pipeline
using the modular pipeline architecture.
"""

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines.wan_progressive import (
    WanProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
    PipelineStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class WanPipeline(LoRAPipeline, ComposedPipelineBase):
    """
    Wan video diffusion pipeline with LoRA support.
    """

    pipeline_name = "WanPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args: ServerArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage()
        self._add_wan_denoising_stage()
        self.add_standard_decoding_stage()

    def _add_wan_denoising_stage(self, stage_name: str = "denoising_stage") -> None:
        def create_stage():
            kwargs = dict(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                vae=self.get_module("vae", None),
            )
            standard = DenoisingStage(**kwargs)
            progressive = WanProgressiveDenoisingStage(**kwargs)

            class _Dispatch(PipelineStage):
                def forward(self, batch: Req, server_args: ServerArgs) -> Req:
                    mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
                    if mode in ("dct", "dct_rewind"):
                        return progressive.forward(batch, server_args)
                    return standard.forward(batch, server_args)

            return _Dispatch()

        self.add_stage_factory(RoleType.DENOISER, create_stage, stage_name)


EntryClass = WanPipeline
