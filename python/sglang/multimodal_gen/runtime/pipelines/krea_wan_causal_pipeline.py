# SPDX-License-Identifier: Apache-2.0
"""
Krea Wan Causal pipeline implementation.

This module wires the causal DMD denoising stage into the modular pipeline.
"""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    FlowUniPCMultistepScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.krea_realtime_video import (
    KreaRealtimeVideoBeforeDenoisingStage,
    KreaRealtimeVideoDenoisingStage,
    KreaRealtimeVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

# isort: on

logger = init_logger(__name__)


class KreaWanCausalPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "KreaWanCausalPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        vae_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
        self.add_stage(InputValidationStage())
        self.add_stage(
            KreaRealtimeVideoTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            ),
        )
        self.add_stage(
            KreaRealtimeVideoBeforeDenoisingStage(
                tokenizer=self.get_module("tokenizer"),
                transformer=self.get_module("transformer"),
                vae=self.get_module("vae"),
                vae_dtype=vae_dtype,
            ),
        )
        self.add_stage(
            KreaRealtimeVideoDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                vae=self.get_module("vae"),
                vae_dtype=vae_dtype,
            ),
        )

    def initialize_pipeline(self, server_args: ServerArgs):
        # We use UniPCMScheduler from Wan2.1 official repo, not the one in diffusers.
        self.modules["scheduler"] = FlowUniPCMultistepScheduler(
            shift=server_args.pipeline_config.flow_shift
        )


EntryClass = KreaWanCausalPipeline
