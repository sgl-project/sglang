# SPDX-License-Identifier: Apache-2.0
"""Realtime API pipeline for the MinWM Wan2.2-5B DMD student."""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DMDTimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm import (
    MinWMCausalDMDDenoisingStage,
    MinWMCausalVaeDecodingStage,
    MinWMChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    RealtimeImageVAEEncodingStage,
    RealtimeInputValidationStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class MinWMCausalDMDPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "MinWMCausalDMDPipeline"
    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args):
        self.modules["scheduler"] = SelfForcingFlowMatchScheduler(
            num_inference_steps=1000,
            shift=server_args.pipeline_config.flow_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

    @staticmethod
    def _validate_sequence_parallelism_args(server_args: ServerArgs) -> None:
        sp_degree = getattr(server_args, "sp_degree", 1) or 1
        ulysses_degree = getattr(server_args, "ulysses_degree", 1) or 1
        ring_degree = getattr(server_args, "ring_degree", 1) or 1
        if (sp_degree, ulysses_degree, ring_degree) != (1, 1, 1):
            raise ValueError(
                "MinWM causal realtime does not support sequence parallelism yet. "
                "Its source-shaped packed attention, causal KV cache, action history, "
                "and absolute-position schedule have not been sharded with the required "
                "all-to-all/state exchange. Use one independent MinWM replica per GPU "
                "instead of "
                f"--sp-degree {sp_degree}, --ulysses-degree {ulysses_degree}, "
                f"--ring-degree {ring_degree}."
            )

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        self._validate_sequence_parallelism_args(server_args)
        self.add_stage(RealtimeInputValidationStage())
        self.add_stage(
            RealtimeTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_stage(RealtimeImageVAEEncodingStage(vae=self.get_module("vae")))
        self.add_stage(DMDTimestepPreparationStage(self.get_module("scheduler")))
        self.add_stage(MinWMChunkLatentPreparationStage(self.get_module("transformer")))
        self.add_stage(
            MinWMCausalDMDDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            )
        )
        self.add_stage(
            MinWMCausalVaeDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


EntryClass = MinWMCausalDMDPipeline
