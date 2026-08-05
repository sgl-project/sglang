# SPDX-License-Identifier: Apache-2.0
"""Realtime API pipeline for the MinWM Wan2.2-5B DMD student."""

from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_unipc_multistep import (
    MinWMFlowUniPCParityScheduler,
)
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_self_forcing_flow_match import (
    SelfForcingFlowMatchScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.lora_pipeline import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DMDTimestepPreparationStage,
    TimestepPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minwm import (
    MinWMCausalDMDDenoisingStage,
    MinWMCausalUniPCDenoisingStage,
    MinWMCausalVaeDecodingStage,
    MinWMChunkLatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime import (
    RealtimeImageVAEEncodingStage,
    RealtimeInputValidationStage,
    RealtimeLatentHandoffStage,
    RealtimeTextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _use_remote_realtime_vae(server_args: ServerArgs) -> bool:
    if bool(getattr(server_args, "realtime_remote_vae_enabled", False)):
        return True
    value = getattr(server_args, "realtime_vae_worker_url", None)
    return isinstance(value, str) and bool(value.strip())


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
        if (sp_degree, ulysses_degree, ring_degree) == (1, 1, 1):
            return
        if ring_degree != 1:
            raise ValueError(
                "MinWM causal realtime supports Ulysses sequence parallelism "
                "with --ring-degree 1 only."
            )
        if sp_degree <= 1 or sp_degree != ulysses_degree:
            raise ValueError(
                "MinWM causal realtime requires --sp-degree == --ulysses-degree > 1."
            )
        num_attention_heads = 24
        pipeline_config = getattr(server_args, "pipeline_config", None)
        dit_config = getattr(pipeline_config, "dit_config", None)
        arch_config = getattr(dit_config, "arch_config", None)
        if arch_config is not None:
            num_attention_heads = int(
                getattr(arch_config, "num_attention_heads", num_attention_heads)
            )
        if num_attention_heads % ulysses_degree != 0:
            raise ValueError(
                f"MinWM attention heads ({num_attention_heads}) must be "
                f"divisible by ulysses_degree ({ulysses_degree})."
            )
        if (getattr(server_args, "tp_size", 1) or 1) != 1:
            raise ValueError(
                "MinWM causal Ulysses cannot be combined with tensor parallelism yet."
            )
        if bool(getattr(server_args, "use_fsdp_inference", False)):
            raise ValueError(
                "MinWM causal Ulysses cannot be combined with FSDP inference yet."
            )
        if bool(getattr(server_args, "enable_torch_compile", False)):
            raise ValueError(
                "MinWM causal Ulysses cannot be combined with whole-DiT "
                "torch.compile yet."
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
        self._add_realtime_output_stage(server_args)

    def _add_realtime_output_stage(self, server_args: ServerArgs) -> None:
        if _use_remote_realtime_vae(server_args):
            self.add_stage(RealtimeLatentHandoffStage())
            return
        self.add_stage(
            MinWMCausalVaeDecodingStage(
                vae=self.get_module("vae"),
                pipeline=self,
            )
        )


def _minwm_unipc_shift(_batch, server_args: ServerArgs):
    return "shift", server_args.pipeline_config.flow_shift


class MinWMCausalUniPCPipeline(MinWMCausalDMDPipeline):
    """Realtime MinWM pipeline matching V3 ``sample_solver: unipc``."""

    pipeline_name = "MinWMCausalUniPCPipeline"

    def initialize_pipeline(self, server_args: ServerArgs) -> None:
        # Native V3 constructs the scheduler at shift=1, then supplies the
        # configured shift to set_timesteps. Applying it in both places changes
        # all four timesteps.
        self.modules["scheduler"] = MinWMFlowUniPCParityScheduler(
            num_train_timesteps=1000,
            shift=1.0,
            use_dynamic_shifting=False,
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
        self.add_stage(
            TimestepPreparationStage(
                self.get_module("scheduler"),
                prepare_extra_set_timesteps_kwargs=[_minwm_unipc_shift],
            )
        )
        self.add_stage(MinWMChunkLatentPreparationStage(self.get_module("transformer")))
        self.add_stage(
            MinWMCausalUniPCDenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            )
        )
        self._add_realtime_output_stage(server_args)


EntryClass = [MinWMCausalDMDPipeline, MinWMCausalUniPCPipeline]
