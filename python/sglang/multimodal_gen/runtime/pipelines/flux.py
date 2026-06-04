# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.flux_progressive import (
    FluxProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
    PipelineStage,
    TextEncodingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# TODO(will): move PRECISION_TO_TYPE to better place

logger = init_logger(__name__)


_PROGRESSIVE_MODES = frozenset({"dct", "dct_rewind"})


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    height = batch.height
    width = batch.width
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    image_seq_len = (int(height) // (vae_scale_factor * 2)) * (
        int(width) // (vae_scale_factor * 2)
    )

    mu = calculate_shift(
        image_seq_len,
        # hard code, since scheduler_config is not in PipelineConfig now
        256,
        4096,
        0.5,
        1.15,
    )
    return "mu", mu


class _FluxDenoisingStageRouter(PipelineStage):
    def __init__(
        self,
        standard_stage: DenoisingStage,
        progressive_stage: FluxProgressiveDenoisingStage,
    ) -> None:
        super().__init__()
        self.standard_stage = standard_stage
        self.progressive_stage = progressive_stage

    @property
    def role_affinity(self):
        return RoleType.DENOISER

    @property
    def parallelism_type(self):
        return self.standard_stage.parallelism_type

    def set_component_residency_manager(self, manager) -> None:
        super().set_component_residency_manager(manager)
        self.standard_stage.set_component_residency_manager(manager)
        self.progressive_stage.set_component_residency_manager(manager)

    def set_registered_stage_name(self, stage_name: str) -> None:
        super().set_registered_stage_name(stage_name)
        self.standard_stage.set_registered_stage_name(stage_name)
        self.progressive_stage.set_registered_stage_name(stage_name)

    def set_profile_stage_name(self, stage_name: str) -> None:
        super().set_profile_stage_name(stage_name)
        self.standard_stage.set_profile_stage_name(stage_name)
        self.progressive_stage.set_profile_stage_name(stage_name)

    def component_uses(self, server_args: ServerArgs, stage_name: str | None = None):
        return self.standard_stage.component_uses(server_args, stage_name)

    def forward(self, batch: Req, server_args: ServerArgs) -> Req:
        mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
        if mode in _PROGRESSIVE_MODES:
            return self.progressive_stage.forward(batch, server_args)
        return self.standard_stage.forward(batch, server_args)


class FluxPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "FluxPipeline"

    _required_config_modules = [
        "text_encoder",
        "text_encoder_2",
        "tokenizer",
        "tokenizer_2",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())

        self.add_stage(
            TextEncodingStage(
                text_encoders=[
                    self.get_module("text_encoder"),
                    self.get_module("text_encoder_2"),
                ],
                tokenizers=[
                    self.get_module("tokenizer"),
                    self.get_module("tokenizer_2"),
                ],
            ),
            "prompt_encoding_stage_primary",
        )

        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_mu])
        self.add_standard_latent_preparation_stage()
        self._add_flux_denoising_router()
        self.add_standard_decoding_stage()

    def _add_flux_denoising_router(self, stage_name: str = "denoising_stage") -> None:
        def create_stage():
            kwargs = dict(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                vae=self.get_module("vae", None),
            )
            return _FluxDenoisingStageRouter(
                standard_stage=DenoisingStage(**kwargs),
                progressive_stage=FluxProgressiveDenoisingStage(**kwargs),
            )

        self.add_stage_factory(RoleType.DENOISER, create_stage, stage_name)


EntryClass = FluxPipeline
