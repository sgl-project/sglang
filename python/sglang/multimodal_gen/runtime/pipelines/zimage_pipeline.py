# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.pipelines.zimage_progressive import (
    ZImageProgressiveDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline, Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    DenoisingStage,
    InputValidationStage,
    PipelineStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


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
    vae_scale_factor = server_args.pipeline_config.vae_config.vae_scale_factor
    image_seq_len = ((int(height) // vae_scale_factor) // 2) * (
        (int(width) // vae_scale_factor) // 2
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


class ZImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "ZImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(InputValidationStage())
        self.add_standard_text_encoding_stage()
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(prepare_extra_kwargs=[prepare_mu])
        self._add_zimage_denoising_stage()
        self.add_standard_decoding_stage()

    def _add_zimage_denoising_stage(self, stage_name: str = "denoising_stage") -> None:
        def create_stage():
            kwargs = dict(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                pipeline=self,
                vae=self.get_module("vae", None),
            )
            standard = DenoisingStage(**kwargs)
            progressive = ZImageProgressiveDenoisingStage(**kwargs)

            class _Dispatch(PipelineStage):
                def forward(self, batch: Req, server_args: ServerArgs) -> Req:
                    mode = getattr(batch, "progressive_mode", "fullres") or "fullres"
                    if mode in ("dct", "dct_rewind"):
                        return progressive.forward(batch, server_args)
                    return standard.forward(batch, server_args)

            return _Dispatch()

        self.add_stage_factory(RoleType.DENOISER, create_stage, stage_name)


EntryClass = ZImagePipeline
