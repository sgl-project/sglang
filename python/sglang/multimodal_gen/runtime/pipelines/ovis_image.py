# SPDX-License-Identifier: Apache-2.0
from typing import ClassVar

from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.diffusion_scheduler_utils import (
    calculate_linear_shift,
)


def prepare_mu(batch, server_args):
    factor = server_args.pipeline_config.vae_config.arch_config.vae_scale_factor * 2
    image_seq_len = (batch.height // factor) * (batch.width // factor)
    config = batch.scheduler.config
    return "mu", calculate_linear_shift(
        image_seq_len,
        base_seq_len=config.get("base_image_seq_len", 256),
        max_seq_len=config.get("max_image_seq_len", 4096),
        base_shift=config.get("base_shift", 0.5),
        max_shift=config.get("max_shift", 1.15),
    )


class OvisImagePipeline(ComposedPipelineBase):
    pipeline_name = "OvisImagePipeline"
    _required_config_modules: ClassVar[list[str]] = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args):
        self.add_standard_t2i_stages(prepare_extra_timestep_kwargs=[prepare_mu])


EntryClass = OvisImagePipeline
