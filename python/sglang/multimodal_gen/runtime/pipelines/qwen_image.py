# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from diffusers.image_processor import VaeImageProcessor

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image_layered import (
    QwenImageLayeredBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

# TODO(will): move PRECISION_TO_TYPE to better place

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
    image_seq_len = (int(height) // vae_scale_factor // 2) * (
        int(width) // vae_scale_factor // 2
    )
    mu = calculate_shift(
        image_seq_len,
        # hard code, since scheduler_config is not in PipelineConfig now
        256,
        8192,
        0.5,
        0.9,
    )
    return "mu", mu


class QwenImagePipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_standard_t2i_stages(prepare_extra_timestep_kwargs=[prepare_mu])


class QwenImageEditPipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImageEditPipeline"

    _required_config_modules = [
        "processor",
        "scheduler",
        "text_encoder",
        "tokenizer",
        "transformer",
        "vae",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        vae_image_processor = VaeImageProcessor(
            vae_scale_factor=server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
            * 2
        )

        self.add_standard_ti2i_stages(
            vae_image_processor=vae_image_processor,
            prompt_encoding="image_encoding",
            image_processor_key="processor",
            prompt_text_encoder_key="text_encoder",
            prepare_extra_timestep_kwargs=[prepare_mu],
        )


class QwenImageEditPlusPipeline(QwenImageEditPipeline):
    pipeline_name = "QwenImageEditPlusPipeline"


def prepare_mu_layered(batch: Req, server_args: ServerArgs):
    base_seqlen = 256 * 256 / 16 / 16
    mu = (batch.image_latent.shape[1] / base_seqlen) ** 0.5
    return "mu", mu


class QwenImageLayeredPipeline(QwenImageEditPipeline):
    pipeline_name = "QwenImageLayeredPipeline"

    _required_config_modules = [
        "vae",
        "tokenizer",
        "processor",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        self.add_stage(
            QwenImageLayeredBeforeDenoisingStage(
                vae=self.get_module("vae"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
                model_path=self.model_path,
            )
        )

        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_mu_layered]
        )
        self.add_standard_denoising_stage()
        self.add_standard_decoding_stage()


EntryClass = [
    QwenImagePipeline,
    QwenImageEditPipeline,
    QwenImageEditPlusPipeline,
    QwenImageLayeredPipeline,
]
