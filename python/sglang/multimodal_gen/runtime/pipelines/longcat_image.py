"""LongCat-Image pipeline for SGLang."""

from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.longcat_image import (
    LongCatPromptRewriteStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


def _prepare_mu(batch, server_args):
    """Compute mu for FlowMatchEulerDiscreteScheduler from the packed latent token count."""
    from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
        _calculate_shift,
    )

    image_seq_len = batch.latents.shape[1]
    mu = _calculate_shift(image_seq_len)
    return "mu", mu


class LongCatImagePipeline(LoRAPipeline, ComposedPipelineBase):
    """Pipeline for LongCat-Image text-to-image generation."""

    pipeline_name = "LongCatImagePipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "text_processor",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        # 1. Prompt rewriting (optional) + request-level setup (generator, cfg renorm).
        rewrite_stage = LongCatPromptRewriteStage(
            text_encoder=self.get_module("text_encoder"),
            text_processor=self.get_module("text_processor"),
            text_encoder_dtype=PRECISION_TO_TYPE[
                server_args.pipeline_config.text_encoder_precisions[0]
            ],
        )
        self.add_stage(rewrite_stage)

        # 2. Text encoding via the standard stage (tokenize_prompt +
        #    postprocess_text_funcs hooks on the pipeline config).
        self.add_standard_text_encoding_stage()

        # 3. Latent preparation (batch-size-aware via pipeline config hooks)
        self.add_standard_latent_preparation_stage()

        # 4. Timestep preparation (mu computed from packed latent token count)
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[_prepare_mu],
        )

        # 5. Standard denoising loop. txt_ids/img_ids are built per-step inside
        #    prepare_*_cond_kwargs; the DiT computes RoPE on the fly from them
        #    (matching diffusers' transformer).
        self.add_standard_denoising_stage()

        # 6. Standard VAE decoding
        self.add_standard_decoding_stage()


EntryClass = [LongCatImagePipeline]
