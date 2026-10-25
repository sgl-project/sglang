# SPDX-License-Identifier: Apache-2.0
"""Nomas T2V pipeline: offline requests and realtime sessions over the standard stages."""

import math

from sglang.multimodal_gen.configs.pipeline_configs.nomas import NomasPipelineConfig
from sglang.multimodal_gen.configs.sample.nomas import NomasSamplingParams
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import (
    NomasDecodingStage,
    NomasDenoisingStage,
    NomasInputValidationStage,
    NomasLatentPreparationStage,
    NomasTextEncodingStage,
)


class NomasPipeline(ComposedPipelineBase):
    pipeline_name = "NomasPipeline"
    pipeline_config_cls = NomasPipelineConfig
    sampling_params_cls = NomasSamplingParams
    _required_config_modules = [
        "tokenizer",
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
    ]

    def initialize_pipeline(self, server_args):
        config = server_args.pipeline_config
        if (config.dit_config.in_channels, config.dit_config.out_channels) != (32, 32):
            raise ValueError("The Nomas pipeline requires f32t4c32 DiT latents")
        text_width = self.get_module("text_encoder").config.text_config.hidden_size
        if text_width != config.dit_config.text_states_dim:
            raise ValueError(
                "Nomas DiT text_states_dim must match the text encoder width"
            )
        scheduler = self.get_module("scheduler")
        if not math.isclose(
            scheduler.shift, config.flow_shift, rel_tol=0, abs_tol=1e-8
        ):
            raise ValueError(
                "Nomas scheduler effective shift must equal pipeline flow_shift"
            )
        if (
            scheduler.config.use_dynamic_shifting
            or scheduler.config.invert_sigmas
            or scheduler.config.shift_terminal is not None
            or scheduler.config.use_karras_sigmas
            or scheduler.config.use_beta_sigmas
            or scheduler.config.use_exponential_sigmas
            or scheduler.config.use_uniform_sigmas
        ):
            raise ValueError("Nomas requires static, non-inverted flow sigmas")
        if scheduler.config.num_train_timesteps != 1000:
            raise ValueError("Nomas DiT requires the 1000-step time convention")

    def create_pipeline_stages(self, server_args):
        self.add_stage(NomasInputValidationStage())
        self.add_stage(
            NomasTextEncodingStage(
                text_encoders=[self.get_module("text_encoder")],
                tokenizers=[self.get_module("tokenizer")],
            )
        )
        self.add_standard_timestep_preparation_stage()
        self.add_stage(
            NomasLatentPreparationStage(
                scheduler=self.get_module("scheduler"),
                transformer=self.get_module("transformer"),
            )
        )
        self.add_stage(
            NomasDenoisingStage(self.get_module("transformer")), "denoising_stage"
        )
        self.add_stage(
            NomasDecodingStage(vae=self.get_module("vae"), pipeline=self),
            "decoding_stage",
        )


EntryClass = NomasPipeline
