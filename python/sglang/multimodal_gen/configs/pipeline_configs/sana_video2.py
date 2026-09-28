# SPDX-License-Identifier: Apache-2.0
"""Native SANA-Video 2.0 pipeline configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.sana_video2 import SanaVideo2Config
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
    SanaVideoPipelineConfig,
)


@dataclass
class SanaVideo2VAEConfig(LTXVideoVAEConfig):
    def update_model_arch(self, source_model_dict):
        config = dict(source_model_dict)
        if "upsample_type" not in config and "decoder_upsample_type" in config:
            config["upsample_type"] = list(reversed(config["decoder_upsample_type"]))
        super().update_model_arch(config)


@dataclass
class SanaVideo2PipelineConfig(SanaVideoPipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V
    flow_shift: float | None = 12.0
    skip_input_image_preprocess: bool = True
    dit_config: SanaVideo2Config = field(default_factory=SanaVideo2Config)
    vae_config: SanaVideo2VAEConfig = field(default_factory=SanaVideo2VAEConfig)
    vae_precision: str = "bf16"
    vae_decode_precision: str = "bf16"
    prompt_instruction: str = ""

    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    def supports_sequential_multi_output_inference(self):
        return True

    def get_decode_scale_and_shift(self, device, dtype, vae):
        mean = vae.latents_mean.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        std = vae.latents_std.to(device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        return self.vae_config.arch_config.scaling_factor / std, mean


def register():
    from sglang.multimodal_gen.configs.sample.sana_video2 import (
        SanaVideo2SamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=SanaVideo2SamplingParams,
        pipeline_config_cls=SanaVideo2PipelineConfig,
        hf_model_paths=["Efficient-Large-Model/SANA-Video_2.0_5B_720p"],
        model_detectors=[
            lambda name: any(
                key in name.lower()
                for key in (
                    "sana-video_2.0",
                    "sana-video2",
                    "sana_video2",
                    "sanavideo2pipeline",
                )
            )
        ],
    )
