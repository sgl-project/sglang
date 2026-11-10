# SPDX-License-Identifier: Apache-2.0
"""Native SANA-Video 2.0 pipeline configuration."""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.sana_video2 import SanaVideo2Config
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (
    SanaVideoPipelineConfig,
)
from sglang.multimodal_gen.configs.task_type import ModelTaskType


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

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # the denoising stage batches both guidance branches on every rank
        return ModelDeploymentConfig(
            auto_enable_cfg_parallel=False, supports_cfg_parallel=False
        )

    def validate_server_args(self, server_args):
        if int(server_args.ring_degree or 1) > 1:
            raise ValueError(
                "SANA-Video 2.0 does not support --ring-degree > 1; use Ulysses "
                "sequence parallelism."
            )
        arch = self.dit_config.arch_config
        softmax_heads = arch.hidden_size // arch.softmax_head_dim
        sp_degree = int(server_args.sp_degree or 1)
        if softmax_heads % sp_degree:
            raise ValueError(
                f"SANA-Video 2.0 spreads its {softmax_heads} softmax attention heads "
                f"over the sequence-parallel ranks; --sp-degree {sp_degree} does not "
                "divide them."
            )
        super().validate_server_args(server_args)

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
