# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.moss_image import MOSSImageDiTConfig
from sglang.multimodal_gen.configs.models.encoders.qwen3_5 import Qwen3_5EncoderConfig
from sglang.multimodal_gen.configs.models.vaes.flux import Flux2VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    SpatialImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    _unpatchify_latents,
    flux2_vae_bn_stats,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.configs.sample.moss_image import resolve_uncond_skip_layers
from sglang.multimodal_gen.configs.task_type import ModelTaskType
from sglang.multimodal_gen.runtime.distributed.cfg_policy import CFGPolicy


@dataclass
class MOSSImagePipelineConfig(SpatialImagePipelineConfig):
    task_type: ModelTaskType = ModelTaskType.T2I
    dit_config: MOSSImageDiTConfig = field(default_factory=MOSSImageDiTConfig)
    vae_config: Flux2VAEConfig = field(default_factory=Flux2VAEConfig)
    dit_precision: str = "bf16"
    vae_precision: str = "fp32"
    text_encoder_precisions: tuple[str, ...] = ("bf16",)
    text_encoder_configs: tuple[Qwen3_5EncoderConfig, ...] = field(
        default_factory=lambda: (Qwen3_5EncoderConfig(),)
    )
    enable_autocast: bool = False
    vae_tiling: bool = False
    vae_sp: bool = False
    flow_shift: float = 5.0
    should_use_guidance: bool = False
    supports_auto_residency: bool = False
    cfg_policy: CFGPolicy = field(
        default_factory=lambda: CFGPolicy(parallel_uses_serial_arithmetic=True)
    )

    def get_model_deployment_config(self):
        return ModelDeploymentConfig(
            keep_resident_min_available_gb=40,
            keep_resident_components=("dit", "text_encoder", "vae"),
            auto_enable_cfg_parallel=False,
            supports_cfg_parallel=True,
        )

    def validate_server_args(self, server_args):
        if server_args.tp_size != 1:
            raise ValueError("MOSSImage supports sequence and CFG parallelism, not TP.")
        if server_args.enable_torch_compile:
            raise ValueError("MOSSImage torch.compile support has not been validated.")
        if not server_args.disable_autocast:
            raise ValueError(
                "MOSSImage requires disabled autocast to preserve reference precision."
            )

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        return {"skip_layers": resolve_uncond_skip_layers(batch.sampling_params)}

    def post_denoising_loop(self, latents, batch):
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        # BN normalization applies to packed channels, before 2x2 unpacking.
        return 1.0, None

    def preprocess_decoding(self, latents, server_args=None, vae=None):
        latents = latents.to(vae.bn.running_mean)
        mean, std = flux2_vae_bn_stats(vae, latents)
        return _unpatchify_latents(latents * std + mean)


def register():
    from sglang.multimodal_gen.configs.sample.moss_image import MOSSImageSamplingParams
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=MOSSImageSamplingParams,
        pipeline_config_cls=MOSSImagePipelineConfig,
        hf_model_paths=[
            "OpenMOSS-Team/MOSS-Image-Base",
            "OpenMOSS-Team/MOSS-Image-SFT",
        ],
        model_detectors=[
            lambda path: "mossimage" in path.lower().replace("-", "").replace("_", "")
        ],
    )
