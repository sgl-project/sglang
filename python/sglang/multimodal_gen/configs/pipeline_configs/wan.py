# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits import WanAnimate2Config, WanVideoConfig
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    CLIPVisionConfig,
    T5Config,
)
from sglang.multimodal_gen.configs.models.vaes import WanVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.condition_expansion import (
    PromptToSampleBatchExpander,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def t5_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    mask: torch.Tensor = outputs.attention_mask
    hidden_state: torch.Tensor = outputs.last_hidden_state
    seq_lens = mask.gt(0).sum(dim=1).long()
    assert torch.isnan(hidden_state).sum() == 0
    prompt_embeds = [u[:v] for u, v in zip(hidden_state, seq_lens, strict=True)]
    prompt_embeds_tensor: torch.Tensor = torch.stack(
        [
            torch.cat([u, u.new_zeros(512 - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
        dim=0,
    )
    return prompt_embeds_tensor


@dataclass
class WanI2VCommonConfig(PipelineConfig):
    # for all wan i2v pipelines
    def adjust_num_frames(self, num_frames, *, log_adjustment: bool = True):
        vae_scale_factor_temporal = self.vae_config.arch_config.scale_factor_temporal
        if num_frames % vae_scale_factor_temporal != 1:
            if log_adjustment:
                logger.warning(
                    f"`num_frames - 1` has to be divisible by {vae_scale_factor_temporal}. Rounding to the nearest number."
                )
            num_frames = (
                num_frames // vae_scale_factor_temporal * vae_scale_factor_temporal + 1
            )
            return num_frames
        return num_frames


@dataclass
class WanT2V480PConfig(PipelineConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    task_type: ModelTaskType = ModelTaskType.T2V
    # WanConfig-specific parameters with defaults
    # DiT
    dit_config: DiTConfig = field(default_factory=WanVideoConfig)

    # VAE
    vae_config: VAEConfig = field(default_factory=WanVAEConfig)
    vae_tiling: bool = False
    vae_sp: bool = False

    # Denoising stage
    flow_shift: float | None = 3.0

    # Text encoding stage
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (T5Config(),)
    )
    postprocess_text_funcs: tuple[Callable[[BaseEncoderOutput], torch.Tensor], ...] = (
        field(default_factory=lambda: (t5_postprocess_text,))
    )

    # Precision for each component
    precision: str = "bf16"
    vae_precision: str = "fp32"
    vae_decode_precision: str = "bf16"
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("fp32",))

    def __post_init__(self):
        self.vae_config.load_encoder = False
        self.vae_config.load_decoder = True

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("memory",),
            keep_resident_min_available_gb=60,
            keep_resident_components=("dit",),
        )

    def expand_conditioning_to_sample_batch(self, batch):
        expander = PromptToSampleBatchExpander.from_batch(batch)
        if expander is None:
            return batch

        for field_name in (
            "prompt_embeds",
            "negative_prompt_embeds",
            "image_embeds",
            "image_latent",
        ):
            expander.expand_field(batch, field_name)
        return batch

    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]


@dataclass
class TurboWanT2V480PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 1.3B pipeline architecture."""

    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [988, 932, 852, 608]
    )


@dataclass
class TurboWanT2V1_3B480PConfig(TurboWanT2V480PConfig):
    """Configuration for TurboWan T2V 1.3B DMD pipeline."""

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("memory",),
            keep_resident_min_available_gb=60,
            keep_resident_components=(
                "dit",
                "text_encoder",
                "image_encoder",
                "vae",
            ),
        )


@dataclass
class WanT2V720PConfig(WanT2V480PConfig):
    """Base configuration for Wan T2V 14B 720P pipeline architecture."""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class WanI2V480PConfig(WanT2V480PConfig, WanI2VCommonConfig):
    """Base configuration for Wan I2V 14B 480P pipeline architecture."""

    max_area: int = 480 * 832
    # WanConfig-specific parameters with defaults
    task_type: ModelTaskType = ModelTaskType.I2V
    # Precision for each component
    image_encoder_config: EncoderConfig = field(default_factory=CLIPVisionConfig)
    image_encoder_precision: str = "fp32"

    image_encoder_extra_args: dict = field(
        default_factory=lambda: dict(
            output_hidden_states=True,
        )
    )

    def postprocess_image(self, image):
        return image.hidden_states[-2]

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True


@dataclass
class WanI2V720PConfig(WanI2V480PConfig):
    """Base configuration for Wan I2V 14B 720P pipeline architecture."""

    max_area: int = 720 * 1280
    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 5.0


@dataclass
class TurboWanI2V720Config(WanI2V720PConfig):
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [996, 932, 852, 608]
    )
    boundary_ratio: float | None = 0.9

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio


@dataclass
class FastWan2_1_T2V_480P_Config(WanT2V480PConfig):
    """Base configuration for FastWan T2V 1.3B 480P pipeline architecture with DMD"""

    # WanConfig-specific parameters with defaults

    # Denoising stage
    flow_shift: float | None = 8.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("memory",),
            keep_resident_min_available_gb=60,
            keep_resident_components=(
                "dit",
                "text_encoder",
                "image_encoder",
                "vae",
            ),
        )


@dataclass
class Wan2_2_TI2V_5B_Config(WanT2V480PConfig, WanI2VCommonConfig):
    flow_shift: float | None = 5.0
    task_type: ModelTaskType = ModelTaskType.TI2V
    expand_timesteps: bool = True
    vae_decode_precision: str = "fp32"
    # ti2v, 5B
    vae_stride = (4, 16, 16)

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        F = num_frames
        z_dim = self.vae_config.arch_config.z_dim
        vae_stride = self.vae_stride
        oh = batch.height
        ow = batch.width
        shape = (batch_size, z_dim, F, oh // vae_stride[1], ow // vae_stride[2])
        return shape

    def __post_init__(self) -> None:
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.dit_config.expand_timesteps = self.expand_timesteps


@dataclass
class FastWan2_2_TI2V_5B_Config(Wan2_2_TI2V_5B_Config):
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 757, 522]
    )


@dataclass
class Wan2_2_T2V_A14B_Config(WanT2V480PConfig):
    flow_shift: float | None = 12.0
    boundary_ratio: float | None = 0.875
    vae_decode_precision: str = "fp32"

    def __post_init__(self) -> None:
        self.dit_config.boundary_ratio = self.boundary_ratio
        self.dit_config.torch_compile_mode = "default"

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("auto", "memory"),
            auto_dit_offload_prefetch_size=2,
        )


@dataclass
class Wan2_2_I2V_A14B_Config(WanI2V720PConfig):
    flow_shift: float | None = 5.0
    boundary_ratio: float | None = 0.900
    vae_decode_precision: str = "fp32"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.dit_config.boundary_ratio = self.boundary_ratio

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("auto", "memory"),
            auto_dit_offload_prefetch_size=2,
        )


@dataclass
class Wan_Animate_2_14B_Config(WanI2V720PConfig):
    """Wan-Animate-2 14B (Wan2.2): reference image + reference video -> video.

    Shares the Wan I2V component bundle (T5, CLIP vision, Wan VAE with the encoder
    loaded) and swaps in the in-context ``WanAnimate2Config`` DiT; a single expert, so no
    ``boundary_ratio`` switching. The bespoke stages read ``flow_shift`` from here
    (carrying the audio track is a per-request sampling parameter); everything else below
    restates the Wan2.2 14B defaults instead of inheriting the Wan2.1 ones from the base.
    """

    # reference image + reference video -> video is I2V-shaped.
    task_type: ModelTaskType = ModelTaskType.I2V

    # In-context animation DiT (in_channels=36 = 16 noise + 20 conditioning).
    dit_config: DiTConfig = field(default_factory=WanAnimate2Config)

    # Denoising stage. Used when building the sigma grid.
    flow_shift: float | None = 5.0

    # The component manager casts the VAE to this dtype at its first use (the reference
    # encode in the before-denoising stage), so it is the precision the whole bespoke
    # VAE path runs at; fp32 like the other Wan2.2 configs changes the output.
    vae_decode_precision: str = "bf16"

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        # The Wan2.2 A14B modes (auto mode may layerwise-offload the 14B DiT) with the
        # resident threshold the base had: at or above 60 GB free the DiT stays resident.
        return ModelDeploymentConfig(
            dit_layerwise_offload_modes=("auto", "memory"),
            auto_dit_offload_prefetch_size=2,
            keep_resident_min_available_gb=60,
            keep_resident_components=("dit",),
        )

    def get_decode_scale_and_shift(
        self,
        device: torch.device | None,
        dtype: torch.dtype | None,
        vae: torch.nn.Module | None,
    ) -> tuple[float, None]:
        """``(1.0, None)``: ``WanAnimate2VaeAdapter.decode`` applies the latent scaling itself,
        so the DecodingStage must not de-normalize again."""
        return 1.0, None

    def supports_disaggregation(self) -> bool:
        # The stages share one WanAnimate2RequestState object (GPU tensors, a torch.Generator,
        # the clip schedule) through batch.extra. The disaggregation transport forwards only
        # JSON scalars and tensor pytrees (dict/list/tuple) from batch.extra and never the
        # generator (disaggregation/extra_tensors.py, scheduler_mixin._EXCLUDE_FIELDS), so a
        # split-role deployment would silently drop the state and fail at the first request.
        return False

    def validate_server_args(self, server_args: Any) -> None:
        if server_args.use_fsdp_inference:
            raise NotImplementedError(
                "Wan-Animate-2 does not support --use-fsdp-inference; use "
                "--dit-layerwise-offload or --tp-size instead."
            )
        # Ring splits the K/V across ranks; the in-context block mask indexes the full
        # [generation | reference-video] K/V on one rank.
        if server_args.ring_degree > 1:
            raise NotImplementedError(
                "Wan-Animate-2 does not support ring sequence parallelism "
                f"(--ring-degree {server_args.ring_degree}); use --ulysses-degree N "
                "with --ring-degree 1."
            )
        num_heads = self.dit_config.num_attention_heads
        if num_heads % server_args.tp_size != 0:
            raise ValueError(
                f"Wan-Animate-2 tensor parallelism needs the {num_heads} attention "
                f"heads divisible by --tp-size {server_args.tp_size}."
            )
        local_num_heads = num_heads // server_args.tp_size
        if local_num_heads % server_args.ulysses_degree != 0:
            raise ValueError(
                f"Wan-Animate-2 Ulysses SP needs the TP-local head count ({local_num_heads}) "
                f"divisible by --ulysses-degree {server_args.ulysses_degree}."
            )
        # The in-context self-attention runs on torch flex_attention whatever backend is
        # selected; only the cross-attention and the reference pass would follow the flag.
        requested_dit_backend = server_args.requested_component_attention_backend(
            "transformer"
        )
        if (
            server_args.is_arg_explicitly_set("attention_backend")
            and server_args.attention_backend is not None
        ) or requested_dit_backend is not None:
            raise ValueError(
                "Wan-Animate-2 does not support selecting the DiT attention backend "
                f"(--attention-backend {server_args.attention_backend} / "
                f"--component-attention-backends transformer={requested_dit_backend}): "
                "its in-context self-attention runs on torch flex_attention only. Drop "
                "the flag; the text encoder still takes "
                "--component-attention-backends text_encoder=<backend>."
            )
        # The parent block applies the offline Q/K rotation inside its own attention
        # path; forward_ref / forward_gen never do, so a rotated checkpoint would load
        # and produce wrong output without a message.
        if envs.SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION:
            raise ValueError(
                "Wan-Animate-2 does not support SGLANG_DIFFUSION_ENABLE_MXFP8_ATTENTION "
                "(offline Q/K rotation); unset it for this model."
            )
        # Upstream issue in the Wan VAE's spatially parallel encoder (wanvae.py,
        # WanEncoder3d): it splits its input over the SP group but its distributed blocks
        # gather over the VAE-decode group; the two differ once TP > 1, so every encoded
        # latent is corrupted. Remove this check once the encoder fix lands upstream.
        if server_args.tp_size > 1 and server_args.sp_degree > 1:
            raise ValueError(
                "Wan-Animate-2 does not support tensor parallelism combined with "
                f"sequence parallelism (--tp-size {server_args.tp_size} with "
                f"--ulysses-degree {server_args.ulysses_degree} / --ring-degree "
                f"{server_args.ring_degree}; sp_degree={server_args.sp_degree}, "
                "auto-derived from --num-gpus when no SP flag is given) because of an "
                "upstream issue in the Wan VAE's spatially parallel encoder "
                "(WanEncoder3d splits over the sequence-parallel group but gathers over "
                "the VAE-decode group). Use --tp-size N alone, --ulysses-degree N alone, "
                "or --enable-cfg-parallel with --ulysses-degree 2 or --tp-size 2."
            )
        super().validate_server_args(server_args)


# =============================================
# ============= Causal Self-Forcing =============
# =============================================
@dataclass
class SelfForcingWanT2V480PConfig(WanT2V480PConfig):
    is_causal: bool = True
    flow_shift: float | None = 5.0
    dmd_denoising_steps: list[int] | None = field(
        default_factory=lambda: [1000, 750, 500, 250]
    )
    warp_denoising_step: bool = True


def register():
    from sglang.multimodal_gen.configs.sample.wan import (
        FastWanT2V480PConfig,
        Turbo_Wan2_2_I2V_A14B_SamplingParam,
        Wan2_1_Fun_1_3B_InP_SamplingParams,
        Wan2_2_I2V_A14B_SamplingParam,
        Wan2_2_T2V_A14B_SamplingParam,
        Wan2_2_TI2V_5B_SamplingParam,
        Wan_Animate_2_14B_SamplingParam,
        WanI2V_14B_480P_SamplingParam,
        WanI2V_14B_720P_SamplingParam,
        WanT2V_1_3B_SamplingParams,
        WanT2V_14B_SamplingParams,
    )
    from sglang.multimodal_gen.registry import register_configs

    register_configs(
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=WanT2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        ],
        model_detectors=[lambda hf_id: "wanpipeline" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=WanT2V_1_3B_SamplingParams,
        pipeline_config_cls=TurboWanT2V1_3B480PConfig,
        hf_model_paths=[
            "IPostYellow/TurboWan2.1-T2V-1.3B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=WanT2V_14B_SamplingParams,
        pipeline_config_cls=WanT2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=WanT2V_14B_SamplingParams,
        pipeline_config_cls=TurboWanT2V480PConfig,
        hf_model_paths=[
            "IPostYellow/TurboWan2.1-T2V-14B-Diffusers",
            "IPostYellow/TurboWan2.1-T2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_480P_SamplingParam,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        ],
        model_detectors=[lambda hf_id: "wanimagetovideo" in hf_id.lower()],
    )
    register_configs(
        sampling_param_cls=WanI2V_14B_720P_SamplingParam,
        pipeline_config_cls=WanI2V720PConfig,
        hf_model_paths=[
            "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Turbo_Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=TurboWanI2V720Config,
        hf_model_paths=[
            "IPostYellow/TurboWan2.2-I2V-A14B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_1_Fun_1_3B_InP_SamplingParams,
        pipeline_config_cls=WanI2V480PConfig,
        hf_model_paths=[
            "weizhou03/Wan2.1-Fun-1.3B-InP-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=Wan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_TI2V_5B_SamplingParam,
        pipeline_config_cls=FastWan2_2_TI2V_5B_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
            "FastVideo/FastWan2.2-TI2V-5B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_T2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_T2V_A14B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
            "nvidia/Wan2.2-T2V-A14B-Diffusers-NVFP4",
        ],
    )
    register_configs(
        sampling_param_cls=Wan2_2_I2V_A14B_SamplingParam,
        pipeline_config_cls=Wan2_2_I2V_A14B_Config,
        hf_model_paths=["Wan-AI/Wan2.2-I2V-A14B-Diffusers"],
    )
    register_configs(
        sampling_param_cls=FastWanT2V480PConfig,
        pipeline_config_cls=FastWan2_1_T2V_480P_Config,
        hf_model_paths=[
            "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
        ],
    )
    register_configs(
        sampling_param_cls=Wan_Animate_2_14B_SamplingParam,
        pipeline_config_cls=Wan_Animate_2_14B_Config,
        hf_model_paths=[
            "Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        ],
        model_detectors=[
            # The registry calls this with the path and with model_index's lower-cased
            # _class_name. Require the "-2-"/"_2" token so a v1 Wan2.2-Animate-14B
            # checkpoint (a different model / pipeline) does not mis-route here.
            lambda p: (
                "wan_animate_2" in p.lower()
                or "wan2.2-animate-2" in p.lower()
                or p.lower() == "wananimate2pipeline"
            )
        ],
    )
