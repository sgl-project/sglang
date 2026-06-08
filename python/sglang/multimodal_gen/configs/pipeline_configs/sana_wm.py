# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.configs.models.dits.sana_wm_refiner import (
    SanaWMRefinerConfig,
)
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
from sglang.multimodal_gen.configs.models.encoders.gemma_3 import Gemma3Config
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.model_deployment_config import (
    ModelDeploymentConfig,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG = 70.0
_SANA_WM_MIN_DEFAULT_FOV_DEG = 25.0
_SANA_WM_MAX_DEFAULT_FOV_DEG = 120.0
_SANA_WM_TORCH_COMPILE_SCOPES: tuple[str, ...] = ("regional", "full", "off")


def _normalize_sana_wm_default_horizontal_fov_deg(value=None) -> float:
    if value is None:
        fov_deg = _SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG
    else:
        try:
            fov_deg = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SANA-WM default horizontal FOV must be a number, " f"got {value!r}."
            ) from exc
    if not _SANA_WM_MIN_DEFAULT_FOV_DEG < fov_deg < _SANA_WM_MAX_DEFAULT_FOV_DEG:
        raise ValueError(
            f"SANA-WM default horizontal FOV must be in "
            f"({_SANA_WM_MIN_DEFAULT_FOV_DEG}, "
            f"{_SANA_WM_MAX_DEFAULT_FOV_DEG}) degrees, got {fov_deg}."
        )
    return fov_deg


def _normalize_sana_wm_choice(
    value,
    *,
    default: str,
    valid_values: tuple[str, ...],
    name: str,
    strict: bool,
) -> str:
    mode = default if value is None else str(value).strip().lower()
    if not mode:
        mode = default
    if mode in valid_values:
        return mode

    if strict:
        raise ValueError(
            f"{name} must be one of {sorted(valid_values)}, got {value!r}."
        )

    logger.warning(
        "Ignoring invalid %s=%r. Expected one of %s; using %r.",
        name,
        value,
        sorted(valid_values),
        default,
    )
    return default


def _normalize_sana_wm_two_stage_residency(
    value,
    *,
    strict: bool = True,
    name: str = "sana_wm_two_stage_residency",
) -> str:
    return _normalize_sana_wm_choice(
        value,
        default="auto",
        valid_values=("auto", "resident", "sequential"),
        name=name,
        strict=strict,
    )


def _normalize_sana_wm_torch_compile_scope(
    value,
    *,
    strict: bool = True,
    name: str = "sana_wm_torch_compile_scope",
) -> str:
    if value is not None:
        value = str(value).strip().lower().replace("-", "_")
        aliases = {
            "0": "off",
            "false": "off",
            "no": "off",
            "none": "off",
            "block": "regional",
            "blocks": "regional",
            "regional_blocks": "regional",
            "module": "full",
            "transformer": "full",
            "full_module": "full",
        }
        value = aliases.get(value, value)
    return _normalize_sana_wm_choice(
        value,
        default="off",
        valid_values=_SANA_WM_TORCH_COMPILE_SCOPES,
        name=name,
        strict=strict,
    )


def _normalize_sana_wm_bool(
    value,
    *,
    name: str,
) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}.")


def _sana_wm_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract Gemma-2 last hidden state as text conditioning (same as SANA T2I)."""
    return outputs.last_hidden_state


_SANA_WM_CHI_PROMPT: tuple[str, ...] = (
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed '
    "visual descriptions suitable for image generation. Evaluate the level of "
    "detail in the user prompt:",
    "- If the prompt is simple, focus on adding specifics about colors, shapes, "
    "sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details "
    "slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled "
    "up in a round shape, sleeping peacefully on a warm sunny windowsill, "
    "surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene "
    "at dusk, featuring glowing street lamps, a diverse crowd of people in "
    "colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and "
    "avoid including any additional commentary or evaluations:",
    "User Prompt: ",
)


@dataclass
class SanaWMPipelineConfig(PipelineConfig):

    task_type: ModelTaskType = ModelTaskType.TI2V

    skip_input_image_preprocess: bool = True

    # --- Guidance ---
    # SANA-WM uses standard CFG via guidance_scale; no embedded guidance token.
    should_use_guidance: bool = False

    # --- Autocast ---
    enable_autocast: bool = False

    # --- DiT ---
    dit_config: DiTConfig = field(default_factory=SanaWMConfig)
    refiner_dit_config: DiTConfig = field(default_factory=SanaWMRefinerConfig)

    # --- SANA-WM runtime controls ---
    sana_wm_two_stage_residency: str = "auto"
    sana_wm_skip_refiner: bool = False
    sana_wm_diagnostics: bool = False
    # SANA-WM torch.compile is opt-in until its graph-break/recompile behavior
    # is validated across request shapes and camera-conditioning modes.
    sana_wm_torch_compile_scope: str = "off"
    sana_wm_torch_compile_mode: str | None = None
    sana_wm_torch_compile_cache_size_limit: int = 128

    # --- VAE: LTX-2 (128ch, 8× temporal, 32× spatial) ---
    vae_config: VAEConfig = field(default_factory=LTXVideoVAEConfig)
    vae_precision: str = "bf16"
    # Match NVlabs SANA-WM inference: long videos must use LTX-2 spatial
    # tiling plus framewise temporal decode to avoid oversized Conv3d pads.
    vae_tiling: bool = True
    vae_sp: bool = False  # no VAE SP for now
    vae_framewise_encoding: bool = True
    vae_framewise_decoding: bool = True
    vae_tile_sample_min_num_frames: int = 96
    vae_tile_sample_stride_num_frames: int = 64

    # Load both encoder and decoder (need encoder for first-frame conditioning)
    def __post_init__(self):
        self.sana_wm_two_stage_residency = _normalize_sana_wm_two_stage_residency(
            self.sana_wm_two_stage_residency
        )
        self.sana_wm_skip_refiner = _normalize_sana_wm_bool(
            self.sana_wm_skip_refiner,
            name="sana_wm_skip_refiner",
        )
        self.sana_wm_diagnostics = _normalize_sana_wm_bool(
            self.sana_wm_diagnostics,
            name="sana_wm_diagnostics",
        )
        self.sana_wm_torch_compile_scope = _normalize_sana_wm_torch_compile_scope(
            self.sana_wm_torch_compile_scope
        )
        if self.sana_wm_torch_compile_mode is not None:
            self.sana_wm_torch_compile_mode = (
                str(self.sana_wm_torch_compile_mode).strip() or None
            )
        self.sana_wm_torch_compile_cache_size_limit = int(
            self.sana_wm_torch_compile_cache_size_limit
        )
        if self.sana_wm_torch_compile_cache_size_limit < 1:
            raise ValueError(
                "sana_wm_torch_compile_cache_size_limit must be positive, got "
                f"{self.sana_wm_torch_compile_cache_size_limit}."
            )
        self.sana_wm_default_horizontal_fov_deg = (
            _normalize_sana_wm_default_horizontal_fov_deg(
                self.sana_wm_default_horizontal_fov_deg
            )
        )

        if hasattr(self.dit_config, "apply_user_flags_to_arch_config"):
            self.dit_config.apply_user_flags_to_arch_config()

        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True
        self.vae_config.use_tiling = self.vae_tiling
        self.vae_config.use_temporal_tiling = self.vae_framewise_decoding
        self.vae_config.tile_sample_min_num_frames = self.vae_tile_sample_min_num_frames
        self.vae_config.tile_sample_stride_num_frames = (
            self.vae_tile_sample_stride_num_frames
        )
        self.vae_config.blend_num_frames = (
            self.vae_tile_sample_min_num_frames - self.vae_tile_sample_stride_num_frames
        )

    def update_config_from_dict(self, args: dict, prefix: str = "") -> None:
        super().update_config_from_dict(args, prefix)
        self.__post_init__()

    # --- Text encoders: stage-1 Gemma-2 and native refiner Gemma-3 ---
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(), Gemma3Config())
    )
    text_encoder_precisions: tuple[str, ...] = field(
        default_factory=lambda: ("bf16", "bf16")
    )
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                # Match NVlabs SANA-WM prompt encoding: positive and negative
                # branches must have the same token dimension for CFG concat.
                "padding": "max_length",
                "return_attention_mask": True,
            },
            {},
        ]
    )
    chi_prompt: tuple[str, ...] = _SANA_WM_CHI_PROMPT
    preprocess_text_funcs: tuple[Callable | None, ...] = field(
        default_factory=lambda: (None, None)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (_sana_wm_postprocess_text, _sana_wm_postprocess_text)
    )

    # --- Scheduler ---
    # linear_flow training schedule from the released config.yaml.
    flow_shift: float = 9.95
    # SANA-WM inference resolves inference_flow_shift first and falls back to
    # flow_shift only when it is absent.
    inference_flow_shift: float | None = 9.8

    # --- Video shape ---
    # VAE strides: (temporal, spatial_h, spatial_w)
    vae_stride: tuple = (8, 32, 32)  # LTX-2 VAE temporal=8, spatial=32

    # --- Camera conditioning ---
    camera_conditioning: bool = True  # set False to disable camera branch
    sana_wm_default_horizontal_fov_deg: float = _SANA_WM_DEFAULT_HORIZONTAL_FOV_DEG

    # --- Deployment ---
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            # On high-memory GPUs, keep the default component-offload path
            # resident. Tensor parallelism remains an explicit launch choice.
            auto_disable_component_offload_min_available_memory_gb=70,
        )

    # --- Latent shape ---
    def prepare_latent_shape(self, batch, batch_size: int, num_frames: int):
        """
        Returns 5D latent shape: (B, 128, T_latent, H_sp, W_sp).
        T_latent = ceil((num_frames - 1) / temporal_stride) + 1
        """
        t_stride = self.vae_stride[0]
        h_stride = self.vae_stride[1]
        w_stride = self.vae_stride[2] if len(self.vae_stride) > 2 else h_stride

        if batch.height % h_stride != 0 or batch.width % w_stride != 0:
            raise ValueError(
                "SANA-WM height/width must be divisible by the LTX-2 spatial "
                f"stride ({h_stride}, {w_stride}); got "
                f"height={batch.height}, width={batch.width}."
            )

        T_latent = (num_frames - 1) // t_stride + 1
        H_sp = batch.height // h_stride
        W_sp = batch.width // w_stride
        z_dim = self.vae_config.arch_config.latent_channels  # 128

        return (batch_size, z_dim, T_latent, H_sp, W_sp)

    def adjust_num_frames(self, num_frames: int) -> int:
        """Round down to the largest VAE-stride-aligned frame count."""
        t_stride = self.vae_stride[0]
        if (num_frames - 1) % t_stride != 0:
            adjusted = max(((num_frames - 1) // t_stride) * t_stride + 1, 1)
            logger.warning(
                f"num_frames - 1 must be divisible by temporal stride {t_stride}. "
                f"Rounding down {num_frames} to {adjusted} to avoid generating "
                "more frames than requested."
            )
            return adjusted
        return num_frames

    # --- Text embedding accessors ---
    def get_pos_prompt_embeds(self, batch):
        return batch.prompt_embeds[0]

    def get_neg_prompt_embeds(self, batch):
        return batch.negative_prompt_embeds[0]

    # --- Conditioning kwargs for DenoisingStage ---
    def prepare_pos_cond_kwargs(self, batch, device, rotary_emb, dtype):
        out = {}

        # Text attention mask
        m = batch.prompt_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m

        # Camera conditioning (built by SanaWMBeforeDenoisingStage)
        if hasattr(batch, "extra") and batch.extra:
            cc = batch.extra.get("camera_conditions", None)
            cp = batch.extra.get("chunk_plucker", None)
            if cc is not None:
                out["camera_conditions"] = cc
            if cp is not None:
                out["chunk_plucker"] = cp

        return out

    def prepare_neg_cond_kwargs(self, batch, device, rotary_emb, dtype):
        out = {}
        m = batch.negative_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        if hasattr(batch, "extra") and batch.extra:
            cc = batch.extra.get("camera_conditions", None)
            cp = batch.extra.get("chunk_plucker", None)
            if cc is not None:
                out["camera_conditions"] = cc
            if cp is not None:
                out["chunk_plucker"] = cp
        return out

    # --- Post-processing ---
    def post_denoising_loop(self, latents: torch.Tensor, batch) -> torch.Tensor:
        """No token un-packing needed; 5D latents are already spatial."""
        return latents

    def shard_latents_for_sp(self, batch, latents):
        # Extension hook for future true SP. Stage-level 5D time sharding would
        # break temporal GDN/GLUMBConvTemp/camera UCPE semantics, so keep stage-1
        # latents replicated until the DiT grows layout-aware token sharding.
        return latents, False

    def gather_latents_for_sp(self, latents, batch=None):
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        latents_mean = getattr(vae, "latents_mean", None)
        latents_std = getattr(vae, "latents_std", None)

        scaling_factor = (
            getattr(getattr(vae, "config", None), "scaling_factor", None)
            or getattr(vae, "scaling_factor", None)
            or getattr(self.vae_config.arch_config, "scaling_factor", None)
            or 1.0
        )
        if isinstance(scaling_factor, (int, float)) and float(scaling_factor) == 0.0:
            scaling_factor = 1.0

        if isinstance(latents_mean, torch.Tensor) and isinstance(
            latents_std, torch.Tensor
        ):
            latents_mean = latents_mean.to(device=device, dtype=dtype).view(
                1, -1, 1, 1, 1
            )
            latents_std = latents_std.to(device=device, dtype=dtype).view(
                1, -1, 1, 1, 1
            )
            sf = torch.tensor(float(scaling_factor), device=device, dtype=dtype).view(
                1, 1, 1, 1, 1
            )
            return sf / latents_std, latents_mean

        sf = torch.tensor(float(scaling_factor), device=device, dtype=dtype).view(
            1, 1, 1, 1, 1
        )
        return sf, None
