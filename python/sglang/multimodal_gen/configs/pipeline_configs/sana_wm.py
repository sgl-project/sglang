# SPDX-License-Identifier: Apache-2.0
#
# Pipeline configuration for SANA-WM TI2V (Text+Image-to-Video) world model.
#
# Key differences vs SanaPipelineConfig (T2I):
#   - task_type = TI2V: requires first-frame image input
#   - 5D latent (B, C, T, H, W) via LTX-2 VAE (8× temporal, 32× spatial, 128 ch)
#   - Camera trajectory conditioning (optional): camera_to_world + intrinsics
#   - flow_shift = 9.95 (linear_flow schedule, from config.yaml)
#   - Two-stage optional: Stage-1 (SANA-WM) + Stage-2 (LTX-2 Refiner)
#
# The DiT forward receives camera_to_world + intrinsics (or plucker) via
# prepare_pos_cond_kwargs. Set them to None for text-only (T2V without camera).

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Optional

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.base import EncoderConfig
from sglang.multimodal_gen.configs.models.encoders.gemma2 import Gemma2Config
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


def sana_wm_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract Gemma-2 last hidden state as text conditioning (same as SANA T2I)."""
    return outputs.last_hidden_state


@dataclass
class SanaWMPipelineConfig(PipelineConfig):
    """
    Pipeline configuration for SANA-WM TI2V world model.

    Supports:
    - Text + first-frame image → video (TI2V)
    - Optional 6-DoF camera trajectory conditioning via camera_to_world + intrinsics
    - Two-stage inference: Stage-1 (SANA-WM DiT) + optional Stage-2 (LTX-2 Refiner)
    """

    task_type: ModelTaskType = ModelTaskType.TI2V

    # SanaWMBeforeDenoisingStage._splice_first_frame handles condition-image
    # resize + VAE-encode itself, so bypass the framework's generic TI2V
    # preprocessing in InputValidationStage. Without this, the framework path
    # reads `vae_config.arch_config.scale_factor_spatial` which LTXVideoVAEArchConfig
    # does not expose (it uses `spatial_compression_ratio` instead). LTX-2 sets
    # the same flag for the same reason -- both are TI2V on LTXVideoVAEConfig.
    skip_input_image_preprocess: bool = True

    # --- Guidance ---
    # SANA-WM uses standard CFG via guidance_scale; no embedded guidance token.
    should_use_guidance: bool = False

    # --- Autocast ---
    enable_autocast: bool = False

    # --- DiT ---
    dit_config: DiTConfig = field(default_factory=SanaWMConfig)

    # --- VAE: LTX-2 (128ch, 8× temporal, 32× spatial) ---
    vae_config: VAEConfig = field(default_factory=LTXVideoVAEConfig)
    vae_precision: str = "bf16"
    vae_tiling: bool = False   # LTX-2 VAE does not use tiling by default
    vae_sp: bool = False        # no VAE SP for now

    # Load both encoder and decoder (need encoder for first-frame conditioning)
    def __post_init__(self):
        self.vae_config.load_encoder = True
        self.vae_config.load_decoder = True

    # --- Text encoder: Gemma-2-2b-it (single encoder, same as SANA T2I) ---
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                "padding": True,
                "return_attention_mask": True,
            }
        ]
    )
    preprocess_text_funcs: tuple[Callable | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_wm_postprocess_text,)
    )

    # --- Scheduler ---
    # linear_flow matching with flow_shift=9.95 (from config.yaml)
    flow_shift: float = 9.95

    # --- Video shape ---
    # VAE strides: (temporal, spatial_h, spatial_w)
    vae_stride: tuple = (8, 32, 32)   # LTX-2 VAE temporal=8, spatial=32

    # --- Camera conditioning ---
    camera_conditioning: bool = True  # set False to disable camera branch

    # --- Deployment ---
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
        )

    # --- Latent shape ---
    def prepare_latent_shape(self, batch, batch_size: int, num_frames: int):
        """
        Returns 5D latent shape: (B, 128, T_latent, H_sp, W_sp).
        T_latent = ceil((num_frames - 1) / temporal_stride) + 1
        """
        t_stride = self.vae_stride[0]
        s_stride = self.vae_stride[1]

        T_latent = (num_frames - 1) // t_stride + 1
        H_sp = batch.height // s_stride
        W_sp = batch.width // s_stride
        z_dim = self.vae_config.arch_config.latent_channels  # 128

        return (batch_size, z_dim, T_latent, H_sp, W_sp)

    def adjust_num_frames(self, num_frames: int) -> int:
        """Ensure (num_frames - 1) is divisible by VAE temporal stride."""
        t_stride = self.vae_stride[0]
        if (num_frames - 1) % t_stride != 0:
            adjusted = ((num_frames - 1) // t_stride) * t_stride + 1
            logger.warning(
                f"num_frames - 1 must be divisible by temporal stride {t_stride}. "
                f"Rounding {num_frames} → {adjusted}."
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
        """Build positive conditioning kwargs passed to SanaWMTransformer3DModel.forward.

        The DiT forward signature consumes:
          * encoder_hidden_states  -- Gemma-2 embeddings (set by DenoisingStage)
          * timestep               -- diffusion step (set by DenoisingStage)
          * encoder_attention_mask -- text padding mask
          * camera_conditions      -- (B, F_orig, 20) UCPE input
          * chunk_plucker          -- (B, 48, T_lat, H, W) packed Plücker raymap
        """
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
        """Build negative conditioning kwargs for CFG (no camera on the negative pass)."""
        out = {}
        m = batch.negative_attention_mask
        if isinstance(m, (list, tuple)):
            out["encoder_attention_mask"] = m[0] if m else None
        elif m is not None:
            out["encoder_attention_mask"] = m
        # Intentionally omit camera_conditions / chunk_plucker -- the negative
        # pass uses the text-unconditional baseline without camera control.
        return out

    # --- Post-processing ---
    def post_denoising_loop(self, latents: torch.Tensor, batch) -> torch.Tensor:
        """No token un-packing needed; 5D latents are already spatial."""
        return latents

    def shard_latents_for_sp(self, batch, latents):
        # SANA-WM uses frame-wise GDN recurrent scan and a temporal depth-wise
        # conv (GLUMBConvTemp, t_kernel=3) that both span across frames. Splitting
        # the latent along T would truncate the GDN hidden state and drop the
        # GLUMBConvTemp halo at rank boundaries, producing silent wrong outputs.
        # Camera/Plücker tensors are also indexed in lockstep with T and would
        # need matching shards. Disable SP until a halo-exchange-aware impl lands.
        return latents, False

    def gather_latents_for_sp(self, latents):
        return latents
