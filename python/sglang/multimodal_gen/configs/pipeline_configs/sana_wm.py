# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable
from dataclasses import dataclass, field

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


SANA_WM_CHI_PROMPT: tuple[str, ...] = (
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
    """Pipeline config for the SANA-WM TI2V world model (text + first-frame image -> video,
    optional 6-DoF camera trajectory, optional Stage-2 LTX-2 refiner)."""

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

    enable_autocast: bool = False

    # --- Streaming self-forcing (S1c) ---
    # When ``streaming`` is set, the pipeline uses the autoregressive
    # SanaWMStreamingDenoisingStage (forward_long, chunk-by-chunk) instead of the
    # one-shot bidirectional denoise. ``num_frame_per_block`` is the streaming
    # chunk size in LATENT frames (distinct from the DiT's intra-attention
    # ``arch.chunk_size``). ``denoising_step_list`` must end in 0.
    streaming: bool = False
    num_frame_per_block: int = 3
    num_cached_blocks: int = 2
    sink_token: bool = True
    denoising_step_list: tuple = (1000, 960, 889, 727, 0)
    streaming_cfg_scale: float = 1.0
    # Streaming refiner (S2b): chunked LTX-2 sink/current refiner.
    sink_size: int = 1
    refiner_block_size: int = 3
    refiner_kv_max_frames: int = 11
    refiner_seed: int = 42
    # True -> chunked streaming refiner (low-latency, causal); False -> whole-clip
    # dense refiner (global context, max quality, non-streaming).
    refiner_chunked: bool = True

    # --- DiT ---
    dit_config: DiTConfig = field(default_factory=SanaWMConfig)

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

    # --- Text encoder: Gemma-2-2b-it (single encoder, same as SANA T2I) ---
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(
        default_factory=lambda: [
            {
                # Match NVlabs SANA-WM prompt encoding: positive and negative
                # branches must have the same token dimension for CFG concat.
                "padding": "max_length",
                "return_attention_mask": True,
            }
        ]
    )
    chi_prompt: tuple[str, ...] = SANA_WM_CHI_PROMPT
    preprocess_text_funcs: tuple[Callable | None, ...] = field(
        default_factory=lambda: (None,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_wm_postprocess_text,)
    )

    # --- Scheduler ---
    # linear_flow training schedule from the released config.yaml.
    flow_shift: float = 9.95
    # Official NVlabs/Sana inference resolves inference_flow_shift first and
    # falls back to flow_shift only when it is absent.
    inference_flow_shift: float | None = 9.8

    # --- Video shape ---
    # VAE strides: (temporal, spatial_h, spatial_w)
    vae_stride: tuple = (8, 32, 32)  # LTX-2 VAE temporal=8, spatial=32

    # --- Camera conditioning ---
    camera_conditioning: bool = True  # set False to disable camera branch

    # --- Deployment ---
    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        return ModelDeploymentConfig(
            auto_dit_layerwise_offload=True,
            # Conservative auto-FSDP gate for the 720p world-model path. Users
            # can still force FSDP explicitly on smaller cards.
            fsdp_auto_min_available_memory_gb=60,
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
          * camera_conditions      -- (B, T_lat, 20) latent-frame UCPE raymap
          * chunk_plucker          -- (B, 48, T_lat, H, W) packed Plücker raymap
        """
        out = {}

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
        """Build negative conditioning kwargs for CFG.

        Camera/plucker are structural video conditions, not text conditions.
        NVlabs SANA-WM duplicates them for both CFG branches and only swaps
        text embeddings/masks.
        """
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
        # SANA-WM uses frame-wise GDN recurrent scan and a temporal depth-wise
        # conv (GLUMBConvTemp, t_kernel=3) that both span across frames. Splitting
        # the latent along T would truncate the GDN hidden state and drop the
        # GLUMBConvTemp halo at rank boundaries, producing silent wrong outputs.
        # Camera/Plücker tensors are also indexed in lockstep with T and would
        # need matching shards. Disable SP until a halo-exchange-aware impl lands.
        return latents, False

    def gather_latents_for_sp(self, latents):
        return latents

    def get_decode_scale_and_shift(self, device, dtype, vae):
        """Invert the LTX-2 latent normalization used before denoising.

        SANA-WM uses the LTX-2 VAE. Upstream encodes as
        ``(z - latents_mean) * scaling_factor / latents_std`` and decodes by
        applying the inverse transform.
        """
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


class SanaWMRealtimeConfig(SanaWMPipelineConfig):
    """Realtime alias of the SANA-WM pipeline config.

    Same numerics/fields as SanaWMPipelineConfig (our correct streaming pipeline);
    exists so the realtime-serving registry can key adapters on a realtime config
    class (matching the upstream/LingBot-World pattern) without renaming the base.
    """

    pass
