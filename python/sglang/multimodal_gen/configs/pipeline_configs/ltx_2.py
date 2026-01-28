import dataclasses
from dataclasses import field
from typing import Callable

import numpy as np
import torch

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.configs.models.encoders import (
    BaseEncoderOutput,
    EncoderConfig,
)
from sglang.multimodal_gen.configs.models.encoders.gemma_3 import Gemma3Config
from sglang.multimodal_gen.configs.models.vaes.ltx_audio import LTXAudioVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ModelTaskType,
    PipelineConfig,
    preprocess_text,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
    sequence_model_parallel_all_gather,
)


def pack_text_embeds(
    text_hidden_states: torch.Tensor,
    sequence_lengths: torch.Tensor,
    padding_side: str = "left",
    scale_factor: int = 8,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Packs and normalizes text encoder hidden states, respecting padding. Normalization is performed per-batch and
    per-layer in a masked fashion (only over non-padded positions).

    Args:
        text_hidden_states (`torch.Tensor` of shape `(batch_size, seq_len, hidden_dim, num_layers)`):
            Per-layer hidden_states from a text encoder (e.g. `Gemma3ForConditionalGeneration`).
        sequence_lengths (`torch.Tensor of shape `(batch_size,)`):
            The number of valid (non-padded) tokens for each batch instance.
        device: (`str` or `torch.device`, *optional*):
            torch device to place the resulting embeddings on
        padding_side: (`str`, *optional*, defaults to `"left"`):
            Whether the text tokenizer performs padding on the `"left"` or `"right"`.
        scale_factor (`int`, *optional*, defaults to `8`):
            Scaling factor to multiply the normalized hidden states by.
        eps (`float`, *optional*, defaults to `1e-6`):
            A small positive value for numerical stability when performing normalization.

    Returns:
        `torch.Tensor` of shape `(batch_size, seq_len, hidden_dim * num_layers)`:
            Normed and flattened text encoder hidden states.
    """
    batch_size, seq_len, hidden_dim, num_layers = text_hidden_states.shape
    original_dtype = text_hidden_states.dtype
    device = text_hidden_states.device

    # Create padding mask
    token_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    if padding_side == "right":
        mask = token_indices < sequence_lengths[:, None]
    elif padding_side == "left":
        start_indices = seq_len - sequence_lengths[:, None]
        mask = token_indices >= start_indices
    else:
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side}")
    mask = mask[:, :, None, None]  # [batch_size, seq_len, 1, 1]

    masked_text_hidden_states = text_hidden_states.masked_fill(~mask, 0.0)
    num_valid_positions = (sequence_lengths * hidden_dim).view(batch_size, 1, 1, 1)
    masked_mean = masked_text_hidden_states.sum(dim=(1, 2), keepdim=True) / (
        num_valid_positions + eps
    )

    x_min = text_hidden_states.masked_fill(~mask, float("inf")).amin(
        dim=(1, 2), keepdim=True
    )
    x_max = text_hidden_states.masked_fill(~mask, float("-inf")).amax(
        dim=(1, 2), keepdim=True
    )

    normalized_hidden_states = (text_hidden_states - masked_mean) / (
        x_max - x_min + eps
    )
    normalized_hidden_states = normalized_hidden_states * scale_factor

    normalized_hidden_states = normalized_hidden_states.flatten(2)
    mask_flat = mask.squeeze(-1).expand(-1, -1, hidden_dim * num_layers)
    normalized_hidden_states = normalized_hidden_states.masked_fill(~mask_flat, 0.0)
    normalized_hidden_states = normalized_hidden_states.to(dtype=original_dtype)

    return normalized_hidden_states


def _gemma_postprocess_func(
    outputs: BaseEncoderOutput, text_inputs: dict
) -> torch.Tensor:
    # LTX-2 requires all hidden states concatenated for the connector
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        # outputs.hidden_states is a tuple of tensors
        # We need to stack them along the last dimension and pack them
        hidden_states = torch.stack(outputs.hidden_states, dim=-1)
        attention_mask = text_inputs["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=-1)
        # Assuming left padding for Gemma as per Diffusers
        return pack_text_embeds(hidden_states, sequence_lengths, padding_side="left")
    else:
        raise AttributeError(
            "Unsupported text encoder output: expected `hidden_states`."
        )


@dataclasses.dataclass
class LTX2PipelineConfig(PipelineConfig):
    """Configuration for LTX-Video pipeline."""

    task_type: ModelTaskType = ModelTaskType.T2V
    dit_config: LTX2Config = field(default_factory=LTX2Config)

    # Model architecture
    in_channels: int = 128
    out_channels: int = 128
    patch_size: int = 1
    patch_size_t: int = 1

    # Audio VAE configuration
    audio_vae_config: LTXAudioVAEConfig = field(default_factory=LTXAudioVAEConfig)
    audio_vae_precision: str = "fp32"
    audio_vae_temporal_compression_ratio: int = 4
    audio_vae_mel_compression_ratio: int = 4

    @property
    def vae_scale_factor(self):
        return getattr(self.vae_config.arch_config, "spatial_compression_ratio", 32)

    @property
    def vae_temporal_compression(self):
        return getattr(self.vae_config.arch_config, "temporal_compression_ratio", 8)

    def prepare_audio_latent_shape(self, batch, batch_size, num_frames):
        # Adapted from diffusers pipeline prepare_audio_latents
        duration_s = num_frames / batch.fps

        sample_rate = self.audio_vae_config.arch_config.sample_rate
        hop_length = self.audio_vae_config.arch_config.mel_hop_length
        temporal_compression = self.audio_vae_temporal_compression_ratio

        latents_per_second = (
            float(sample_rate) / float(hop_length) / float(temporal_compression)
        )
        latent_length = round(duration_s * latents_per_second)

        num_mel_bins = self.audio_vae_config.arch_config.mel_bins
        mel_compression_ratio = self.audio_vae_mel_compression_ratio
        latent_mel_bins = num_mel_bins // mel_compression_ratio

        # Default to 8
        num_channels_latents = self.audio_vae_config.arch_config.latent_channels

        shape = (batch_size, num_channels_latents, latent_length, latent_mel_bins)

        return shape

    # Text encoding stage (Gemma)
    # LTX-2 needs separate contexts for video/audio streams. We model this as
    # two logical encoders sharing the same underlying `text_encoder` module.
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma3Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))
    text_encoder_extra_args: list[dict] = field(default_factory=lambda: [{}])

    preprocess_text_funcs: tuple[Callable[[str], str], ...] = field(
        default_factory=lambda: (preprocess_text,)
    )
    postprocess_text_funcs: tuple[
        Callable[[BaseEncoderOutput, dict], torch.Tensor], ...
    ] = field(default_factory=lambda: (_gemma_postprocess_func,))

    def prepare_sigmas(self, sigmas, num_inference_steps):
        if sigmas is None:
            steps = int(num_inference_steps)
            if steps <= 0:
                raise ValueError(f"num_inference_steps must be positive, got {steps}")
            return np.linspace(1.0, 1.0 / float(steps), steps).tolist()
        return sigmas

    def tokenize_prompt(self, prompt: list[str], tokenizer, tok_kwargs) -> dict:
        # Adapted from diffusers_pipeline.py _get_gemma_prompt_embeds
        # But we only need tokenization here, the embedding happens in TextEncodingStage

        # Gemma expects left padding for chat-style prompts
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        max_sequence_length = tok_kwargs.get(
            "max_length", 1024
        )  # Default from diffusers pipeline

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs

    def maybe_pack_latents(self, latents, batch_size, batch):
        # Unpacked latents of shape are [B, C, F, H, W] are patched into tokens of shape [B, C, F // p_t, p_t, H // p, p, W // p, p].
        # The patch dimensions are then permuted and collapsed into the channel dimension of shape:
        # [B, F // p_t * H // p * W // p, C * p_t * p * p] (an ndim=3 tensor).
        # dim=0 is the batch size, dim=1 is the effective video sequence length, dim=2 is the effective number of input features
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_patch_num_frames = num_frames // self.patch_size_t
        post_patch_height = height // self.patch_size
        post_patch_width = width // self.patch_size
        latents = latents.reshape(
            batch_size,
            -1,
            post_patch_num_frames,
            self.patch_size_t,
            post_patch_height,
            self.patch_size,
            post_patch_width,
            self.patch_size,
        )
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    def _infer_video_latent_frames_and_tokens_per_frame(
        self, batch, seq_len: int
    ) -> tuple[int, int]:
        """Infer latent-frame count and tokens-per-frame for packed token latents [B, S, D].

        Notes:
        - This assumes `patch_size_t == 1` (no temporal patching).
        - Tokens are ordered as (frame, height, width) after packing.
        """
        if int(self.patch_size_t) != 1:
            raise ValueError(
                "LTX-2 SP time-sharding for packed token latents currently requires "
                f"{self.patch_size_t=}. (Expected 1)"
            )
        if int(seq_len) <= 0:
            raise ValueError(f"Expected {seq_len=} > 0 for packed token latents.")
        if int(self.vae_scale_factor) <= 0:
            raise ValueError(f"Invalid {self.vae_scale_factor=}. Must be > 0.")
        if int(self.patch_size) <= 0:
            raise ValueError(f"Invalid {self.patch_size=}. Must be > 0.")

        latent_height = int(batch.height) // int(self.vae_scale_factor)
        latent_width = int(batch.width) // int(self.vae_scale_factor)
        if latent_height <= 0 or latent_width <= 0:
            raise ValueError(
                "Invalid latent H/W computed from batch.height/width: "
                f"{batch.height=} {batch.width=} {self.vae_scale_factor=}"
            )
        if (latent_height % int(self.patch_size)) != 0 or (
            latent_width % int(self.patch_size)
        ) != 0:
            raise ValueError(
                "Invalid spatial patching for packed token latents. Expected latent H/W "
                "to be divisible by patch_size, got "
                f"{latent_height=} {latent_width=} {self.patch_size=}."
            )

        post_patch_h = latent_height // int(self.patch_size)
        post_patch_w = latent_width // int(self.patch_size)
        tokens_per_frame = int(post_patch_h) * int(post_patch_w)
        if tokens_per_frame <= 0:
            raise ValueError(
                f"Invalid tokens_per_frame={tokens_per_frame} from "
                f"{latent_height=} {latent_width=} {self.patch_size=}"
            )
        if int(seq_len) % int(tokens_per_frame) != 0:
            raise ValueError(
                f"LTX-2 token latents seq_len={seq_len} is not divisible by "
                f"tokens_per_frame={tokens_per_frame}. Cannot time-shard for SP."
            )
        latent_num_frames = int(seq_len) // int(tokens_per_frame)
        return int(latent_num_frames), int(tokens_per_frame)

    def shard_latents_for_sp(self, batch, latents):
        """Shard LTX-2 packed token latents across SP ranks by latent time (frame) dimension."""
        sp_world_size = get_sp_world_size()
        if sp_world_size <= 1:
            return latents, False

        # Default behavior for 5D latents.
        if isinstance(latents, torch.Tensor) and latents.ndim == 5:
            return super().shard_latents_for_sp(batch, latents)

        # LTX-2 packed token latents [B, S, D]
        if not (isinstance(latents, torch.Tensor) and latents.ndim == 3):
            return latents, False

        sp_rank = get_sp_parallel_rank()
        seq_len = int(latents.shape[1])
        latent_frames, tokens_per_frame = (
            self._infer_video_latent_frames_and_tokens_per_frame(batch, seq_len)
        )

        # Pad whole frames so `latent_frames` is divisible by `sp_world_size`.
        pad_frames = (sp_world_size - (latent_frames % sp_world_size)) % sp_world_size
        if pad_frames:
            pad_tokens = int(pad_frames) * int(tokens_per_frame)
            pad = torch.zeros(
                (latents.shape[0], pad_tokens, latents.shape[2]),
                device=latents.device,
                dtype=latents.dtype,
            )
            latents = torch.cat([latents, pad], dim=1)
            latent_frames = int(latent_frames) + int(pad_frames)

        local_frames = int(latent_frames) // int(sp_world_size)
        start_frame = int(sp_rank) * int(local_frames)
        start = int(start_frame) * int(tokens_per_frame)
        end = int(start) + int(local_frames) * int(tokens_per_frame)
        latents = latents[:, start:end, :]

        # Store SP metadata for denoising (TI2V gating) and model-side RoPE shift.
        batch.sp_video_latent_num_frames = int(local_frames)
        batch.sp_video_start_frame = int(start_frame)
        batch.sp_video_tokens_per_frame = int(tokens_per_frame)

        return latents, True

    def gather_latents_for_sp(self, latents):
        """Gather latents after SP. For packed token latents [B, S_local, D], gather on dim=1."""
        if get_sp_world_size() <= 1:
            return latents
        if isinstance(latents, torch.Tensor) and latents.ndim == 3:
            return sequence_model_parallel_all_gather(latents.contiguous(), dim=1)
        return super().gather_latents_for_sp(latents)

    def maybe_pack_audio_latents(self, latents, batch_size, batch):
        # Audio latents shape: [B, C, L, M], where L is the latent audio length and M is the number of mel bins
        # We need to pack them if patch_size/patch_size_t are defined for audio (not standard DiT patch size)

        # So for LTX-2 (unless we change patch sizes), we just do:
        latents = latents.transpose(1, 2).flatten(
            2, 3
        )  # [B, C, L, M] --> [B, L, C * M]
        return latents

    def get_pos_prompt_embeds(self, batch):
        # LTX-2 returns multiple prompt embed tensors (video/audio contexts).
        return (
            batch.prompt_embeds[0]
            if isinstance(batch.prompt_embeds, list)
            else batch.prompt_embeds
        )

    def get_neg_prompt_embeds(self, batch):
        return (
            batch.negative_prompt_embeds[0]
            if isinstance(batch.negative_prompt_embeds, list)
            else batch.negative_prompt_embeds
        )

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

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions)
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(
            batch_size,
            num_frames,
            height,
            width,
            -1,
            patch_size_t,
            patch_size,
            patch_size,
        )
        latents = (
            latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        return latents

    @staticmethod
    def _denormalize_latents(
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    def _denormalize_audio_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ):
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return (latents * latents_std) + latents_mean

    @staticmethod
    def _unpack_audio_latents(
        latents: torch.Tensor,
        latent_length: int,
        num_mel_bins: int,
        patch_size: int | None = None,
        patch_size_t: int | None = None,
    ) -> torch.Tensor:
        # Unpacks an audio patch sequence of shape [B, S, D] into a latent spectrogram tensor of shape [B, C, L, M],
        # where L is the latent audio length and M is the number of mel bins.
        if patch_size is not None and patch_size_t is not None:
            batch_size = latents.size(0)
            latents = latents.reshape(
                batch_size, latent_length, num_mel_bins, -1, patch_size_t, patch_size
            )
            latents = latents.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)
        else:
            # Assume [B, S, D] = [B, L, C * M], which implies that patch_size = M and patch_size_t = 1.
            latents = latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)
        return latents

    def _unpad_and_unpack_latents(self, latents, audio_latents, batch, vae, audio_vae):
        # Calculate latent dimensions
        # Assuming batch has height, width, num_frames
        height = batch.height
        width = batch.width
        num_frames = batch.num_frames

        # Get compression ratios
        # Default LTX-2 values if not present in config
        vae_spatial_compression_ratio = getattr(
            self.vae_config.arch_config, "spatial_compression_ratio", 32
        )
        vae_temporal_compression_ratio = getattr(
            self.vae_config.arch_config, "temporal_compression_ratio", 8
        )

        latent_height = height // vae_spatial_compression_ratio
        latent_width = width // vae_spatial_compression_ratio
        latent_num_frames = (num_frames - 1) // vae_temporal_compression_ratio + 1

        latents = self._unpack_latents(
            latents,
            latent_num_frames,
            latent_height,
            latent_width,
            self.patch_size,
            self.patch_size_t,
        )

        sample_rate = self.audio_vae_config.arch_config.sample_rate
        hop_length = self.audio_vae_config.arch_config.mel_hop_length
        temporal_compression = self.audio_vae_temporal_compression_ratio
        duration_s = num_frames / batch.fps

        latents_per_second = (
            float(sample_rate) / float(hop_length) / float(temporal_compression)
        )
        audio_num_frames = round(duration_s * latents_per_second)

        num_mel_bins = self.audio_vae_config.arch_config.mel_bins
        mel_compression_ratio = self.audio_vae_mel_compression_ratio
        latent_mel_bins = num_mel_bins // mel_compression_ratio

        audio_latents_mean = getattr(audio_vae, "latents_mean", None)
        audio_latents_std = getattr(audio_vae, "latents_std", None)
        if (
            isinstance(audio_latents_mean, torch.Tensor)
            and isinstance(audio_latents_std, torch.Tensor)
            and audio_latents_mean.numel() == audio_latents_std.numel()
        ):
            audio_latents_mean = audio_latents_mean.to(
                device=audio_latents.device, dtype=audio_latents.dtype
            )
            audio_latents_std = audio_latents_std.to(
                device=audio_latents.device, dtype=audio_latents.dtype
            )
            if audio_latents.ndim == 3:
                if audio_latents.shape[-1] != audio_latents_mean.numel():
                    raise ValueError(
                        f"audio_latents last dim {audio_latents.shape[-1]} "
                        f"does not match audio_vae stats {audio_latents_mean.numel()}"
                    )
                audio_latents = audio_latents * audio_latents_std.view(
                    1, 1, -1
                ) + audio_latents_mean.view(1, 1, -1)
            elif audio_latents.ndim == 2:
                if audio_latents.shape[-1] != audio_latents_mean.numel():
                    raise ValueError(
                        f"audio_latents last dim {audio_latents.shape[-1]} "
                        f"does not match audio_vae stats {audio_latents_mean.numel()}"
                    )
                audio_latents = audio_latents * audio_latents_std.view(
                    1, -1
                ) + audio_latents_mean.view(1, -1)
            else:
                audio_latents = audio_latents * audio_latents_std + audio_latents_mean

        audio_latents = self._unpack_audio_latents(
            audio_latents, audio_num_frames, num_mel_bins=latent_mel_bins
        )

        return latents, audio_latents


@dataclasses.dataclass
class LTX2I2VPipelineConfig(LTX2PipelineConfig):
    task_type: ModelTaskType = ModelTaskType.TI2V
