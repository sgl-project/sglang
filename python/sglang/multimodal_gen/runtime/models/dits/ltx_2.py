from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rotary_pos_embed
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.ltx_2_layers import (
    AdaLayerNormSingle,
    BasicAVTransformerBlock,
    PixArtAlphaTextProjection,
    TransformerArgs,
    TransformerConfig,
)


class LTX2AudioVideoRotaryPosEmbed(nn.Module):
    """
    Video and audio rotary positional embeddings (RoPE) helper for LTX-2.
    Adapted from Diffusers to support coordinate generation.
    """

    def __init__(
        self,
        dim: int,
        patch_size: int = 1,
        patch_size_t: int = 1,
        base_num_frames: int = 20,
        base_height: int = 2048,
        base_width: int = 2048,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        scale_factors: Tuple[int, ...] = (8, 32, 32),
        theta: float = 10000.0,
        causal_offset: int = 1,
        modality: str = "video",
        double_precision: bool = True,
        rope_type: str = "interleaved",
        num_attention_heads: int = 32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.rope_type = rope_type
        self.base_num_frames = base_num_frames
        self.num_attention_heads = num_attention_heads
        self.base_height = base_height
        self.base_width = base_width
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.scale_factors = scale_factors
        self.theta = theta
        self.causal_offset = causal_offset
        self.modality = modality
        self.double_precision = double_precision

    def prepare_video_coords(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        fps: int = 24,
    ) -> torch.Tensor:
        # 1. Generate coordinates in the frame (time) dimension
        grid_f = (
            torch.arange(num_frames, dtype=torch.float32, device=device)
            * self.patch_size_t
        )

        # 2. Generate coordinates in the height dimension
        grid_h = torch.arange(
            0, height, self.patch_size, dtype=torch.float32, device=device
        )

        # 3. Generate coordinates in the width dimension
        grid_w = torch.arange(
            0, width, self.patch_size, dtype=torch.float32, device=device
        )

        # indexing='ij' ensures that the dimensions are kept in order as (frames, height, width)
        grid = torch.meshgrid(grid_f, grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0)  # [3, N_F, N_H, N_W]

        # 2. Get the patch boundaries with respect to the latent video grid
        patch_size = (self.patch_size_t, self.patch_size, self.patch_size)
        patch_size_delta = torch.tensor(
            patch_size, dtype=grid.dtype, device=grid.device
        )
        patch_ends = grid + patch_size_delta.view(3, 1, 1, 1)

        # Combine the start (grid) and end (patch_ends) coordinates along new trailing dimension
        latent_coords = torch.stack([grid, patch_ends], dim=-1)  # [3, N_F, N_H, N_W, 2]
        # Reshape to (batch_size, 3, num_patches, 2)
        latent_coords = latent_coords.flatten(1, 3)
        latent_coords = latent_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        # 3. Calculate the pixel space patch boundaries from the latent boundaries.
        scale_tensor = torch.tensor(self.scale_factors, device=latent_coords.device)
        # Broadcast the VAE scale factors such that they are compatible with latent_coords's shape
        broadcast_shape = [1] * latent_coords.ndim
        broadcast_shape[1] = -1  # This is the (frame, height, width) dim
        # Apply per-axis scaling to convert latent coordinates to pixel space coordinates
        pixel_coords = latent_coords * scale_tensor.view(*broadcast_shape)

        # As the VAE temporal stride for the first frame is 1 instead of self.vae_scale_factors[0], we need to shift
        # and clamp to keep the first-frame timestamps causal and non-negative.
        pixel_coords[:, 0, ...] = (
            pixel_coords[:, 0, ...] + self.causal_offset - self.scale_factors[0]
        ).clamp(min=0)

        # Scale the temporal coordinates by the video FPS
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps

        return pixel_coords

    def prepare_audio_coords(
        self,
        batch_size: int,
        num_frames: int,
        device: torch.device,
        shift: int = 0,
    ) -> torch.Tensor:
        # 1. Generate coordinates in the frame (time) dimension.
        grid_f = torch.arange(
            start=shift,
            end=num_frames + shift,
            step=self.patch_size_t,
            dtype=torch.float32,
            device=device,
        )

        # 2. Calculate start timestamps in seconds with respect to the original spectrogram grid
        audio_scale_factor = self.scale_factors[0]
        # Scale back to mel spectrogram space
        grid_start_mel = grid_f * audio_scale_factor
        # Handle first frame causal offset, ensuring non-negative timestamps
        grid_start_mel = (
            grid_start_mel + self.causal_offset - audio_scale_factor
        ).clip(min=0)
        # Convert mel bins back into seconds
        grid_start_s = grid_start_mel * self.hop_length / self.sampling_rate

        # 3. Calculate start timestamps in seconds with respect to the original spectrogram grid
        grid_end_mel = (grid_f + self.patch_size_t) * audio_scale_factor
        grid_end_mel = (grid_end_mel + self.causal_offset - audio_scale_factor).clip(
            min=0
        )
        grid_end_s = grid_end_mel * self.hop_length / self.sampling_rate

        audio_coords = torch.stack(
            [grid_start_s, grid_end_s], dim=-1
        )  # [num_patches, 2]
        audio_coords = audio_coords.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch_size, num_patches, 2]
        audio_coords = audio_coords.unsqueeze(1)  # [batch_size, 1, num_patches, 2]
        return audio_coords

    def forward(
        self, coords: torch.Tensor, device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = device or coords.device

        # Number of spatiotemporal dimensions (3 for video, 1 (temporal) for audio and cross attn)
        num_pos_dims = coords.shape[1]

        # 1. If the coords are patch boundaries [start, end), use the midpoint of these boundaries as the patch
        # position index
        if coords.ndim == 4:
            coords_start, coords_end = coords.chunk(2, dim=-1)
            coords = (coords_start + coords_end) / 2.0
            coords = coords.squeeze(-1)  # [B, num_pos_dims, num_patches]

        # 2. Get coordinates as a fraction of the base data shape
        if self.modality == "video":
            max_positions = (self.base_num_frames, self.base_height, self.base_width)
        elif self.modality == "audio":
            max_positions = (self.base_num_frames,)
        # [B, num_pos_dims, num_patches] --> [B, num_patches, num_pos_dims]
        grid = torch.stack(
            [coords[:, i] / max_positions[i] for i in range(num_pos_dims)], dim=-1
        ).to(device)
        # Number of spatiotemporal dimensions (3 for video, 1 for audio and cross attn) times 2 for cos, sin
        num_rope_elems = num_pos_dims * 2

        # 3. Create a 1D grid of frequencies for RoPE
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        pow_indices = torch.pow(
            self.theta,
            torch.linspace(
                start=0.0,
                end=1.0,
                steps=self.dim // num_rope_elems,
                dtype=freqs_dtype,
                device=device,
            ),
        )
        freqs = (pow_indices * torch.pi / 2.0).to(dtype=torch.float32)

        # 4. Tensor-vector outer product between pos ids tensor of shape (B, 3, num_patches) and freqs vector of shape
        # (self.dim // num_elems,)
        freqs = (
            grid.unsqueeze(-1) * 2 - 1
        ) * freqs  # [B, num_patches, num_pos_dims, self.dim // num_elems]
        freqs = freqs.transpose(-1, -2).flatten(2)  # [B, num_patches, self.dim // 2]

        # 5. Get real, interleaved (cos, sin) frequencies, padded to self.dim
        if self.rope_type == "interleaved":
            cos_freqs = freqs.cos().repeat_interleave(2, dim=-1)
            sin_freqs = freqs.sin().repeat_interleave(2, dim=-1)

            if self.dim % num_rope_elems != 0:
                cos_padding = torch.ones_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                sin_padding = torch.zeros_like(
                    cos_freqs[:, :, : self.dim % num_rope_elems]
                )
                cos_freqs = torch.cat([cos_padding, cos_freqs], dim=-1)
                sin_freqs = torch.cat([sin_padding, sin_freqs], dim=-1)

        elif self.rope_type == "split":
            expected_freqs = self.dim // 2
            current_freqs = freqs.shape[-1]
            pad_size = expected_freqs - current_freqs
            cos_freq = freqs.cos()
            sin_freq = freqs.sin()

            if pad_size != 0:
                cos_padding = torch.ones_like(cos_freq[:, :, :pad_size])
                sin_padding = torch.zeros_like(sin_freq[:, :, :pad_size])

                cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
                sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)

            # Reshape freqs to be compatible with multi-head attention
            b = cos_freq.shape[0]
            t = cos_freq.shape[1]

            cos_freq = cos_freq.reshape(b, t, self.num_attention_heads, -1)
            sin_freq = sin_freq.reshape(b, t, self.num_attention_heads, -1)

            cos_freqs = torch.swapaxes(cos_freq, 1, 2)  # (B,H,T,D//2)
            sin_freqs = torch.swapaxes(sin_freq, 1, 2)  # (B,H,T,D//2)

        return cos_freqs, sin_freqs


class LTX2VideoTransformer3DModel(BaseDiT):
    """
    LTX-2 Asymmetric Dual-Stream Transformer implementation for SGLang.
    """

    param_names_mapping = LTX2Config().arch_config.param_names_mapping
    _fsdp_shard_conditions = [
        lambda name, module: "transformer_blocks" in name,
    ]
    _compile_conditions: list = []

    def __init__(self, config: LTX2Config, hf_config: dict[str, Any], **kwargs):
        super().__init__(config, hf_config)
        self.config = config.arch_config

        self.in_channels = self.config.in_channels
        self.out_channels = self.config.out_channels
        self.inner_dim = (
            self.config.num_attention_heads * self.config.attention_head_dim
        )

        self.audio_in_channels = self.config.audio_in_channels
        self.audio_out_channels = self.config.audio_out_channels
        self.audio_inner_dim = (
            self.config.audio_num_attention_heads * self.config.audio_attention_head_dim
        )

        # --- Video Stream Components ---
        # Match diffusers/ltx-core naming used by safetensors keys.
        self.proj_in = nn.Linear(self.in_channels, self.inner_dim)
        self.time_embed = AdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels,
            hidden_size=self.inner_dim,
        )

        self.scale_shift_table = nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels)

        # --- Audio Stream Components ---
        self.audio_proj_in = nn.Linear(self.audio_in_channels, self.audio_inner_dim)
        self.audio_time_embed = AdaLayerNormSingle(self.audio_inner_dim)
        self.audio_caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels,
            hidden_size=self.audio_inner_dim,
        )

        self.audio_scale_shift_table = nn.Parameter(
            torch.empty(2, self.audio_inner_dim)
        )
        self.audio_norm_out = nn.RMSNorm(
            self.audio_inner_dim, eps=1e-6, elementwise_affine=False
        )
        self.audio_proj_out = nn.Linear(self.audio_inner_dim, self.audio_out_channels)

        # --- Audio-Video Cross Components ---
        self.av_cross_attn_video_scale_shift = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=4
        )
        self.av_cross_attn_audio_scale_shift = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=4
        )
        self.av_cross_attn_video_a2v_gate = AdaLayerNormSingle(
            self.inner_dim, embedding_coefficient=1
        )
        self.av_cross_attn_audio_v2a_gate = AdaLayerNormSingle(
            self.audio_inner_dim, embedding_coefficient=1
        )

        # --- Blocks ---
        video_config = TransformerConfig(
            dim=self.inner_dim,
            heads=self.config.num_attention_heads,
            d_head=self.config.attention_head_dim,
            context_dim=self.config.cross_attention_dim,
        )
        audio_config = TransformerConfig(
            dim=self.audio_inner_dim,
            heads=self.config.audio_num_attention_heads,
            d_head=self.config.audio_attention_head_dim,
            context_dim=self.config.audio_cross_attention_dim,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicAVTransformerBlock(
                    idx=i,
                    video=video_config,
                    audio=audio_config,
                )
                for i in range(self.config.num_layers)
            ]
        )

        # --- RoPE helper for coordinates ---
        # Used by pipeline/scheduler to generate coordinate grids
        self.rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.config.attention_head_dim,
            patch_size=self.config.patch_size,
            patch_size_t=self.config.patch_size_t,
            scale_factors=(
                self.config.vae_temporal_compression_ratio,
                self.config.vae_spatial_compression_ratio,
                self.config.vae_spatial_compression_ratio,
            ),
            modality="video",
            num_attention_heads=self.config.num_attention_heads,
        )

        self.audio_rope = LTX2AudioVideoRotaryPosEmbed(
            dim=self.config.audio_attention_head_dim,
            patch_size=self.config.patch_size,  # Assuming same spatial patch size? Audio is 1D usually
            patch_size_t=self.config.patch_size_t,  # Audio uses 1D?
            # Audio specific settings
            sampling_rate=16000,  # Default from Diffusers
            hop_length=128,  # Default from Diffusers for LTX-Audio
            scale_factors=(4, 1, 1),  # Temporal compression 4 for audio?
            modality="audio",
            num_attention_heads=self.config.audio_num_attention_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Video latents
        audio_hidden_states: Optional[torch.Tensor] = None,  # Audio latents
        encoder_hidden_states: Optional[torch.Tensor] = None,  # Text embeddings
        timestep: Optional[torch.Tensor] = None,
        video_coords: Optional[torch.Tensor] = None,
        audio_coords: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        audio_timestep = kwargs.get("audio_timestep", timestep)
        audio_encoder_hidden_states = kwargs.get(
            "audio_encoder_hidden_states", encoder_hidden_states
        )
        encoder_attention_mask = kwargs.get("encoder_attention_mask", None)
        audio_encoder_attention_mask = kwargs.get("audio_encoder_attention_mask", None)

        def _rope_dim_list_for_head_dim(d: int) -> list[int]:
            base = int(d) // 6
            return [int(d) - 4 * base, 2 * base, 2 * base]

        video_shape = None
        if hidden_states.ndim == 5:
            video_shape = hidden_states.shape
            b, c, f, h, w = video_shape
            hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)

        x = self.proj_in(hidden_states)
        batch_size = x.shape[0]
        video_pe = None

        # Calculate Video RoPE
        if video_coords is not None:
            video_pe = self.rope(video_coords, device=x.device)
        elif video_shape is not None:
            # Fallback to SGLang default if no coords provided (should prefer coords)
            _, _, f, h, w = video_shape
            head_dim = int(self.inner_dim // self.config.num_attention_heads)
            rope_dim_list = _rope_dim_list_for_head_dim(head_dim)
            cos, sin = get_rotary_pos_embed(
                rope_sizes=(int(f), int(h), int(w)),
                hidden_size=int(self.inner_dim),
                heads_num=int(self.config.num_attention_heads),
                rope_dim_list=rope_dim_list,
                rope_theta=10000.0,
                dtype=torch.float32,
                device=x.device,
            )
            video_pe = (cos, sin)

        audio_shape = None
        x_audio = None
        if audio_hidden_states is not None:
            if (
                audio_hidden_states.ndim == 4
            ):  # [B, C, T, F] (channels, frames, mel_bins)
                audio_shape = audio_hidden_states.shape
                b, c, t, f = audio_shape
                audio_hidden_states = audio_hidden_states.permute(0, 2, 1, 3).reshape(
                    b, t, c * f
                )  # [B, T, C*F] == [B, frames, audio_in_channels]
            x_audio = self.audio_proj_in(audio_hidden_states)

        audio_pe = None
        # Calculate Audio RoPE
        if audio_coords is not None:
            audio_pe = self.audio_rope(audio_coords, device=x.device)
        elif audio_shape is not None and x_audio is not None:
            _, _, t, _ = audio_shape
            audio_head_dim = int(
                self.audio_inner_dim // self.config.audio_num_attention_heads
            )
            audio_rope_dim_list = _rope_dim_list_for_head_dim(audio_head_dim)
            cos, sin = get_rotary_pos_embed(
                rope_sizes=(int(t), 1, 1),
                hidden_size=int(self.audio_inner_dim),
                heads_num=int(self.config.audio_num_attention_heads),
                rope_dim_list=audio_rope_dim_list,
                rope_theta=10000.0,
                dtype=torch.float32,
                device=x.device,
            )
            audio_pe = (cos, sin)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.size(-1)
        )

        temb_audio, audio_embedded_timestep = self.audio_time_embed(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x_audio.dtype if x_audio is not None else x.dtype,
        )
        temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
        audio_embedded_timestep = audio_embedded_timestep.view(
            batch_size, -1, audio_embedded_timestep.size(-1)
        )

        video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x.dtype,
        )
        video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
            batch_size, -1, video_cross_attn_scale_shift.shape[-1]
        )
        video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
            batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
        )

        audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x_audio.dtype if x_audio is not None else x.dtype,
        )
        audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
            audio_timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=x_audio.dtype if x_audio is not None else x.dtype,
        )
        audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
            batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
        )
        audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
            batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
        )

        if encoder_hidden_states is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, x.size(-1)
            )
        if audio_encoder_hidden_states is not None:
            audio_encoder_hidden_states = self.audio_caption_projection(
                audio_encoder_hidden_states
            )
            audio_encoder_hidden_states = audio_encoder_hidden_states.view(
                batch_size,
                -1,
                x_audio.size(-1) if x_audio is not None else self.audio_inner_dim,
            )

        video_args = TransformerArgs(
            x=x,
            timesteps=temb,
            positional_embeddings=video_pe,
            context=encoder_hidden_states,
            context_mask=encoder_attention_mask,
            cross_scale_shift_timestep=video_cross_attn_scale_shift,
            cross_gate_timestep=video_cross_attn_a2v_gate,
        )

        audio_args = TransformerArgs(
            x=x_audio,
            timesteps=temb_audio,
            positional_embeddings=audio_pe,
            context=audio_encoder_hidden_states,
            context_mask=audio_encoder_attention_mask,
            cross_scale_shift_timestep=audio_cross_attn_scale_shift,
            cross_gate_timestep=audio_cross_attn_v2a_gate,
        )

        # 3. Blocks
        for block in self.transformer_blocks:
            video_args, audio_args = block(video_args, audio_args)

        x = video_args.x
        x_audio = audio_args.x

        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = self.norm_out(x)
        x = x * (1 + scale) + shift
        x = self.proj_out(x)

        if x_audio is not None:
            audio_scale_shift_values = (
                self.audio_scale_shift_table[None, None]
                + audio_embedded_timestep[:, :, None]
            )
            audio_shift, audio_scale = (
                audio_scale_shift_values[:, :, 0],
                audio_scale_shift_values[:, :, 1],
            )

            x_audio = self.audio_norm_out(x_audio)
            x_audio = x_audio * (1 + audio_scale) + audio_shift
            x_audio = self.audio_proj_out(x_audio)

        if video_shape is not None:
            b, _, f, h, w = video_shape
            x = x.reshape(b, f, h, w, x.shape[-1]).permute(0, 4, 1, 2, 3).contiguous()

        if audio_shape is not None and x_audio is not None:
            b, c, t, f = audio_shape
            x_audio = x_audio.reshape(b, t, c, f).permute(0, 2, 1, 3).contiguous()

        return x, x_audio


EntryClass = LTX2VideoTransformer3DModel
