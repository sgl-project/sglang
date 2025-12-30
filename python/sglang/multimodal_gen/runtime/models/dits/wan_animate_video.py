# Adapted from diffusers transformer_wan_animate.py for SGLang

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.configs.models.dits import WanVideoConfig
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_sp_world_size
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift, RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding import NDRotaryEmbedding
from sglang.multimodal_gen.runtime.layers.visual_embedding import PatchEmbed
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.wanvideo import (
    WanTimeTextImageEmbedding,
    WanTransformerBlock,
)
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


# Default channel sizes for the motion encoder at different resolutions
WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES = {
    "4": 512,
    "8": 512,
    "16": 512,
    "32": 512,
    "64": 256,
    "128": 128,
    "256": 64,
    "512": 32,
    "1024": 16,
}


class FusedLeakyReLU(nn.Module):
    """
    Fused LeakyRelu with scale factor and channel-wise bias.
    """

    def __init__(
        self,
        negative_slope: float = 0.2,
        scale: float = 2**0.5,
        bias_channels: int | None = None,
    ):
        super().__init__()
        self.negative_slope = negative_slope
        self.scale = scale
        self.channels = bias_channels

        if self.channels is not None:
            self.bias = nn.Parameter(torch.zeros(self.channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if self.bias is not None:
            # Expand self.bias to have all singleton dims except at channel_dim
            expanded_shape = [1] * x.ndim
            expanded_shape[channel_dim] = self.bias.shape[0]
            bias = self.bias.reshape(*expanded_shape)
            x = x + bias
        return F.leaky_relu(x, self.negative_slope) * self.scale


class MotionConv2d(nn.Module):
    """
    Conv2d layer with optional blur and fused activation for motion encoding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        blur_kernel: tuple[int, ...] | None = None,
        blur_upsample_factor: int = 1,
        use_activation: bool = True,
    ):
        super().__init__()
        self.use_activation = use_activation
        self.in_channels = in_channels

        # Handle blurring (applying a FIR filter with the given kernel) if available
        self.blur = False
        if blur_kernel is not None:
            p = (len(blur_kernel) - stride) + (kernel_size - 1)
            self.blur_padding = ((p + 1) // 2, p // 2)

            kernel = torch.tensor(blur_kernel, dtype=torch.float32, device="cuda")
            # Convert kernel to 2D if necessary
            if kernel.ndim == 1:
                kernel = kernel[None, :] * kernel[:, None]
            # Normalize kernel
            kernel = kernel / kernel.sum()
            if blur_upsample_factor > 1:
                kernel = kernel * (blur_upsample_factor**2)
            self.register_buffer("blur_kernel", kernel, persistent=False)
            self.blur = True

        # Main Conv2d parameters (with scale factor)
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channels * kernel_size**2)

        self.stride = stride
        self.padding = padding

        # If using an activation function, the bias will be fused into the activation
        if bias and not self.use_activation:
            self.conv_bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.conv_bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_channels)
        else:
            self.act_fn = None

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        # Apply blur if using
        if self.blur:
            expanded_kernel = self.blur_kernel[None, None, :, :].expand(
                self.in_channels, 1, -1, -1
            )
            x = F.conv2d(
                x, expanded_kernel, padding=self.blur_padding, groups=self.in_channels
            )

        # Main Conv2D with scaling
        x = F.conv2d(
            x,
            self.weight * self.scale,
            bias=self.conv_bias,
            stride=self.stride,
            padding=self.padding,
        )

        # Activation with fused bias, if using
        if self.use_activation:
            x = self.act_fn(x, channel_dim=channel_dim)
        return x


class MotionLinear(nn.Module):
    """
    Linear layer with optional fused activation for motion encoding.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        use_activation: bool = False,
    ):
        super().__init__()
        self.use_activation = use_activation

        # Linear weight with scale factor
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        self.scale = 1 / math.sqrt(in_dim)

        # If an activation is present, the bias will be fused to it
        if bias and not self.use_activation:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        if self.use_activation:
            self.act_fn = FusedLeakyReLU(bias_channels=out_dim)
        else:
            self.act_fn = None

    def forward(self, input: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        out = F.linear(input, self.weight * self.scale, bias=self.bias)
        if self.use_activation:
            out = self.act_fn(out, channel_dim=channel_dim)
        return out


class MotionEncoderResBlock(nn.Module):
    """
    Residual block for motion encoder with downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        kernel_size_skip: int = 1,
        blur_kernel: tuple[int, ...] = (1, 3, 3, 1),
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.downsample_factor = downsample_factor

        # 3 x 3 Conv + fused leaky ReLU
        self.conv1 = MotionConv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2,
            use_activation=True,
        )

        # 3 x 3 Conv that downsamples 2x + fused leaky ReLU
        self.conv2 = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.downsample_factor,
            padding=0,
            blur_kernel=blur_kernel,
            use_activation=True,
        )

        # 1 x 1 Conv that downsamples 2x in skip connection
        self.conv_skip = MotionConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_skip,
            stride=self.downsample_factor,
            padding=0,
            bias=False,
            blur_kernel=blur_kernel,
            use_activation=False,
        )

    def forward(self, x: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        x_out = self.conv1(x, channel_dim)
        x_out = self.conv2(x_out, channel_dim)

        x_skip = self.conv_skip(x, channel_dim)

        x_out = (x_out + x_skip) / math.sqrt(2)
        return x_out


class WanAnimateMotionEncoder(nn.Module):
    """
    Motion encoder that extracts motion features from face video frames.
    Uses a convolutional appearance encoder followed by linear motion encoding
    with QR-based Linear Motion Decomposition.
    """

    def __init__(
        self,
        size: int = 512,
        style_dim: int = 512,
        motion_dim: int = 20,
        out_dim: int = 512,
        motion_blocks: int = 5,
        channels: dict[str, int] | None = None,
    ):
        super().__init__()
        self.size = size

        # Appearance encoder: conv layers
        if channels is None:
            channels = WAN_ANIMATE_MOTION_ENCODER_CHANNEL_SIZES

        self.conv_in = MotionConv2d(3, channels[str(size)], 1, use_activation=True)

        self.res_blocks = nn.ModuleList()
        in_channels = channels[str(size)]
        log_size = int(math.log(size, 2))
        for i in range(log_size, 2, -1):
            out_channels = channels[str(2 ** (i - 1))]
            self.res_blocks.append(MotionEncoderResBlock(in_channels, out_channels))
            in_channels = out_channels

        self.conv_out = MotionConv2d(
            in_channels, style_dim, 4, padding=0, bias=False, use_activation=False
        )

        # Motion encoder: linear layers
        linears = [MotionLinear(style_dim, style_dim) for _ in range(motion_blocks - 1)]
        linears.append(MotionLinear(style_dim, motion_dim))
        self.motion_network = nn.ModuleList(linears)

        self.motion_synthesis_weight = nn.Parameter(torch.randn(out_dim, motion_dim))

    def forward(self, face_image: torch.Tensor, channel_dim: int = 1) -> torch.Tensor:
        if (face_image.shape[-2] != self.size) or (face_image.shape[-1] != self.size):
            raise ValueError(
                f"Face pixel values has resolution ({face_image.shape[-1]}, {face_image.shape[-2]}) but is expected"
                f" to have resolution ({self.size}, {self.size})"
            )

        # Appearance encoding through convs
        face_image = self.conv_in(face_image, channel_dim)
        for block in self.res_blocks:
            face_image = block(face_image, channel_dim)
        face_image = self.conv_out(face_image, channel_dim)
        motion_feat = face_image.squeeze(-1).squeeze(-1)

        # Motion feature extraction
        for linear_layer in self.motion_network:
            motion_feat = linear_layer(motion_feat, channel_dim=channel_dim)

        # Motion synthesis via Linear Motion Decomposition
        weight = self.motion_synthesis_weight + 1e-8
        # Upcast the QR orthogonalization operation to FP32
        original_motion_dtype = motion_feat.dtype
        motion_feat = motion_feat.to(torch.float32)
        weight = weight.to(torch.float32)

        Q = torch.linalg.qr(weight)[0].to(device=motion_feat.device)

        motion_feat_diag = torch.diag_embed(motion_feat)  # Alpha, diagonal matrix
        motion_decomposition = torch.matmul(motion_feat_diag, Q.T)
        motion_vec = torch.sum(motion_decomposition, dim=1)

        motion_vec = motion_vec.to(dtype=original_motion_dtype)

        return motion_vec


class WanAnimateFaceEncoder(nn.Module):
    """
    Face encoder that processes motion features through temporal 1D convolutions.
    Produces temporally-aligned features for cross-attention with video latents.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 1024,
        num_heads: int = 4,
        kernel_size: int = 3,
        eps: float = 1e-6,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.time_causal_padding = (kernel_size - 1, 0)
        self.pad_mode = pad_mode

        self.act = nn.SiLU()

        self.conv1_local = nn.Conv1d(
            in_dim, hidden_dim * num_heads, kernel_size=kernel_size, stride=1
        )
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, stride=2)

        self.norm1 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(hidden_dim, eps, elementwise_affine=False)

        self.out_proj = nn.Linear(hidden_dim, out_dim)

        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Reshape to channels-first to apply causal Conv1d over frame dim
        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv1_local(x)  # [B, C, T_padded] --> [B, N * C, T]
        x = x.unflatten(1, (self.num_heads, -1)).flatten(0, 1)  # [B * N, C, T]
        # Reshape back to channels-last to apply LayerNorm over channel dim
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.norm2(x)
        x = self.act(x)

        x = x.permute(0, 2, 1)
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.norm3(x)
        x = self.act(x)

        x = self.out_proj(x)
        x = x.unflatten(0, (batch_size, -1)).permute(
            0, 2, 1, 3
        )  # [B * N, T, C_out] --> [B, T, N, C_out]

        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1).to(
            device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, padding], dim=-2)  # [B, T, N, C_out] --> [B, T, N + 1, C_out]

        return x


class WanAnimateFaceBlockCrossAttention(nn.Module):
    """
    Temporally-aligned cross attention with the face motion signal in the Wan Animate Face Blocks.
    Uses pre-normalization and reshapes Q to align temporally with motion K/V.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-6,
        cross_attention_dim_head: int | None = None,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
    ):
        super().__init__()
        self.inner_dim = dim_head * num_heads
        self.num_heads = num_heads
        self.cross_attention_head_dim = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * num_heads
        )

        # 1. Pre-Attention Norms for the hidden_states (video latents) and encoder_hidden_states (motion vector)
        self.pre_norm_q = nn.LayerNorm(dim, eps, elementwise_affine=False)
        self.pre_norm_kv = nn.LayerNorm(dim, eps, elementwise_affine=False)

        # 2. QKV and Output Projections
        self.to_q = ReplicatedLinear(dim, self.inner_dim, bias=True)
        self.to_k = ReplicatedLinear(dim, self.kv_inner_dim, bias=True)
        self.to_v = ReplicatedLinear(dim, self.kv_inner_dim, bias=True)
        self.to_out = ReplicatedLinear(self.inner_dim, dim, bias=True)

        # 3. QK Norm (applied after the reshape, so only over dim_head rather than dim_head * heads)
        self.norm_q = RMSNorm(dim_head, eps=eps)
        self.norm_k = RMSNorm(dim_head, eps=eps)

        # 4. Attention layer
        self.attn = USPAttention(
            num_heads=num_heads,
            head_size=dim_head,
            causal=False,
            supported_attention_backends=supported_attention_backends,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: Video latent features [B, S, C] where S is sequence length
            encoder_hidden_states: Motion features [B, T, N, C] where T is temporal length,
                                   N is num_heads + 1
            attention_mask: Optional mask for attention
        """
        # Pre-normalize
        hidden_states = self.pre_norm_q(hidden_states)
        encoder_hidden_states = self.pre_norm_kv(encoder_hidden_states)

        B, T, N, C = encoder_hidden_states.shape

        # Compute query, key, value
        query, _ = self.to_q(hidden_states)
        key, _ = self.to_k(encoder_hidden_states)
        value, _ = self.to_v(encoder_hidden_states)

        query = query.unflatten(2, (self.num_heads, -1))  # [B, S, H, D]
        key = key.view(B, T, N, self.num_heads, -1)  # [B, T, N, H, D_kv]
        value = value.view(B, T, N, self.num_heads, -1)

        # Apply QK norm
        query = self.norm_q(query)
        key = self.norm_k(key)

        # Reshape for temporal alignment:
        # Q: [B, S, H, D] -> [B * T, S / T, H, D]
        # K, V: [B, T, N, H, D_kv] -> [B * T, N, H, D_kv]
        query = query.unflatten(1, (T, -1)).flatten(0, 1)  # [B * T, S / T, H, D]
        key = key.flatten(0, 1)  # [B * T, N, H, D_kv]
        value = value.flatten(0, 1)

        # Compute attention
        hidden_states = self.attn(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.unflatten(0, (B, T)).flatten(1, 2)

        hidden_states, _ = self.to_out(hidden_states)

        if attention_mask is not None:
            # attention_mask is assumed to be a multiplicative mask
            attention_mask = attention_mask.flatten(start_dim=1)
            hidden_states = hidden_states * attention_mask

        return hidden_states


class WanAnimateTransformer3DModel(BaseDiT):
    """
    Transformer model for Wan2.2-Animate video generation.
    Extends WanTransformer3DModel with motion encoder, face encoder, and face adapter blocks.
    """

    _fsdp_shard_conditions = WanVideoConfig()._fsdp_shard_conditions
    _compile_conditions = WanVideoConfig()._compile_conditions
    _supported_attention_backends = WanVideoConfig()._supported_attention_backends
    param_names_mapping = WanVideoConfig().param_names_mapping
    reverse_param_names_mapping = WanVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = WanVideoConfig().lora_param_names_mapping

    def __init__(self, config: WanVideoConfig, hf_config: dict[str, Any]) -> None:
        super().__init__(config=config, hf_config=hf_config)

        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_channels_latents = config.num_channels_latents
        self.patch_size = config.patch_size

        # Wan Animate specific config with defaults
        self.latent_channels = getattr(config, "latent_channels", 16)
        self.in_channels = getattr(
            config, "in_channels", 2 * self.latent_channels + 4
        )  # 36 for Animate
        self.out_channels = getattr(config, "out_channels", self.latent_channels)
        self.motion_encoder_size = getattr(config, "motion_encoder_size", 512)
        self.motion_style_dim = getattr(config, "motion_style_dim", 512)
        self.motion_dim = getattr(config, "motion_dim", 20)
        self.motion_encoder_dim = getattr(config, "motion_encoder_dim", 512)
        self.face_encoder_hidden_dim = getattr(config, "face_encoder_hidden_dim", 1024)
        self.face_encoder_num_heads = getattr(config, "face_encoder_num_heads", 4)
        self.inject_face_latents_blocks = getattr(
            config, "inject_face_latents_blocks", 5
        )
        self.motion_encoder_batch_size = getattr(config, "motion_encoder_batch_size", 8)

        # 1. Patch & position embedding
        self.patch_embedding = PatchEmbed(
            in_chans=self.in_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )
        self.pose_patch_embedding = PatchEmbed(
            in_chans=self.latent_channels,
            embed_dim=inner_dim,
            patch_size=config.patch_size,
            flatten=False,
        )

        # 2. Condition embeddings
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=config.freq_dim,
            text_embed_dim=config.text_dim,
            image_embed_dim=config.image_dim,
        )

        # 3. Motion encoder
        self.motion_encoder = WanAnimateMotionEncoder(
            size=self.motion_encoder_size,
            style_dim=self.motion_style_dim,
            motion_dim=self.motion_dim,
            out_dim=self.motion_encoder_dim,
        )

        # 4. Face encoder
        self.face_encoder = WanAnimateFaceEncoder(
            in_dim=self.motion_encoder_dim,
            out_dim=inner_dim,
            hidden_dim=self.face_encoder_hidden_dim,
            num_heads=self.face_encoder_num_heads,
        )

        # 5. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    config.ffn_dim,
                    config.num_attention_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                    config.added_kv_proj_dim,
                    self._supported_attention_backends,
                    prefix=f"{config.prefix}.blocks.{i}",
                )
                for i in range(config.num_layers)
            ]
        )

        # 6. Face adapter blocks (applied every inject_face_latents_blocks)
        num_face_adapters = config.num_layers // self.inject_face_latents_blocks
        self.face_adapter = nn.ModuleList(
            [
                WanAnimateFaceBlockCrossAttention(
                    dim=inner_dim,
                    num_heads=config.num_attention_heads,
                    dim_head=inner_dim // config.num_attention_heads,
                    eps=config.eps,
                    cross_attention_dim_head=inner_dim // config.num_attention_heads,
                    supported_attention_backends=self._supported_attention_backends,
                )
                for _ in range(num_face_adapters)
            ]
        )

        # 7. Output norm & projection
        self.norm_out = LayerNormScaleShift(
            inner_dim,
            norm_type="layer",
            eps=config.eps,
            elementwise_affine=False,
            dtype=torch.float32,
            compute_dtype=torch.float32,
        )
        self.proj_out = nn.Linear(
            inner_dim, self.out_channels * math.prod(config.patch_size)
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

        self.__post_init__()

        # misc
        self.sp_size = get_sp_world_size()

        # Get rotary embeddings
        d = self.hidden_size // self.num_attention_heads
        self.rope_dim_list = [d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)]

        self.rotary_emb = NDRotaryEmbedding(
            rope_dim_list=self.rope_dim_list,
            rope_theta=10000,
            dtype=torch.float32 if current_platform.is_mps() else torch.float64,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        pose_hidden_states: torch.Tensor | None = None,
        face_pixel_values: torch.Tensor | None = None,
        motion_encode_batch_size: int | None = None,
        guidance: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Wan2.2-Animate transformer model.

        Args:
            hidden_states: Input noisy video latents [B, 2C + 4, T + 1, H, W]
            encoder_hidden_states: Text embeddings from text encoder
            timestep: Current timestep in the denoising loop
            encoder_hidden_states_image: CLIP visual features of the reference image
            pose_hidden_states: Pose video latents [B, C, T, H, W]
            face_pixel_values: Face video in pixel space [B, C', S, H', W']
            motion_encode_batch_size: Batch size for motion encoding
        """
        orig_dtype = hidden_states.dtype

        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        # Validate shapes
        if pose_hidden_states is not None:
            if pose_hidden_states.shape[2] + 1 != hidden_states.shape[2]:
                raise ValueError(
                    f"pose_hidden_states frame dim is {pose_hidden_states.shape[2]} but must be one less than "
                    f"hidden_states frame dim: {hidden_states.shape[2]}"
                )

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        # 1. Rotary position embedding
        freqs_cos, freqs_sin = self.rotary_emb.forward_from_grid(
            (
                post_patch_num_frames * self.sp_size,
                post_patch_height,
                post_patch_width,
            ),
            shard_dim=0,
            start_frame=0,
            device=hidden_states.device,
        )
        assert freqs_cos.dtype == torch.float32
        freqs_cis = (freqs_cos.float(), freqs_sin.float())

        # 2. Patch embedding
        hidden_states = self.patch_embedding(hidden_states)
        pose_hidden_states = self.pose_patch_embedding(pose_hidden_states)

        # Add pose embeddings to frames 1+ (skip first frame which is the reference)
        hidden_states[:, :, 1:] = hidden_states[:, :, 1:] + pose_hidden_states

        # Flatten for transformer
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

        # 3. Condition embeddings (time, text, image)
        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(
                timestep, encoder_hidden_states, encoder_hidden_states_image
            )
        )
        # batch_size, 6, inner_dim
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        encoder_hidden_states = (
            encoder_hidden_states.to(orig_dtype)
            if current_platform.is_mps()
            else encoder_hidden_states
        )

        # 4. Get motion features from the face video
        if face_pixel_values is not None:
            batch_size_face, channels, num_face_frames, face_height, face_width = (
                face_pixel_values.shape
            )
            # Rearrange from (B, C, T, H, W) to (B*T, C, H, W)
            face_pixel_values = face_pixel_values.permute(0, 2, 1, 3, 4).reshape(
                -1, channels, face_height, face_width
            )

            # Extract motion features using motion encoder (batched for memory efficiency)
            motion_encode_batch_size = (
                motion_encode_batch_size or self.motion_encoder_batch_size
            )
            face_batches = torch.split(face_pixel_values, motion_encode_batch_size)
            motion_vec_batches = []
            for face_batch in face_batches:
                motion_vec_batch = self.motion_encoder(face_batch)
                motion_vec_batches.append(motion_vec_batch)
            motion_vec = torch.cat(motion_vec_batches)
            motion_vec = motion_vec.view(batch_size_face, num_face_frames, -1)

            # Get face features from the motion vector
            motion_vec = self.face_encoder(motion_vec)

            # Add padding at the beginning (prepend zeros for first frame)
            pad_face = torch.zeros_like(motion_vec[:, :1])
            motion_vec = torch.cat([pad_face, motion_vec], dim=1)
        else:
            motion_vec = None

        # 5. Transformer blocks with face adapter integration
        for block_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states, encoder_hidden_states, timestep_proj, freqs_cis
            )

            # Face adapter integration: apply after every inject_face_latents_blocks blocks
            if (
                motion_vec is not None
                and block_idx % self.inject_face_latents_blocks == 0
            ):
                face_adapter_block_idx = block_idx // self.inject_face_latents_blocks
                face_adapter_output = self.face_adapter[face_adapter_block_idx](
                    hidden_states, motion_vec
                )
                # Handle potential device mismatch with model parallelism
                face_adapter_output = face_adapter_output.to(
                    device=hidden_states.device
                )
                hidden_states = face_adapter_output + hidden_states

        # 6. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)

        hidden_states = self.norm_out(hidden_states, shift, scale)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        return output


EntryClass = WanAnimateTransformer3DModel
