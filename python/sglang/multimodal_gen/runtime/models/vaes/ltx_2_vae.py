from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE


class PerChannelRMSNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.

    For each element along the chosen dimension, this layer normalizes the tensor by the root-mean-square of its values
    across that dimension:

        y = x / sqrt(mean(x^2, dim=dim, keepdim=True) + eps)
    """

    def __init__(self, channel_dim: int = 1, eps: float = 1e-8) -> None:
        """
        Args:
            dim: Dimension along which to compute the RMS (typically channels).
            eps: Small constant added for numerical stability.
        """
        super().__init__()
        self.channel_dim = channel_dim
        self.eps = eps

    def forward(
        self, x: torch.Tensor, channel_dim: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply RMS normalization along the configured dimension.
        """
        channel_dim = channel_dim or self.channel_dim
        # Compute mean of squared values along `dim`, keep dimensions for broadcasting.
        mean_sq = torch.mean(x**2, dim=self.channel_dim, keepdim=True)
        # Normalize by the root-mean-square (RMS).
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


# Like LTXCausalConv3d, but whether causal inference is performed can be specified at runtime
class LTX2VideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size, kernel_size)
        )

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if causal:
            pad_left = hidden_states[:, :, :1, :, :].repeat(
                (1, 1, time_kernel_size - 1, 1, 1)
            )
            hidden_states = torch.concatenate([pad_left, hidden_states], dim=2)
        else:
            pad_left = hidden_states[:, :, :1, :, :].repeat(
                (1, 1, (time_kernel_size - 1) // 2, 1, 1)
            )
            pad_right = hidden_states[:, :, -1:, :, :].repeat(
                (1, 1, (time_kernel_size - 1) // 2, 1, 1)
            )
            hidden_states = torch.concatenate(
                [pad_left, hidden_states, pad_right], dim=2
            )

        hidden_states = self.conv(hidden_states)
        return hidden_states


# Like LTXVideoResnetBlock3d, but uses new causal Conv3d, normal Conv3d for the conv_shortcut, and the spatial padding
# mode is configurable
class LTX2VideoResnetBlock3d(nn.Module):
    r"""
    A 3D ResNet block used in the LTX 2.0 audiovisual model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        elementwise_affine (`bool`, defaults to `False`):
            Whether to enable elementwise affinity in the normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = PerChannelRMSNorm()
        self.conv1 = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm2 = PerChannelRMSNorm()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = LTX2VideoCausalConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(
                in_channels, eps=eps, elementwise_affine=True, bias=True
            )
            # LTX 2.0 uses a normal nn.Conv3d here rather than LTXVideoCausalConv3d
            self.conv_shortcut = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
            )

        self.per_channel_scale1 = None
        self.per_channel_scale2 = None
        if inject_noise:
            self.per_channel_scale1 = nn.Parameter(torch.zeros(in_channels, 1, 1))
            self.per_channel_scale2 = nn.Parameter(torch.zeros(in_channels, 1, 1))

        self.scale_shift_table = None
        if timestep_conditioning:
            self.scale_shift_table = nn.Parameter(
                torch.randn(4, in_channels) / in_channels**0.5
            )

    def forward(
        self,
        inputs: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.scale_shift_table is not None:
            temb = (
                temb.unflatten(1, (4, -1))
                + self.scale_shift_table[None, ..., None, None, None]
            )
            shift_1, scale_1, shift_2, scale_2 = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale_1) + shift_1

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states, causal=causal)

        if self.per_channel_scale1 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape,
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = (
                hidden_states
                + (spatial_noise * self.per_channel_scale1)[None, :, None, ...]
            )

        hidden_states = self.norm2(hidden_states)

        if self.scale_shift_table is not None:
            hidden_states = hidden_states * (1 + scale_2) + shift_2

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, causal=causal)

        if self.per_channel_scale2 is not None:
            spatial_shape = hidden_states.shape[-2:]
            spatial_noise = torch.randn(
                spatial_shape,
                generator=generator,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )[None]
            hidden_states = (
                hidden_states
                + (spatial_noise * self.per_channel_scale2)[None, :, None, ...]
            )

        if self.norm3 is not None:
            inputs = self.norm3(inputs.movedim(1, -1)).movedim(-1, 1)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


# Like LTX 1.0 LTXVideoDownsampler3d, but uses new causal Conv3d
class LTXVideoDownsampler3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.group_size = (
            in_channels * stride[0] * stride[1] * stride[2]
        ) // out_channels

        out_channels = out_channels // (
            self.stride[0] * self.stride[1] * self.stride[2]
        )

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        hidden_states = torch.cat(
            [hidden_states[:, :, : self.stride[0] - 1], hidden_states], dim=2
        )

        residual = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        residual = residual.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        residual = residual.unflatten(1, (-1, self.group_size))
        residual = residual.mean(dim=2)

        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = (
            hidden_states.unflatten(4, (-1, self.stride[2]))
            .unflatten(3, (-1, self.stride[1]))
            .unflatten(2, (-1, self.stride[0]))
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 7, 2, 4, 6).flatten(1, 4)
        hidden_states = hidden_states + residual

        return hidden_states


# Like LTX 1.0 LTXVideoUpsampler3d, but uses new causal Conv3d
class LTXVideoUpsampler3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stride: Union[int, Tuple[int, int, int]] = 1,
        residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.residual = residual
        self.upscale_factor = upscale_factor

        out_channels = (
            in_channels * stride[0] * stride[1] * stride[2]
        ) // upscale_factor

        self.conv = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor, causal: bool = True) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        if self.residual:
            residual = hidden_states.reshape(
                batch_size,
                -1,
                self.stride[0],
                self.stride[1],
                self.stride[2],
                num_frames,
                height,
                width,
            )
            residual = (
                residual.permute(0, 1, 5, 2, 6, 3, 7, 4)
                .flatten(6, 7)
                .flatten(4, 5)
                .flatten(2, 3)
            )
            repeats = (
                self.stride[0] * self.stride[1] * self.stride[2]
            ) // self.upscale_factor
            residual = residual.repeat(1, repeats, 1, 1, 1)
            residual = residual[:, :, self.stride[0] - 1 :]

        hidden_states = self.conv(hidden_states, causal=causal)
        hidden_states = hidden_states.reshape(
            batch_size,
            -1,
            self.stride[0],
            self.stride[1],
            self.stride[2],
            num_frames,
            height,
            width,
        )
        hidden_states = (
            hidden_states.permute(0, 1, 5, 2, 6, 3, 7, 4)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )
        hidden_states = hidden_states[:, :, self.stride[0] - 1 :]

        if self.residual:
            hidden_states = hidden_states + residual

        return hidden_states


# Like LTX 1.0 LTXVideo095DownBlock3D, but with the updated LTX2VideoResnetBlock3d
class LTX2VideoDownBlock3D(nn.Module):
    r"""
    Down block used in the LTXVideo model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        downsample_type: str = "conv",
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.downsamplers = None
        if spatio_temporal_scale:
            self.downsamplers = nn.ModuleList()

            if downsample_type == "conv":
                self.downsamplers.append(
                    LTX2VideoCausalConv3d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatial":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(1, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "temporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 1, 1),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )
            elif downsample_type == "spatiotemporal":
                self.downsamplers.append(
                    LTXVideoDownsampler3d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=(2, 2, 2),
                        spatial_padding_mode=spatial_padding_mode,
                    )
                )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        r"""Forward method of the `LTXDownBlock3D` class."""

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, generator, causal
                )
            else:
                hidden_states = resnet(hidden_states, temb, generator, causal=causal)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, causal=causal)

        return hidden_states


# Adapted from diffusers.models.autoencoders.autoencoder_kl_cogvideox.CogVideoMidBlock3d
# Like LTX 1.0 LTXVideoMidBlock3d, but with the updated LTX2VideoResnetBlock3d
class LTX2VideoMidBlock3d(nn.Module):
    r"""
    A middle block used in the LTXVideo model.

    Args:
        in_channels (`int`):
            Number of input channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        spatial_padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        r"""Forward method of the `LTXMidBlock3D` class."""

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, generator, causal
                )
            else:
                hidden_states = resnet(hidden_states, temb, generator, causal=causal)

        return hidden_states


# Like LTXVideoUpBlock3d but with no conv_in and the updated LTX2VideoResnetBlock3d
class LTX2VideoUpBlock3d(nn.Module):
    r"""
    Up block used in the LTXVideo model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        inject_noise: bool = False,
        timestep_conditioning: bool = False,
        upsample_residual: bool = False,
        upscale_factor: int = 1,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.time_embedder = None
        if timestep_conditioning:
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                in_channels * 4, 0
            )

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = LTX2VideoResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                inject_noise=inject_noise,
                timestep_conditioning=timestep_conditioning,
                spatial_padding_mode=spatial_padding_mode,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList(
                [
                    LTXVideoUpsampler3d(
                        out_channels * upscale_factor,
                        stride=(2, 2, 2),
                        residual=upsample_residual,
                        upscale_factor=upscale_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                LTX2VideoResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    inject_noise=inject_noise,
                    timestep_conditioning=timestep_conditioning,
                    spatial_padding_mode=spatial_padding_mode,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states, temb, generator, causal=causal)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, causal=causal)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    resnet, hidden_states, temb, generator, causal
                )
            else:
                hidden_states = resnet(hidden_states, temb, generator, causal=causal)

        return hidden_states


# Like LTX 1.0 LTXVideoEncoder3d but with different default args - the spatiotemporal downsampling pattern is
# different, as is the layers_per_block (the 2.0 VAE is bigger)
class LTX2VideoEncoder3d(nn.Module):
    r"""
    The `LTXVideoEncoder3d` layer of a variational autoencoder that encodes input video samples to its latent
    representation.

    Args:
        in_channels (`int`, defaults to 3):
            Number of input channels.
        out_channels (`int`, defaults to 128):
            Number of latent channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(256, 512, 1024, 2048)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, True)`:
            Whether a block should contain spatio-temporal downscaling layers or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 6, 6, 2, 2)`):
            The number of layers per block.
        downsample_type (`Tuple[str, ...]`, defaults to `("spatial", "temporal", "spatiotemporal", "spatiotemporal")`):
            The spatiotemporal downsampling pattern per block. Per-layer values can be
                - `"spatial"` (downsample spatial dims by 2x)
                - `"temporal"` (downsample temporal dim by 2x)
                - `"spatiotemporal"` (downsample both spatial and temporal dims by 2x)
        patch_size (`int`, defaults to `4`):
            The size of spatial patches.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 128,
        block_out_channels: Tuple[int, ...] = (256, 512, 1024, 2048),
        down_block_types: Tuple[str, ...] = (
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
            "LTX2VideoDownBlock3D",
        ),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, True),
        layers_per_block: Tuple[int, ...] = (4, 6, 6, 2, 2),
        downsample_type: Tuple[str, ...] = (
            "spatial",
            "temporal",
            "spatiotemporal",
            "spatiotemporal",
        ),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = True,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.in_channels = in_channels * patch_size**2
        self.is_causal = is_causal

        output_channel = out_channels

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=self.in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # down blocks
        num_block_out_channels = len(block_out_channels)
        self.down_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            if down_block_types[i] == "LTX2VideoDownBlock3D":
                down_block = LTX2VideoDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    num_layers=layers_per_block[i],
                    resnet_eps=resnet_norm_eps,
                    spatio_temporal_scale=spatio_temporal_scaling[i],
                    downsample_type=downsample_type[i],
                    spatial_padding_mode=spatial_padding_mode,
                )
            else:
                raise ValueError(f"Unknown down block type: {down_block_types[i]}")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[-1],
            resnet_eps=resnet_norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=out_channels + 1,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.gradient_checkpointing = False

    def forward(
        self, hidden_states: torch.Tensor, causal: Optional[bool] = None
    ) -> torch.Tensor:
        r"""The forward method of the `LTXVideoEncoder3d` class."""

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p
        post_patch_width = width // p
        causal = causal or self.is_causal

        hidden_states = hidden_states.reshape(
            batch_size,
            num_channels,
            post_patch_num_frames,
            p_t,
            post_patch_height,
            p,
            post_patch_width,
            p,
        )
        # Thanks for driving me insane with the weird patching order :(
        hidden_states = hidden_states.permute(0, 1, 3, 7, 5, 2, 4, 6).flatten(1, 4)
        hidden_states = self.conv_in(hidden_states, causal=causal)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for down_block in self.down_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    down_block, hidden_states, None, None, causal
                )

            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states, None, None, causal
            )
        else:
            for down_block in self.down_blocks:
                hidden_states = down_block(hidden_states, causal=causal)

            hidden_states = self.mid_block(hidden_states, causal=causal)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        last_channel = hidden_states[:, -1:]
        last_channel = last_channel.repeat(1, hidden_states.size(1) - 2, 1, 1, 1)
        hidden_states = torch.cat([hidden_states, last_channel], dim=1)

        return hidden_states


# Like LTX 1.0 LTXVideoDecoder3d, but has only 3 symmetric up blocks which are causal and residual with upsample_factor 2
class LTX2VideoDecoder3d(nn.Module):
    r"""
    The `LTXVideoDecoder3d` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, defaults to 128):
            Number of latent channels.
        out_channels (`int`, defaults to 3):
            Number of output channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal upscaling layers or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        patch_size (`int`, defaults to `4`):
            The size of spatial patches.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
        is_causal (`bool`, defaults to `False`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
        timestep_conditioning (`bool`, defaults to `False`):
            Whether to condition the model on timesteps.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (256, 512, 1024),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True),
        layers_per_block: Tuple[int, ...] = (5, 5, 5, 5),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        inject_noise: Tuple[bool, ...] = (False, False, False),
        timestep_conditioning: bool = False,
        upsample_residual: Tuple[bool, ...] = (True, True, True),
        upsample_factor: Tuple[bool, ...] = (2, 2, 2),
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels * patch_size**2
        self.is_causal = is_causal

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        inject_noise = tuple(reversed(inject_noise))
        upsample_residual = tuple(reversed(upsample_residual))
        upsample_factor = tuple(reversed(upsample_factor))
        output_channel = block_out_channels[0]

        self.conv_in = LTX2VideoCausalConv3d(
            in_channels=in_channels,
            out_channels=output_channel,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        self.mid_block = LTX2VideoMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            inject_noise=inject_noise[0],
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel // upsample_factor[i]
            output_channel = block_out_channels[i] // upsample_factor[i]

            up_block = LTX2VideoUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                inject_noise=inject_noise[i + 1],
                timestep_conditioning=timestep_conditioning,
                upsample_residual=upsample_residual[i],
                upscale_factor=upsample_factor[i],
                spatial_padding_mode=spatial_padding_mode,
            )

            self.up_blocks.append(up_block)

        # out
        self.norm_out = PerChannelRMSNorm()
        self.conv_act = nn.SiLU()
        self.conv_out = LTX2VideoCausalConv3d(
            in_channels=output_channel,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode=spatial_padding_mode,
        )

        # timestep embedding
        self.time_embedder = None
        self.scale_shift_table = None
        self.timestep_scale_multiplier = None
        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(
                torch.tensor(1000.0, dtype=torch.float32)
            )
            self.time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                output_channel * 2, 0
            )
            self.scale_shift_table = nn.Parameter(
                torch.randn(2, output_channel) / output_channel**0.5
            )

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
    ) -> torch.Tensor:
        causal = causal or self.is_causal

        hidden_states = self.conv_in(hidden_states, causal=causal)

        if self.timestep_scale_multiplier is not None:
            temb = temb * self.timestep_scale_multiplier

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                self.mid_block, hidden_states, temb, None, causal
            )

            for up_block in self.up_blocks:
                hidden_states = self._gradient_checkpointing_func(
                    up_block, hidden_states, temb, None, causal
                )
        else:
            hidden_states = self.mid_block(hidden_states, temb, causal=causal)

            for up_block in self.up_blocks:
                hidden_states = up_block(hidden_states, temb, causal=causal)

        hidden_states = self.norm_out(hidden_states)

        if self.time_embedder is not None:
            temb = self.time_embedder(
                timestep=temb.flatten(),
                resolution=None,
                aspect_ratio=None,
                batch_size=hidden_states.size(0),
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(hidden_states.size(0), -1, 1, 1, 1).unflatten(1, (2, -1))
            temb = temb + self.scale_shift_table[None, ..., None, None, None]
            shift, scale = temb.unbind(dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states, causal=causal)

        p = self.patch_size
        p_t = self.patch_size_t

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size, -1, p_t, p, p, num_frames, height, width
        )
        hidden_states = (
            hidden_states.permute(0, 1, 5, 2, 6, 4, 7, 3)
            .flatten(6, 7)
            .flatten(4, 5)
            .flatten(2, 3)
        )

        return hidden_states


class AutoencoderKLLTX2Video(ParallelTiledVAE):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False

    def __init__(self, config: LTXVideoVAEConfig):
        super().__init__(config=config)
        in_channels = config.arch_config.in_channels
        latent_channels = config.arch_config.latent_channels
        out_channels = config.arch_config.out_channels
        block_out_channels = config.arch_config.block_out_channels
        down_block_types = config.arch_config.down_block_types
        spatio_temporal_scaling = config.arch_config.spatio_temporal_scaling
        layers_per_block = config.arch_config.layers_per_block
        downsample_type = config.arch_config.downsample_type
        patch_size = config.arch_config.patch_size
        patch_size_t = config.arch_config.patch_size_t
        resnet_norm_eps = config.arch_config.resnet_norm_eps
        encoder_causal = config.arch_config.encoder_causal
        encoder_spatial_padding_mode = config.arch_config.encoder_spatial_padding_mode

        decoder_block_out_channels = config.arch_config.decoder_block_out_channels
        decoder_spatio_temporal_scaling = (
            config.arch_config.decoder_spatio_temporal_scaling
        )
        decoder_layers_per_block = config.arch_config.decoder_layers_per_block
        decoder_causal = config.arch_config.decoder_causal
        decoder_spatial_padding_mode = config.arch_config.decoder_spatial_padding_mode

        self.encoder = LTX2VideoEncoder3d(
            in_channels,
            latent_channels,
            block_out_channels,
            down_block_types,
            spatio_temporal_scaling,
            layers_per_block,
            downsample_type,
            patch_size,
            patch_size_t,
            resnet_norm_eps,
            encoder_causal,
            encoder_spatial_padding_mode,
        )

        self.decoder = LTX2VideoDecoder3d(
            latent_channels,
            out_channels,
            decoder_block_out_channels,
            decoder_spatio_temporal_scaling,
            decoder_layers_per_block,
            patch_size,
            patch_size_t,
            resnet_norm_eps,
            decoder_causal,
            decoder_spatial_padding_mode,
        )

        latents_mean = torch.zeros((latent_channels,), requires_grad=False)
        latents_std = torch.ones((latent_channels,), requires_grad=False)
        self.register_buffer("latents_mean", latents_mean, persistent=True)
        self.register_buffer("latents_std", latents_std, persistent=True)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # When decoding temporally long video latents, the memory requirement is very high. By decoding latent frames
        # at a fixed frame batch size (based on `self.num_latent_frames_batch_sizes`), the memory requirement can be lowered.
        self.use_framewise_encoding = False
        self.use_framewise_decoding = False

        # This can be configured based on the amount of GPU memory available.
        # `16` for sample frames and `2` for latent frames are sensible defaults for consumer GPUs.
        # Setting it to higher values results in higher memory usage.
        self.num_sample_frames_batch_size = 16
        self.num_latent_frames_batch_size = 2

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 512
        self.tile_sample_min_width = 512
        self.tile_sample_min_num_frames = 16

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 448
        self.tile_sample_stride_width = 448
        self.tile_sample_stride_num_frames = 8

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_min_num_frames: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
        tile_sample_stride_num_frames: Optional[float] = None,
    ) -> None:
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The minimum height required for a sample to be separated into tiles across the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The minimum width required for a sample to be separated into tiles across the width dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
        """
        self.use_tiling = True
        self.tile_sample_min_height = (
            tile_sample_min_height or self.tile_sample_min_height
        )
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_num_frames = (
            tile_sample_min_num_frames or self.tile_sample_min_num_frames
        )
        self.tile_sample_stride_height = (
            tile_sample_stride_height or self.tile_sample_stride_height
        )
        self.tile_sample_stride_width = (
            tile_sample_stride_width or self.tile_sample_stride_width
        )
        self.tile_sample_stride_num_frames = (
            tile_sample_stride_num_frames or self.tile_sample_stride_num_frames
        )

    def _encode(self, x: torch.Tensor, causal: Optional[bool] = None) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > self.tile_sample_min_num_frames:
            return self._temporal_tiled_encode(x, causal=causal)

        if self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            return self.tiled_encode(x, causal=causal)

        enc = self.encoder(x, causal=causal)

        return enc

    def encode(
        self, x: torch.Tensor, causal: Optional[bool] = None, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [
                self._encode(x_slice, causal=causal) for x_slice in x.split(1)
            ]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x, causal=causal)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )

        if self.use_framewise_decoding and num_frames > tile_latent_min_num_frames:
            return self._temporal_tiled_decode(
                z, temb, causal=causal, return_dict=return_dict
            )

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height
        ):
            return self.tiled_decode(z, temb, causal=causal, return_dict=return_dict)

        dec = self.decoder(z, temb, causal=causal)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        if self.use_slicing and z.shape[0] > 1:
            if temb is not None:
                decoded_slices = [
                    self._decode(z_slice, t_slice, causal=causal).sample
                    for z_slice, t_slice in (z.split(1), temb.split(1))
                ]
            else:
                decoded_slices = [
                    self._decode(z_slice, causal=causal).sample
                    for z_slice in z.split(1)
                ]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z, temb, causal=causal).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_t(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (
                1 - x / blend_extent
            ) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def tiled_encode(
        self, x: torch.Tensor, causal: Optional[bool] = None
    ) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                time = self.encoder(
                    x[
                        :,
                        :,
                        :,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ],
                    causal=causal,
                )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width]
                )
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor],
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """

        batch_size, num_channels, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                time = self.decoder(
                    z[
                        :,
                        :,
                        :,
                        i : i + tile_latent_min_height,
                        j : j + tile_latent_min_width,
                    ],
                    temb,
                    causal=causal,
                )

                row.append(time)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(
                    tile[
                        :,
                        :,
                        :,
                        : self.tile_sample_stride_height,
                        : self.tile_sample_stride_width,
                    ]
                )
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(
        self, x: torch.Tensor, causal: Optional[bool] = None
    ) -> AutoencoderKLOutput:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (
                height > self.tile_sample_min_height
                or width > self.tile_sample_min_width
            ):
                tile = self.tiled_encode(tile, causal=causal)
            else:
                tile = self.encoder(tile, causal=causal)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self,
        z: torch.Tensor,
        temb: Optional[torch.Tensor],
        causal: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (
                tile.shape[-1] > tile_latent_min_width
                or tile.shape[-2] > tile_latent_min_height
            ):
                decoded = self.tiled_decode(
                    tile, temb, causal=causal, return_dict=True
                ).sample
            else:
                decoded = self.decoder(tile, temb, causal=causal)
            if i > 0:
                decoded = decoded[:, :, :-1, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                tile = tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                result_row.append(tile)
            else:
                result_row.append(
                    tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :]
                )

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        sample_posterior: bool = False,
        encoder_causal: Optional[bool] = None,
        decoder_causal: Optional[bool] = None,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.Tensor, torch.Tensor]:
        x = sample
        posterior = self.encode(x, causal=encoder_causal).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, temb, causal=decoder_causal)
        if not return_dict:
            return (dec.sample,)
        return dec


EntryClass = AutoencoderKLLTX2Video
