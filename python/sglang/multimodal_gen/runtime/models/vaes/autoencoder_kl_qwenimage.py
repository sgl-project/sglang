# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.activations import get_activation
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.configs.models.vaes.qwenimage import QwenImageVAEConfig
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)  # pylint: disable=invalid-name

CACHE_T = 2


class QwenImageCausalConv3d(nn.Conv3d):
    r"""
    A custom 3D causal convolution layer with feature caching support.

    This layer extends the standard Conv3D layer by ensuring causality in the time dimension and handling feature
    caching for efficient inference.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to all three sides of the input. Default: 0
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

        # Set up causal padding
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class QwenImageRMS_norm(nn.Module):
    r"""
    A custom RMS normalization layer.

    Args:
        dim (int): The number of dimensions to normalize over.
        channel_first (bool, optional): Whether the input tensor has channels as the first dimension.
            Default is True.
        images (bool, optional): Whether the input represents image data. Default is True.
        bias (bool, optional): Whether to include a learnable bias term. Default is False.
    """

    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return (
            F.normalize(x, dim=(1 if self.channel_first else -1))
            * self.scale
            * self.gamma
            + self.bias
        )


class QwenImageUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class QwenImageResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
            - 'downsample2d': 2D downsampling with zero-padding and convolution.
            - 'downsample3d': 3D downsampling with zero-padding, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                QwenImageUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim // 2, 3, padding=1),
            )
            self.time_conv = QwenImageCausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0)
            )

        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = QwenImageCausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] != "Rep"
                    ):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :]
                                .unsqueeze(2)
                                .to(cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (
                        cache_x.shape[2] < 2
                        and feat_cache[idx] is not None
                        and feat_cache[idx] == "Rep"
                    ):
                        cache_x = torch.cat(
                            [torch.zeros_like(cache_x).to(cache_x.device), cache_x],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                    )
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x


class QwenImageResidualBlock(nn.Module):
    r"""
    A custom residual block module.

    Args:
        in_dim (int): Number of input channels.
        out_dim (int): Number of output channels.
        dropout (float, optional): Dropout rate for the dropout layer. Default is 0.0.
        non_linearity (str, optional): Type of non-linearity to use. Default is "silu".
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation(non_linearity)

        # layers
        self.norm1 = QwenImageRMS_norm(in_dim, images=False)
        self.conv1 = QwenImageCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImageRMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImageCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            QwenImageCausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # Apply shortcut connection
        h = self.conv_shortcut(x)

        # First normalization and activation
        x = self.norm1(x)
        x = self.nonlinearity(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # Second normalization and activation
        x = self.norm2(x)
        x = self.nonlinearity(x)

        # Dropout
        x = self.dropout(x)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )

            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        # Add residual connection
        return x + h


class QwenImageAttentionBlock(nn.Module):
    r"""
    Causal self-attention with a single head.

    Args:
        dim (int): The number of channels in the input tensor.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = QwenImageRMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()

        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)

        # compute query, key, value
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(q, k, v)

        x = (
            x.squeeze(1)
            .permute(0, 2, 1)
            .reshape(batch_size * time, channels, height, width)
        )

        # output projection
        x = self.proj(x)

        # Reshape back: [(b*t), c, h, w] -> [b, c, t, h, w]
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)

        return x + identity


class QwenImageMidBlock(nn.Module):
    """
    Middle block for QwenImageVAE encoder and decoder.

    Args:
        dim (int): Number of input/output channels.
        dropout (float): Dropout rate.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        num_layers: int = 1,
    ):
        super().__init__()
        self.dim = dim

        # Create the components
        resnets = [QwenImageResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImageAttentionBlock(dim))
            resnets.append(QwenImageResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        # First residual block
        x = self.resnets[0](x, feat_cache, feat_idx)

        # Process through attention and residual blocks
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)

            x = resnet(x, feat_cache, feat_idx)

        return x


class QwenImageEncoder3d(nn.Module):
    r"""
    A 3D encoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_downsample (list of bool): Whether to downsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        # dim = config.arch_config.dim
        # z_dim = config.arch_config.z_dim
        # dim_mult = config.arch_config.dim_mult
        # num_res_blocks = config.arch_config.num_res_blocks
        # attn_scales = config.arch_config.attn_scales
        # temperal_downsample = config.arch_config.temperal_downsample
        # dropout = config.arch_config.dropout
        # non_linearity = config.arch_config.non_linearity
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = QwenImageCausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    QwenImageResidualBlock(in_dim, out_dim, dropout)
                )
                if scale in attn_scales:
                    self.down_blocks.append(QwenImageAttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(QwenImageResample(out_dim, mode=mode))
                scale /= 2.0

        # middle blocks
        self.mid_block = QwenImageMidBlock(
            out_dim, dropout, non_linearity, num_layers=1
        )

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, z_dim, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## downsamples
        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class QwenImageUpBlock(nn.Module):
    """
    A block that handles upsampling for the QwenImageVAE decoder.

    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
        num_res_blocks (int): Number of residual blocks
        dropout (float): Dropout rate
        upsample_mode (str, optional): Mode for upsampling ('upsample2d' or 'upsample3d')
        non_linearity (str): Type of non-linearity to use
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: Optional[str] = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Create layers list
        resnets = []
        # Add residual blocks and attention if needed
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                QwenImageResidualBlock(current_dim, out_dim, dropout, non_linearity)
            )
            current_dim = out_dim

        self.resnets = nn.ModuleList(resnets)

        # Add upsampling layer if needed
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList(
                [QwenImageResample(out_dim, mode=upsample_mode)]
            )

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        """
        Forward pass through the upsampling block.

        Args:
            x (torch.Tensor): Input tensor
            feat_cache (list, optional): Feature cache for causal convolutions
            feat_idx (list, optional): Feature index for cache management

        Returns:
            torch.Tensor: Output tensor
        """
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache, feat_idx)
            else:
                x = resnet(x)

        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache, feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x


class QwenImageDecoder3d(nn.Module):
    r"""
    A 3D decoder module.

    Args:
        dim (int): The base number of channels in the first layer.
        z_dim (int): The dimensionality of the latent space.
        dim_mult (list of int): Multipliers for the number of channels in each block.
        num_res_blocks (int): Number of residual blocks in each block.
        attn_scales (list of float): Scales at which to apply attention mechanisms.
        temperal_upsample (list of bool): Whether to upsample temporally in each block.
        dropout (float): Dropout rate for the dropout layers.
        non_linearity (str): Type of non-linearity to use.
    """

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        self.nonlinearity = get_activation(non_linearity)

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 ** (len(dim_mult) - 2)

        # init block
        self.conv_in = QwenImageCausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.mid_block = QwenImageMidBlock(
            dims[0], dropout, non_linearity, num_layers=1
        )

        # upsample blocks
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0:
                in_dim = in_dim // 2

            # Determine if we need upsampling
            upsample_mode = None
            if i != len(dim_mult) - 1:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            # Create and add the upsampling block
            up_block = QwenImageUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                dropout=dropout,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
            )
            self.up_blocks.append(up_block)

            # Update scale for next iteration
            if upsample_mode is not None:
                scale *= 2.0

        # output blocks
        self.norm_out = QwenImageRMS_norm(out_dim, images=False)
        self.conv_out = QwenImageCausalConv3d(out_dim, 3, 3, padding=1)

        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        ## middle
        x = self.mid_block(x, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache, feat_idx)

        ## head
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x


class AutoencoderKLQwenImage(nn.Module):
    r"""
    A VAE model with KL loss for encoding videos into latents and decoding latent representations into videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    _supports_gradient_checkpointing = False

    # fmt: off
    def __init__(
        self,
        config: QwenImageVAEConfig,
    ) -> None:
        # fmt: on
        super().__init__()
        base_dim = config.arch_config.base_dim
        z_dim = config.arch_config.z_dim
        dim_mult = config.arch_config.dim_mult
        num_res_blocks = config.arch_config.num_res_blocks
        attn_scales = config.arch_config.attn_scales
        temperal_downsample = config.arch_config.temperal_downsample
        dropout = config.arch_config.dropout
        # non_linearity = config.arch_config.non_linearity
        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        self.encoder = QwenImageEncoder3d(
            base_dim, z_dim * 2, dim_mult, num_res_blocks, attn_scales, self.temperal_downsample, dropout
        )
        self.quant_conv = QwenImageCausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImageCausalConv3d(z_dim, z_dim, 1)

        self.decoder = QwenImageDecoder3d(
            base_dim, z_dim, dim_mult, num_res_blocks, attn_scales, self.temperal_upsample, dropout
        )

        self.spatial_compression_ratio = 2 ** len(self.temperal_downsample)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When decoding spatially large video latents, the memory requirement is very high. By breaking the video latent
        # frames spatially into smaller tiles and performing multiple forward passes for decoding, and then blending the
        # intermediate tiles together, the memory requirement can be lowered.
        self.use_tiling = False

        # The minimal tile height and width for spatial tiling to be used
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256

        # The minimal distance between two spatial tiles
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192

        # Precompute and cache conv counts for encoder and decoder for clear_cache speedup
        self._cached_conv_counts = {
            "decoder": sum(isinstance(m, QwenImageCausalConv3d) for m in self.decoder.modules())
            if self.decoder is not None
            else 0,
            "encoder": sum(isinstance(m, QwenImageCausalConv3d) for m in self.encoder.modules())
            if self.encoder is not None
            else 0,
        }
        cuda_device = get_local_torch_device()
        # FIXME: hardcode
        dtype = torch.bfloat16
        latent_channels = config.arch_config.z_dim

        self.shift_factor = (
            torch.tensor(
                config.arch_config.latents_mean
            )
            .view(1, latent_channels, 1, 1, 1)
            .to(cuda_device, dtype)
        )
        latents_std_tensor = torch.tensor(config.arch_config.latents_std, dtype=dtype, device=cuda_device)
        self.scaling_factor = (1.0 / latents_std_tensor).view(1, latent_channels, 1, 1, 1)

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_sample_stride_height: Optional[float] = None,
        tile_sample_stride_width: Optional[float] = None,
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
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False

    def enable_slicing(self) -> None:
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_slicing = True

    def disable_slicing(self) -> None:
        r"""
        Disable sliced VAE decoding. If `enable_slicing` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_slicing = False

    def clear_cache(self):
        def _count_conv3d(model):
            count = 0
            for m in model.modules():
                if isinstance(m, QwenImageCausalConv3d):
                    count += 1
            return count

        self._conv_num = _count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = _count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        self.clear_cache()
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1): 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
                out = torch.cat([out, out_], 2)

        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> DiagonalGaussianDistribution:
        r"""
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
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)

        return posterior

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i: i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                out_ = self.decoder(x[:, :, i: i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)

        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)

        return DecoderOutput(sample=out)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        r"""
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
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        return decoded

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, num_frames, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, self.tile_sample_stride_height):
            row = []
            for j in range(0, width, self.tile_sample_stride_width):
                self.clear_cache()
                time = []
                frame_range = 1 + (num_frames - 1) // 4
                for k in range(frame_range):
                    self._enc_conv_idx = [0]
                    if k == 0:
                        tile = x[:, :, :1, i: i + self.tile_sample_min_height, j: j + self.tile_sample_min_width]
                    else:
                        tile = x[
                            :,
                            :,
                            1 + 4 * (k - 1): 1 + 4 * k,
                            i: i + self.tile_sample_min_height,
                            j: j + self.tile_sample_min_width,
                        ]
                    tile = self.encoder(tile, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
                    tile = self.quant_conv(tile)
                    time.append(tile)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

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
                result_row.append(tile[:, :, :, :tile_latent_stride_height, :tile_latent_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
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
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k: k + 1, i: i + tile_latent_min_height, j: j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile, feat_cache=self._feat_map, feat_idx=self._conv_idx)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()

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
                result_row.append(tile[:, :, :, : self.tile_sample_stride_height, : self.tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        """
        Args:
            sample (`torch.Tensor`): Input sample.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec


EntryClass = AutoencoderKLQwenImage
