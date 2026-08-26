# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from einops import rearrange

from sglang.multimodal_gen.configs.models.vaes.magi2 import (
    Magi2TurboVAEArchConfig,
    Magi2TurboVAEConfig,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.models.vaes.common import ParallelTiledVAE
from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanUpsample, unpatchify

# Hardcoded inside every resnet norm; resnet_norm_eps only reaches the shortcut norm.
RESNET_INNER_NORM_EPS = 1e-8


def rms_norm_channels(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(dim=1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)).to(x.dtype)


def pad_time_replicate(x: torch.Tensor, kernel_t: int) -> torch.Tensor:
    """This decoder is non-causal, so a window needs one latent frame of context on both sides."""
    if kernel_t == 1:
        return x
    pad = (kernel_t - 1) // 2
    left = x[:, :, :1].repeat(1, 1, pad, 1, 1)
    right = x[:, :, -1:].repeat(1, 1, pad, 1, 1)
    return torch.cat([left, x, right], dim=2)


def strip_turbo_vae_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A training checkpoint: tensors may sit under ``ema_state_dict`` / ``state_dict``, carry ``module.``, or be rooted at the decoder."""
    for wrapper_key in ("ema_state_dict", "state_dict"):
        if wrapper_key in state_dict:
            state_dict = state_dict[wrapper_key]
            break

    stripped = {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    if any(key.startswith("decoder.") for key in stripped):
        return stripped
    return {f"decoder.{key}": value for key, value in stripped.items()}


class Magi2TurboConv3d(nn.Module):
    """Replicate padding on time, zero padding on space."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.kernel_t = kernel_size[0]
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(0, kernel_size[1] // 2, kernel_size[2] // 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(pad_time_replicate(x, self.kernel_t))


class Magi2TurboDepthwiseConv3d(nn.Module):

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.kernel_t = kernel_size
        self.depthwise_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            padding=(0, kernel_size // 2, kernel_size // 2),
        )
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(pad_time_replicate(x, self.kernel_t))
        return self.pointwise_conv(x)


class Magi2TurboResnetBlock(nn.Module):
    """Norms are affine-free, so they hold no weights."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        is_dw_conv: bool,
        dw_kernel_size: int,
        after_upsample: bool = False,
    ) -> None:
        super().__init__()
        conv_cls = Magi2TurboDepthwiseConv3d if is_dw_conv else Magi2TurboConv3d
        kernel_size = dw_kernel_size if is_dw_conv else 3

        self.norm_eps = norm_eps
        # Distillation swapped the entry activation of post-upsample blocks.
        self.nonlinearity = get_act_fn("relu" if after_upsample else "silu")
        self.nonlinearity2 = get_act_fn("silu")

        self.conv1 = conv_cls(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.conv2 = conv_cls(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.conv_shortcut = (
            conv_cls(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = rms_norm_channels(x, eps=RESNET_INNER_NORM_EPS)
        h = self.conv1(self.nonlinearity(h))
        h = rms_norm_channels(h, eps=RESNET_INNER_NORM_EPS)
        h = self.conv2(self.nonlinearity2(h))

        if self.conv_shortcut is not None:
            x = self.conv_shortcut(rms_norm_channels(x, eps=self.norm_eps))
        return h + x


class Magi2TurboUpsample(nn.Module):

    def __init__(self, *, dim: int, temporal: bool) -> None:
        super().__init__()
        self.temporal = temporal
        self.resample = nn.Sequential(
            WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        if temporal:
            self.time_conv = Magi2TurboConv3d(
                in_channels=dim, out_channels=dim * 2, kernel_size=(3, 1, 1)
            )

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        batch = x.shape[0]
        if self.temporal:
            x = rearrange(self.time_conv(x), "b (n c) t h w -> b c (t n) h w", n=2)
            if is_first_chunk:
                # No left context, so the leading interpolated frame is unbacked.
                x = x[:, :, 1:]

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=batch)


class Magi2TurboMidBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_layers: int,
        norm_eps: float,
        is_dw_conv: bool,
        dw_kernel_size: int,
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                Magi2TurboResnetBlock(
                    in_channels=dim,
                    out_channels=dim,
                    norm_eps=norm_eps,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        return x


class Magi2TurboUpBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        norm_eps: float,
        upsample: bool,
        spatio_only: bool,
        is_dw_conv: bool,
        dw_kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv_in = (
            Magi2TurboResnetBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                is_dw_conv=is_dw_conv,
                dw_kernel_size=dw_kernel_size,
            )
            if in_channels != out_channels
            else None
        )

        self.upsamplers = (
            nn.ModuleList(
                [Magi2TurboUpsample(dim=out_channels, temporal=not spatio_only)]
            )
            if upsample
            else None
        )

        self.resnets = nn.ModuleList(
            [
                Magi2TurboResnetBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    norm_eps=norm_eps,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    after_upsample=upsample,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x, is_first_chunk=is_first_chunk)
        for resnet in self.resnets:
            x = resnet(x)
        return x


class Magi2TurboDecoder3d(nn.Module):
    """Config lists are in encoder order, so every one of them is reversed."""

    def __init__(self, *, arch_config: Magi2TurboVAEArchConfig) -> None:
        super().__init__()
        block_out_channels = tuple(reversed(arch_config.decoder_block_out_channels))
        layers_per_block = tuple(reversed(arch_config.decoder_layers_per_block))
        upsampling = tuple(reversed(arch_config.decoder_spatio_temporal_scaling))
        spatio_only = tuple(reversed(arch_config.decoder_spatio_only))
        is_dw_conv = tuple(reversed(arch_config.decoder_is_dw_conv))
        norm_eps = arch_config.resnet_norm_eps
        dw_kernel_size = arch_config.decoder_dw_kernel_size

        self.patch_size = arch_config.patch_size
        self.conv_act = get_act_fn("silu")

        dim = block_out_channels[0]
        self.conv_in = Magi2TurboConv3d(
            in_channels=arch_config.latent_channels, out_channels=dim, kernel_size=3
        )
        self.mid_block = Magi2TurboMidBlock(
            dim=dim,
            num_layers=layers_per_block[0],
            norm_eps=norm_eps,
            is_dw_conv=is_dw_conv[0],
            dw_kernel_size=dw_kernel_size,
        )

        self.up_blocks = nn.ModuleList()
        for i, out_dim in enumerate(block_out_channels):
            self.up_blocks.append(
                Magi2TurboUpBlock(
                    in_channels=dim,
                    out_channels=out_dim,
                    num_layers=layers_per_block[i + 1],
                    norm_eps=norm_eps,
                    upsample=upsampling[i],
                    spatio_only=spatio_only[i],
                    is_dw_conv=is_dw_conv[i + 1],
                    dw_kernel_size=dw_kernel_size,
                )
            )
            dim = out_dim

        # The last 2x of the spatial ratio folds into channels; unpatchify recovers it.
        self.conv_out = Magi2TurboConv3d(
            in_channels=dim,
            out_channels=arch_config.out_channels * self.patch_size**2,
            kernel_size=3,
        )

    def forward(self, x: torch.Tensor, *, is_first_chunk: bool) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x, is_first_chunk=is_first_chunk)

        x = rms_norm_channels(x, eps=RESNET_INNER_NORM_EPS)
        x = self.conv_out(self.conv_act(x))
        return unpatchify(x, patch_size=self.patch_size)


class Magi2TurboVAE(ParallelTiledVAE):
    """Latents must arrive un-normalized; that is owned by ``Magi2PipelineConfig.get_decode_scale_and_shift``, not this module."""

    def __init__(self, config: Magi2TurboVAEConfig) -> None:
        nn.Module.__init__(self)
        ParallelTiledVAE.__init__(self, config)

        arch_config = config.arch_config
        if not arch_config.use_unpatchify:
            raise ValueError(
                "Magi2TurboVAE only ships the use_unpatchify=True head; the "
                "pixel-shuffle head is a training-time variant"
            )

        self.z_dim = arch_config.latent_channels
        self.first_chunk_size = arch_config.first_chunk_size
        self.step_size = arch_config.step_size
        self.decoder = Magi2TurboDecoder3d(arch_config=arch_config)

    def _encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "Magi2TurboVAE is a distilled decoder; encode with the Wan2.2 VAE"
        )

    def _pad_to_window_grid(self, z: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Frame count must be first_chunk_size + n * step_size."""
        num_frames = z.shape[2]
        if num_frames < self.first_chunk_size:
            num_padding = self.first_chunk_size - num_frames
        else:
            num_padding = -(num_frames - self.first_chunk_size) % self.step_size

        if num_padding:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, num_padding, 1, 1)], dim=2)
        return z, num_padding

    def _decode_windows(self, z: torch.Tensor) -> torch.Tensor:
        num_frames = z.shape[2]
        first = self.first_chunk_size
        step = self.step_size
        # One latent frame of context on each side costs this many pixel frames.
        overlap = self.temporal_compression_ratio

        if num_frames == first:
            return self.decoder(z, is_first_chunk=True)

        chunks = [
            self.decoder(z[:, :, : first + 1], is_first_chunk=True)[:, :, :-overlap]
        ]
        for start in range(first, num_frames, step):
            is_last = start + step == num_frames
            stop = start + step if is_last else start + step + 1
            out = self.decoder(z[:, :, start - 1 : stop], is_first_chunk=False)
            chunks.append(
                out[:, :, overlap:] if is_last else out[:, :, overlap:-overlap]
            )
        return torch.cat(chunks, dim=2)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z, num_padding = self._pad_to_window_grid(z)
        frames = self._decode_windows(z)
        if num_padding:
            frames = frames[:, :, : -num_padding * self.temporal_compression_ratio]
        return frames


EntryClass = Magi2TurboVAE
