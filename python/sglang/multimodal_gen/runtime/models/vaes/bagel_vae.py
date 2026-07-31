# Copyright 2024 Black Forest Labs and contributors.
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""FLUX-style autoencoder used by BAGEL generation and image editing.

Adapted from the Apache-2.0 BAGEL autoencoder. Encoding applies the official
latent scale/shift exactly once. Decoding expects SGLang's standard
``DecodingStage`` to remove that normalization before calling ``decode``.

Source: https://github.com/ByteDance-Seed/Bagel/blob/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f/modeling/autoencoder.py
"""

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sglang.multimodal_gen.configs.models.vaes.bagel import BagelVAEConfig


def _swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class _AttnBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def attention(self, hidden_states: Tensor) -> Tensor:
        """Apply spatial self-attention while preserving NCHW layout."""
        hidden_states = self.norm(hidden_states)
        batch, channels, height, width = hidden_states.shape

        # Conv2d produces NCHW.  Move channels last before flattening spatial
        # positions; a direct reshape would silently interleave channel data.
        query = (
            self.q(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(batch, 1, height * width, channels)
        )
        key = (
            self.k(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(batch, 1, height * width, channels)
        )
        value = (
            self.v(hidden_states)
            .permute(0, 2, 3, 1)
            .reshape(batch, 1, height * width, channels)
        )
        attended = F.scaled_dot_product_attention(query, key, value)
        return (
            attended.reshape(batch, height, width, channels)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.proj_out(self.attention(hidden_states))


class _ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.conv1(_swish(self.norm1(hidden_states)))
        hidden_states = self.conv2(_swish(self.norm2(hidden_states)))
        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(residual)
        return residual + hidden_states


class _Upsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, hidden_states: Tensor) -> Tensor:
        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        return self.conv(hidden_states)


class _Downsample(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, hidden_states: Tensor) -> Tensor:
        # PyTorch Conv2d has no asymmetric padding argument. Match the official
        # encoder by extending only the right and bottom edges before stride 2.
        hidden_states = F.pad(hidden_states, (0, 1, 0, 1), value=0.0)
        return self.conv(hidden_states)


class _Encoder(nn.Module):
    def __init__(self, config: BagelVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        num_resolutions = len(arch.ch_mult)
        self.conv_in = nn.Conv2d(arch.in_channels, arch.ch, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        block_in = arch.ch
        for level in range(num_resolutions):
            block_out = arch.ch * arch.ch_mult[level]
            down = nn.Module()
            down.block = nn.ModuleList()
            down.attn = nn.ModuleList()
            for _ in range(arch.num_res_blocks):
                down.block.append(_ResnetBlock(block_in, block_out))
                block_in = block_out
            if level != num_resolutions - 1:
                down.downsample = _Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)
        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(
            block_in, 2 * arch.z_channels, kernel_size=3, padding=1
        )
        self._num_resolutions = num_resolutions
        self._num_res_blocks = arch.num_res_blocks

    def forward(self, images: Tensor) -> Tensor:
        hidden_states = self.conv_in(images)
        for level in range(self._num_resolutions):
            for block_index in range(self._num_res_blocks):
                hidden_states = self.down[level].block[block_index](hidden_states)
            if hasattr(self.down[level], "downsample"):
                hidden_states = self.down[level].downsample(hidden_states)

        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)
        return self.conv_out(_swish(self.norm_out(hidden_states)))


class _Decoder(nn.Module):
    def __init__(self, config: BagelVAEConfig) -> None:
        super().__init__()
        arch = config.arch_config
        num_resolutions = len(arch.ch_mult)
        block_in = arch.ch * arch.ch_mult[-1]

        self.conv_in = nn.Conv2d(arch.z_channels, block_in, kernel_size=3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = _ResnetBlock(block_in, block_in)
        self.mid.attn_1 = _AttnBlock(block_in)
        self.mid.block_2 = _ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for level in reversed(range(num_resolutions)):
            block_out = arch.ch * arch.ch_mult[level]
            up = nn.Module()
            up.block = nn.ModuleList()
            for block_index in range(arch.num_res_blocks + 1):
                input_channels = block_in if block_index == 0 else block_out
                up.block.append(_ResnetBlock(input_channels, block_out))
            block_in = block_out
            if level != 0:
                up.upsample = _Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(32, block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, arch.out_channels, kernel_size=3, padding=1)
        self._num_resolutions = num_resolutions
        self._num_res_blocks = arch.num_res_blocks

    def forward(self, latents: Tensor) -> Tensor:
        hidden_states = self.conv_in(latents)
        hidden_states = self.mid.block_1(hidden_states)
        hidden_states = self.mid.attn_1(hidden_states)
        hidden_states = self.mid.block_2(hidden_states)

        for level in reversed(range(self._num_resolutions)):
            for block_index in range(self._num_res_blocks + 1):
                hidden_states = self.up[level].block[block_index](hidden_states)
            if hasattr(self.up[level], "upsample"):
                hidden_states = self.up[level].upsample(hidden_states)

        return self.conv_out(_swish(self.norm_out(hidden_states)))


class BagelVAE(nn.Module):
    """BAGEL's configurable FLUX-style VAE encoder and decoder.

    Args:
        config: Autoencoder architecture and lifecycle configuration.
    """

    def __init__(self, config: BagelVAEConfig | None = None) -> None:
        super().__init__()
        self.config = config or BagelVAEConfig()
        if self.config.load_encoder:
            self.encoder = _Encoder(self.config)
        if self.config.load_decoder:
            self.decoder = _Decoder(self.config)
        if not self.config.load_encoder and not self.config.load_decoder:
            raise ValueError("BAGEL VAE must load at least one of encoder or decoder")
        self.scaling_factor = self.config.arch_config.scaling_factor
        self.shift_factor = self.config.arch_config.shift_factor
        self.use_parallel_decode = self.config.use_parallel_decode
        self.requires_grad_(False)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def decode(self, latents: Tensor) -> Tensor:
        """Decode already de-normalized latent tensors.

        Args:
            latents: Tensor shaped ``[batch, channels, height, width]``.

        Returns:
            Raw image tensor shaped ``[batch, 3, height * 8, width * 8]``.

        Raises:
            RuntimeError: If this component was created without a decoder.
            ValueError: If ``latents`` is not a four-dimensional tensor.
        """
        if not hasattr(self, "decoder"):
            raise RuntimeError("BAGEL VAE decoder is not loaded")
        if latents.ndim != 4:
            raise ValueError(
                f"BAGEL VAE expects NCHW latents, got shape {tuple(latents.shape)}"
            )
        return self.decoder(latents)

    def encode(
        self,
        images: Tensor,
        *,
        generator: torch.Generator,
    ) -> Tensor:
        """Encode and sample normalized latents with a request-owned RNG.

        Args:
            images: Normalized image tensor shaped ``[batch, 3, height, width]``.
            generator: Explicit generator whose device matches ``images``. The
                method never consumes process-global CPU or CUDA RNG state.

        Returns:
            Sampled and normalized latent tensor shaped
            ``[batch, z_channels, height / 8, width / 8]``.

        Raises:
            RuntimeError: If this component was created without an encoder.
            ValueError: If input rank/channels or generator device is invalid.
        """
        if not hasattr(self, "encoder"):
            raise RuntimeError("BAGEL VAE encoder is not loaded")
        if images.ndim != 4 or images.shape[1] != self.config.arch_config.in_channels:
            raise ValueError(
                f"BAGEL VAE expects NCHW RGB images, got shape {tuple(images.shape)}"
            )
        image_device = images.device
        generator_device = torch.device(generator.device)
        if generator_device != image_device:
            raise ValueError(
                "BAGEL VAE generator must use the image device; "
                f"got {generator.device} for {image_device}"
            )

        moments = self.encoder(images)
        mean, log_variance = moments.chunk(2, dim=1)
        noise = torch.randn(
            mean.shape,
            generator=generator,
            device=mean.device,
            dtype=mean.dtype,
        )
        sampled = mean + torch.exp(0.5 * log_variance) * noise
        return self.scaling_factor * (sampled - self.shift_factor)

    def forward(self, latents: Tensor) -> Tensor:
        """Alias for :meth:`decode` used by generic module tooling."""
        return self.decode(latents)

    def load_weights(
        self,
        weights: Iterable[tuple[str, Tensor]],
        *,
        strict: bool = True,
    ) -> set[str]:
        """Stream configured weights from the official autoencoder checkpoint.

        Args:
            weights: Iterator of ``(checkpoint_name, tensor)`` pairs.
            strict: Require complete configured-component coverage and reject
                unknown autoencoder keys.

        Returns:
            Names of target parameters populated by the iterator.

        Raises:
            ValueError: If a required decoder parameter is missing or an
                unclassified checkpoint key is encountered.
        """
        params = dict(self.named_parameters())
        required = set(params)
        loaded: set[str] = set()
        unexpected: list[str] = []

        for source_name, tensor in weights:
            if source_name.startswith("reg."):
                continue
            if source_name.startswith("encoder.") and not self.config.load_encoder:
                continue
            if source_name.startswith("decoder.") and not self.config.load_decoder:
                continue
            if not source_name.startswith(("encoder.", "decoder.")):
                unexpected.append(source_name)
                continue
            if source_name not in params:
                unexpected.append(source_name)
                continue
            self._load_parameter(source_name, params[source_name], tensor)
            loaded.add(source_name)

        missing = sorted(required - loaded)
        if strict and (missing or unexpected):
            details = []
            if missing:
                details.append(f"missing VAE weights: {missing}")
            if unexpected:
                details.append(f"unexpected VAE weights: {sorted(unexpected)}")
            raise ValueError("; ".join(details))
        return loaded

    def _load_parameter(
        self, name: str, parameter: nn.Parameter, tensor: Tensor
    ) -> None:
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise ValueError(
                f"VAE weight shape mismatch for {name}: expected "
                f"{tuple(parameter.shape)}, got {tuple(tensor.shape)}"
            )
        if parameter.is_meta:
            parent: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(
                parent,
                parts[-1],
                nn.Parameter(tensor.to(dtype=parameter.dtype), requires_grad=False),
            )
            return
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


EntryClass = BagelVAE
