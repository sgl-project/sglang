# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
# SPDX-License-Identifier: Apache-2.0

import torch
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput

from sglang.multimodal_gen.runtime.models.vaes.parallel.diffusers_spatial import (
    spatial_parallel_diffusers_decode,
)


class AutoencoderKLMixin:
    """Shared 2D KL VAE operations; constructors and public return types stay model-specific."""

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if self.use_tiling and (
            width > self.tile_sample_min_size or height > self.tile_sample_min_size
        ):
            return self._tiled_encode(x)
        enc = self.encoder(x)
        if self.quant_conv is not None:
            enc = self.quant_conv(enc)
        return enc

    def _decode(self, z: torch.Tensor, return_dict: bool = True):
        if self.use_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.tiled_decode(z, return_dict=return_dict)
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        if self._spatial_parallel_decode_enabled:
            dec = spatial_parallel_diffusers_decode(
                self.decoder, z, self._spatial_parallel_upsample_count
            )
        else:
            dec = self.decoder(z)
        return DecoderOutput(sample=dec) if return_dict else (dec,)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for y in range(blend_extent):
            b[:, :, y, :] = a[:, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[
                :, :, y, :
            ] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, x] = a[:, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[
                :, :, :, x
            ] * (x / blend_extent)
        return b

    def _blend_tiles(self, rows, blend_extent: int, row_limit: int) -> torch.Tensor:
        # blending mutates tiles; preserve above-then-left order and row traversal
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=3))
        return torch.cat(result_rows, dim=2)

    def _tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Blend overlapping image tiles; tiled and untiled outputs need not be identical."""
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent
        rows = []
        for i in range(0, x.shape[2], overlap_size):
            row = []
            for j in range(0, x.shape[3], overlap_size):
                tile = x[
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                if self.config.use_quant_conv:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        return self._blend_tiles(rows, blend_extent, row_limit)

    def tiled_encode(self, x: torch.Tensor, return_dict: bool = True):
        posterior = DiagonalGaussianDistribution(self._tiled_encode(x))
        return (
            AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)
        )

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        rows = []
        for i in range(0, z.shape[2], overlap_size):
            row = []
            for j in range(0, z.shape[3], overlap_size):
                tile = z[
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                if self.config.use_post_quant_conv:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        dec = self._blend_tiles(rows, blend_extent, row_limit)
        return DecoderOutput(sample=dec) if return_dict else (dec,)
