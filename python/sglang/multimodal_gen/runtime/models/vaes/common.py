# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import Iterator
from math import prod
from typing import Optional, cast

import numpy as np
import torch
import torch.distributed as dist
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.utils.torch_utils import randn_tensor

from sglang.multimodal_gen.configs.models import VAEConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_parallel_rank,
    get_sp_world_size,
)


class ParallelTiledVAE(ABC):
    tile_sample_min_height: int
    tile_sample_min_width: int
    tile_sample_min_num_frames: int
    tile_sample_stride_height: int
    tile_sample_stride_width: int
    tile_sample_stride_num_frames: int
    blend_num_frames: int
    use_tiling: bool
    use_temporal_tiling: bool
    use_parallel_tiling: bool

    def __init__(self, config: VAEConfig, **kwargs) -> None:
        self.config = config
        self.tile_sample_min_height = config.tile_sample_min_height
        self.tile_sample_min_width = config.tile_sample_min_width
        self.tile_sample_min_num_frames = config.tile_sample_min_num_frames
        self.tile_sample_stride_height = config.tile_sample_stride_height
        self.tile_sample_stride_width = config.tile_sample_stride_width
        self.tile_sample_stride_num_frames = config.tile_sample_stride_num_frames
        self.blend_num_frames = config.blend_num_frames
        self.use_tiling = config.use_tiling
        self.use_temporal_tiling = config.use_temporal_tiling
        self.use_parallel_tiling = config.use_parallel_tiling

    def to(self, device) -> "ParallelTiledVAE":
        # TODO: implement this
        return self

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def temporal_compression_ratio(self) -> int:
        return cast(int, self.config.temporal_compression_ratio)

    @property
    def spatial_compression_ratio(self) -> int:
        return cast(int, self.config.spatial_compression_ratio)

    @property
    def scaling_factor(self) -> float | torch.Tensor:
        return cast(float | torch.Tensor, self.config.scaling_factor)

    @abstractmethod
    def _encode(self, *args, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def _decode(self, *args, **kwargs) -> torch.Tensor:
        pass

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:
        batch_size, num_channels, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        if (
            self.use_tiling
            and self.use_temporal_tiling
            and num_frames > self.tile_sample_min_num_frames
        ):
            latents = self.tiled_encode(x)[:, :, :latent_num_frames]
        elif self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            latents = self.spatial_tiled_encode(x)[:, :, :latent_num_frames]
        else:
            latents = self._encode(x)[:, :, :latent_num_frames]
        return DiagonalGaussianDistribution(latents)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = z.shape
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        if self.use_tiling and self.use_parallel_tiling and get_sp_world_size() > 1:
            return self.parallel_tiled_decode(z)[:, :, :num_sample_frames]
        if (
            self.use_tiling
            and self.use_temporal_tiling
            and num_frames > tile_latent_min_num_frames
        ):
            return self.tiled_decode(z)[:, :, :num_sample_frames]

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height
        ):
            return self.spatial_tiled_decode(z)[:, :, :num_sample_frames]

        return self._decode(z)[:, :, :num_sample_frames]

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
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

    def spatial_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, _, height, width = x.shape
        # latent_height = height // self.spatial_compression_ratio
        # latent_width = width // self.spatial_compression_ratio

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
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self._encode(tile)
                row.append(tile)
            rows.append(row)

        return self._merge_spatial_tiles(
            rows,
            blend_height,
            blend_width,
            tile_latent_stride_height,
            tile_latent_stride_width,
        )

    def _parallel_data_generator(
        self, gathered_results, gathered_dim_metadata
    ) -> Iterator[tuple[torch.Tensor, int]]:
        global_idx = 0
        for i, per_rank_metadata in enumerate(gathered_dim_metadata):
            _start_shape = 0
            for shape in per_rank_metadata:
                mul_shape = prod(shape)
                yield (
                    gathered_results[
                        i, _start_shape : _start_shape + mul_shape
                    ].reshape(shape),
                    global_idx,
                )
                _start_shape += mul_shape
                global_idx += 1

    def parallel_tiled_decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        Parallel version of tiled_decode that distributes both temporal and spatial computation across GPUs
        """
        world_size, rank = get_sp_world_size(), get_sp_parallel_rank()
        B, C, T, H, W = z.shape

        # Calculate parameters
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        # Calculate tile dimensions
        num_t_tiles = (
            T + tile_latent_stride_num_frames - 1
        ) // tile_latent_stride_num_frames
        num_h_tiles = (H + tile_latent_stride_height - 1) // tile_latent_stride_height
        num_w_tiles = (W + tile_latent_stride_width - 1) // tile_latent_stride_width
        total_spatial_tiles = num_h_tiles * num_w_tiles
        total_tiles = num_t_tiles * total_spatial_tiles

        # Calculate tiles per rank and padding
        tiles_per_rank = (total_tiles + world_size - 1) // world_size
        start_tile_idx = rank * tiles_per_rank
        end_tile_idx = min((rank + 1) * tiles_per_rank, total_tiles)

        local_results = []
        local_dim_metadata = []
        # Process assigned tiles
        for local_idx, global_idx in enumerate(range(start_tile_idx, end_tile_idx)):
            t_idx = global_idx // total_spatial_tiles
            spatial_idx = global_idx % total_spatial_tiles
            h_idx = spatial_idx // num_w_tiles
            w_idx = spatial_idx % num_w_tiles

            # Calculate positions
            t_start = t_idx * tile_latent_stride_num_frames
            h_start = h_idx * tile_latent_stride_height
            w_start = w_idx * tile_latent_stride_width

            # Extract and process tile
            tile = z[
                :,
                :,
                t_start : t_start + tile_latent_min_num_frames + 1,
                h_start : h_start + tile_latent_min_height,
                w_start : w_start + tile_latent_min_width,
            ]

            # Process tile
            tile = self._decode(tile)

            if t_start > 0:
                tile = tile[:, :, 1:, :, :]

            # Store metadata
            shape = tile.shape
            # Store decoded data (flattened)
            decoded_flat = tile.reshape(-1)
            local_results.append(decoded_flat)
            local_dim_metadata.append(shape)

        results = torch.cat(local_results, dim=0).contiguous()
        del local_results
        # first gather size to pad the results
        local_size = torch.tensor(
            [results.size(0)], device=results.device, dtype=torch.int64
        )
        all_sizes = [
            torch.zeros(1, device=results.device, dtype=torch.int64)
            for _ in range(world_size)
        ]
        dist.all_gather(all_sizes, local_size)
        max_size = max(size.item() for size in all_sizes)
        padded_results = torch.zeros(max_size, device=results.device)
        padded_results[: results.size(0)] = results
        del results

        # Gather all results
        gathered_dim_metadata = [None] * world_size
        gathered_results = (
            torch.zeros_like(padded_results)
            .repeat(world_size, *[1] * len(padded_results.shape))
            .contiguous()
        )  # use contiguous to make sure it won't copy data in the following operations
        # TODO (PY): use sgl_diffusion distributed methods
        dist.all_gather_into_tensor(gathered_results, padded_results)
        dist.all_gather_object(gathered_dim_metadata, local_dim_metadata)
        # Process gathered results
        data: list = [
            [[[] for _ in range(num_w_tiles)] for _ in range(num_h_tiles)]
            for _ in range(num_t_tiles)
        ]
        for current_data, global_idx in self._parallel_data_generator(
            gathered_results, gathered_dim_metadata
        ):
            t_idx = global_idx // total_spatial_tiles
            spatial_idx = global_idx % total_spatial_tiles
            h_idx = spatial_idx // num_w_tiles
            w_idx = spatial_idx % num_w_tiles
            data[t_idx][h_idx][w_idx] = current_data
        # Merge results
        result_slices = []
        last_slice_data = None
        for i, tem_data in enumerate(data):
            slice_data = self._merge_spatial_tiles(
                tem_data,
                blend_height,
                blend_width,
                self.tile_sample_stride_height,
                self.tile_sample_stride_width,
            )
            if i > 0:
                slice_data = self.blend_t(
                    last_slice_data, slice_data, self.blend_num_frames
                )
                result_slices.append(
                    slice_data[:, :, : self.tile_sample_stride_num_frames, :, :]
                )
            else:
                result_slices.append(
                    slice_data[:, :, : self.tile_sample_stride_num_frames + 1, :, :]
                )
            last_slice_data = slice_data
        dec = torch.cat(result_slices, dim=2)

        return dec

    def _merge_spatial_tiles(
        self, tiles, blend_height, blend_width, stride_height, stride_width
    ) -> torch.Tensor:
        """Helper function to merge spatial tiles with blending"""
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(tiles[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :stride_height, :stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def spatial_tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.

        Returns:
            `torch.Tensor`:
                The decoded images.
        """

        _, _, _, height, width = z.shape
        # sample_height = height * self.spatial_compression_ratio
        # sample_width = width * self.spatial_compression_ratio

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
                tile = z[
                    :,
                    :,
                    :,
                    i : i + tile_latent_min_height,
                    j : j + tile_latent_min_width,
                ]
                decoded = self._decode(tile)
                row.append(decoded)
            rows.append(row)
        return self._merge_spatial_tiles(
            rows,
            blend_height,
            blend_width,
            self.tile_sample_stride_height,
            self.tile_sample_stride_width,
        )

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape

        # tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )

        row = []
        for i in range(0, num_frames, self.tile_sample_stride_num_frames):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (
                height > self.tile_sample_min_height
                or width > self.tile_sample_min_width
            ):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self._encode(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, self.blend_num_frames)
                result_row.append(tile[:, :, :tile_latent_stride_num_frames, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])
        enc = torch.cat(result_row, dim=2)
        return enc

    def tiled_decode(self, z: torch.Tensor) -> torch.Tensor:
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
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )

        row = []
        for i in range(0, num_frames, tile_latent_stride_num_frames):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (
                tile.shape[-1] > tile_latent_min_width
                or tile.shape[-2] > tile_latent_min_height
            ):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self._decode(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, self.blend_num_frames)
                result_row.append(
                    tile[:, :, : self.tile_sample_stride_num_frames, :, :]
                )
            else:
                result_row.append(
                    tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :]
                )

        dec = torch.cat(result_row, dim=2)
        return dec

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_num_frames: int | None = None,
        tile_sample_stride_height: int | None = None,
        tile_sample_stride_width: int | None = None,
        tile_sample_stride_num_frames: int | None = None,
        blend_num_frames: int | None = None,
        use_tiling: bool | None = None,
        use_temporal_tiling: bool | None = None,
        use_parallel_tiling: bool | None = None,
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
            tile_sample_min_num_frames (`int`, *optional*):
                The minimum number of frames required for a sample to be separated into tiles across the frame
                dimension.
            tile_sample_stride_height (`int`, *optional*):
                The minimum amount of overlap between two consecutive vertical tiles. This is to ensure that there are
                no tiling artifacts produced across the height dimension.
            tile_sample_stride_width (`int`, *optional*):
                The stride between two consecutive horizontal tiles. This is to ensure that there are no tiling
                artifacts produced across the width dimension.
            tile_sample_stride_num_frames (`int`, *optional*):
                The stride between two consecutive frame tiles. This is to ensure that there are no tiling artifacts
                produced across the frame dimension.
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
        if blend_num_frames is not None:
            self.blend_num_frames = blend_num_frames
        else:
            self.blend_num_frames = (
                self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
            )
        self.use_tiling = use_tiling or self.use_tiling
        self.use_temporal_tiling = use_temporal_tiling or self.use_temporal_tiling
        self.use_parallel_tiling = use_parallel_tiling or self.use_parallel_tiling

    def disable_tiling(self) -> None:
        r"""
        Disable tiled VAE decoding. If `enable_tiling` was previously enabled, this method will go back to computing
        decoding in one step.
        """
        self.use_tiling = False


# adapted from https://github.com/huggingface/diffusers/blob/e7ffeae0a191f710881d1fbde00cd6ff025e81f2/src/diffusers/models/autoencoders/vae.py#L691
class DiagonalGaussianDistribution:

    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(
        self, other: Optional["DiagonalGaussianDistribution"] = None
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(
        self, sample: torch.Tensor, dims: tuple[int, ...] = (1, 2, 3)
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
