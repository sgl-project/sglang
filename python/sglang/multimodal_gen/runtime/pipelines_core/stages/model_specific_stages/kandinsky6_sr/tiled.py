# SPDX-License-Identifier: Apache-2.0
"""Tile preparation, denoising and blending for Kandinsky 6 SR.

Stages group work by component to avoid swapping weights per tile.
Each chunk uses seed + its first tile index, matching the reference.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence

import msgspec
import torch
from torch.nn import functional as F

from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latents import (
    build_initial_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
    bf16_autocast,
    cast_to_module_dtype,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    RESOLUTIONS,
    VAE_SPATIAL_FACTOR,
    TileGrid,
    closest_base_resolution,
    compute_tile_grid_even,
    extract_all_tiles,
    latent_tile_grid_from_pixel_grid,
    stitch_tiles_hanning,
)

LatentUpscaleFn = Callable[[torch.Tensor], torch.Tensor]
STITCH_FRAME_CHUNK = 8


class TilePlan(msgspec.Struct, frozen=True):
    """Tile geometry of one run (all sizes in pixels of the frame that is tiled)."""

    tiling_scale: int
    frame_hw: tuple[int, int]
    base_hw: tuple[int, int]
    tile_hw: tuple[int, int]
    pixel_grid: TileGrid


def plan_tiles(
    *,
    frame_hw: tuple[int, int],
    visual_size: int,
    tiling_scale: int,
    tile_min_overlap: float,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
    resolutions: Mapping[int, Sequence[tuple[int, int]]] = RESOLUTIONS,
) -> TilePlan:
    """Even tile grid for a frame; the frame must hold at least one full tile."""
    height, width = frame_hw
    base_hw = closest_base_resolution(height, width, visual_size, resolutions)
    if any(size % tiling_scale for size in base_hw):
        raise ValueError(
            f"resolution_scale={tiling_scale} does not divide the base resolution "
            f"{base_hw[0]}x{base_hw[1]} exactly"
        )
    tile_hw = (base_hw[0] // tiling_scale, base_hw[1] // tiling_scale)
    grid = compute_tile_grid_even(
        height, width, tile_hw, tile_min_overlap, spatial_factor
    )
    if height < tile_hw[0] or width < tile_hw[1]:
        raise ValueError(
            f"The {width}x{height} frame is smaller than one SR tile "
            f"({tile_hw[1]}x{tile_hw[0]}) at x{tiling_scale}; use a larger input or a "
            "higher scale."
        )
    return TilePlan(
        tiling_scale=tiling_scale,
        frame_hw=(height, width),
        base_hw=base_hw,
        tile_hw=tile_hw,
        pixel_grid=grid,
    )


def chunk_ranges(num_tiles: int, tiles_batch_size: int) -> Iterator[tuple[int, int]]:
    """``(start, stop)`` tile ranges; ``start`` also offsets the chunk's seed."""
    for start in range(0, num_tiles, tiles_batch_size):
        yield start, min(start + tiles_batch_size, num_tiles)


@torch.no_grad()
def encode_video_to_lr_latent(
    video: torch.Tensor, vae: torch.nn.Module, *, device: torch.device
) -> torch.Tensor:
    """Encode a whole ``[T, C, H, W]`` uint8 video to the raw latent ``[T', C', h, w]``.

    The latent is *not* multiplied by the VAE scaling factor (the LU does that itself).
    """
    if video.ndim != 4:
        raise ValueError(f"video must be [T, C, H, W], got {tuple(video.shape)}")
    pixel = video.permute(1, 0, 2, 3).unsqueeze(0).to(device=device)
    pixel = cast_to_module_dtype(vae, vae.normalize_data(pixel.float()))
    latent = vae.encode(pixel)[0]
    return latent.squeeze(0).permute(1, 0, 2, 3).float()


@torch.no_grad()
def upscale_lr_latent_tile(
    tile: torch.Tensor,
    upscale_fn: LatentUpscaleFn,
    *,
    lu_dtype: torch.dtype | None,
    scaling_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """LU on one raw tile ``[T', C', h, w]`` -> scaled latent ``[T', Hb, Wb, C']``."""
    z = tile.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float32)
    if lu_dtype is not None and z.dtype != lu_dtype:
        z = z.to(dtype=lu_dtype)
    with bf16_autocast(device):
        upscaled = upscale_fn(z * scaling_factor)
    return upscaled.squeeze(0).permute(1, 2, 3, 0).float()


@torch.no_grad()
def encode_pixel_tile(
    tile: torch.Tensor,
    vae: torch.nn.Module,
    *,
    scaling_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """Pixel path: ``[T, Hb, Wb, 3]`` floats in [0, 255] -> scaled ``[T', Hb', Wb', C']``.

    The tile goes through bf16 first (also on CPU), like the reference.
    """
    lq = tile.permute(3, 0, 1, 2).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    lq = cast_to_module_dtype(vae, vae.normalize_data(lq))
    latent = vae.encode(lq)[0]
    return latent.squeeze(0).permute(1, 2, 3, 0).float() * scaling_factor


@torch.no_grad()
def prepare_tile_latents(
    source: torch.Tensor,
    plan: TilePlan,
    *,
    tile_encoder: Callable[[torch.Tensor], torch.Tensor],
    latent_path: bool,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
) -> list[torch.Tensor]:
    """Encode/upscale one tile batch at a time; keep prepared chunks on CPU."""
    grid = (
        latent_tile_grid_from_pixel_grid(plan.pixel_grid, spatial_factor)
        if latent_path
        else plan.pixel_grid
    )
    tiles = extract_all_tiles(source, grid)
    chunks = []
    for start, stop in chunk_ranges(len(tiles), spec.tiles_batch_size):
        inputs = tiles[start:stop]
        if not latent_path:
            inputs = [
                F.interpolate(
                    tile.float(),
                    size=plan.base_hw,
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                for tile in inputs
            ]
        lq_tiles = [tile_encoder(tile) for tile in inputs]
        lq = torch.cat(lq_tiles, dim=0)
        x = build_initial_latent(
            instruct_type=dit_spec.instruct_type,
            visual_cond=dit_spec.visual_cond,
            in_visual_dim=dit_spec.in_visual_dim,
            lq_latent=lq,
            device=device,
            seed=spec.seed + start,
            lq_noise_scale=spec.lq_noise_scale,
            lq_noise_type=spec.lq_noise_type,
            lq_channel_noise_scale=spec.lq_channel_noise_scale,
            # conditioning noise must round in the loaded DiT dtype
            dtype=dit_spec.dtype,
        )
        chunk = x.reshape(len(lq_tiles), -1, *x.shape[1:])
        del lq, x
        chunks.append(chunk.cpu())
    return chunks


@torch.no_grad()
def _decode_tile(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    *,
    scaling_factor: float,
    device: torch.device,
) -> torch.Tensor:
    """Decode one ``[1, T', H', W', C]`` tile to uint8 ``[3, T, H, W]`` on device."""
    with bf16_autocast(device):
        z = (latents.to(device) / scaling_factor).permute(0, 4, 1, 2, 3)
        latent = cast_to_module_dtype(vae, z)
        # preserve the reference's FP32 rescale before uint8 quantization
        decoded = vae.decode(latent).sample.float()
        pixels = (vae.denormalize_data(decoded) / 255.0).clamp(0.0, 1.0)
        return (pixels * 255.0).to(torch.uint8).squeeze(0)


def decode_chunks(
    chunks: Sequence[torch.Tensor],
    vae: torch.nn.Module,
    *,
    scaling_factor: float,
    device: torch.device,
    group: GroupCoordinator | None = None,
) -> list[torch.Tensor]:
    """Decode in tile order, optionally sharing independent tiles within one DP replica."""
    latents = [tile for chunk in chunks for tile in chunk.split(1)]
    if group is None or group.world_size == 1 or len(latents) == 1:
        return [
            _decode_tile(tile, vae, scaling_factor=scaling_factor, device=device).cpu()
            for tile in latents
        ]

    tiles = []
    for start in range(0, len(latents), group.world_size):
        index = start + group.rank_in_group
        if index < len(latents):
            decoded = _decode_tile(
                latents[index], vae, scaling_factor=scaling_factor, device=device
            )
        else:
            _, frames, height, width, _ = latents[start].shape
            decoded = torch.zeros(
                (
                    3,
                    1 + (frames - 1) * vae.temporal_factor,
                    height * vae.spatial_factor,
                    width * vae.spatial_factor,
                ),
                device=device,
                dtype=torch.uint8,
            )
        # bound device staging to one tile per rank, not the whole decoded video
        gathered = group.all_gather(decoded.unsqueeze(0))[: len(latents) - start].cpu()
        del decoded
        tiles.extend(gathered.unbind(0))
    return tiles


@torch.no_grad()
def stitch_tiles(tiles: Sequence[torch.Tensor], plan: TilePlan) -> torch.Tensor:
    """Hann-blend [C, T, H, W] tiles in frame chunks to bound memory."""
    height, width = plan.frame_hw
    scale = plan.tiling_scale
    channels, frames = tiles[0].shape[:2]
    out = torch.empty(
        (channels, frames, height * scale, width * scale), dtype=torch.uint8
    )
    for start in range(0, frames, STITCH_FRAME_CHUNK):
        stop = min(start + STITCH_FRAME_CHUNK, frames)
        blended = stitch_tiles_hanning(
            [tile[:, start:stop].float() for tile in tiles],
            plan.pixel_grid,
            height,
            width,
            scale=scale,
        )
        out[:, start:stop] = blended.clamp(0, 255).to(torch.uint8)
    return out
