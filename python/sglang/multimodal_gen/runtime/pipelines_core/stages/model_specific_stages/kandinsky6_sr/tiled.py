# SPDX-License-Identifier: Apache-2.0
"""Framework-independent tile orchestration of Kandinsky 6 video SR.

The whole tiled algorithm as plain functions over tensors and duck-typed components, so
it can run (and be compared to the reference) on CPU without a pipeline:

1. :func:`plan_tiles` - tile geometry for a frame size and scale;
2. *prepare* - per chunk of ``tiles_batch_size`` tiles, the initial noisy latent
   ``[B, T', H', W', C]``: the latent-upscaler (LU) path (:func:`prepare_lu_tile_latents`,
   the whole video encoded once by :func:`encode_video_to_lr_latent`) or the pixel path
   (:func:`prepare_pixel_tile_latents`, every tile bilinearly enlarged and encoded);
3. *denoise* - :func:`denoise_chunks` runs the bundle's own scheduler (pi-Flow or flow-Euler);
4. *decode* - :func:`decode_chunks` turns latents into uint8 tiles;
5. :func:`stitch_tiles` - Hann-window blending back into one video.

The stages run these phases one component at a time (LU or VAE, then DiT, then VAE) so
that CPU offloading moves every component once per request instead of once per tile;
:func:`super_resolve` chains them for tests.  Chunk ``k`` is seeded with
``seed + first_tile_index_of_chunk`` exactly like the reference.

Components: ``vae`` needs ``encode(x) -> (latent, _)``, ``decode(z).sample``,
``normalize_data`` / ``denormalize_data`` and parameters; the LU is a callable
``[1, C, T, h, w] -> [1, C, T, H, W]`` on the *scaled* latent.
"""

from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import msgspec
import torch

from sglang.multimodal_gen.runtime.distributed.group_coordinator import GroupCoordinator
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.latents import (
    build_initial_latent,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.sampling import (
    DitSpec,
    SamplingSpec,
    StepCallback,
    StepContext,
    bf16_autocast,
    cast_to_module_dtype,
    check_sampler_options,
    denoise_with_scheduler,
    euler_start_timestep,
    make_dit_fn,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6_sr.tiling import (
    RESOLUTIONS,
    VAE_SPATIAL_FACTOR,
    TileGrid,
    extract_all_tiles,
    latent_tile_grid_from_pixel_grid,
    stitch_tiles_hanning,
    tile_geometry,
    upsample_tiles_to_base,
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
    base_hw, tile_hw, grid = tile_geometry(
        height,
        width,
        visual_size,
        tiling_scale,
        0.25,
        spatial_factor,
        "even",
        tile_min_overlap,
        resolutions,
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


def build_chunk_latent(
    lq_tiles: Sequence[torch.Tensor],
    *,
    seed: int,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
) -> torch.Tensor:
    """Initial noisy latent ``[B, T', H', W', C]`` of one chunk of scaled LQ tiles."""
    lq = torch.cat(list(lq_tiles), dim=0)
    batch = len(lq_tiles)
    x = build_initial_latent(
        instruct_type=dit_spec.instruct_type,
        visual_cond=dit_spec.visual_cond,
        in_visual_dim=dit_spec.in_visual_dim,
        lq_latent=lq,
        batch_size=batch,
        duration=lq.shape[0] // batch,
        height=lq.shape[1],
        width=lq.shape[2],
        device=device,
        seed=seed,
        lq_noise_scale=spec.lq_noise_scale,
        lq_noise_type=spec.lq_noise_type,
        lq_channel_noise_scale=spec.lq_channel_noise_scale,
        # The DiT's own loaded dtype (reference: cast_to_module_dtype(dit, ...) before
        # building the initial latent), not a hard-coded fp32 regardless of precision.
        dtype=dit_spec.dtype,
    )
    return x.reshape(batch, -1, *x.shape[1:])


@torch.no_grad()
def prepare_lu_tile_latents(
    lr_latent: torch.Tensor,
    plan: TilePlan,
    *,
    upscale_fn: LatentUpscaleFn,
    lu_dtype: torch.dtype | None,
    scaling_factor: float,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
) -> list[torch.Tensor]:
    """LU path: initial latents (CPU) of every chunk from the whole-video latent."""
    latent_grid = latent_tile_grid_from_pixel_grid(plan.pixel_grid, spatial_factor)
    tiles = extract_all_tiles(lr_latent, latent_grid)
    chunks: list[torch.Tensor] = []
    for start, stop in chunk_ranges(len(tiles), spec.tiles_batch_size):
        lq_tiles = [
            upscale_lr_latent_tile(
                tile,
                upscale_fn,
                lu_dtype=lu_dtype,
                scaling_factor=scaling_factor,
                device=device,
            )
            for tile in tiles[start:stop]
        ]
        chunk = build_chunk_latent(
            lq_tiles,
            seed=spec.seed + start,
            dit_spec=dit_spec,
            spec=spec,
            device=device,
        )
        chunks.append(chunk.cpu())
    return chunks


@torch.no_grad()
def prepare_pixel_tile_latents(
    video: torch.Tensor,
    plan: TilePlan,
    *,
    vae: torch.nn.Module,
    scaling_factor: float,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
) -> list[torch.Tensor]:
    """Pixel path: bilinear tile -> KVAE encode -> initial latents (CPU) per chunk."""
    tiles = extract_all_tiles(video, plan.pixel_grid)
    chunks: list[torch.Tensor] = []
    for start, stop in chunk_ranges(len(tiles), spec.tiles_batch_size):
        enlarged = upsample_tiles_to_base(tiles[start:stop], *plan.base_hw)
        lq_tiles = [
            encode_pixel_tile(tile, vae, scaling_factor=scaling_factor, device=device)
            for tile in enlarged
        ]
        chunk = build_chunk_latent(
            lq_tiles,
            seed=spec.seed + start,
            dit_spec=dit_spec,
            spec=spec,
            device=device,
        )
        chunks.append(chunk.cpu())
    return chunks


def reset_scheduler_for_chunk(scheduler: Any) -> None:
    """Rewind a scheduler already configured by ``set_timesteps`` back to its first step.

    The same scheduler object denoises every chunk of a request in turn (``set_timesteps`` is
    only called once -- it depends on the bundle and the requested step count, not on any one
    chunk), so its step counter must be rewound between chunks or the second chunk would
    silently continue from where the first left off. Mirrors the shared ``DenoisingStage``'s
    own ``_reset_scheduler_loop_state`` / ``set_begin_index(0)`` pair.
    """
    scheduler._step_index = None
    scheduler.set_begin_index(0)


@torch.no_grad()
def denoise_chunk(
    x: torch.Tensor,
    dit: Callable[..., torch.Tensor],
    scheduler: Any,
    *,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    step_context: StepContext | None = None,
    on_step: StepCallback | None = None,
) -> torch.Tensor:
    """Run the sampler on one chunk ``[B, T', H', W', C]`` -> ``[B, T', H', W', C]``.

    ``scheduler`` must already have had ``set_timesteps`` called on it (once per request, by
    the caller); this only rewinds its step counter so repeated calls (one per chunk) each
    start from the first timestep.
    """
    is_piflow = spec.piflow is not None
    check_sampler_options(
        piflow=spec.piflow,
        cap_noise_timestep=spec.cap_noise_timestep,
        instruct_type=dit_spec.instruct_type,
    )
    if not is_piflow:
        # The Euler start timestep is only meaningful for the scheduler this chunk is about to
        # run: re-derive it instead of baking a stale one into ``set_timesteps`` up front.
        start = euler_start_timestep(
            cap_noise_timestep=spec.cap_noise_timestep,
            lq_noise_scale=spec.lq_noise_scale,
            instruct_type=dit_spec.instruct_type,
        )
        sigmas = torch.linspace(start, 0.0, spec.num_steps + 1)[:-1].tolist()
        scheduler.set_timesteps(sigmas=sigmas, device=x.device)
    reset_scheduler_for_chunk(scheduler)
    dit_fn = make_dit_fn(
        dit,
        latent_frames_hw=tuple(x.shape[1:4]),
        patch_size=dit_spec.patch_size,
        scale_factor=spec.scale_factor,
        use_motion_score=dit_spec.use_motion_score and not is_piflow,
        step_context=step_context,
    )
    with bf16_autocast(x.device):
        return denoise_with_scheduler(
            x,
            dit_fn,
            scheduler,
            channels=dit_spec.in_visual_dim,
            is_piflow=is_piflow,
            on_step=on_step,
        )


def denoise_chunks(
    chunks: Sequence[torch.Tensor],
    dit: Callable[..., torch.Tensor],
    scheduler: Any,
    *,
    dit_spec: DitSpec,
    spec: SamplingSpec,
    device: torch.device,
    step_context: StepContext | None = None,
    on_step: StepCallback | None = None,
) -> list[torch.Tensor]:
    """Denoise every chunk on ``device`` with the same scheduler; results come back on the CPU.

    A pi-Flow scheduler's step grid depends only on the bundle (``nfe``), so it is built once,
    before the first chunk, and every chunk just rewinds it (:func:`reset_scheduler_for_chunk`).
    A flow-Euler scheduler's grid also depends on ``start_timestep`` (capped for some
    ``instruct_type``s), which :func:`denoise_chunk` re-derives per chunk from ``spec`` -- cheap
    to recompute and avoids a redundant "is this piflow" branch here.
    """
    if spec.piflow is not None:
        scheduler.set_timesteps(spec.piflow.nfe, device=device)
    return [
        denoise_chunk(
            chunk.to(device),
            dit,
            scheduler,
            dit_spec=dit_spec,
            spec=spec,
            step_context=step_context,
            on_step=on_step,
        ).cpu()
        for chunk in chunks
    ]


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
    """Hann-blend uint8 tiles ``[3, T, Hb, Wb]`` into one uint8 ``[3, T, H*s, W*s]``.

    Frames are independent, so the blend runs in frame chunks (bounded memory) with the
    same per-pixel arithmetic as one call over the whole clip.
    """
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


@torch.no_grad()
def super_resolve(
    video: torch.Tensor,
    *,
    vae: torch.nn.Module,
    scaling_factor: float,
    dit: Callable[..., torch.Tensor],
    dit_spec: DitSpec,
    spec: SamplingSpec,
    scheduler: Any,
    device: torch.device | str,
    upscale_fn: LatentUpscaleFn | None = None,
    lu_dtype: torch.dtype | None = None,
    spatial_factor: int = VAE_SPATIAL_FACTOR,
    resolutions: Mapping[int, Sequence[tuple[int, int]]] = RESOLUTIONS,
) -> torch.Tensor:
    """Tiled SR of a ``[T, C, H, W]`` uint8 video -> uint8 ``[3, T, H*s, W*s]``.

    ``upscale_fn`` selects the LU path; without it the pixel path runs. ``scheduler`` is the
    already-resolved effective scheduler (``run_spec.effective_scheduler``) -- a real loaded
    component, or one synthesized from ``spec.piflow`` / ``spec.scheduler_scale`` when none fits.
    """
    device = torch.device(device)
    common = dict(
        visual_size=spec.visual_size,
        tiling_scale=spec.tiling_scale,
        tile_min_overlap=spec.tile_min_overlap,
        spatial_factor=spatial_factor,
        resolutions=resolutions,
    )
    if upscale_fn is not None:
        lr_latent = encode_video_to_lr_latent(video, vae, device=device).cpu()
        frame_hw = (
            lr_latent.shape[-2] * spatial_factor,
            lr_latent.shape[-1] * spatial_factor,
        )
        plan = plan_tiles(frame_hw=frame_hw, **common)
        chunks = prepare_lu_tile_latents(
            lr_latent,
            plan,
            upscale_fn=upscale_fn,
            lu_dtype=lu_dtype,
            scaling_factor=scaling_factor,
            dit_spec=dit_spec,
            spec=spec,
            device=device,
            spatial_factor=spatial_factor,
        )
    else:
        plan = plan_tiles(frame_hw=tuple(video.shape[-2:]), **common)
        chunks = prepare_pixel_tile_latents(
            video,
            plan,
            vae=vae,
            scaling_factor=scaling_factor,
            dit_spec=dit_spec,
            spec=spec,
            device=device,
        )
    denoised = denoise_chunks(
        chunks, dit, scheduler, dit_spec=dit_spec, spec=spec, device=device
    )
    tiles = decode_chunks(denoised, vae, scaling_factor=scaling_factor, device=device)
    return stitch_tiles(tiles, plan)
