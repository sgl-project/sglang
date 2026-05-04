# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.srt.ug.context import UGContextBundle
from sglang.srt.ug.denoiser import UGMiddleBridge
from sglang.srt.ug.interleaved import UGGKind, UGGSegmentResult


class U1PixelFlowGSegmentExecutor:
    """SenseNova U1 pixel-flow G executor skeleton."""

    required_g_kind: UGGKind = "pixel_flow"
    patch_size: int = 16

    def __call__(
        self,
        *,
        bridge: UGMiddleBridge,
        contexts: UGContextBundle,
        batch: Req,
        server_args: ServerArgs,
    ) -> UGGSegmentResult:
        if getattr(bridge, "g_kind", None) != self.required_g_kind:
            raise ValueError(
                "SenseNova U1 pixel-flow executor requires g_kind='pixel_flow', got "
                f"{getattr(bridge, 'g_kind', None)!r}"
            )
        height, width = _u1_image_size(batch=batch, server_args=server_args)
        grid = _u1_patch_grid(
            height=height,
            width=width,
            patch_size=self.patch_size,
        )
        timesteps = _u1_timesteps(
            num_inference_steps=batch.sampling_params.num_inference_steps,
            timestep_shift=batch.sampling_params.timestep_shift,
        )
        guidance = _u1_guidance_branch(batch.sampling_params)
        image = _u1_unpatchify_placeholder(
            height=height,
            width=width,
            grid=grid,
            timesteps=timesteps,
            guidance=guidance,
            seed=_first_int(getattr(batch, "seed", None), default=0),
            session_id=contexts.full.session.session_id
            if contexts.full.session is not None
            else "",
        )
        return UGGSegmentResult(
            type="image",
            image=image,
            metadata={
                "g_kind": "pixel_flow",
                "grid": grid,
                "timesteps": len(timesteps),
                "guidance": guidance,
                "patch_size": self.patch_size,
                "temporary_g_kv": False,
            },
        )


def _u1_patch_grid(*, height: int, width: int, patch_size: int) -> tuple[int, int]:
    if height <= 0 or width <= 0:
        raise ValueError(f"U1 image size must be positive, got {height}x{width}")
    if patch_size <= 0:
        raise ValueError(f"U1 patch_size must be positive, got {patch_size}")
    return (
        math.ceil(height / patch_size),
        math.ceil(width / patch_size),
    )


def _u1_timesteps(
    *,
    num_inference_steps: int | None,
    timestep_shift: float,
) -> list[float]:
    steps = int(num_inference_steps or 0)
    if steps <= 0:
        raise ValueError(f"num_inference_steps must be positive, got {steps}")
    if timestep_shift <= 0:
        raise ValueError(f"timestep_shift must be positive, got {timestep_shift}")
    if steps == 1:
        base = [1.0]
    else:
        base = [1.0 - i / (steps - 1) for i in range(steps)]
    return [
        timestep_shift * timestep / (1 + (timestep_shift - 1) * timestep)
        if timestep > 0
        else 0.0
        for timestep in base
    ]


def _u1_guidance_branch(sampling_params) -> str:
    text_scale = float(getattr(sampling_params, "cfg_text_scale", 1.0))
    image_scale = float(getattr(sampling_params, "cfg_img_scale", 1.0))
    text_guided = text_scale > 1.0
    image_guided = image_scale > 1.0
    if text_guided and image_guided:
        return "text_image"
    if text_guided:
        return "text"
    if image_guided:
        return "image"
    return "none"


def _u1_image_size(*, batch: Req, server_args: ServerArgs) -> tuple[int, int]:
    params = batch.sampling_params
    cfg = server_args.pipeline_config
    height = _first_int(
        getattr(batch, "height", None),
        getattr(params, "height", None),
        getattr(cfg, "default_height", None),
        default=1024,
    )
    width = _first_int(
        getattr(batch, "width", None),
        getattr(params, "width", None),
        getattr(cfg, "default_width", None),
        default=1024,
    )
    return height, width


def _first_int(*values, default: int) -> int:
    for value in values:
        if value is not None:
            return int(value)
    return int(default)


def _u1_unpatchify_placeholder(
    *,
    height: int,
    width: int,
    grid: tuple[int, int],
    timesteps: list[float],
    guidance: str,
    seed: int,
    session_id: str,
) -> Image.Image:
    fingerprint = (
        seed
        + grid[0] * 17
        + grid[1] * 31
        + len(timesteps) * 13
        + sum(session_id.encode("utf-8"))
        + sum(guidance.encode("utf-8"))
    )
    y, x = np.indices((height, width), dtype=np.uint16)
    array = np.stack(
        [
            (x + fingerprint) % 256,
            (y * 2 + fingerprint // 2) % 256,
            ((x + y) * 3 + fingerprint // 3) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(array, "RGB")
