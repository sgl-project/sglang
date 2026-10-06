# SPDX-License-Identifier: Apache-2.0
"""Sampling parameters of Kandinsky 6 video super-resolution (video -> video)."""

import os
import time
from dataclasses import dataclass
from typing import Any

from sglang.multimodal_gen.configs.sample.kandinsky6_sr_resolution import (
    SUPPORTED_RESOLUTION_SCALES,
    resolve_target_hw,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

SR_RESIZE_MODES = ("fit", "exact")


@dataclass
class Kandinsky6SRSamplingParams(SamplingParams):
    """Knobs of the tiled SR run.

    The input video comes from ``video_path``; ``height`` / ``width`` / ``num_frames`` /
    ``fps`` are outputs of the SR input stage (they follow the source video), and the
    prompt is ignored (the SR DiT is text-free).  ``seed`` seeds the LQ noising of the
    tiles exactly like the reference (chunk ``k`` uses ``seed + first tile index``).
    """

    # No prompt-driven generation: keep the guidance defaults inert.
    negative_prompt: str = ""
    guidance_scale: float = 1.0

    # 2, 4 or 2.25 (= x1.125 pixel pre-upscale, then x2 tiles).
    sr_resolution_scale: float = 2.25
    # DiT calls per tile (the repo-wide convention; counts network calls, not timestep grid
    # points). Flow-matching bundles want 4; a pi-Flow bundle ignores this and runs its own
    # ``nfe`` instead (the denoising stage warns if this is set to something else).
    num_inference_steps: int = 4
    # Tiles denoised together (memory / speed trade-off).
    sr_tiles_batch_size: int = 1
    # Minimum tile overlap of the tile grid, as a fraction of the tile size.
    sr_tile_min_overlap: float = 0.20
    # Optional delivery resolution: "hd", "fullhd", "2k" or "WxH" (never upscales).
    sr_target_resolution: str | None = None
    # "fit" keeps the aspect ratio inside the target bucket; "exact" squeezes to it.
    sr_target_resize_mode: str = "fit"

    # The SR DiT is text-free: a multipart request needs no ``prompt`` field.
    prompt_optional = True

    @classmethod
    def video_request_extra_fields(cls) -> frozenset[str]:
        # num_inference_steps is not listed here: it is already a core field of the base
        # video request schema (every model gets it), unlike the sr_* knobs below, which the
        # base schema does not declare and which therefore only survive as pydantic
        # model_extra on a multipart / JSON request.
        return frozenset(
            {
                "sr_resolution_scale",
                "sr_tiles_batch_size",
                "sr_tile_min_overlap",
                "sr_target_resolution",
                "sr_target_resize_mode",
            }
        )

    @classmethod
    def lower_video_request_kwargs(
        cls, request: Any, kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        """Lift the ``sr_*`` extras (declared above, kept by the multipart / JSON
        request layer as pydantic ``model_extra``) into the kwargs this class is
        constructed from; the base class only forwards a fixed, generic field set."""
        extra = getattr(request, "model_extra", None) or {}
        for name in cls.video_request_extra_fields():
            if name in extra and extra[name] is not None:
                kwargs[name] = extra[name]
        return kwargs

    def _validate(self):
        super()._validate()
        if self.sr_resolution_scale not in SUPPORTED_RESOLUTION_SCALES:
            raise ValueError(
                f"sr_resolution_scale must be one of {list(SUPPORTED_RESOLUTION_SCALES)}, "
                f"got {self.sr_resolution_scale!r}"
            )
        # num_inference_steps >= 1 is already enforced generically by the base class; the
        # denoising stage additionally rejects it there with a message naming the model, and
        # warns when a pi-Flow checkpoint is asked to run something other than its own nfe.
        if self.sr_tiles_batch_size < 1:
            raise ValueError(
                f"sr_tiles_batch_size must be >= 1, got {self.sr_tiles_batch_size}"
            )
        if not 0.0 <= self.sr_tile_min_overlap < 1.0:
            raise ValueError(
                f"sr_tile_min_overlap must be in [0, 1), got {self.sr_tile_min_overlap}"
            )
        if self.sr_target_resize_mode not in SR_RESIZE_MODES:
            raise ValueError(
                f"sr_target_resize_mode must be one of {list(SR_RESIZE_MODES)}, "
                f"got {self.sr_target_resize_mode!r}"
            )
        # "exact" mode only parses the spec: a malformed tier / WxH fails early.
        resolve_target_hw(self.sr_target_resolution, (1080, 1920), "exact")

    def _validate_with_pipeline_config(self, pipeline_config):
        super()._validate_with_pipeline_config(pipeline_config)
        if self.source_video_path() is None:
            raise ValueError(
                "Kandinsky6 SR needs an input video: pass --video-path (a local "
                "mp4/mkv file)."
            )

    def source_video_path(self) -> str | None:
        """The single input video path (a one-element list is accepted)."""
        path = self.video_path
        if isinstance(path, list):
            if len(path) != 1:
                raise ValueError(
                    f"Kandinsky6 SR takes exactly one video, got {len(path)} paths"
                )
            path = path[0]
        return path

    def _set_output_file_name(self):
        path = self.source_video_path() if self.video_path is not None else None
        if self.output_file_name is None and path:
            stem = os.path.splitext(os.path.basename(path))[0]
            stamp = time.strftime("%Y%m%d-%H%M%S")
            self.output_file_name = f"{stem}_sr{self.sr_resolution_scale:g}x_{stamp}"
        super()._set_output_file_name()


__all__ = ["Kandinsky6SRSamplingParams"]
