"""Post-decode output geometry for HunyuanImage-3."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
    OutputBatch,
    Req,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.server_args import ServerArgs

from .resolution import OUTPUT_GEOMETRY_EXTRA_KEY


@dataclass(frozen=True)
class _SpatialPlan:
    source_size: tuple[int, int]
    crop_box: tuple[int, int, int, int] | None = None
    padding: tuple[int, int, int, int] | None = None
    pad_value: float = 0.0
    resize_size: tuple[int, int] | None = None


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _crop_box(
    width: int,
    height: int,
    crop_width: int,
    crop_height: int,
    anchor: tuple[float, float],
) -> tuple[int, int, int, int]:
    excess_x, excess_y = width - crop_width, height - crop_height
    left = min(max(_round_half_up(excess_x * anchor[0]), 0), excess_x)
    top = min(max(_round_half_up(excess_y * anchor[1]), 0), excess_y)
    return left, top, left + crop_width, top + crop_height


def _maximum_exact_crop_size(
    width: int, height: int, p: int, q: int
) -> tuple[int, int]:
    """Largest integer crop whose ratio matches p:q within pixel rounding.

    A strict p:q-multiple crop degenerates whenever the reduced ratio does
    not divide the decoded bucket -- the common case for edit requests whose
    reference size is coprime (e.g. 1361x907 -> ratio 1361:907): the
    multiplier collapses to 0 (ValueError) or discards most of the image
    even though a near-exact crop with sub-pixel ratio error exists.
    Anchoring on the limiting dimension and rounding keeps the maximum area
    with at most half a pixel of ratio deviation per axis.
    """
    ratio = p / q
    if width / height >= ratio:
        crop_height = height
        crop_width = min(width, _round_half_up(crop_height * ratio))
    else:
        crop_width = width
        crop_height = min(height, _round_half_up(crop_width / ratio))
    if crop_width <= 0 or crop_height <= 0:
        raise ValueError(
            f"Cannot crop decoded {width}x{height} pixels to ratio {p}:{q}"
        )
    return crop_width, crop_height


def _maximum_approximate_crop_size(
    width: int,
    height: int,
    p: int,
    q: int,
    max_ratio_error: float,
) -> tuple[int, int]:
    """Return max-area integer crop within the declared relative ratio error."""
    best: tuple[int, int] | None = None
    best_area, best_error = -1, math.inf
    for crop_height in range(1, height + 1):
        nominal_width = crop_height * p / q
        lower = max(1, math.ceil(nominal_width * (1.0 - max_ratio_error)))
        upper = min(width, math.floor(nominal_width * (1.0 + max_ratio_error)))
        if lower > upper:
            continue
        crop_width = upper
        ratio_error = abs((crop_width * q) / (crop_height * p) - 1.0)
        area = crop_width * crop_height
        if area > best_area or (area == best_area and ratio_error < best_error):
            best = crop_width, crop_height
            best_area, best_error = area, ratio_error
    if best is None:
        raise ValueError(
            f"Cannot crop decoded {width}x{height} pixels within ratio-error "
            f"limit {max_ratio_error} for {p}:{q}"
        )
    return best


def _minimum_exact_canvas_size(
    width: int, height: int, p: int, q: int
) -> tuple[int, int]:
    multiplier = max(math.ceil(width / p), math.ceil(height / q))
    return multiplier * p, multiplier * q


def _build_spatial_plan(
    decoded_size: tuple[int, int], geometry: dict[str, object]
) -> tuple[_SpatialPlan, dict[str, object]]:
    width, height = decoded_size
    p, q = (int(value) for value in geometry["requested_aspect_ratio"])
    anchor = tuple(float(value) for value in geometry["crop_anchor"])
    strategy = str(geometry["strategy"])
    ratio_policy = str(geometry["ratio_policy"])
    size_mode = str(geometry["size_mode"])

    metadata = dict(geometry)
    metadata.setdefault("native_bucket_size", [width, height])
    metadata["decoded_size"] = [width, height]
    metadata["resampled"] = False

    if strategy == "native_crop":
        if ratio_policy == "exact":
            crop_width, crop_height = _maximum_exact_crop_size(width, height, p, q)
        else:
            crop_width, crop_height = _maximum_approximate_crop_size(
                width,
                height,
                p,
                q,
                float(geometry["max_ratio_error"]),
            )
        crop_box = _crop_box(width, height, crop_width, crop_height, anchor)
        plan = _SpatialPlan(source_size=decoded_size, crop_box=crop_box)
        retained_fraction = crop_width * crop_height / (width * height)
        metadata["crop_box"] = list(crop_box)
        metadata["retained_pixel_fraction"] = retained_fraction
        theoretical_area = (
            height * height * p / q
            if width * q >= height * p
            else width * width * q / p
        )
        metadata["integer_constraint_extra_loss"] = max(
            0.0, (theoretical_area - crop_width * crop_height) / (width * height)
        )
        cropped_size = crop_width, crop_height
    elif strategy == "native_pad":
        canvas_width, canvas_height = _minimum_exact_canvas_size(width, height, p, q)
        left, top, right, bottom = _crop_box(
            canvas_width, canvas_height, width, height, anchor
        )
        padding = left, canvas_width - right, top, canvas_height - bottom
        plan = _SpatialPlan(
            source_size=decoded_size,
            padding=padding,
            pad_value=float(geometry["pad_value"]),
        )
        metadata["pad"] = list(padding)
        metadata["retained_pixel_fraction"] = 1.0
        metadata["integer_constraint_extra_loss"] = 0.0
        cropped_size = canvas_width, canvas_height
    else:
        raise ValueError(f"Unsupported HunyuanImage-3 output strategy: {strategy}")

    ratio_error = abs((cropped_size[0] * q) / (cropped_size[1] * p) - 1.0)
    metadata["relative_ratio_error"] = ratio_error
    if size_mode == "exact_size":
        requested_size = tuple(int(value) for value in geometry["requested_size"])
        plan = _SpatialPlan(
            source_size=plan.source_size,
            crop_box=plan.crop_box,
            padding=plan.padding,
            pad_value=plan.pad_value,
            resize_size=requested_size,
        )
        metadata["resampled"] = cropped_size != requested_size
        if metadata["resampled"]:
            metadata["resample_mode"] = "bicubic_antialias"
        output_size = requested_size
    elif size_mode == "aspect_ratio":
        output_size = cropped_size
    else:
        raise ValueError(f"Unsupported HunyuanImage-3 output size mode: {size_mode}")
    metadata["output_size"] = list(output_size)
    return plan, metadata


def _resize_frames(frames: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
    output_width, output_height = output_size
    input_height, input_width = frames.shape[-2:]
    if (input_width, input_height) == output_size:
        return frames

    original_dtype = frames.dtype
    if frames.ndim == 5:
        batch_size, channels, num_frames, _, _ = frames.shape
        flat_frames = frames.permute(0, 2, 1, 3, 4).reshape(
            batch_size * num_frames, channels, input_height, input_width
        )
    elif frames.ndim == 4:
        batch_size, channels, _, _ = frames.shape
        num_frames = None
        flat_frames = frames
    else:
        raise ValueError(
            "HunyuanImage-3 decoded frames must be BCHW or BCFHW, "
            f"but got shape {tuple(frames.shape)}"
        )

    compute_dtype = torch.float64 if original_dtype == torch.float64 else torch.float32
    resized = F.interpolate(
        flat_frames.to(compute_dtype),
        size=(output_height, output_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).clamp_(0, 1)
    if original_dtype.is_floating_point:
        resized = resized.to(original_dtype)
    if num_frames is None:
        return resized
    return (
        resized.reshape(batch_size, num_frames, channels, output_height, output_width)
        .permute(0, 2, 1, 3, 4)
        .contiguous()
    )


def apply_hunyuan_image3_spatial_plan(
    frames: torch.Tensor, plan: _SpatialPlan
) -> torch.Tensor:
    """Apply the exact same post-decode spatial transform to images or paths."""
    if frames.ndim not in (4, 5):
        raise ValueError(
            "HunyuanImage-3 decoded frames must be BCHW or BCFHW, "
            f"but got shape {tuple(frames.shape)}"
        )
    source_width, source_height = plan.source_size
    if tuple(frames.shape[-2:]) != (source_height, source_width):
        raise ValueError(
            "HunyuanImage-3 output and trajectory decoded to different spatial "
            f"sizes: expected {source_width}x{source_height}, got "
            f"{frames.shape[-1]}x{frames.shape[-2]}"
        )
    if plan.crop_box is not None:
        left, top, right, bottom = plan.crop_box
        frames = frames[..., top:bottom, left:right]
    if plan.padding is not None:
        frames = F.pad(frames, plan.padding, value=plan.pad_value)
    if plan.resize_size is not None:
        frames = _resize_frames(frames, plan.resize_size)
    return frames


class HunyuanImage3DecodingStage(DecodingStage):
    """Decode a complete native bucket, then apply request-scoped geometry."""

    @torch.no_grad()
    def forward(self, batch: Req, server_args: ServerArgs) -> OutputBatch:
        output_batch = super().forward(batch, server_args)
        geometry = batch.extra.get(OUTPUT_GEOMETRY_EXTRA_KEY)
        if geometry is None or not isinstance(output_batch.output, torch.Tensor):
            return output_batch
        if not envs.SGLANG_HI3_OUTPUT_CROP:
            # Env-gated escape hatch: return the full decoded native bucket
            # without any crop/pad/resample. batch.width/height already hold
            # the bucket dims, and the geometry contract stays in extra for
            # introspection.
            return output_batch

        source_size = output_batch.output.shape[-1], output_batch.output.shape[-2]
        plan, metadata = _build_spatial_plan(source_size, geometry)
        output_batch.output = apply_hunyuan_image3_spatial_plan(
            output_batch.output, plan
        )
        if output_batch.trajectory_decoded is not None:
            output_batch.trajectory_decoded = [
                apply_hunyuan_image3_spatial_plan(frames, plan)
                for frames in output_batch.trajectory_decoded
            ]
        batch.width, batch.height = metadata["output_size"]
        batch.extra[OUTPUT_GEOMETRY_EXTRA_KEY] = metadata
        return output_batch
