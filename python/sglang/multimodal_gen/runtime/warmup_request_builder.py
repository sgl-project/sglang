# SPDX-License-Identifier: Apache-2.0
"""Build synthetic diffusion warmup requests.

Default server warmup should cover a representative serving path before the
first real request, without copying user traffic. It starts from the model's
sampling defaults, then keeps startup bounded by choosing common low-cost
resolution/frame buckets and trimming the denoising step count.

Image models may run a tiny second step because first/last step paths often
initialize different kernels or scheduler state. Video models cap frames and
steps to keep startup bounded. Explicit warmup resolutions share this builder;
callers send them through the scheduler client so warmup exercises the same
request transport path as real generation.
"""

from copy import copy
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import (
    DataType,
    SamplingParams,
)
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import (
    ServerArgs,
    is_ltx2_two_stage_pipeline_name,
)
from sglang.multimodal_gen.runtime.utils.common import parse_size
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

DEFAULT_PLACEHOLDER_PROMPT = "warmup"
DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION = (64, 64)
SERVER_WARMUP_IMAGE_FALLBACK_RESOLUTION = (512, 512)
SERVER_WARMUP_VIDEO_FALLBACK_RESOLUTION = (832, 480)
SERVER_WARMUP_IMAGE_MAX_AREA = 768 * 768
SERVER_WARMUP_DIFFUSERS_IMAGE_MAX_AREA = 512 * 512
SERVER_WARMUP_VIDEO_MAX_AREA = 832 * 480
SERVER_WARMUP_MAX_VIDEO_FRAMES = 17
SERVER_WARMUP_IMAGE_STEPS = 2
SERVER_WARMUP_VIDEO_STEPS = 2


def get_model_sampling_defaults(server_args: ServerArgs) -> SamplingParams:
    pipeline_class_name = server_args.pipeline_class_name
    if pipeline_class_name:
        config_classes = get_pipeline_config_classes(pipeline_class_name)
        if config_classes is not None:
            _, sampling_params_cls = config_classes
            return sampling_params_cls()

    return SamplingParams.from_pretrained(
        server_args.model_path,
        backend=server_args.backend,
        model_id=server_args.model_id,
    )


def _resolve_default_warmup_resolution(
    server_args: ServerArgs,
    sampling_defaults: SamplingParams,
    *,
    server_based_warmup: bool,
) -> tuple[int, int]:
    """Return the default warmup resolution.

    Prefer the model's sampling-default resolution — the most likely real
    request shape — so warmup specializes kernels for it. Server-based image
    warmup used to shrink this to an area cap (``SERVER_WARMUP_IMAGE_MAX_AREA``,
    768x768) to bound startup, but that left a residual first-request
    cold-start when the real request is larger (e.g. 1024x1024 paid ~0.1s of
    first-shape kernel autotuning, measured on H100).
    """
    width = sampling_defaults.width
    height = sampling_defaults.height
    is_image_gen = server_args.pipeline_config.task_type.is_image_gen()
    if (
        width is not None
        and height is not None
        and (not server_based_warmup or is_image_gen)
    ):
        return width, height

    if server_based_warmup:
        return _resolve_representative_warmup_resolution(server_args, sampling_defaults)

    supported_resolutions = sampling_defaults.supported_resolutions
    if supported_resolutions:
        return min(supported_resolutions, key=lambda size: size[0] * size[1])

    if server_args.pipeline_config.task_type.is_image_gen():
        return DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION

    return (
        width or DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[0],
        height or DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[1],
    )


def _resolve_representative_warmup_resolution(
    server_args: ServerArgs,
    sampling_defaults: SamplingParams,
) -> tuple[int, int]:
    target_area = _target_warmup_area(server_args)
    alignment = _warmup_resolution_alignment(server_args)

    supported_resolution = _select_supported_warmup_resolution(
        sampling_defaults.supported_resolutions, target_area, alignment
    )
    if supported_resolution is not None:
        return supported_resolution

    width = sampling_defaults.width
    height = sampling_defaults.height
    if width is not None and height is not None:
        return _fit_resolution_to_area(width, height, target_area, alignment)

    width, height = _fallback_warmup_resolution(server_args)
    return _fit_resolution_to_area(width, height, target_area, alignment)


def _target_warmup_area(server_args: ServerArgs) -> int:
    if server_args.pipeline_config.task_type.is_image_gen():
        if getattr(server_args, "backend", None) == "diffusers":
            return SERVER_WARMUP_DIFFUSERS_IMAGE_MAX_AREA
        return SERVER_WARMUP_IMAGE_MAX_AREA
    if _is_video_warmup_task(server_args):
        return SERVER_WARMUP_VIDEO_MAX_AREA
    return (
        DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[0]
        * DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[1]
    )


def _fallback_warmup_resolution(server_args: ServerArgs) -> tuple[int, int]:
    if server_args.pipeline_config.task_type.is_image_gen():
        return SERVER_WARMUP_IMAGE_FALLBACK_RESOLUTION
    if _is_video_warmup_task(server_args):
        return SERVER_WARMUP_VIDEO_FALLBACK_RESOLUTION
    return DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION


def _is_video_warmup_task(server_args: ServerArgs) -> bool:
    return server_args.pipeline_config.task_type.data_type() == DataType.VIDEO


def _warmup_resolution_alignment(server_args: ServerArgs) -> int:
    pipeline_config = server_args.pipeline_config
    alignment = 16

    vae_stride = getattr(pipeline_config, "vae_stride", None)
    if vae_stride is not None:
        spatial_stride = (
            vae_stride[-2:] if isinstance(vae_stride, (tuple, list)) else (vae_stride,)
        )
        for stride in spatial_stride:
            alignment = max(alignment, int(stride))

    vae_scale_factor = getattr(pipeline_config, "vae_scale_factor", None)
    if vae_scale_factor is not None:
        alignment = max(alignment, int(vae_scale_factor))

    arch_config = getattr(
        getattr(pipeline_config, "vae_config", None), "arch_config", None
    )
    for attr in ("vae_scale_factor", "spatial_compression_ratio"):
        value = getattr(arch_config, attr, None)
        if value is not None:
            alignment = max(alignment, int(value))

    if is_ltx2_two_stage_pipeline_name(server_args.pipeline_class_name):
        vae_scale_factor = pipeline_config.vae_scale_factor
        alignment = max(alignment, 64, int(vae_scale_factor) * 2)

    return alignment


def _select_supported_warmup_resolution(
    supported_resolutions: list[tuple[int, int]] | None,
    target_area: int,
    alignment: int,
) -> tuple[int, int] | None:
    if not supported_resolutions:
        return None

    candidates = [
        resolution
        for resolution in supported_resolutions
        if resolution[0] * resolution[1] <= target_area
        and _is_resolution_aligned(resolution, alignment)
    ]
    if candidates:
        return max(candidates, key=lambda size: size[0] * size[1])

    aligned_resolutions = [
        resolution
        for resolution in supported_resolutions
        if _is_resolution_aligned(resolution, alignment)
    ]
    if aligned_resolutions:
        return min(aligned_resolutions, key=lambda size: size[0] * size[1])
    return None


def _fit_resolution_to_area(
    width: int, height: int, target_area: int, alignment: int
) -> tuple[int, int]:
    """adjust the warmup resolution to balance between warmup time and warmup effect"""
    area = width * height
    if area > target_area:
        scale = (target_area / area) ** 0.5
        width = int(width * scale)
        height = int(height * scale)

    return (
        max(alignment, width // alignment * alignment),
        max(alignment, height // alignment * alignment),
    )


def _is_resolution_aligned(resolution: tuple[int, int], alignment: int) -> bool:
    width, height = resolution
    return width % alignment == 0 and height % alignment == 0


def _resolve_warmup_num_frames(
    server_args: ServerArgs,
    sampling_defaults: SamplingParams,
    *,
    server_based_warmup: bool,
) -> int:
    num_frames = sampling_defaults.num_frames
    if (
        not server_based_warmup
        or not _is_video_warmup_task(server_args)
        or num_frames is None
    ):
        # use default num frames
        return num_frames

    return min(num_frames, SERVER_WARMUP_MAX_VIDEO_FRAMES)


def _effective_cfg_scale(sampling_defaults: SamplingParams) -> float | None:
    if sampling_defaults.true_cfg_scale is not None:
        return sampling_defaults.true_cfg_scale
    return sampling_defaults.guidance_scale


def _resolve_warmup_steps(
    server_args: ServerArgs,
    sampling_defaults: SamplingParams,
    *,
    server_based_warmup: bool,
) -> int:
    warmup_steps = server_args.warmup_steps
    default_steps = sampling_defaults.num_inference_steps

    # Breakable CUDA graph captures one graph per step-branch at warmup so that
    # serving never records a fresh graph. Run the model's full recommended
    # steps (uncapped) so every step-branch signature is captured up front.
    if (
        getattr(server_args, "enable_breakable_cuda_graph", False) is True
        and default_steps
    ):
        return max(int(default_steps), warmup_steps)

    if not server_based_warmup:
        return warmup_steps

    if default_steps is None or default_steps <= warmup_steps:
        return warmup_steps

    if _is_video_warmup_task(server_args):
        return min(default_steps, max(warmup_steps, SERVER_WARMUP_VIDEO_STEPS))

    if server_args.pipeline_config.task_type.is_image_gen():
        return min(default_steps, max(warmup_steps, SERVER_WARMUP_IMAGE_STEPS))

    return warmup_steps


def should_include_warmup_image(
    server_args: ServerArgs, server_based_warmup: bool
) -> bool:
    task_type = server_args.pipeline_config.task_type
    if not task_type.accepts_image_input():
        return False
    if task_type.requires_image_input():
        return True
    if server_based_warmup:
        return task_type in (ModelTaskType.TI2I, ModelTaskType.TI2V)
    return True


def build_warmup_reqs(
    server_args: ServerArgs,
    *,
    warmup_resolutions: list[str] | None,
    warmup_input_path: str | None = None,
    return_warmup_result: bool = False,
    server_based_warmup: bool = False,
) -> list[Req]:
    task_type = server_args.pipeline_config.task_type
    sampling_defaults = get_model_sampling_defaults(server_args)

    if warmup_resolutions is None:
        width, height = _resolve_default_warmup_resolution(
            server_args,
            sampling_defaults,
            server_based_warmup=server_based_warmup,
        )
        resolutions: list[tuple[int, int]] = [(width, height)]
    else:
        resolutions = [parse_size(resolution) for resolution in warmup_resolutions]

    negative_prompt: Any = sampling_defaults.negative_prompt
    cfg_scale = _effective_cfg_scale(sampling_defaults)
    warmup_steps = _resolve_warmup_steps(
        server_args,
        sampling_defaults,
        server_based_warmup=server_based_warmup,
    )
    warmup_num_frames = _resolve_warmup_num_frames(
        server_args,
        sampling_defaults,
        server_based_warmup=server_based_warmup,
    )

    # build warmup reqs
    warmup_reqs = []
    include_warmup_image = should_include_warmup_image(server_args, server_based_warmup)
    for width, height in resolutions:
        req_kwargs = dict(
            data_type=task_type.data_type(),
            width=width,
            height=height,
            prompt=DEFAULT_PLACEHOLDER_PROMPT,
        )
        req_kwargs["sampling_params"] = copy(sampling_defaults)
        req_kwargs.update(
            negative_prompt=negative_prompt,
            guidance_scale=sampling_defaults.guidance_scale,
            guidance_scale_2=sampling_defaults.guidance_scale_2,
            true_cfg_scale=sampling_defaults.true_cfg_scale,
            num_inference_steps=sampling_defaults.num_inference_steps,
            num_frames=warmup_num_frames,
        )
        if include_warmup_image:
            if warmup_input_path is None:
                raise RuntimeError(
                    "Warmup image path is required for image-input model"
                )
            req_kwargs["prompt"] = DEFAULT_PLACEHOLDER_PROMPT
            req_kwargs["image_path"] = [warmup_input_path]
        if server_args.enable_cfg_parallel:
            if not req_kwargs.get("negative_prompt"):
                req_kwargs["negative_prompt"] = DEFAULT_PLACEHOLDER_PROMPT
            req_kwargs["do_classifier_free_guidance"] = True
        elif negative_prompt is not None and cfg_scale is not None and cfg_scale > 1.0:
            req_kwargs["do_classifier_free_guidance"] = True

        req = Req(**req_kwargs)
        req.set_as_warmup(warmup_steps)
        if return_warmup_result:
            req.extra["return_warmup_result"] = True
        if server_based_warmup:
            req.extra["server_based_warmup"] = True
        warmup_reqs.append(req)

    return warmup_reqs
