# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import tempfile
from copy import copy
from typing import Any

from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.registry import get_pipeline_config_classes
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _parse_size,
    save_image_to_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

MINIMUM_PICTURE_BASE64_FOR_WARMUP = "data:image/jpg;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

DEFAULT_PLACEHOLDER_PROMPT = "warmup"
DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION = (64, 64)


def get_first_generation_req(req_or_group: Any) -> Req | None:
    """Extract the first req"""
    if isinstance(req_or_group, Req):
        return req_or_group
    if isinstance(req_or_group, list) and req_or_group:
        first_req = req_or_group[0]
        if isinstance(first_req, Req):
            return first_req
    return None


def is_warmup_req(req_or_group: Any) -> bool:
    """either server-based or req-based"""
    req = get_first_generation_req(req_or_group)
    return req.is_warmup if req is not None else False


def is_server_based_warmup(req_or_group: Any) -> bool:
    req = get_first_generation_req(req_or_group)
    return (
        req is not None and req.is_warmup and bool(req.extra.get("server_based_warmup"))
    )


def should_return_warmup_result(req_or_group: Any) -> bool:
    # server-based warmup needs to return to the http server to finish the startup
    req = get_first_generation_req(req_or_group)
    return (
        req is not None
        and req.is_warmup
        and bool(req.extra.get("return_warmup_result"))
    )


def get_model_sampling_defaults(server_args: ServerArgs) -> SamplingParams:
    pipeline_class_name = server_args.pipeline_class_name
    try:
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
    except Exception:
        logger.debug("Falling back to base SamplingParams for server warmup")
        return SamplingParams()


async def prepare_warmup_image_path(server_args: ServerArgs) -> str:
    if server_args.input_save_path is not None:
        uploads_dir = server_args.input_save_path
        os.makedirs(uploads_dir, exist_ok=True)
    else:
        uploads_dir = tempfile.mkdtemp(prefix="sglang_input_")

    warmup_image_base = os.path.join(uploads_dir, "warmup_image")
    return await save_image_to_path(
        MINIMUM_PICTURE_BASE64_FOR_WARMUP, warmup_image_base
    )


def prepare_warmup_image_path_sync(server_args: ServerArgs) -> str:
    return asyncio.run(prepare_warmup_image_path(server_args))


def _resolve_default_warmup_resolution(
    server_args: ServerArgs,
    sampling_defaults: SamplingParams,
) -> tuple[int, int]:
    supported_resolutions = sampling_defaults.supported_resolutions
    if supported_resolutions:
        return min(supported_resolutions, key=lambda size: size[0] * size[1])

    width = sampling_defaults.width
    height = sampling_defaults.height
    if width is not None and height is not None:
        return width, height

    if server_args.pipeline_config.task_type.is_image_gen():
        return DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION

    return (
        width or DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[0],
        height or DEFAULT_LIGHTWEIGHT_IMAGE_RESOLUTION[1],
    )


def _effective_cfg_scale(sampling_defaults: SamplingParams) -> float | None:
    if sampling_defaults.true_cfg_scale is not None:
        return sampling_defaults.true_cfg_scale
    return sampling_defaults.guidance_scale


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
    use_model_sampling_defaults: bool = False,
) -> list[Req]:
    task_type = server_args.pipeline_config.task_type
    if warmup_resolutions is None or use_model_sampling_defaults:
        sampling_defaults = get_model_sampling_defaults(server_args)
    else:
        sampling_defaults = SamplingParams()

    if warmup_resolutions is None:
        width, height = _resolve_default_warmup_resolution(
            server_args, sampling_defaults
        )
        resolutions: list[tuple[int, int]] = [(width, height)]
    else:
        resolutions = [_parse_size(resolution) for resolution in warmup_resolutions]

    negative_prompt: Any = (
        sampling_defaults.negative_prompt if use_model_sampling_defaults else None
    )
    cfg_scale = (
        _effective_cfg_scale(sampling_defaults) if use_model_sampling_defaults else None
    )

    warmup_reqs = []
    include_warmup_image = should_include_warmup_image(server_args, server_based_warmup)
    for width, height in resolutions:
        req_kwargs = dict(
            data_type=task_type.data_type(),
            width=width,
            height=height,
            prompt=DEFAULT_PLACEHOLDER_PROMPT,
        )
        if use_model_sampling_defaults:
            req_kwargs["sampling_params"] = copy(sampling_defaults)
            req_kwargs.update(
                negative_prompt=negative_prompt,
                guidance_scale=sampling_defaults.guidance_scale,
                guidance_scale_2=sampling_defaults.guidance_scale_2,
                true_cfg_scale=sampling_defaults.true_cfg_scale,
                num_inference_steps=sampling_defaults.num_inference_steps,
            )
        if include_warmup_image:
            if warmup_input_path is None:
                raise RuntimeError(
                    "Warmup image path is required for image-input model"
                )
            req_kwargs["prompt"] = DEFAULT_PLACEHOLDER_PROMPT
            if not use_model_sampling_defaults:
                req_kwargs["negative_prompt"] = ""
            req_kwargs["image_path"] = [warmup_input_path]
        if (
            server_args.enable_cfg_parallel
            and req_kwargs.get("negative_prompt") is None
        ):
            req_kwargs["negative_prompt"] = DEFAULT_PLACEHOLDER_PROMPT
            req_kwargs["do_classifier_free_guidance"] = True
        elif (
            use_model_sampling_defaults
            and negative_prompt is not None
            and cfg_scale is not None
            and cfg_scale > 1.0
        ):
            req_kwargs["do_classifier_free_guidance"] = True

        req = Req(**req_kwargs)
        req.set_as_warmup(server_args.warmup_steps)
        if return_warmup_result:
            req.extra["return_warmup_result"] = True
        if server_based_warmup:
            req.extra["server_based_warmup"] = True
        warmup_reqs.append(req)

    return warmup_reqs
