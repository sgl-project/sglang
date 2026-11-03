# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 adapter for the generic action endpoint."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def cosmos3_action_metadata(server_args: ServerArgs) -> dict[str, Any]:
    pipeline_config = server_args.pipeline_config
    defaults = Cosmos3SamplingParams()
    max_batch_size = max(1, int(getattr(server_args, "batching_max_size", 1)))
    return {
        "object": "action.metadata",
        "model": server_args.served_model_name,
        "model_path": server_args.model_path,
        "policy_family": "cosmos3",
        "input": {
            "modalities": ["image", "video"],
            "supported_resolutions": [
                list(resolution) for resolution in defaults.supported_resolutions
            ],
            "state_dim": None,
        },
        "output": {
            "action_type": "continuous",
            "action_horizon": 16,
            "action_dim": None,
            "padded_action_dim": pipeline_config.dit_config.arch_config.action_dim,
            "dtype": "float32",
        },
        "runtime": {
            "parallelism": {
                "num_gpus": server_args.num_gpus,
                "tp_size": server_args.tp_size,
                "sp_degree": server_args.sp_degree,
                "ulysses_degree": server_args.ulysses_degree,
                "ring_degree": server_args.ring_degree,
            }
        },
        "defaults": {
            "action_mode": "policy",
            "action_horizon": 16,
            "num_inference_steps": defaults.num_inference_steps,
            "height": 480,
            "width": 832,
            "fps": 5,
        },
        "capabilities": {
            "action_modes": ["policy", "inverse_dynamics"],
            "realtime_websocket": True,
            "openpi_websocket": False,
            "batch_inputs": max_batch_size > 1,
            "max_batch_size": max_batch_size,
            "batched_action_modes": ["policy"],
            "multiple_candidates": False,
        },
    }


def _images_from_observation(observation: dict[str, Any]) -> list[Any]:
    image = None
    for name in ("image", "image_path", "input_reference"):
        if name in observation:
            image = observation[name]
            break

    if image is None:
        images = observation.get("images")
        if images is None or (isinstance(images, dict) and not images):
            return []
        if not isinstance(images, dict) or len(images) != 1:
            raise ValueError(
                "Cosmos3 action input accepts one image field; use a list or "
                "a [B, H, W, C] array in that field for batched observations"
            )
        image = next(iter(images.values()))

    if isinstance(image, (list, tuple)):
        images = list(image)
    elif isinstance(image, np.ndarray) and image.ndim == 4:
        images = list(image)
    else:
        images = [image]

    normalized_images: list[Any] = []
    for item in images:
        if not isinstance(item, np.ndarray):
            normalized_images.append(item)
            continue
        if item.dtype != np.uint8:
            raise ValueError("Cosmos3 observation image arrays must use uint8 dtype")
        if item.ndim not in (2, 3):
            raise ValueError(
                "Cosmos3 observation image arrays must have shape [H, W] "
                f"or [H, W, C], got {tuple(item.shape)}"
            )
        normalized_images.append(Image.fromarray(item))
    return normalized_images


def _action_prompt(prompt: Any, batch_size: int) -> str | list[str]:
    if isinstance(prompt, str):
        return prompt if batch_size == 1 else [prompt] * batch_size
    if not isinstance(prompt, (list, tuple)) or not prompt:
        raise ValueError("Cosmos3 action prompt must be a string or non-empty list")
    if not all(isinstance(item, str) for item in prompt):
        raise ValueError("Cosmos3 action prompt list must contain only strings")
    prompts = list(prompt)
    if len(prompts) == 1 and batch_size > 1:
        prompts *= batch_size
    if len(prompts) != batch_size:
        raise ValueError(
            "Cosmos3 batched action input requires one prompt per image, got "
            f"{len(prompts)} prompt(s) and {batch_size} image(s)"
        )
    return prompts[0] if batch_size == 1 else prompts


def build_cosmos3_action_sampling_params(
    payload: dict[str, Any],
    observation: dict[str, Any],
    server_args: ServerArgs,
    sampling_params_cls: type[Cosmos3SamplingParams],
) -> Cosmos3SamplingParams:
    parameters = dict(payload.get("parameters") or {})
    options = {**observation, **parameters}
    action_mode = str(options.get("action_mode", "policy")).strip().lower()
    if action_mode == "forward_dynamics":
        raise ValueError(
            "Cosmos3 forward_dynamics produces video; use /v1/videos instead"
        )
    if action_mode not in ("policy", "inverse_dynamics"):
        raise ValueError(
            "Cosmos3 action endpoint supports action_mode='policy' or "
            "'inverse_dynamics'"
        )

    action_horizon = options.get("action_horizon")
    num_frames = options.get("num_frames")
    if action_horizon is None and num_frames is None:
        action_horizon = 16
    if action_horizon is not None:
        action_horizon = int(action_horizon)
        if action_horizon <= 0:
            raise ValueError("action_horizon must be a positive integer")
        expected_num_frames = action_horizon + 1
        if num_frames is not None and int(num_frames) != expected_num_frames:
            raise ValueError(
                "Cosmos3 requires num_frames == action_horizon + 1, got "
                f"num_frames={num_frames}, action_horizon={action_horizon}"
            )
        num_frames = expected_num_frames
    else:
        num_frames = int(num_frames)
        if num_frames <= 1:
            raise ValueError("Cosmos3 action num_frames must be greater than 1")
    if (num_frames - 1) % 4 != 0:
        raise ValueError(
            "Cosmos3 action_horizon must be divisible by 4 so num_frames "
            "is compatible with the temporal VAE"
        )

    images = _images_from_observation(observation)
    video_path = options.get("video_path") or observation.get("video")
    if action_mode == "policy" and not images:
        raise ValueError("Cosmos3 policy input requires an observation image")
    if action_mode == "inverse_dynamics" and video_path is None:
        raise ValueError("Cosmos3 inverse_dynamics input requires an observation video")
    if images and video_path is not None:
        raise ValueError("Cosmos3 action requests accept either an image or a video")
    batch_size = len(images) if images else 1
    max_batch_size = max(1, int(getattr(server_args, "batching_max_size", 1)))
    if batch_size > max_batch_size:
        raise ValueError(
            f"Cosmos3 action batch size {batch_size} exceeds "
            f"--batching-max-size={max_batch_size}"
        )
    image_path = None if not images else images[0] if batch_size == 1 else images

    domain_id = options.get("domain_id")
    domain_name = options.get("domain_name")
    raw_action_dim = options.get("raw_action_dim")
    if domain_id is None and not domain_name:
        raise ValueError("Cosmos3 action requests require domain_name or domain_id")
    if domain_id is not None and not domain_name and raw_action_dim is None:
        raise ValueError("raw_action_dim is required when only domain_id is provided")

    prompt = observation.get("prompt")
    if prompt is None:
        prompt = observation.get("task", "")
    if action_mode != "policy" and not isinstance(prompt, str):
        raise ValueError("Cosmos3 inverse_dynamics prompt must be a string")
    prompt = _action_prompt(prompt, batch_size)
    sampling_kwargs = {
        "request_id": payload.get("request_id") or payload.get("id"),
        "prompt": prompt,
        "image_path": image_path,
        "video_path": video_path,
        "action_mode": action_mode,
        "domain_id": domain_id,
        "domain_name": domain_name,
        "raw_action_dim": raw_action_dim,
        "action_fps": options.get("action_fps"),
        "action_view_point": options.get("action_view_point", "ego_view"),
        "action_normalization": options.get("action_normalization", "quantile"),
        "action_stats_path": server_args.pipeline_config.action_stats_path,
        "num_frames": num_frames,
        "fps": int(options.get("fps", 5)),
        "height": int(options.get("height", 480)),
        "width": int(options.get("width", 832)),
        "num_inference_steps": int(options.get("num_inference_steps", 35)),
        "guidance_scale": float(options.get("guidance_scale", 1.0)),
        "seed": int(options.get("seed", 42)),
        "flow_shift": options.get("flow_shift"),
        "max_sequence_length": options.get("max_sequence_length"),
        "condition_frame_indexes": options.get("condition_frame_indexes"),
        "condition_video_keep": options.get("condition_video_keep", "first"),
        "use_duration_template": False,
        "use_system_prompt": False,
        "use_guardrails": options.get("use_guardrails"),
        "save_output": False,
        "return_file_paths_only": False,
        "return_frames": False,
    }
    supported_fields = {field.name for field in dataclasses.fields(sampling_params_cls)}
    sampling_params = sampling_params_cls(
        **{
            name: value
            for name, value in sampling_kwargs.items()
            if name in supported_fields and value is not None
        }
    )
    sampling_params._adjust(server_args)
    return sampling_params
