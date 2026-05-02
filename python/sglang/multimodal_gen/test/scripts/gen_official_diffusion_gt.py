"""
Generate diffusion consistency GT from official/Diffusers pipelines.

This intentionally does not call SGLang native generation or SGLang's diffusers
backend. It reuses SGLang CI case definitions only to recover the exact request
sampling parameters, then executes the upstream Diffusers pipeline directly.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import tempfile
import types
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
from diffusers import DiffusionPipeline
from huggingface_hub import snapshot_download
from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import DataType
from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    DEFAULT_FPS,
    DEFAULT_VIDEO_SECONDS,
)
from sglang.multimodal_gen.runtime.entrypoints.utils import (
    post_process_sample,
    prepare_request,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.test.server.gpu_cases import (
    ONE_GPU_CASES,
    ONE_GPU_MODELOPT_CASES,
    TWO_GPU_CASES,
)
from sglang.multimodal_gen.test.server.test_server_utils import (
    download_image_from_url,
    parse_dimensions,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionTestCase,
)
from sglang.multimodal_gen.test.test_utils import (
    _consistency_gt_filenames,
    extract_key_frames_from_video,
    is_image_url,
    output_format_to_ext,
)

SUITE_CASES = {
    "1-gpu": ONE_GPU_CASES,
    "2-gpu": TWO_GPU_CASES,
    "1-gpu-b200": ONE_GPU_MODELOPT_CASES,
}

UNSUPPORTED_OFFICIAL_CASES = {
    "fastwan2_2_ti2v_5b": (
        "The HF repo declares WanDMDPipeline, but the current Diffusers package "
        "does not provide that class and the repo has no custom pipeline.py."
    ),
    "turbo_wan2_1_t2v_1.3b": (
        "The HF repo declares WanDMDPipeline, but the current Diffusers package "
        "does not provide that class and the repo has no custom pipeline.py."
    ),
    "wan2_2_ti2v_5b": (
        "The current Diffusers WanPipeline signature does not accept image input, "
        "so it would generate a T2V sample instead of the TI2V CI case."
    ),
    "zimage_image_t2i_multi_lora": (
        "Diffusers Z-Image LoRA conversion currently expects alpha keys that are "
        "not present in the CI LoRA checkpoints; keep the existing GT for this case."
    ),
}

WAN_OFFICIAL_CASES = {
    "wan2_1_t2v_1.3b": {
        "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
        "config_key": "t2v-1.3B",
    },
}

SAMPLING_KWARGS = (
    "prompt",
    "negative_prompt",
    "num_inference_steps",
    "guidance_scale",
    "guidance_scale_2",
    "true_cfg_scale",
    "guidance_rescale",
    "height",
    "width",
    "num_frames",
    "fps",
    "cfg_normalize",
    "use_en_prompt",
)


class UnsupportedCaseError(RuntimeError):
    pass


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[5],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, Image.Image):
        return f"<PIL.Image mode={value.mode} size={value.size}>"
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_extra_args(extra_args: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    tokens: list[str] = []
    for item in extra_args:
        tokens.extend(shlex.split(item))

    index = 0
    while index < len(tokens):
        arg = tokens[index]
        if not arg.startswith("--"):
            index += 1
            continue

        if "=" in arg:
            key, value = arg[2:].split("=", 1)
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            key = arg[2:]
            index += 1
            value = tokens[index]
        else:
            key = arg[2:]
            value = True

        parsed[key.replace("-", "_")] = value
        index += 1
    return parsed


def _server_args_for_case(case: DiffusionTestCase) -> ServerArgs:
    extra = _parse_extra_args(case.server_args.extras)
    kwargs: dict[str, Any] = {
        "model_path": case.server_args.model_path,
        "num_gpus": case.server_args.num_gpus,
        "trust_remote_code": True,
        "enable_cfg_parallel": bool(case.server_args.cfg_parallel),
    }
    for field in ("tp_size", "ulysses_degree", "ring_degree"):
        value = getattr(case.server_args, field)
        if value is not None:
            kwargs[field] = value
        elif field in extra:
            kwargs[field] = int(extra[field])
    if "pipeline_class_name" in extra:
        kwargs["pipeline_class_name"] = extra["pipeline_class_name"]

    server_args = ServerArgs.from_kwargs(**kwargs)
    server_args.enable_cache_dit = case.server_args.enable_cache_dit
    server_args.dit_layerwise_offload = case.server_args.dit_layerwise_offload
    server_args.dit_offload_prefetch_size = case.server_args.dit_offload_prefetch_size
    server_args.text_encoder_cpu_offload = case.server_args.text_encoder_cpu_offload
    return server_args


def _sampling_user_kwargs(
    sampling_params: DiffusionSamplingParams, output_size: str
) -> dict[str, Any]:
    kwargs: dict[str, Any] = dict(sampling_params.extras)
    for name in (
        "prompt",
        "image_path",
        "num_frames",
        "fps",
        "num_outputs_per_prompt",
    ):
        value = getattr(sampling_params, name)
        if value is not None:
            kwargs[name] = value

    width, height = parse_dimensions(output_size)
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    kwargs.setdefault("seed", 42)
    return kwargs


def _final_request_for_case(case: DiffusionTestCase):
    server_args = _server_args_for_case(case)
    output_size = os.environ.get(
        "SGLANG_TEST_OUTPUT_SIZE", case.sampling_params.output_size
    )
    user_kwargs = _sampling_user_kwargs(case.sampling_params, output_size)
    if case.server_args.modality == "video":
        seconds = case.sampling_params.seconds or DEFAULT_VIDEO_SECONDS
        fps = user_kwargs.get("fps") or DEFAULT_FPS
        user_kwargs.setdefault("fps", fps)
        user_kwargs.setdefault("num_frames", int(fps) * int(seconds))

    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

    sampling_params = SamplingParams.from_user_sampling_params_args(
        case.server_args.model_path,
        server_args,
        **user_kwargs,
    )
    return (
        prepare_request(server_args, sampling_params),
        sampling_params,
        server_args,
        output_size,
    )


def _torch_dtype(dtype_arg: str) -> torch.dtype:
    if dtype_arg == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_arg]


def _load_pipe(
    case: DiffusionTestCase,
    *,
    dtype: torch.dtype,
    device_map: str,
    cpu_offload: bool,
) -> DiffusionPipeline:
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(True)

    load_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
    }
    if device_map != "none":
        load_kwargs["device_map"] = device_map

    try:
        pipe = DiffusionPipeline.from_pretrained(
            case.server_args.model_path, **load_kwargs
        )
    except (TypeError, ValueError) as exc:
        if "device_map" not in load_kwargs:
            raise
        print(
            f"[official-gt] {case.id}: retrying without device_map after load error: {exc}",
            flush=True,
        )
        load_kwargs.pop("device_map", None)
        device_map = "none"
        pipe = DiffusionPipeline.from_pretrained(
            case.server_args.model_path, **load_kwargs
        )
    except AttributeError:
        load_kwargs["custom_pipeline"] = case.server_args.model_path
        load_kwargs["trust_remote_code"] = True
        pipe = DiffusionPipeline.from_pretrained(
            case.server_args.model_path, **load_kwargs
        )

    if device_map == "none" and not cpu_offload:
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        elif torch.backends.mps.is_available():
            pipe = pipe.to("mps")

    if cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    return pipe


def _call_signature(pipe: DiffusionPipeline) -> inspect.Signature | None:
    try:
        return inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return None


def _filter_kwargs(
    pipe: DiffusionPipeline, kwargs: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    sig = _call_signature(pipe)
    if sig is None:
        return kwargs, []

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs, []

    valid = set(params) - {"self"}
    kept = {k: v for k, v in kwargs.items() if k in valid}
    ignored = sorted(k for k in kwargs if k not in valid)
    return kept, ignored


def _load_input_images(
    image_path: Any, *, force_rgba: bool = False
) -> list[Image.Image]:
    if image_path is None:
        return []
    paths = image_path if isinstance(image_path, list) else [image_path]
    images: list[Image.Image] = []
    for path in paths:
        local_path = download_image_from_url(path) if is_image_url(path) else Path(path)
        image = Image.open(local_path)
        if force_rgba or image.mode in ("RGBA", "LA") or "transparency" in image.info:
            images.append(image.convert("RGBA"))
        else:
            images.append(image.convert("RGB"))
    return images


def _apply_sglang_condition_size(
    kwargs: dict[str, Any],
    input_images: list[Image.Image],
    server_args: ServerArgs,
    req: Any,
) -> None:
    """Mirror SGLang's image-conditioned output size adjustment for official runs."""
    if not input_images:
        return

    explicit_fields = set(getattr(req, "extra", {}).get("explicit_fields", []))
    if "width" in explicit_fields and "height" in explicit_fields:
        return

    config = server_args.pipeline_config
    if "flux" in server_args.model_path.lower():
        target_area = 1024 * 1024
        multiple_of = 16

        def calculate_size(image: Image.Image) -> tuple[int, int] | None:
            width, height = image.size
            if width * height > target_area:
                scale = math.sqrt(target_area / (width * height))
                width = int(width * scale)
                height = int(height * scale)
            width = width // multiple_of * multiple_of
            height = height // multiple_of * multiple_of
            if (width, height) != image.size:
                return width, height
            return None

        calculated_size = calculate_size(input_images[-1])
    else:
        calculate_size = lambda image: config.calculate_condition_image_size(
            image, image.width, image.height
        )
        calculated_size = config.prepare_calculated_size(input_images[-1])

    for i, image in enumerate(input_images):
        image_size = calculate_size(image)
        if image_size is not None and image.size != image_size:
            input_images[i] = image.resize(image_size, Image.Resampling.LANCZOS)
    if calculated_size is None:
        return

    calculated_width, calculated_height = calculated_size
    width = kwargs.get("width") if "width" in explicit_fields else calculated_width
    height = kwargs.get("height") if "height" in explicit_fields else calculated_height

    vae_arch = config.vae_config.arch_config
    vae_scale_factor = getattr(vae_arch, "vae_scale_factor", None)
    if vae_scale_factor is None:
        vae_scale_factor = vae_arch.spatial_compression_ratio
    multiple_of = vae_scale_factor * 2
    kwargs["width"] = width // multiple_of * multiple_of
    kwargs["height"] = height // multiple_of * multiple_of


def _generator(device_arg: str, seed: int) -> torch.Generator:
    if device_arg == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_arg
    return torch.Generator(device=device).manual_seed(seed)


def _apply_lora_if_needed(
    pipe: DiffusionPipeline, case: DiffusionTestCase
) -> dict[str, Any]:
    lora_path = case.server_args.lora_path or case.server_args.dynamic_lora_path
    if not lora_path:
        return {}

    if not hasattr(pipe, "load_lora_weights"):
        raise UnsupportedCaseError(
            f"{case.id}: official pipeline {type(pipe).__name__} has no load_lora_weights"
        )

    extra = _parse_extra_args(case.server_args.extras)
    kwargs = {}
    if "lora_weight_name" in extra:
        kwargs["weight_name"] = extra["lora_weight_name"]
    pipe.load_lora_weights(lora_path, **kwargs)
    return {"lora_path": lora_path, "lora_kwargs": kwargs}


def _build_call_kwargs(
    pipe: DiffusionPipeline,
    case: DiffusionTestCase,
    req: Any,
    sampling_params: Any,
    server_args: ServerArgs,
    *,
    generator_device: str,
) -> tuple[dict[str, Any], list[str]]:
    kwargs: dict[str, Any] = {}
    for name in SAMPLING_KWARGS:
        value = getattr(req, name, None)
        if value is not None:
            kwargs[name] = value

    seed = getattr(req, "seed", None)
    if seed is not None:
        kwargs["generator"] = _generator(generator_device, int(seed))

    if getattr(req, "num_outputs_per_prompt", 1) > 1:
        kwargs["num_images_per_prompt"] = req.num_outputs_per_prompt

    force_rgba = type(pipe).__name__ == "QwenImageLayeredPipeline"
    input_images = _load_input_images(
        getattr(req, "image_path", None), force_rgba=force_rgba
    )
    if input_images:
        _apply_sglang_condition_size(kwargs, input_images, server_args, req)
        sig = _call_signature(pipe)
        valid = set(sig.parameters) if sig is not None else set()
        if "images" in valid:
            kwargs["images"] = input_images
        elif "input_image" in valid:
            kwargs["input_image"] = input_images[0]
        else:
            kwargs["image"] = input_images if len(input_images) > 1 else input_images[0]

    diffusers_kwargs = getattr(sampling_params, "diffusers_kwargs", None)
    if diffusers_kwargs:
        kwargs.update(diffusers_kwargs)

    for key, value in case.sampling_params.extras.items():
        if key.startswith("enable_") or key.endswith("_scale") or key.endswith("_exp"):
            continue
        kwargs.setdefault(key, value)

    if "guidance_scale_2" in kwargs and getattr(pipe, "boundary_ratio", None) is None:
        kwargs.pop("guidance_scale_2")

    if (
        type(pipe).__name__.startswith("QwenImage")
        and "guidance_scale" in kwargs
        and "true_cfg_scale" not in kwargs
    ):
        kwargs["true_cfg_scale"] = kwargs["guidance_scale"]
        kwargs["guidance_scale"] = None

    kwargs.setdefault("output_type", "pil")
    return _filter_kwargs(pipe, kwargs)


def _build_dry_run_kwargs(case: DiffusionTestCase, req: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for name in SAMPLING_KWARGS:
        value = getattr(req, name, None)
        if value is not None:
            kwargs[name] = value
    if getattr(req, "seed", None) is not None:
        kwargs["generator"] = "<generator>"
    if getattr(req, "num_outputs_per_prompt", 1) > 1:
        kwargs["num_images_per_prompt"] = req.num_outputs_per_prompt
    if getattr(req, "image_path", None) is not None:
        kwargs["image"] = req.image_path
    for key, value in case.sampling_params.extras.items():
        if key.startswith("enable_") or key.endswith("_scale") or key.endswith("_exp"):
            continue
        kwargs.setdefault(key, value)
    if (
        case.server_args.model_path.lower().startswith("qwen/")
        and "guidance_scale" in kwargs
        and "true_cfg_scale" not in kwargs
    ):
        kwargs["true_cfg_scale"] = kwargs["guidance_scale"]
        kwargs["guidance_scale"] = None
    kwargs["output_type"] = "pil"
    return kwargs


def _to_uint8_rgb(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        return np.array(value.convert("RGB"))
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 3 and value.shape[0] in (1, 3, 4):
            value = value.permute(1, 2, 0)
        arr = value.float().numpy()
    else:
        arr = np.asarray(value)

    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if arr.min(initial=0) < 0 and arr.max(initial=0) <= 1.0:
            arr = (arr + 1.0) / 2.0 * 255.0
        elif arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr[..., :3]


def _flatten_output(output: Any) -> list[np.ndarray]:
    for attr in ("frames", "videos", "images", "sample"):
        value = getattr(output, attr, None)
        if value is not None:
            output = value
            break

    if isinstance(output, torch.Tensor):
        tensor = output.detach().cpu()
        if tensor.ndim == 5:
            tensor = tensor[0]
            if tensor.shape[0] in (1, 3, 4):
                tensor = tensor.permute(1, 2, 3, 0)
        elif tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor[0]
        elif tensor.ndim == 4 and tensor.shape[0] in (3, 4):
            tensor = tensor.permute(1, 2, 3, 0)
        if tensor.ndim == 4:
            return [_to_uint8_rgb(frame) for frame in tensor]
        return [_to_uint8_rgb(tensor)]

    if isinstance(output, np.ndarray):
        if output.ndim == 5:
            output = output[0]
        if output.ndim == 4:
            return [_to_uint8_rgb(frame) for frame in output]
        return [_to_uint8_rgb(output)]

    if isinstance(output, list):
        if output and isinstance(output[0], list):
            output = output[0]
        return [_to_uint8_rgb(item) for item in output]

    return [_to_uint8_rgb(output)]


def _postprocess_frames(
    frames: list[np.ndarray], case: DiffusionTestCase, req: Any, is_video: bool
) -> list[np.ndarray]:
    extras = case.sampling_params.extras
    data_type = DataType.VIDEO if is_video else DataType.IMAGE
    sample = np.stack(frames) if is_video or len(frames) > 1 else frames[0]
    return post_process_sample(
        sample,
        data_type=data_type,
        fps=getattr(req, "fps", 24) or 24,
        save_output=False,
        enable_frame_interpolation=bool(extras.get("enable_frame_interpolation")),
        frame_interpolation_exp=int(extras.get("frame_interpolation_exp", 1)),
        frame_interpolation_scale=float(extras.get("frame_interpolation_scale", 1.0)),
        frame_interpolation_model_path=extras.get("frame_interpolation_model_path"),
        enable_upscaling=bool(extras.get("enable_upscaling")),
        upscaling_model_path=extras.get("upscaling_model_path"),
        upscaling_scale=int(extras.get("upscaling_scale", 4)),
    )


def _extract_ci_encoded_video_key_frames(
    frames: list[np.ndarray], req: Any
) -> list[np.ndarray]:
    output_compression = getattr(req, "output_compression", None)
    quality = output_compression / 10 if output_compression is not None else 5

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        imageio.mimsave(
            tmp_path,
            frames,
            fps=getattr(req, "fps", 24) or 24,
            format=DataType.VIDEO.get_default_extension(),
            codec="libx264",
            quality=quality,
        )
        return extract_key_frames_from_video(Path(tmp_path).read_bytes())
    finally:
        os.unlink(tmp_path)


def _save_gt_frames(
    frames: list[np.ndarray],
    case: DiffusionTestCase,
    out_dir: Path,
    *,
    is_video: bool,
    req: Any,
) -> list[str]:
    num_gpus = case.server_args.num_gpus
    saved: list[str] = []
    if is_video:
        selected = _extract_ci_encoded_video_key_frames(frames, req)
        filenames = _consistency_gt_filenames(case.id, num_gpus, is_video=True)
        for frame, filename in zip(selected, filenames):
            Image.fromarray(frame).save(out_dir / filename)
            saved.append(filename)
        return saved

    ext = output_format_to_ext(case.sampling_params.output_format)
    filenames = _consistency_gt_filenames(
        case.id, num_gpus, is_video=False, output_format=ext
    )
    save_kwargs = {"quality": 75} if ext in ("jpg", "jpeg") else {}
    Image.fromarray(frames[0]).save(out_dir / filenames[0], **save_kwargs)
    saved.append(filenames[0])
    return saved


def _select_cases(args: argparse.Namespace) -> list[DiffusionTestCase]:
    cases = [case for case in SUITE_CASES[args.suite] if case.run_consistency_check]
    if args.case_ids:
        wanted = set(args.case_ids)
        cases = [case for case in cases if case.id in wanted]

    if args.partition_id is not None:
        cases = [
            case
            for index, case in enumerate(cases)
            if index % args.total_partitions == args.partition_id
        ]
    return cases


def _run_case(
    case: DiffusionTestCase,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    req, sampling_params, server_args, output_size = _final_request_for_case(case)
    if case.id in WAN_OFFICIAL_CASES:
        if args.dry_run:
            case_info = WAN_OFFICIAL_CASES[case.id]
            return {
                "case_id": case.id,
                "model_path": case.server_args.model_path,
                "official_repo_id": case_info["repo_id"],
                "pipeline_class": "WanT2V",
                "output_size": output_size,
                "device_map": "wan-official",
                "call_kwargs": _jsonable(
                    _build_wan_official_dry_run_kwargs(req, server_args)
                ),
                "ignored_kwargs": [],
                "lora": {},
                "dry_run": True,
            }
        return _run_wan_official_case(
            case, req, server_args, output_size, out_dir, args
        )

    is_video = case.server_args.modality == "video"
    device_map = args.device_map
    if device_map == "case":
        device_map = "balanced" if case.server_args.num_gpus > 1 else "none"

    if args.dry_run:
        return {
            "case_id": case.id,
            "model_path": case.server_args.model_path,
            "pipeline_class": "<not loaded>",
            "output_size": output_size,
            "device_map": device_map,
            "call_kwargs": _jsonable(_build_dry_run_kwargs(case, req)),
            "ignored_kwargs": [],
            "lora": {
                "lora_path": case.server_args.lora_path
                or case.server_args.dynamic_lora_path
            },
            "dry_run": True,
        }

    pipe = _load_pipe(
        case,
        dtype=_torch_dtype(args.dtype),
        device_map=device_map,
        cpu_offload=args.cpu_offload,
    )
    lora_info = _apply_lora_if_needed(pipe, case)
    call_kwargs, ignored_kwargs = _build_call_kwargs(
        pipe,
        case,
        req,
        sampling_params,
        server_args,
        generator_device=args.generator_device,
    )

    with torch.inference_mode():
        output = pipe(**call_kwargs)
    frames = _flatten_output(output)
    frames = _postprocess_frames(frames, case, req, is_video)
    saved_files = _save_gt_frames(frames, case, out_dir, is_video=is_video, req=req)

    return {
        "case_id": case.id,
        "model_path": case.server_args.model_path,
        "pipeline_class": type(pipe).__name__,
        "output_size": output_size,
        "device_map": device_map,
        "saved_files": saved_files,
        "num_frames_after_postprocess": len(frames),
        "call_kwargs": {
            k: ("<generator>" if k == "generator" else _jsonable(v))
            for k, v in call_kwargs.items()
        },
        "ignored_kwargs": ignored_kwargs,
        "lora": lora_info,
        "dry_run": False,
    }


def _resolve_wan_official_repo_dir(args: argparse.Namespace) -> Path:
    repo_dir = (
        args.wan_official_repo_dir
        or os.environ.get("WAN_OFFICIAL_REPO_DIR")
        or str(_repo_root() / "3rdparty" / "Wan2.1")
    )
    path = Path(repo_dir).expanduser().resolve()
    if not (path / "wan" / "text2video.py").exists():
        raise UnsupportedCaseError(
            "Wan official repo is required for Wan official GT. "
            f"Expected {path}/wan/text2video.py; pass --wan-official-repo-dir."
        )
    return path


def _download_wan_official_checkpoint(case_info: dict[str, str]) -> str:
    return snapshot_download(
        case_info["repo_id"],
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "*.pth",
            "google/umt5-xxl/*",
        ],
    )


def _install_wan_official_compat_modules() -> None:
    try:
        import flash_attn
    except ModuleNotFoundError:
        flash_attn = None

    if flash_attn is not None and not hasattr(flash_attn, "flash_attn_varlen_func"):

        def _torch_flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
            deterministic=False,
        ):
            if window_size != (-1, -1):
                raise NotImplementedError(
                    "Wan official GT fallback only supports full attention"
                )

            outputs = []
            batch_size = cu_seqlens_q.numel() - 1
            for i in range(batch_size):
                q_start = int(cu_seqlens_q[i].item())
                q_end = int(cu_seqlens_q[i + 1].item())
                k_start = int(cu_seqlens_k[i].item())
                k_end = int(cu_seqlens_k[i + 1].item())

                qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
                ki = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
                vi = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
                if qi.shape[1] != ki.shape[1]:
                    repeat = qi.shape[1] // ki.shape[1]
                    ki = ki.repeat_interleave(repeat, dim=1)
                    vi = vi.repeat_interleave(repeat, dim=1)

                out = torch.nn.functional.scaled_dot_product_attention(
                    qi,
                    ki,
                    vi,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                )
                outputs.append(out.squeeze(0).transpose(0, 1))
            return torch.cat(outputs, dim=0)

        flash_attn.flash_attn_varlen_func = _torch_flash_attn_varlen_func

    if "easydict" not in sys.modules:
        module = types.ModuleType("easydict")

        class EasyDict(dict):
            def __getattr__(self, name):
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def __setattr__(self, name, value):
                self[name] = value

            def __delattr__(self, name):
                try:
                    del self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

        module.EasyDict = EasyDict
        sys.modules["easydict"] = module
    if "ftfy" not in sys.modules:
        module = types.ModuleType("ftfy")
        module.fix_text = lambda text: text
        sys.modules["ftfy"] = module


def _build_wan_official_dry_run_kwargs(
    req: Any, server_args: ServerArgs
) -> dict[str, Any]:
    return {
        "prompt": req.prompt,
        "size": [req.width, req.height],
        "frame_num": req.num_frames,
        "shift": server_args.pipeline_config.flow_shift,
        "sample_solver": "unipc",
        "sampling_steps": req.num_inference_steps,
        "guide_scale": req.guidance_scale,
        "n_prompt": req.negative_prompt,
        "seed": req.seed,
        "offload_model": True,
    }


def _run_wan_official_case(
    case: DiffusionTestCase,
    req: Any,
    server_args: ServerArgs,
    output_size: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_info = WAN_OFFICIAL_CASES[case.id]
    repo_dir = _resolve_wan_official_repo_dir(args)
    _install_wan_official_compat_modules()
    sys.path.insert(0, str(repo_dir))
    try:
        from wan.configs import WAN_CONFIGS
        from wan.text2video import WanT2V
    finally:
        sys.path.pop(0)

    checkpoint_dir = _download_wan_official_checkpoint(case_info)
    cfg = WAN_CONFIGS[case_info["config_key"]]
    seed = int(getattr(req, "seed", 42))
    width = int(req.width)
    height = int(req.height)
    num_frames = int(req.num_frames)
    num_inference_steps = int(req.num_inference_steps)
    guidance_scale = float(req.guidance_scale)
    negative_prompt = req.negative_prompt or cfg.sample_neg_prompt
    shift = float(server_args.pipeline_config.flow_shift)

    pipe = WanT2V(
        config=cfg,
        checkpoint_dir=checkpoint_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
    )
    output = pipe.generate(
        req.prompt,
        size=(width, height),
        frame_num=num_frames,
        shift=shift,
        sample_solver="unipc",
        sampling_steps=num_inference_steps,
        guide_scale=guidance_scale,
        n_prompt=negative_prompt,
        seed=seed,
        offload_model=True,
    )
    frames = _flatten_output(output)
    frames = _postprocess_frames(frames, case, req, is_video=True)
    saved_files = _save_gt_frames(frames, case, out_dir, is_video=True, req=req)
    return {
        "case_id": case.id,
        "model_path": case.server_args.model_path,
        "official_repo_id": case_info["repo_id"],
        "official_repo_dir": str(repo_dir),
        "checkpoint_dir": checkpoint_dir,
        "pipeline_class": "WanT2V",
        "output_size": output_size,
        "device_map": "wan-official",
        "saved_files": saved_files,
        "num_frames_after_postprocess": len(frames),
        "call_kwargs": {
            "prompt": req.prompt,
            "size": [width, height],
            "frame_num": num_frames,
            "shift": shift,
            "sample_solver": "unipc",
            "sampling_steps": num_inference_steps,
            "guide_scale": guidance_scale,
            "n_prompt": negative_prompt,
            "seed": seed,
            "offload_model": True,
        },
        "ignored_kwargs": [],
        "lora": {},
        "dry_run": False,
    }


def _cleanup_accelerators() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _manifest_path(out_dir: Path, args: argparse.Namespace) -> Path:
    suffix = args.suite.replace("-", "_")
    if args.partition_id is not None:
        suffix = f"{suffix}_part{args.partition_id}_of_{args.total_partitions}"
    return out_dir / f"official_gt_manifest_{suffix}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(SUITE_CASES), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--partition-id", type=int)
    parser.add_argument("--total-partitions", type=int)
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--device-map",
        choices=("case", "none", "auto", "balanced"),
        default="case",
        help="case uses balanced only for multi-GPU cases; none calls pipe.to(device).",
    )
    parser.add_argument("--generator-device", default="auto")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument(
        "--wan-official-repo-dir",
        help="Path to the official Wan2.1 repository checkout for Wan official GT.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if (args.partition_id is None) != (args.total_partitions is None):
        parser.error("--partition-id and --total-partitions must be provided together")
    if args.partition_id is not None and not (
        0 <= args.partition_id < args.total_partitions
    ):
        parser.error("--partition-id must be in [0, total-partitions)")

    cases = _select_cases(args)
    if args.list_cases:
        for case in cases:
            print(case.id)
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generator": "official-diffusers",
        "git_sha": _git_sha(),
        "suite": args.suite,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cases": [],
        "skipped": [],
        "failures": [],
    }
    manifest_path = _manifest_path(out_dir, args)

    for case in cases:
        if case.id in UNSUPPORTED_OFFICIAL_CASES:
            skip = {"case_id": case.id, "reason": UNSUPPORTED_OFFICIAL_CASES[case.id]}
            manifest["skipped"].append(skip)
            print(f"[official-gt] SKIPPED {case.id}: {skip['reason']}", flush=True)
            continue

        print(f"[official-gt] generating {case.id}", flush=True)
        try:
            manifest["cases"].append(_run_case(case, out_dir, args))
        except Exception as exc:
            failure = {
                "case_id": case.id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            manifest["failures"].append(failure)
            print(f"[official-gt] FAILED {case.id}: {failure}", flush=True)
            if not args.continue_on_error:
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                raise
        finally:
            _cleanup_accelerators()

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if manifest["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
