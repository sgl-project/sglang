# SPDX-License-Identifier: Apache-2.0
"""Build an inference-only AdaLN cache from a local MiniMax H3 transformer."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.denoise_loop import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.time_request import (
    minimax_h3_image_pdd_grid,
    minimax_h3_time_shift_sigmas,
)

_HIDDEN_SIZE = 5376
_TIMESTEP_INPUT_DIM = 256
_NUM_BLOCKS = 50
_BLOCK_PARAM_WIDTH = 18 * _HIDDEN_SIZE
_FINAL_PARAM_WIDTH = 2 * _HIDDEN_SIZE
_CACHE_MODES = {
    "t2va": ("video", "audio"),
    "fl2va": ("video", "audio", "image"),
    "ref2va-image": ("video", "audio", "image"),
    "ref2va-audio": ("video", "audio", "audio_ref"),
    "ref2va-mixed": ("video", "audio", "image", "audio_ref"),
    # One image sidecar covers both declared tasks; see _image_timestep_plans.
    "image": ("video", "image"),
}
_IMAGE_TASK_FIELDS = {
    "t2i": ("video",),
    "edit": ("video", "image"),
}
_MODE_VARIANTS = {
    "t2va": "fl2va",
    "fl2va": "fl2va",
    "ref2va-image": "ref2va",
    "ref2va-audio": "ref2va",
    "ref2va-mixed": "ref2va",
    "image": "image",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MiniMax H3 inference-only AdaLN cache from local weights."
    )
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-variant", choices=("fl2va", "ref2va", "image"), required=True
    )
    parser.add_argument("--mode", choices=tuple(_CACHE_MODES), default="t2va")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
    parser.add_argument(
        "--pdd-config",
        type=Path,
        help="Image releases list their distilled sigma grid here; this "
        "replaces --num-inference-steps/--flow-shift.",
    )
    parser.add_argument(
        "--imgvid-cond-noise-aug",
        type=float,
        default=MINIMAX_H3_IMGVID_COND_TIMESTEP,
    )
    parser.add_argument(
        "--audio-cond-noise-aug",
        type=float,
        default=MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    )
    parser.add_argument(
        "--timesteps",
        type=float,
        nargs="+",
        help="Override the scheduler-derived timestep plan.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cache_timestep_plans(args: argparse.Namespace) -> list[torch.Tensor]:
    if args.timesteps is not None:
        return [torch.tensor(args.timesteps, dtype=torch.float32).unique(sorted=True)]

    if args.pdd_config is not None:
        # Image releases step the distilled grid; no audio branch, so both
        # sigma lists are the same.
        with args.pdd_config.open() as f:
            video_sigmas = list(minimax_h3_image_pdd_grid(json.load(f)))
        audio_sigmas = video_sigmas
    else:
        video_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=args.num_inference_steps,
            shift_scale=args.flow_shift,
        )
        audio_sigmas = minimax_h3_time_shift_sigmas(
            num_steps=args.num_inference_steps,
            shift_scale=args.audio_flow_shift,
        )
    if args.mode == "image":
        return _dedupe(
            _image_timestep_plans(
                video_sigmas, cond_timestep=args.imgvid_cond_noise_aug
            )
        )

    fields = _CACHE_MODES[args.mode]
    plans = []
    for video_sigma, audio_sigma in zip(video_sigmas[:-1], audio_sigmas[:-1]):
        video_timestep = 1.0 - video_sigma
        audio_timestep = 1.0 - audio_sigma
        candidates = {
            "video": video_timestep,
            "audio": audio_timestep,
            "image": max(video_timestep, args.imgvid_cond_noise_aug),
            "audio_ref": max(audio_timestep, args.audio_cond_noise_aug),
        }
        plans.append(
            torch.tensor(
                [candidates[field] for field in fields], dtype=torch.float32
            ).unique(sorted=True)
        )
    return _dedupe(plans)


def _image_timestep_plans(
    sigmas: list[float], *, cond_timestep: float
) -> list[torch.Tensor]:
    """Plans the image loop looks up: t2i the step's timestep, edit also the
    pinned reference timestep (``pin_imgvid_cond``)."""
    plans = []
    for sigma in sigmas[:-1]:
        timestep = 1.0 - sigma
        candidates = {"video": timestep, "image": cond_timestep}
        for fields in _IMAGE_TASK_FIELDS.values():
            plans.append(
                torch.tensor(
                    [candidates[field] for field in fields], dtype=torch.float32
                ).unique(sorted=True)
            )
    return plans


def _dedupe(plans: list[torch.Tensor]) -> list[torch.Tensor]:
    deduplicated = []
    seen = set()
    for plan in plans:
        key = tuple(plan.tolist())
        if key not in seen:
            seen.add(key)
            deduplicated.append(plan)
    return deduplicated


def _time_embed(
    timesteps: torch.Tensor,
    *,
    proj_in_weight: torch.Tensor,
    proj_in_bias: torch.Tensor,
    proj_out_weight: torch.Tensor,
    proj_out_bias: torch.Tensor,
) -> torch.Tensor:
    half = _TIMESTEP_INPUT_DIM // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None] * freqs[None]
    t_freq = torch.cat((torch.cos(args), torch.sin(args)), dim=-1)
    return F.linear(
        F.silu(F.linear(t_freq, proj_in_weight, proj_in_bias)),
        proj_out_weight,
        proj_out_bias,
    )


def _load_tensor(
    name: str,
    *,
    weight_map: dict[str, str],
    files: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    tensor_file = files[weight_map[name]]
    return tensor_file.get_tensor(name).to(device)


def main() -> None:
    args = _parse_args()
    if args.num_inference_steps < 1 and args.timesteps is None:
        raise ValueError("--num-inference-steps must be at least 1")
    mode_variant = _MODE_VARIANTS[args.mode]
    if args.model_variant != mode_variant:
        raise ValueError(f"--mode {args.mode} requires {mode_variant}")
    if (args.model_variant == "image") != (args.pdd_config is not None):
        raise ValueError(
            "--model-variant image takes the distilled grid from --pdd-config "
            "(the release's pdd_config.json); no other variant accepts one"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("MiniMax H3 AdaLN cache must be built on CUDA")

    index_path = args.transformer_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    plans = _cache_timestep_plans(args)
    if not plans or any(plan.numel() == 0 for plan in plans):
        raise ValueError("AdaLN cache must cover at least one timestep plan")
    max_plan_length = max(plan.numel() for plan in plans)
    plan_timesteps = torch.zeros((len(plans), max_plan_length), dtype=torch.float32)
    plan_lengths = torch.tensor([plan.numel() for plan in plans], dtype=torch.int64)
    block_params = torch.empty(
        (len(plans), max_plan_length, _NUM_BLOCKS, _BLOCK_PARAM_WIDTH),
        dtype=torch.bfloat16,
    )
    final_params = torch.empty(
        (len(plans), max_plan_length, _FINAL_PARAM_WIDTH), dtype=torch.bfloat16
    )

    with ExitStack() as stack:
        files = {
            filename: stack.enter_context(
                safe_open(
                    str(args.transformer_path / filename),
                    framework="pt",
                    device="cpu",
                )
            )
            for filename in set(weight_map.values())
        }
        time_kwargs = {
            f"{module}_{name}": _load_tensor(
                f"time_embedder.{module}.{name}",
                weight_map=weight_map,
                files=files,
                device=device,
            )
            for module, name in (
                ("proj_in", "weight"),
                ("proj_in", "bias"),
                ("proj_out", "weight"),
                ("proj_out", "bias"),
            )
        }
        adaln_inputs = []
        for plan_index, plan in enumerate(plans):
            plan_length = plan.numel()
            plan_timesteps[plan_index, :plan_length].copy_(plan)
            adaln_inputs.append(
                F.silu(_time_embed(plan.to(device), **time_kwargs)).to(torch.bfloat16)
            )

        for index in range(_NUM_BLOCKS):
            prefix = f"blocks.{index}.adaln_proj.linear"
            weight = _load_tensor(
                f"{prefix}.weight",
                weight_map=weight_map,
                files=files,
                device=device,
            )
            bias = _load_tensor(
                f"{prefix}.bias",
                weight_map=weight_map,
                files=files,
                device=device,
            )
            for plan_index, adaln_input in enumerate(adaln_inputs):
                plan_length = adaln_input.shape[0]
                block_params[plan_index, :plan_length, index].copy_(
                    F.linear(adaln_input, weight, bias).cpu()
                )

        prefix = "final_layer.adaln_proj.linear"
        weight = _load_tensor(
            f"{prefix}.weight",
            weight_map=weight_map,
            files=files,
            device=device,
        )
        bias = _load_tensor(
            f"{prefix}.bias",
            weight_map=weight_map,
            files=files,
            device=device,
        )
        for plan_index, adaln_input in enumerate(adaln_inputs):
            plan_length = adaln_input.shape[0]
            final_params[plan_index, :plan_length].copy_(
                F.linear(adaln_input, weight, bias).cpu()
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "plan_timesteps": plan_timesteps,
            "plan_lengths": plan_lengths,
            "block_params": block_params,
            "final_params": final_params,
        },
        str(args.output),
        metadata={
            "format_version": "2",
            "model_variant": args.model_variant,
        },
    )


if __name__ == "__main__":
    main()
