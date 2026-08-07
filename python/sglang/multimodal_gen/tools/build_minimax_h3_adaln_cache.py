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
    minimax_h3_time_shift_sigmas,
)

_HIDDEN_SIZE = 5376
_TIME_EMBED_DIM = 2688
_TIMESTEP_INPUT_DIM = 256
_NUM_BLOCKS = 50
_BLOCK_PARAM_WIDTH = 18 * _HIDDEN_SIZE
_FINAL_PARAM_WIDTH = 2 * _HIDDEN_SIZE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MiniMax H3 inference-only AdaLN cache from local weights."
    )
    parser.add_argument("--transformer-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-variant", choices=("fl2va", "ref2va"), required=True)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--audio-flow-shift", type=float, default=3.0)
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
        help="Override the scheduler-derived timestep coverage.",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cache_timesteps(args: argparse.Namespace) -> torch.Tensor:
    if args.timesteps is not None:
        return torch.tensor(args.timesteps, dtype=torch.float32).unique(sorted=True)

    video_sigmas = minimax_h3_time_shift_sigmas(
        num_steps=args.num_inference_steps,
        shift_scale=args.flow_shift,
    )
    audio_sigmas = minimax_h3_time_shift_sigmas(
        num_steps=args.num_inference_steps,
        shift_scale=args.audio_flow_shift,
    )
    candidates = []
    for video_sigma, audio_sigma in zip(video_sigmas[:-1], audio_sigmas[:-1]):
        video_timestep = 1.0 - video_sigma
        audio_timestep = 1.0 - audio_sigma
        candidates.extend(
            (
                video_timestep,
                audio_timestep,
                max(video_timestep, args.imgvid_cond_noise_aug),
                max(audio_timestep, args.audio_cond_noise_aug),
            )
        )
    return torch.tensor(candidates, dtype=torch.float32).unique(sorted=True)


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
    if args.num_inference_steps < 2 and args.timesteps is None:
        raise ValueError("--num-inference-steps must be at least 2")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("MiniMax H3 AdaLN cache must be built on CUDA")

    index_path = args.transformer_path / "model.safetensors.index.json"
    with index_path.open() as f:
        weight_map = json.load(f)["weight_map"]

    timesteps = _cache_timesteps(args)
    if timesteps.numel() == 0:
        raise ValueError("AdaLN cache must cover at least one timestep")
    adaln_inputs = torch.empty(
        (timesteps.numel(), _TIME_EMBED_DIM), dtype=torch.bfloat16
    )
    block_params = torch.empty(
        (timesteps.numel(), _NUM_BLOCKS, _BLOCK_PARAM_WIDTH), dtype=torch.bfloat16
    )
    final_params = torch.empty(
        (timesteps.numel(), _FINAL_PARAM_WIDTH), dtype=torch.bfloat16
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
        adaln_input = F.silu(_time_embed(timesteps.to(device), **time_kwargs)).to(
            torch.bfloat16
        )
        adaln_inputs.copy_(adaln_input.cpu())

        for index in range(_NUM_BLOCKS):
            prefix = f"blocks.{index}.adaln_proj.linear"
            block_params[:, index].copy_(
                F.linear(
                    adaln_input,
                    _load_tensor(
                        f"{prefix}.weight",
                        weight_map=weight_map,
                        files=files,
                        device=device,
                    ),
                    _load_tensor(
                        f"{prefix}.bias",
                        weight_map=weight_map,
                        files=files,
                        device=device,
                    ),
                ).cpu()
            )

        prefix = "final_layer.adaln_proj.linear"
        final_params.copy_(
            F.linear(
                adaln_input,
                _load_tensor(
                    f"{prefix}.weight",
                    weight_map=weight_map,
                    files=files,
                    device=device,
                ),
                _load_tensor(
                    f"{prefix}.bias",
                    weight_map=weight_map,
                    files=files,
                    device=device,
                ),
            ).cpu()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "adaln_inputs": adaln_inputs,
            "block_params": block_params,
            "final_params": final_params,
        },
        str(args.output),
        metadata={
            "format_version": "1",
            "model_variant": args.model_variant,
        },
    )


if __name__ == "__main__":
    main()
