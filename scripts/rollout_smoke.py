#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for rollout log-prob trajectory on native denoising pipeline.

Usage:
  PYTHONPATH=python python scripts/rollout_smoke.py \
      --model-path <native-model-path>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model


def _tensor_to_pil_image(sample: torch.Tensor) -> Image.Image:
    t = sample.detach().cpu().float()
    if t.ndim == 4:
        t = t[0]

    if t.ndim != 3:
        raise RuntimeError(f"Expected 3D image tensor, got shape={tuple(t.shape)}")

    if t.shape[0] in (1, 3, 4):  # CHW
        t = t.clamp(0, 1)
        arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    elif t.shape[-1] in (1, 3, 4):  # HWC
        t = t.clamp(0, 1)
        arr = (t.numpy() * 255.0).round().astype(np.uint8)
    else:
        raise RuntimeError(f"Cannot infer channel dimension from shape={tuple(t.shape)}")

    if arr.shape[-1] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def _build_request(
    generator: DiffGenerator,
    args: argparse.Namespace,
    *,
    rollout: bool,
    rollout_sde_type: str | None = None,
):
    sampling_kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "rollout": rollout,
        "save_output": False,
        "return_frames": False,
    }
    if rollout and rollout_sde_type is not None:
        sampling_kwargs["rollout_sde_type"] = rollout_sde_type
    if args.negative_prompt is not None:
        sampling_kwargs["negative_prompt"] = args.negative_prompt

    sampling_params = SamplingParams.from_user_sampling_params_args(
        generator.server_args.model_path,
        server_args=generator.server_args,
        **sampling_kwargs,
    )
    req = prepare_request(server_args=generator.server_args, sampling_params=sampling_params)
    return req


def _run_once(
    generator: DiffGenerator,
    args: argparse.Namespace,
    run_name: str,
    output_dir: Path,
    *,
    rollout: bool,
    rollout_sde_type: str | None = None,
) -> Path:
    req = _build_request(
        generator=generator,
        args=args,
        rollout=rollout,
        rollout_sde_type=rollout_sde_type,
    )
    output_batch = generator._send_to_scheduler_and_wait_for_response([req])

    if output_batch.error:
        raise RuntimeError(f"{run_name} run failed: {output_batch.error}")
    if output_batch.output is None:
        raise RuntimeError(f"{run_name} run returned empty output.")

    sample = output_batch.output[0]
    image = _tensor_to_pil_image(sample)
    image_path = output_dir / f"rollout_{run_name}.png"
    image.save(image_path)
    print(f"[{run_name}] image: {image_path}")
    return image_path

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Native rollout smoke test (no rollout / sde / cps)."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pipeline-class-name", type=str, default=None)
    parser.add_argument(
        "--prompt", type=str, default="a photo of a cat wearing sunglasses"
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=None,
        help="Optional negative prompt. Use a non-empty value if you want CFG.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--backend",
        type=str,
        default="sglang",
        help="Backend to use. Defaults to 'sglang' for native-only smoke testing.",
    )
    parser.add_argument("--local-mode", action="store_true", default=True)
    parser.add_argument("--remote-mode", action="store_true", default=False)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rollout_smoke_native",
        help="Directory to save generated images.",
    )
    args = parser.parse_args()
    if args.num_inference_steps < 2:
        print(
            f"[smoke] num_inference_steps={args.num_inference_steps} is too small for stable timestep preparation; overriding to 2.",
            file=sys.stderr,
        )
        args.num_inference_steps = 2

    local_mode = not args.remote_mode if args.remote_mode else args.local_mode
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the original model ID/path for registry matching. If that fails, retry
    # with the resolved local snapshot path for offline compatibility.
    resolved_model_path: str | None = None
    try:
        resolved_model_path = maybe_download_model(args.model_path)
    except Exception:
        resolved_model_path = None

    init_kwargs: dict[str, Any] = {
        "num_gpus": args.num_gpus,
    }
    if args.pipeline_class_name:
        init_kwargs["pipeline_class_name"] = args.pipeline_class_name
    if args.backend:
        init_kwargs["backend"] = args.backend

    model_path_candidates = [args.model_path]
    if resolved_model_path and resolved_model_path != args.model_path:
        model_path_candidates.append(resolved_model_path)

    generator: DiffGenerator | None = None
    last_error: Exception | None = None
    for candidate_path in model_path_candidates:
        try:
            generator = DiffGenerator.from_pretrained(
                local_mode=local_mode,
                model_path=candidate_path,
                **init_kwargs,
            )
            break
        except Exception as e:
            last_error = e
            if (
                candidate_path == args.model_path
                and len(model_path_candidates) > 1
            ):
                print(
                    f"[smoke] failed with model_path='{args.model_path}', retrying with resolved local path.",
                    file=sys.stderr,
                )
            else:
                raise

    if generator is None:
        raise RuntimeError(
            f"Failed to initialize generator for model_path='{args.model_path}': {last_error}"
        )
    try:
        _run_once(
            generator=generator,
            args=args,
            run_name="native_no_rollout",
            output_dir=output_dir,
            rollout=False,
        )
        _run_once(
            generator=generator,
            args=args,
            run_name="native_rollout_sde",
            output_dir=output_dir,
            rollout=True,
            rollout_sde_type="sde",
        )
        _run_once(
            generator=generator,
            args=args,
            run_name="native_rollout_cps",
            output_dir=output_dir,
            rollout=True,
            rollout_sde_type="cps",
        )
    finally:
        generator.shutdown()

    print(f"native rollout smoke test passed (saved 3 images under {output_dir})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"native rollout smoke test failed: {e}", file=sys.stderr)
        raise SystemExit(1)
