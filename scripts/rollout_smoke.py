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


def _build_request(generator: DiffGenerator, args: argparse.Namespace):
    sampling_kwargs: dict[str, Any] = {
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "guidance_scale": args.guidance_scale,
        "rollout": True,
        "rollout_sde_type": args.rollout_sde_type,
        "save_output": False,
        "return_frames": False,
    }
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
) -> dict[str, Any]:
    req = _build_request(generator=generator, args=args)
    output_batch = generator._send_to_scheduler_and_wait_for_response([req])

    if output_batch.error:
        raise RuntimeError(f"{run_name} run failed: {output_batch.error}")
    if output_batch.output is None:
        raise RuntimeError(f"{run_name} run returned empty output.")

    sample = output_batch.output[0]
    image = _tensor_to_pil_image(sample)
    image_path = output_dir / f"rollout_{run_name}.png"
    image.save(image_path)

    log_probs = output_batch.trajectory_log_probs
    if log_probs is None:
        raise RuntimeError(f"{run_name} run returned no trajectory_log_probs.")

    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.as_tensor(log_probs)
    log_probs = log_probs.detach().cpu()

    if log_probs.numel() == 0:
        raise RuntimeError(f"{run_name} trajectory_log_probs is empty.")
    if log_probs.ndim < 2:
        raise RuntimeError(
            f"{run_name} trajectory_log_probs should be [B, T], got {tuple(log_probs.shape)}"
        )
    if int(log_probs.shape[-1]) != args.num_inference_steps:
        raise RuntimeError(
            f"{run_name} steps mismatch: expected {args.num_inference_steps}, "
            f"got {int(log_probs.shape[-1])} (shape={tuple(log_probs.shape)})"
        )
    if not torch.isfinite(log_probs).all():
        raise RuntimeError(f"{run_name} trajectory_log_probs contains NaN/Inf.")

    logprob_path = output_dir / f"trajectory_log_probs_{run_name}.pt"
    torch.save(log_probs, logprob_path)

    latents = output_batch.trajectory_latents
    if latents is None:
        raise RuntimeError(f"{run_name} run returned no trajectory_latents.")
    if not isinstance(latents, torch.Tensor):
        latents = torch.as_tensor(latents)
    latents = latents.detach().cpu()

    if latents.numel() == 0:
        raise RuntimeError(f"{run_name} trajectory_latents is empty.")
    if latents.ndim < 3:
        raise RuntimeError(
            f"{run_name} trajectory_latents should be [B, T, ...], got {tuple(latents.shape)}"
        )
    if int(latents.shape[1]) != args.num_inference_steps:
        raise RuntimeError(
            f"{run_name} latent steps mismatch: expected {args.num_inference_steps}, "
            f"got {int(latents.shape[1])} (shape={tuple(latents.shape)})"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError(f"{run_name} trajectory_latents contains NaN/Inf.")

    latents_path = output_dir / f"trajectory_latents_{run_name}.pt"
    torch.save(latents, latents_path)

    print(f"[{run_name}] image: {image_path}")
    print(f"[{run_name}] log_probs: {logprob_path}")
    print(f"[{run_name}] latents: {latents_path}")
    print(f"[{run_name}] log_probs.shape={tuple(log_probs.shape)}")
    print(f"[{run_name}] latents.shape={tuple(latents.shape)}")
    print(f"[{run_name}] log_probs.mean={log_probs.float().mean().item():.6f}")

    return {
        "image_path": image_path,
        "log_probs_path": logprob_path,
        "log_probs": log_probs,
        "latents_path": latents_path,
        "latents": latents,
    }


def _assert_exact_match(
    *,
    metric_name: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    if first.shape != second.shape:
        raise RuntimeError(
            f"[determinism] {metric_name} shape mismatch: "
            f"{tuple(first.shape)} vs {tuple(second.shape)}"
        )
    if first.dtype != second.dtype:
        raise RuntimeError(
            f"[determinism] {metric_name} dtype mismatch: "
            f"{first.dtype} vs {second.dtype}"
        )
    if torch.equal(first, second):
        return

    max_abs_diff = (first.float() - second.float()).abs().max().item()
    raise RuntimeError(
        f"[determinism] {metric_name} mismatch with same seed. "
        f"max_abs_diff={max_abs_diff:.6e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Native rollout log_prob smoke test")
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
        "--rollout-sde-type",
        type=str,
        default="sde",
        choices=["sde", "cps"],
        help="Rollout step objective for native denoising.",
    )
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
        help="Directory to save generated images and rollout tensors.",
    )
    parser.add_argument(
        "--test-determinism",
        action="store_true",
        default=False,
        help="Run twice with the same seed and require identical log_probs/latents.",
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
        first = _run_once(
            generator=generator,
            args=args,
            run_name="native",
            output_dir=output_dir,
        )

        if args.test_determinism:
            print("[determinism] running repeat pass with identical seed...")
            second = _run_once(
                generator=generator,
                args=args,
                run_name="native_repeat",
                output_dir=output_dir,
            )
            _assert_exact_match(
                metric_name="trajectory_log_probs",
                first=first["log_probs"],
                second=second["log_probs"],
            )
            _assert_exact_match(
                metric_name="trajectory_latents",
                first=first["latents"],
                second=second["latents"],
            )
            print("[determinism] passed")
    finally:
        generator.shutdown()

    print(f"native rollout smoke test passed (saved under {output_dir})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"native rollout smoke test failed: {e}", file=sys.stderr)
        raise SystemExit(1)
