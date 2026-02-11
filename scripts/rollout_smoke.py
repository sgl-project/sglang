#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke test for rollout log-prob trajectory (SD3 diffusers backend).

It can run one mode (`sde` or `cps`) or both and save generated images/log-probs
to the same output directory for side-by-side comparison.

Usage:
  PYTHONPATH=python python scripts/rollout_smoke.py \
      --model-path stabilityai/stable-diffusion-3-medium-diffusers \
      --pipeline-class-name DiffusersPipeline \
      --sde-type both
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
    sde_type: str,
    noise_level: float | None = None,
):
    sampling_params = SamplingParams.from_user_sampling_params_args(
        generator.server_args.model_path,
        server_args=generator.server_args,
        prompt=args.prompt,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        rollout=True,
        save_output=False,
        return_frames=False,
    )
    req = prepare_request(server_args=generator.server_args, sampling_params=sampling_params)
    rollout_cfg: dict[str, Any] = {"enabled": True, "sde_type": sde_type}
    if noise_level is not None:
        rollout_cfg["noise_level"] = float(noise_level)
    req.extra["rollout"] = rollout_cfg
    return req


def _run_single_mode(
    generator: DiffGenerator,
    args: argparse.Namespace,
    mode_name: str,
    sde_type: str,
    output_dir: Path,
    noise_level: float | None = None,
) -> dict[str, Any]:
    req = _build_request(
        generator, args, sde_type=sde_type, noise_level=noise_level
    )
    output_batch = generator._send_to_scheduler_and_wait_for_response([req])
    if output_batch.error:
        raise RuntimeError(f"{mode_name} run failed: {output_batch.error}")
    if output_batch.output is None:
        raise RuntimeError(f"{mode_name} run returned empty output.")

    sample = output_batch.output[0]
    image = _tensor_to_pil_image(sample)
    image_path = output_dir / f"rollout_{mode_name}.png"
    image.save(image_path)

    log_probs = output_batch.trajectory_log_probs
    if log_probs is None:
        raise RuntimeError(f"{mode_name} run returned no trajectory_log_probs.")

    if not isinstance(log_probs, torch.Tensor):
        log_probs = torch.as_tensor(log_probs)
    log_probs = log_probs.detach().cpu()

    if log_probs.numel() == 0:
        raise RuntimeError(f"{mode_name} trajectory_log_probs is empty.")
    if log_probs.ndim < 2:
        raise RuntimeError(
            f"{mode_name} trajectory_log_probs should be [B, T], got {tuple(log_probs.shape)}"
        )
    if int(log_probs.shape[-1]) != args.num_inference_steps:
        raise RuntimeError(
            f"{mode_name} steps mismatch: expected {args.num_inference_steps}, "
            f"got {int(log_probs.shape[-1])} (shape={tuple(log_probs.shape)})"
        )
    if not torch.isfinite(log_probs).all():
        raise RuntimeError(f"{mode_name} trajectory_log_probs contains NaN/Inf.")

    logprob_path = output_dir / f"trajectory_log_probs_{mode_name}.pt"
    torch.save(log_probs, logprob_path)

    latents = output_batch.trajectory_latents
    if latents is None:
        raise RuntimeError(f"{mode_name} run returned no trajectory_latents.")
    if not isinstance(latents, torch.Tensor):
        latents = torch.as_tensor(latents)
    latents = latents.detach().cpu()

    if latents.numel() == 0:
        raise RuntimeError(f"{mode_name} trajectory_latents is empty.")
    if latents.ndim < 3:
        raise RuntimeError(
            f"{mode_name} trajectory_latents should be [B, T, ...], got {tuple(latents.shape)}"
        )
    if int(latents.shape[1]) != args.num_inference_steps:
        raise RuntimeError(
            f"{mode_name} latent steps mismatch: expected {args.num_inference_steps}, "
            f"got {int(latents.shape[1])} (shape={tuple(latents.shape)})"
        )
    if not torch.isfinite(latents).all():
        raise RuntimeError(f"{mode_name} trajectory_latents contains NaN/Inf.")

    latents_path = output_dir / f"trajectory_latents_{mode_name}.pt"
    torch.save(latents, latents_path)

    noise_str = f", noise_level={noise_level}" if noise_level is not None else ""
    print(f"[{mode_name}] image: {image_path}")
    print(f"[{mode_name}] log_probs: {logprob_path}")
    print(f"[{mode_name}] latents: {latents_path}")
    print(f"[{mode_name}] log_probs.shape={tuple(log_probs.shape)}{noise_str}")
    print(f"[{mode_name}] latents.shape={tuple(latents.shape)}{noise_str}")
    print(f"[{mode_name}] log_probs.mean={log_probs.float().mean().item():.6f}")
    return {
        "image_path": image_path,
        "log_probs_path": logprob_path,
        "log_probs": log_probs,
        "latents_path": latents_path,
        "latents": latents,
        "sde_type": sde_type,
        "noise_level": noise_level,
    }


def _assert_exact_match(
    *,
    metric_name: str,
    mode_name: str,
    first: torch.Tensor,
    second: torch.Tensor,
) -> None:
    if first.shape != second.shape:
        raise RuntimeError(
            f"[determinism][{mode_name}] {metric_name} shape mismatch: "
            f"{tuple(first.shape)} vs {tuple(second.shape)}"
        )
    if first.dtype != second.dtype:
        raise RuntimeError(
            f"[determinism][{mode_name}] {metric_name} dtype mismatch: "
            f"{first.dtype} vs {second.dtype}"
        )
    if torch.equal(first, second):
        return

    max_abs_diff = (first.float() - second.float()).abs().max().item()
    raise RuntimeError(
        f"[determinism][{mode_name}] {metric_name} mismatch with same seed. "
        f"max_abs_diff={max_abs_diff:.6e}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollout log_prob smoke test")
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--pipeline-class-name", type=str, default=None)
    parser.add_argument(
        "--prompt", type=str, default="a photo of a cat wearing sunglasses"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, default=1)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--local-mode", action="store_true", default=True)
    parser.add_argument("--remote-mode", action="store_true", default=False)
    parser.add_argument(
        "--sde-type",
        type=str,
        default="both",
        choices=["sde", "cps", "both"],
        help="Which rollout path to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/rollout_smoke",
        help="Directory to save generated images and log_prob tensors.",
    )
    parser.add_argument(
        "--test-sde-zero-variance",
        action="store_true",
        default=False,
        help="Run an extra rollout test with sde_type=sde and noise_level=0.",
    )
    parser.add_argument(
        "--test-determinism",
        action="store_true",
        default=False,
        help="Run each selected mode twice with the same seed and require identical log_probs/latents.",
    )
    args = parser.parse_args()

    local_mode = not args.remote_mode if args.remote_mode else args.local_mode
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs: dict[str, Any] = {
        "model_path": args.model_path,
        "num_gpus": args.num_gpus,
    }
    if args.pipeline_class_name:
        init_kwargs["pipeline_class_name"] = args.pipeline_class_name
    if args.backend:
        init_kwargs["backend"] = args.backend

    run_specs: list[tuple[str, str, float | None]] = []
    if args.sde_type == "both":
        run_specs.extend([("sde", "sde", None), ("cps", "cps", None)])
    else:
        run_specs.append((args.sde_type, args.sde_type, None))

    if args.test_sde_zero_variance:
        run_specs.append(("sde_var0", "sde", 0.0))

    results: dict[str, dict[str, Any]] = {}
    repeat_results: dict[str, dict[str, Any]] = {}

    generator = DiffGenerator.from_pretrained(local_mode=local_mode, **init_kwargs)
    try:
        for mode_name, sde_type, noise_level in run_specs:
            results[mode_name] = _run_single_mode(
                generator=generator,
                args=args,
                mode_name=mode_name,
                sde_type=sde_type,
                output_dir=output_dir,
                noise_level=noise_level,
            )

        if args.test_determinism:
            print("[determinism] running repeat pass with identical seed...")
            for mode_name, sde_type, noise_level in run_specs:
                repeat_results[mode_name] = _run_single_mode(
                    generator=generator,
                    args=args,
                    mode_name=f"{mode_name}_repeat",
                    sde_type=sde_type,
                    output_dir=output_dir,
                    noise_level=noise_level,
                )
    finally:
        generator.shutdown()

    if "sde" in results and "cps" in results:
        sde_lp = results["sde"]["log_probs"]
        cps_lp = results["cps"]["log_probs"]
        if sde_lp.shape == cps_lp.shape:
            mean_abs_diff = (sde_lp - cps_lp).abs().float().mean().item()
            print(f"[compare] mean_abs_diff(log_probs)={mean_abs_diff:.6f}")
        else:
            print(
                f"[compare] shape mismatch: sde={tuple(sde_lp.shape)}, cps={tuple(cps_lp.shape)}"
            )

    if args.test_determinism:
        for mode_name in results:
            _assert_exact_match(
                metric_name="trajectory_log_probs",
                mode_name=mode_name,
                first=results[mode_name]["log_probs"],
                second=repeat_results[mode_name]["log_probs"],
            )
            _assert_exact_match(
                metric_name="trajectory_latents",
                mode_name=mode_name,
                first=results[mode_name]["latents"],
                second=repeat_results[mode_name]["latents"],
            )
            print(f"[determinism][{mode_name}] passed")

    print(f"rollout smoke test passed (saved under {output_dir})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"rollout smoke test failed: {e}", file=sys.stderr)
        raise SystemExit(1)
