#!/usr/bin/env python3
"""Run minWM main/V3 through the same direct T2V path as wan_inference.py."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import (
    load_cases,
    resolve_case_contract,
    save_video,
    sha256_file,
    trajectory_action_labels,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--minwm-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--case", action="append", dest="selected_cases")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deterministic = os.environ.get("MINWM_PARITY_DETERMINISTIC", "1").strip().lower()
    deterministic = deterministic not in {"0", "false", "no", "off"}
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    minwm_root = Path(args.minwm_root).resolve()
    wan_root = minwm_root / "Wan21"
    checkpoint = Path(args.checkpoint).resolve()
    pretrained = Path(args.pretrained_dir).resolve()
    config_path = Path(args.config).resolve()
    for required in (
        wan_root,
        checkpoint,
        config_path,
        pretrained / "transformer",
        pretrained / "text_encoder",
        pretrained / "tokenizer",
        pretrained / "vae",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ["PRETRAINED_DIR"] = str(pretrained)
    sys.path.insert(0, str(wan_root))
    sys.path.insert(0, str(minwm_root))

    import torch
    from einops import rearrange

    from configs.configuration import PretrainedConfig
    from pipeline import PipelineBase
    from wan_utils.misc import set_seed

    manifest = load_cases(args.cases)
    contract = manifest["contract"]
    selected = set(args.selected_cases or [])
    cases = [
        case for case in manifest["cases"] if not selected or case["id"] in selected
    ]
    unknown = selected - {case["id"] for case in manifest["cases"]}
    if unknown:
        raise ValueError(f"unknown case ids: {sorted(unknown)}")

    config = PretrainedConfig.from_pretrained(str(config_path))
    expected_config = {
        "height": int(contract["height"]),
        "width": int(contract["width"]),
        "num_frame_first_block": int(contract["latent_chunk_sizes"][0]),
        "guidance_scale": 0.0,
    }
    actual_config = {
        "height": int(config.height),
        "width": int(config.width),
        "num_frame_first_block": int(config.num_frame_first_block),
        "guidance_scale": float(config.guidance_scale),
    }
    if actual_config != expected_config:
        raise ValueError(
            f"baseline config does not match manifest: {actual_config} != "
            f"{expected_config}"
        )
    if int(config.num_frame_per_block) != 4:
        raise ValueError("step-1600 baseline requires num_frame_per_block=4")
    if list(config.denoising_step_list) != [1000, 750, 500, 250]:
        raise ValueError("unexpected regular DMD schedule")
    if list(config.denoising_step_list_first_block) != [1000, 750, 500, 250]:
        raise ValueError("unexpected first-block DMD schedule")

    pipeline = PipelineBase.from_pretrained(
        config,
        str(checkpoint),
        torch.device("cuda"),
        low_memory=False,
    )
    parity_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
    parity_dir = Path(parity_root) / "baseline" if parity_root else None
    if parity_dir is not None:
        parity_dir.mkdir(parents=True, exist_ok=True)
        original_model_flow = pipeline._model_flow
        forward_index = 0

        def model_flow_with_dump(
            latents,
            conditional_dict,
            timestep,
            action,
            cache,
            self_cache_update,
            condition_switch=None,
        ):
            nonlocal forward_index
            output = original_model_flow(
                latents,
                conditional_dict,
                timestep,
                action,
                cache,
                self_cache_update,
                condition_switch,
            )
            if forward_index < 6:
                torch.save(
                    {
                        "latent_model_input": latents.detach().cpu(),
                        "prompt_embeds": conditional_dict["flat_prompt_embeds"]
                        .detach()
                        .cpu(),
                        "prompt_lens": conditional_dict["prompt_lens"],
                        "timestep": timestep.detach().cpu(),
                        "action": (None if action is None else action.detach().cpu()),
                        "self_cache_update": self_cache_update,
                        "condition_switch": condition_switch,
                        "output": output.detach().cpu(),
                    },
                    parity_dir / f"forward_{forward_index:03d}.pt",
                )
            forward_index += 1
            return output

        pipeline._model_flow = model_flow_with_dump
    results = Path(args.results).resolve()
    run_records = []
    for case in cases:
        case_contract = resolve_case_contract(case, contract)
        case_dir = results / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        action_labels = trajectory_action_labels(
            case["trajectory"],
            expected_frames=int(case_contract["generated_latent_frames"]),
        )
        latent_shape = (
            1,
            int(case_contract["generated_latent_frames"]),
            int(config.vae_config.z_dim),
            int(case_contract["height"]) // int(config.vae_config.scale_factor_spatial),
            int(case_contract["width"]) // int(config.vae_config.scale_factor_spatial),
        )
        expected_video_shape = (
            int(case_contract["generated_pixel_frames"]),
            int(case_contract["height"]),
            int(case_contract["width"]),
            3,
        )

        # The source script uses torchrun with SP=1 and at most one sample per
        # rank in each archived batch. Every case therefore receives rank-local
        # CUDA seed 0's first BFCHW draw, rather than a serial RNG continuation.
        set_seed(int(contract["seed"]), deterministic=deterministic)
        noise = torch.randn(
            latent_shape,
            device="cuda",
            dtype=torch.bfloat16,
        )
        action = torch.tensor(
            action_labels,
            device="cuda",
            dtype=torch.long,
        ).unsqueeze(0)
        if parity_dir is not None:
            torch.save(noise.detach().cpu(), parity_dir / "initial_noise_bfchw.pt")
            torch.save(action.detach().cpu(), parity_dir / "action_labels.pt")

        torch.cuda.synchronize()
        started = time.perf_counter()
        video, latents = pipeline(
            noise=noise,
            text_prompts=[case["prompt"]],
            return_latents=True,
            action=action,
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = (
            (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].cpu())
            .to(torch.uint8)
            .numpy()
        )
        if frames.shape != expected_video_shape:
            raise AssertionError(
                f"{case['id']}: video shape {frames.shape} != {expected_video_shape}"
            )

        np.save(case_dir / "baseline.npy", frames, allow_pickle=False)
        save_video(case_dir / "baseline.mp4", frames, int(contract["fps"]))
        torch.save(latents.detach().cpu(), case_dir / "baseline_latents.pt")
        record = {
            "id": case["id"],
            "trajectory": case["trajectory"],
            "contract": case_contract,
            "elapsed_s": elapsed,
            "frames": int(frames.shape[0]),
            "frames_sha256": sha256_file(case_dir / "baseline.npy"),
            "video_sha256": sha256_file(case_dir / "baseline.mp4"),
            "latents_sha256": sha256_file(case_dir / "baseline_latents.pt"),
        }
        write_json(case_dir / "baseline.json", record)
        run_records.append(record)
        pipeline.vae.model.clear_cache()
        print(json.dumps(record, sort_keys=True), flush=True)

    write_json(
        results / "baseline_run.json",
        {
            "engine": "minwm-main-v3-direct-t2v",
            "minwm_git_sha": subprocess.check_output(
                ["git", "-C", str(minwm_root), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "seed_contract": (
                "reset per case; matches first sample on each SP=1 torchrun rank"
            ),
            "deterministic": deterministic,
            "cases": run_records,
        },
    )


if __name__ == "__main__":
    main()
