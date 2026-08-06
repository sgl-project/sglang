#!/usr/bin/env python3
"""Run current minWM main/V3 as the numerical baseline for manifest cases."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from common import (
    action_label_sequence,
    action_weights,
    build_minwm_message,
    load_cases,
    materialize_first_frame,
    prompt_switch_boundary,
    save_video,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--minwm-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--config",
        help="Defaults to Wan21/configs/eval/wan22_5b_varlen_dmd.yaml.",
    )
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Discard this many complete runs of each case before measuring it.",
    )
    parser.add_argument(
        "--local-attn-size",
        type=int,
        help="Override MinWM V3 generator_config.local_attn_size.",
    )
    parser.add_argument(
        "--sink-size",
        type=int,
        help="Override MinWM V3 generator_config.sink_size.",
    )
    parser.add_argument(
        "--fp8-calibration-output",
        help=(
            "Write per-module input activation maxima for the 300 MinWM block "
            "linears. This synchronizes CUDA and is for calibration, not timing."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.local_attn_size is not None and (
        args.local_attn_size == 0 or args.local_attn_size < -1
    ):
        raise ValueError("--local-attn-size must be -1 or positive")
    if args.sink_size is not None and args.sink_size < 0:
        raise ValueError("--sink-size must be non-negative")
    if (
        args.local_attn_size is not None
        and args.local_attn_size != -1
        and args.sink_size is not None
        and args.sink_size >= args.local_attn_size
    ):
        raise ValueError("--sink-size must be smaller than finite --local-attn-size")
    deterministic = os.environ.get("MINWM_PARITY_DETERMINISTIC", "1").strip().lower()
    deterministic = deterministic not in {"0", "false", "no", "off"}
    os.environ["MINWM_DETERMINISTIC_ATTENTION"] = "true" if deterministic else "false"
    if deterministic:
        # Must be configured before the first CUDA BLAS handle is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    minwm_root = Path(args.minwm_root).resolve()
    wan_root = minwm_root / "Wan21"
    config_path = (
        Path(args.config).resolve()
        if args.config
        else (wan_root / "configs/eval/wan22_5b_varlen_dmd.yaml")
    )
    pretrained_dir = Path(args.pretrained_dir).resolve()
    for required in (
        wan_root,
        config_path,
        Path(args.checkpoint),
        pretrained_dir / "transformer",
        pretrained_dir / "text_encoder",
        pretrained_dir / "tokenizer",
        pretrained_dir / "vae",
    ):
        if not required.exists():
            raise FileNotFoundError(required)

    os.environ["PRETRAINED_DIR"] = str(pretrained_dir)
    sys.path.insert(0, str(wan_root))
    sys.path.insert(0, str(minwm_root))

    import torch
    from configs.configuration import PretrainedConfig
    from dataloader.processors.wan_packed import WanPackedProcessor
    from einops import rearrange
    from omegaconf import OmegaConf
    from pipeline import PipelineBase
    from wan_utils.misc import set_seed

    manifest = load_cases(args.cases)
    contract = manifest["contract"]
    selected = set(args.selected_cases or [])
    cases = [
        case for case in manifest["cases"] if not selected or case["id"] in selected
    ]
    if selected - {case["id"] for case in cases}:
        raise ValueError(
            f"unknown case ids: {sorted(selected - {case['id'] for case in cases})}"
        )

    results = Path(args.results).resolve()
    inputs = results / "inputs"
    config = PretrainedConfig.from_pretrained(str(config_path))
    runtime_overrides = {
        "height": int(contract["height"]),
        "width": int(contract["width"]),
        "guidance_scale": 0.0,
        "num_frames": 1 + int(contract["generated_latent_frames"]),
        "dataloader": {
            "processor_kwargs": {
                "action_output_format": contract.get("action_output_format", "label_81")
            }
        },
    }
    generator_overrides = {}
    if args.local_attn_size is not None:
        generator_overrides["local_attn_size"] = args.local_attn_size
    if args.sink_size is not None:
        generator_overrides["sink_size"] = args.sink_size
    if generator_overrides:
        runtime_overrides["generator_config"] = generator_overrides
    config = OmegaConf.merge(config, OmegaConf.create(runtime_overrides))
    OmegaConf.resolve(config)
    effective_local_attn_size = int(config.generator_config.local_attn_size)
    effective_sink_size = int(config.generator_config.sink_size)
    configured_window_size = config.generator_config.get("window_size")
    if configured_window_size is not None:
        configured_window_size = int(configured_window_size)
    if (
        effective_local_attn_size != -1
        and effective_sink_size >= effective_local_attn_size
    ):
        raise ValueError(
            "effective sink_size must be smaller than finite local_attn_size"
        )

    pipeline = PipelineBase.from_pretrained(
        config,
        str(Path(args.checkpoint).resolve()),
        torch.device("cuda"),
        low_memory=False,
    )
    processor = WanPackedProcessor(config)

    calibration_amax: dict[str, float] = {}
    calibration_samples: dict[str, int] = {}
    calibration_handles = []
    if args.fp8_calibration_output:
        block_linear_pattern = re.compile(
            r"^blocks\.\d+\." r"(?:self_attn\.[qkvo]|cross_attn\.[qkvo]|ffn\.(?:0|2))$"
        )
        calibration_modules = {
            name: module
            for name, module in pipeline.generator.named_modules()
            if isinstance(module, torch.nn.Linear)
            and block_linear_pattern.fullmatch(name)
        }
        if len(calibration_modules) != 300:
            raise RuntimeError(
                "expected exactly 300 MinWM block linears for FP8 calibration, "
                f"found {len(calibration_modules)}"
            )

        def capture_input_amax(name: str):
            def hook(_module, hook_args):
                if not hook_args or not isinstance(hook_args[0], torch.Tensor):
                    raise TypeError(f"{name}: expected tensor as first linear input")
                value = float(hook_args[0].detach().abs().amax().float().item())
                if not math.isfinite(value):
                    raise ValueError(f"{name}: non-finite activation maximum {value}")
                calibration_amax[name] = max(calibration_amax.get(name, 0.0), value)
                calibration_samples[name] = calibration_samples.get(name, 0) + 1

            return hook

        calibration_handles = [
            module.register_forward_pre_hook(capture_input_amax(name))
            for name, module in calibration_modules.items()
        ]

    dump_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
    if dump_root and args.warmup_runs:
        raise ValueError(
            "MINWM_PARITY_DUMP_DIR cannot be combined with --warmup-runs because "
            "the hooks would capture the discarded run"
        )
    if dump_root:
        dump_dir = Path(dump_root) / "baseline"
        dump_dir.mkdir(parents=True, exist_ok=True)
        original_forward = pipeline.generator.forward
        forward_index = 0
        patch_index = 0
        block_index = 0

        def dump_patch_output(_module, _args, output):
            nonlocal patch_index
            if patch_index < 2:
                torch.save(
                    output.detach().cpu(),
                    dump_dir / f"patch_output_{patch_index:03d}.pt",
                )
            patch_index += 1

        def dump_first_block_output(_module, hook_args, output):
            nonlocal block_index
            if block_index < 2:
                block_output = output[0] if isinstance(output, tuple) else output
                torch.save(
                    hook_args[0].detach().cpu(),
                    dump_dir / f"block0_input_{block_index:03d}.pt",
                )
                torch.save(
                    block_output.detach().cpu(),
                    dump_dir / f"block0_output_{block_index:03d}.pt",
                )
            block_index += 1

        pipeline.generator.patch_embedding.register_forward_hook(dump_patch_output)
        pipeline.generator.blocks[0].register_forward_hook(dump_first_block_output)

        detail_counters: dict[str, int] = {}

        def dump_detail(name: str):
            def hook(_module, hook_args, output):
                index = detail_counters.get(name, 0)
                if index < 2:
                    value = output[0] if isinstance(output, tuple) else output
                    torch.save(
                        value.detach().cpu(),
                        dump_dir / f"{name}_{index:03d}.pt",
                    )
                    if name in {"self_q", "cross_q", "self_residual_norm"}:
                        torch.save(
                            hook_args[0].detach().cpu(),
                            dump_dir / f"{name}_input_{index:03d}.pt",
                        )
                detail_counters[name] = index + 1

            return hook

        block0 = pipeline.generator.blocks[0]
        detail_modules = {
            "time_embed": pipeline.generator.time_embedding,
            "time_projection": pipeline.generator.time_projection,
            "text_embed": pipeline.generator.text_embedding,
            "self_q": block0.self_attn.q,
            "self_k": block0.self_attn.k,
            "self_v": block0.self_attn.v,
            "self_out": block0.self_attn.o,
            "self_residual_norm": block0.norm3,
            "cross_q": block0.cross_attn.q,
            "cross_k": block0.cross_attn.k,
            "cross_v": block0.cross_attn.v,
            "cross_out": block0.cross_attn.o,
            "ffn": block0.ffn,
        }
        for detail_name, module in detail_modules.items():
            module.register_forward_hook(dump_detail(detail_name))

        def parity_forward(*forward_args, **forward_kwargs):
            nonlocal forward_index
            output = original_forward(*forward_args, **forward_kwargs)
            x = forward_kwargs.get("x")
            if x is None and forward_args:
                x = forward_args[0]
            record = {
                "x": x.detach().cpu(),
                "t": forward_kwargs["t"].detach().cpu(),
                "action": (
                    forward_kwargs["action"].detach().cpu()
                    if forward_kwargs.get("action") is not None
                    else None
                ),
                "context": (
                    forward_kwargs["context"].detach().cpu()
                    if forward_index == 0
                    else None
                ),
                "context_lens": (
                    forward_kwargs["context_lens"].detach().cpu()
                    if forward_index == 0
                    else None
                ),
                "output": output.detach().cpu(),
            }
            for name in (
                "seq_lens",
                "block_idx",
                "position_ids",
                "clean_x",
                "aug_t",
                "action_seq_lens",
                "action_token_nums",
                "action_mask",
                "cross_seq_lens",
                "attention_tag",
            ):
                if name not in forward_kwargs:
                    continue
                value = forward_kwargs.get(name)
                record[name] = (
                    value.detach().cpu() if isinstance(value, torch.Tensor) else value
                )
            torch.save(record, dump_dir / f"forward_{forward_index:03d}.pt")
            forward_index += 1
            return output

        pipeline.generator.forward = parity_forward

    def encode(frames: torch.Tensor) -> torch.Tensor:
        normalized = (
            frames.to(device="cuda", dtype=torch.bfloat16).div_(127.5).sub_(1.0)
        )
        if dump_root:
            torch.save(normalized.detach().cpu(), dump_dir / "vae_input.pt")
        return pipeline.vae.encode_to_latent(normalized).cpu()

    processor.encode = encode
    run_records = []
    for case in cases:
        case_dir = results / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        first_frame = materialize_first_frame(case, inputs)
        message = build_minwm_message(case, contract, first_frame)
        write_json(case_dir / "input.json", message)

        warmup_elapsed_s = []
        for run_index in range(args.warmup_runs + 1):
            pipeline.vae.model.clear_cache()
            batch = processor.process_inference_messages(message)
            switch_boundary = prompt_switch_boundary(case, contract)
            if switch_boundary is not None:
                expected_switch_latent = 1 + int(
                    case["prompt_switch"]["target_chunk"]
                ) * int(contract["latent_frames_per_chunk"])
                expected_prompt_lengths = [
                    expected_switch_latent,
                    1
                    + int(contract["generated_latent_frames"])
                    - expected_switch_latent,
                ]
                if batch["prompts"] != [
                    case["prompt"],
                    case["prompt_switch"]["prompt"],
                ]:
                    raise AssertionError(
                        f"{case['id']}: processor prompt segments do not match manifest"
                    )
                if batch["prompt_seqlens"].tolist() != expected_prompt_lengths:
                    raise AssertionError(
                        f"{case['id']}: processor prompt lengths "
                        f"{batch['prompt_seqlens'].tolist()} != "
                        f"{expected_prompt_lengths}"
                    )
            if dump_root:
                torch.save(batch["clean_x"], dump_dir / "clean_x.pt")
            if contract.get("action_output_format") == "primitive_float":
                temporal_factor = int(config.vae_config.scale_factor_temporal)
                expected_actions = np.zeros(
                    (
                        1 + int(contract["generated_latent_frames"]),
                        temporal_factor,
                        8,
                    ),
                    dtype=np.float32,
                )
                expected_actions[1:] = np.asarray(
                    action_weights(case), dtype=np.float32
                )
                actual_actions = batch["action"][0].numpy()
                if not np.array_equal(actual_actions, expected_actions):
                    raise AssertionError(
                        f"{case['id']}: processor action weight windows do not "
                        "match manifest"
                    )
            else:
                expected_actions = [0] + action_label_sequence(case, contract)
                actual_actions = batch["action"][0].tolist()
                if actual_actions != expected_actions:
                    raise AssertionError(
                        f"{case['id']}: processor action labels {actual_actions} "
                        f"!= {expected_actions}"
                    )

            # Reset immediately before V3's one-shot BFCHW noise draw. Reference
            # encoding is deterministic posterior-mode and intentionally outside it.
            set_seed(int(contract["seed"]), deterministic=deterministic)
            torch.cuda.synchronize()
            started = time.perf_counter()
            video, latents = pipeline.run_processor_batch(batch, return_latents=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started

            frames = (
                (255.0 * rearrange(video, "b t c h w -> b t h w c")[0].cpu())
                .to(torch.uint8)
                .numpy()
            )
            if run_index < args.warmup_runs:
                warmup_elapsed_s.append(elapsed)
                print(
                    json.dumps(
                        {
                            "id": case["id"],
                            "warmup": run_index + 1,
                            "warmup_runs": args.warmup_runs,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        expected_pixel_frames = int(contract["reference_pixel_frames"]) + int(
            contract["generated_pixel_frames"]
        )
        if frames.shape != (
            expected_pixel_frames,
            int(contract["height"]),
            int(contract["width"]),
            3,
        ):
            raise AssertionError(f"{case['id']}: unexpected video shape {frames.shape}")

        np.save(case_dir / "baseline.npy", frames, allow_pickle=False)
        save_video(case_dir / "baseline.mp4", frames, int(contract["fps"]))
        torch.save(latents.detach().cpu(), case_dir / "baseline_latents.pt")
        record = {
            "id": case["id"],
            "elapsed_s": elapsed,
            "frames": int(frames.shape[0]),
            "warmup_runs": args.warmup_runs,
            "warmup_elapsed_s": warmup_elapsed_s,
            "video_sha256": sha256_file(case_dir / "baseline.mp4"),
            "frames_sha256": sha256_file(case_dir / "baseline.npy"),
            "latents_sha256": sha256_file(case_dir / "baseline_latents.pt"),
            "prompt_switch": (
                {
                    **case["prompt_switch"],
                    "pixel_frame_boundary": switch_boundary,
                    "latent_frame_boundary": (
                        1
                        + int(case["prompt_switch"]["target_chunk"])
                        * int(contract["latent_frames_per_chunk"])
                    ),
                }
                if switch_boundary is not None
                else None
            ),
        }
        write_json(case_dir / "baseline.json", record)
        run_records.append(record)
        pipeline.vae.model.clear_cache()
        print(json.dumps(record, sort_keys=True), flush=True)

    write_json(
        results / "baseline_run.json",
        {
            "engine": "minwm-main-v3",
            "minwm_git_sha": subprocess.check_output(
                ["git", "-C", str(minwm_root), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_size": Path(args.checkpoint).stat().st_size,
            "deterministic": deterministic,
            "deterministic_attention": bool(config.deterministic_attention),
            "local_attn_size": effective_local_attn_size,
            "sink_size": effective_sink_size,
            "window_size": configured_window_size,
            "warmup_runs": args.warmup_runs,
            "cases": run_records,
        },
    )
    if args.fp8_calibration_output:
        for handle in calibration_handles:
            handle.remove()
        missing = sorted(set(calibration_modules) - set(calibration_amax))
        if missing:
            raise RuntimeError(
                "FP8 calibration did not exercise all block linears: "
                + ", ".join(missing[:10])
            )
        write_json(
            Path(args.fp8_calibration_output).resolve(),
            {
                "format": "minwm-static-fp8-calibration-v1",
                "module_count": len(calibration_modules),
                "modules": {
                    name: {
                        "input_amax": calibration_amax[name],
                        "samples": calibration_samples[name],
                    }
                    for name in sorted(calibration_modules)
                },
            },
        )


if __name__ == "__main__":
    main()
