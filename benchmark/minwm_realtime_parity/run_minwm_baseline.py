#!/usr/bin/env python3
"""Run current minWM main/V3 as the numerical baseline for ten fixed cases."""

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
    action_weights,
    build_minwm_message,
    load_cases,
    materialize_first_frame,
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
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
    from einops import rearrange
    from omegaconf import OmegaConf

    from configs.configuration import PretrainedConfig
    from dataloader.processors.wan_packed import WanPackedProcessor
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
    config = OmegaConf.merge(
        config,
        OmegaConf.create(
            {
                "height": int(contract["height"]),
                "width": int(contract["width"]),
                "guidance_scale": 0.0,
                "num_frames": 1 + int(contract["generated_latent_frames"]),
                "dataloader": {
                    "processor_kwargs": {
                        "action_output_format": contract.get(
                            "action_output_format", "label_81"
                        )
                    }
                },
            }
        ),
    )
    OmegaConf.resolve(config)

    pipeline = PipelineBase.from_pretrained(
        config,
        str(Path(args.checkpoint).resolve()),
        torch.device("cuda"),
        low_memory=False,
    )
    processor = WanPackedProcessor(config)

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
                    if name == "self_q":
                        torch.save(
                            hook_args[0].detach().cpu(),
                            dump_dir / f"self_q_input_{index:03d}.pt",
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
                expected_actions = [0] + [int(case["action_label"])] * int(
                    contract["generated_latent_frames"]
                )
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
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_size": Path(args.checkpoint).stat().st_size,
            "deterministic": deterministic,
            "deterministic_attention": bool(config.deterministic_attention),
            "warmup_runs": args.warmup_runs,
            "cases": run_records,
        },
    )


if __name__ == "__main__":
    main()
