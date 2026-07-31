#!/usr/bin/env python3
"""Replay Tianpeng's gap12 sample through native minWM DirectorSession."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from common import save_video, sha256_file, write_json
from tianpeng_alignment import (
    EXPECTED_MINWM_COMMIT,
    EXPECTED_SAMPLE_ID,
    EXPECTED_VIDEO_SHA256,
    _fetch_bytes,
    _run_ffmpeg_metric,
    load_contract,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-url", required=True)
    parser.add_argument("--minwm-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Stop after this many DirectorSession blocks for parity debugging.",
    )
    return parser.parse_args()


def _latent_actions(pixel_actions: list[list[int]], temporal_factor: int) -> list:
    if len(pixel_actions) % temporal_factor:
        raise ValueError("pixel action timeline is not VAE-factor aligned")
    latent = [[0] * 8]
    for start in range(0, len(pixel_actions), temporal_factor):
        rows = pixel_actions[start : start + temporal_factor]
        if any(row != rows[0] for row in rows[1:]):
            raise ValueError(f"action changes inside latent frame at pixel {start}")
        latent.append(rows[0])
    return latent


def _video_boundary_to_latent(boundary: int, temporal_factor: int) -> int:
    if boundary == 0:
        return 0
    if (boundary - 1) % temporal_factor:
        raise ValueError(f"unaligned video boundary: {boundary}")
    return 1 + (boundary - 1) // temporal_factor


def main() -> None:
    args = parse_args()
    minwm_root = Path(args.minwm_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    results = Path(args.results).resolve()
    results.mkdir(parents=True, exist_ok=True)
    for required in (minwm_root / "Wan21", checkpoint, config):
        if not required.exists():
            raise FileNotFoundError(required)

    minwm_sha = subprocess.check_output(
        ["git", "-C", str(minwm_root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected_sha = subprocess.check_output(
        ["git", "-C", str(minwm_root), "rev-parse", EXPECTED_MINWM_COMMIT],
        text=True,
    ).strip()
    if minwm_sha != expected_sha:
        raise ValueError(f"native minWM checkout {minwm_sha} != {expected_sha}")

    contract = load_contract(args.alignment_url)
    sample = contract["sample"]
    target = next(
        message
        for message in sample["messages"]
        if message.get("role") == "target" and message.get("type") == "video"
    )
    world_prompt = next(
        message["content"]
        for message in sample["messages"]
        if message.get("role") == "user" and message.get("type") == "text"
    )
    controls = {control["type"]: control for control in target["controls"]}
    segments = controls["text_prompt_interval"]["segments"]
    block_lengths = [
        int(value) for value in sample["reproducibility"]["latent_block_lengths"]
    ]
    seeds = [int(value) for value in sample["reproducibility"]["seed_schedule"]]
    temporal_factor = 4
    actions = _latent_actions(
        controls["keyboard_direction_frame_interval"]["actions"], temporal_factor
    )
    if len(actions) != sum(block_lengths):
        raise ValueError("latent action and block timelines differ")

    sys.path[:0] = [str(minwm_root / "Wan21"), str(minwm_root)]
    import torch

    from demo_utils.director_service import DirectedSession, _load_runtime

    pipeline, processor = _load_runtime(
        SimpleNamespace(
            config_path=str(config),
            checkpoint_path=str(checkpoint),
            device=args.device,
            low_memory=False,
        )
    )
    dump_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
    if dump_root:
        dump_dir = Path(dump_root) / "baseline"
        dump_dir.mkdir(parents=True, exist_ok=True)
        original_model_flow = pipeline._model_flow
        forward_index = 0
        counters: dict[str, int] = {}

        from wan.modules import flex_attn as minwm_flex_attn

        original_qk_norm = minwm_flex_attn.qk_norm_op
        qk_norm_index = 0

        def qk_norm_with_dump(*qk_args, **qk_kwargs):
            nonlocal qk_norm_index
            query, key = original_qk_norm(*qk_args, **qk_kwargs)
            if qk_norm_index < 2:
                torch.save(
                    query.detach().cpu(),
                    dump_dir / f"self_q_norm_{qk_norm_index:03d}.pt",
                )
                torch.save(
                    key.detach().cpu(),
                    dump_dir / f"self_k_norm_{qk_norm_index:03d}.pt",
                )
            qk_norm_index += 1
            return query, key

        minwm_flex_attn.qk_norm_op = qk_norm_with_dump
        original_apply_rope = minwm_flex_attn.apply_rope_varlen_op
        rope_index = 0

        def apply_rope_with_dump(*rope_args, **rope_kwargs):
            nonlocal rope_index
            value = original_apply_rope(*rope_args, **rope_kwargs)
            forward = rope_index // 2
            if forward < 2:
                kind = "k" if rope_index % 2 == 0 else "q"
                torch.save(
                    value.detach().cpu(),
                    dump_dir / f"self_{kind}_roped_{forward:03d}.pt",
                )
            rope_index += 1
            return value

        minwm_flex_attn.apply_rope_varlen_op = apply_rope_with_dump

        def dump_output(name: str, *, include_input: bool = False):
            counters[name] = 0

            def hook(_module, hook_args, output):
                index = counters[name]
                if index < 2:
                    value = output[0] if isinstance(output, tuple) else output
                    torch.save(
                        value.detach().cpu(),
                        dump_dir / f"{name}_output_{index:03d}.pt",
                    )
                    if include_input:
                        torch.save(
                            hook_args[0].detach().cpu(),
                            dump_dir / f"{name}_input_{index:03d}.pt",
                        )
                counters[name] = index + 1

            return hook

        block0 = pipeline.generator.blocks[0]
        torch.save(
            block0.self_attn.q.weight.detach().cpu(),
            dump_dir / "self_q_weight.pt",
        )
        torch.save(
            block0.self_attn.q.bias.detach().cpu(),
            dump_dir / "self_q_bias.pt",
        )
        detail_modules = {
            "patch": pipeline.generator.patch_embedding,
            "block0": block0,
            "time_embed": pipeline.generator.time_embedding,
            "time_projection": pipeline.generator.time_projection,
            "text_embed": pipeline.generator.text_embedding,
            "self_q": block0.self_attn.q,
            "self_k": block0.self_attn.k,
            "self_v": block0.self_attn.v,
            "self_norm_q": block0.self_attn.norm_q,
            "self_norm_k": block0.self_attn.norm_k,
            "self_out": block0.self_attn.o,
            "self_residual_norm": block0.norm3,
            "cross_q": block0.cross_attn.q,
            "cross_k": block0.cross_attn.k,
            "cross_v": block0.cross_attn.v,
            "cross_norm_q": block0.cross_attn.norm_q,
            "cross_norm_k": block0.cross_attn.norm_k,
            "cross_out": block0.cross_attn.o,
            "ffn": block0.ffn,
        }
        for name, module in detail_modules.items():
            module.register_forward_hook(
                dump_output(
                    name,
                    include_input=name
                    in {
                        "block0",
                        "self_q",
                        "self_out",
                        "self_residual_norm",
                        "cross_q",
                        "cross_out",
                    },
                )
            )

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
            torch.save(
                {
                    "latent_model_input": latents.detach().cpu(),
                    "prompt_embeds": (
                        conditional_dict["flat_prompt_embeds"].detach().cpu()
                        if forward_index == 0
                        else None
                    ),
                    "prompt_lens": conditional_dict["prompt_lens"],
                    "timestep": timestep.detach().cpu(),
                    "action": None if action is None else action.detach().cpu(),
                    "self_cache_update": self_cache_update,
                    "condition_switch": condition_switch,
                    "output": output.detach().cpu(),
                },
                dump_dir / f"forward_{forward_index:03d}.pt",
            )
            forward_index += 1
            return output

        pipeline._model_flow = model_flow_with_dump
    expected_config = {
        "height": 480,
        "width": 832,
        "num_frame_first_block": 1,
        "num_frame_per_block": 4,
        "denoising_step_list": [1000, 750, 500, 250],
        "denoising_step_list_first_block": [1000, 750, 500, 250],
        "guidance_scale": 0.0,
        "local_attn_size": 32,
        "sink_size": 8,
        "rope_position_mode": "block_relative",
        "rope_max_frame_gap": 12,
        "prompt_first_frame_pin_enabled": True,
        "action_type": "primitive_token_residual",
        "action_output_format": "primitive_float",
        "action_effective_ratio": 0.0,
        "target_fps": 24,
    }
    actual_config = {
        "height": int(pipeline.config.height),
        "width": int(pipeline.config.width),
        "num_frame_first_block": int(pipeline.config.num_frame_first_block),
        "num_frame_per_block": int(pipeline.config.num_frame_per_block),
        "denoising_step_list": list(pipeline.config.denoising_step_list),
        "denoising_step_list_first_block": list(
            pipeline.config.denoising_step_list_first_block
        ),
        "guidance_scale": float(pipeline.config.guidance_scale),
        "local_attn_size": int(pipeline.config.generator_config.local_attn_size),
        "sink_size": int(pipeline.config.generator_config.sink_size),
        "rope_position_mode": str(pipeline.config.generator_config.rope_position_mode),
        "rope_max_frame_gap": int(pipeline.config.generator_config.rope_max_frame_gap),
        "prompt_first_frame_pin_enabled": bool(
            pipeline.config.multi_shot_config["condition_switch"][
                "prompt_first_frame_pin_enabled"
            ]
        ),
        "action_type": str(pipeline.config.action_config.type),
        "action_output_format": str(processor.action_output_format),
        "action_effective_ratio": float(processor.action_effective_ratio),
        "target_fps": int(processor.target_fps),
    }
    if actual_config != expected_config:
        raise ValueError(
            f"native baseline config changed: {actual_config} != {expected_config}"
        )

    session = DirectedSession(
        pipeline,
        processor,
        world_prompt=world_prompt,
        metadata=sample.get("metadata"),
    )
    block_cursor = 0
    latent_cursor = 0
    step_timings = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    max_chunks = args.max_chunks
    if max_chunks is not None and not 1 <= max_chunks <= len(block_lengths):
        raise ValueError(f"max_chunks must be in [1, {len(block_lengths)}]")
    for segment_index, segment in enumerate(segments):
        segment_start = _video_boundary_to_latent(
            int(segment["start"]), temporal_factor
        )
        segment_end = _video_boundary_to_latent(int(segment["end"]), temporal_factor)
        if segment_start != latent_cursor:
            raise ValueError("prompt segments do not cover latent timeline")
        session.begin(f"prompt-{segment_index}", segment["text"])
        while latent_cursor < segment_end:
            block_length = block_lengths[block_cursor]
            block_end = latent_cursor + block_length
            if block_end > segment_end:
                raise ValueError("prompt boundary falls inside a director block")
            torch.cuda.synchronize()
            step_started = time.perf_counter()
            session.step(
                seed=seeds[block_cursor],
                actions=actions[latent_cursor:block_end],
            )
            torch.cuda.synchronize()
            step_timings.append(
                {
                    "chunk_index": block_cursor,
                    "latent_frames": block_length,
                    "seconds": time.perf_counter() - step_started,
                }
            )
            latent_cursor = block_end
            block_cursor += 1
            if max_chunks is not None and block_cursor >= max_chunks:
                break
        session.commit()
        if max_chunks is not None and block_cursor >= max_chunks:
            break
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - started
    if max_chunks is None and block_cursor != len(block_lengths):
        raise ValueError("not every director block was generated")

    torch.cuda.synchronize()
    decode_started = time.perf_counter()
    video, state = session.finalize()
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_started
    frames = (
        video[0]
        .permute(0, 2, 3, 1)
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    expected_frames = 1 + (latent_cursor - 1) * temporal_factor
    expected_shape = (expected_frames, 480, 832, 3)
    if frames.shape != expected_shape:
        raise ValueError(f"native output {frames.shape} != {expected_shape}")

    np.save(results / "native_minwm.npy", frames, allow_pickle=False)
    save_video(results / "native_minwm.mp4", frames, int(processor.target_fps))
    torch.save(
        session._committed_latents.clone(),
        results / "native_minwm_latents.pt",
    )
    baseline = results / "baseline.mp4"
    baseline.write_bytes(_fetch_bytes(contract["reference_url"]))
    if sha256_file(baseline) != EXPECTED_VIDEO_SHA256:
        raise ValueError("published baseline video SHA-256 changed")

    record = {
        "sample_id": EXPECTED_SAMPLE_ID,
        "minwm_git_sha": minwm_sha,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_contract": actual_config,
        "state": state,
        "generation_seconds": generation_seconds,
        "decode_seconds": decode_seconds,
        "raw_video_fps": frames.shape[0] / (generation_seconds + decode_seconds),
        "step_timings": step_timings,
        "native_frames_sha256": sha256_file(results / "native_minwm.npy"),
        "native_video_sha256": sha256_file(results / "native_minwm.mp4"),
        "native_latents_sha256": sha256_file(results / "native_minwm_latents.pt"),
        "published_baseline_sha256": sha256_file(baseline),
        "published_baseline_psnr": (
            _run_ffmpeg_metric("psnr", baseline, results / "native_minwm.mp4")
            if max_chunks is None
            else None
        ),
        "published_baseline_ssim": (
            _run_ffmpeg_metric("ssim", baseline, results / "native_minwm.mp4")
            if max_chunks is None
            else None
        ),
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
    }
    write_json(results / "native_baseline.json", record)
    print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
