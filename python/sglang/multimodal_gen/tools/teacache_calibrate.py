# SPDX-License-Identifier: Apache-2.0
"""Generate TeaCache polynomial coefficients for a diffusion model.

TeaCache skips redundant denoising steps using per-model polynomial
coefficients + a threshold; a model that ships none (e.g. LongCat-Image,
Wan2.2) cannot use TeaCache. This tool produces them: it runs generations with
TeaCache disabled (so every step is computed), records per-step
(Δmodulated_input, Δoutput), fits the degree-4 polynomial + threshold, and
writes a JSON consumable by ``TeaCacheParams``. Stable coefficients need >=50
diverse prompts; use --prompts-file and shard across GPUs.

Two modes:

  record (default): run generations for this process's shard of prompts; the
  worker subprocess dumps accumulated rows to --rows-out.
    CUDA_VISIBLE_DEVICES=0 python -m \
      sglang.multimodal_gen.tools.teacache_calibrate \
      --model-path /path/LongCat-Image --prompts-file prompts.txt \
      --shard 0/4 --width 1024 --height 1024 --num-inference-steps 50 \
      --guidance-scale 4.5 --rows-out rows_0.json

  fit: merge shard row files (count-weighted) and fit the coefficients.
    python -m sglang.multimodal_gen.tools.teacache_calibrate \
      --merge-rows rows_0.json,rows_1.json,rows_2.json,rows_3.json \
      --output ./fast_infer/teacache.json

Single-transformer models yield ``{"coefficients": [...], "teacache_thresh": ...}``;
MoE dual-transformer models yield ``{"high": {...}, "low": {...}}``.
"""

from __future__ import annotations

import argparse
import json
import os


def build_server_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "backend": "sglang",
        "num_gpus": args.num_gpus,
        "ulysses_degree": args.ulysses_degree,
    }
    # Distinct ports per shard so parallel per-GPU processes don't collide on
    # the otherwise-fixed master_port/port.
    if args.port_base is not None:
        kwargs["master_port"] = args.port_base
        kwargs["port"] = args.port_base + 1
        kwargs["scheduler_port"] = args.port_base + 2
    # Keep everything resident when VRAM is ample: the auto memory policy may
    # layerwise-offload the text encoder / VAE, which only slows the one-time
    # text-encode and decode. These explicit flags win over the auto policy.
    if args.no_offload:
        kwargs["dit_layerwise_offload"] = False
        kwargs["text_encoder_cpu_offload"] = False
        kwargs["vae_cpu_offload"] = False
        kwargs["layerwise_offload_components"] = []
    return kwargs


def build_sampling_kwargs(args: argparse.Namespace, item: dict) -> dict:
    kwargs: dict = {
        "prompt": item["prompt"],
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        # TeaCache OFF: compute every step so the calibrator sees the true
        # step-to-step deltas rather than cached residuals.
        "enable_teacache": False,
    }
    image_path = item.get("image_path", args.image_path)
    if image_path is not None:
        kwargs["image_path"] = image_path
    if args.negative_prompt is not None:
        kwargs["negative_prompt"] = args.negative_prompt
    if args.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = args.guidance_scale_2
    if args.num_frames is not None:
        kwargs["num_frames"] = args.num_frames
    return kwargs


def load_prompts(args: argparse.Namespace) -> list[dict]:
    """One item per generation: {"prompt": str, "image_path": str | None}.

    A prompts file is JSONL (``{"prompt": ..., "image_path": ...}`` per line) or
    plain text (one prompt per line). Without a file, the single --prompt is
    repeated --samples times.
    """
    if args.prompts_file is None:
        return [{"prompt": args.prompt}] * args.samples
    items: list[dict] = []
    with open(args.prompts_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line) if line[0] in "{[" else {"prompt": line})
    return items


def shard_items(items: list[dict], shard: str | None) -> list[dict]:
    if shard is None:
        return items
    idx, total = (int(x) for x in shard.split("/"))
    return items[idx::total]


def run_fit(args: argparse.Namespace) -> None:
    from sglang.multimodal_gen.runtime.cache.teacache_calibrate import (
        TeaCacheCalibrator,
        merge_rows_dicts,
    )

    rows_dicts = []
    for path in args.merge_rows.split(","):
        with open(path.strip()) as f:
            rows_dicts.append(json.load(f))
    merged = merge_rows_dicts(rows_dicts)
    calibrator = TeaCacheCalibrator.from_rows_dict(
        merged, degree=args.degree, slope_threshold=args.slope_threshold
    )
    result = calibrator.save_json(args.output)
    print(f"[teacache-calibrate] merged {len(rows_dicts)} shard(s) -> {args.output}")
    print(json.dumps(result, indent=2))


def run_record(args: argparse.Namespace) -> None:
    items = shard_items(load_prompts(args), args.shard)
    if not items:
        raise ValueError("No prompts to record for this shard.")

    # The DiT forward runs in a worker subprocess, so calibration is activated by
    # env vars the subprocess inherits: the worker records every step and dumps
    # the accumulated rows to this file.
    rows_out = args.rows_out or (args.output + ".rows.json")
    os.environ["SGLANG_TEACACHE_CALIBRATE"] = "1"
    os.environ["SGLANG_TEACACHE_CALIBRATE_OUT"] = rows_out
    if os.path.exists(rows_out):
        os.remove(rows_out)

    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    with DiffGenerator.from_pretrained(
        local_mode=True, **build_server_kwargs(args)
    ) as generator:
        for i, item in enumerate(items):
            print(
                f"[teacache-calibrate] shard={args.shard or 'all'} "
                f"prompt {i + 1}/{len(items)}",
                flush=True,
            )
            generator.generate(sampling_params_kwargs=build_sampling_kwargs(args, item))

    print(f"[teacache-calibrate] recorded rows -> {rows_out}")
    if not args.no_fit:
        from sglang.multimodal_gen.runtime.cache.teacache_calibrate import (
            TeaCacheCalibrator,
        )

        with open(rows_out) as f:
            rows = json.load(f)
        result = TeaCacheCalibrator.from_rows_dict(
            rows, degree=args.degree, slope_threshold=args.slope_threshold
        ).save_json(args.output)
        print(f"[teacache-calibrate] wrote {args.output}")
        print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--model-id")
    parser.add_argument("--prompt")
    parser.add_argument("--prompts-file", help="JSONL or plain-text prompts.")
    parser.add_argument("--shard", help="IDX/TOTAL, e.g. 0/4 (round-robin subset).")
    parser.add_argument("--negative-prompt")
    parser.add_argument("--image-path", help="Default input image for edit models.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-frames", type=int, help="Video frame count (T2V/I2V).")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--guidance-scale-2", type=float)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--ulysses-degree", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--slope-threshold", type=float, default=0.01)
    parser.add_argument("--rows-out", help="Where this shard dumps its rows.")
    parser.add_argument(
        "--port-base",
        type=int,
        help="Base port (master/port/scheduler) so parallel shards don't collide.",
    )
    parser.add_argument(
        "--no-fit", action="store_true", help="Record only; fit later via --merge-rows."
    )
    parser.add_argument(
        "--no-offload",
        action="store_true",
        help="Force text_encoder/VAE/DiT resident (skip auto layerwise offload).",
    )
    parser.add_argument(
        "--merge-rows", help="Comma-separated rows files to merge+fit (no generation)."
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.merge_rows:
        run_fit(args)
    else:
        if args.model_path is None:
            parser.error("--model-path is required for recording.")
        run_record(args)


if __name__ == "__main__":
    main()
