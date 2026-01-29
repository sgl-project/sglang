# This script benchmarks MRotaryEmbedding.get_rope_index_glm4v (GLM4V mrope index builder).
# It generates synthetic multimodal input_ids + attention_mask (+ optional image/video grids),
# runs benchmarks.
#
# == Usage Examples ==
#
# python3 benchmark_rope_index.py --device cuda --num-tokens 1024 2048 --benchmark-iter 200

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang.srt.layers.rotary_embedding import MRotaryEmbedding


# -----------------------------
# Minimal config objects
# -----------------------------
@dataclass
class DummyVisionConfig:
    spatial_merge_size: int = 2


@dataclass
class DummyHFConfig:
    image_token_id: int = 32000
    video_start_token_id: int = 32001
    video_end_token_id: int = 32002
    vision_config: DummyVisionConfig = field(
        default_factory=lambda: DummyVisionConfig(spatial_merge_size=2)
    )


# -----------------------------
# Helpers
# -----------------------------
def calculate_stats(times: list[float]) -> dict[str, float]:
    """Calculate statistics from a list of times."""
    times_array = np.array(times, dtype=np.float64)
    return {
        "mean": float(np.mean(times_array)),
        "median": float(np.median(times_array)),
        "p99": float(np.percentile(times_array, 99)),
        "min": float(np.min(times_array)),
        "max": float(np.max(times_array)),
    }


def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _approx_hw(patches: int, merge: int) -> tuple[int, int]:
    # want (h/merge)*(w/merge) ~= patches
    gh = int(math.sqrt(max(1, patches)))
    gw = max(1, patches // max(1, gh))
    return gh * merge, gw * merge


def generate_test_data(
    num_tokens: int,
    batch_size: int,
    hf_config: DummyHFConfig,
    dtype: torch.dtype,
    device: torch.device,
    pad_ratio: float,
    num_images_per_sample: int,
    image_patch_tokens: int,
    num_videos_per_sample: int,
    video_patch_tokens: int,
    seed: int,
):
    """
    Generate synthetic (input_ids, attention_mask, image_grid_thw, video_grid_thw).

    NOTE:
      - image_grid_thw / video_grid_thw are global lists across the entire batch in encounter order,
        matching the function's image_index/video_index behavior.
      - image patches are represented by repeated image_token_id.
      - video patches are represented by image_token_id wrapped with start/end tokens.
    """
    torch.manual_seed(seed)

    forbidden = {
        0,
        hf_config.image_token_id,
        hf_config.video_start_token_id,
        hf_config.video_end_token_id,
    }
    vocab_size = 50000

    def rand_text(n: int) -> torch.Tensor:
        # generate random ids not in forbidden
        out = torch.randint(1, vocab_size, (n,), device=device, dtype=torch.long)
        # fix forbidden by +1 until ok (cheap, deterministic enough for benchmark data)
        for bad in forbidden:
            out = torch.where(out == bad, out + 1, out)
        return out

    image_grids: list[list[int]] = []
    video_grids: list[list[int]] = []

    input_ids = torch.zeros((batch_size, num_tokens), device=device, dtype=torch.long)
    attention_mask = torch.zeros(
        (batch_size, num_tokens), device=device, dtype=torch.long
    )

    eff_len = int(round(num_tokens * (1.0 - pad_ratio)))
    eff_len = max(1, min(num_tokens, eff_len))

    min_needed = 1
    min_needed += num_images_per_sample * image_patch_tokens
    min_needed += num_videos_per_sample * (2 + video_patch_tokens)
    if eff_len < min_needed:
        num_images_per_sample = 0
        num_videos_per_sample = 0

    for b in range(batch_size):
        blocks: list[torch.Tensor] = []

        reserved = (
            num_images_per_sample * image_patch_tokens
            + num_videos_per_sample * (2 + video_patch_tokens)
        )
        reserved = min(reserved, max(0, eff_len - 1))
        text_budget = max(1, eff_len - reserved)

        n_text_chunks = num_images_per_sample + num_videos_per_sample + 1
        base = text_budget // n_text_chunks
        rem = text_budget % n_text_chunks
        text_chunks = [base + (1 if i < rem else 0) for i in range(n_text_chunks)]

        tci = 0
        for _ in range(num_images_per_sample):
            blocks.append(rand_text(text_chunks[tci]))
            tci += 1
            blocks.append(
                torch.full(
                    (image_patch_tokens,),
                    hf_config.image_token_id,
                    device=device,
                    dtype=torch.long,
                )
            )

            h, w = _approx_hw(
                image_patch_tokens, hf_config.vision_config.spatial_merge_size
            )
            image_grids.append([1, h, w])

        for _ in range(num_videos_per_sample):
            blocks.append(rand_text(text_chunks[tci]))
            tci += 1
            blocks.append(
                torch.tensor(
                    [hf_config.video_start_token_id], device=device, dtype=torch.long
                )
            )
            blocks.append(
                torch.full(
                    (video_patch_tokens,),
                    hf_config.image_token_id,
                    device=device,
                    dtype=torch.long,
                )
            )
            blocks.append(
                torch.tensor(
                    [hf_config.video_end_token_id], device=device, dtype=torch.long
                )
            )

            h, w = _approx_hw(
                video_patch_tokens, hf_config.vision_config.spatial_merge_size
            )
            # first field = group count used by code; set to 1
            video_grids.append([1, h, w])

        blocks.append(rand_text(text_chunks[tci]))

        tokens = torch.cat(blocks, dim=0)[:eff_len]
        pad = torch.zeros(
            (num_tokens - tokens.numel(),), device=device, dtype=torch.long
        )
        ids = torch.cat([tokens, pad], dim=0)

        mask = torch.cat(
            [
                torch.ones((tokens.numel(),), device=device, dtype=torch.long),
                torch.zeros(
                    (num_tokens - tokens.numel(),), device=device, dtype=torch.long
                ),
            ],
            dim=0,
        )

        input_ids[b] = ids
        attention_mask[b] = mask

    image_grid_thw = (
        torch.tensor(image_grids, device=device, dtype=torch.long)
        if len(image_grids)
        else None
    )
    video_grid_thw = (
        torch.tensor(video_grids, device=device, dtype=torch.long)
        if len(video_grids)
        else None
    )
    return (
        input_ids.to(dtype=torch.long),
        attention_mask.to(dtype=torch.long),
        image_grid_thw,
        video_grid_thw,
    )


def benchmark_rope_index(
    model_name: str,
    tp_size: int,
    num_tokens: int,
    batch_size: int,
    pad_ratio: float,
    spatial_merge_size: int,
    num_images: int,
    image_patch_tokens: int,
    num_videos: int,
    video_patch_tokens: int,
    dtype: torch.dtype,
    seed: int,
    warmup_iter: int,
    benchmark_iter: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    hf_config = DummyHFConfig(
        image_token_id=32000,
        video_start_token_id=32001,
        video_end_token_id=32002,
        vision_config=DummyVisionConfig(spatial_merge_size=spatial_merge_size),
    )

    print(80 * "=")
    print(
        f"Evaluating: {model_name} tp_size={tp_size} "
        f"num_tokens={num_tokens} batch={batch_size} pad_ratio={pad_ratio} "
        f"images/sample={num_images} image_patch_tokens={image_patch_tokens} "
        f"videos/sample={num_videos} video_patch_tokens={video_patch_tokens} "
        f"dtype={dtype} device={device}"
    )

    input_ids, attention_mask, image_grid_thw, video_grid_thw = generate_test_data(
        num_tokens=num_tokens,
        batch_size=batch_size,
        hf_config=hf_config,
        dtype=dtype,
        device=device,
        pad_ratio=pad_ratio,
        num_images_per_sample=num_images,
        image_patch_tokens=image_patch_tokens,
        num_videos_per_sample=num_videos,
        video_patch_tokens=video_patch_tokens,
        seed=seed,
    )

    # Smoke test
    has_mm = (image_grid_thw is not None) or (video_grid_thw is not None)
    if has_mm:
        pos, delta = MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        assert pos.shape == (3, batch_size, num_tokens)
        assert delta.shape == (batch_size, 1)

    # Warm up
    for _ in range(warmup_iter):
        if has_mm:
            MRotaryEmbedding.get_rope_index_glm4v(
                input_ids=input_ids,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )
        MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids,
            hf_config=hf_config,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

    _sync(device)

    # Time multimodal branch
    multimodal_times = []
    for _ in range(benchmark_iter):
        _sync(device)
        start = time.time()
        MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids,
            hf_config=hf_config,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        _sync(device)
        multimodal_times.append(time.time() - start)

    # Time fallback branch
    fallback_times = []
    for _ in range(benchmark_iter):
        _sync(device)
        start = time.time()
        MRotaryEmbedding.get_rope_index_glm4v(
            input_ids=input_ids,
            hf_config=hf_config,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )
        _sync(device)
        fallback_times.append(time.time() - start)

    multimodal_stats = calculate_stats(multimodal_times)
    fallback_stats = calculate_stats(fallback_times)

    print(f"\nPerformance for config (B={batch_size}, T={num_tokens}):")
    print(
        f"Multimodal: mean={multimodal_stats['mean']:.8f}s, "
        f"median={multimodal_stats['median']:.8f}s, "
        f"p99={multimodal_stats['p99']:.8f}s"
    )
    print(
        f"Fallback:   mean={fallback_stats['mean']:.8f}s, "
        f"median={fallback_stats['median']:.8f}s, "
        f"p99={fallback_stats['p99']:.8f}s"
    )

    if has_mm:
        speedup = (
            multimodal_stats["mean"] / fallback_stats["mean"]
            if fallback_stats["mean"] > 0
            else float("inf")
        )
        print(f"Fallback Speedup over Multimodal: {speedup:.8f}x")
    else:
        speedup = float("nan")
        print(
            "[INFO] num_tokens too small for multimodal segments; skip multimodal benchmark."
        )

    print(f"Fallback Speedup over Multimodal: {speedup:.8f}x")

    return multimodal_stats, fallback_stats, speedup


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark GLM4V get_rope_index_glm4v."
    )
    parser.add_argument("--model-name", type=str, default="GLM4V")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup-iter", type=int, default=10)
    parser.add_argument("--benchmark-iter", type=int, default=100)
    parser.add_argument("--dtype", type=str, choices=["int64"], default="int64")
    parser.add_argument("--seed", type=int, default=0)

    # token length sweep
    parser.add_argument("--num-tokens", type=int, nargs="+", required=False)

    # data shape knobs
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pad-ratio", type=float, default=0.0)
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--image-patch-tokens", type=int, default=256)
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--video-patch-tokens", type=int, default=256)

    # output
    parser.add_argument("--out-dir", type=str, default=".")
    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)

    if args.num_tokens is None:
        num_tokens_list = [2**i for i in range(0, 18)]
    else:
        num_tokens_list = args.num_tokens

    rows: list[dict[str, Any]] = []

    for num_tokens in num_tokens_list:
        multimodal_stats, fallback_stats, speedup = benchmark_rope_index(
            model_name=args.model_name,
            tp_size=args.tp_size,
            num_tokens=num_tokens,
            batch_size=args.batch_size,
            pad_ratio=args.pad_ratio,
            spatial_merge_size=args.spatial_merge_size,
            num_images=args.num_images,
            image_patch_tokens=args.image_patch_tokens,
            num_videos=args.num_videos,
            video_patch_tokens=args.video_patch_tokens,
            dtype=getattr(torch, args.dtype),
            seed=args.seed,
            warmup_iter=args.warmup_iter,
            benchmark_iter=args.benchmark_iter,
            device=device,
        )
