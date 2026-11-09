# SPDX-License-Identifier: Apache-2.0
"""Real H3 quality switching: compare baseline and FP8-enabled deployments.

Run once without --fp8, then with --fp8 and the same --output directory.
Frames are compared before lossy file encoding; timings include quantization.
Requires a CUDA SM100+ GPU with room for the unquantized resident pipeline.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator

PROMPTS = (
    "A quiet sunlit kitchen, detailed wooden furniture, slow camera pan, birdsong outside.",
    "A cyclist rides through a busy rainy city street at night, reflections and traffic sounds.",
    "Close-up of a person speaking beside a window, natural skin detail and soft daylight.",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--short-edge", type=int, default=768)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    label = "fp8" if args.fp8 else "baseline"
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "steps": args.steps,
        "short_edge": args.short_edge,
        "requests": [],
    }
    generator = DiffGenerator.from_pretrained(
        model_path="MiniMaxAI/MiniMax-H3",
        model_variant="fl2va",
        num_gpus=1,
        performance_mode="speed",
        enable_torch_compile=False,
        warmup_mode="off",
        component_quantizations={"video_vae": "fp8"} if args.fp8 else {},
    )
    try:
        for index, prompt in enumerate(PROMPTS):
            order = (
                ("exact", "lossless", "high", "lossless", "high", "exact")
                if args.fp8
                else ("exact", "lossless", "exact", "lossless")
            )
            for iteration, quality in enumerate(order):
                start = time.perf_counter()
                result = generator.generate(
                    {
                        "prompt": prompt,
                        "task": "t2va",
                        "conditions": [],
                        "target": {
                            "short_edge": args.short_edge,
                            "aspect_ratio": "16:9",
                            "duration_seconds": 4.0,
                        },
                        "num_inference_steps": args.steps,
                        "seed": 1101 + index,
                        "quality": quality,
                        "save_output": False,
                        "return_file_paths_only": False,
                    }
                )
                elapsed = time.perf_counter() - start
                if result is None:
                    raise RuntimeError(f"generation failed: {index}/{quality}")
                frames = np.stack(result.frames)
                audio_hash = hashlib.sha256(
                    np.asarray(result.audio).tobytes()
                ).hexdigest()
                reference_path = args.output / f"reference-{index}-{quality}.npz"
                if not args.fp8 and not reference_path.exists():
                    np.savez(reference_path, frames=frames, audio_hash=audio_hash)
                row = {
                    "prompt": index,
                    "quality": quality,
                    "iteration": iteration,
                    "wall_s": elapsed,
                    "generation_s": result.generation_time,
                    "peak_memory_mb": result.peak_memory_mb,
                    "metrics": result.metrics,
                }
                if quality == "high":
                    reference_path = args.output / f"reference-{index}-lossless.npz"
                with np.load(reference_path) as reference:
                    expected = reference["frames"]
                    assert frames.shape == expected.shape
                    assert audio_hash == str(reference["audio_hash"])
                    if quality != "high":
                        np.testing.assert_array_equal(frames, expected)
                    else:
                        mse = ((frames.astype(np.float64) - expected) ** 2).mean(
                            axis=(1, 2, 3)
                        )
                        row["worst_frame_psnr"] = float(
                            np.min(10 * np.log10(255**2 / np.maximum(mse, 1e-12)))
                        )
                        row["worst_frame_ssim"] = min(
                            structural_similarity(a, b, channel_axis=-1, data_range=255)
                            for a, b in zip(frames, expected)
                        )
                        assert row["worst_frame_psnr"] >= 40
                        assert row["worst_frame_ssim"] >= 0.98
                report["requests"].append(row)
                (args.output / f"{label}.json").write_text(json.dumps(report, indent=2))
                print(json.dumps(row), flush=True)
    finally:
        generator.shutdown()


if __name__ == "__main__":
    main()
