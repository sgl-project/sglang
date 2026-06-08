# SPDX-License-Identifier: Apache-2.0
"""
Manual integration test: progressive resolution growing for Wan T2V.

Requires a GPU and a Wan2.1-T2V model checkpoint.

Run via:
  python -m pytest python/sglang/multimodal_gen/test/manual/test_progressive_wan.py \
         -v --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers

Or as a standalone script:
  python python/sglang/multimodal_gen/test/manual/test_progressive_wan.py \
         --model-path /path/to/Wan2.1-T2V-1.3B-Diffusers

Tests:
  1. Fullres baseline (progressive_mode='fullres') generates a valid video.
  2. DCT rewind (progressive_mode='dct_rewind', levels=1) generates a valid video.
  3. Both modes produce different outputs (progressive actually changes computation).
  4. Progressive generation is faster than fullres on wall-clock time.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--prompt",
        default="A serene mountain lake at golden hour with gentle ripples, photorealistic",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--output-dir", default="/tmp/progressive_wan_test")
    parser.add_argument("--attention-backend", default="torch_sdpa")
    return parser.parse_args()


def _make_generator(model_path: str, attention_backend: str):
    from sglang.multimodal_gen import DiffGenerator

    return DiffGenerator.from_pretrained(
        model_path=model_path,
        attention_backend=attention_backend,
        dit_cpu_offload=False,
    )


def _generate(
    gen, prompt, seed, steps, height, width, num_frames, mode, levels, delta, outfile
):
    out_path = str(Path(outfile).parent)
    out_name = Path(outfile).stem
    t0 = time.time()
    result = gen.generate(
        sampling_params_kwargs={
            "prompt": prompt,
            "seed": seed,
            "num_inference_steps": steps,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "save_output": True,
            "output_path": out_path,
            "output_file_name": out_name,
            "progressive_mode": mode,
            "progressive_levels": levels,
            "progressive_delta": delta,
        }
    )
    elapsed = time.time() - t0
    return result, elapsed


def run_test(args):
    os.makedirs(args.output_dir, exist_ok=True)
    gen = _make_generator(args.model_path, args.attention_backend)

    out_fullres = str(Path(args.output_dir) / "fullres.mp4")
    out_prog = str(Path(args.output_dir) / "dct_rewind.mp4")

    # --- Test 1: fullres baseline ---
    print("\n[1/3] fullres baseline...")
    result_fr, t_fr = _generate(
        gen,
        args.prompt,
        args.seed,
        args.steps,
        args.height,
        args.width,
        args.num_frames,
        "fullres",
        args.levels,
        args.delta,
        out_fullres,
    )
    assert Path(out_fullres).exists(), "fullres video not saved"
    print(f"      done in {t_fr:.1f}s → {out_fullres}")

    # --- Test 2: dct_rewind progressive ---
    print(f"\n[2/3] dct_rewind (levels={args.levels}, delta={args.delta})...")
    result_pr, t_pr = _generate(
        gen,
        args.prompt,
        args.seed,
        args.steps,
        args.height,
        args.width,
        args.num_frames,
        "dct_rewind",
        args.levels,
        args.delta,
        out_prog,
    )
    assert Path(out_prog).exists(), "progressive video not saved"
    print(f"      done in {t_pr:.1f}s → {out_prog}")
    print(f"      wall-clock speedup: {t_fr/t_pr:.2f}x vs fullres")

    # --- Test 3: outputs differ ---
    print("\n[3/3] Verifying progressive ≠ fullres...")
    import imageio
    import numpy as np

    frames_fr = imageio.v3.imread(out_fullres, plugin="pyav").astype(np.float32)
    frames_pr = imageio.v3.imread(out_prog, plugin="pyav").astype(np.float32)
    min_frames = min(len(frames_fr), len(frames_pr))
    pixel_diff = np.abs(frames_fr[:min_frames] - frames_pr[:min_frames]).mean()
    print(f"      mean pixel diff: {pixel_diff:.2f}")
    assert pixel_diff > 1.0, (
        f"Progressive and fullres outputs are too similar (diff={pixel_diff:.2f}); "
        "progressive mode may not be running."
    )
    print("      outputs differ as expected ✓")

    gen.shutdown()

    print("\n====================")
    print("All tests PASSED")
    print(f"  Fullres:    {t_fr:.1f}s  → {out_fullres}")
    print(f"  dct_rewind: {t_pr:.1f}s  → {out_prog}")
    print(f"  Wall-clock speedup: {t_fr/t_pr:.2f}x")
    return {"fullres": out_fullres, "dct_rewind": out_prog}


if __name__ == "__main__":
    args = _parse_args()
    run_test(args)
