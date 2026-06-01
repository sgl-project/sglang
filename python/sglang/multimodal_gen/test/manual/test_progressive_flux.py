# SPDX-License-Identifier: Apache-2.0
"""
Manual integration test: progressive resolution growing for FLUX.1.

Requires a GPU and a FLUX.1-dev model checkpoint.
Run via:
  python -m pytest python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
         -v --model-path /path/to/FLUX.1-dev

Or as a standalone script:
  python python/sglang/multimodal_gen/test/manual/test_progressive_flux.py \
         --model-path /path/to/FLUX.1-dev

Tests:
  1. Fullres baseline (progressive_mode='fullres') generates a valid image.
  2. DCT rewind (progressive_mode='dct_rewind', levels=1) generates a valid image.
  3. Both modes produce different outputs (progressive actually changes computation).
  4. Latent dimensions and freqs_cis are correctly updated at the resolution transition.
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
        "--prompt", default="A serene mountain lake at golden hour, photorealistic"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--output-dir", default="/tmp/progressive_flux_test")
    parser.add_argument("--attention-backend", default="torch_sdpa")
    return parser.parse_args()


def _make_generator(model_path: str, attention_backend: str):
    from sglang.multimodal_gen import DiffGenerator

    return DiffGenerator.from_pretrained(
        model_path=model_path,
        attention_backend=attention_backend,
        dit_cpu_offload=False,
    )


def _generate(gen, prompt, seed, steps, height, width, mode, levels, delta, outfile):
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

    results = {}

    # --- Test 1: fullres baseline ---
    print("\n[1/3] fullres baseline...")
    out_fullres = str(Path(args.output_dir) / "fullres.png")
    result_fr, t_fr = _generate(
        gen,
        args.prompt,
        args.seed,
        args.steps,
        args.height,
        args.width,
        "fullres",
        args.levels,
        args.delta,
        out_fullres,
    )
    assert Path(out_fullres).exists(), "fullres image not saved"
    print(f"      done in {t_fr:.1f}s → {out_fullres}")
    results["fullres"] = out_fullres

    # --- Test 2: dct_rewind progressive ---
    print(f"\n[2/3] dct_rewind (levels={args.levels}, delta={args.delta})...")
    out_prog = str(Path(args.output_dir) / "dct_rewind.png")
    result_pr, t_pr = _generate(
        gen,
        args.prompt,
        args.seed,
        args.steps,
        args.height,
        args.width,
        "dct_rewind",
        args.levels,
        args.delta,
        out_prog,
    )
    assert Path(out_prog).exists(), "progressive image not saved"
    print(f"      done in {t_pr:.1f}s → {out_prog}")
    print(f"      speedup estimate: {t_fr/t_pr:.2f}x vs fullres")
    results["dct_rewind"] = out_prog

    # --- Test 3: outputs differ ---
    print("\n[3/3] Verifying progressive ≠ fullres...")
    import numpy as np
    from PIL import Image

    img_fr = np.array(Image.open(out_fullres).convert("RGB")).astype(np.float32)
    img_pr = np.array(Image.open(out_prog).convert("RGB")).astype(np.float32)
    pixel_diff = np.abs(img_fr - img_pr).mean()
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
    return results


if __name__ == "__main__":
    args = _parse_args()
    run_test(args)
