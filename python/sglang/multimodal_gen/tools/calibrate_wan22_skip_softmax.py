#!/usr/bin/env python
"""Offline BLASST skip-softmax calibration for Wan2.2-T2V-A14B.

The ``flashinfer_trtllm_skip_softmax`` attention backend consumes a calibration
JSON via ``--attention-backend-config target_sparsity=<t>,calibration_path=<file>``.
This script *produces* that JSON.

It loads the real Wan2.2-T2V-A14B pipeline (BF16) and runs NVIDIA Model
Optimizer's official skip-softmax calibration (``DynamicThresholdCalibrator``:
sweeps threshold trials, measures achieved tile-sparsity, and fits
``scale_factor = a * exp(b * target_sparsity)``) on each denoiser
(``transformer`` and, for the 14B MoE-style model, ``transformer_2``). The output
JSON carries per-component ``(a, b)`` plus a flat top-level ``(a, b)`` that the
backend resolves to for any block via the ``_default`` fallback.

Calibration is resolution dependent: the ``(a, b)`` fitted at one resolution do
not transfer to another (the ``/seq_len`` normalization does not fully cancel the
tile-distribution shift). Calibrate at the resolution you intend to serve.

Requirements
------------
- ``nvidia-modelopt`` installed, and a checkout of the Model Optimizer repo that
  ships ``examples/diffusers/sparsity/wan22_skip_softmax.py`` (this script reuses
  that example's forward-loop / sparse-config builders). Point ``--modelopt-example``
  (or ``MODELOPT_WAN22_EXAMPLE``) at that file.
- ``diffusers`` with ``WanPipeline`` / ``AutoencoderKLWan`` support.
- A GPU with enough memory for the 14B pipeline in BF16 (e.g. B200 / H100 80GB+).

Example
-------
    python python/sglang/multimodal_gen/tools/calibrate_wan22_skip_softmax.py \
        --modelopt-example /path/Model-Optimizer/examples/diffusers/sparsity/wan22_skip_softmax.py \
        --width 1280 --height 720 --num-frames 81 \
        --target-sparsity 0.5 --calib-steps 8 --calib-size 2 \
        --out ./wan22_calib_14b_720p.json

Then serve with:
    --attention-backend flashinfer_trtllm_skip_softmax \
    --attention-backend-config 'target_sparsity=0.5,calibration_path=./wan22_calib_14b_720p.json'
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

# A few diverse video captions used only to drive forward passes so the
# calibrator can observe attention-score distributions.
INLINE_PROMPTS = [
    "A cinematic shot of a snow leopard prowling across a rocky mountain ridge at dawn.",
    "Time-lapse of a bustling city intersection at night with streaks of car headlights.",
    "A chef in a busy kitchen flambeing a pan, flames rising, steam everywhere.",
    "Underwater footage of a coral reef teeming with colorful tropical fish.",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--model-path",
        default=os.environ.get("WAN22_MODEL_PATH", "Wan-AI/Wan2.2-T2V-A14B-Diffusers"),
        help="HF repo or local path of the (BF16) Wan2.2-T2V-A14B pipeline.",
    )
    p.add_argument(
        "--modelopt-example",
        default=os.environ.get("MODELOPT_WAN22_EXAMPLE"),
        help="Path to Model Optimizer's examples/diffusers/sparsity/wan22_skip_softmax.py "
        "(or set MODELOPT_WAN22_EXAMPLE).",
    )
    p.add_argument("--width", type=int, default=int(os.environ.get("WAN22_CALIB_W", "1280")))
    p.add_argument("--height", type=int, default=int(os.environ.get("WAN22_CALIB_H", "720")))
    p.add_argument("--num-frames", type=int, default=int(os.environ.get("WAN22_CALIB_F", "81")))
    p.add_argument(
        "--calib-steps",
        type=int,
        default=int(os.environ.get("WAN22_CALIB_STEPS", "8")),
        help="Denoising steps per calibration prompt.",
    )
    p.add_argument(
        "--calib-size",
        type=int,
        default=int(os.environ.get("WAN22_CALIB_SIZE", "2")),
        help="Number of prompts used to drive the forward loop.",
    )
    p.add_argument(
        "--target-sparsity",
        type=float,
        default=float(os.environ.get("WAN22_CALIB_TARGET", "0.5")),
    )
    p.add_argument(
        "--out",
        default=os.environ.get("WAN22_CALIB_OUT", "./wan22_calib_14b.json"),
        help="Output calibration JSON path.",
    )
    return p.parse_args()


def load_example(example_path: str):
    if not example_path:
        sys.exit(
            "ERROR: pass --modelopt-example (or set MODELOPT_WAN22_EXAMPLE) to Model "
            "Optimizer's examples/diffusers/sparsity/wan22_skip_softmax.py"
        )
    if not os.path.isfile(example_path):
        sys.exit(f"ERROR: modelopt example not found: {example_path}")
    spec = importlib.util.spec_from_file_location("wan22_example", example_path)
    ex = importlib.util.module_from_spec(spec)
    sys.modules["wan22_example"] = ex
    spec.loader.exec_module(ex)
    # Use inline prompts instead of the example's OpenVid download.
    ex.load_calib_prompts = lambda calib_size: INLINE_PROMPTS[:calib_size]
    return ex


def extract_ab(transformer):
    """Pull the single calibrated (a, b) for prefill from any enabled module."""
    from modelopt.torch.sparsity.attention_sparsity.sparse_attention import (
        SparseAttentionModule,
    )

    for _name, mod in transformer.named_modules():
        if isinstance(mod, SparseAttentionModule) and mod.is_enabled:
            cp = getattr(mod._sparse_method_instance, "calibration_params", None)
            if isinstance(cp, dict):
                pf = cp.get("prefill")
                if isinstance(pf, dict) and "a" in pf and "b" in pf:
                    return {
                        "a": float(pf["a"]),
                        "b": float(pf["b"]),
                        "min_observed_sparsity": float(pf.get("min_observed_sparsity", 0.0)),
                        "max_observed_sparsity": float(pf.get("max_observed_sparsity", 0.0)),
                    }
    return None


def main() -> None:
    args = parse_args()

    import modelopt.torch.sparsity.attention_sparsity as mtsa
    from diffusers import AutoencoderKLWan, WanPipeline

    ex = load_example(args.modelopt_example)

    print(f"=== loading {args.model_path} (bf16) ===", flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.model_path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(args.model_path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")

    transformers = [("transformer", pipe.transformer)]
    if getattr(pipe, "transformer_2", None) is not None:
        transformers.append(("transformer_2", pipe.transformer_2))
    is_14b = len(transformers) > 1
    print(f"transformers: {[n for n, _ in transformers]} is_14b={is_14b}", flush=True)

    sparse_args = argparse.Namespace(
        skip_softmax_threshold=None,
        skip_first_last=2,
        triton_baseline=False,
        calibrate=True,
        target_sparsity=args.target_sparsity,
        # Export-only metadata for modelopt >=0.46 example API (0 = no-op).
        initial_disabled_steps=0,
    )

    print(
        f"=== forward loop ({args.calib_size} prompts x {args.calib_steps} steps "
        f"@ {args.width}x{args.height}x{args.num_frames}) ===",
        flush=True,
    )
    forward_loop = ex.build_calibration_forward_loop(
        pipe,
        calib_size=args.calib_size,
        num_steps=args.calib_steps,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        seed=42,
        guidance_scale=4.0,
        guidance_scale_2=3.0 if is_14b else None,
        negative_prompt="",
    )

    per_component = {}
    for name, transformer in transformers:
        nb = ex._get_num_blocks(transformer)
        print(f"\n=== calibrating {name} ({nb} blocks), target_sparsity={args.target_sparsity} ===", flush=True)
        config = ex.build_sparse_config(sparse_args, num_blocks=nb)
        mtsa.sparsify(transformer, config, forward_loop=forward_loop)
        ab = extract_ab(transformer)
        if ab is None:
            print(f"WARN: {name} produced no calibration (a,b)", flush=True)
        else:
            ab["n_blocks"] = nb
            per_component[name] = ab
            print(
                f"{name}: a={ab['a']:.6g} b={ab['b']:.6g} "
                f"observed_sparsity=[{ab['min_observed_sparsity']:.3f},{ab['max_observed_sparsity']:.3f}]",
                flush=True,
            )

    if not per_component:
        sys.exit("FATAL: no calibration extracted")

    # If transformer_2 produced no data, reuse transformer's (a, b).
    if "transformer" in per_component and "transformer_2" not in per_component:
        per_component["transformer_2"] = dict(per_component["transformer"])
        per_component["transformer_2"]["_reused_from"] = "transformer"

    a_all = statistics.mean(v["a"] for v in per_component.values())
    b_all = statistics.mean(v["b"] for v in per_component.values())

    out = {
        "_meta": {
            "source": "Wan2.2-T2V-A14B BLASST skip-softmax calibration",
            "model_path": args.model_path,
            "resolution": f"{args.width}x{args.height}x{args.num_frames}",
            "calib_size": args.calib_size,
            "calib_steps": args.calib_steps,
            "target_sparsity": args.target_sparsity,
        },
        "per_component": per_component,
        "transformer": {"a": per_component["transformer"]["a"], "b": per_component["transformer"]["b"]},
        "transformer_2": {"a": per_component["transformer_2"]["a"], "b": per_component["transformer_2"]["b"]},
        "a": a_all,
        "b": b_all,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== wrote {args.out} ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
    print(f"\nFLAT (a,b) = ({a_all:.6g}, {b_all:.6g})", flush=True)


if __name__ == "__main__":
    main()
