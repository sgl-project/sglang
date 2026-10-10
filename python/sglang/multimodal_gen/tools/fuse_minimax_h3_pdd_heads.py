#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fuse PDD's 32 position-level output heads into one per block, i.e. 8.

    python3 -m sglang.multimodal_gen.tools.fuse_minimax_h3_pdd_heads <pdd dir> \
        [--video-shift 12.0 --audio-shift 3.0]

## Why the fusion is exact

MiniMax-H3's scheduler has "ancestral" in its name but takes the eta0 branch, which
injects no noise:

    x_next = r*x + (1-r)*x0,  r = sigma_next/sigma_curr,  x0 = x - sigma_curr*v
           = x - (sigma_curr - sigma_next)*v

That is plain Euler, and linear in v. The L heads in one PDD block share a single
backbone evaluation, so

    x_{n+L} = x_n - sum_j dsigma_{n+j} * W_{n+j} h
            = x_n - (sum_j dsigma_{n+j}) * (sum_j w_j W_{n+j}) h,  w_j = dsigma_{n+j} / sum dsigma

and the bracket is the fused head -- the paper's "fuse the layers into a single
linear layer that directly predicts a step across the full block". Each block then
costs one forward and one scheduler step, and the denoise loop needs no changes.

This depends on eta0. Under true ancestral sampling (noise injected every step) the
state at the block's second step is no longer the block's first state, the sum
cannot be moved onto W, and the fusion is invalid -- the L sub-steps have to be run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sigma_grid(num_points: int, shift: float) -> torch.Tensor:
    base = torch.linspace(1.0, 0.0, num_points, dtype=torch.float64)
    return shift * base / (1 + (shift - 1) * base)


def fuse(heads: torch.Tensor, sigmas: torch.Tensor, block: int) -> torch.Tensor:
    """heads: (N, ...) -> (N//block, ...), averaged within a block weighted by dsigma."""
    steps = heads.shape[0]
    if block <= 0 or steps == 0 or steps % block:
        raise ValueError("PDD head count must be positive and divisible by block size")
    if sigmas.numel() != steps + 1:
        raise SystemExit(
            f"sigma grid has {sigmas.numel()} points but there are {steps} heads"
        )
    dsigma = (sigmas[:-1] - sigmas[1:]).clamp(min=0)
    out = []
    for b in range(steps // block):
        w = dsigma[b * block : (b + 1) * block]
        if float(w.sum()) <= 0:
            raise SystemExit(f"block {b} has an all-zero dsigma")
        w = (w / w.sum()).to(heads.dtype)
        chunk = heads[b * block : (b + 1) * block].float()
        out.append((chunk * w.float().view(-1, *([1] * (chunk.dim() - 1)))).sum(0))
    return torch.stack(out).to(heads.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("pdd_dir")
    ap.add_argument("--video-shift", type=float, default=12.0)
    ap.add_argument("--audio-shift", type=float, default=3.0)
    args = ap.parse_args()

    d = Path(args.pdd_dir)
    cfg = json.loads((d / "pdd_config.json").read_text())
    n, block = cfg["num_steps"], cfg["block_size"]
    with safe_open(str(d / "pdd_heads.safetensors"), "pt") as f:
        heads = {k: f.get_tensor(k) for k in f.keys()}

    sv = sigma_grid(n + 1, args.video_shift)
    sa = sigma_grid(n + 1, args.audio_shift)
    fused = {
        "video_out.weight": fuse(heads["proj_out.weight"], sv, block),
        "video_out.bias": fuse(heads["proj_out.bias"], sv, block),
        "audio_out.weight": fuse(heads["audio_proj_out.weight"], sa, block),
        "audio_out.bias": fuse(heads["audio_proj_out.bias"], sa, block),
    }
    save_file(fused, str(d / "pdd_fused_heads.safetensors"), metadata={"format": "pt"})
    cfg["fused_steps"] = n // block
    cfg["num_inference_steps"] = n // block + 1  # H3 counts steps as sigma grid points
    cfg["video_shift"], cfg["audio_shift"] = args.video_shift, args.audio_shift
    (d / "pdd_config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    print(f"fused {n} -> {n // block} heads (block size {block})")
    for k, v in fused.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"run with --num-inference-steps {cfg['num_inference_steps']}")
    print(
        "first 9 points of the video sigma grid:",
        [round(float(x), 4) for x in sv[::block]],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
