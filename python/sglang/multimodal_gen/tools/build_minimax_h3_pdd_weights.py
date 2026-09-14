#!/usr/bin/env python3
"""Bake alibaba-pai's PDD LoRA into the official MiniMax-H3 weights.

    python3 -m sglang.multimodal_gen.tools.build_minimax_h3_pdd_weights \
        <base transformer dir> <lora.safetensors> <out dir>

Produces two things:

* a complete transformer checkpoint with the 2D LoRA deltas merged in (sglang and
  VideoX-Fun both load it as an ordinary checkpoint);
* `pdd_heads.safetensors`: the 32 position-level output heads, plus a
  `pdd_config.json` recording num_steps / block_size. Those cannot be merged --
  they are what PDD *is*. See below.

## What PDD is, and why the output heads cannot be merged

Parallel Decoding Distillation (https://research.nvidia.com/labs/genair/pdd/)
splits the time axis into N intervals grouped into blocks of L, and "the parallel
decoder utilizes the same backbone, but with the final linear layer replicated N
times". At inference one backbone evaluation advances a whole block, i.e. N/L
forwards -- here 32/4 = 8, which is where "8Step" comes from.

So the checkpoint's `proj_out.weight` is (32, 96, 5376): 32 copies of the final
linear layer, one position dimension more than the base model's (96, 5376).
Numerically each is a small perturbation of that base layer (1-5% relative, which
`--verify` prints), exactly what "replicate N times and fine-tune" looks like.

The L heads inside a block share one backbone evaluation. The paper notes that at
generation time one can "fuse the layers into a single linear layer that directly
predicts a step across the full block"; for flow matching that is a weighted sum
over the per-step sigma deltas, because the h in

    x_{n+L} = x_n + sum_j dsigma_{n+j} * W_{n+j} h

is shared and the sum can be moved onto the weights. This script does NOT do that
fusion: the caller applies the 4 heads one at a time through 4 scheduler steps,
which is bit-equivalent to the fused form without us having to re-derive the sigma
grid and the weighting. The arithmetic saved is negligible (a 96x5376 GEMM).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HEAD_DIM = 128


def _lora_delta(down: torch.Tensor, up: torch.Tensor, scale: float) -> torch.Tensor:
    return (up.float() @ down.float()) * scale


def _interleave_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[3 x (heads*head_dim), in] -> the official per-head interleaved layout."""
    heads = q.shape[0] // HEAD_DIM
    in_dim = q.shape[1]
    grouped = torch.empty(heads, 3 * HEAD_DIM, in_dim, dtype=q.dtype)
    grouped[:, 0 * HEAD_DIM : 1 * HEAD_DIM] = q.view(heads, HEAD_DIM, in_dim)
    grouped[:, 1 * HEAD_DIM : 2 * HEAD_DIM] = k.view(heads, HEAD_DIM, in_dim)
    grouped[:, 2 * HEAD_DIM : 3 * HEAD_DIM] = v.view(heads, HEAD_DIM, in_dim)
    return grouped.reshape(heads * 3 * HEAD_DIM, in_dim)


def _target_of(lora_key: str) -> str | None:
    """LoRA module name -> official weight name. None means "not handled here"."""
    k = lora_key
    if k.startswith("token_refiner.refiner_blocks."):
        k = "token_refiner.blocks." + k[len("token_refiner.refiner_blocks.") :]
    elif k.startswith("transformer_blocks."):
        k = "blocks." + k[len("transformer_blocks.") :]
    else:
        return None
    for src, dst in (
        (".attn.to_out.0", ".attn.out_proj"),
        (".ff.net.0.proj", ".mlp.fc1"),
        (".ff.net.2", ".mlp.fc2"),
        (".adaln_proj.linear", ".adaln_proj.linear"),
    ):
        if k.endswith(src):
            return k[: -len(src)] + dst
    return k  # attn.to_q/k/v pass through; the caller interleaves them


def load_lora(path: str) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    with safe_open(path, "pt") as f:
        meta = f.metadata() or {}
        return {k: f.get_tensor(k) for k in f.keys()}, meta


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir")
    ap.add_argument("lora_path")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="print the relative magnitude of each class of change",
    )
    args = ap.parse_args()

    lora, meta = load_lora(args.lora_path)
    rank = int(meta.get("lora_rank", 64))
    alpha = float(meta.get("lora_alpha", rank))
    scale = alpha / rank
    pdd = {
        "num_steps": int(meta.get("pdd_num_steps", 32)),
        "block_size": int(meta.get("pdd_block_size", 4)),
    }
    print(f"LoRA: rank={rank} alpha={alpha} scale={scale} pdd={pdd}")

    # Group lora_down/lora_up per module; q/k/v are held back and interleaved at the end.
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in lora.items():
        for suffix in (".lora_down", ".lora_up"):
            if key.endswith(suffix):
                pairs.setdefault(key[: -len(suffix)], {})[suffix[1:]] = value
                break

    deltas: dict[str, torch.Tensor] = {}  # official weight name -> delta W
    qkv: dict[str, dict[str, torch.Tensor]] = {}  # module prefix -> {q,k,v: delta W}
    for module, dv in pairs.items():
        if "lora_down" not in dv or "lora_up" not in dv:
            raise SystemExit(f"{module} has only one half of the LoRA pair")
        delta = _lora_delta(dv["lora_down"], dv["lora_up"], scale)
        target = _target_of(module)
        if target is None:
            raise SystemExit(f"no mapping rule for {module}")
        for role in ("to_q", "to_k", "to_v"):
            if target.endswith(f".attn.{role}"):
                qkv.setdefault(target[: -len(f".attn.{role}")], {})[role[-1]] = delta
                break
        else:
            deltas[target + ".weight"] = delta

    for prefix, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise SystemExit(f"{prefix} is missing part of q/k/v: {sorted(parts)}")
        deltas[prefix + ".attn.qkv_proj.weight"] = _interleave_qkv(
            parts["q"], parts["k"], parts["v"]
        )
    print(f"2D deltas to merge: {len(deltas)} ({len(qkv)} of them qkv)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    applied, shards = 0, []
    for shard in sorted(glob.glob(os.path.join(args.base_dir, "*.safetensors"))):
        with safe_open(shard, "pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        for key in list(tensors):
            if key in deltas:
                base = tensors[key]
                merged = base.float() + deltas[key].to(base.device)
                if args.verify:
                    rel = ((merged - base.float()).norm() / base.float().norm()).item()
                    print(f"  {key}: relative change {rel:.4f}")
                tensors[key] = merged.to(base.dtype)
                applied += 1
        name = os.path.basename(shard)
        save_file(tensors, str(out / name), metadata={"format": "pt"})
        shards.append(name)
    print(f"merged {applied}/{len(deltas)} tensors")
    if applied != len(deltas):
        missing = sorted(set(deltas) - set())
        raise SystemExit(
            f"{len(deltas) - applied} deltas found no target; do not use this output"
        )

    for extra in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
        src = os.path.join(args.base_dir, extra)
        if os.path.exists(src):
            shutil.copy(src, out / extra)

    heads = {
        k: lora[k]
        for k in (
            "proj_out.weight",
            "proj_out.bias",
            "audio_proj_out.weight",
            "audio_proj_out.bias",
        )
    }
    save_file(heads, str(out / "pdd_heads.safetensors"), metadata={"format": "pt"})
    (out / "pdd_config.json").write_text(json.dumps(pdd, indent=2) + "\n")
    print(
        f"position-level output heads -> {out / 'pdd_heads.safetensors'} "
        f"({pdd['num_steps']} of them, block size {pdd['block_size']}, "
        f"i.e. {pdd['num_steps'] // pdd['block_size']} forwards)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
