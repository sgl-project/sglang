#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Merge alibaba-pai's PDD LoRA into a native MiniMax-H3 checkpoint.

    python3 -m sglang.multimodal_gen.tools.build_minimax_h3_pdd_weights \
        <base transformer dir> <lora.safetensors> <out dir>

Writes merged weights to `transformer/` and interval-specific output heads to
`pdd_heads.safetensors`, alongside `pdd_config.json`. The heads must stay outside
`transformer/` so the model loader does not treat them as checkpoint shards.
Run `fuse_minimax_h3_pdd_heads.py` on the output directory before serving.

Serve with `--transformer-weights-path <out dir>/transformer` and set
`SGLANG_DIFFUSION_MINIMAX_H3_PDD_HEADS=<out dir>/pdd_fused_heads.safetensors`.
Keep `pdd_config.json` beside the fused heads for schedule validation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

HEAD_DIM = 128


def _interleave_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """[3 x (heads*head_dim), in] -> the official per-head interleaved layout."""
    in_dim = q.shape[1]
    return torch.stack(
        [x.reshape(-1, HEAD_DIM, in_dim) for x in (q, k, v)], dim=1
    ).reshape(-1, in_dim)


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
    ):
        if k.endswith(src):
            return k[: -len(src)] + dst
    return k  # attn.to_q/k/v pass through; the caller interleaves them


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("base_dir", type=Path)
    ap.add_argument("lora_path")
    ap.add_argument("out_dir", type=Path)
    ap.add_argument(
        "--verify",
        action="store_true",
        help="print the relative magnitude of each merged weight change",
    )
    args = ap.parse_args(argv)

    with safe_open(args.lora_path, "pt") as f:
        meta = f.metadata() or {}
        lora = {k: f.get_tensor(k) for k in f.keys()}
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
        delta = (dv["lora_up"].float() @ dv["lora_down"].float()) * scale
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

    out = args.out_dir
    transformer_out = out / "transformer"
    transformer_out.mkdir(parents=True, exist_ok=True)
    applied = 0
    for shard in sorted(args.base_dir.glob("*.safetensors")):
        tensors = load_file(str(shard))
        for key, base in tensors.items():
            if key in deltas:
                merged = base.float() + deltas[key]
                if args.verify:
                    rel = ((merged - base.float()).norm() / base.float().norm()).item()
                    print(f"  {key}: relative change {rel:.4f}")
                tensors[key] = merged.to(base.dtype)
                applied += 1
        save_file(tensors, str(transformer_out / shard.name), metadata={"format": "pt"})
    print(f"merged {applied}/{len(deltas)} tensors")
    if applied != len(deltas):
        raise SystemExit(
            f"{len(deltas) - applied} deltas found no target; do not use this output"
        )

    for extra in (
        "config.json",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        src = args.base_dir / extra
        if src.exists():
            shutil.copy(src, transformer_out / extra)

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
