"""Freeze a small Qwen3-MoE checkpoint using an existing dense SID fixture.

No pretrained download is needed. Existing fixture contents are verified, never
silently regenerated. Run with the pinned SGLang Python environment.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

SEED = 20261014


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path):
    import torch
    from safetensors.torch import load_file, save_file

    original = json.loads((source / "manifest.json").read_text())
    for name, digest in original["files"].items():
        if sha(source / name) != digest:
            raise ValueError(f"Dense input fixture changed: {name}")
    if (output / "manifest.json").exists():
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["dense_source_files"] != original["files"]:
            raise ValueError("MoE fixture came from a different dense fixture")
        for name, digest in manifest["files"].items():
            if sha(output / name) != digest:
                raise ValueError(f"Frozen MoE fixture changed: {name}")
        return manifest
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty, unverified fixture")
    config = json.loads((source / "config.json").read_text())
    config.update(
        architectures=["Qwen3MoeForCausalLM"],
        model_type="qwen3_moe",
        _name_or_path="dummy-qwen3-moe-v1",
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=1024,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )
    tensors = {
        k: v
        for k, v in load_file(str(source / "model.safetensors")).items()
        if ".mlp." not in k
    }
    rng = torch.Generator().manual_seed(SEED)
    h, m = config["hidden_size"], config["moe_intermediate_size"]
    for layer in range(config["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.mlp."
        shapes = {"gate.weight": (8, h)}
        for e in range(8):
            shapes.update(
                {
                    f"experts.{e}.gate_proj.weight": (m, h),
                    f"experts.{e}.up_proj.weight": (m, h),
                    f"experts.{e}.down_proj.weight": (h, m),
                }
            )
        for name, shape in shapes.items():
            tensors[prefix + name] = (
                (torch.randn(shape, generator=rng) * 0.02)
                .to(torch.bfloat16)
                .contiguous()
            )
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(
        tensors,
        str(output / "model.safetensors"),
        metadata={"format": "pt", "seed": str(SEED), "purpose": "fixed-dummy-moe"},
    )
    files = ["config.json", "model.safetensors", "workloads.json"]
    files += sorted(p.name for p in source.glob("catalog-*.jsonl"))
    for name in files[2:]:
        shutil.copyfile(source / name, output / name)
    manifest = {
        "schema": 1,
        "seed": SEED,
        "dtype": "bfloat16",
        "parameters": sum(v.numel() for v in tensors.values()),
        "dense_source_files": original["files"],
        "files": {n: sha(output / n) for n in files},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dense-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.dense_fixture, a.output), indent=2))
