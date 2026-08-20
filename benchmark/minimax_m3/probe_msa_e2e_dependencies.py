#!/usr/bin/env python3
"""Read-only, offline preflight for the MiniMax-M3 TP4 MSA gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
import subprocess
from importlib import metadata
from pathlib import Path


REQUIRED_GPQA_COLUMNS = {
    "Question",
    "Correct Answer",
    "Incorrect Answer 1",
    "Incorrect Answer 2",
    "Incorrect Answer 3",
}
REQUIRED_SGLANG_KERNEL_VERSION = "0.4.6.post1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(source: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(source), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_checkpoint(model: str) -> Path:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return local_path.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(repo_id=model, local_files_only=True)
    ).resolve()


def probe_checkpoint(model: str) -> dict:
    checkpoint = resolve_checkpoint(model)
    config = checkpoint / "config.json"
    if not config.is_file():
        raise RuntimeError(f"checkpoint has no readable config.json: {checkpoint}")

    indexes = sorted(checkpoint.glob("*.safetensors.index.json"))
    shards: set[str] = set()
    for index in indexes:
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        shards.update(str(value) for value in weight_map.values())
    if shards:
        missing = sorted(shard for shard in shards if not (checkpoint / shard).is_file())
        if missing:
            raise RuntimeError(
                f"checkpoint is missing {len(missing)} indexed weight shards; "
                f"first missing shard: {missing[0]}"
            )
    else:
        standalone_weights = list(checkpoint.glob("*.safetensors")) + list(
            checkpoint.glob("pytorch_model*.bin")
        )
        if not standalone_weights:
            raise RuntimeError(f"checkpoint contains no locally cached weights: {checkpoint}")
        shards = {path.name for path in standalone_weights}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=True
    )
    return {
        "requested_model": model,
        "resolved_path": str(checkpoint),
        "config_sha256": sha256(config),
        "num_weight_shards": len(shards),
        "tokenizer_class": type(tokenizer).__name__,
    }


def probe_datasets(longbench_subset: Path, gpqa_dataset: Path) -> dict:
    manifest_path = Path(str(longbench_subset) + ".manifest.json")
    if not longbench_subset.is_file() or not manifest_path.is_file():
        raise RuntimeError("LongBench-v2 subset and manifest must both be readable")
    manifest = json.loads(manifest_path.read_text())
    observed_subset_sha = sha256(longbench_subset)
    if manifest.get("subset_sha256") != observed_subset_sha:
        raise RuntimeError("LongBench-v2 subset SHA-256 does not match its manifest")
    if int(manifest.get("num_examples", 0)) != 100:
        raise RuntimeError("LongBench-v2 subset must contain exactly 100 examples")
    if int(manifest.get("minimum_observed_tokens", 0)) < 32768:
        raise RuntimeError("LongBench-v2 subset contains a prompt shorter than 32K")

    with gpqa_dataset.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        columns = set(reader.fieldnames or ())
    missing_columns = REQUIRED_GPQA_COLUMNS - columns
    if missing_columns:
        raise RuntimeError(f"GPQA-Diamond CSV is missing columns: {missing_columns}")
    if len(rows) != 198:
        raise RuntimeError(f"GPQA-Diamond CSV contains {len(rows)} rows, expected 198")
    return {
        "longbench_v2": {
            "path": str(longbench_subset.resolve()),
            "sha256": observed_subset_sha,
            "num_examples": 100,
            "minimum_observed_tokens": manifest["minimum_observed_tokens"],
        },
        "gpqa_diamond": {
            "path": str(gpqa_dataset.resolve()),
            "sha256": sha256(gpqa_dataset),
            "num_examples": len(rows),
        },
    }


def probe_flashinfer(source: Path, expected_head: str) -> dict:
    source = source.resolve()
    head = git(source, "rev-parse", "HEAD")
    if head != expected_head:
        raise RuntimeError(f"FlashInfer HEAD is {head}, expected {expected_head}")
    dirty = git(source, "status", "--short", "--untracked-files=no")
    if dirty:
        raise RuntimeError(
            "FlashInfer tracked source differs from HEAD; evidence needs exact source"
        )
    public_module = source / "flashinfer" / "msa_ops" / "__init__.py"
    if not public_module.is_file():
        raise RuntimeError(f"FlashInfer source API is missing: {public_module}")

    flashinfer = importlib.import_module("flashinfer")
    msa_ops = importlib.import_module("flashinfer.msa_ops")
    installed_path = Path(inspect.getfile(msa_ops)).resolve()
    expected_package = (source / "flashinfer" / "msa_ops").resolve()
    if not installed_path.is_relative_to(expected_package):
        raise RuntimeError(
            f"imported flashinfer.msa_ops from {installed_path}, expected {expected_package}"
        )
    required = {
        "msa_sparse_attention",
        "msa_sparse_decode_attention",
        "msa_topk_select",
        "MSASparseAttentionWorkspace",
    }
    missing = sorted(name for name in required if not callable(getattr(msa_ops, name, None)))
    if missing:
        raise RuntimeError(f"installed flashinfer.msa_ops is missing callables: {missing}")
    return {
        "source_path": str(source),
        "source_head": head,
        "installed_path": str(installed_path),
        "version": getattr(flashinfer, "__version__", None),
        "prefill_signature": str(inspect.signature(msa_ops.msa_sparse_attention)),
        "decode_signature": str(inspect.signature(msa_ops.msa_sparse_decode_attention)),
    }


def probe_fmha_sm100() -> dict:
    module = importlib.import_module("fmha_sm100")
    for name in ("fmha_sm100", "fmha_sm100_plan"):
        if not callable(getattr(module, name, None)):
            raise RuntimeError(f"fmha_sm100 installation has no callable {name}")
    return {"installed_path": str(Path(inspect.getfile(module)).resolve())}


def probe_sglang(repo: Path) -> dict:
    module = importlib.import_module("sglang")
    installed_path = Path(inspect.getfile(module)).resolve()
    expected_package = (repo / "python" / "sglang").resolve()
    if not installed_path.is_relative_to(expected_package):
        raise RuntimeError(
            f"Python imports SGLang from {installed_path}, not this checkout "
            f"({expected_package})"
        )
    kernel_version = metadata.version("sglang-kernel")
    if kernel_version != REQUIRED_SGLANG_KERNEL_VERSION:
        raise RuntimeError(
            f"sglang-kernel is {kernel_version}, expected "
            f"{REQUIRED_SGLANG_KERNEL_VERSION}"
        )
    return {
        "repo": str(repo.resolve()),
        "head": git(repo, "rev-parse", "HEAD"),
        "installed_path": str(installed_path),
        "sglang_kernel_version": kernel_version,
    }


def probe_gpus(expected_gpus: int, expected_capability: tuple[int, int]) -> dict:
    import torch

    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise RuntimeError(f"{count} CUDA devices are visible, expected {expected_gpus}")
    devices = []
    for index in range(count):
        capability = torch.cuda.get_device_capability(index)
        if capability != expected_capability:
            raise RuntimeError(
                f"GPU {index} capability is {capability}, expected {expected_capability}"
            )
        devices.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(capability),
            }
        )
    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,compute_cap,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "devices": devices,
        "nvidia_smi": nvidia_smi,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--longbench-subset", type=Path, required=True)
    parser.add_argument("--gpqa-dataset", type=Path, required=True)
    parser.add_argument("--flashinfer-source-dir", type=Path, required=True)
    parser.add_argument("--expected-flashinfer-head", required=True)
    parser.add_argument("--expected-gpus", type=int, default=4)
    parser.add_argument("--expected-capability", default="10.3")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    capability_parts = tuple(int(value) for value in args.expected_capability.split("."))
    if len(capability_parts) != 2:
        raise SystemExit("--expected-capability must look like 10.3")
    repo = Path(__file__).resolve().parents[2]
    report = {
        "offline": True,
        "sglang": probe_sglang(repo),
        "flashinfer": probe_flashinfer(
            args.flashinfer_source_dir, args.expected_flashinfer_head
        ),
        "fmha_sm100": probe_fmha_sm100(),
        "checkpoint": probe_checkpoint(args.model),
        "datasets": probe_datasets(args.longbench_subset, args.gpqa_dataset),
        "gpu": probe_gpus(args.expected_gpus, capability_parts),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
