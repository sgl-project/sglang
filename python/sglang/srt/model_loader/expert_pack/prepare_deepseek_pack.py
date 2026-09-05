#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Validate or build the DeepSeek expert-pack used by the RTX 5090 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

try:
    from .format import read_header
except ImportError:
    from format import read_header  # type: ignore[no-redef]


FORMAT = "SGLANG-EXPERTPACK-v1"
EXPERT_PACK_FILENAME = "DeepSeek-V4-Flash.expert-pack"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def normalize_manifest_identity(manifest_path: Path, manifest: dict) -> None:
    model = manifest["model"]
    normalized_model = {
        "ref": model["ref"],
        "model_identity_sha256": model["model_identity_sha256"],
        "config_sha256": model["config_sha256"],
        "num_layers": model["num_layers"],
        "num_routed_experts": model["num_routed_experts"],
        "top_k": model["top_k"],
        "single_gpu": model["single_gpu"],
    }
    if model != normalized_model:
        manifest["model"] = normalized_model
        write_json_atomic(manifest_path, manifest)


def artifact_paths(source: Path) -> tuple[Path, Path, Path]:
    pack = source.parent / EXPERT_PACK_FILENAME
    manifest = source.parent / f"{EXPERT_PACK_FILENAME}.manifest.json"
    checkpoint = source.parent / f"{EXPERT_PACK_FILENAME}.checkpoint.json"
    return pack, manifest, checkpoint


def validate_pack(
    pack: Path, manifest_path: Path, expected_source: Path | None = None
) -> tuple[bool, str, dict | None]:
    try:
        pack = pack.resolve(strict=True)
        manifest_path = manifest_path.resolve(strict=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT or manifest.get("complete") is not True:
            raise ValueError(
                "manifest is not a completed SGLANG-EXPERTPACK-v1 manifest"
            )
        if Path(manifest["pack_path"]).resolve() != pack:
            raise ValueError(
                "manifest pack_path does not match the fixed expert-pack path"
            )
        if pack.stat().st_size != int(manifest["pack_size"]):
            raise ValueError("pack size does not match manifest")

        source = Path(manifest["source"]["path"]).resolve(strict=True)
        if expected_source is not None and source != expected_source.resolve(
            strict=True
        ):
            raise ValueError("manifest source path does not match --gguf")
        if source.stat().st_size != int(manifest["source"]["size"]):
            raise ValueError("source GGUF size does not match manifest")

        with pack.open("rb", buffering=0) as stream:
            header = read_header(stream)
            stream.seek(header.header_bytes)
            raw_index = stream.read(header.index_count * header.entry_bytes)
        if len(raw_index) != header.index_count * header.entry_bytes:
            raise ValueError("pack index is truncated")
        if hashlib.sha256(raw_index).hexdigest() != manifest["index_sha256"]:
            raise ValueError("pack index SHA-256 does not match manifest")

        model = manifest["model"]
        model_identity_sha256 = model.get(
            "model_identity_sha256", header.model_identity_sha256
        )
        model["model_identity_sha256"] = model_identity_sha256
        expected = (
            (header.index_count, manifest["index_count"], "index count"),
            (header.data_start, manifest["data_start"], "data start"),
            (header.num_layers, model["num_layers"], "layer count"),
            (header.num_experts, model["num_routed_experts"], "expert count"),
            (header.top_k, model["top_k"], "top-k"),
            (
                header.source_blob_sha256,
                manifest["source"]["sha256"],
                "source digest",
            ),
            (
                header.model_identity_sha256,
                model_identity_sha256,
                "model identity digest",
            ),
            (header.config_sha256, model["config_sha256"], "config digest"),
        )
        for actual, wanted, label in expected:
            if actual != wanted:
                raise ValueError(f"pack header {label} does not match manifest")
        return True, "header, index, source path and sizes are valid", manifest
    except Exception as exc:
        return False, str(exc), None


def load_build_inputs(args: argparse.Namespace) -> dict[str, object]:
    source = args.gguf.resolve(strict=True)
    expert_pack, expert_pack_manifest, checkpoint = artifact_paths(source)
    model_config_path = args.model_config.resolve(strict=True)
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    source_sha256 = sha256_file(source)
    config_sha256 = sha256_file(model_config_path)
    model_identity = hashlib.sha256(
        f"sglang-deepseek-expert-pack-v1:{source_sha256}:{config_sha256}".encode(
            "ascii"
        )
    ).hexdigest()
    return {
        "source": source,
        "source_sha256": source_sha256,
        "expert_pack": expert_pack,
        "expert_pack_manifest": expert_pack_manifest,
        "checkpoint": checkpoint,
        "config_blob": model_config_path,
        "config_sha256": config_sha256,
        "model_identity_sha256": model_identity,
        "num_layers": int(model_config["num_hidden_layers"]),
        "num_experts": int(model_config["n_routed_experts"]),
        "top_k": int(model_config["num_experts_per_tok"]),
    }


def remove_invalid_outputs(pack: Path, manifest: Path, checkpoint: Path) -> None:
    for path in (pack, manifest, pack.with_name(pack.name + ".partial"), checkpoint):
        if path.exists():
            print(f"EXPERT_PACK_REMOVE_INVALID path={path}", flush=True)
            path.unlink()


def build_pack(args: argparse.Namespace, inputs: dict[str, object]) -> None:
    build_script = Path(__file__).with_name("build.py")
    expert_pack = Path(inputs["expert_pack"])
    manifest = Path(inputs["expert_pack_manifest"])
    checkpoint = Path(inputs["checkpoint"])
    partial = expert_pack.with_name(expert_pack.name + ".partial")
    resume = partial.is_file() and checkpoint.is_file() and not expert_pack.exists()
    if not resume:
        remove_invalid_outputs(expert_pack, manifest, checkpoint)

    command = [
        sys.executable,
        str(build_script),
        "--source",
        str(inputs["source"]),
        "--source-sha256",
        str(inputs["source_sha256"]),
        "--output",
        str(expert_pack),
        "--manifest",
        str(manifest),
        "--checkpoint",
        str(checkpoint),
        "--model-ref",
        args.model_ref,
        "--model-identity-sha256",
        str(inputs["model_identity_sha256"]),
        "--config-blob",
        str(inputs["config_blob"]),
        "--config-sha256",
        str(inputs["config_sha256"]),
        "--num-layers",
        str(inputs["num_layers"]),
        "--num-experts",
        str(inputs["num_experts"]),
        "--top-k",
        str(inputs["top_k"]),
        "--alignment",
        str(args.alignment),
        "--safety-margin-gib",
        str(args.safety_margin_gib),
    ]
    if args.inventory and args.inventory.is_file():
        command.extend(("--inventory", str(args.inventory.resolve())))
    if resume:
        command.append("--resume")
    print(
        f"EXPERT_PACK_BUILD_START output={expert_pack} resume={str(resume).lower()}",
        flush=True,
    )
    subprocess.run(command, check=True)
    print(f"EXPERT_PACK_BUILD_COMPLETE output={expert_pack}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--model-ref", default="deepseek-v4-flash")
    parser.add_argument("--alignment", type=int, default=4096)
    parser.add_argument("--safety-margin-gib", type=float, default=16.0)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    return args


def main() -> int:
    args = parse_args()
    source = args.gguf.resolve(strict=True)
    expert_pack, expert_pack_manifest, _ = artifact_paths(source)
    valid, reason, manifest = validate_pack(expert_pack, expert_pack_manifest, source)
    if valid and manifest is not None:
        normalize_manifest_identity(expert_pack_manifest, manifest)
        print(f"EXPERT_PACK_VALID path={expert_pack} detail={reason}", flush=True)
        return 0
    print(f"EXPERT_PACK_INVALID path={expert_pack} detail={reason}", flush=True)
    if args.check_only:
        return 1

    existing_valid, existing_reason, existing_manifest = validate_pack(
        expert_pack, expert_pack_manifest
    )
    if existing_valid and existing_manifest is not None:
        existing_source = Path(existing_manifest["source"]["path"])
        raise RuntimeError(
            f"the fixed expert-pack already belongs to a different GGUF: {existing_source}; "
            f"move the requested GGUF to its own directory instead of overwriting {expert_pack}"
        )

    inputs = load_build_inputs(args)
    build_pack(args, inputs)
    valid, reason, manifest = validate_pack(expert_pack, expert_pack_manifest, source)
    if not valid or manifest is None:
        raise RuntimeError(f"generated expert-pack failed validation: {reason}")
    normalize_manifest_identity(expert_pack_manifest, manifest)
    print(f"EXPERT_PACK_READY path={expert_pack} detail={reason}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
