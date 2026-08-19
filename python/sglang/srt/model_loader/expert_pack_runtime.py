# SPDX-License-Identifier: Apache-2.0
"""Internal preparation of source assets for the expert-pack load format."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

METADATA_FORMAT_VERSION = 3
GGUF_SHARD_SUFFIX_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")


def cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def artifact_dir_for_source(gguf: Path) -> Path:
    stat = gguf.stat()
    fingerprint = hashlib.sha256(
        f"{gguf.parent.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:"
        f"{METADATA_FORMAT_VERSION}".encode()
    ).hexdigest()[:20]
    return cache_root() / "sglang-expert-pack" / "kimi-k3" / fingerprint


def _tokenizer_candidate(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and any(
            (path / name).is_file()
            for name in ("tokenizer.json", "tiktoken.model", "tokenizer_config.json")
        )
    )


def resolve_kimi_tokenizer(gguf: Path, explicit: str | None = None) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not _tokenizer_candidate(candidate):
            raise ValueError(f"Kimi tokenizer directory is invalid: {candidate}")
        return candidate

    candidates = [gguf.parent / "tokenizer", gguf.parent.parent / "kimi-k3-tokenizer"]
    candidates.extend(
        sorted(path for path in gguf.parent.parent.glob("*tokenizer*") if path.is_dir())
    )
    tokenizers = []
    for path in candidates:
        path = path.resolve()
        if path not in tokenizers and _tokenizer_candidate(path):
            tokenizers.append(path)
    if len(tokenizers) != 1:
        names = ", ".join(str(path) for path in tokenizers) or "none"
        raise RuntimeError(
            f"could not uniquely derive Kimi tokenizer beside {gguf.parent}; "
            f"candidates: {names}"
        )
    return tokenizers[0]


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_kimi_model_metadata(tokenizer_dir: Path, artifact_dir: Path) -> Path:
    tokenizer_dir = tokenizer_dir.resolve(strict=True)
    source_config = json.loads(
        (tokenizer_dir / "config.json").read_text(encoding="utf-8")
    )
    if "text_config" not in source_config:
        raise ValueError("Kimi tokenizer config does not contain text_config")
    config = dict(source_config["text_config"])
    config["architectures"] = ["KimiK3LinearForCausalLM"]
    config["model_type"] = "kimi_linear"
    config.pop("auto_map", None)
    config.pop("quantization_config", None)

    output_dir = artifact_dir / "model-meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source in tokenizer_dir.iterdir():
        if source.is_file() and source.name != "config.json":
            destination = output_dir / source.name
            if (
                not destination.is_file()
                or destination.stat().st_size != source.stat().st_size
                or destination.stat().st_mtime_ns != source.stat().st_mtime_ns
            ):
                shutil.copy2(source, destination)
    _write_json_atomic(output_dir / "config.json", config)
    return output_dir


def _expert_pack_path(gguf: Path) -> Path:
    match = GGUF_SHARD_SUFFIX_RE.search(gguf.name)
    if match is None:
        raise ValueError(f"Kimi GGUF is not a numbered shard: {gguf}")
    return gguf.parent / f"{gguf.name[: match.start()]}.expert-major.pack"


def _repo_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "tools" / "expert_pack" / "prepare_kimi_pack.py").is_file():
            return candidate
    raise RuntimeError(
        "expert_pack cannot auto-build Kimi artifacts from an installed package; "
        "run from an SGLang source checkout"
    )


def ensure_kimi_assets(
    gguf: Path,
    *,
    tokenizer_dir: str | None = None,
) -> dict[str, Path]:
    """Build or reuse Kimi artifacts and return the internal serving paths."""
    gguf = gguf.expanduser().resolve(strict=True)
    if not gguf.is_file() or gguf.suffix != ".gguf":
        raise ValueError(f"expert_pack expects a local Kimi GGUF shard, got {gguf}")
    if "KIMI-K3" not in gguf.name.upper():
        raise ValueError(
            "raw GGUF auto-preparation currently supports only Kimi-K3; "
            "provide the model metadata and loader artifacts for other models"
        )
    gguf_dir = gguf.parent
    artifact_dir = artifact_dir_for_source(gguf).resolve()
    pack = _expert_pack_path(gguf).resolve()
    manifest = artifact_dir / "kimi-k3-expert-pack.manifest.json"
    tokenizer = resolve_kimi_tokenizer(gguf, tokenizer_dir)
    lock_path = pack.with_name(pack.name + ".startup.lock")
    repo = _repo_root()
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        model_dir = prepare_kimi_model_metadata(tokenizer, artifact_dir)
        subprocess.run(
            [
                sys.executable,
                str(repo / "tools" / "expert_pack" / "prepare_kimi_pack.py"),
                "--gguf",
                str(gguf),
                "--model-config",
                str(model_dir / "config.json"),
            ],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(repo / "tools" / "expert_pack" / "prepare_kimi_manifest.py"),
                "--gguf-dir",
                str(gguf_dir),
                "--expert-pack",
                str(pack),
                "--model-config",
                str(model_dir / "config.json"),
                "--tokenizer-dir",
                str(tokenizer),
                "--output",
                str(manifest),
                "--payload-samples",
                "6",
            ],
            cwd=repo,
            check=True,
        )
    return {
        "gguf": gguf,
        "gguf_dir": gguf_dir,
        "tokenizer_dir": tokenizer,
        "model_dir": model_dir,
        "pack_path": pack,
        "manifest_path": manifest,
        "stats_path": artifact_dir / "kimi-k3-expert-pack.stats.json",
        "artifact_dir": artifact_dir,
    }


def prepare_raw_kimi_server_args(
    server_args: Any, loader_config: dict[str, Any]
) -> None:
    """Resolve a raw GGUF model path into the normal loader inputs."""
    model_path = Path(server_args.model_path).expanduser()
    if not model_path.is_file() or model_path.suffix.lower() != ".gguf":
        return
    tokenizer_path = server_args.tokenizer_path
    if tokenizer_path and Path(tokenizer_path).expanduser() == model_path:
        tokenizer_path = None
    assets = ensure_kimi_assets(
        model_path,
        tokenizer_dir=tokenizer_path,
    )
    server_args.model_path = str(assets["model_dir"])
    server_args.tokenizer_path = str(assets["model_dir"])
    for key in ("pack_path", "manifest_path", "stats_path"):
        loader_config.setdefault(key, str(assets[key]))
    loader_config.setdefault("source_path", str(assets["gguf"]))
