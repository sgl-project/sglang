# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Torch-free Hugging Face Hub and local model-file management helpers."""

import os
import re
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download


def download_from_hf(
    model_path: str,
    allow_patterns: Optional[Union[str, list]] = None,
):
    if os.path.exists(model_path):
        return model_path

    if not allow_patterns:
        allow_patterns = ["*.json", "*.bin", "*.model"]

    return snapshot_download(model_path, allow_patterns=allow_patterns)


def _resolve_local_or_cached_file(model_name_or_path, filename, revision=None):
    """Resolve a file from a local directory or HF hub cache (no network)."""
    local_path = Path(model_name_or_path) / filename
    if local_path.is_file():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model_name_or_path, filename, revision=revision, local_files_only=True
    )


def _cached_file_exists(model_name_or_path, filename, revision=None) -> bool:
    """Whether *filename* is available locally or in the HF cache (no network)."""
    try:
        _resolve_local_or_cached_file(model_name_or_path, filename, revision)
        return True
    except Exception:
        return False


def _remote_file_exists(repo_id, filename, revision=None) -> bool:
    """Whether *filename* exists on the HF hub (HEAD request only, no download).

    Returns False on any error (offline, gated, network, invalid id) so callers
    fall back to their default path instead of crashing.
    """
    from huggingface_hub.constants import HF_HUB_OFFLINE

    if HF_HUB_OFFLINE:
        return False
    try:
        from huggingface_hub import HfApi

        return HfApi().file_exists(repo_id, filename, revision=revision)
    except Exception:
        return False


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"


def _is_remote_url(url: Union[str, Path]) -> bool:
    if isinstance(url, Path):
        return False
    return re.match(r"(.+)://(.*)", url) is not None


def resolve_hf_gguf_reference(
    model: str, revision: Optional[str] = None
) -> Optional[str]:
    """Download a .gguf named by Hub reference and return its local path.

    owner/repo/path/inside/repo.gguf   -> exactly that file
    owner/repo:QUANT_TYPE              -> the only matching quantization
    owner/repo                         -> the only .gguf in the repo
    """
    if not model or os.path.exists(model) or _is_remote_url(model):
        return None

    from huggingface_hub import hf_hub_download

    if ":" in model:
        repo_id, _, quant_type = model.rpartition(":")
        if repo_id.count("/") != 1 or not quant_type:
            return None

        from huggingface_hub import HfApi

        files = [
            sibling.rfilename
            for sibling in HfApi().repo_info(repo_id, revision=revision).siblings
        ]
        suffix = f"-{quant_type}.gguf"
        candidates = [filename for filename in files if filename.endswith(suffix)]
        if not candidates:
            available = sorted(
                filename for filename in files if filename.endswith(".gguf")
            )
            raise ValueError(
                f"No file matching quant type {quant_type!r} in {repo_id}. "
                f"Available GGUF files: {available}"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"Quant type {quant_type!r} is ambiguous in {repo_id}: "
                f"{sorted(candidates)}. Pass the full owner/repo/path/file.gguf "
                "reference instead."
            )
        return hf_hub_download(repo_id, candidates[0], revision=revision)

    parts = model.strip("/").split("/")
    if len(parts) < 2:
        return None

    if len(parts) > 2 and model.endswith(".gguf"):
        repo_id = "/".join(parts[:2])
        filename = "/".join(parts[2:])
        return hf_hub_download(repo_id, filename, revision=revision)

    if len(parts) != 2:
        return None

    from huggingface_hub import HfApi

    try:
        files = [
            s.rfilename for s in HfApi().repo_info(model, revision=revision).siblings
        ]
    except Exception:
        return None
    if any(f == "config.json" for f in files):
        return None

    candidates = [f for f in files if f.endswith(".gguf")]
    if not candidates:
        return None
    if len(candidates) > 1:
        listing = "\n  ".join(f"{model}/{f}" for f in sorted(candidates))
        raise ValueError(
            f"{model} contains {len(candidates)} .gguf files; name the one to "
            f"serve:\n  {listing}"
        )
    return hf_hub_download(model, candidates[0], revision=revision)


def gguf_sidecar_dir(
    gguf_path: Union[str, os.PathLike], sentinel: str
) -> Optional[Path]:
    """Directory containing *sentinel* next to a .gguf file, if there is one."""
    directory = Path(gguf_path).parent
    return directory if (directory / sentinel).is_file() else None
