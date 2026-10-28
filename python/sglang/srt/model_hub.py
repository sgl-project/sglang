# SPDX-License-Identifier: Apache-2.0
"""Lightweight Hub reference parsing and startup path resolution."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock
from huggingface_hub.utils import HFValidationError, validate_repo_id

logger = logging.getLogger(__name__)

_METADATA_PATTERNS = [
    "*.json",
    "*.py",
    "*.model*",
    "*.txt",
    "*.tiktoken",
    "*.jinja",
    "*.bpe",
    "*.vocab",
    "*.spm",
]
_WEIGHT_PATTERNS = {
    "auto": ["*.safetensors", "*.bin", "*.pt"],
    "safetensors": ["*.safetensors"],
    "fastsafetensors": ["*.safetensors"],
    "instanttensor": ["*.safetensors"],
    "mistral": ["consolidated*.safetensors"],
    "pt": ["*.pt"],
    "npcache": ["*.bin"],
    "dummy": [],
}


def parse_model_hub_reference(path: str | None) -> tuple[str, str] | None:
    """Parse hf://owner/repo or modelscope://owner/repo, leaving other paths alone."""
    if not isinstance(path, str) or not path:
        return None
    source, separator, repo_id = path.partition("://")
    if not separator or source not in ("hf", "modelscope"):
        return None
    try:
        if repo_id.count("/") != 1:
            raise HFValidationError("Expected owner/repo")
        validate_repo_id(repo_id)
    except HFValidationError as exc:
        raise ValueError(
            f"Expected {source}://owner/repo without a file path, query, or "
            "embedded revision; use the existing revision arguments."
        ) from exc
    return source, repo_id


def _localize_hub_code(snapshot: Path, repo_id: str, download, cache_dir) -> str:
    """Keep cross-repository auto_map code on the explicitly selected Hub."""
    code_paths = {}

    def local_reference(value):
        if isinstance(value, list):
            return [local_reference(item) for item in value]
        if not isinstance(value, str) or "--" not in value:
            return value
        code_repo, class_name = value.split("--", 1)
        validate_repo_id(code_repo)
        if code_repo not in code_paths:
            code_paths[code_repo] = (
                str(snapshot)
                if code_repo == repo_id
                else download(code_repo, revision=None, allow_patterns=["*.py"])
            )
        namespace = "hub_code_" + hashlib.sha256(code_repo.encode()).hexdigest()[:16]
        return f"{namespace}/{class_name}"

    def localize(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "auto_map" and isinstance(item, dict):
                    value[key] = {
                        name: local_reference(ref) for name, ref in item.items()
                    }
                else:
                    localize(item)
        elif isinstance(value, list):
            for item in value:
                localize(item)

    configs = {}
    for path in snapshot.glob("*config*.json"):
        config = json.loads(path.read_text())
        original = json.dumps(config)
        localize(config)
        if json.dumps(config) != original:
            configs[path.name] = config
    if not configs:
        return str(snapshot)

    # Do not rewrite shared Hub snapshots; publish a local view atomically.
    files = sorted(snapshot.iterdir())
    identity = json.dumps(
        [str(snapshot), [p.name for p in files], configs, code_paths], sort_keys=True
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()
    from huggingface_hub.constants import HF_HUB_CACHE

    root = Path(cache_dir or Path(HF_HUB_CACHE).parent) / "sglang_hub_sources"
    root.mkdir(parents=True, exist_ok=True)
    overlay = root / digest / repo_id.split("/")[-1]
    with FileLock(str(root / f"{digest}.lock")):
        if not overlay.exists():
            overlay.parent.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(dir=overlay.parent) as temporary:
                staging = Path(temporary) / overlay.name
                staging.mkdir()
                for path in files:
                    if path.name not in configs:
                        (staging / path.name).symlink_to(path.resolve())
                for name, config in configs.items():
                    (staging / name).write_text(json.dumps(config))
                for repo, code_path in code_paths.items():
                    namespace = (
                        "hub_code_" + hashlib.sha256(repo.encode()).hexdigest()[:16]
                    )
                    (staging / namespace).symlink_to(Path(code_path).resolve())
                staging.rename(overlay)
    return str(overlay)


def _download_model(reference, revision, cfg, load_format):
    from functools import partial

    from huggingface_hub.constants import HF_HUB_OFFLINE

    source, repo_id = reference
    if source == "hf":
        from huggingface_hub import snapshot_download
    else:
        from modelscope.hub.snapshot_download import snapshot_download

    download = partial(
        snapshot_download,
        revision=revision,
        cache_dir=cfg.download_dir,
        local_files_only=HF_HUB_OFFLINE,
        ignore_patterns=["original/**"],
    )
    try:
        snapshot = Path(download(repo_id, allow_patterns=_METADATA_PATTERNS))
        for pattern in _WEIGHT_PATTERNS[load_format]:
            patterns = [pattern]
            index_name = (
                "consolidated.safetensors.index.json"
                if load_format == "mistral"
                else "model.safetensors.index.json"
            )
            index = snapshot / index_name
            # Auto-detection of native Mistral checkpoints needs consolidated weights.
            if (
                pattern.endswith(".safetensors")
                and index.is_file()
                and (load_format != "auto" or not (snapshot / "params.json").exists())
            ):
                weights = json.loads(index.read_text()).get("weight_map", {})
                if weights:
                    # Preserve bundled drafts and multimodal submodels as well.
                    patterns = sorted(set(weights.values())) + ["**/*.safetensors"]
            download(repo_id, allow_patterns=patterns)
            if any(
                p.relative_to(snapshot).parts[0] != "original"
                for p in snapshot.rglob(pattern)
            ):
                break
        if cfg.trust_remote_code:
            return _localize_hub_code(snapshot, repo_id, download, cfg.download_dir)
        return str(snapshot)
    except Exception:
        logger.error("Failed to resolve %s://%s", source, repo_id)
        raise


def resolve_model_hub_paths(cfg: Any) -> dict[str, str]:
    """Validate explicit sources, then resolve models before tokenizer-only paths."""
    references = {
        field: parse_model_hub_reference(getattr(cfg, field))
        for field in ("model_path", "speculative_draft_model_path", "tokenizer_path")
    }
    model_reference = references["model_path"]
    if model_reference is not None and cfg.tokenizer_path is None:
        references["tokenizer_path"] = model_reference
    formats = {}
    for field, reference in references.items():
        if reference is None:
            continue
        load_format = (
            cfg.speculative_draft_load_format
            if field == "speculative_draft_model_path"
            else None
        ) or cfg.load_format
        formats[field] = "dummy" if field == "tokenizer_path" else load_format
        if formats[field] not in _WEIGHT_PATTERNS:
            raise ValueError(
                f"Explicit Hub sources do not support load_format={load_format!r}; "
                "download a local snapshot for this loader."
            )
    resolved = {}
    if model_reference is not None and cfg.served_model_name is None:
        resolved["served_model_name"] = model_reference[1]
    snapshots = {}
    for field, reference in references.items():
        if reference is None:
            continue
        draft = field == "speculative_draft_model_path"
        revision = cfg.speculative_draft_model_revision if draft else cfg.revision
        key = (reference, revision, formats[field])
        if field == "tokenizer_path":
            key = next((k for k in snapshots if k[:2] == key[:2]), key)
        if key not in snapshots:
            snapshots[key] = _download_model(reference, revision, cfg, formats[field])
        resolved[field] = snapshots[key]
    return resolved
