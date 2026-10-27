# SPDX-License-Identifier: Apache-2.0
"""Lightweight Hub reference parsing and startup path resolution."""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ID = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
_WEIGHT_PATTERNS = ["*.bin", "*.safetensors", "*.pt", "*.pth", "*.gguf"]


def parse_model_hub_reference(path: str | None) -> tuple[str, str] | None:
    """Parse hf://owner/repo or modelscope://owner/repo, leaving other paths alone."""
    if not isinstance(path, str) or not path:
        return None
    source, separator, repo_id = path.partition("://")
    if not separator or source not in ("hf", "modelscope"):
        return None
    if (
        not _REPO_ID.fullmatch(repo_id)
        or "." in repo_id.split("/")
        or ".." in repo_id
        or "--" in repo_id
        or repo_id.endswith(".git")
    ):
        raise ValueError(
            f"Expected {source}://owner/repo without a file path, query, or "
            "embedded revision; use the existing revision arguments."
        )
    return source, repo_id


def resolve_model_hub_paths(cfg: Any) -> dict[str, str]:
    """Validate explicit sources, then resolve models before tokenizer-only paths."""
    references = {
        field: parse_model_hub_reference(getattr(cfg, field))
        for field in ("model_path", "speculative_draft_model_path", "tokenizer_path")
    }
    model_reference = references["model_path"]
    if model_reference is not None and cfg.tokenizer_path is None:
        references["tokenizer_path"] = model_reference
    resolved = {}
    if model_reference is not None and cfg.served_model_name is None:
        resolved["served_model_name"] = model_reference[1]
    snapshots = {}
    for field, reference in references.items():
        if reference is None:
            continue
        draft = field == "speculative_draft_model_path"
        revision = cfg.speculative_draft_model_revision if draft else cfg.revision
        load_format = (
            cfg.speculative_draft_load_format if draft else None
        ) or cfg.load_format
        metadata_only = field == "tokenizer_path" or load_format == "dummy"
        key = (reference, revision, metadata_only)
        full_key = (reference, revision, False)
        if key not in snapshots and full_key not in snapshots:
            from huggingface_hub.constants import HF_HUB_OFFLINE

            source, repo_id = reference
            kwargs = dict(
                revision=revision,
                cache_dir=cfg.download_dir,
                local_files_only=HF_HUB_OFFLINE,
            )
            if metadata_only:
                kwargs["ignore_patterns"] = _WEIGHT_PATTERNS
            try:
                if source == "hf":
                    from huggingface_hub import snapshot_download
                else:
                    from modelscope.hub.snapshot_download import snapshot_download

                snapshots[key] = snapshot_download(repo_id, **kwargs)
            except Exception:
                logger.error("Failed to resolve %s://%s", source, repo_id)
                raise
        resolved[field] = snapshots.get(full_key) or snapshots[key]
    return resolved
