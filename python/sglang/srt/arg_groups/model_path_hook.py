# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the model source paths."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from sglang.srt.arg_groups.overrides import (
    declare_resolution,
    resolving_view,
)
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

logger = logging.getLogger(__name__)


def handle_model_source_paths(server_args: Any):
    """Prepare metadata for model paths backed by remote object stores."""
    cfg = resolving_view(server_args)
    server_args._resolve_hf_gguf_model_path()

    seen_paths = set()
    for model_path in (
        cfg.model_path,
        cfg.tokenizer_path,
        cfg.speculative_draft_model_path,
    ):
        if (
            model_path is not None
            and model_path not in seen_paths
            and is_runai_obj_uri(model_path)
        ):
            ObjectStorageModel.download_and_get_path(model_path)
            seen_paths.add(model_path)


def resolve_hf_gguf_model_path(server_args: Any):
    """Turn a Hub reference to a .gguf into a local file path."""
    cfg = resolving_view(server_args)
    from sglang.srt.utils.hf_transformers_utils import resolve_hf_gguf_reference

    resolved = resolve_hf_gguf_reference(cfg.model_path, revision=cfg.revision)
    if resolved is not None:
        logger.info("Resolved GGUF %s -> %s", cfg.model_path, resolved)
        if cfg.tokenizer_path == cfg.model_path:
            declare_resolution(
                server_args,
                "_resolve_hf_gguf_model_path",
                tokenizer_path=resolved,
            )
        declare_resolution(
            server_args,
            "_resolve_hf_gguf_model_path",
            model_path=resolved,
        )

    # A speculative draft can be a .gguf too, and it is loaded by path, so it
    # needs the same Hub-reference resolution as the target.
    if cfg.speculative_draft_model_path:
        resolved_draft = resolve_hf_gguf_reference(
            cfg.speculative_draft_model_path,
            revision=cfg.speculative_draft_model_revision,
        )
        if resolved_draft is not None:
            logger.info(
                "Resolved draft GGUF %s -> %s",
                cfg.speculative_draft_model_path,
                resolved_draft,
            )
            declare_resolution(
                server_args,
                "_resolve_hf_gguf_model_path",
                speculative_draft_model_path=resolved_draft,
            )


def handle_modelscope_paths(server_args: Any):
    """Resolve model / tokenizer / speculative-draft paths from the local
    ModelScope cache when possible, falling back to snapshot_download
    for any path that is not already present on disk.

    Note: speculative_token_map is intentionally NOT handled here
    because its value uses repo_id/filename semantics rather than a
    plain repo ID.  That resolution lives in
    :func:`sglang.srt.speculative.spec_utils.load_token_map`.
    """
    cfg = resolving_view(server_args)

    ms_root = None
    ms_snapshot_download = None

    def _resolve_or_download(
        path: Optional[str],
        ignore_patterns: Optional[list] = None,
        revision: Optional[str] = None,
    ) -> Optional[str]:
        nonlocal ms_root, ms_snapshot_download
        if path is None:
            return None
        if not path or os.path.exists(path):
            return path

        if ms_snapshot_download is None:
            from modelscope.hub.snapshot_download import (
                snapshot_download as _ms_snapshot_download,
            )
            from modelscope.utils.file_utils import get_model_cache_root

            ms_snapshot_download = _ms_snapshot_download
            ms_root = get_model_cache_root()

        # Check ModelScope default cache
        cached = os.path.join(ms_root, path)
        if os.path.exists(cached):
            return cached
        # Check user-specified download dir
        if cfg.download_dir:
            alt = os.path.join(cfg.download_dir, path)
            if os.path.exists(alt):
                return alt

        # Cache miss — download from ModelScope hub
        return ms_snapshot_download(
            path,
            cache_dir=cfg.download_dir,
            revision=revision,
            **({"ignore_patterns": ignore_patterns} if ignore_patterns else {}),
        )

    declare_resolution(
        server_args,
        "_handle_modelscope_paths",
        model_path=_resolve_or_download(cfg.model_path, revision=cfg.revision),
    )
    declare_resolution(
        server_args,
        "_handle_modelscope_paths",
        tokenizer_path=_resolve_or_download(
            cfg.tokenizer_path,
            ignore_patterns=["*.bin", "*.safetensors"],
            revision=cfg.revision,
        ),
    )
    if cfg.speculative_draft_model_path:
        declare_resolution(
            server_args,
            "_handle_modelscope_paths",
            speculative_draft_model_path=_resolve_or_download(
                cfg.speculative_draft_model_path,
                revision=cfg.speculative_draft_model_revision or "main",
            ),
        )
