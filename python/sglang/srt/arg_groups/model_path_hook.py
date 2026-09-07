# SPDX-License-Identifier: Apache-2.0
"""Server-argument resolution for the model source paths."""

from __future__ import annotations

import glob
import importlib
import logging
import os
from typing import Any, Optional

from sglang.srt.arg_groups.overrides import (
    _gguf_quantization,
    declare_resolution,
    resolving_view,
    run_post_process_pass,
)
from sglang.srt.utils.common import is_remote_url
from sglang.srt.utils.hf_transformers_utils import check_gguf_file
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

logger = logging.getLogger(__name__)


def handle_model_source_paths(server_args: Any):
    """Prepare metadata for model paths backed by remote object stores."""
    cfg = resolving_view(server_args)
    resolve_hf_gguf_model_path(server_args)

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


def handle_load_format(server_args: Any):
    # The quantization side of the gguf coupling moved to the pipeline
    # (arg_groups/overrides.py: _gguf_quantization); load_format itself is
    # genuine config (runtime user updates write it) and stays imperative.
    cfg = resolving_view(server_args)

    run_post_process_pass(server_args, _gguf_quantization)
    if (cfg.load_format == "auto" or cfg.load_format == "gguf") and check_gguf_file(
        cfg.model_path
    ):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="gguf",
        )

    if cfg.load_format == "auto" and is_mistral_native_format(server_args):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="mistral",
        )
        logger.info(
            "Detected Mistral native format checkpoint, setting load_format='mistral'"
        )

    if is_runai_obj_uri(cfg.model_path):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="runai_streamer",
        )
    elif is_remote_url(cfg.model_path):
        declare_resolution(
            server_args,
            "_handle_load_format",
            load_format="remote",
        )

    if (
        cfg.speculative_draft_model_path is not None
        and is_runai_obj_uri(cfg.speculative_draft_model_path)
        and cfg.speculative_draft_load_format is None
    ):
        declare_resolution(
            server_args,
            "_handle_load_format",
            speculative_draft_load_format="runai_streamer",
        )

    if cfg.custom_weight_loader is None:
        declare_resolution(server_args, "_handle_load_format", custom_weight_loader=[])

    if cfg.load_format == "remote_instance":
        if cfg.remote_instance_weight_loader_backend != "modelexpress" and (
            cfg.remote_instance_weight_loader_seed_instance_ip is None
            or cfg.remote_instance_weight_loader_seed_instance_service_port is None
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to incomplete remote instance weight loader settings."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )
        elif (
            cfg.remote_instance_weight_loader_send_weights_group_ports is None
            and cfg.remote_instance_weight_loader_backend == "nccl"
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to incomplete remote instance weight loader NCCL group ports settings."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )
        elif (
            cfg.remote_instance_weight_loader_backend == "transfer_engine"
            and not validate_transfer_engine(server_args)
        ):
            logger.warning(
                "Fallback load_format to 'auto' due to 'transfer_engine' backend is not supported."
            )
            declare_resolution(
                server_args,
                "_handle_load_format",
                load_format="auto",
            )

    # Check whether TransferEngine can be used when users want to start seed service that supports TransferEngine backend.
    if cfg.remote_instance_weight_loader_start_seed_via_transfer_engine:
        declare_resolution(
            server_args,
            "_handle_load_format",
            remote_instance_weight_loader_start_seed_via_transfer_engine=validate_transfer_engine(
                server_args
            ),
        )

    # "ipc_cache" is an internal-only load format: ModelRunner sets it
    # automatically when the weight cache is enabled, and it is not a public
    # --load-format choice. Setting it directly is always wrong (no daemon is
    # launched, and fallback_load_format inherits a nonsensical format), so
    # reject it and point at the knob (defense-in-depth; the CLI already
    # rejects it via LOAD_FORMAT_CHOICES).
    if cfg.load_format == "ipc_cache":
        raise ValueError(
            "load_format='ipc_cache' is an internal-only format and must not "
            "be set directly. Enable the weight cache via --weight-cache-mode "
            "client (connect to an existing daemon) or daemon (launch one); "
            "that selects IPC loading automatically."
        )

    # Speculative decoding loads an extra draft model whose weights the
    # daemon does not export, so refuse the combination up front instead of
    # failing deep inside draft-worker load (draft-model daemon TBD).
    if cfg.weight_cache_mode != "off" and cfg.speculative_algorithm is not None:
        raise ValueError(
            "--weight-cache-mode is not supported together with speculative "
            "decoding (--speculative-algorithm): the weight cache daemon does "
            "not export the draft model's weights. Disable one of them "
            "(--weight-cache-mode off) for this configuration."
        )


def validate_transfer_engine(server_args: Any):
    cfg = resolving_view(server_args)
    try:
        mooncake_available = importlib.util.find_spec("mooncake.engine") is not None
    except (ModuleNotFoundError, ValueError):
        mooncake_available = False
    if not mooncake_available:
        logger.warning(
            "Failed to import mooncake.engine. Does not support using TransferEngine as remote instance weight loader backend."
        )
        return False
    elif cfg.enable_memory_saver:
        logger.warning(
            "Memory saver is enabled, which is not compatible with TransferEngine. Does not support using TransferEngine as remote instance weight loader backend."
        )
        return False
    else:
        return True


def is_mistral_native_format(server_args: Any) -> bool:
    """True iff the checkpoint requires load_format=mistral.

    Looks for consolidated*.safetensors with no competing
    model*.safetensors; when both weight formats ship in the
    same checkpoint (e.g. Mistral-7B-Instruct-v0.3) the HF path is
    preferred to avoid loading Mistral-named weights into an
    HF-named architecture.

    Name override: mistral-large-3 / mistral-small-4 /
    leanstral always treat as Mistral-native when params.json
    is present -- those families need Mistral weight loading
    regardless of which weight files happen to be present.
    """
    cfg = resolving_view(server_args)
    _MISTRAL_NATIVE_PATTERNS = (
        "mistral-large-3",
        "mistral-small-4",
        "leanstral",
    )
    name_matches = any(
        p in str(cfg.model_path).lower() for p in _MISTRAL_NATIVE_PATTERNS
    )

    def _check_format(has_params, has_consolidated, has_hf_weights) -> bool:
        if has_params and name_matches:
            return True
        return has_consolidated and not has_hf_weights

    if os.path.isdir(cfg.model_path):
        return _check_format(
            has_params=os.path.exists(os.path.join(cfg.model_path, "params.json")),
            has_consolidated=bool(
                glob.glob(os.path.join(cfg.model_path, "consolidated*.safetensors"))
            ),
            has_hf_weights=bool(
                glob.glob(os.path.join(cfg.model_path, "model*.safetensors"))
            ),
        )

    try:
        from huggingface_hub import HfApi

        files = {s.rfilename for s in HfApi().model_info(cfg.model_path).siblings}
        return _check_format(
            has_params="params.json" in files,
            has_consolidated=any(
                f.startswith("consolidated") and f.endswith(".safetensors")
                for f in files
            ),
            has_hf_weights=any(
                f.startswith("model") and f.endswith(".safetensors") and "/" not in f
                for f in files
            ),
        )
    except Exception:
        return False
