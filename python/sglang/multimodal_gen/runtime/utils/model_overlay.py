# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import glob
import hashlib
import importlib.util
import json
import os
import shutil
from typing import Any, Callable, cast

from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException

from sglang.multimodal_gen.runtime.loader.weight_utils import get_lock
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.utils import load_diffusion_overlay_registry_from_env

logger = init_logger(__name__)

# Built-in diffusion model overlay registry.
BUILTIN_MODEL_OVERLAY_REGISTRY: dict[str, dict[str, Any]] = {
    "Lightricks/LTX-2.3": {
        "overlay_repo_id": "MickJ/LTX-2.3-overlay",
        "overlay_revision": "main",
        "bundled_overlay_subdir": "ltx_2_3",
    },
}


MODEL_OVERLAY_METADATA_PATTERNS = [
    "*.json",
    "*.md",
    "*.py",
    "*.txt",
    "**/*.json",
    "**/*.md",
    "**/*.py",
    "**/*.txt",
]

_MODEL_OVERLAY_REGISTRY_CACHE: dict[str, dict[str, Any]] | None = None


def _resolve_bundled_overlay_dir(overlay_spec: dict[str, Any]) -> str | None:
    bundled_overlay_subdir = overlay_spec.get("bundled_overlay_subdir")
    if not bundled_overlay_subdir:
        return None
    bundled_overlay_dir = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "model_overlays",
            str(bundled_overlay_subdir),
        )
    )
    if not os.path.isdir(bundled_overlay_dir):
        return None
    if load_overlay_manifest_if_present(bundled_overlay_dir) is None:
        return None
    return bundled_overlay_dir


def get_diffusion_cache_root() -> str:
    return os.path.expanduser(
        os.getenv("SGLANG_DIFFUSION_CACHE_ROOT", "~/.cache/sgl_diffusion")
    )


def clear_model_overlay_registry_cache() -> None:
    global _MODEL_OVERLAY_REGISTRY_CACHE
    _MODEL_OVERLAY_REGISTRY_CACHE = None


def _load_model_overlay_registry() -> dict[str, dict[str, Any]]:
    global _MODEL_OVERLAY_REGISTRY_CACHE
    if _MODEL_OVERLAY_REGISTRY_CACHE is not None:
        return _MODEL_OVERLAY_REGISTRY_CACHE

    # Built-in registry is the stable default path; env only overrides it.
    normalized = _normalize_model_overlay_registry(BUILTIN_MODEL_OVERLAY_REGISTRY)

    env_registry = load_diffusion_overlay_registry_from_env()
    if not env_registry:
        _MODEL_OVERLAY_REGISTRY_CACHE = normalized
        return _MODEL_OVERLAY_REGISTRY_CACHE

    normalized.update(_normalize_model_overlay_registry(env_registry))
    _MODEL_OVERLAY_REGISTRY_CACHE = normalized
    return _MODEL_OVERLAY_REGISTRY_CACHE


def _normalize_model_overlay_registry(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for source_model_id, spec in payload.items():
        if isinstance(spec, str):
            normalized[source_model_id] = {"overlay_repo_id": spec}
            continue
        if not isinstance(spec, dict):
            raise ValueError(
                "Overlay registry values must be either strings or JSON objects"
            )
        overlay_repo_id = spec.get("overlay_repo_id")
        if not overlay_repo_id:
            raise ValueError(
                f"Overlay registry entry for {source_model_id!r} is missing overlay_repo_id"
            )
        normalized[source_model_id] = dict(spec)
    return normalized


def resolve_model_overlay(model_name_or_path: str) -> dict[str, Any] | None:
    registry = _load_model_overlay_registry()
    return registry.get(model_name_or_path)


def resolve_model_overlay_target(
    model_name_or_path: str,
) -> tuple[str, dict[str, Any]] | None:
    registry = _load_model_overlay_registry()

    exact = registry.get(model_name_or_path)
    if exact is not None:
        return model_name_or_path, exact

    if os.path.exists(model_name_or_path):
        # Local source dirs do not have a repo id, so match them by basename.
        base_name = os.path.basename(os.path.normpath(model_name_or_path))
        for source_model_id, spec in registry.items():
            if base_name == source_model_id.rsplit("/", 1)[-1]:
                return source_model_id, spec

    return None


def load_overlay_manifest_if_present(overlay_dir: str) -> dict[str, Any] | None:
    overlay_manifest_path = os.path.join(
        overlay_dir, "_overlay", "overlay_manifest.json"
    )
    if not os.path.exists(overlay_manifest_path):
        return None
    with open(overlay_manifest_path, encoding="utf-8") as f:
        manifest = cast(dict[str, Any], json.load(f))
    return manifest


def load_model_index_from_dir(model_dir: str) -> dict[str, Any]:
    model_index_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(model_index_path):
        raise ValueError(f"model_index.json not found under {model_dir}")
    with open(model_index_path, encoding="utf-8") as f:
        config = cast(dict[str, Any], json.load(f))
    if "_class_name" not in config or "_diffusers_version" not in config:
        raise ValueError(f"Invalid model_index.json under {model_dir}")
    config["pipeline_name"] = config["_class_name"]
    return config


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _find_missing_required_paths(
    root_dir: str, required_paths: list[str] | tuple[str, ...]
) -> list[str]:
    missing: list[str] = []
    for rel_path in required_paths:
        if not os.path.exists(os.path.join(root_dir, rel_path)):
            missing.append(rel_path)
    return missing


def _link_or_copy_file(src: str, dst: str) -> None:
    src = os.path.realpath(src)
    _ensure_dir(os.path.dirname(dst))
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return
    except OSError:
        pass
    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass
    shutil.copy2(src, dst)


def _copytree_link_or_copy(src_dir: str, dst_dir: str) -> None:
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        target_root = dst_dir if rel_root == "." else os.path.join(dst_dir, rel_root)
        _ensure_dir(target_root)
        for file_name in files:
            src_file = os.path.join(root, file_name)
            dst_file = os.path.join(target_root, file_name)
            _link_or_copy_file(src_file, dst_file)


def ensure_overlay_source_dir_complete(
    *,
    source_model_id: str,
    source_dir: str,
    manifest: dict[str, Any],
    local_dir: str | None,
    allow_patterns: list[str] | None,
    download: bool,
    snapshot_download_fn: Callable[..., str],
) -> str:
    required_source_files = cast(
        list[str], list(manifest.get("required_source_files", []))
    )
    if not required_source_files:
        return source_dir

    # Metadata-only overlays often need a partial source snapshot. Re-download
    # only when the current source dir is missing required files.
    missing_paths = _find_missing_required_paths(source_dir, required_source_files)
    if not missing_paths:
        return source_dir

    if not download:
        raise ValueError(
            f"Overlay source model {source_model_id} is missing required files "
            f"{missing_paths} and download=False."
        )

    logger.warning(
        "Overlay source model %s is missing required files %s. "
        "Re-downloading source snapshot.",
        source_model_id,
        missing_paths,
    )
    source_allow_patterns = manifest.get("source_allow_patterns")
    effective_allow_patterns = (
        cast(list[str] | None, source_allow_patterns)
        if source_allow_patterns is not None
        else allow_patterns
    )
    with get_lock(source_model_id).acquire(poll_interval=2):
        source_dir = snapshot_download_fn(
            repo_id=source_model_id,
            ignore_patterns=["*.onnx", "*.msgpack"],
            allow_patterns=effective_allow_patterns,
            local_dir=local_dir,
            max_workers=8,
            force_download=True,
        )
    missing_after_redownload = _find_missing_required_paths(
        source_dir, required_source_files
    )
    if missing_after_redownload:
        raise ValueError(
            f"Overlay source model {source_model_id} is still missing required files "
            f"{missing_after_redownload} after re-download."
        )
    return str(source_dir)


def resolve_direct_overlay_repo(
    model_name_or_path: str,
    *,
    hf_hub_download_fn: Callable[..., str],
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    if os.path.exists(model_name_or_path):
        manifest = load_overlay_manifest_if_present(model_name_or_path)
        if manifest is None:
            return None
        source_model_id = manifest.get("source_model_id")
        if not source_model_id:
            raise ValueError(
                f"Overlay repo {model_name_or_path} is missing source_model_id in _overlay/overlay_manifest.json"
            )
        overlay_spec = {
            "overlay_repo_id": model_name_or_path,
            "overlay_revision": "local",
        }
        return overlay_spec, model_name_or_path, manifest

    try:
        manifest_path = hf_hub_download_fn(
            repo_id=model_name_or_path,
            filename="_overlay/overlay_manifest.json",
        )
        overlay_dir = os.path.dirname(os.path.dirname(manifest_path))
    except (
        RepositoryNotFoundError,
        RevisionNotFoundError,
        LocalEntryNotFoundError,
        RequestsConnectionError,
        RequestException,
    ):
        return None
    except Exception:
        return None

    manifest = load_overlay_manifest_if_present(overlay_dir)
    if manifest is None:
        return None
    source_model_id = manifest.get("source_model_id")
    if not source_model_id:
        raise ValueError(
            f"Overlay repo {model_name_or_path} is missing source_model_id in _overlay/overlay_manifest.json"
        )
    overlay_spec = {
        "overlay_repo_id": model_name_or_path,
        "overlay_revision": "main",
    }
    return overlay_spec, overlay_dir, manifest


def download_overlay_metadata(
    source_model_id: str,
    overlay_spec: dict[str, Any],
    *,
    snapshot_download_fn: Callable[..., str],
) -> str:
    bundled_overlay_dir = _resolve_bundled_overlay_dir(overlay_spec)
    if bundled_overlay_dir is not None:
        logger.info(
            "Using bundled overlay metadata for %s from %s",
            source_model_id,
            bundled_overlay_dir,
        )
        return bundled_overlay_dir

    overlay_repo_id = str(overlay_spec["overlay_repo_id"])
    if os.path.exists(overlay_repo_id):
        logger.info(
            "Using local overlay metadata for %s from %s",
            source_model_id,
            overlay_repo_id,
        )
        return overlay_repo_id
    revision = overlay_spec.get("overlay_revision")
    logger.info(
        "Downloading overlay metadata for %s from %s",
        source_model_id,
        overlay_repo_id,
    )
    return str(
        snapshot_download_fn(
            repo_id=overlay_repo_id,
            allow_patterns=MODEL_OVERLAY_METADATA_PATTERNS,
            revision=revision,
            max_workers=4,
        )
    )


def _apply_overlay_file_mappings(
    *,
    source_dir: str,
    output_dir: str,
    file_mappings: list[dict[str, Any]],
) -> None:
    for mapping in file_mappings:
        mapping_type = mapping.get("type", "file")
        src_rel = mapping.get("src")
        if not src_rel:
            raise ValueError(f"Overlay file mapping is missing src: {mapping}")
        src_path = os.path.join(source_dir, src_rel)
        if mapping_type == "tree":
            if not os.path.isdir(src_path):
                raise ValueError(f"Tree mapping source does not exist: {src_path}")
            dst_dir = os.path.join(output_dir, str(mapping.get("dst_dir", src_rel)))
            _copytree_link_or_copy(src_path, dst_dir)
            continue
        if mapping_type == "glob":
            matched = glob.glob(src_path, recursive=True)
            if not matched:
                raise ValueError(f"Glob mapping matched no files: {src_path}")
            for matched_path in matched:
                if os.path.isdir(matched_path):
                    continue
                rel_path = os.path.relpath(matched_path, source_dir)
                dst_path = os.path.join(output_dir, rel_path)
                _link_or_copy_file(matched_path, dst_path)
            continue

        if not os.path.isfile(src_path):
            raise ValueError(f"File mapping source does not exist: {src_path}")
        dst_rel = str(mapping.get("dst", os.path.basename(src_rel)))
        dst_path = os.path.join(output_dir, dst_rel)
        _link_or_copy_file(src_path, dst_path)


def _run_overlay_custom_materializer(
    *,
    overlay_dir: str,
    source_dir: str,
    output_dir: str,
    manifest: dict[str, Any],
) -> None:
    custom_materializer = manifest.get("custom_materializer")
    if not custom_materializer:
        return
    script_path = os.path.join(overlay_dir, str(custom_materializer))
    if not os.path.exists(script_path):
        raise ValueError(f"Custom materializer script not found: {script_path}")

    spec = importlib.util.spec_from_file_location(
        "_sglang_overlay_materializer", script_path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Failed to import custom materializer: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    materialize_fn = getattr(module, "materialize", None)
    if materialize_fn is None:
        raise ValueError(
            f"Custom materializer {script_path} must define materialize(...)"
        )

    materialize_fn(
        overlay_dir=overlay_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        manifest=manifest,
    )


def materialize_overlay_model(
    *,
    source_model_id: str,
    overlay_spec: dict[str, Any],
    overlay_dir: str,
    source_dir: str,
    verify_diffusers_model_complete_fn: Callable[[str], bool],
) -> str:
    overlay_manifest_path = os.path.join(
        overlay_dir, "_overlay", "overlay_manifest.json"
    )
    if not os.path.exists(overlay_manifest_path):
        raise ValueError(
            f"Overlay repo for {source_model_id} is missing _overlay/overlay_manifest.json"
        )

    with open(overlay_manifest_path, encoding="utf-8") as f:
        manifest = cast(dict[str, Any], json.load(f))

    materializer_version = str(manifest.get("materializer_version", "v1"))
    overlay_repo_id = str(overlay_spec["overlay_repo_id"])
    overlay_revision = str(overlay_spec.get("overlay_revision", "main"))
    cache_key = hashlib.sha256(
        json.dumps(
            {
                "source_model_id": source_model_id,
                "overlay_repo_id": overlay_repo_id,
                "overlay_revision": overlay_revision,
                "materializer_version": materializer_version,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_root = os.path.join(get_diffusion_cache_root(), "materialized_models")
    _ensure_dir(cache_root)
    safe_name = source_model_id.replace("/", "__")
    final_dir = os.path.join(cache_root, f"{safe_name}-{cache_key}")
    marker_path = os.path.join(final_dir, ".sglang_overlay_materialized.json")
    if verify_diffusers_model_complete_fn(final_dir) and os.path.exists(marker_path):
        return final_dir

    lock_name = (
        f"overlay-materialize::{source_model_id}::{overlay_repo_id}::{overlay_revision}"
    )
    with get_lock(lock_name).acquire(poll_interval=2):
        if verify_diffusers_model_complete_fn(final_dir) and os.path.exists(
            marker_path
        ):
            return final_dir

        logger.info(
            "Materializing overlay model for %s into %s",
            source_model_id,
            final_dir,
        )
        logger.info(
            "Overlay source repo: %s, overlay repo: %s@%s",
            source_model_id,
            overlay_repo_id,
            overlay_revision,
        )
        tmp_dir = final_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        logger.info("Copying overlay metadata into temporary materialized directory")
        shutil.copytree(
            overlay_dir,
            tmp_dir,
            ignore=shutil.ignore_patterns("*.safetensors", "*.bin", "*.pth", "*.pt"),
        )

        overlay_hidden_dir = os.path.join(tmp_dir, "_overlay")
        if os.path.isdir(overlay_hidden_dir):
            shutil.rmtree(overlay_hidden_dir)

        file_mappings = manifest.get("file_mappings", [])
        if file_mappings:
            logger.info("Applying %d overlay file mappings", len(file_mappings))
            _apply_overlay_file_mappings(
                source_dir=source_dir,
                output_dir=tmp_dir,
                file_mappings=cast(list[dict[str, Any]], file_mappings),
            )
        if manifest.get("custom_materializer"):
            logger.info(
                "Running custom overlay materializer: %s",
                manifest["custom_materializer"],
            )
        _run_overlay_custom_materializer(
            overlay_dir=overlay_dir,
            source_dir=source_dir,
            output_dir=tmp_dir,
            manifest=manifest,
        )

        with open(marker_path.replace(final_dir, tmp_dir), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_model_id": source_model_id,
                    "source_dir": source_dir,
                    "overlay_repo_id": overlay_repo_id,
                    "overlay_revision": overlay_revision,
                    "materializer_version": materializer_version,
                },
                f,
                indent=2,
                sort_keys=True,
            )

        os.replace(tmp_dir, final_dir)
        logger.info("Overlay materialization finished: %s", final_dir)

    return final_dir


def maybe_load_overlay_model_index(
    model_name_or_path: str,
    *,
    snapshot_download_fn: Callable[..., str],
    hf_hub_download_fn: Callable[..., str],
) -> dict[str, Any] | None:
    if os.path.exists(model_name_or_path):
        # A local overlay repo already contains the model_index we need.
        if load_overlay_manifest_if_present(model_name_or_path) is not None:
            return load_model_index_from_dir(model_name_or_path)
        return None

    overlay_target = resolve_model_overlay_target(model_name_or_path)
    if overlay_target is not None:
        # Registry-mapped source model ids first resolve to overlay metadata.
        source_model_id, overlay_spec = overlay_target
        overlay_dir = download_overlay_metadata(
            source_model_id,
            overlay_spec,
            snapshot_download_fn=snapshot_download_fn,
        )
        return load_model_index_from_dir(overlay_dir)

    direct_overlay = resolve_direct_overlay_repo(
        model_name_or_path, hf_hub_download_fn=hf_hub_download_fn
    )
    if direct_overlay is None:
        return None

    _, overlay_dir, _ = direct_overlay
    return load_model_index_from_dir(overlay_dir)


def maybe_resolve_overlay_model_path(
    model_name_or_path: str,
    *,
    local_dir: str | None,
    download: bool,
    allow_patterns: list[str] | None,
    snapshot_download_fn: Callable[..., str],
    hf_hub_download_fn: Callable[..., str],
    verify_diffusers_model_complete_fn: Callable[[str], bool],
    base_model_download_fn: Callable[..., str],
) -> str | None:
    overlay_target = resolve_model_overlay_target(model_name_or_path)
    if overlay_target is not None:
        source_model_id, overlay_spec = overlay_target
        overlay_dir = download_overlay_metadata(
            source_model_id,
            overlay_spec,
            snapshot_download_fn=snapshot_download_fn,
        )
        manifest = load_overlay_manifest_if_present(overlay_dir)
        if manifest is None:
            # Full diffusers overlays do not need materialization.
            return base_model_download_fn(
                str(overlay_spec["overlay_repo_id"]),
                local_dir=local_dir,
                download=download,
                allow_patterns=allow_patterns,
                force_diffusers_model=True,
                skip_overlay_resolution=True,
            )
        source_allow_patterns = cast(
            list[str] | None, manifest.get("source_allow_patterns")
        )
        # For local source paths, reuse the directory directly instead of
        # round-tripping through snapshot_download.
        source_dir = (
            model_name_or_path
            if os.path.exists(model_name_or_path)
            else base_model_download_fn(
                source_model_id,
                local_dir=local_dir,
                download=download,
                allow_patterns=source_allow_patterns or allow_patterns,
                force_diffusers_model=False,
                skip_overlay_resolution=True,
            )
        )
        source_dir = ensure_overlay_source_dir_complete(
            source_model_id=source_model_id,
            source_dir=source_dir,
            manifest=manifest,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
            download=download,
            snapshot_download_fn=snapshot_download_fn,
        )
        return materialize_overlay_model(
            source_model_id=source_model_id,
            overlay_spec=overlay_spec,
            overlay_dir=overlay_dir,
            source_dir=source_dir,
            verify_diffusers_model_complete_fn=verify_diffusers_model_complete_fn,
        )

    direct_overlay = resolve_direct_overlay_repo(
        model_name_or_path, hf_hub_download_fn=hf_hub_download_fn
    )
    if direct_overlay is None:
        return None

    overlay_spec, overlay_dir, manifest = direct_overlay
    source_model_id = str(manifest["source_model_id"])
    # Direct overlay repos are always metadata-only; they need the original
    # source weights before they can be materialized into a diffusers-like dir.
    source_allow_patterns = cast(
        list[str] | None, manifest.get("source_allow_patterns")
    )
    source_dir = base_model_download_fn(
        source_model_id,
        local_dir=local_dir,
        download=download,
        allow_patterns=source_allow_patterns or allow_patterns,
        force_diffusers_model=False,
        skip_overlay_resolution=True,
    )
    source_dir = ensure_overlay_source_dir_complete(
        source_model_id=source_model_id,
        source_dir=source_dir,
        manifest=manifest,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
        download=download,
        snapshot_download_fn=snapshot_download_fn,
    )
    return materialize_overlay_model(
        source_model_id=source_model_id,
        overlay_spec=overlay_spec,
        overlay_dir=overlay_dir,
        source_dir=source_dir,
        verify_diffusers_model_complete_fn=verify_diffusers_model_complete_fn,
    )
