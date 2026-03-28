# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from SGLang: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/hf_transformers_utils.py

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Huggingface Transformers."""

import contextlib
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import time
from functools import reduce
from pathlib import Path
from typing import Any, Optional, Union, cast

from diffusers.loaders.lora_base import (
    _best_guess_weight_name,  # watch out for potetential removal from diffusers
)
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from transformers import AutoConfig, PretrainedConfig

from sglang.multimodal_gen.runtime.loader.utils import _clean_hf_config_inplace
from sglang.multimodal_gen.runtime.loader.weight_utils import get_lock
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs
from sglang.utils import is_in_ci

logger = init_logger(__name__)

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


def _get_diffusion_cache_root() -> str:
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

    raw_value = os.getenv("SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY", "").strip()
    if not raw_value:
        _MODEL_OVERLAY_REGISTRY_CACHE = {}
        return _MODEL_OVERLAY_REGISTRY_CACHE

    try:
        if raw_value.startswith("{"):
            payload = json.loads(raw_value)
        else:
            with open(os.path.expanduser(raw_value), encoding="utf-8") as f:
                payload = json.load(f)
    except Exception as exc:
        raise ValueError(
            "Failed to parse SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "SGLANG_DIFFUSION_MODEL_OVERLAY_REGISTRY must be a JSON object"
        )

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

    _MODEL_OVERLAY_REGISTRY_CACHE = normalized
    return _MODEL_OVERLAY_REGISTRY_CACHE


def resolve_model_overlay(model_name_or_path: str) -> dict[str, Any] | None:
    registry = _load_model_overlay_registry()
    return registry.get(model_name_or_path)


def _load_overlay_manifest_if_present(overlay_dir: str) -> dict[str, Any] | None:
    overlay_manifest_path = os.path.join(
        overlay_dir, "_overlay", "overlay_manifest.json"
    )
    if not os.path.exists(overlay_manifest_path):
        return None
    with open(overlay_manifest_path, encoding="utf-8") as f:
        manifest = cast(dict[str, Any], json.load(f))
    return manifest


def _find_missing_required_paths(
    root_dir: str, required_paths: list[str] | tuple[str, ...]
) -> list[str]:
    missing: list[str] = []
    for rel_path in required_paths:
        if not os.path.exists(os.path.join(root_dir, rel_path)):
            missing.append(rel_path)
    return missing


def _ensure_overlay_source_dir_complete(
    *,
    source_model_id: str,
    source_dir: str,
    manifest: dict[str, Any],
    local_dir: str | None,
    allow_patterns: list[str] | None,
    download: bool,
) -> str:
    required_source_files = cast(
        list[str], list(manifest.get("required_source_files", []))
    )
    if not required_source_files:
        return source_dir

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
        source_dir = snapshot_download(
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


def _resolve_direct_overlay_repo(
    model_name_or_path: str,
) -> tuple[dict[str, Any], str, dict[str, Any]] | None:
    if os.path.exists(model_name_or_path):
        manifest = _load_overlay_manifest_if_present(model_name_or_path)
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
        manifest_path = hf_hub_download(
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

    manifest = _load_overlay_manifest_if_present(overlay_dir)
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


def _download_overlay_metadata(
    source_model_id: str,
    overlay_spec: dict[str, Any],
) -> str:
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
        snapshot_download(
            repo_id=overlay_repo_id,
            allow_patterns=MODEL_OVERLAY_METADATA_PATTERNS,
            revision=revision,
            max_workers=4,
        )
    )


def _load_model_index_from_dir(model_dir: str) -> dict[str, Any]:
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


def _materialize_overlay_model(
    *,
    source_model_id: str,
    overlay_spec: dict[str, Any],
    overlay_dir: str,
    source_dir: str,
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
    cache_root = os.path.join(_get_diffusion_cache_root(), "materialized_models")
    _ensure_dir(cache_root)
    safe_name = source_model_id.replace("/", "__")
    final_dir = os.path.join(cache_root, f"{safe_name}-{cache_key}")
    marker_path = os.path.join(final_dir, ".sglang_overlay_materialized.json")
    if _verify_diffusers_model_complete(final_dir) and os.path.exists(marker_path):
        return final_dir

    lock_name = (
        f"overlay-materialize::{source_model_id}::{overlay_repo_id}::{overlay_revision}"
    )
    with get_lock(lock_name).acquire(poll_interval=2):
        if _verify_diffusers_model_complete(final_dir) and os.path.exists(marker_path):
            return final_dir

        tmp_dir = final_dir + ".tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
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
            _apply_overlay_file_mappings(
                source_dir=source_dir,
                output_dir=tmp_dir,
                file_mappings=cast(list[dict[str, Any]], file_mappings),
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

    return final_dir


def _check_index_files_for_missing_shards(
    model_path: str,
) -> tuple[bool, list[str], list[str]]:
    """
    Check all subdirectories for missing shards based on index files.

    This catches cases where a model download was interrupted, leaving
    some safetensors shards missing while the index file exists.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (all_valid, missing_files, checked_subdirs)
    """
    missing_files = []
    checked_subdirs = []

    # Add common subdirectories for diffusers models
    try:
        subdirs = os.listdir(model_path)
    except OSError as e:
        logger.warning("Failed to list model directory %s: %s", model_path, e)
        return True, [], []  # Assume valid if we can't check

    # Check the root directory and all subdirectories that might contain model weights
    dirs_to_check = [model_path]

    for subdir in subdirs:
        subdir_path = os.path.join(model_path, subdir)
        if os.path.isdir(subdir_path):
            dirs_to_check.append(subdir_path)

    for dir_path in dirs_to_check:
        # Find all safetensors index files
        index_files = glob.glob(os.path.join(dir_path, "*.safetensors.index.json"))

        for index_file in index_files:
            checked_subdirs.append(os.path.basename(dir_path))
            try:
                with open(index_file) as f:
                    index_data = json.load(f)

                weight_map = index_data.get("weight_map", {})
                if not weight_map:
                    continue

                # Get unique files referenced in weight_map
                required_files = set(weight_map.values())

                for file_name in required_files:
                    file_path = os.path.join(dir_path, file_name)
                    if not os.path.exists(file_path):
                        relative_path = os.path.relpath(file_path, model_path)
                        missing_files.append(relative_path)

            except Exception as e:
                logger.warning("Failed to read index file %s: %s", index_file, e)
                continue

    return len(missing_files) == 0, missing_files, checked_subdirs


def _cleanup_model_cache(model_path: str, reason: str) -> bool:
    """
    Remove the model cache directory to force a clean re-download.

    Args:
        model_path: Path to the model directory (snapshot path)
        reason: Reason for cleanup (for logging)

    Returns:
        True if cleanup was performed, False otherwise
    """
    # Navigate up to the model root directory: snapshots/hash -> snapshots -> model_root
    # HF cache structure: models--org--name/snapshots/hash/
    try:
        snapshot_dir = os.path.abspath(model_path)
        snapshots_dir = os.path.dirname(snapshot_dir)
        repo_folder = os.path.dirname(snapshots_dir)

        # Verify this looks like an HF cache structure
        if os.path.basename(snapshots_dir) != "snapshots":
            logger.warning(
                "Model path %s doesn't appear to be in HF cache structure, skipping cleanup",
                model_path,
            )
            return False

        logger.warning(
            "Removing model cache at %s. Reason: %s",
            repo_folder,
            reason,
        )
        shutil.rmtree(repo_folder)
        logger.info("Successfully removed corrupted cache directory")
        return True
    except Exception as e:
        logger.error(
            "Failed to remove corrupted cache directory %s: %s. "
            "Manual cleanup may be required.",
            model_path,
            e,
        )
        return False


def _ci_validate_diffusers_model(model_path: str) -> tuple[bool, bool]:
    """
    CI-specific validation for diffusers models.

    Checks all subdirectories (transformer, transformer_2, vae, etc.) for
    missing shards based on their index files. If issues are found in CI,
    cleans up the cache to force re-download.

    Args:
        model_path: Path to the model directory

    Returns:
        Tuple of (is_valid, cleanup_performed)
        - is_valid: True if the model is valid
        - cleanup_performed: True if cleanup was performed (only relevant when is_valid=False)
    """
    if not is_in_ci():
        return True, False
    is_valid, missing_files, checked_subdirs = _check_index_files_for_missing_shards(
        model_path
    )

    if not is_valid:
        logger.error(
            "CI validation failed for %s. Missing %d file(s): %s. "
            "Checked subdirectories: %s",
            model_path,
            len(missing_files),
            missing_files[:5] if len(missing_files) > 5 else missing_files,
            checked_subdirs,
        )
        cleanup_performed = _cleanup_model_cache(
            model_path,
            f"Missing {len(missing_files)} shard file(s): {missing_files[:3]}",
        )
        return False, cleanup_performed

    if checked_subdirs:
        logger.info(
            "CI validation passed for %s. Checked subdirectories: %s",
            model_path,
            checked_subdirs,
        )

    return True, False


def _verify_diffusers_model_complete(path: str) -> bool:
    """Check if a diffusers model directory has all required component subdirectories."""
    config_path = os.path.join(path, "model_index.json")
    if not os.path.exists(config_path):
        return False

    try:
        with open(config_path) as config_file:
            model_index = json.load(config_file)
    except Exception as exc:
        logger.warning("Failed to read model_index.json at %s: %s", config_path, exc)
        return False

    component_keys = [
        key
        for key, value in model_index.items()
        if isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ]
    if component_keys:
        return all(os.path.exists(os.path.join(path, key)) for key in component_keys)

    return os.path.exists(os.path.join(path, "transformer")) and os.path.exists(
        os.path.join(path, "vae")
    )


_CONFIG_REGISTRY: dict[str, type[PretrainedConfig]] = {
    # ChatGLMConfig.model_type: ChatGLMConfig,
    # DbrxConfig.model_type: DbrxConfig,
    # ExaoneConfig.model_type: ExaoneConfig,
    # Qwen2_5_VLConfig.model_type: Qwen2_5_VLConfig,
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(model_path: str):
    if os.path.exists(model_path):
        return model_path

    return snapshot_download(model_path, allow_patterns=["*.json", "*.bin", "*.model"])


def get_hf_config(
    component_model_path: str,
    trust_remote_code: bool,
    revision: str | None = None,
    model_override_args: dict | None = None,
    **kwargs,
) -> PretrainedConfig:
    if check_gguf_file(component_model_path):
        raise NotImplementedError("GGUF models are not supported.")

    config = AutoConfig.from_pretrained(
        component_model_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
        **kwargs,
    )
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(component_model_path, revision=revision)
        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        config._name_or_path = component_model_path
    if model_override_args:
        config.update(model_override_args)

    return config


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    return AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )


def load_dict(file_path):
    if not os.path.exists(file_path):
        return {}
    try:
        # Load the config directly from the file
        with open(file_path) as f:
            config_dict: dict[str, Any] = json.load(f)
        if "_diffusers_version" in config_dict:
            config_dict.pop("_diffusers_version")
        # TODO(will): apply any overrides from inference args
        return config_dict
    except Exception as e:
        raise RuntimeError(
            f"Failed to load diffusers config from {file_path}: {e}"
        ) from e


def get_diffusers_component_config(
    component_path: str,
) -> dict[str, Any]:
    """Gets a configuration of a submodule for the given diffusers model."""
    # Download from HuggingFace Hub if path doesn't exist locally
    if not os.path.exists(component_path):
        component_path = maybe_download_model(component_path)

    config_names = ["generation_config.json"]
    # By default, we load config.json, but scheduler_config.json for scheduler
    if "scheduler" in component_path:
        config_names.append("scheduler_config.json")
    else:
        config_names.append("config.json")

    config_file_paths = [
        os.path.join(component_path, config_name) for config_name in config_names
    ]

    combined_config = reduce(
        lambda acc, path: acc | load_dict(path), config_file_paths, {}
    )

    _clean_hf_config_inplace(combined_config)

    logger.debug("HF model config: %s", combined_config)

    return combined_config


# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = {
            tokenizer.get_added_vocab()["<|eom_id|>"]
        }
    else:
        tokenizer.additional_stop_token_ids = None


def check_gguf_file(model: str | os.PathLike) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"


def maybe_download_lora(
    model_name_or_path: str, local_dir: str | None = None, download: bool = True
) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.
    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub

    Returns:
        Local path to the model
    """
    allow_patterns = ["*.json", "*.safetensors", "*.bin"]

    local_path = maybe_download_model(
        model_name_or_path,
        local_dir,
        download,
        is_lora=True,
        allow_patterns=allow_patterns,
    )
    # return directly if local_path is a file
    if os.path.isfile(local_path):
        return local_path

    weight_name = _best_guess_weight_name(local_path, file_extension=".safetensors")
    # AMD workaround: PR 15813 changed from model_name_or_path to local_path,
    # which can return None. Fall back to original behavior on ROCm.
    if weight_name is None and current_platform.is_rocm():
        weight_name = _best_guess_weight_name(
            model_name_or_path, file_extension=".safetensors"
        )
    return os.path.join(local_path, weight_name)


def verify_model_config_and_directory(model_path: str) -> dict[str, Any]:
    """
    Verify that the model directory contains a valid diffusers configuration.

    Args:
        model_path: Path to the model directory

    Returns:
        The loaded model configuration as a dictionary
    """

    # Check for model_index.json which is required for diffusers models
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        raise ValueError(
            f"Model directory {model_path} does not contain model_index.json. "
            "Only HuggingFace diffusers format is supported."
        )

    # Load the config
    with open(config_path) as f:
        config = json.load(f)

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        raise ValueError("model_index.json does not contain _diffusers_version")

    logger.info("Diffusers version: %s", config["_diffusers_version"])

    component_keys = [
        key
        for key, value in config.items()
        if isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ]
    if component_keys:
        missing_components = [
            component_key
            for component_key in component_keys
            if not os.path.exists(os.path.join(model_path, component_key))
        ]
        if missing_components:
            missing_str = ", ".join(missing_components)
            raise ValueError(
                f"Model directory {model_path} is missing required component "
                f"directories: {missing_str}."
            )
    else:
        transformer_dir = os.path.join(model_path, "transformer")
        vae_dir = os.path.join(model_path, "vae")
        if not os.path.exists(transformer_dir):
            raise ValueError(
                f"Model directory {model_path} does not contain a transformer/ directory."
            )
        if not os.path.exists(vae_dir):
            raise ValueError(
                f"Model directory {model_path} does not contain a vae/ directory."
            )
    return cast(dict[str, Any], config)


def maybe_download_model_index(model_name_or_path: str) -> dict[str, Any]:
    """
    Download and extract just the model_index.json for a Hugging Face model.

    Args:
        model_name_or_path: Path or HF Hub model ID

    Returns:
        The parsed model_index.json as a dictionary
    """
    import tempfile

    from huggingface_hub.errors import EntryNotFoundError

    # If it's a local path, verify it directly
    if os.path.exists(model_name_or_path):
        manifest = _load_overlay_manifest_if_present(model_name_or_path)
        if manifest is not None:
            return _load_model_index_from_dir(model_name_or_path)
        try:
            return verify_model_config_and_directory(model_name_or_path)
        except ValueError:
            # Not a pipeline, maybe a single model.
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                return config
            raise

    overlay_spec = resolve_model_overlay(model_name_or_path)
    if overlay_spec is not None:
        overlay_metadata_dir = _download_overlay_metadata(
            model_name_or_path, overlay_spec
        )
        return _load_model_index_from_dir(overlay_metadata_dir)

    direct_overlay = _resolve_direct_overlay_repo(model_name_or_path)
    if direct_overlay is not None:
        _, overlay_dir, _ = direct_overlay
        return _load_model_index_from_dir(overlay_dir)

    # For remote models, download just the model_index.json
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Download just the model_index.json file
            model_index_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="model_index.json",
                local_dir=tmp_dir,
            )

            # Load the model_index.json
            with open(model_index_path) as f:
                config: dict[str, Any] = json.load(f)

            # Verify it has the required fields
            if "_class_name" not in config:
                raise ValueError(
                    f"model_index.json for {model_name_or_path} does not contain _class_name field"
                )

            if "_diffusers_version" not in config:
                raise ValueError(
                    f"model_index.json for {model_name_or_path} does not contain _diffusers_version field"
                )

            # Add the pipeline name for downstream use
            config["pipeline_name"] = config["_class_name"]

            logger.debug(
                "Downloaded model_index.json for %s, pipeline: %s",
                model_name_or_path,
                config["_class_name"],
            )
            return config
    except EntryNotFoundError:
        logger.warning(
            "model_index.json not found for %s. Assuming it is a single model and downloading it.",
            model_name_or_path,
        )
        local_path = maybe_download_model(model_name_or_path)
        config_path = os.path.join(local_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(
                f"Failed to find config.json for {model_name_or_path} after failing to find model_index.json"
                f"You might be looking for models ending with '-Diffusers'"
            )
        with open(config_path) as f:
            config = json.load(f)
        return config
    except Exception as e:
        raise ValueError(
            f"Failed to download or parse model_index.json for {model_name_or_path}: {e}"
        ) from e


def maybe_download_model(
    model_name_or_path: str,
    local_dir: str | None = None,
    download: bool = True,
    is_lora: bool = False,
    allow_patterns: list[str] | None = None,
    force_diffusers_model: bool = False,
    skip_overlay_resolution: bool = False,
) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.

    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub
        is_lora: If True, skip model completeness verification (LoRA models don't have transformer/vae directories)
        force_diffusers_model: If True, apply diffusers model check. Otherwise it should be a component model
    Returns:
        Local path to the model
    """

    overlay_spec = None
    if (
        force_diffusers_model
        and not skip_overlay_resolution
        and not os.path.exists(model_name_or_path)
    ):
        overlay_spec = resolve_model_overlay(model_name_or_path)
        if overlay_spec is not None:
            overlay_metadata_dir = _download_overlay_metadata(
                model_name_or_path, overlay_spec
            )
            manifest = _load_overlay_manifest_if_present(overlay_metadata_dir)
            if manifest is not None:
                source_allow_patterns = cast(
                    list[str] | None, manifest.get("source_allow_patterns")
                )
                source_dir = maybe_download_model(
                    model_name_or_path,
                    local_dir=local_dir,
                    download=download,
                    allow_patterns=source_allow_patterns or allow_patterns,
                    force_diffusers_model=False,
                    skip_overlay_resolution=True,
                )
                source_dir = _ensure_overlay_source_dir_complete(
                    source_model_id=model_name_or_path,
                    source_dir=source_dir,
                    manifest=manifest,
                    local_dir=local_dir,
                    allow_patterns=allow_patterns,
                    download=download,
                )
                return _materialize_overlay_model(
                    source_model_id=model_name_or_path,
                    overlay_spec=overlay_spec,
                    overlay_dir=overlay_metadata_dir,
                    source_dir=source_dir,
                )
            return maybe_download_model(
                str(overlay_spec["overlay_repo_id"]),
                local_dir=local_dir,
                download=download,
                is_lora=is_lora,
                allow_patterns=allow_patterns,
                force_diffusers_model=True,
                skip_overlay_resolution=True,
            )

    if force_diffusers_model and not skip_overlay_resolution:
        direct_overlay = _resolve_direct_overlay_repo(model_name_or_path)
        if direct_overlay is not None:
            overlay_spec, overlay_dir, manifest = direct_overlay
            source_model_id = str(manifest["source_model_id"])
            source_allow_patterns = cast(
                list[str] | None, manifest.get("source_allow_patterns")
            )
            source_dir = maybe_download_model(
                source_model_id,
                local_dir=local_dir,
                download=download,
                allow_patterns=source_allow_patterns or allow_patterns,
                force_diffusers_model=False,
                skip_overlay_resolution=True,
            )
            source_dir = _ensure_overlay_source_dir_complete(
                source_model_id=source_model_id,
                source_dir=source_dir,
                manifest=manifest,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                download=download,
            )
            return _materialize_overlay_model(
                source_model_id=source_model_id,
                overlay_spec=overlay_spec,
                overlay_dir=overlay_dir,
                source_dir=source_dir,
            )

    # 1. Local path check: if path exists locally, verify it's complete (skip for LoRA)
    if os.path.exists(model_name_or_path):
        if not force_diffusers_model:
            return model_name_or_path
        if is_lora or _verify_diffusers_model_complete(model_name_or_path):
            if not is_lora:
                is_valid, cleanup_performed = _ci_validate_diffusers_model(
                    model_name_or_path
                )
                if not is_valid:
                    if cleanup_performed:
                        logger.warning(
                            "CI validation failed for local model at %s, "
                            "cache has been cleaned up, will re-download",
                            model_name_or_path,
                        )
                        # Fall through to download
                    else:
                        raise ValueError(
                            f"CI validation failed for local model at {model_name_or_path}. "
                            "Some safetensors shards are missing. "
                            "Please manually delete the model directory and retry."
                        )
                else:
                    logger.info("Model already exists locally and is complete")
                    return model_name_or_path
            else:
                logger.info("Model already exists locally and is complete")
                return model_name_or_path
        else:
            logger.warning(
                "Local model at %s appears incomplete (missing required components), "
                "will attempt re-download",
                model_name_or_path,
            )

    # 2. Cache-first strategy (Fast Path)
    # Try to read from HF cache without network access
    try:
        logger.info(
            "Checking for cached model in HF Hub cache for %s...", model_name_or_path
        )
        local_path = snapshot_download(
            repo_id=model_name_or_path,
            ignore_patterns=["*.onnx", "*.msgpack"],
            local_dir=local_dir,
            local_files_only=True,
            max_workers=8,
        )
        if not force_diffusers_model:
            return str(local_path)
        if is_lora or _verify_diffusers_model_complete(local_path):
            if not is_lora:
                is_valid, cleanup_performed = _ci_validate_diffusers_model(local_path)
                if not is_valid:
                    logger.warning(
                        "CI validation failed for cached model at %s, "
                        "%s, will re-download",
                        local_path,
                        (
                            "cache has been cleaned up"
                            if cleanup_performed
                            else "cleanup was not performed"
                        ),
                    )
                    # Fall through to download
                else:
                    logger.info("Found complete model in cache at %s", local_path)
                    return str(local_path)
            else:
                logger.info("Found complete model in cache at %s", local_path)
                return str(local_path)
        else:
            if not download:
                raise ValueError(
                    f"Model {model_name_or_path} found in cache but is incomplete and download=False."
                )
            logger.info(
                "Model found in cache but incomplete, will download from HF Hub"
            )
    except LocalEntryNotFoundError:
        if not download:
            raise ValueError(
                f"Model {model_name_or_path} not found in local cache and download=False."
            )
        logger.info("Model not found in cache, will download from HF Hub")
    except Exception as e:
        logger.warning(
            "Unexpected error while checking cache for %s: %s, will attempt download",
            model_name_or_path,
            e,
        )
        if not download:
            raise ValueError(
                f"Error checking cache for {model_name_or_path} and download=False: {e}"
            ) from e

    # 3. Download strategy (with retry mechanism)
    MAX_RETRIES = 5
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(
                "Downloading model snapshot from HF Hub for %s (attempt %d/%d)...",
                model_name_or_path,
                attempt + 1,
                MAX_RETRIES,
            )
            with get_lock(model_name_or_path).acquire(poll_interval=2):
                local_path = snapshot_download(
                    repo_id=model_name_or_path,
                    ignore_patterns=["*.onnx", "*.msgpack"],
                    allow_patterns=allow_patterns,
                    local_dir=local_dir,
                    max_workers=8,
                )

            if not force_diffusers_model:
                return str(local_path)
            # Verify downloaded model is complete (skip for LoRA)
            elif not is_lora and not _verify_diffusers_model_complete(local_path):
                logger.warning(
                    "Downloaded model at %s is incomplete, retrying with force_download=True",
                    local_path,
                )
                with get_lock(model_name_or_path).acquire(poll_interval=2):
                    local_path = snapshot_download(
                        repo_id=model_name_or_path,
                        ignore_patterns=["*.onnx", "*.msgpack"],
                        local_dir=local_dir,
                        max_workers=8,
                        force_download=True,
                    )
                if not _verify_diffusers_model_complete(local_path):
                    raise ValueError(
                        f"Downloaded model at {local_path} is still incomplete after forced re-download. "
                        "The model repository may be missing required components (model_index.json, transformer/, or vae/)."
                    )

            # CI validation: check all subdirectories for missing shards after download
            if not is_lora:
                is_valid, cleanup_performed = _ci_validate_diffusers_model(local_path)
                if not is_valid:
                    # In CI, if validation fails after download, we have a serious issue
                    # If cleanup was performed, the next retry should get a fresh download
                    raise ValueError(
                        f"CI validation failed for downloaded model at {local_path}. "
                        f"Some safetensors shards are missing. Cleanup performed: {cleanup_performed}."
                    )

            logger.info("Downloaded model to %s", local_path)
            return str(local_path)

        except (RepositoryNotFoundError, RevisionNotFoundError) as e:
            raise ValueError(
                f"Model or revision not found at {model_name_or_path}. "
                f"Please check the model ID or ensure you have access to the repository. Error: {e}"
            ) from e
        except (RequestException, RequestsConnectionError) as e:
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"Could not find model at {model_name_or_path} and failed to download from HF Hub "
                    f"after {MAX_RETRIES} attempts due to network error: {e}"
                ) from e
            wait_time = 2**attempt
            logger.warning(
                "Download failed (attempt %d/%d) due to network error: %s. "
                "Retrying in %d seconds...",
                attempt + 1,
                MAX_RETRIES,
                e,
                wait_time,
            )
            time.sleep(wait_time)
        except Exception as e:
            raise ValueError(
                f"Could not find model at {model_name_or_path} and failed to download from HF Hub: {e}"
            ) from e


# Unified download functions with Hugging Face-compatible names
def hf_hub_download(
    repo_id: str,
    filename: str,
    local_dir: Optional[Union[str, Path]] = None,
    **kwargs,
) -> str:
    """Unified hf_hub_download that supports both Hugging Face Hub and ModelScope."""
    if envs.SGLANG_USE_MODELSCOPE.get():
        from modelscope import model_file_download

        return model_file_download(
            model_id=repo_id,
            file_path=filename,
            cache_dir=local_dir,
            **kwargs,
        )
    else:
        from huggingface_hub import hf_hub_download as _hf_hub_download

        return _hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            **kwargs,
        )


def snapshot_download(
    repo_id: str,
    local_dir: Optional[Union[str, Path]] = None,
    ignore_patterns: Optional[Union[list[str], str]] = None,
    allow_patterns: Optional[Union[list[str], str]] = None,
    local_files_only: bool = False,
    max_workers: int = 8,
    **kwargs,
) -> str:
    """Unified snapshot_download that supports both Hugging Face Hub and ModelScope."""
    if envs.SGLANG_USE_MODELSCOPE.get():
        from modelscope import snapshot_download as _ms_snapshot_download

        ms_kwargs = {
            "model_id": repo_id,
            "local_dir": local_dir,
            "ignore_patterns": ignore_patterns,
            "allow_patterns": allow_patterns,
            "local_files_only": local_files_only,
            "max_workers": max_workers,
        }
        ms_kwargs.update(kwargs)
        return _ms_snapshot_download(**ms_kwargs)
    else:
        from huggingface_hub import snapshot_download as _hf_snapshot_download

        hf_kwargs = {
            "repo_id": repo_id,
            "local_dir": local_dir,
            "ignore_patterns": ignore_patterns,
            "allow_patterns": allow_patterns,
            "local_files_only": local_files_only,
            "max_workers": max_workers,
            "etag_timeout": 60,
        }
        hf_kwargs.update(kwargs)
        return _hf_snapshot_download(**hf_kwargs)
