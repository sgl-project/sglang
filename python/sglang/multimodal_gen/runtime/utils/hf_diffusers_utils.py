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
    EntryNotFoundError,
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
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    maybe_load_overlay_model_index,
    maybe_resolve_overlay_model_path,
)
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    normalize_flat_modelopt_quant_config,
)
from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers import check_gguf_file
from sglang.utils import is_in_ci

logger = init_logger(__name__)


_NON_WEIGHT_DIFFUSERS_COMPONENT_HINTS = (
    "tokenizer",
    "scheduler",
    "processor",
    "feature_extractor",
)
_WEIGHT_FILE_PATTERNS = (
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "*.ckpt",
)


def _model_hub_name() -> str:
    return "ModelScope" if envs.SGLANG_USE_MODELSCOPE.get() else "Hugging Face Hub"


def _is_revisionless_snapshot_root(local_path: str) -> bool:
    """Detect a resolved "snapshot" that is really the ``snapshots/`` parent.

    An empty ``refs/<revision>`` makes the offline resolver join ``""`` onto
    ``snapshots/`` and return the parent, which holds only revision subdirectories.
    The ``models--*`` folder above is required too, since a ``local_dir`` may
    legitimately be named ``snapshots``.
    """
    head, tail = os.path.split(os.path.normpath(local_path))
    return tail == "snapshots" and os.path.basename(head).split("--")[0] in (
        "models",
        "datasets",
        "spaces",
    )


def _snapshot_has_files(
    local_path: str,
    allow_patterns: Optional[Union[list[str], str]],
) -> bool:
    patterns = (
        [allow_patterns]
        if isinstance(allow_patterns, str)
        else allow_patterns or ["**/*"]
    )
    return any(
        os.path.isfile(candidate)
        for pattern in patterns
        for candidate in glob.iglob(
            os.path.join(local_path, pattern),
            recursive=True,
        )
    )


def _is_modelscope_not_found_error(error: BaseException) -> bool:
    if not envs.SGLANG_USE_MODELSCOPE.get():
        return False

    from modelscope.hub.errors import NotExistError

    current: Optional[BaseException] = error
    while current is not None:
        if isinstance(current, NotExistError):
            return True
        current = current.__cause__
    return False


def _is_diffusers_component_entry(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(item is None or isinstance(item, str) for item in value)
    )


def _is_weight_bearing_diffusers_component(key: str, value: Any) -> bool:
    if (
        key.startswith("_")
        or not _is_diffusers_component_entry(value)
        or not any(item is not None for item in value)
    ):
        return False

    key_lower = key.lower()
    return not any(hint in key_lower for hint in _NON_WEIGHT_DIFFUSERS_COMPONENT_HINTS)


def _get_declared_weight_component_dirs(model_path: str) -> list[str]:
    model_index_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(model_index_path):
        return []

    try:
        with open(model_index_path) as f:
            model_index = json.load(f)
    except Exception as exc:
        logger.warning(
            "Failed to read model_index.json at %s: %s", model_index_path, exc
        )
        return []

    return [
        key
        for key, value in model_index.items()
        if _is_weight_bearing_diffusers_component(key, value)
    ]


def _has_local_weight_files(component_path: str) -> bool:
    return any(
        glob.glob(os.path.join(component_path, pattern))
        for pattern in _WEIGHT_FILE_PATTERNS
    )


def _get_missing_declared_weight_components(model_path: str) -> list[str]:
    missing_files = []
    for component_dir in _get_declared_weight_component_dirs(model_path):
        component_path = os.path.join(model_path, component_dir)
        if not os.path.isdir(component_path):
            missing_files.append(f"{component_dir}/")
        elif not _has_local_weight_files(component_path):
            missing_files.append(f"{component_dir}/<weights>")
    return missing_files


def _is_metadata_only_pipeline_snapshot(model_path: str) -> bool:
    """Detect a snapshot holding only pipeline metadata, with no component weights.

    ``maybe_download_model_index`` probes a repo by fetching just ``model_index.json``;
    that single-file fetch materializes a full cache entry, so a later
    ``local_files_only`` snapshot resolves it as a hit — offline there is no remote file
    list to tell "cached" from "fully cached", so completeness must be read off disk.

    Requires *all* declared components missing, not any: a partially populated snapshot
    is legitimate (``allow_patterns``-filtered fetch), so only total absence is
    unambiguously the probe stub. No declarations means no evidence to act on.
    """
    declared = _get_declared_weight_component_dirs(model_path)
    if not declared:
        return False
    return len(_get_missing_declared_weight_components(model_path)) == len(declared)


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
    checked_subdir_set = set()

    def _record_checked_subdir(dir_path: str) -> None:
        subdir = os.path.basename(dir_path)
        if not subdir:
            subdir = "."
        if subdir not in checked_subdir_set:
            checked_subdirs.append(subdir)
            checked_subdir_set.add(subdir)

    # Add common subdirectories for diffusers models
    try:
        subdirs = os.listdir(model_path)
    except OSError as e:
        logger.warning("Failed to list model directory %s: %s", model_path, e)
        return True, [], []  # Assume valid if we can't check

    # Check the root directory and all subdirectories that might contain model weights
    dirs_to_check = [model_path]

    for component_dir in _get_declared_weight_component_dirs(model_path):
        _record_checked_subdir(os.path.join(model_path, component_dir))
    missing_files.extend(_get_missing_declared_weight_components(model_path))

    for subdir in subdirs:
        subdir_path = os.path.join(model_path, subdir)
        if os.path.isdir(subdir_path):
            dirs_to_check.append(subdir_path)

    for dir_path in dirs_to_check:
        # Find all safetensors index files
        index_files = glob.glob(os.path.join(dir_path, "*.safetensors.index.json"))

        for index_file in index_files:
            _record_checked_subdir(dir_path)
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
        if _is_diffusers_component_entry(value)
        and any(item is not None for item in value)
    ]
    if component_keys:
        return all(
            os.path.exists(os.path.join(path, key)) for key in component_keys
        ) and not _get_missing_declared_weight_components(path)

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


def _split_hf_subfolder(path: str) -> tuple[str, str | None]:
    """Split 'namespace/repo/subfolder' into (repo_id, subfolder), or return (path, None)."""
    if os.path.isabs(path):
        return path, None
    parts = path.split("/")
    if len(parts) > 2:
        return "/".join(parts[:2]), "/".join(parts[2:])
    return path, None


def prepare_diffusers_component_path_for_loading(component_path: str) -> str:
    """Download component repos if needed and patch legacy flat ModelOpt configs."""
    if os.path.exists(component_path):
        local_component_path = component_path
    else:
        repo_id, subfolder = _split_hf_subfolder(component_path)
        if subfolder is not None:
            # component_path is 'namespace/repo/subfolder' — download only that subfolder
            local_repo = maybe_download_model(
                repo_id,
                allow_patterns=[f"{subfolder}/**", f"{subfolder}/*"],
            )
            local_component_path = os.path.join(local_repo, subfolder)
        else:
            local_component_path = maybe_download_model(component_path)
    config_path = os.path.join(local_component_path, "config.json")
    if not os.path.exists(config_path):
        return local_component_path

    with get_lock(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                config = cast(dict[str, Any], json.load(f))
        except Exception as exc:
            logger.warning("Failed to read component config %s: %s", config_path, exc)
            return local_component_path

        quant_config = config.get("quantization_config")
        normalized_quant_config = normalize_flat_modelopt_quant_config(quant_config)
        if normalized_quant_config == quant_config:
            return local_component_path

        config["quantization_config"] = normalized_quant_config
        try:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, sort_keys=True)
                f.write("\n")
        except OSError as exc:
            logger.warning(
                "Could not persist normalized ModelOpt config at %s (%s); "
                "normalization will be applied in memory at load time.",
                config_path,
                exc,
            )
        else:
            logger.warning(
                "Patched legacy flat ModelOpt quantization_config at %s with quant_type=%s "
                "for diffusers compatibility.",
                config_path,
                normalized_quant_config.get("quant_type"),
            )

    return local_component_path


def get_diffusers_component_config(
    component_path: str,
) -> dict[str, Any]:
    """Gets a configuration of a submodule for the given diffusers model."""
    # Download from HuggingFace Hub if path doesn't exist locally
    component_path = prepare_diffusers_component_path_for_loading(component_path)

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

    quant_config = combined_config.get("quantization_config")
    if quant_config is not None:
        combined_config["quantization_config"] = normalize_flat_modelopt_quant_config(
            quant_config
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


def maybe_download_lora(
    model_name_or_path: str,
    local_dir: str | None = None,
    download: bool = True,
    weight_name: str | None = None,
) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.
    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub
        weight_name: Specific safetensors filename to load (pins deterministic selection
                     for repos with multiple weight files)

    Returns:
        Local path to the model
    """
    # Repositories often publish several adapter revisions side by side.  If a
    # filename is pinned, do not download every weight before selecting it.
    # Keep JSON metadata so PEFT's lora_alpha remains available.
    allow_patterns = (
        ["*.json", weight_name, f"**/{weight_name}"]
        if weight_name is not None
        else ["*.json", "*.safetensors", "*.bin"]
    )

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

    if weight_name is not None:
        target = os.path.join(local_path, weight_name)
        if not os.path.isfile(target):
            raise FileNotFoundError(
                f"Specified lora_weight_name '{weight_name}' not found in {local_path}"
            )
        return target

    guessed = _best_guess_weight_name(local_path, file_extension=".safetensors")
    # AMD workaround: PR 15813 changed from model_name_or_path to local_path,
    # which can return None. Fall back to original behavior on ROCm.
    if guessed is None and current_platform.is_rocm():
        guessed = _best_guess_weight_name(
            model_name_or_path, file_extension=".safetensors"
        )
    return os.path.join(local_path, guessed)


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


def _resolve_remote_repo_model_index_path(model_name_or_path: str) -> str:
    """Return a local path to a remote repo's ``model_index.json``"""
    try:
        # Cache-aware: no local_dir, so the selected Hub reuses its cache and
        # revalidates the remote file when online.
        return hf_hub_download(repo_id=model_name_or_path, filename="model_index.json")
    except EntryNotFoundError:
        # Repo exists but has no model_index.json (single-model repo); let the
        # caller fall through to the single-model path.
        raise
    except Exception as online_err:
        cached_path = None
        if not envs.SGLANG_USE_MODELSCOPE.get():
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                repo_id=model_name_or_path, filename="model_index.json"
            )
            if isinstance(cached, str) and os.path.exists(cached):
                cached_path = cached
        if cached_path is not None:
            logger.warning(
                "Could not fetch model_index.json for '%s' from the Hugging Face "
                "Hub (%s); using the locally cached copy at '%s'. The cached copy "
                "may be out of date — provide an HF token or clear the cache to "
                "force a refresh.",
                model_name_or_path,
                online_err,
                cached_path,
            )
            return cached_path
        raise


def maybe_download_model_index(model_name_or_path: str) -> dict[str, Any]:
    """
    Download and extract just the model_index.json for a Hugging Face model.

    Args:
        model_name_or_path: Path or HF Hub model ID

    Returns:
        The parsed model_index.json as a dictionary
    """
    overlay_config = maybe_load_overlay_model_index(
        model_name_or_path,
        snapshot_download_fn=snapshot_download,
        hf_hub_download_fn=hf_hub_download,
    )
    if overlay_config is not None:
        return overlay_config

    # If it's a local path, verify it directly.
    if os.path.exists(model_name_or_path):
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

    # For remote models, resolve model_index.json (Hub-first, cache fallback).
    try:
        model_index_path = _resolve_remote_repo_model_index_path(model_name_or_path)

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
            "Resolved model_index.json for %s, pipeline: %s",
            model_name_or_path,
            config["_class_name"],
        )
        return config
    except EntryNotFoundError:
        logger.debug(
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
    revision: str | None = None,
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
        revision: Specific Hugging Face Hub revision to resolve
    Returns:
        Local path to the model
    """
    if force_diffusers_model and not skip_overlay_resolution:
        # return overlay model path if applicable
        overlay_model_path = maybe_resolve_overlay_model_path(
            model_name_or_path,
            local_dir=local_dir,
            download=download,
            allow_patterns=allow_patterns,
            snapshot_download_fn=snapshot_download,
            hf_hub_download_fn=hf_hub_download,
            verify_diffusers_model_complete_fn=_verify_diffusers_model_complete,
            base_model_download_fn=maybe_download_model,
        )
        if overlay_model_path is not None:
            return overlay_model_path

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
            "Checking for cached model in %s cache for %s...",
            _model_hub_name(),
            model_name_or_path,
        )
        local_path = snapshot_download(
            repo_id=model_name_or_path,
            ignore_patterns=["*.onnx", "*.msgpack"],
            allow_patterns=allow_patterns,
            local_dir=local_dir,
            local_files_only=True,
            max_workers=8,
            revision=revision,
        )
        if _is_revisionless_snapshot_root(local_path):
            # A cache miss, so the download below re-resolves and rewrites the ref.
            raise LocalEntryNotFoundError(
                f"Cached ref for {model_name_or_path} is corrupt: resolved to the "
                f"snapshots parent {local_path!r} instead of a revision directory."
            )
        if not force_diffusers_model:
            # maybe_download_model_index's model_index.json fetch materializes a full
            # cache entry, so this resolve reports that stub as a hit; returning it
            # would skip the download. LoRA repos declare no components.
            if not is_lora and _is_metadata_only_pipeline_snapshot(local_path):
                if not download:
                    raise ValueError(
                        f"Model {model_name_or_path} found in cache but only contains "
                        "pipeline metadata (no component weights) and download=False."
                    )
                logger.info(
                    "Cached snapshot for %s only contains pipeline metadata, "
                    "will download component weights from %s",
                    model_name_or_path,
                    _model_hub_name(),
                )
            else:
                return str(local_path)
        elif is_lora or _verify_diffusers_model_complete(local_path):
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
                "Model found in cache but incomplete, will download from %s",
                _model_hub_name(),
            )
    except LocalEntryNotFoundError:
        if not download:
            raise ValueError(
                f"Model {model_name_or_path} not found in local cache and download=False."
            )
        logger.info(
            "Model not found in cache, will download from %s", _model_hub_name()
        )
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
                "Downloading model snapshot from %s for %s (attempt %d/%d)...",
                _model_hub_name(),
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
                    revision=revision,
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
                        allow_patterns=allow_patterns,
                        local_dir=local_dir,
                        max_workers=8,
                        force_download=True,
                        revision=revision,
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
        except (RequestException, RequestsConnectionError, ConnectionError) as e:
            if _is_modelscope_not_found_error(e):
                raise ValueError(
                    f"Model or revision not found at {model_name_or_path}. "
                    "Please check the model ID or ensure you have access to the repository."
                ) from e
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"Could not find model at {model_name_or_path} and failed to download from {_model_hub_name()} "
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
        except RuntimeError as e:
            if "client has been closed" not in str(e).lower():
                raise ValueError(
                    f"Could not find model at {model_name_or_path} and failed to download from {_model_hub_name()}: {e}"
                ) from e
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"Could not find model at {model_name_or_path} and failed to download from {_model_hub_name()} "
                    f"after {MAX_RETRIES} attempts due to network error: {e}"
                ) from e
            from huggingface_hub.utils._http import close_session

            close_session()
            wait_time = 2**attempt
            logger.warning(
                "Download failed (attempt %d/%d) because the Hugging Face client was closed. "
                "Retrying in %d seconds...",
                attempt + 1,
                MAX_RETRIES,
                wait_time,
            )
            time.sleep(wait_time)
        except Exception as e:
            raise ValueError(
                f"Could not find model at {model_name_or_path} and failed to download from {_model_hub_name()}: {e}"
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
        from modelscope.hub.errors import NotExistError

        try:
            return model_file_download(
                model_id=repo_id,
                file_path=filename,
                local_dir=str(local_dir) if local_dir is not None else None,
                **kwargs,
            )
        except NotExistError as exc:
            # Keep the Hugging Face-compatible exception contract used by
            # maybe_download_model_index for repositories without model_index.json.
            raise EntryNotFoundError(str(exc)) from exc
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

        # ModelScope validates cached files on every online snapshot request and
        # has no force_download argument. Dropping it preserves the caller's
        # intended online revalidation without leaking Hub-specific kwargs.
        kwargs.pop("force_download", None)
        ms_kwargs = {
            "model_id": repo_id,
            "local_dir": str(local_dir) if local_dir is not None else None,
            "ignore_patterns": ignore_patterns,
            "allow_patterns": allow_patterns,
            "local_files_only": local_files_only,
            "max_workers": max_workers,
        }
        ms_kwargs.update(kwargs)
        local_path = _ms_snapshot_download(**ms_kwargs)
        if local_files_only and not _snapshot_has_files(local_path, allow_patterns):
            raise LocalEntryNotFoundError(
                f"No cached files for {repo_id} match {allow_patterns or '**/*'}"
            )
        return local_path
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
