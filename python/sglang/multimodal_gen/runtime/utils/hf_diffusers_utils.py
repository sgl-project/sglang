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
from typing import Any, Dict, List, Optional, Union, cast

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
from safetensors import safe_open
from transformers import AutoConfig, PretrainedConfig

from sglang.multimodal_gen.runtime.layers.quantization import (
    QuantizationConfig,
    get_quantization_config,
)
from sglang.multimodal_gen.runtime.loader.utils import _clean_hf_config_inplace
from sglang.multimodal_gen.runtime.loader.weight_utils import get_lock
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.environ import envs
from sglang.utils import is_in_ci

logger = init_logger(__name__)


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


def replace_prefix(key: str, prefix_mapping: dict[str, str]) -> str:
    for prefix, new_prefix in prefix_mapping.items():
        if key.startswith(prefix):
            key = key.replace(prefix, new_prefix, 1)
    return key


def get_quant_config(
    model_config,
    packed_modules_mapping: Dict[str, List[str]] = {},
    remap_prefix: Dict[str, str] | None = None,
) -> QuantizationConfig:
    if "quantization_config" not in model_config:
        return None
    quant_cls = get_quantization_config(
        model_config["quantization_config"]["quant_method"]
    )

    # GGUF doesn't have config file
    if model_config["quantization_config"]["quant_method"] == "gguf":
        return quant_cls.from_config({})

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = model_config["quantization_config"]
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config, "compression_config", None)
    if hf_quant_config is not None:
        hf_quant_config["packed_modules_mapping"] = packed_modules_mapping
        return quant_cls.from_config(hf_quant_config)
    # In case of bitsandbytes/QLoRA, get quant config from the adapter model.
    else:
        model_name_or_path = model_config["model_path"]
    is_local = os.path.isdir(model_name_or_path)
    hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(
            f"Cannot find the config file for {model_config['quantization_config']['quant_method']}"
        )
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config['quantization_config']['quant_method']}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)
        if remap_prefix is not None:
            exclude_modules = [
                replace_prefix(key, remap_prefix)
                for key in config["quantization"]["exclude_modules"]
            ]
            config["quantization"]["exclude_modules"] = exclude_modules
        config["packed_modules_mapping"] = packed_modules_mapping
        return quant_cls.from_config(config)


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


def get_metadata_from_safetensors_file(file_path: str):
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            return metadata
    except Exception as e:
        logger.warning(e)


def get_quant_config_from_safetensors_metadata(
    file_path: str,
) -> Optional[QuantizationConfig]:
    """Extract quantization config from a safetensors file's metadata header.
    Returns None if no recognizable quantization metadata is found.
    """
    metadata = get_metadata_from_safetensors_file(file_path)
    if not metadata:
        return None

    quant_config_str = metadata.get("_quantization_metadata")
    if not quant_config_str:
        return None
    try:
        quant_config_dict = json.loads(quant_config_str)
    except Exception as _e:
        return None

    # handle diffusers fp8 safetensors metadata format
    if (
        "quant_method" not in quant_config_dict
        and "format_version" in quant_config_dict
        and "layers" in quant_config_dict
    ):
        layers = quant_config_dict.get("layers", {})
        if any(
            isinstance(v, dict) and "float8" in v.get("format", "")
            for v in layers.values()
        ):
            quant_config_dict["quant_method"] = "fp8"
            quant_config_dict["activation_scheme"] = "dynamic"

    quant_method = quant_config_dict.get("quant_method")
    if not quant_method:
        return None

    try:
        quant_cls = get_quantization_config(quant_method)
        config = quant_cls.from_config(quant_config_dict)
        logger.debug(f"Get quantization config from safetensors file: {file_path}")
        return config
    except Exception as _e:
        return None
