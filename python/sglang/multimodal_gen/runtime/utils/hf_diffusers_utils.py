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
import json
import os
import time
from functools import reduce
from pathlib import Path
from typing import Any, Optional, cast

from diffusers.loaders.lora_base import (
    _best_guess_weight_name,  # watch out for potetential removal from diffusers
)
from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import RequestException
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.multimodal_gen.runtime.loader.weight_utils import get_lock
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)
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
    is_gguf = check_gguf_file(component_model_path)
    if is_gguf:
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

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config


def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    try:
        config = AutoConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except ValueError as e:
        raise e

    return config


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
    model_path: str,
) -> dict[str, Any]:
    """Gets a configuration of a submodule for the given diffusers model.

    Args:
        model_path: the path of the submodule (can be local path or HuggingFace model ID)

    Returns:
        The loaded configuration.
    """

    # Download from HuggingFace Hub if path doesn't exist locally
    if not os.path.exists(model_path):
        model_path = maybe_download_model(model_path)

    # tokenizer
    config_names = ["generation_config.json"]
    # By default, we load config.json, but scheduler_config.json for scheduler
    if "scheduler" in model_path:
        config_names.append("scheduler_config.json")
    else:
        config_names.append("config.json")

    config_file_paths = [
        os.path.join(model_path, config_name) for config_name in config_names
    ]

    combined_config = reduce(
        lambda acc, path: acc | load_dict(path), config_file_paths, {}
    )

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
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
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

    # Check for transformer and vae directories
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

    # Load the config
    with open(config_path) as f:
        config = json.load(f)

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        raise ValueError("model_index.json does not contain _diffusers_version")

    logger.info("Diffusers version: %s", config["_diffusers_version"])
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

    from huggingface_hub import hf_hub_download
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

            logger.info(
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
) -> str:
    """
    Check if the model path is a Hugging Face Hub model ID and download it if needed.

    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the model
        download: Whether to download the model from Hugging Face Hub
        is_lora: If True, skip model completeness verification (LoRA models don't have transformer/vae directories)

    Returns:
        Local path to the model
    """

    def _verify_model_complete(path: str) -> bool:
        """Check if model directory has required subdirectories."""
        transformer_dir = os.path.join(path, "transformer")
        vae_dir = os.path.join(path, "vae")
        config_path = os.path.join(path, "model_index.json")
        return (
            os.path.exists(config_path)
            and os.path.exists(transformer_dir)
            and os.path.exists(vae_dir)
        )

    # 1. Local path check: if path exists locally, verify it's complete (skip for LoRA)
    if os.path.exists(model_name_or_path):
        if is_lora or _verify_model_complete(model_name_or_path):
            logger.info("Model already exists locally and is complete")
            return model_name_or_path
        else:
            logger.warning(
                "Local model at %s appears incomplete (missing transformer/ or vae/), "
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
            resume_download=True,
            max_workers=8,
            etag_timeout=60,
        )
        if is_lora or _verify_model_complete(local_path):
            logger.info("Found complete model in cache at %s", local_path)
            return str(local_path)
        else:
            # Model found in cache but incomplete
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
                    resume_download=True,
                    max_workers=8,
                    etag_timeout=120,
                )

            # Verify downloaded model is complete (skip for LoRA)
            if not is_lora and not _verify_model_complete(local_path):
                logger.warning(
                    "Downloaded model at %s is incomplete, retrying with force_download=True",
                    local_path,
                )
                with get_lock(model_name_or_path).acquire(poll_interval=2):
                    local_path = snapshot_download(
                        repo_id=model_name_or_path,
                        ignore_patterns=["*.onnx", "*.msgpack"],
                        local_dir=local_dir,
                        resume_download=True,
                        max_workers=8,
                        etag_timeout=60,
                        force_download=True,
                    )
                if not _verify_model_complete(local_path):
                    raise ValueError(
                        f"Downloaded model at {local_path} is still incomplete after forced re-download. "
                        "The model repository may be missing required components (model_index.json, transformer/, or vae/)."
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
