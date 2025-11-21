import hashlib
import json
import logging
import os
import tempfile
from typing import Optional

import filelock
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

temp_dir = tempfile.gettempdir()


def _get_lock(model_name_or_path: str, cache_dir: Optional[str] = None):
    lock_dir = cache_dir or temp_dir
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    lock_file_name = hash_name + model_name + ".lock"
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


# Copied and adapted from hf_diffusers_utils.py
def _maybe_download_model(
    model_name_or_path: str, local_dir: str | None = None, download: bool = True
) -> str:
    """
    Resolve a model path. If it's a local directory, return it.
    If it's a Hugging Face Hub ID, download only the config file
    (`model_index.json` or `config.json`) and return its directory.

    Args:
        model_name_or_path: Local path or Hugging Face Hub model ID
        local_dir: Local directory to save the downloaded file (if any)
        download: Whether to download from Hugging Face Hub when needed

    Returns:
        Local directory path that contains the downloaded config file, or the original local directory.
    """

    if os.path.exists(model_name_or_path):
        logger.info("Model already exists locally")
        return model_name_or_path

    if not download:
        return model_name_or_path

    with _get_lock(model_name_or_path):
        # Try `model_index.json` first (diffusers models)
        try:
            logger.info(
                "Downloading model_index.json from HF Hub for %s...",
                model_name_or_path,
            )
            file_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="model_index.json",
                local_dir=local_dir,
            )
            logger.info("Downloaded to %s", file_path)
            return os.path.dirname(file_path)
        except Exception as e_index:
            logger.debug("model_index.json not found or failed: %s", e_index)

        # Fallback to `config.json`
        try:
            logger.info(
                "Downloading config.json from HF Hub for %s...", model_name_or_path
            )
            file_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="config.json",
                local_dir=local_dir,
            )
            logger.info("Downloaded to %s", file_path)
            return os.path.dirname(file_path)
        except Exception as e_config:
            raise ValueError(
                (
                    "Could not find model locally at %s and failed to download "
                    "model_index.json/config.json from HF Hub: %s"
                )
                % (model_name_or_path, e_config)
            ) from e_config


# Copied and adapted from hf_diffusers_utils.py
def is_diffusers_model_path(model_path: str) -> True:
    """
    Verify if the model directory contains a valid diffusers configuration.

    Args:
        model_path: Path to the model directory

    Returns:
        The loaded model configuration as a dictionary if the model is a diffusers model
        None if the model is not a diffusers model
    """

    # Prefer model_index.json which indicates a diffusers pipeline
    config_path = os.path.join(model_path, "model_index.json")
    if not os.path.exists(config_path):
        return False

    # Load the config
    with open(config_path) as f:
        config = json.load(f)

    # Verify diffusers version exists
    if "_diffusers_version" not in config:
        return False
    return True


def get_is_diffusion_model(model_path: str):
    model_path = _maybe_download_model(model_path)
    is_diffusion_model = is_diffusers_model_path(model_path)
    if is_diffusion_model:
        logger.info("Diffusion model detected")
    return is_diffusion_model


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg == "--model-path":
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path="):
            model_path = arg.split("=", 1)[1]
            break

    if model_path is None:
        # Fallback for --help or other cases where model-path is not provided
        if any(h in extra_argv for h in ["-h", "--help"]):
            raise Exception(
                "Usage: sglang serve --model-path <model-name-or-path> [additional-arguments]\n\n"
                "This command can launch either a standard language model server or a diffusion model server.\n"
                "The server type is determined by the model path.\n"
                "For specific arguments, please provide a model_path."
            )
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path
