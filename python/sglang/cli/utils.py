import json
import logging
import os
import subprocess
from functools import lru_cache

from huggingface_hub import HfApi

from sglang.srt.environ import envs
from sglang.utils import (
    has_diffusion_overlay_registry_match,
    is_known_non_diffusers_diffusion_model,
    load_diffusion_overlay_registry_from_env,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_overlay_registry() -> dict:
    return load_diffusion_overlay_registry_from_env()


def _is_overlay_diffusion_model(model_path: str) -> bool:
    return has_diffusion_overlay_registry_match(model_path, _load_overlay_registry())


def _is_registered_diffusion_model(model_path: str) -> bool:
    try:
        from sglang.multimodal_gen.registry import has_registered_diffusion_model_path
    except ImportError:
        # if diffusion dependencies are not installed
        return False

    return has_registered_diffusion_model_path(model_path)


def _is_diffusers_model_dir(model_dir: str) -> bool:
    """Check if a local directory contains a valid diffusers model_index.json."""
    config_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path) as f:
        config = json.load(f)

    return "_diffusers_version" in config


def _is_gated_diffusion_repo(repo_id: str) -> bool:
    """Query HF model card metadata to check if a gated repo is a diffusers model."""
    try:
        info = HfApi().model_info(repo_id)
        return getattr(info, "library_name", None) == "diffusers"
    except Exception:
        return False


def get_is_diffusion_model(model_path: str) -> bool:
    """Detect whether model_path points to a diffusion model.

    For local directories, checks the filesystem directly.
    For HF/ModelScope model IDs, attempts to fetch only model_index.json.
    For gated repos where file download fails, falls back to HF model card
    metadata (library_name == "diffusers").
    Returns False on any failure (network error, 404, offline mode, etc.)
    so that the caller falls through to the standard LLM server path.
    """
    if _is_overlay_diffusion_model(model_path):
        # short-circuit, if applicable for the overlay mechanism (diffusion-only)
        return True

    if os.path.isdir(model_path):
        if _is_diffusers_model_dir(model_path):
            return True
        return is_known_non_diffusers_diffusion_model(model_path)

    if is_known_non_diffusers_diffusion_model(model_path):
        return True

    if _is_registered_diffusion_model(model_path):
        return True

    try:
        if envs.SGLANG_USE_MODELSCOPE.get():
            from modelscope import model_file_download

            file_path = model_file_download(
                model_id=model_path, file_path="model_index.json"
            )
        else:
            from huggingface_hub import hf_hub_download

            file_path = hf_hub_download(repo_id=model_path, filename="model_index.json")

        return _is_diffusers_model_dir(os.path.dirname(file_path))
    except Exception as e:
        logger.debug("Failed to auto-detect diffusion model for %s: %s", model_path, e)
        return False


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
                "The server type is determined by the --model-path.\n"
            )
        else:
            raise Exception(
                "Error: --model-path is required. "
                "Please provide the path to the model."
            )
    return model_path


@lru_cache(maxsize=1)
def get_git_commit_hash() -> str:
    try:
        commit_hash = os.environ.get("SGLANG_GIT_COMMIT")
        if not commit_hash:
            commit_hash = (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
                )
                .strip()
                .decode("utf-8")
            )
        _CACHED_COMMIT_HASH = commit_hash
        return commit_hash
    except (subprocess.CalledProcessError, FileNotFoundError):
        _CACHED_COMMIT_HASH = "N/A"
        return "N/A"
