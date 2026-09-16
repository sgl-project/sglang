import importlib
import json
import logging
import os
import subprocess
import sys
from functools import lru_cache

from huggingface_hub import HfApi, ModelCard

from sglang.srt.environ import envs
from sglang.utils import (
    has_diffusion_overlay_registry_match,
    load_diffusion_overlay_registry_from_env,
)

logger = logging.getLogger(__name__)

_DIFFUSION_OUTPUT_MODALITIES = frozenset({"3d", "audio", "image", "video"})


@lru_cache(maxsize=1)
def _load_overlay_registry() -> dict:
    return load_diffusion_overlay_registry_from_env()


def _is_overlay_diffusion_model(model_path: str) -> bool:
    return has_diffusion_overlay_registry_match(model_path, _load_overlay_registry())


def _is_registered_diffusion_model(model_path: str) -> bool:
    registry = sys.modules.get("sglang.multimodal_gen.registry")
    if registry is None and os.environ.get("SGLANG_EXTERNAL_MODEL_PACKAGE"):
        try:
            registry = importlib.import_module("sglang.multimodal_gen.registry")
        except ImportError:
            return False
    if registry is None:
        return False
    return registry.is_registered_diffusion_model_path(model_path)


def _is_diffusers_model_dir(model_dir: str) -> bool:
    """Check if a local directory contains a valid diffusers model_index.json."""
    config_path = os.path.join(model_dir, "model_index.json")
    if not os.path.exists(config_path):
        return False

    with open(config_path) as f:
        config = json.load(f)

    return "_diffusers_version" in config


def _metadata_marks_diffusion(metadata) -> bool:
    library_name = (getattr(metadata, "library_name", None) or "").lower()
    tags = {str(tag).lower() for tag in (getattr(metadata, "tags", None) or [])}
    if library_name == "diffusers" or "diffusers" in tags:
        return True

    pipeline_tag = (getattr(metadata, "pipeline_tag", None) or "").lower()
    output = pipeline_tag.rsplit("-to-", 1)[-1]
    return any(
        modality in output.split("-") for modality in _DIFFUSION_OUTPUT_MODALITIES
    ) and ("-to-" in pipeline_tag or pipeline_tag.endswith("-generation"))


def _is_diffusion_model_from_local_metadata(model_dir: str) -> bool:
    readme_path = os.path.join(model_dir, "README.md")
    if not os.path.isfile(readme_path):
        return False
    try:
        return _metadata_marks_diffusion(ModelCard.load(readme_path).data)
    except Exception:
        return False


def _is_diffusion_model_from_hub_metadata(repo_id: str) -> bool:
    """Query model-card metadata without importing the diffusion runtime."""
    try:
        return _metadata_marks_diffusion(HfApi().model_info(repo_id))
    except Exception:
        return False


def get_is_diffusion_model(model_path: str) -> bool:
    """Detect whether model_path points to a diffusion model.

    Local directories are checked for Diffusers config or model-card metadata.
    Remote models are detected from model_index.json or Hub metadata. The
    diffusion registry is consulted only when already loaded or explicitly
    configured, so auto-detecting an LLM does not import diffusion operators.
    Returns False on any failure (network error, 404, offline mode, etc.)
    so that the caller falls through to the standard LLM server path.
    """
    if _is_overlay_diffusion_model(model_path):
        # short-circuit, if applicable for the overlay mechanism (diffusion-only)
        return True

    if os.path.isdir(model_path):
        if _is_diffusers_model_dir(
            model_path
        ) or _is_diffusion_model_from_local_metadata(model_path):
            return True
        return _is_registered_diffusion_model(model_path)

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

        if _is_diffusers_model_dir(os.path.dirname(file_path)):
            return True
    except Exception as e:
        logger.debug("Failed to auto-detect diffusion model for %s: %s", model_path, e)

    if envs.SGLANG_USE_MODELSCOPE.get():
        return False
    return _is_diffusion_model_from_hub_metadata(model_path)


def try_get_model_path(extra_argv) -> str | None:
    """Return a model path from command-line arguments when one is present."""

    model_path = None
    for i, arg in enumerate(extra_argv):
        if arg in ("--model-path", "--model"):
            if i + 1 < len(extra_argv):
                model_path = extra_argv[i + 1]
                break
        elif arg.startswith("--model-path=") or arg.startswith("--model="):
            model_path = arg.split("=", 1)[1]
            break

    return model_path


def get_model_path(extra_argv):
    # Find the model_path argument
    model_path = try_get_model_path(extra_argv)

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
                "Error: --model-path is required. Please provide the path to the model."
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
