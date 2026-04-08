import glob
import json
import os
import re

import safetensors
import torch
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
    LatentUpsampler,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

UPSAMPLER_CONSTRUCTOR_KEYS = {
    "in_channels",
    "mid_channels",
    "num_blocks_per_stage",
    "dims",
    "spatial_upsample",
    "temporal_upsample",
    "spatial_scale",
    "rational_resampler",
}

_HF_BLOB_URL_RE = re.compile(
    r"https?://huggingface\.co/([^/]+/[^/]+)/blob/([^/]+)/(.*)"
)
_HF_RESOLVE_URL_RE = re.compile(
    r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.*)"
)


def _parse_hf_url(path: str):
    m = _HF_BLOB_URL_RE.match(path) or _HF_RESOLVE_URL_RE.match(path)
    if m:
        return m.group(1), m.group(2), m.group(3)
    return None


def _download_hf_file(repo_id: str, filename: str, revision: str = "main") -> str:
    from huggingface_hub import hf_hub_download

    logger.info("Downloading %s from %s (revision=%s)", filename, repo_id, revision)
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)


def _find_safetensors_file(path: str) -> str:
    """Resolve path to a single safetensors file (local path, directory, HF URL, or HF repo id)."""
    if os.path.isfile(path) and path.endswith(".safetensors"):
        return path

    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
        if len(files) == 1:
            return files[0]
        elif len(files) > 1:
            raise ValueError(
                f"Found {len(files)} safetensors files in {path}, expected 1"
            )

    hf = _parse_hf_url(path)
    if hf:
        repo_id, revision, filename = hf
        return _download_hf_file(repo_id, filename, revision)

    try:
        maybe_downloaded = maybe_download_model(path)
        if os.path.isdir(maybe_downloaded):
            files = sorted(glob.glob(os.path.join(maybe_downloaded, "*.safetensors")))
            if len(files) == 1:
                return files[0]
            elif len(files) > 1:
                raise ValueError(
                    f"Found {len(files)} safetensors files in {maybe_downloaded}, expected 1"
                )
    except Exception:
        pass

    raise FileNotFoundError(
        f"No safetensors file found at {path}. "
        "Provide a local .safetensors file, a directory containing one, "
        "a HuggingFace URL (https://huggingface.co/<repo>/blob/main/<path>), "
        "or a HuggingFace repo id."
    )


def _normalize_config(raw: dict) -> dict:
    """Map diffusers / original-repo config fields to LatentUpsampler kwargs."""
    config = {k: v for k, v in raw.items() if k in UPSAMPLER_CONSTRUCTOR_KEYS}

    # diffusers uses rational_spatial_scale instead of rational_resampler + spatial_scale
    if "rational_spatial_scale" in raw and "rational_resampler" not in config:
        config["rational_resampler"] = True
        config.setdefault("spatial_scale", raw["rational_spatial_scale"])

    return config


def _infer_config_from_state_dict(state_dict: dict[str, torch.Tensor]) -> dict:
    """Infer LatentUpsampler kwargs from weight shapes and key names.

    Works even when no config.json or safetensors metadata is available.
    """
    config: dict = {}

    w = state_dict.get("initial_conv.weight")
    if w is not None:
        config["mid_channels"] = w.shape[0]
        config["in_channels"] = w.shape[1]
        config["dims"] = 3 if w.ndim == 5 else 2

    num_blocks = sum(
        1
        for k in state_dict
        if k.startswith("res_blocks.") and k.endswith(".conv1.weight")
    )
    if num_blocks > 0:
        config["num_blocks_per_stage"] = num_blocks

    # Detect upsampler type from key patterns
    has_rational = any(k.startswith("upsampler.blur_down.") for k in state_dict)
    if has_rational:
        config["rational_resampler"] = True
        config["spatial_upsample"] = True
        config["temporal_upsample"] = False
        config["spatial_scale"] = 2.0
    else:
        up_w = state_dict.get("upsampler.0.weight")
        if up_w is not None and up_w.ndim == 5:
            ratio = up_w.shape[0] // up_w.shape[1]
            if ratio == 8:
                config["spatial_upsample"] = True
                config["temporal_upsample"] = True
            elif ratio == 2:
                config["spatial_upsample"] = False
                config["temporal_upsample"] = True
            else:
                config["spatial_upsample"] = True
                config["temporal_upsample"] = False
        else:
            config["spatial_upsample"] = True
            config["temporal_upsample"] = False

    return config


def _load_config(
    safetensors_path: str,
    original_path: str,
    state_dict: dict[str, torch.Tensor],
) -> dict:
    """Load upsampler config with fallback chain:
    1. safetensors metadata ("config" key) - original LTX-2 repo format
    2. sibling config.json - diffusers format
    3. config.json from HF (if original_path was a URL)
    4. infer from state dict shapes (always works)
    """
    with safetensors.safe_open(safetensors_path, framework="pt") as f:
        meta = f.metadata()
        if meta and "config" in meta:
            logger.info("Using config from safetensors metadata")
            return _normalize_config(json.loads(meta["config"]))

    config_json_path = os.path.join(os.path.dirname(safetensors_path), "config.json")
    if os.path.isfile(config_json_path):
        with open(config_json_path) as fp:
            logger.info("Using config from sibling config.json")
            return _normalize_config(json.load(fp))

    hf = _parse_hf_url(original_path)
    if hf:
        repo_id, revision, filename = hf
        config_filename = os.path.dirname(filename) + "/config.json"
        try:
            local = _download_hf_file(repo_id, config_filename, revision)
            with open(local) as fp:
                logger.info("Using config from HF config.json")
                return _normalize_config(json.load(fp))
        except Exception:
            pass

    logger.info("No explicit config found, inferring from state dict")
    return _infer_config_from_state_dict(state_dict)


class UpsamplerLoader(ComponentLoader):
    component_names = ["spatial_upsampler"]
    expected_library = "diffusers"

    def should_offload(self, server_args: ServerArgs, model_config=None):
        return server_args.vae_cpu_offload

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
    ):
        safetensors_path = _find_safetensors_file(component_model_path)
        state_dict = safetensors_load_file(safetensors_path)
        config = _load_config(safetensors_path, component_model_path, state_dict)

        logger.info("Loading LatentUpsampler with config: %s", config)

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        with torch.device("meta"):
            model = LatentUpsampler(**config)

        model.load_state_dict(state_dict, assign=True)
        model = model.to(device=target_device, dtype=torch.bfloat16).eval()

        logger.info("Loaded LatentUpsampler to %s", target_device)
        return model
