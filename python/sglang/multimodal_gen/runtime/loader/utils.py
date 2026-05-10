# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""

import contextlib
import glob
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Dict, Type

import torch
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_QUANTIZED_DTYPES = {
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
}


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)


def get_param_names_mapping(
    mapping_dict: dict[str, str | tuple[str, int, int]],
) -> Callable[[str], tuple[str, Any, Any]]:
    """
    Creates a mapping function that transforms parameter names using regex patterns.

    Args:
        mapping_dict (Dict[str, str]): Dictionary mapping regex patterns to replacement patterns

    Returns:
        Callable[[str], str]: A function that maps parameter names from source to target format
    """

    def mapping_fn(name: str) -> tuple[str, Any, Any]:
        # support chained conversions, e.g.:
        # transformer.xxx.lora_down -> xxx.lora_down -> xxx.proj_down
        merge_index = None
        total_split_params = None
        max_steps = max(8, len(mapping_dict) * 2)
        applied_patterns: set[str] = set()
        visited_names: set[str] = {name}

        for _ in range(max_steps):
            transformed = False
            for pattern, replacement in mapping_dict.items():
                # avoid re-applying the same rule on its own output
                if pattern in applied_patterns:
                    continue
                if re.match(pattern, name) is None:
                    continue

                curr_merge_index = None
                curr_total_split_params = None
                if isinstance(replacement, tuple):
                    curr_merge_index = replacement[1]
                    curr_total_split_params = replacement[2]
                    replacement = replacement[0]

                new_name = re.sub(pattern, replacement, name)

                if new_name != name:
                    if curr_merge_index is not None:
                        merge_index = curr_merge_index
                        total_split_params = curr_total_split_params

                    name = new_name
                    applied_patterns.add(pattern)
                    if name in visited_names:
                        transformed = False
                        break
                    visited_names.add(name)
                    transformed = True
                    break

            if not transformed:
                break

        return name, merge_index, total_split_params

    return mapping_fn


def hf_to_custom_state_dict(
    hf_param_sd: dict[str, torch.Tensor] | Iterator[tuple[str, torch.Tensor]],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
    valid_target_names: set[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[str, Any, Any]]]:
    """
    Converts a Hugging Face parameter state dictionary to a custom parameter state dictionary.

    Args:
        hf_param_sd (Dict[str, torch.Tensor]): The Hugging Face parameter state dictionary
        param_names_mapping (Callable[[str], tuple[str, Any, Any]]): A function that maps parameter names from source to target format

    Returns:
        custom_param_sd (Dict[str, torch.Tensor]): The custom formatted parameter state dict
        reverse_param_names_mapping (Dict[str, Tuple[str, Any, Any]]): Maps back from custom to hf
    """
    custom_param_sd = {}
    to_merge_params = defaultdict(dict)  # type: ignore
    reverse_param_names_mapping = {}
    if isinstance(hf_param_sd, dict):
        hf_param_sd = hf_param_sd.items()  # type: ignore
    for source_param_name, full_tensor in hf_param_sd:  # type: ignore
        target_param_name, merge_index, num_params_to_merge = param_names_mapping(
            source_param_name
        )
        if (
            valid_target_names is not None
            and target_param_name != source_param_name
            and source_param_name in valid_target_names
            and target_param_name not in valid_target_names
        ):
            target_param_name = source_param_name
            merge_index = None
            num_params_to_merge = None
        if target_param_name == "" or target_param_name is None:  # type: ignore[comparison-overlap]
            continue
        reverse_param_names_mapping[target_param_name] = (
            source_param_name,
            merge_index,
            num_params_to_merge,
        )
        if merge_index is not None:
            to_merge_params[target_param_name][merge_index] = full_tensor
            if len(to_merge_params[target_param_name]) == num_params_to_merge:
                # cat at output dim according to the merge_index order
                sorted_tensors = [
                    to_merge_params[target_param_name][i]
                    for i in range(num_params_to_merge)
                ]
                full_tensor = torch.cat(sorted_tensors, dim=0)
                del to_merge_params[target_param_name]
            else:
                continue
        existing_tensor = custom_param_sd.get(target_param_name)
        if existing_tensor is not None and existing_tensor.dtype != full_tensor.dtype:
            existing_is_quantized = existing_tensor.dtype in _QUANTIZED_DTYPES
            current_is_quantized = full_tensor.dtype in _QUANTIZED_DTYPES
            if existing_is_quantized and not current_is_quantized:
                logger.debug(
                    "Keeping quantized duplicate for %s: existing=%s new=%s",
                    target_param_name,
                    existing_tensor.dtype,
                    full_tensor.dtype,
                )
                continue
            if current_is_quantized and not existing_is_quantized:
                logger.debug(
                    "Replacing non-quantized duplicate for %s: existing=%s new=%s",
                    target_param_name,
                    existing_tensor.dtype,
                    full_tensor.dtype,
                )
        custom_param_sd[target_param_name] = full_tensor
    return custom_param_sd, reverse_param_names_mapping


class skip_init_modules:
    def __enter__(self):
        # Save originals
        self._orig_reset = {}
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None  # skip init

    def __exit__(self, exc_type, exc_value, traceback):
        # restore originals
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig


def _normalize_component_type(module_type: str) -> str:
    """Normalize module types like 'text_encoder_2' -> 'text_encoder'."""
    return re.sub(r"_\d+$", "", module_type)


def _clean_hf_config_inplace(model_config: dict) -> None:
    """Remove common extraneous HF fields if present."""
    for key in (
        "_name_or_path",
        "transformers_version",
        "model_type",
        "tokenizer_class",
        "torch_dtype",
    ):
        model_config.pop(key, None)


def _try_redownload_missing_shards(model_path: str, missing: list[str]) -> bool:
    """Try to re-download missing safetensors shards from HuggingFace Hub.

    Parses the repo_id and revision from the HF cache path structure
    (models--{org}--{repo}/snapshots/{revision}) and calls hf_hub_download
    for each missing shard. Returns True if all shards were recovered.
    """
    try:
        from huggingface_hub import hf_hub_download

        match = re.search(
            r"models--([^/\\]+)--([^/\\]+)[/\\]snapshots[/\\]([^/\\]+)", model_path
        )
        if not match:
            return False

        repo_id = f"{match.group(1)}/{match.group(2)}"
        revision = match.group(3)
        logger.warning(
            "Incomplete checkpoint for %s (revision %.8s) — missing shards: %s. "
            "Attempting auto-repair via HuggingFace Hub...",
            repo_id,
            revision,
            missing,
        )
        for shard in missing:
            hf_hub_download(repo_id=repo_id, filename=shard, revision=revision)
        logger.info("Auto-repair succeeded for %s.", repo_id)
        return True
    except Exception as e:
        logger.warning("Auto-repair failed: %s", e)
        return False


def _list_safetensors_files(model_path: str) -> list[str]:
    """List all .safetensors files under a directory.

    If a safetensors index file is present, verifies that every shard listed
    in the index actually exists on disk. Missing shards are first repaired
    automatically via HuggingFace Hub (if the path is an HF cache entry);
    if repair fails a clear RuntimeError is raised.
    """
    found = sorted(glob.glob(os.path.join(str(model_path), "*.safetensors")))

    index_path = os.path.join(
        str(model_path), "diffusion_pytorch_model.safetensors.index.json"
    )
    if os.path.exists(index_path):
        import json

        with open(index_path) as f:
            index = json.load(f)
        expected_shards = sorted(set(index.get("weight_map", {}).values()))
        found_basenames = {os.path.basename(p) for p in found}
        missing = [s for s in expected_shards if s not in found_basenames]
        if missing:
            repaired = _try_redownload_missing_shards(model_path, missing)
            if repaired:
                found = sorted(
                    glob.glob(os.path.join(str(model_path), "*.safetensors"))
                )
            else:
                raise RuntimeError(
                    f"Checkpoint at '{model_path}' is incomplete — the following "
                    f"shard(s) listed in the index are missing from disk: "
                    f"{missing}. Re-download the checkpoint (e.g. "
                    f"`huggingface-cli download {os.path.basename(model_path)}`)."
                )

    return found


BYTES_PER_GB = 1024**3


def get_memory_usage_of_component(module) -> float | None:
    """
    returned value is in GB, rounded to 2 decimal digits
    """
    if not isinstance(module, nn.Module):
        return None
    if hasattr(module, "get_memory_footprint"):
        usage = module.get_memory_footprint() / BYTES_PER_GB
    else:
        # manually
        param_size = sum(p.numel() * p.element_size() for p in module.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in module.buffers())

        total_size_bytes = param_size + buffer_size
        usage = total_size_bytes / (1024**3)

    return round(usage, 2)


# component name ->  ComponentLoader class
component_name_to_loader_cls: Dict[str, Type[Any]] = {}
