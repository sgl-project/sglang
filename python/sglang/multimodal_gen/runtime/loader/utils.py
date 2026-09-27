# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Utilities for selecting and loading models."""

import bisect
import contextlib
import glob
import json
import os
import re
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any, Dict, Type

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.weights.source import (
    filter_duplicate_precision_variant_safetensors,
)

logger = init_logger(__name__)

_DEFAULT_SAFETENSORS_INDEX = "diffusion_pytorch_model.safetensors.index.json"

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
        for cls in (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Embedding):
            self._orig_reset[cls] = cls.reset_parameters
            cls.reset_parameters = lambda self: None  # skip init
        from transformers.modeling_utils import PreTrainedModel

        self._pretrained_model_cls = PreTrainedModel
        self._orig_post_init = PreTrainedModel.post_init
        PreTrainedModel.post_init = lambda self: None

    def __exit__(self, exc_type, exc_value, traceback):
        # restore originals
        for cls, orig in self._orig_reset.items():
            cls.reset_parameters = orig
        self._pretrained_model_cls.post_init = self._orig_post_init


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


def checkpoint_bytes(model_path: str) -> int:
    """On-disk size of the selected safetensors checkpoint files."""
    if os.path.isfile(model_path):
        return os.path.getsize(model_path)

    paths = sorted(
        glob.glob(os.path.join(str(model_path), "**", "*.safetensors"), recursive=True)
    )
    total = 0
    for path in filter_duplicate_precision_variant_safetensors(paths):
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def keep_checkpoint_mapped(*, weight_bytes: int, component: str) -> bool:
    """Whether a component's weights should stay on their file mapping.

    Judged against the whole deployment rather than the one component: on a
    host that cannot afford copies of everything it is about to serve, every
    byte of anonymous memory a copy takes is a byte the pin budget for the
    stepped components loses. On a host with room, the copy is the faster
    choice -- its pages are resident, where a mapping's first use pays a fault.
    """
    from sglang.multimodal_gen.runtime.managers.memory_managers.host_memory_budget import (
        host_copies_would_not_fit,
        host_memory_available_bytes,
    )

    if not host_copies_would_not_fit(weight_bytes):
        return False
    logger.info(
        "%s stays on its checkpoint mapping: the deployment is %.2f GiB of "
        "weights against %.2f GiB of host memory, so copies are host memory "
        "the streamed components need more.",
        component,
        weight_bytes / 1024**3,
        host_memory_available_bytes() / 1024**3,
    )
    return True


def _select_safetensors_index_file(model_path: str, preferred_name: str) -> str | None:
    preferred_path = os.path.join(str(model_path), preferred_name)
    if os.path.exists(preferred_path):
        return preferred_path

    candidates = filter_duplicate_precision_variant_safetensors(
        sorted(glob.glob(os.path.join(str(model_path), "*.safetensors.index.json")))
    )
    return candidates[0] if len(candidates) == 1 else None


def _list_safetensors_files(
    model_path: str,
    *,
    index_file: str = _DEFAULT_SAFETENSORS_INDEX,
    key_filter: Callable[[str], bool] | None = None,
    raw_candidates: bool = False,
) -> list[str]:
    """Resolve the safetensors files to load from a local component path.

    An index is authoritative when present. Otherwise canonical files are
    preferred over precision-suffixed copies. ``raw_candidates`` is reserved
    for model-specific selectors that must choose a precision variant first.
    """
    if os.path.isfile(model_path):
        return [str(model_path)] if str(model_path).endswith(".safetensors") else []

    found = sorted(glob.glob(os.path.join(str(model_path), "*.safetensors")))

    index_path = _select_safetensors_index_file(model_path, index_file)
    if index_path is not None:
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        expected_shards = sorted(set(weight_map.values()))
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

        if not raw_candidates:
            selected_shards = {
                shard
                for weight_name, shard in weight_map.items()
                if key_filter is None or key_filter(weight_name)
            }
            return [
                os.path.join(str(model_path), shard)
                for shard in sorted(selected_shards)
            ]

    if raw_candidates:
        return found
    return filter_duplicate_precision_variant_safetensors(found)


def load_safetensors_state_dict(model_path: str) -> dict[str, torch.Tensor]:
    """Load one safetensors checkpoint, including an indexed sharded set."""
    index_path = _select_safetensors_index_file(model_path, _DEFAULT_SAFETENSORS_INDEX)
    safetensors_files = _list_safetensors_files(model_path)
    if index_path is not None:
        state_dict: dict[str, torch.Tensor] = {}
        for path in safetensors_files:
            state_dict.update(safetensors_load_file(path))
        return state_dict

    if not safetensors_files:
        raise ValueError(f"No safetensors files found in {model_path}")
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Found {len(safetensors_files)} safetensors files in {model_path} "
            "and no index to disambiguate them."
        )
    return safetensors_load_file(safetensors_files[0])


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


def _read_process_mappings() -> tuple[list[int], list[int], list[bool]] | None:
    """Sorted (start, end, is_file_backed) of this process's address space.

    Linux only; returns None where /proc is unavailable, and the caller then
    reports host bytes without splitting file-backed from anonymous.
    """
    try:
        with open("/proc/self/maps") as handle:
            rows = []
            for line in handle:
                fields = line.split(maxsplit=5)
                low, _, high = fields[0].partition("-")
                path = fields[5].strip() if len(fields) > 5 else ""
                # pseudo-paths like [heap] and [stack] are anonymous
                rows.append(
                    (int(low, 16), int(high, 16), bool(path) and path[0] != "[")
                )
    except OSError:
        return None
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


class MappedRegions:
    """Answers whether a tensor's bytes live in a file mapping.

    Built once and reused. The lookup table comes from /proc/self/maps, so
    rebuilding it per tensor would be quadratic over a checkpoint's worth of
    weights -- H3's DiT alone has tens of thousands.

    A snapshot, not a live view: mappings created after construction are
    unknown to it. Callers that need to classify freshly loaded weights should
    build one after loading, which is when the mappings exist.
    """

    def __init__(self) -> None:
        self._maps = _read_process_mappings()

    @property
    def available(self) -> bool:
        """False where /proc is absent, in which case nothing is classified."""
        return self._maps is not None

    def holds_pointer(self, pointer: int) -> bool:
        if self._maps is None or pointer == 0:
            return False
        starts, ends, backed = self._maps
        index = bisect.bisect_right(starts, pointer) - 1
        if index < 0 or pointer >= ends[index]:
            return False
        return backed[index]

    def holds(self, tensor: torch.Tensor) -> bool:
        if tensor.device.type != "cpu":
            return False
        try:
            return self.holds_pointer(tensor.untyped_storage().data_ptr())
        except Exception:
            return False


def component_residency_bytes(module) -> Dict[str, int]:
    """Where a component's weights actually sit, in bytes.

    Four buckets, ordered by what the kernel can do with them: device memory,
    pinned host memory (which it cannot reclaim at all), file-backed host
    memory (which it can drop without swapping), and anonymous host memory.

    Two caveats. `host_mapped` counts the size of the file mapping, not the
    pages currently resident in it -- a mapped safetensors file is faulted in
    lazily, so the real footprint is at most this. And pinned is tested first
    because CUDA's host allocator sits behind a named mapping, which the
    file-backed check alone would misread.

    Layerwise-offloaded weights are absent from parameters()/buffers(): the
    module keeps (1,) placeholders while its offload managers own the host
    copy, so those managers are walked too. Sizes are taken from the storage
    and deduped by it, because one flat host buffer backs many logical weights.
    """
    if not isinstance(module, nn.Module):
        return {}

    totals = {"vram": 0, "host_pinned": 0, "host_mapped": 0, "host": 0}
    seen: set[int] = set()
    regions = MappedRegions()

    def is_file_backed(pointer: int) -> bool:
        return regions.holds_pointer(pointer)

    def add(tensor: torch.Tensor) -> None:
        try:
            storage = tensor.untyped_storage()
            pointer = storage.data_ptr()
        except Exception:
            return
        # a zero pointer is an empty offload placeholder, not a weight
        if pointer == 0 or pointer in seen:
            return
        seen.add(pointer)
        if tensor.device.type != "cpu":
            totals["vram"] += storage.nbytes()
            return
        try:
            pinned = tensor.is_pinned()
        except Exception:
            pinned = False
        if pinned:
            bucket = "host_pinned"
        elif is_file_backed(pointer):
            bucket = "host_mapped"
        else:
            bucket = "host"
        totals[bucket] += storage.nbytes()

    for tensor in module.parameters():
        add(tensor)
    for tensor in module.buffers():
        add(tensor)
    for manager in getattr(module, "layerwise_offload_managers", None) or []:
        iter_cpu_weights = getattr(manager, "iter_cpu_weights", None)
        if iter_cpu_weights is None:
            continue
        for _, tensor in iter_cpu_weights():
            add(tensor)

    return totals


def format_component_residency(module) -> str:
    """Name the places a component's weights are, skipping the empty ones.

    A component that streams from the host reports no VRAM at rest, which is
    the point; saying so beats reporting a zero delta that reads as free.
    """
    totals = component_residency_bytes(module)
    # `pinned` and `pageable` are the standard CUDA pair, and naming mmap after
    # the call says what it is: labels a reader has to guess at defeat the point
    # of splitting host bytes in the first place.
    labels = (
        ("vram", "vram"),
        ("host_pinned", "host pinned"),
        ("host_mapped", "host mmap"),
        ("host", "host pageable"),
    )
    parts = [
        f"{label}: {totals[key] / BYTES_PER_GB:.2f} GB"
        for key, label in labels
        if totals.get(key)
    ]
    return ", ".join(parts) if parts else "weights: none"


# component name ->  ComponentLoader class
component_name_to_loader_cls: Dict[str, Type[Any]] = {}
