"""Memory-aware ordering for pipeline component weight loads to avoid OOM while loading.

Load the VRAM-intensive components earlier than others

The pipeline owns component selection, path resolution, and actual loading; this
module only ranks already-selected load specs.
"""

import glob
import json
import os
from dataclasses import dataclass

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload_components import (
    is_dit_component_name,
    is_image_encoder_component_name,
    is_text_encoder_component_name,
    is_vae_component_name,
)


@dataclass(frozen=True)
class ComponentLoadSpec:
    """One pipeline component that still needs a real weight load."""

    module_name: str
    load_module_name: str
    component_model_path: str
    transformers_or_diffusers: str
    architecture: str | None
    index: int


_WEIGHT_FILE_SUFFIXES = (".bin", ".pt", ".pth")


def _component_base_name(component_name: str) -> str:
    prefix, separator, suffix = component_name.rpartition("_")
    if separator and suffix.isdigit():
        return prefix
    return component_name


def _component_variant_priority(component_name: str) -> int:
    _, separator, suffix = component_name.rpartition("_")
    if separator and suffix.isdigit():
        return -int(suffix)
    return 0


def component_load_risk_rank(component_name: str) -> int:
    """Fallback type rank when checkpoint size cannot be inferred."""
    candidate_names = (component_name, _component_base_name(component_name))
    if any(is_dit_component_name(name) for name in candidate_names):
        return 0
    if any(is_text_encoder_component_name(name) for name in candidate_names):
        return 1
    if any(is_image_encoder_component_name(name) for name in candidate_names):
        return 2
    if any(is_vae_component_name(name) for name in candidate_names):
        return 3
    return 10


def _safe_file_size(file_path: str) -> int | None:
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def _safetensors_payload_size_bytes(file_path: str) -> int | None:
    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                return _safe_file_size(file_path)
            header_size = int.from_bytes(header_size_bytes, "little")
            header = json.loads(f.read(header_size))
    except (OSError, json.JSONDecodeError, ValueError):
        return _safe_file_size(file_path)

    payload_size = 0
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        offsets = tensor_info.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            return _safe_file_size(file_path)
        payload_size += offsets[1] - offsets[0]
    return payload_size


def _safetensors_files_from_index(component_model_path: str) -> list[str]:
    indexed_files: set[str] = set()
    index_paths = sorted(
        glob.glob(os.path.join(component_model_path, "*.safetensors.index.json"))
    )
    for index_path in index_paths:
        try:
            with open(index_path) as f:
                weight_map = json.load(f).get("weight_map", {})
        except (OSError, json.JSONDecodeError):
            continue
        for shard_name in weight_map.values():
            shard_path = os.path.join(component_model_path, shard_name)
            if os.path.isfile(shard_path):
                indexed_files.add(shard_path)
    return sorted(indexed_files)


def _list_component_safetensors_files(component_model_path: str) -> list[str]:
    if os.path.isfile(component_model_path):
        if component_model_path.endswith(".safetensors"):
            return [component_model_path]
        return []
    if not os.path.isdir(component_model_path):
        return []

    indexed_files = _safetensors_files_from_index(component_model_path)
    if indexed_files:
        return indexed_files
    return sorted(glob.glob(os.path.join(component_model_path, "*.safetensors")))


def infer_component_weight_size_bytes(component_model_path: str) -> int | None:
    """Infer checkpoint payload size from safetensors without materializing tensors."""
    safetensors_files = _list_component_safetensors_files(component_model_path)
    if safetensors_files:
        sizes = [
            size
            for size in (
                _safetensors_payload_size_bytes(file_path)
                for file_path in safetensors_files
            )
            if size is not None
        ]
        return sum(sizes) if sizes else None

    if os.path.isfile(component_model_path):
        if component_model_path.endswith(_WEIGHT_FILE_SUFFIXES):
            return _safe_file_size(component_model_path)
        return None
    if not os.path.isdir(component_model_path):
        return None

    weight_files = []
    for suffix in _WEIGHT_FILE_SUFFIXES:
        weight_files.extend(glob.glob(os.path.join(component_model_path, f"*{suffix}")))
    if not weight_files:
        return None
    sizes = [
        size
        for size in (_safe_file_size(file_path) for file_path in weight_files)
        if size is not None
    ]
    return sum(sizes) if sizes else None


def order_component_load_specs(
    component_specs: list[ComponentLoadSpec],
) -> list[ComponentLoadSpec]:
    # load larger weight payloads before small helpers to reduce startup peak OOMs
    return sorted(
        component_specs,
        key=lambda spec: (
            # 1. model size inferred from checkpoints
            -(infer_component_weight_size_bytes(spec.component_model_path) or 0),
            # 2. infer from component name
            component_load_risk_rank(spec.load_module_name),
            _component_variant_priority(spec.load_module_name),
            spec.index,
        ),
    )
