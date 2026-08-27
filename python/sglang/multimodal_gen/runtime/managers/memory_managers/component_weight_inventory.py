"""Pre-load weight inventory for pipeline placement planning."""

import glob
import json
import math
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentWeightSource:
    """One physical checkpoint source selected by the pipeline."""

    component_name: str
    component_model_path: str
    checkpoint_bytes: int | None = None
    parameter_count: int | None = None
    target_element_size: int | None = None
    supports_fsdp_loading: bool = False


@dataclass(frozen=True)
class ComponentWeightEstimate:
    component_name: str
    component_model_path: str
    checkpoint_bytes: int | None
    parameter_count: int | None
    target_element_size: int | None
    supports_fsdp_loading: bool = False

    def materialized_bytes(self) -> int | None:
        if self.parameter_count is not None and self.target_element_size is not None:
            return self.parameter_count * self.target_element_size
        return self.checkpoint_bytes


_WEIGHT_FILE_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".gguf")


def _safe_file_size(file_path: str) -> int | None:
    try:
        return os.path.getsize(file_path)
    except OSError:
        return None


def _read_safetensors_header(file_path: str) -> dict | None:
    try:
        with open(file_path, "rb") as f:
            header_size_bytes = f.read(8)
            if len(header_size_bytes) != 8:
                return None
            return json.loads(f.read(int.from_bytes(header_size_bytes, "little")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _tensor_stats(tensor_info: dict) -> tuple[int, int] | None:
    offsets = tensor_info.get("data_offsets")
    shape = tensor_info.get("shape")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not isinstance(shape, list)
        or any(not isinstance(dimension, int) for dimension in shape)
    ):
        return None
    return offsets[1] - offsets[0], math.prod(shape)


def _safetensors_payload_stats(file_path: str) -> tuple[int, int | None] | None:
    header = _read_safetensors_header(file_path)
    if header is None:
        size = _safe_file_size(file_path)
        return None if size is None else (size, None)

    checkpoint_bytes = 0
    parameter_count = 0
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        stats = _tensor_stats(tensor_info)
        if stats is None:
            size = _safe_file_size(file_path)
            return None if size is None else (size, None)
        checkpoint_bytes += stats[0]
        parameter_count += stats[1]
    return checkpoint_bytes, parameter_count


def infer_safetensors_weight_stats(
    file_paths: list[str],
) -> tuple[int | None, int | None]:
    """Infer aggregate stats for the exact safetensors files selected to load."""
    stats = [
        stat
        for file_path in file_paths
        if (stat := _safetensors_payload_stats(file_path)) is not None
    ]
    if not stats:
        return None, None
    parameter_counts = [parameters for _, parameters in stats]
    return (
        sum(checkpoint_bytes for checkpoint_bytes, _ in stats),
        (
            sum(parameter_counts)
            if all(parameters is not None for parameters in parameter_counts)
            else None
        ),
    )


def infer_safetensors_weight_stats_by_prefix(
    file_path: str,
) -> dict[str, tuple[int, int]] | None:
    """Return payload bytes and parameter count by first tensor-key segment."""
    header = _read_safetensors_header(file_path)
    if header is None:
        return None

    result: dict[str, tuple[int, int]] = {}
    for tensor_name, tensor_info in header.items():
        if tensor_name == "__metadata__":
            continue
        stats = _tensor_stats(tensor_info)
        if stats is None:
            return None
        prefix = tensor_name.split(".", 1)[0]
        previous = result.get(prefix, (0, 0))
        result[prefix] = (previous[0] + stats[0], previous[1] + stats[1])
    return result


def _safetensors_files_from_index(component_model_path: str) -> list[str]:
    indexed_files: set[str] = set()
    for index_path in sorted(
        glob.glob(os.path.join(component_model_path, "*.safetensors.index.json"))
    ):
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
        return (
            [component_model_path]
            if component_model_path.endswith(".safetensors")
            else []
        )
    if not os.path.isdir(component_model_path):
        return []

    indexed_files = _safetensors_files_from_index(component_model_path)
    return indexed_files or sorted(
        glob.glob(os.path.join(component_model_path, "*.safetensors"))
    )


def infer_component_weight_stats(
    component_model_path: str,
) -> tuple[int | None, int | None]:
    """Infer checkpoint bytes and parameter count without loading tensors."""
    safetensors_files = _list_component_safetensors_files(component_model_path)
    if safetensors_files:
        return infer_safetensors_weight_stats(safetensors_files)

    if os.path.isfile(component_model_path):
        if component_model_path.endswith(_WEIGHT_FILE_SUFFIXES):
            return _safe_file_size(component_model_path), None
        return None, None
    if not os.path.isdir(component_model_path):
        return None, None

    weight_files = [
        file_path
        for suffix in _WEIGHT_FILE_SUFFIXES
        for file_path in glob.glob(os.path.join(component_model_path, f"*{suffix}"))
    ]
    if not weight_files:
        return 0, 0
    sizes = [
        size
        for file_path in weight_files
        if (size := _safe_file_size(file_path)) is not None
    ]
    return (sum(sizes), None) if sizes else (None, None)


def infer_component_weight_size_bytes(component_model_path: str) -> int | None:
    return infer_component_weight_stats(component_model_path)[0]


def estimate_component_weight_inventory(
    sources: list[ComponentWeightSource],
) -> list[ComponentWeightEstimate]:
    """Resolve the selected checkpoint sources without loading tensors."""
    seen_names: set[str] = set()
    inventory = []
    for source in sources:
        if source.component_name in seen_names:
            raise ValueError(
                f"duplicate component weight source {source.component_name!r}"
            )
        seen_names.add(source.component_name)
        inferred_bytes, inferred_parameters = (
            infer_component_weight_stats(source.component_model_path)
            if source.checkpoint_bytes is None
            else (None, None)
        )
        inventory.append(
            ComponentWeightEstimate(
                component_name=source.component_name,
                component_model_path=source.component_model_path,
                checkpoint_bytes=(
                    source.checkpoint_bytes
                    if source.checkpoint_bytes is not None
                    else inferred_bytes
                ),
                parameter_count=(
                    source.parameter_count
                    if source.parameter_count is not None
                    else inferred_parameters
                ),
                target_element_size=source.target_element_size,
                supports_fsdp_loading=source.supports_fsdp_loading,
            )
        )
    return inventory
