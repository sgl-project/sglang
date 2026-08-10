# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch
import torch.distributed.tensor as dist_tensor
from safetensors.torch import safe_open
from torch.distributed.fsdp import FSDPModule

from sglang.multimodal_gen.runtime.distributed import get_tp_rank, get_tp_world_size
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import _scan_safetensors_files
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

QUANTIZED_DTYPES = (
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.int8,
)
_QUANTIZED_SAFETENSORS_DTYPES = {
    "F8_E4M3",
    "F8_E5M2",
    "I8",
    "U8",
}


@dataclass(frozen=True)
class SafetensorsSource:
    file_path: str
    param_name: str
    shape: tuple[int, ...]
    dtype: str
    merge_index: int | None
    num_params_to_merge: int | None


@dataclass(frozen=True)
class LocalFSDPShard:
    tensor: torch.Tensor


@dataclass(frozen=True)
class LocalTPShard:
    tensor: torch.Tensor


def get_param_for_weight_loading(
    model: torch.nn.Module,
    param_dict: dict[str, torch.nn.Parameter],
    param_name: str,
) -> torch.nn.Parameter | None:
    actual_param = param_dict.get(param_name)
    if actual_param is not None and actual_param.__dict__.get("weight_loader"):
        return actual_param

    pre_fsdp_weight_loader_params = model.__dict__.get(
        "_pre_fsdp_weight_loader_params", {}
    )
    pre_fsdp_param = pre_fsdp_weight_loader_params.get(param_name)
    if pre_fsdp_param is not None:
        return pre_fsdp_param

    return actual_param


def _mapped_param_name(
    source_param_name: str,
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
    valid_target_names: set[str],
) -> tuple[str, int | None, int | None]:
    target_param_name, merge_index, num_params_to_merge = param_names_mapping(
        source_param_name
    )
    if (
        target_param_name != source_param_name
        and source_param_name in valid_target_names
        and target_param_name not in valid_target_names
    ):
        return source_param_name, None, None
    return target_param_name, merge_index, num_params_to_merge


def assembled_source_shape(
    sources: list[SafetensorsSource],
) -> tuple[int, ...] | None:
    if len(sources) == 1 and sources[0].merge_index is None:
        return sources[0].shape
    if any(source.merge_index is None for source in sources):
        return None

    expected_count = sources[0].num_params_to_merge
    if expected_count is None or len(sources) != expected_count:
        return None
    if any(source.num_params_to_merge != expected_count for source in sources):
        return None
    if {source.merge_index for source in sources} != set(range(expected_count)):
        return None

    reference_shape = sources[0].shape
    if not reference_shape:
        return None
    if any(source.shape[1:] != reference_shape[1:] for source in sources):
        return None
    return (sum(source.shape[0] for source in sources), *reference_shape[1:])


def _collect_safetensors_sources(
    weight_files: list[str],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
    valid_target_names: set[str],
) -> (
    tuple[
        dict[str, list[SafetensorsSource]],
        dict[str, tuple[str, Any, Any]],
    ]
    | None
):
    corrupted_files, duplicate_files_by_key = _scan_safetensors_files(weight_files)
    if corrupted_files or duplicate_files_by_key:
        return None

    sources_by_target: dict[str, list[SafetensorsSource]] = defaultdict(list)
    reverse_param_names_mapping: dict[str, tuple[str, Any, Any]] = {}
    for file_path in weight_files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for source_param_name in handle.keys():  # noqa: SIM118
                target_param_name, merge_index, num_params_to_merge = (
                    _mapped_param_name(
                        source_param_name,
                        param_names_mapping,
                        valid_target_names,
                    )
                )
                if not target_param_name:
                    continue
                source_slice = handle.get_slice(source_param_name)
                sources_by_target[target_param_name].append(
                    SafetensorsSource(
                        file_path=file_path,
                        param_name=source_param_name,
                        shape=tuple(source_slice.get_shape()),
                        dtype=source_slice.get_dtype(),
                        merge_index=merge_index,
                        num_params_to_merge=num_params_to_merge,
                    )
                )
                reverse_param_names_mapping[target_param_name] = (
                    source_param_name,
                    merge_index,
                    num_params_to_merge,
                )

    return sources_by_target, reverse_param_names_mapping


def read_rank_local_tensor(
    sources: list[SafetensorsSource],
    handles: dict[str, Any],
    local_shape: tuple[int, ...],
    global_offset: tuple[int, ...],
) -> torch.Tensor:
    if not local_shape:
        if len(sources) != 1:
            raise RuntimeError("Scalar checkpoint parameters cannot be merged")
        source = sources[0]
        return handles[source.file_path].get_tensor(source.param_name)

    ordered_sources = sorted(
        sources,
        key=lambda source: (
            source.merge_index is None,
            source.merge_index if source.merge_index is not None else 0,
        ),
    )
    if any(size == 0 for size in local_shape):
        source = ordered_sources[0]
        empty_slices = (slice(0, 0),) + (slice(None),) * (len(source.shape) - 1)
        return (
            handles[source.file_path]
            .get_slice(source.param_name)[empty_slices]
            .contiguous()
            .reshape(local_shape)
        )

    local_start = global_offset[0]
    local_end = local_start + local_shape[0]
    source_start = 0
    local_parts: list[torch.Tensor] = []
    for source in ordered_sources:
        source_end = source_start + source.shape[0]
        intersection_start = max(local_start, source_start)
        intersection_end = min(local_end, source_end)
        if intersection_start < intersection_end:
            slices = [
                slice(
                    intersection_start - source_start,
                    intersection_end - source_start,
                )
            ]
            slices.extend(
                slice(offset, offset + size)
                for offset, size in zip(global_offset[1:], local_shape[1:])
            )
            local_parts.append(
                handles[source.file_path]
                .get_slice(source.param_name)[tuple(slices)]
                .contiguous()
            )
        source_start = source_end

    if not local_parts:
        raise RuntimeError(
            f"No checkpoint slice overlaps local FSDP shard at offset={global_offset}, shape={local_shape}"
        )
    local_tensor = (
        local_parts[0] if len(local_parts) == 1 else torch.cat(local_parts, dim=0)
    )
    if tuple(local_tensor.shape) != local_shape:
        raise RuntimeError(
            "Rank-local checkpoint slice shape mismatch: "
            f"loaded={tuple(local_tensor.shape)}, expected={local_shape}"
        )
    return local_tensor


def _resolve_tp_shard_dim(
    actual_param: torch.nn.Parameter,
) -> tuple[bool, int | None]:
    weight_loader = actual_param.__dict__.get("weight_loader")
    if weight_loader is None:
        return True, None
    if not isinstance(weight_loader, MethodType):
        return False, None

    owner = weight_loader.__self__
    if isinstance(owner, ReplicatedLinear):
        return True, None
    if isinstance(owner, ColumnParallelLinear):
        output_dim = actual_param.__dict__.get("output_dim")
        return output_dim is not None, output_dim
    if isinstance(owner, RowParallelLinear):
        input_dim = actual_param.__dict__.get("input_dim")
        return True, input_dim
    return False, None


def tp_local_shape(
    sources: list[SafetensorsSource],
    shard_dim: int | None,
    tp_size: int,
) -> tuple[int, ...] | None:
    assembled_shape = assembled_source_shape(sources)
    if assembled_shape is None or shard_dim is None:
        return assembled_shape
    if shard_dim >= len(assembled_shape):
        return None
    if any(source.shape[shard_dim] % tp_size != 0 for source in sources):
        return None

    local_shape = list(assembled_shape)
    local_shape[shard_dim] //= tp_size
    return tuple(local_shape)


def read_tp_local_tensor(
    sources: list[SafetensorsSource],
    handles: dict[str, Any],
    shard_dim: int | None,
    tp_rank: int,
    tp_size: int,
) -> torch.Tensor:
    if shard_dim is None:
        assembled_shape = assembled_source_shape(sources)
        if assembled_shape is None:
            raise RuntimeError("Invalid checkpoint sources for replicated TP parameter")
        return read_rank_local_tensor(
            sources,
            handles,
            assembled_shape,
            (0,) * len(assembled_shape),
        )

    ordered_sources = sorted(
        sources,
        key=lambda source: (
            source.merge_index is None,
            source.merge_index if source.merge_index is not None else 0,
        ),
    )
    local_parts = []
    for source in ordered_sources:
        shard_size = source.shape[shard_dim] // tp_size
        slices = [slice(None)] * len(source.shape)
        slices[shard_dim] = slice(
            tp_rank * shard_size,
            (tp_rank + 1) * shard_size,
        )
        local_parts.append(
            handles[source.file_path]
            .get_slice(source.param_name)[tuple(slices)]
            .contiguous()
        )
    return local_parts[0] if len(local_parts) == 1 else torch.cat(local_parts, dim=0)


def try_load_rank_local_tp_state_dict(
    model: torch.nn.Module,
    weight_files: list[str],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
) -> (
    tuple[
        dict[str, LocalTPShard],
        dict[str, tuple[str, Any, Any]],
    ]
    | None
):
    tp_size = get_tp_world_size()
    if tp_size == 1:
        return None

    meta_sd = model.state_dict()
    param_dict = dict(model.named_parameters())
    checkpoint_sources = _collect_safetensors_sources(
        weight_files,
        param_names_mapping,
        set(meta_sd),
    )
    if checkpoint_sources is None:
        return None
    sources_by_target, reverse_param_names_mapping = checkpoint_sources

    shard_dims: dict[str, int | None] = {}
    for target_param_name, sources in sources_by_target.items():
        meta_param = meta_sd.get(target_param_name)
        if meta_param is None or isinstance(meta_param, dist_tensor.DTensor):
            return None
        if meta_param.dtype in QUANTIZED_DTYPES:
            return None
        if any(source.dtype in _QUANTIZED_SAFETENSORS_DTYPES for source in sources):
            return None

        actual_param = get_param_for_weight_loading(
            model,
            param_dict,
            target_param_name,
        )
        if actual_param is None:
            supported, shard_dim = True, None
        else:
            supported, shard_dim = _resolve_tp_shard_dim(actual_param)
        if not supported:
            return None
        if tp_local_shape(sources, shard_dim, tp_size) != tuple(meta_param.shape):
            return None
        shard_dims[target_param_name] = shard_dim

    local_param_sd: dict[str, LocalTPShard] = {}
    local_bytes = 0
    tp_rank = get_tp_rank()
    with ExitStack() as stack:
        handles = {
            file_path: stack.enter_context(
                safe_open(file_path, framework="pt", device="cpu")
            )
            for file_path in weight_files
        }
        for target_param_name in sorted(sources_by_target):
            tensor = read_tp_local_tensor(
                sources_by_target[target_param_name],
                handles,
                shard_dims[target_param_name],
                tp_rank,
                tp_size,
            )
            local_param_sd[target_param_name] = LocalTPShard(tensor)
            local_bytes += tensor.numel() * tensor.element_size()

    logger.info(
        "Loaded rank-local TP checkpoint slices: rank=%d, tensors=%d, bytes=%.2f GiB",
        torch.distributed.get_rank(),
        len(local_param_sd),
        local_bytes / (1024**3),
    )
    return local_param_sd, reverse_param_names_mapping


def try_load_rank_local_fsdp_state_dict(
    model: FSDPModule,
    weight_files: list[str],
    param_names_mapping: Callable[[str], tuple[str, Any, Any]],
) -> (
    tuple[
        dict[str, torch.Tensor | LocalFSDPShard],
        dict[str, tuple[str, Any, Any]],
    ]
    | None
):
    if get_tp_world_size() != 1:
        return None

    meta_sd = model.state_dict()
    checkpoint_sources = _collect_safetensors_sources(
        weight_files,
        param_names_mapping,
        set(meta_sd),
    )
    if checkpoint_sources is None:
        return None
    sources_by_target, reverse_param_names_mapping = checkpoint_sources

    for target_param_name, sources in sources_by_target.items():
        meta_param = meta_sd.get(target_param_name)
        assembled_shape = assembled_source_shape(sources)
        if meta_param is None or assembled_shape != tuple(meta_param.shape):
            return None
        if meta_param.dtype in QUANTIZED_DTYPES:
            return None
        if any(source.dtype in _QUANTIZED_SAFETENSORS_DTYPES for source in sources):
            return None

    local_param_sd: dict[str, torch.Tensor | LocalFSDPShard] = {}
    local_bytes = 0
    with ExitStack() as stack:
        handles = {
            file_path: stack.enter_context(
                safe_open(file_path, framework="pt", device="cpu")
            )
            for file_path in weight_files
        }
        for target_param_name in sorted(sources_by_target):
            meta_param = meta_sd[target_param_name]
            if isinstance(meta_param, dist_tensor.DTensor):
                local_shape, global_offset = (
                    dist_tensor._utils.compute_local_shape_and_global_offset(
                        meta_param.shape,
                        meta_param.device_mesh,
                        meta_param.placements,
                    )
                )
                tensor = read_rank_local_tensor(
                    sources_by_target[target_param_name],
                    handles,
                    tuple(local_shape),
                    tuple(global_offset),
                )
                local_param_sd[target_param_name] = LocalFSDPShard(tensor)
            else:
                tensor = read_rank_local_tensor(
                    sources_by_target[target_param_name],
                    handles,
                    tuple(meta_param.shape),
                    (0,) * meta_param.ndim,
                )
                local_param_sd[target_param_name] = tensor
            local_bytes += tensor.numel() * tensor.element_size()

    logger.info(
        "Loaded rank-local FSDP checkpoint slices: rank=%d, tensors=%d, bytes=%.2f GiB",
        torch.distributed.get_rank(),
        len(local_param_sd),
        local_bytes / (1024**3),
    )
    return local_param_sd, reverse_param_names_mapping
