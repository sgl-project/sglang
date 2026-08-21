# SPDX-License-Identifier: Apache-2.0
"""Immutable heterogeneous weight transfer between daemons.

Source daemons collectively publish one address-bearing runtime manifest over
a lightweight HTTP registry. Target daemons construct their requested parallel
layout with the dummy loader, reject overlapping physical GPUs on the same node,
then use Mooncake's manifest planner and Transfer Engine reader to pull exactly
the ranges needed by each target rank. No Engine control-plane participation is
required and both daemon-owned models remain immutable.

The weight cache has no data-parallel ownership dimension, so this module
supports TP, PP and static EP layouts while intentionally keeping DP=1.  The
Mooncake planner consumes layout-independent logical tensor boxes; PP and EP
support therefore belongs here, above Mooncake, rather than in the transport.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import msgspec

from sglang.srt.environ import envs
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

from .weight_runtime_manifest import (
    WeightParallelTopology,
    create_weight_runtime_manifest_builder,
    model_identity_from_config,
)

logger = logging.getLogger(__name__)


class WeightHeterogeneousTransferError(RuntimeError):
    pass


@dataclass(frozen=True)
class WeightParallelLayout:
    """Process and weight ownership dimensions managed by weight cache."""

    tp_size: int
    pp_size: int = 1
    ep_size: int = 1

    def __post_init__(self) -> None:
        if min(self.tp_size, self.pp_size, self.ep_size) < 1:
            raise ValueError("TP, PP and EP sizes must be positive")
        if self.tp_size % self.ep_size != 0:
            raise ValueError(
                f"tp_size={self.tp_size} must be divisible by ep_size={self.ep_size}"
            )

    @property
    def world_size(self) -> int:
        # EP partitions the TP group and is not an independent process axis.
        return self.tp_size * self.pp_size

    def iter_ranks(self):
        for pp_rank in range(self.pp_size):
            for tp_rank in range(self.tp_size):
                yield pp_rank, tp_rank


def rebuild_transferred_weight_state(model: Any) -> None:
    """Recompute non-persistent tensors derived from immutable parameters.

    Normal disk loaders update these objects while copying each checkpoint
    weight.  A manifest transfer writes parameter storage directly, so the
    same derived state must be rebuilt after the pull and before readiness.
    """
    post_load_weights = getattr(model, "post_load_weights", None)
    if callable(post_load_weights):
        # DeepSeek MLA derives w_kc/w_vc from kv_b_proj here. This is the
        # model-specific hook also used by checkpoint-engine weight updates;
        # it does not re-run generic quantization transforms.
        post_load_weights()

    from sglang.srt.model_loader.loader import PreshardedModelLoader

    PreshardedModelLoader._rebind_parameter_aliases(model)


def rebuild_client_derived_weight_state(model: Any) -> None:
    """Rebuild client-local views without writing daemon-owned IPC tensors."""
    import torch

    for _, module in model.named_modules():
        kv_b_proj = getattr(module, "kv_b_proj", None)
        weight = getattr(kv_b_proj, "weight", None)
        if (
            isinstance(weight, torch.Tensor)
            and weight.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and hasattr(module, "qk_nope_head_dim")
            and hasattr(module, "v_head_dim")
            and hasattr(module, "w_kc")
            and hasattr(module, "w_vc")
        ):
            w_kc, w_vc = weight.unflatten(
                0, (-1, module.qk_nope_head_dim + module.v_head_dim)
            ).split([module.qk_nope_head_dim, module.v_head_dim], dim=1)
            module.w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)
            module.w_vc = w_vc.contiguous().transpose(1, 2)
            if hasattr(kv_b_proj, "weight_scale"):
                module.w_scale = kv_b_proj.weight_scale


def _compute_weight_content_checksums(
    model: Any,
    runtime_inventory: Any,
) -> tuple[dict[str, Any], ...]:
    """Compute layout-independent checksums over logical parameter bytes."""
    import numpy as np
    import torch

    parameters = dict(model.named_parameters(remove_duplicate=False))
    mask = (1 << 64) - 1
    multiplier = np.uint64(0x9E3779B185EBCA87)
    increment = np.uint64(0xC2B2AE3D27D4EB4F)
    max_chunk_bytes = 4 * 1024 * 1024
    checksums = []

    for tensor_record in runtime_inventory.tensors:
        parameter = parameters.get(tensor_record.runtime_name)
        if parameter is None:
            raise WeightHeterogeneousTransferError(
                "manifest parameter is missing at checksum time: "
                f"{tensor_record.runtime_name}"
            )
        itemsize = int(tensor_record.itemsize)
        if tensor_record.byte_offset % itemsize or tensor_record.nbytes % itemsize:
            raise WeightHeterogeneousTransferError(
                f"unaligned manifest bytes for {tensor_record.runtime_name}: "
                f"offset={tensor_record.byte_offset}, "
                f"nbytes={tensor_record.nbytes}, itemsize={itemsize}"
            )
        element_offset = int(tensor_record.byte_offset // itemsize)
        element_count = int(tensor_record.nbytes // itemsize)
        flat = parameter.detach().reshape(-1)
        if element_offset + element_count > flat.numel():
            raise WeightHeterogeneousTransferError(
                f"manifest checksum range exceeds {tensor_record.runtime_name} storage"
            )

        global_shape = tuple(int(value) for value in tensor_record.global_shape)
        global_offset = tuple(int(value) for value in tensor_record.global_offset)
        local_shape = tuple(int(value) for value in tensor_record.local_shape)
        local_strides = []
        global_strides = []
        for shape, output in (
            (local_shape, local_strides),
            (global_shape, global_strides),
        ):
            stride = 1
            reverse = []
            for size in reversed(shape):
                reverse.append(stride)
                stride *= size
            output.extend(reversed(reverse))

        byte_sum = 0
        weighted_sum = 0
        elements_per_chunk = max(1, max_chunk_bytes // itemsize)
        for local_start in range(0, element_count, elements_per_chunk):
            count = min(elements_per_chunk, element_count - local_start)
            raw = (
                flat[
                    element_offset + local_start : element_offset + local_start + count
                ]
                .view(torch.uint8)
                .cpu()
                .numpy()
                .reshape(-1)
            )
            local_indices = np.arange(
                local_start,
                local_start + count,
                dtype=np.uint64,
            )
            global_indices = np.zeros(count, dtype=np.uint64)
            for local_stride, local_size, offset, global_stride in zip(
                local_strides,
                local_shape,
                global_offset,
                global_strides,
            ):
                coordinate = (
                    (local_indices // np.uint64(local_stride)) % np.uint64(local_size)
                ) + np.uint64(offset)
                global_indices += coordinate * np.uint64(global_stride)

            byte_sum = (byte_sum + int(raw.sum(dtype=np.uint64))) & mask
            global_byte_base = global_indices * np.uint64(itemsize)
            for lane in range(itemsize):
                values = raw[lane::itemsize].astype(np.uint64)
                byte_indices = global_byte_base + np.uint64(lane)
                weights = byte_indices * multiplier + increment
                weighted_sum = (
                    weighted_sum + int((values * weights).sum(dtype=np.uint64))
                ) & mask

        checksums.append(
            {
                "tensor_id": tensor_record.tensor_id,
                "global_shape": global_shape,
                "global_offset": global_offset,
                "local_shape": local_shape,
                "nbytes": int(tensor_record.nbytes),
                "byte_sum": byte_sum,
                "weighted_sum": weighted_sum,
            }
        )
    return tuple(checksums)


def _aggregate_weight_content_checksums(
    groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, tuple[tuple[int, ...], int, int, int]]:
    """Deduplicate replicas and fold arbitrary TP boxes by logical tensor ID."""
    mask = (1 << 64) - 1
    boxes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for group in groups:
        for record in group:
            key = (
                record["tensor_id"],
                tuple(record["global_offset"]),
                tuple(record["local_shape"]),
            )
            existing = boxes.get(key)
            if existing is not None and existing != record:
                raise WeightHeterogeneousTransferError(
                    f"replicated logical box has inconsistent content: {key}"
                )
            boxes[key] = record

    totals: dict[str, tuple[tuple[int, ...], int, int, int]] = {}
    for record in boxes.values():
        tensor_id = record["tensor_id"]
        global_shape = tuple(record["global_shape"])
        previous = totals.get(tensor_id, (global_shape, 0, 0, 0))
        if previous[0] != global_shape:
            raise WeightHeterogeneousTransferError(
                f"logical tensor global shape differs across fragments: {tensor_id}"
            )
        totals[tensor_id] = (
            global_shape,
            previous[1] + int(record["nbytes"]),
            (previous[2] + int(record["byte_sum"])) & mask,
            (previous[3] + int(record["weighted_sum"])) & mask,
        )
    return totals


def _register_weight_fragment_allocations(
    transfer_engine: Any,
    fragments: Sequence[Any],
) -> tuple[tuple[int, int], ...]:
    """Register full CUDA allocator regions owning manifest fragments.

    Transfer Engine can report success yet mishandle tiny fragmented CUDA
    registrations.  Registering the owning cudaMalloc segment matches the
    proven sglang-manifest path and also deduplicates aliased weights.
    """
    import torch

    fragment_ranges = sorted(
        {
            (int(fragment.address), int(fragment.address + fragment.nbytes))
            for fragment in fragments
        }
    )
    allocator_snapshot = torch.cuda.memory.memory_snapshot()
    active_blocks: list[tuple[int, int, int]] = []
    for segment_index, segment in enumerate(allocator_snapshot):
        for block in segment.get("blocks", ()):
            address = block.get("address", -1)
            size = block.get("size", -1)
            if (
                type(address) is int
                and type(size) is int
                and address >= 0
                and size > 0
                and block.get("state") == "active_allocated"
            ):
                active_blocks.append((address, address + size, segment_index))
    active_blocks.sort()
    block_starts = [block[0] for block in active_blocks]
    selected_segments = set()
    for begin, end in fragment_ranges:
        index = bisect_right(block_starts, begin) - 1
        if index < 0 or end > active_blocks[index][1]:
            raise WeightHeterogeneousTransferError(
                f"manifest range {begin}:{end} is not covered by an active CUDA allocation"
            )
        selected_segments.add(active_blocks[index][2])

    registered_memory_ranges: list[tuple[int, int]] = []
    try:
        for segment_index in sorted(selected_segments):
            segment = allocator_snapshot[segment_index]
            segment_begin = segment.get("address", -1)
            segment_size = segment.get("total_size", -1)
            if (
                type(segment_begin) is int
                and type(segment_size) is int
                and segment_begin >= 0
                and segment_size > 0
                and transfer_engine.register_memory(segment_begin, segment_size) == 0
            ):
                registered_memory_ranges.append((segment_begin, segment_size))
                continue
            merged: list[tuple[int, int]] = []
            for begin, end, block_segment_index in active_blocks:
                if block_segment_index != segment_index:
                    continue
                if merged and merged[-1][1] == begin:
                    merged[-1] = (merged[-1][0], end)
                else:
                    merged.append((begin, end))
            for begin, end in merged:
                result = transfer_engine.register_memory(begin, end - begin)
                if result != 0:
                    raise WeightHeterogeneousTransferError(
                        f"Mooncake register_memory failed for {begin}+{end - begin}: {result}"
                    )
                registered_memory_ranges.append((begin, end - begin))

        registered_intervals = sorted(
            (address, address + nbytes) for address, nbytes in registered_memory_ranges
        )
        registered_starts = [item[0] for item in registered_intervals]
        for begin, end in fragment_ranges:
            index = bisect_right(registered_starts, begin) - 1
            if index < 0 or end > registered_intervals[index][1]:
                raise WeightHeterogeneousTransferError(
                    f"manifest range {begin}:{end} is not covered by registered memory"
                )
        return tuple(registered_memory_ranges)
    except BaseException:
        for address, _ in reversed(registered_memory_ranges):
            try:
                transfer_engine.unregister_memory(address)
            except Exception:
                logger.exception("failed to undo Mooncake memory registration")
        raise


def validate_weight_heterogeneous_transfer_configuration(
    *,
    tp_size: int,
    dp_size: int,
    ep_size: int,
    pp_size: int,
    nnodes: int = 1,
    quantization: Optional[str] = None,
) -> None:
    unsupported = []
    if nnodes != 1:
        unsupported.append(f"nnodes={nnodes}")
    if dp_size != 1:
        unsupported.append(f"dp_size={dp_size}")
    if quantization is not None:
        unsupported.append(f"quantization={quantization!r}")
    if unsupported:
        raise WeightHeterogeneousTransferError(
            "weight-cache heterogeneous transfer supports only single-node, DP=1 "
            "and unquantized weights; unsupported: " + ", ".join(unsupported)
        )
    try:
        WeightParallelLayout(tp_size=tp_size, pp_size=pp_size, ep_size=ep_size)
    except ValueError as error:
        raise WeightHeterogeneousTransferError(str(error)) from error


def _resolve_active_moe_runner_backend(model: Any, configured: str) -> str:
    """Return the backend that owns the model's in-memory MoE weight layout.

    ``auto`` remains in the published config even after an unquantized
    FusedMoE layer has selected its concrete runner.  The manifest adapter
    needs that concrete value because runner-specific layouts are not
    interchangeable.
    """
    active = set()
    for module in model.modules():
        quant_method = getattr(module, "quant_method", None)
        for runner_name in ("runner", "_aiter_runner"):
            runner = getattr(quant_method, runner_name, None)
            backend = getattr(runner, "runner_backend", None)
            backend = getattr(backend, "value", backend)
            if backend and backend != "auto":
                active.add(str(backend))
    if len(active) > 1:
        raise WeightHeterogeneousTransferError(
            "weight manifest requires one concrete MoE runner backend; "
            f"model has {sorted(active)}"
        )
    return next(iter(active), configured)


def _build_weight_runtime_inventory(
    *,
    model: Any,
    model_config: Any,
    tp_size: int,
    tp_rank: int,
    pp_size: int,
    pp_rank: int,
    ep_size: int,
    gpu_id: int,
    transfer_endpoint: str,
    model_id: str,
    revision: str,
):
    from sglang.srt.distributed.parallel_state import (
        get_attn_tensor_model_parallel_rank,
        get_attn_tensor_model_parallel_world_size,
        get_moe_expert_parallel_rank,
        get_moe_expert_parallel_world_size,
        get_moe_tensor_parallel_rank,
        get_moe_tensor_parallel_world_size,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_world_size,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )
    from sglang.srt.runtime_context import get_exec

    parallel_topology = WeightParallelTopology(
        dp_rank=0,
        dp_size=1,
        tp_rank=get_tensor_model_parallel_rank(),
        tp_size=get_tensor_model_parallel_world_size(),
        pp_rank=get_pipeline_model_parallel_rank(),
        pp_size=get_pipeline_model_parallel_world_size(),
        ep_rank=get_moe_expert_parallel_rank(),
        ep_size=get_moe_expert_parallel_world_size(),
        moe_tp_rank=get_moe_tensor_parallel_rank(),
        moe_tp_size=get_moe_tensor_parallel_world_size(),
        attention_tp_rank=get_attn_tensor_model_parallel_rank(),
        attention_tp_size=get_attn_tensor_model_parallel_world_size(),
    )
    expected_topology = (tp_size, tp_rank, pp_size, pp_rank, ep_size)
    actual_topology = (
        parallel_topology.tp_size,
        parallel_topology.tp_rank,
        parallel_topology.pp_size,
        parallel_topology.pp_rank,
        parallel_topology.ep_size,
    )
    if actual_topology != expected_topology:
        raise WeightHeterogeneousTransferError(
            "distributed parallel topology differs from daemon arguments: "
            f"expected={expected_topology}, actual={actual_topology}"
        )
    manifest_builder = create_weight_runtime_manifest_builder(
        model=model,
        config=model_config.hf_config,
        topology=parallel_topology,
        is_multimodal=model_config.is_multimodal,
        moe_runner_backend=_resolve_active_moe_runner_backend(
            model, get_exec().moe.moe_runner_backend
        ),
    )
    worker_id = (
        f"weight-cache:{os.getpid()}:gpu{gpu_id}:pp{pp_rank}:tp{tp_rank}:"
        f"ep{parallel_topology.ep_rank}:mtp{parallel_topology.moe_tp_rank}"
    )
    runtime_inventory = manifest_builder.build(
        model_id=model_id,
        revision=revision,
        instance_id=worker_id,
        worker_id=worker_id,
        endpoint=transfer_endpoint,
    )
    return runtime_inventory


def _initialize_weight_transfer_engine():
    try:
        from mooncake.engine import TransferEngine
    except ImportError as error:
        raise WeightHeterogeneousTransferError(
            "Mooncake Transfer Engine is required for heterogeneous weight "
            "transfer. Run the "
            "daemon from the manifest conda environment."
        ) from error

    transfer_engine = TransferEngine()
    transfer_host = get_local_ip_auto()
    transfer_protocol = envs.MOONCAKE_PROTOCOL.get()
    transfer_device = envs.MOONCAKE_DEVICE.get()
    result = transfer_engine.initialize(
        transfer_host,
        "P2PHANDSHAKE",
        transfer_protocol,
        transfer_device,
    )
    if result != 0:
        raise WeightHeterogeneousTransferError(
            "Mooncake TransferEngine.initialize failed: "
            f"protocol={transfer_protocol!r}, device={transfer_device!r}, "
            f"result={result}"
        )
    transfer_endpoint = NetworkAddress(
        transfer_host, transfer_engine.get_rpc_port()
    ).to_host_port_str()
    return transfer_engine, transfer_endpoint


@dataclass
class WeightsManifestState:
    transfer_engine: Any
    runtime_inventory: Any
    runtime_manifest: Any
    registered_memory_ranges: tuple[tuple[int, int], ...]
    fragment_registrations: tuple[Any, ...]
    source_content_checksums: tuple[dict[str, Any], ...]

    @classmethod
    def create(
        cls,
        *,
        model: Any,
        model_config: Any,
        tp_size: int,
        tp_rank: int,
        pp_size: int,
        pp_rank: int,
        ep_size: int,
        gpu_id: int,
        revision: str,
        is_source_daemon: bool,
    ) -> WeightsManifestState:
        try:
            from mooncake.weight_transfer import (
                MemoryRegistrationLease,
                RuntimeManifest,
            )
        except ImportError as error:
            raise WeightHeterogeneousTransferError(
                "mooncake.weight_transfer is unavailable in this environment"
            ) from error

        transfer_engine, transfer_endpoint = _initialize_weight_transfer_engine()
        registered_memory_ranges: list[tuple[int, int]] = []
        try:
            runtime_inventory = _build_weight_runtime_inventory(
                model=model,
                model_config=model_config,
                tp_size=tp_size,
                tp_rank=tp_rank,
                pp_size=pp_size,
                pp_rank=pp_rank,
                ep_size=ep_size,
                gpu_id=gpu_id,
                transfer_endpoint=transfer_endpoint,
                model_id=model_identity_from_config(model_config.hf_config),
                revision=revision,
            )
            runtime_manifest = RuntimeManifest.from_runtime_inventory(runtime_inventory)
            registered_memory_ranges.extend(
                _register_weight_fragment_allocations(
                    transfer_engine,
                    runtime_manifest.fragments,
                )
            )
            fragment_registrations = (
                ()
                if is_source_daemon
                else tuple(
                    MemoryRegistrationLease.from_fragment(fragment)
                    for fragment in runtime_manifest.fragments
                )
            )
            return cls(
                transfer_engine=transfer_engine,
                runtime_inventory=runtime_inventory,
                runtime_manifest=runtime_manifest,
                registered_memory_ranges=tuple(registered_memory_ranges),
                fragment_registrations=fragment_registrations,
                source_content_checksums=(
                    _compute_weight_content_checksums(model, runtime_inventory)
                    if is_source_daemon
                    else ()
                ),
            )
        except BaseException:
            for address, _ in reversed(registered_memory_ranges):
                try:
                    transfer_engine.unregister_memory(address)
                except Exception:
                    logger.exception(
                        "Failed to unregister Mooncake memory after setup error"
                    )
            raise

    def runtime_inventory_for_wire(self) -> Any:
        return msgspec.to_builtins(self.runtime_inventory)

    def close(self) -> None:
        failures = []
        for address, _ in reversed(self.registered_memory_ranges):
            try:
                result = self.transfer_engine.unregister_memory(address)
            except Exception as error:
                failures.append((address, repr(error)))
                continue
            if result != 0:
                failures.append((address, result))
        if failures:
            logger.error("weight manifest cleanup failures: %s", failures)


@dataclass(frozen=True)
class SourceWeightsManifest:
    node_id: str
    parallel_layout: WeightParallelLayout
    gpu_ids: tuple[int, ...]
    runtime_inventories: tuple[Any, ...]
    content_checksum_groups: tuple[tuple[dict[str, Any], ...], ...]

    def to_wire(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parallel_layout": {
                "tp_size": self.parallel_layout.tp_size,
                "pp_size": self.parallel_layout.pp_size,
                "ep_size": self.parallel_layout.ep_size,
            },
            "gpu_ids": self.gpu_ids,
            "runtime_inventories": self.runtime_inventories,
            "content_checksum_groups": self.content_checksum_groups,
        }

    @classmethod
    def from_wire(cls, wire_manifest: dict[str, Any]) -> SourceWeightsManifest:
        try:
            node_id = str(wire_manifest["node_id"])
            parallel_layout_data = wire_manifest["parallel_layout"]
            parallel_layout = WeightParallelLayout(
                tp_size=int(parallel_layout_data["tp_size"]),
                pp_size=int(parallel_layout_data["pp_size"]),
                ep_size=int(parallel_layout_data["ep_size"]),
            )
            gpu_ids = tuple(int(value) for value in wire_manifest["gpu_ids"])
            runtime_inventories = tuple(wire_manifest["runtime_inventories"])
            content_checksum_groups = tuple(
                tuple(group) for group in wire_manifest["content_checksum_groups"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise WeightHeterogeneousTransferError(
                "registry returned an invalid source weight manifest"
            ) from error

        rank_counts = (
            len(gpu_ids),
            len(runtime_inventories),
            len(content_checksum_groups),
        )
        if rank_counts != (parallel_layout.world_size,) * 3:
            raise WeightHeterogeneousTransferError(
                "source manifest rank count mismatch: "
                f"expected={parallel_layout.world_size}, actual={rank_counts}"
            )
        if len(gpu_ids) != len(set(gpu_ids)) or any(
            not group for group in content_checksum_groups
        ):
            raise WeightHeterogeneousTransferError(
                "source manifest GPU ranks or checksums are invalid"
            )
        return cls(
            node_id=node_id,
            parallel_layout=parallel_layout,
            gpu_ids=gpu_ids,
            runtime_inventories=runtime_inventories,
            content_checksum_groups=content_checksum_groups,
        )


def register_source_weights_manifest(
    registry_url: str,
    *,
    global_rank: int,
    gpu_id: int,
    tp_size: int,
    pp_size: int,
    ep_size: int,
    manifest_state: WeightsManifestState,
    timeout: float = 30.0,
) -> None:
    """Register one immutable source rank with the manifest server."""
    import requests

    payload = {
        "node_id": get_local_ip_auto(fallback=socket.gethostname()),
        "global_rank": global_rank,
        "gpu_id": gpu_id,
        "parallel_layout": {
            "tp_size": tp_size,
            "pp_size": pp_size,
            "ep_size": ep_size,
        },
        "runtime_inventory": manifest_state.runtime_inventory_for_wire(),
        "content_checksums": manifest_state.source_content_checksums,
    }
    url = registry_url.rstrip("/") + "/register_weight_manifest"
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = requests.put(url, json=payload, timeout=5)
            if response.status_code == 200:
                return
            if response.status_code == 409:
                raise WeightHeterogeneousTransferError(
                    f"manifest server rejected source rank: {response.text}"
                )
            last_error = RuntimeError(
                f"manifest server returned {response.status_code}: {response.text}"
            )
        except WeightHeterogeneousTransferError:
            raise
        except Exception as error:
            last_error = error
        time.sleep(0.2)
    raise WeightHeterogeneousTransferError(
        f"failed to register source manifest at {url}: {last_error}"
    )


def fetch_source_weights_manifest(
    registry_url: str,
    *,
    timeout: float = 30.0,
) -> SourceWeightsManifest:
    """Wait for and fetch one complete source manifest from the registry."""
    import requests

    url = registry_url.rstrip("/") + "/get_source_weights_manifest"
    deadline = time.monotonic() + timeout
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return SourceWeightsManifest.from_wire(
                    response.json()["source_weights_manifest"]
                )
            if response.status_code == 409:
                last_error = RuntimeError(response.text)
            else:
                last_error = RuntimeError(
                    f"manifest server returned {response.status_code}: {response.text}"
                )
        except Exception as error:
            last_error = error
        time.sleep(0.2)
    raise WeightHeterogeneousTransferError(
        f"failed to fetch source manifest from {url}: {last_error}"
    )


def validate_transferred_weights(
    *,
    target_manifest_state: WeightsManifestState,
    model: Any,
    source_content_checksum_groups: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, int]:
    """Validate logical weight bytes across source and target layouts."""
    import torch.distributed as dist

    target_content_checksums = _compute_weight_content_checksums(
        model, target_manifest_state.runtime_inventory
    )
    target_rank_checksums: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(target_rank_checksums, target_content_checksums)
    expected_weights = _aggregate_weight_content_checksums(
        source_content_checksum_groups
    )
    transferred_weights = _aggregate_weight_content_checksums(target_rank_checksums)
    if expected_weights != transferred_weights:
        source_ids = set(expected_weights)
        target_ids = set(transferred_weights)
        differing = sorted(
            tensor_id
            for tensor_id in source_ids & target_ids
            if expected_weights[tensor_id] != transferred_weights[tensor_id]
        )
        raise WeightHeterogeneousTransferError(
            "heterogeneous weight transfer checksum mismatch: "
            f"source_only={sorted(source_ids - target_ids)[:5]}, "
            f"target_only={sorted(target_ids - source_ids)[:5]}, "
            f"differing={differing[:5]}"
        )
    return {
        "validated_tensors": len(expected_weights),
        "validated_bytes": sum(item[1] for item in expected_weights.values()),
    }


def pull_weights_from_source(
    *,
    target_manifest_state: WeightsManifestState,
    source_runtime_inventories: Sequence[Any],
) -> dict[str, int]:
    try:
        from mooncake.weight_transfer import (
            MemoryRegistrationLease,
            MooncakeTransferEngineReader,
            RuntimeManifest,
            plan_runtime_transfer_to_local_target,
        )
    except ImportError as error:
        raise WeightHeterogeneousTransferError(
            "mooncake.weight_transfer is unavailable in this environment"
        ) from error

    source_manifests = tuple(
        RuntimeManifest.from_runtime_inventory(runtime_inventory)
        for runtime_inventory in source_runtime_inventories
    )
    if not source_manifests:
        raise WeightHeterogeneousTransferError("source manifest set is empty")
    for source_manifest in source_manifests:
        if (
            source_manifest.model_id != target_manifest_state.runtime_manifest.model_id
            or source_manifest.revision
            != target_manifest_state.runtime_manifest.revision
        ):
            raise WeightHeterogeneousTransferError(
                "source and target manifest model identity differs: "
                f"source={source_manifest.model_id}@{source_manifest.revision}, "
                f"target={target_manifest_state.runtime_manifest.model_id}@"
                f"{target_manifest_state.runtime_manifest.revision}"
            )

    source_fragment_registrations = tuple(
        MemoryRegistrationLease.from_fragment(
            fragment,
        )
        for source_manifest in source_manifests
        for fragment in source_manifest.fragments
    )
    transfer_plan = plan_runtime_transfer_to_local_target(
        source_manifests,
        target_manifest_state.runtime_manifest,
    )
    if transfer_plan.total_bytes <= 0 or not transfer_plan.operations:
        raise WeightHeterogeneousTransferError(
            "Mooncake produced an empty heterogeneous weight transfer plan"
        )

    transfer_reader = MooncakeTransferEngineReader(
        target_manifest_state.transfer_engine,
        max_batch_operations=8192,
    )
    transfer_receipts = transfer_reader.execute(
        transfer_plan,
        source_manifests,
        target_manifest_state.runtime_manifest,
        source_pre_registered=True,
        source_registrations=source_fragment_registrations,
        target_pre_registered=True,
        target_registrations=target_manifest_state.fragment_registrations,
    )
    wire_bytes = sum(receipt.nbytes for receipt in transfer_receipts)
    wire_operations = sum(receipt.operation_count for receipt in transfer_receipts)
    if wire_bytes != transfer_plan.total_bytes:
        raise WeightHeterogeneousTransferError(
            "incomplete heterogeneous weight transfer: "
            f"planned={transfer_plan.total_bytes}, "
            f"completed={wire_bytes}"
        )
    logger.info(
        "heterogeneous weight transfer complete: logical_bytes=%d operations=%d "
        "endpoints=%d",
        transfer_plan.total_bytes,
        wire_operations,
        len(transfer_receipts),
    )
    return {
        "logical_bytes": int(transfer_plan.total_bytes),
        "wire_bytes": int(wire_bytes),
        "wire_operations": int(wire_operations),
        "source_ranks": len(source_manifests),
    }


def transfer_weights_from_source_daemons(
    *,
    target_manifest_state: WeightsManifestState,
    model: Any,
    gpu_id: int,
    registry_url: str,
) -> dict[str, int]:
    """Complete a target copy using only the source and target daemons."""
    import torch.distributed as dist

    # Fetch the source once, then broadcast it to every target daemon rank.
    source_manifest_result: list[Optional[dict[str, Any]]] = [None]
    if dist.get_rank() == 0:
        try:
            source_manifest_result[0] = {
                "manifest": fetch_source_weights_manifest(registry_url),
                "error": None,
            }
        except Exception as error:
            source_manifest_result[0] = {
                "manifest": None,
                "error": str(error),
            }
    dist.broadcast_object_list(source_manifest_result, src=0)
    result = source_manifest_result[0]
    if result is None:
        raise WeightHeterogeneousTransferError(
            "target rank did not receive a manifest registry result"
        )
    if result["error"] is not None:
        raise WeightHeterogeneousTransferError(result["error"])
    source_weights_manifest = result["manifest"]

    # Enforce the deliberately narrow no-overlap contract on a shared node
    # before planning or writing any target weight bytes.
    target_gpu_ids: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(target_gpu_ids, gpu_id)
    target_gpu_ids = [int(value) for value in target_gpu_ids]
    if len(target_gpu_ids) != len(set(target_gpu_ids)):
        raise WeightHeterogeneousTransferError(
            f"target GPU ranks are not unique: {tuple(target_gpu_ids)}"
        )
    target_node_id = get_local_ip_auto(fallback=socket.gethostname())
    if source_weights_manifest.node_id == target_node_id:
        overlap = sorted(set(source_weights_manifest.gpu_ids) & set(target_gpu_ids))
        if overlap:
            raise WeightHeterogeneousTransferError(
                "source and target GPU ranks must not overlap on the same node; "
                f"overlapping ranks: {overlap}"
            )

    transfer_stats = pull_weights_from_source(
        target_manifest_state=target_manifest_state,
        source_runtime_inventories=source_weights_manifest.runtime_inventories,
    )
    rebuild_transferred_weight_state(model)
    transfer_stats.update(
        validate_transferred_weights(
            target_manifest_state=target_manifest_state,
            model=model,
            source_content_checksum_groups=(
                source_weights_manifest.content_checksum_groups
            ),
        )
    )
    return transfer_stats
