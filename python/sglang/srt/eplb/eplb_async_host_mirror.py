import atexit
import contextlib
import ctypes
import hashlib
import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed import P2POp
from tqdm.auto import tqdm

from sglang.srt.distributed import get_world_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import (
    ModelConfigForExpertLocation,
    get_global_expert_location_metadata,
)

logger = logging.getLogger(__name__)

_NCCL_UNSUPPORTED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


def _nccl_safe_view(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret tensors with NCCL-unsupported dtypes as uint8 for P2P ops.
    The underlying storage is shared, so after irecv the original tensor
    reads the correct bytes."""
    if t.dtype in _NCCL_UNSUPPORTED_DTYPES:
        return t.view(torch.uint8)
    return t


def _sanitize_name(raw: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", raw)


def _get_host_mirror_model_name(model_config) -> str:
    architectures = getattr(model_config.hf_config, "architectures", None)
    if architectures and len(architectures) > 0:
        return architectures[0]
    return (
        os.path.basename(model_config.model_path.rstrip("/")) or model_config.model_path
    )


def _element_size(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def _format_nbytes(num_bytes: int) -> str:
    return f"{num_bytes} B ({num_bytes / (1024 * 1024 * 1024):.2f} GB)"


def _compute_node_owner_physical_ids(
    physical_to_logical_map: torch.Tensor,
    *,
    node_physical_start: int,
    node_physical_end: int,
    num_logical_experts: int,
) -> list[int]:
    owners = [-1] * num_logical_experts
    for global_physical_expert_id in range(node_physical_start, node_physical_end):
        logical_expert_id = int(
            physical_to_logical_map[global_physical_expert_id].item()
        )
        if owners[logical_expert_id] == -1:
            owners[logical_expert_id] = global_physical_expert_id
    return owners


def _compute_node_owner_global_ranks(
    physical_to_logical_map: torch.Tensor,
    *,
    node_physical_start: int,
    node_physical_end: int,
    num_local_physical_experts: int,
    num_logical_experts: int,
) -> list[int]:
    """Compute the owner global rank for each logical expert within a node.

    Returns a list of length ``num_logical_experts``.  Entry *i* is the
    **global** rank that owns logical expert *i* on the given node, or -1 if
    the node does not host that expert.

    Note: ``global_physical_expert_id // num_local_physical_experts`` yields the
    global rank (not the node-local rank).  Callers convert to a local rank via
    ``% local_world_size`` when needed.
    """
    owner_physical_ids = _compute_node_owner_physical_ids(
        physical_to_logical_map,
        node_physical_start=node_physical_start,
        node_physical_end=node_physical_end,
        num_logical_experts=num_logical_experts,
    )
    owner_global_ranks = [-1] * num_logical_experts
    for logical_expert_id, global_physical_expert_id in enumerate(owner_physical_ids):
        if global_physical_expert_id == -1:
            continue
        owner_global_ranks[logical_expert_id] = (
            global_physical_expert_id // num_local_physical_experts
        )
    return owner_global_ranks


def _build_cross_node_transfer_plan(
    availability: torch.Tensor,
) -> list[tuple[int, int, int]]:
    plan = []
    num_nodes, num_logical_experts = availability.shape
    for logical_expert_id in range(num_logical_experts):
        src_nodes = torch.nonzero(
            availability[:, logical_expert_id], as_tuple=False
        ).flatten()
        if src_nodes.numel() == 0:
            continue
        src_node_rank = int(src_nodes[0].item())
        for dst_node_rank in range(num_nodes):
            if int(availability[dst_node_rank, logical_expert_id].item()) == 0:
                plan.append((logical_expert_id, src_node_rank, dst_node_rank))
    return plan


def _build_node_availability_from_metadata(
    physical_to_logical_map: torch.Tensor,
    *,
    num_nodes: int,
    local_world_size: int,
    num_local_physical_experts: int,
    num_logical_experts: int,
) -> torch.Tensor:
    availability = torch.zeros((num_nodes, num_logical_experts), dtype=torch.uint8)
    experts_per_node = local_world_size * num_local_physical_experts
    for node_rank in range(num_nodes):
        node_physical_start = node_rank * experts_per_node
        node_physical_end = node_physical_start + experts_per_node
        owners = _compute_node_owner_physical_ids(
            physical_to_logical_map,
            node_physical_start=node_physical_start,
            node_physical_end=node_physical_end,
            num_logical_experts=num_logical_experts,
        )
        for logical_expert_id, global_physical_expert_id in enumerate(owners):
            if global_physical_expert_id != -1:
                availability[node_rank, logical_expert_id] = 1
    return availability


def _group_cross_node_transfer_plan(
    transfer_plan: list[tuple[int, int, int]],
) -> list[tuple[int, int, tuple[int, ...]]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for logical_expert_id, src_node_rank, dst_node_rank in transfer_plan:
        grouped.setdefault((src_node_rank, dst_node_rank), []).append(logical_expert_id)
    return [
        (src_node_rank, dst_node_rank, tuple(logical_expert_ids))
        for (src_node_rank, dst_node_rank), logical_expert_ids in sorted(
            grouped.items()
        )
    ]


def _build_rank_parallel_cross_node_transfer_plan(
    availability: torch.Tensor,
    *,
    physical_to_logical_map: torch.Tensor,
    local_world_size: int,
    num_local_physical_experts: int,
    num_logical_experts: int,
) -> list[tuple[int, int, int, int, tuple[int, ...]]]:
    grouped: dict[tuple[int, int, int, int], list[int]] = {}
    num_nodes = availability.shape[0]
    experts_per_node = local_world_size * num_local_physical_experts

    owner_global_ranks_by_node = []
    for node_rank in range(num_nodes):
        node_physical_start = node_rank * experts_per_node
        node_physical_end = node_physical_start + experts_per_node
        owner_global_ranks_by_node.append(
            _compute_node_owner_global_ranks(
                physical_to_logical_map,
                node_physical_start=node_physical_start,
                node_physical_end=node_physical_end,
                num_local_physical_experts=num_local_physical_experts,
                num_logical_experts=num_logical_experts,
            )
        )

    for logical_expert_id in range(num_logical_experts):
        src_nodes = torch.nonzero(
            availability[:, logical_expert_id], as_tuple=False
        ).flatten()
        if src_nodes.numel() == 0:
            continue
        src_node_rank = int(src_nodes[0].item())
        src_global_rank = owner_global_ranks_by_node[src_node_rank][logical_expert_id]
        if src_global_rank < 0:
            continue
        src_local_rank = src_global_rank % local_world_size
        dst_local_rank = logical_expert_id % local_world_size
        for dst_node_rank in range(num_nodes):
            if int(availability[dst_node_rank, logical_expert_id].item()) == 1:
                continue
            grouped.setdefault(
                (src_node_rank, src_local_rank, dst_node_rank, dst_local_rank), []
            ).append(logical_expert_id)

    return [
        (
            src_node_rank,
            src_local_rank,
            dst_node_rank,
            dst_local_rank,
            tuple(logical_expert_ids),
        )
        for (
            src_node_rank,
            src_local_rank,
            dst_node_rank,
            dst_local_rank,
        ), logical_expert_ids in sorted(grouped.items())
    ]


def _compute_local_owner_indices(
    physical_to_logical_map: torch.Tensor,
    *,
    rank: int,
    num_local_physical_experts: int,
    node_physical_start: int,
    node_physical_end: int,
    num_logical_experts: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_owner_physical_ids = _compute_node_owner_physical_ids(
        physical_to_logical_map,
        node_physical_start=node_physical_start,
        node_physical_end=node_physical_end,
        num_logical_experts=num_logical_experts,
    )
    local_physical_start = rank * num_local_physical_experts
    local_expert_ids = []
    logical_expert_ids = []
    for local_expert_id in range(num_local_physical_experts):
        global_physical_expert_id = local_physical_start + local_expert_id
        logical_expert_id = int(
            physical_to_logical_map[global_physical_expert_id].item()
        )
        if node_owner_physical_ids[logical_expert_id] == global_physical_expert_id:
            local_expert_ids.append(local_expert_id)
            logical_expert_ids.append(logical_expert_id)
    return (
        torch.tensor(local_expert_ids, dtype=torch.int64),
        torch.tensor(logical_expert_ids, dtype=torch.int64),
    )


@dataclass
class _ShmRecord:
    shm: shared_memory.SharedMemory
    tensor: torch.Tensor
    name: str
    is_owner: bool
    unlink_on_close: bool


class EPLBAsyncHostMirrorManager:
    def __init__(self, server_args, model_config):
        config = ModelConfigForExpertLocation.from_model_config(model_config)
        assert config is not None, "EPLB async requires a MoE model."

        self._server_args = server_args
        self._world_group = get_world_group()
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._local_world_size = self._world_group.local_size or max(
            1, self._world_size // max(1, server_args.nnodes)
        )
        self._local_rank = self._rank % self._local_world_size
        self._node_rank = server_args.node_rank
        self._num_nodes = server_args.nnodes
        self._is_owner = self._rank % self._local_world_size == 0
        self._num_logical_experts = config.num_logical_experts
        self._records: dict[tuple[int, int], _ShmRecord] = {}
        self._valid_records: dict[int, _ShmRecord] = {}
        self._layer_tensors: dict[int, list[torch.Tensor]] = {}
        self._dummy_layer_tensors: dict[int, list[torch.Tensor]] = {}
        self._layer_num_tensors: dict[int, int] = {}
        self._tensor_devices: dict[tuple[int, int], torch.device] = {}
        self._gpu_staging_buffers: dict[
            tuple[torch.dtype, str, tuple[int, ...]], torch.Tensor
        ] = {}
        self._cpu_transfer_buffers: dict[
            tuple[torch.dtype, tuple[int, ...]], torch.Tensor
        ] = {}
        self._closed = False
        self._registered_atexit = False
        self._lock = threading.Lock()
        self._reuse_existing_shm = envs.SGLANG_EPLB_ASYNC_HOST_MIRROR_REUSE_SHM.get()
        self._dummy_h2d = envs.SGLANG_EPLB_ASYNC_DUMMY_H2D.get()
        self._checksum_shm = None
        self._checksum_shm_name = None

        model_name = _sanitize_name(_get_host_mirror_model_name(model_config))
        master_port = _sanitize_name(os.environ.get("MASTER_PORT", "0"))
        # Note: base_name intentionally excludes node_rank because host mirror
        # data should be identical across all nodes. Using the same name allows
        # each node to independently verify and reuse local shared memory.
        self._base_name = f"sglang_eplb_async_{model_name}_p{master_port}"

    def build_from_loaded_model(self, routed_experts_weights_of_layer) -> None:
        if self._dummy_h2d:
            self._build_dummy_from_loaded_model(routed_experts_weights_of_layer)
            return

        metadata = get_global_expert_location_metadata()
        assert metadata is not None, "EPLB async host mirror requires expert metadata."

        build_start = time.time()
        num_layers = len(routed_experts_weights_of_layer)
        num_tensor_records = sum(
            len(tensors) for tensors in routed_experts_weights_of_layer.values()
        )
        num_record_attach_steps = (
            num_layers + num_tensor_records if self._world_size > 1 else 0
        )
        num_local_fill_steps = num_layers * metadata.num_local_physical_experts
        layer_transfer_plan = self._build_grouped_cross_node_transfer_plan(
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
            metadata=metadata,
        )
        (
            local_remote_transfer_steps,
            remote_transfer_experts,
            remote_transfer_bytes,
        ) = self._summarize_transfer_plan(
            layer_transfer_plan,
            routed_experts_weights_of_layer=routed_experts_weights_of_layer,
        )
        pbar = tqdm(
            total=(
                num_layers
                + num_tensor_records
                + num_record_attach_steps
                + num_local_fill_steps
                + local_remote_transfer_steps
                + num_tensor_records
            ),
            desc="Building EPLB async host mirror",
            disable=self._rank != 0,
            dynamic_ncols=True,
        )
        reused_existing_data = False
        phase_times = {
            "create_records": 0.0,
            "populate_local": 0.0,
            "remote_transfer": 0.0,
            "attach": 0.0,
        }

        try:
            # Early pre-check: attach only valid tensors to quickly determine
            # if existing SHM data is complete, avoiding full attach + populate
            # when reuse is possible.
            phase_start = time.time()
            reused_existing_data = self._preflight_reuse_check(
                routed_experts_weights_of_layer
            )
            phase_times["create_records"] = time.time() - phase_start
            if reused_existing_data:
                pbar.update(num_local_fill_steps + local_remote_transfer_steps)
                remote_transfer_experts = 0
                remote_transfer_bytes = 0
            else:
                # Preflight failed — clean up stale SHM records before rebuild.
                self._cleanup_stale_records()
                self._barrier_all_ranks()
                self._create_records(routed_experts_weights_of_layer, pbar=pbar)
                phase_times["create_records"] = time.time() - phase_start
                phase_start = time.time()
                self._populate_local_node_shards(
                    routed_experts_weights_of_layer=routed_experts_weights_of_layer,
                    metadata=metadata,
                    pbar=pbar,
                )
                phase_times["populate_local"] = time.time() - phase_start
                self._barrier_all_ranks()
                phase_start = time.time()
                self._fill_missing_from_remote_nodes(
                    layer_transfer_plan=layer_transfer_plan,
                    pbar=pbar,
                )
                phase_times["remote_transfer"] = time.time() - phase_start
                self._barrier_all_ranks()
            self._validate_completeness()

            phase_start = time.time()
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                attached = []
                for tensor_index, _ in enumerate(tensors):
                    record = self._records.get((layer_id, tensor_index))
                    if record is None:
                        raise RuntimeError(
                            f"EPLB async host mirror is incomplete for layer={layer_id} tensor_index={tensor_index}."
                        )
                    attached.append(record.tensor)
                    pbar.update(1)
                self._layer_tensors[layer_id] = attached
            phase_times["attach"] = time.time() - phase_start
        finally:
            pbar.close()

        self._maybe_register_atexit()

        # Store weight checksum for future reuse verification.
        if not reused_existing_data:
            self._store_checksum_after_build(routed_experts_weights_of_layer)
            self._barrier_all_ranks()

        if self._rank == 0:
            shm_nbytes = self._get_total_shm_nbytes()
            logger.info(
                "EPLB async host mirror build complete: reused_existing_data=%s "
                "layers=%s tensor_records=%s remote_transfer_experts=%s "
                "remote_transfer_bytes=%s shm=%s "
                "create_records=%.2fs populate_local=%.2fs remote_transfer=%.2fs "
                "attach=%.2fs elapsed=%.2fs",
                reused_existing_data,
                num_layers,
                num_tensor_records,
                remote_transfer_experts,
                _format_nbytes(remote_transfer_bytes),
                _format_nbytes(shm_nbytes),
                phase_times["create_records"],
                phase_times["populate_local"],
                phase_times["remote_transfer"],
                phase_times["attach"],
                time.time() - build_start,
            )

    def get_expert_tensors(self, layer_id: int, logical_expert_id: int):
        if self._dummy_h2d:
            if layer_id not in self._dummy_layer_tensors:
                raise KeyError(
                    f"Layer {layer_id} does not exist in async dummy host mirror."
                )
            return self._dummy_layer_tensors[layer_id]
        if layer_id not in self._layer_tensors:
            raise KeyError(f"Layer {layer_id} does not exist in async host mirror.")
        return [tensor[logical_expert_id] for tensor in self._layer_tensors[layer_id]]

    def close(self):
        if self._closed:
            return
        self._closed = True

        # Never unlink SHM — it must persist across server restarts.
        # Only close file descriptors. No resource_tracker.unregister needed
        # because SHM was created with track=False (register patched to no-op).
        for record in self._records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
        for record in self._valid_records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
        if self._checksum_shm is not None:
            with contextlib.suppress(Exception):
                self._checksum_shm.close()
        self._records.clear()
        self._valid_records.clear()
        self._layer_tensors.clear()
        self._dummy_layer_tensors.clear()
        self._layer_num_tensors.clear()
        self._tensor_devices.clear()
        self._gpu_staging_buffers.clear()
        self._cpu_transfer_buffers.clear()

    def _build_dummy_from_loaded_model(self, routed_experts_weights_of_layer) -> None:
        self._dummy_layer_tensors.clear()
        self._layer_num_tensors.clear()

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
            fake_tensors = []
            for tensor in tensors:
                fake = torch.empty_like(tensor[0], device="cpu", pin_memory=True)
                fake.zero_()
                fake_tensors.append(fake)
            self._dummy_layer_tensors[layer_id] = fake_tensors

        self._maybe_register_atexit()
        if self._rank == 0:
            logger.info(
                "EPLB async dummy H2D enabled: skipped host mirror build, layers=%s tensor_records=%s",
                len(routed_experts_weights_of_layer),
                sum(
                    len(tensors) for tensors in routed_experts_weights_of_layer.values()
                ),
            )

    def _preflight_reuse_check(self, routed_experts_weights_of_layer) -> bool:
        """Check if existing SHM data can be reused.

        Two conditions must be met:
        1. All valid bits are 1 (SHM data is complete)
        2. Weight checksum matches (device weights unchanged since last build)

        Returns ``True`` when SHM data is complete and consistent
        (skip populate/transfer); ``False`` otherwise.
        """
        if not self._reuse_existing_shm:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._layer_num_tensors[layer_id] = len(tensors)
            return False

        can_skip_check = len(routed_experts_weights_of_layer) == 0
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
            if not can_skip_check:
                self._get_or_create_valid_tensor(layer_id)
        self._barrier_all_ranks()

        # Phase 1: Check valid bits (SHM data is complete)
        local_all_valid = can_skip_check or all(
            torch.all(record.tensor == 1) for record in self._valid_records.values()
        )

        # Phase 2: Check weight checksum (device weights unchanged)
        local_checksum_ok = can_skip_check
        if local_all_valid and not can_skip_check:
            local_checksum_ok = self._verify_checksum(routed_experts_weights_of_layer)

        # Both conditions must be met on ALL ranks
        local_ok = local_all_valid and local_checksum_ok
        all_ok = self._all_gather_bool(local_ok)

        if not all_ok:
            if self._rank == 0:
                logger.info(
                    "EPLB async host mirror reuse check failed: "
                    "local_all_valid=%s local_checksum_ok=%s. Will rebuild.",
                    local_all_valid,
                    local_checksum_ok,
                )
            return False

        # Data is complete and consistent — attach all data tensors.
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            for tensor_index, tensor in enumerate(tensors):
                self._get_or_create_tensor(
                    layer_id=layer_id,
                    tensor_index=tensor_index,
                    sample_tensor=tensor,
                )
        self._barrier_all_ranks()
        return True

    def _all_gather_bool(self, local_value: bool) -> bool:
        """All-gather boolean values from all ranks and return True only if all are True."""
        if not dist.is_initialized() or self._world_size <= 1:
            return local_value
        tensor = torch.tensor([int(local_value)], dtype=torch.int32, device="cuda")
        gathered_list = [
            torch.zeros(1, dtype=torch.int32, device="cuda")
            for _ in range(self._world_size)
        ]
        dist.all_gather(gathered_list, tensor, group=self._world_group.device_group)
        return bool(torch.cat(gathered_list).all().item())

    def _maybe_register_atexit(self):
        if self._registered_atexit:
            return
        self._registered_atexit = True
        atexit.register(self.close)

    def _create_records(self, routed_experts_weights_of_layer, pbar=None) -> None:
        for layer_id, tensors in routed_experts_weights_of_layer.items():
            self._layer_num_tensors[layer_id] = len(tensors)
        if self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                if pbar is not None:
                    pbar.update(1)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )
                    if pbar is not None:
                        pbar.update(1)
        self._barrier_all_ranks()
        if not self._is_owner:
            for layer_id, tensors in routed_experts_weights_of_layer.items():
                self._get_or_create_valid_tensor(layer_id)
                for tensor_index, tensor in enumerate(tensors):
                    self._get_or_create_tensor(
                        layer_id=layer_id,
                        tensor_index=tensor_index,
                        sample_tensor=tensor,
                    )
        self._barrier_all_ranks()
        if pbar is not None and self._rank == 0 and self._world_size > 1:
            pbar.update(
                sum(
                    1 + len(tensors)
                    for tensors in routed_experts_weights_of_layer.values()
                )
            )

    def _populate_local_node_shards(
        self, *, routed_experts_weights_of_layer, metadata, pbar=None
    ) -> None:
        num_local_physical_experts = metadata.num_local_physical_experts
        node_physical_start = (
            self._node_rank * self._local_world_size * num_local_physical_experts
        )
        node_physical_end = (
            node_physical_start + self._local_world_size * num_local_physical_experts
        )

        # Pre-allocate CPU pinned buffer for D2H transfer to avoid GPU memory growth
        cpu_pinned_buffers: dict[tuple[torch.dtype, tuple[int, ...]], torch.Tensor] = {}

        for layer_id, tensors in routed_experts_weights_of_layer.items():
            local_expert_ids_cpu, logical_expert_ids_cpu = _compute_local_owner_indices(
                metadata.physical_to_logical_map_cpu[layer_id],
                rank=self._rank,
                num_local_physical_experts=num_local_physical_experts,
                node_physical_start=node_physical_start,
                node_physical_end=node_physical_end,
                num_logical_experts=self._num_logical_experts,
            )
            if local_expert_ids_cpu.numel() == 0:
                continue
            valid_tensor = self._valid_records[layer_id].tensor
            num_local_experts = local_expert_ids_cpu.numel()

            for tensor_index, tensor in enumerate(tensors):
                record = self._records[(layer_id, tensor_index)]
                tail_shape = tuple(tensor.shape[1:])
                buffer_key = (tensor.dtype, tail_shape)

                # Get or create CPU pinned buffer for this tensor shape
                cpu_buffer = cpu_pinned_buffers.get(buffer_key)
                if cpu_buffer is None or cpu_buffer.shape[0] < num_local_experts:
                    cpu_buffer = torch.empty(
                        (num_local_experts, *tail_shape),
                        dtype=tensor.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_pinned_buffers[buffer_key] = cpu_buffer

                # D2H: Copy entire tensor to CPU pinned memory, then index on CPU
                # This avoids creating temporary GPU tensors from index_select
                tensor_cpu = tensor.to(device="cpu", non_blocking=False)
                packed = tensor_cpu.index_select(0, local_expert_ids_cpu)
                _index_copy_host_tensor(record.tensor, logical_expert_ids_cpu, packed)

            valid_tensor.index_fill_(0, logical_expert_ids_cpu, 1)
            if pbar is not None:
                pbar.update(local_expert_ids_cpu.numel())

    def _fill_missing_from_remote_nodes(
        self, *, layer_transfer_plan, pbar=None
    ) -> None:
        if (not dist.is_initialized()) or self._num_nodes <= 1:
            return

        for layer_id in sorted(layer_transfer_plan):
            transfer_groups = layer_transfer_plan[layer_id]
            if len(transfer_groups) == 0:
                self._barrier_all_ranks()
                continue

            for (
                src_node_rank,
                src_local_rank,
                dst_node_rank,
                dst_local_rank,
                logical_expert_ids,
            ) in transfer_groups:
                if (self._node_rank, self._local_rank) not in (
                    (src_node_rank, src_local_rank),
                    (dst_node_rank, dst_local_rank),
                ):
                    continue
                self._transfer_logical_expert_group(
                    layer_id=layer_id,
                    src_node_rank=src_node_rank,
                    src_local_rank=src_local_rank,
                    dst_node_rank=dst_node_rank,
                    dst_local_rank=dst_local_rank,
                    logical_expert_ids=logical_expert_ids,
                )
                if pbar is not None:
                    pbar.update(self._layer_num_tensors[layer_id])

            self._barrier_all_ranks()

    def _transfer_logical_expert_group(
        self,
        *,
        layer_id: int,
        src_node_rank: int,
        src_local_rank: int,
        dst_node_rank: int,
        dst_local_rank: int,
        logical_expert_ids: tuple[int, ...],
    ) -> None:
        src_global_rank = src_node_rank * self._local_world_size + src_local_rank
        dst_global_rank = dst_node_rank * self._local_world_size + dst_local_rank
        logical_expert_ids_cpu = torch.tensor(logical_expert_ids, dtype=torch.int64)
        num_logical_experts = len(logical_expert_ids)

        if (self._node_rank, self._local_rank) == (dst_node_rank, dst_local_rank):
            valid_tensor = self._valid_records[layer_id].tensor
        else:
            valid_tensor = None

        p2p_ops = []
        recv_infos = []
        for tensor_index in range(self._layer_num_tensors[layer_id]):
            record = self._records[(layer_id, tensor_index)]
            gpu_staging, cpu_transfer = self._get_or_create_transfer_buffers(
                sample_tensor=record.tensor,
                device=self._tensor_devices[(layer_id, tensor_index)],
                num_logical_experts=num_logical_experts,
            )
            if (self._node_rank, self._local_rank) == (src_node_rank, src_local_rank):
                gpu_staging.copy_(
                    record.tensor.index_select(0, logical_expert_ids_cpu),
                    non_blocking=False,
                )
                p2p_ops.append(
                    P2POp(
                        op=dist.isend,
                        tensor=_nccl_safe_view(gpu_staging),
                        peer=dst_global_rank,
                        group=self._world_group.device_group,
                    )
                )
            elif (self._node_rank, self._local_rank) == (dst_node_rank, dst_local_rank):
                p2p_ops.append(
                    P2POp(
                        op=dist.irecv,
                        tensor=_nccl_safe_view(gpu_staging),
                        peer=src_global_rank,
                        group=self._world_group.device_group,
                    )
                )
                recv_infos.append((record.tensor, gpu_staging, cpu_transfer))

        self._execute_batch_p2p_ops(p2p_ops)

        if (self._node_rank, self._local_rank) == (dst_node_rank, dst_local_rank):
            for dst_tensor, gpu_staging, cpu_transfer in recv_infos:
                cpu_transfer.copy_(gpu_staging, non_blocking=False)
                _index_copy_host_tensor(
                    dst_tensor, logical_expert_ids_cpu, cpu_transfer
                )

        if valid_tensor is not None:
            valid_tensor.index_fill_(0, logical_expert_ids_cpu, 1)

    def _execute_batch_p2p_ops(self, p2p_ops: list[P2POp]) -> None:
        if len(p2p_ops) == 0:
            return
        reqs = dist.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    def _validate_completeness(self) -> None:
        for layer_id, valid_record in self._valid_records.items():
            missing = (
                torch.nonzero(valid_record.tensor == 0, as_tuple=False)
                .flatten()
                .tolist()
            )
            if missing:
                raise RuntimeError(
                    "EPLB async host mirror is incomplete after initialization: "
                    f"node_rank={self._node_rank} layer_id={layer_id} missing_logical_expert_ids={missing}"
                )

    def _barrier_all_ranks(self) -> None:
        if dist.is_initialized():
            self._world_group.barrier()

    def _get_or_create_tensor(
        self,
        *,
        layer_id: int,
        tensor_index: int,
        sample_tensor: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_id, tensor_index)
        if key in self._records:
            return self._records[key].tensor

        shm_name = self._tensor_shm_name(layer_id, tensor_index)
        num_bytes = self._num_bytes_for_tensor(sample_tensor)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=self._mirror_shape(sample_tensor),
            dtype=sample_tensor.dtype,
            strides=self._mirror_strides(sample_tensor),
        )
        if created_new:
            tensor.zero_()
        _cuda_host_register_tensor(tensor)
        self._records[key] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        self._tensor_devices[key] = sample_tensor.device
        return tensor

    def _get_or_create_valid_tensor(self, layer_id: int) -> torch.Tensor:
        if layer_id in self._valid_records:
            return self._valid_records[layer_id].tensor

        shm_name = self._valid_shm_name(layer_id)
        num_bytes = self._num_logical_experts * _element_size(torch.uint8)
        shm, is_owner, unlink_on_close, created_new = self._open_or_create_shm(
            shm_name, num_bytes
        )
        tensor = _create_buffer_tensor(
            shm=shm,
            shape=(self._num_logical_experts,),
            dtype=torch.uint8,
        )
        if created_new:
            tensor.zero_()
        self._valid_records[layer_id] = _ShmRecord(
            shm=shm,
            tensor=tensor,
            name=shm_name,
            is_owner=is_owner,
            unlink_on_close=unlink_on_close,
        )
        return tensor

    def _mirror_shape(self, tensor: torch.Tensor):
        return (self._num_logical_experts, *tensor.shape[1:])

    def _mirror_strides(self, tensor: torch.Tensor):
        """Return strides matching the GPU tensor's physical layout.

        If the GPU tensor is contiguous, returns None (use default contiguous
        strides). Otherwise returns a stride tuple that mirrors the GPU's
        physical storage order so that cudaMemcpyAsync can be used directly
        for H2D without elementwise kernels.
        """
        if tensor.is_contiguous():
            return None
        # Verify storage is dense (no gaps/overlap) — required for direct memcpy.
        numel_per_expert = math.prod(tensor.shape[1:])
        assert tensor.stride()[0] == numel_per_expert, (
            f"Non-dense storage: shape={tuple(tensor.shape)} "
            f"stride={tuple(tensor.stride())}. stride[0]={tensor.stride()[0]} "
            f"!= numel_per_expert={numel_per_expert}. "
            f"Cannot use stride-aligned host mirror for direct H2D memcpy."
        )
        return (numel_per_expert, *tensor.stride()[1:])

    def _num_bytes_for_tensor(self, tensor: torch.Tensor) -> int:
        return math.prod(self._mirror_shape(tensor)) * _element_size(tensor.dtype)

    def _tensor_shm_name(self, layer_id: int, tensor_index: int) -> str:
        return f"{self._base_name}_l{layer_id}_t{tensor_index}"

    def _valid_shm_name(self, layer_id: int) -> str:
        return f"{self._base_name}_l{layer_id}_valid"

    def _open_or_create_shm(self, shm_name: str, num_bytes: int):
        if self._is_owner:
            return self._open_or_create_owner_shm(shm_name, num_bytes)

        for _ in range(200):
            try:
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, False, False, False
            except FileNotFoundError:
                time.sleep(0.05)

        raise RuntimeError(
            f"Timed out waiting for host mirror shared memory {shm_name}."
        )

    def _open_or_create_owner_shm(self, shm_name: str, num_bytes: int):
        """Open existing SHM or create new. Always track=False and
        unlink_on_close=False so SHM persists across server restarts.
        """
        try:
            shm = _open_shared_memory(
                name=shm_name,
                create=True,
                size=num_bytes,
                track=False,
            )
            return shm, True, False, True
        except FileExistsError:
            # SHM exists from previous run — open for reuse check.
            try:
                shm = _open_shared_memory(name=shm_name, track=False)
                self._validate_shm_size(shm, shm_name, num_bytes)
                return shm, True, False, False
            except RuntimeError:
                # Size mismatch (model config changed) — unlink & recreate.
                logger.warning(
                    "EPLB async SHM size mismatch for %s, recreating.",
                    shm_name,
                )
                with contextlib.suppress(Exception):
                    shm.close()
                stale = _open_shared_memory(name=shm_name, track=False)
                stale.close()
                _unlink_shm_no_track(stale)
                shm = _open_shared_memory(
                    name=shm_name, create=True, size=num_bytes, track=False
                )
                return shm, True, False, True

    def _validate_shm_size(self, shm, shm_name: str, num_bytes: int):
        actual_size = getattr(shm, "size", None)
        if actual_size is not None and actual_size != num_bytes:
            shm.close()
            raise RuntimeError(
                f"EPLB async shared memory size mismatch for {shm_name}: "
                f"expected={num_bytes} actual={actual_size}"
            )

    def _can_reuse_existing_data(self) -> bool:
        if not self._reuse_existing_shm:
            return False
        if len(self._valid_records) == 0:
            return False
        for valid_record in self._valid_records.values():
            if not torch.all(valid_record.tensor == 1):
                return False
        return True

    # ------------------------------------------------------------------
    # Weight checksum: detect device weight changes across server restarts
    # ------------------------------------------------------------------

    _CHECKSUM_SIZE = 16  # MD5 digest size in bytes

    def _compute_weights_checksum(self, routed_experts_weights_of_layer) -> bytes:
        """Compute MD5 checksum of device weights for this rank.

        Uses tensor.sum() as a fast GPU-side fingerprint — only one scalar
        per tensor is transferred D2H.  Includes dtype and shape to detect
        configuration changes.
        """
        md5 = hashlib.md5()
        for layer_id in sorted(routed_experts_weights_of_layer):
            tensors = routed_experts_weights_of_layer[layer_id]
            for tensor in tensors:
                md5.update(str(tensor.sum().item()).encode())
                md5.update(str(tensor.dtype).encode())
                md5.update(str(tuple(tensor.shape)).encode())
        return md5.digest()

    def _verify_checksum(self, routed_experts_weights_of_layer) -> bool:
        """Open checksum SHM and verify stored checksum matches current weights.

        Returns False if the checksum SHM does not exist (first run) or if
        the checksum does not match (weights changed).
        """
        shm_name = f"{self._base_name}_checksum"
        try:
            shm = _open_shared_memory(name=shm_name, track=False)
        except FileNotFoundError:
            return False  # First run or checksum SHM was lost
        self._checksum_shm = shm
        self._checksum_shm_name = shm_name

        current = self._compute_weights_checksum(routed_experts_weights_of_layer)
        offset = self._local_rank * self._CHECKSUM_SIZE
        stored = bytes(shm.buf[offset : offset + self._CHECKSUM_SIZE])
        if current != stored:
            logger.info(
                "EPLB async host mirror checksum mismatch on rank=%d "
                "node_rank=%d. Weights changed, will rebuild.",
                self._rank,
                self._node_rank,
            )
            return False
        return True

    def _store_checksum_after_build(self, routed_experts_weights_of_layer):
        """Create/open checksum SHM and store current weights checksum.

        Called after a fresh build (not on reuse).  The owner creates the
        SHM; non-owners wait for it.  Each rank writes its 16-byte MD5 at
        offset ``local_rank * 16``.
        """
        shm_name = f"{self._base_name}_checksum"
        num_bytes = self._local_world_size * self._CHECKSUM_SIZE

        # Close existing checksum SHM if open from preflight.
        if self._checksum_shm is not None:
            with contextlib.suppress(Exception):
                self._checksum_shm.close()
            self._checksum_shm = None

        if self._is_owner:
            try:
                shm = _open_shared_memory(
                    name=shm_name, create=True, size=num_bytes, track=False
                )
            except FileExistsError:
                shm = _open_shared_memory(name=shm_name, track=False)
        else:
            for _ in range(200):
                try:
                    shm = _open_shared_memory(name=shm_name, track=False)
                    break
                except FileNotFoundError:
                    time.sleep(0.05)
            else:
                raise RuntimeError(f"Timed out waiting for checksum SHM {shm_name}")
        self._checksum_shm = shm
        self._checksum_shm_name = shm_name

        checksum = self._compute_weights_checksum(routed_experts_weights_of_layer)
        offset = self._local_rank * self._CHECKSUM_SIZE
        shm.buf[offset : offset + self._CHECKSUM_SIZE] = checksum

    # ------------------------------------------------------------------
    # Stale record cleanup
    # ------------------------------------------------------------------

    def _cleanup_stale_records(self):
        """Close and unlink stale SHM records from a failed preflight check.

        Called before rebuild so that ``_create_records`` can create fresh
        SHM segments instead of reusing stale data.
        """
        # Use _unlink_shm_no_track instead of shm.unlink() — SHM was created
        # with track=False (register patched to no-op in _open_shared_memory),
        # so calling shm.unlink() directly would send UNREGISTER to the daemon
        # for a name never registered, causing KeyError in the daemon.
        for record in self._records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if self._is_owner:
                with contextlib.suppress(FileNotFoundError):
                    _unlink_shm_no_track(record.shm)
        for record in self._valid_records.values():
            with contextlib.suppress(Exception):
                record.shm.close()
            if self._is_owner:
                with contextlib.suppress(FileNotFoundError):
                    _unlink_shm_no_track(record.shm)
        if self._checksum_shm is not None:
            with contextlib.suppress(Exception):
                self._checksum_shm.close()
            if self._is_owner:
                with contextlib.suppress(FileNotFoundError):
                    _unlink_shm_no_track(self._checksum_shm)
            self._checksum_shm = None
            self._checksum_shm_name = None
        self._records.clear()
        self._valid_records.clear()

    def _get_total_shm_nbytes(self) -> int:
        total = 0
        for record in self._records.values():
            total += getattr(record.shm, "size", 0)
        for record in self._valid_records.values():
            total += getattr(record.shm, "size", 0)
        return total

    def _build_grouped_cross_node_transfer_plan(
        self, *, routed_experts_weights_of_layer, metadata
    ) -> dict[int, list[tuple[int, int, int, int, tuple[int, ...]]]]:
        plan = {}
        num_local_physical_experts = metadata.num_local_physical_experts
        for layer_id in sorted(routed_experts_weights_of_layer):
            availability = _build_node_availability_from_metadata(
                metadata.physical_to_logical_map_cpu[layer_id],
                num_nodes=self._num_nodes,
                local_world_size=self._local_world_size,
                num_local_physical_experts=num_local_physical_experts,
                num_logical_experts=self._num_logical_experts,
            )
            plan[layer_id] = _build_rank_parallel_cross_node_transfer_plan(
                availability,
                physical_to_logical_map=metadata.physical_to_logical_map_cpu[layer_id],
                local_world_size=self._local_world_size,
                num_local_physical_experts=num_local_physical_experts,
                num_logical_experts=self._num_logical_experts,
            )
        return plan

    def _summarize_transfer_plan(
        self, layer_transfer_plan, *, routed_experts_weights_of_layer
    ) -> tuple[int, int, int]:
        local_steps = 0
        total_experts = 0
        total_bytes = 0
        for layer_id, transfer_groups in layer_transfer_plan.items():
            num_tensors = len(routed_experts_weights_of_layer[layer_id])
            for (
                src_node_rank,
                src_local_rank,
                dst_node_rank,
                dst_local_rank,
                logical_expert_ids,
            ) in transfer_groups:
                num_group_experts = len(logical_expert_ids)
                total_experts += num_group_experts
                if (self._node_rank, self._local_rank) in (
                    (src_node_rank, src_local_rank),
                    (dst_node_rank, dst_local_rank),
                ):
                    local_steps += num_tensors
                for tensor in routed_experts_weights_of_layer[layer_id]:
                    total_bytes += (
                        num_group_experts * tensor[0].numel() * tensor.element_size()
                    )
        return local_steps, total_experts, total_bytes

    def _get_or_create_transfer_buffers(
        self,
        *,
        sample_tensor: torch.Tensor,
        device: torch.device,
        num_logical_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tail_shape = tuple(sample_tensor.shape[1:])
        gpu_key = (
            sample_tensor.dtype,
            str(device),
            tail_shape,
        )
        cpu_key = (
            sample_tensor.dtype,
            tail_shape,
        )
        gpu_staging = self._gpu_staging_buffers.get(gpu_key)
        cpu_transfer = self._cpu_transfer_buffers.get(cpu_key)
        if gpu_staging is None or gpu_staging.shape[0] < num_logical_experts:
            self._gpu_staging_buffers[gpu_key] = torch.empty(
                (num_logical_experts, *tail_shape),
                dtype=sample_tensor.dtype,
                device=device,
            )
            gpu_staging = self._gpu_staging_buffers[gpu_key]
        if cpu_transfer is None or cpu_transfer.shape[0] < num_logical_experts:
            self._cpu_transfer_buffers[cpu_key] = torch.empty(
                (num_logical_experts, *tail_shape),
                dtype=sample_tensor.dtype,
                pin_memory=True,
            )
            cpu_transfer = self._cpu_transfer_buffers[cpu_key]
        return gpu_staging[:num_logical_experts], cpu_transfer[:num_logical_experts]


def _create_buffer_tensor(
    *,
    shm: shared_memory.SharedMemory,
    shape,
    dtype: torch.dtype,
    strides=None,
) -> torch.Tensor:
    numel = math.prod(shape)
    raw = torch.frombuffer(shm.buf, dtype=torch.uint8)
    typed = raw.view(dtype)[:numel]
    if strides is None:
        return typed.view(*shape)
    # Use as_strided to match GPU's physical storage layout.
    # This allows cudaMemcpyAsync to work correctly for non-contiguous
    # GPU tensors (e.g. mxfp4 weight_scale after preprocess_mxfp4_scales).
    return torch.as_strided(typed, shape, strides)


def _index_copy_host_tensor(
    dst: torch.Tensor, index: torch.Tensor, src: torch.Tensor
) -> None:
    _UNSUPPORTED_INDEX_COPY_DTYPES = {
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.uint16,
    }
    if dst.dtype not in _UNSUPPORTED_INDEX_COPY_DTYPES:
        dst.index_copy_(0, index, src)
        return

    # For strided host tensors (e.g. mxfp4 scale after preprocess_mxfp4_scales),
    # view(uint8) requires stride(-1)==1 which may not hold. Fall back to
    # per-row copy_ which correctly handles arbitrary strides via PyTorch dispatch.
    # This is only used during one-time host mirror build, not on the H2D hot path.
    if dst.stride(-1) != 1:
        for i in range(index.numel()):
            dst[index[i].item()].copy_(src[i])
        return

    row_nbytes = dst[0].numel() * dst.element_size()
    dst_bytes = dst.view(torch.uint8).view(dst.shape[0], row_nbytes)
    src_bytes = src.view(torch.uint8).view(src.shape[0], row_nbytes)
    dst_bytes.index_copy_(0, index, src_bytes)


def _open_shared_memory(
    *, name: str, create: bool = False, size: int = 0, track: bool = True
):
    if track:
        return shared_memory.SharedMemory(name=name, create=create, size=size)
    # Patch register to no-op so the SHM is NOT tracked by resource_tracker.
    # This ensures the SHM segment survives process exit (including SIGKILL)
    # and can be reused on server restart.
    #
    # Do NOT call resource_tracker.unregister here — since register was patched
    # to a no-op, the SHM was never registered. Calling unregister would cause
    # a KeyError in the resource_tracker daemon process (which runs in a
    # separate process and cannot be caught by try/except in this process).
    with patch.object(resource_tracker, "register", lambda *args, **kwargs: None):
        shm = shared_memory.SharedMemory(name=name, create=create, size=size)
    # Suppress UNREGISTER during close()/del/unlink: in Python 3.13+
    # SharedMemory.close() and unlink() check _track before sending UNREGISTER.
    # In Python 3.12, close() does not call unregister but unlink() always does.
    # Setting _track=False prevents 3.13+ from sending UNREGISTER on close/unlink.
    # For 3.12, callers must use _unlink_shm_no_track() instead of shm.unlink().
    if hasattr(shm, "_track"):
        shm._track = False
    return shm


def _unlink_shm_no_track(shm):
    """Unlink shared memory without sending UNREGISTER to resource_tracker.

    SHM segments created with track=False (register patched to no-op) are
    never registered in the daemon's cache.  Python 3.12's unlink() always
    calls resource_tracker.unregister() unconditionally, which sends
    UNREGISTER to the daemon — causing a KeyError because the name was
    never registered.  This helper patches unregister to a no-op during
    the unlink call to prevent the spurious UNREGISTER message.
    """
    with patch.object(resource_tracker, "unregister", lambda *args, **kwargs: None):
        shm.unlink()


def _cuda_host_register_tensor(tensor: torch.Tensor):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "EPLB async requires CUDA, but torch.cuda.is_available() is False."
        )

    libcudart = ctypes.CDLL("libcudart.so")

    # Check if memory logging is enabled via environment variable
    _memory_log_enabled = getattr(
        _cuda_host_register_tensor, "_memory_log_enabled", None
    )
    if _memory_log_enabled is None:
        _cuda_host_register_tensor._memory_log_enabled = (
            envs.SGLANG_EPLB_ASYNC_HOST_MIRROR_MEMORY_LOG.get()
        )
        _memory_log_enabled = _cuda_host_register_tensor._memory_log_enabled

    # Periodic GPU memory logging for debugging (only when enabled)
    should_log = False
    if _memory_log_enabled:
        _cuda_host_register_tensor._counter = (
            getattr(_cuda_host_register_tensor, "_counter", 0) + 1
        )
        _cuda_host_register_tensor._total_bytes = getattr(
            _cuda_host_register_tensor, "_total_bytes", 0
        )

        counter = _cuda_host_register_tensor._counter
        num_bytes = tensor.numel() * tensor.element_size()
        _cuda_host_register_tensor._total_bytes += num_bytes

        # Log every 10 calls or for large tensors (>100MB)
        should_log = counter % 10 == 0 or num_bytes > 100 * 1024 * 1024

        if should_log:
            # Use cudaMemGetInfo to get real GPU memory usage (not just PyTorch's allocator)
            free_before = ctypes.c_size_t()
            total_before = ctypes.c_size_t()
            libcudart.cudaMemGetInfo(
                ctypes.byref(free_before), ctypes.byref(total_before)
            )
            used_before = total_before.value - free_before.value
    else:
        num_bytes = tensor.numel() * tensor.element_size()

    cuda_host_register_portable = 0x01
    result = libcudart.cudaHostRegister(
        ctypes.c_void_p(tensor.data_ptr()),
        ctypes.c_size_t(num_bytes),
        ctypes.c_uint(cuda_host_register_portable),
    )
    if result != 0:
        raise RuntimeError(
            f"cudaHostRegister failed with CUDA error code {result} "
            f"for tensor shape={tuple(tensor.shape)} dtype={tensor.dtype}"
        )

    if should_log:
        free_after = ctypes.c_size_t()
        total_after = ctypes.c_size_t()
        libcudart.cudaMemGetInfo(ctypes.byref(free_after), ctypes.byref(total_after))
        used_after = total_after.value - free_after.value

        logger.info(
            "[cudaHostRegister] count=%d total_registered=%.2fGB "
            "tensor_shape=%s dtype=%s size=%.2fMB "
            "GPU used: %.2fGB -> %.2fGB (delta: %+dMB) total: %.2fGB",
            counter,
            _cuda_host_register_tensor._total_bytes / (1024**3),
            tuple(tensor.shape),
            tensor.dtype,
            num_bytes / (1024**2),
            used_before / (1024**3),
            used_after / (1024**3),
            int((used_after - used_before) / (1024**2)),
            total_after.value / (1024**3),
        )


_INSTANCE = None


def get_global_eplb_async_host_mirror_manager():
    return _INSTANCE


def set_global_eplb_async_host_mirror_manager(value):
    global _INSTANCE
    _INSTANCE = value
