# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import atexit
import collections
import hashlib
import logging
import mmap
import os
import re
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np
import torch

from sglang.srt.mem_cache.storage.tensorcast_store.config import (
    TensorcastHiCacheConfig,
)
from sglang.srt.mem_cache.storage.tensorcast_store.host_shared_slot_state import (
    HostSharedPageSlotToken,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
        TensorcastHostRegionBinding,
    )


@dataclass(frozen=True)
class TensorcastBatchExistsResult:
    existence_mask: tuple[bool, ...]
    rpc_elapsed_s: float


@dataclass(frozen=True)
class TensorcastBatchTransferResult:
    success_mask: tuple[bool, ...]
    adopted_duplicate_count: int
    pack_elapsed_s: float
    stage_copy_elapsed_s: float
    rpc_elapsed_s: float
    host_fill_elapsed_s: float
    operation_id: str = ""


class TensorcastPageClient(Protocol):
    def artifact_id_for(self, logical_key: str) -> str: ...

    def activate_stable_local_backing(
        self, region_id: str, *, slot_bytes: int
    ) -> None: ...

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult: ...

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        source_region_binding: "TensorcastHostRegionBinding | None" = None,
    ) -> TensorcastBatchTransferResult: ...

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        target_region_binding: "TensorcastHostRegionBinding | None" = None,
    ) -> TensorcastBatchTransferResult: ...


@dataclass
class _HostSharedRegionState:
    tensor: torch.Tensor
    region_id: str
    capacity_bytes: int
    handle: object
    mapping: mmap.mmap
    array: np.ndarray


_CGID_SAFE_PATTERN = re.compile(r"[^-._~A-Za-z0-9]+")


def _cpu_tensor_as_uint8_view(tensor: torch.Tensor) -> torch.Tensor:
    flat = tensor.reshape(-1)
    if not flat.is_contiguous():
        flat = flat.contiguous()
    return flat.view(torch.uint8)


def _sanitize_cgid_segment(value: str) -> str:
    sanitized = _CGID_SAFE_PATTERN.sub("_", str(value).strip())
    sanitized = sanitized.strip("_")
    return sanitized or "default"


def _compact_cgid_segment(value: str, *, prefix: str, max_len: int = 24) -> str:
    sanitized = _sanitize_cgid_segment(value)
    if len(sanitized) <= max_len:
        return sanitized
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _engine_key_payload(engine_key_prefix: str, logical_key: str) -> bytes:
    prefix = str(engine_key_prefix).encode("utf-8")
    logical_key_text = str(logical_key).strip()
    if logical_key_text and len(logical_key_text) % 2 == 0:
        try:
            key_bytes = bytes.fromhex(logical_key_text)
        except ValueError:
            key_bytes = logical_key_text.encode("utf-8")
    else:
        key_bytes = logical_key_text.encode("utf-8")
    return prefix + b":" + key_bytes


class _HostSharedRegionManager:
    def __init__(
        self,
        *,
        store,
        host_shared_region_class,
        region_memory_kind,
        ttl_ms: int,
        name: str,
    ) -> None:
        self._store = store
        self._host_shared_region_class = host_shared_region_class
        self._region_memory_kind = region_memory_kind
        self._ttl_ms = int(ttl_ms)
        self._name = name
        self._state: _HostSharedRegionState | None = None
        self._lock = threading.Lock()

    def ensure_capacity(self, required_bytes: int) -> _HostSharedRegionState:
        if required_bytes <= 0:
            raise ValueError("required_bytes must be positive")
        with self._lock:
            state = self._state
            if state is not None and state.capacity_bytes >= required_bytes:
                return state
            self._release_locked()
            handle = None
            try:
                handle = self._store.register_region(
                    memory_kind=self._region_memory_kind,
                    size_bytes=int(required_bytes),
                    ttl_ms=self._ttl_ms,
                    daemon_managed=True,
                    host_shared_region_class=self._host_shared_region_class,
                    name=self._name,
                )
                attachment = self._store.attach_host_shared_region(handle)
                fd = int(attachment.fd)
                try:
                    mapping = mmap.mmap(
                        fd,
                        int(attachment.size_bytes),
                        flags=mmap.MAP_SHARED,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE,
                    )
                finally:
                    os.close(fd)
                array = np.ndarray(
                    (int(attachment.size_bytes),), dtype=np.uint8, buffer=mapping
                )
                tensor = torch.from_numpy(array)
            except Exception:
                with suppress(Exception):
                    if handle is not None:
                        self._store.release_host_shared_region(handle)
                with suppress(Exception):
                    if handle is not None:
                        self._store.unregister_region(str(handle.region_id), force=True)
                raise
            self._state = _HostSharedRegionState(
                tensor=tensor,
                region_id=str(handle.region_id),
                capacity_bytes=int(required_bytes),
                handle=handle,
                mapping=mapping,
                array=array,
            )
            return self._state

    def close(self) -> None:
        with self._lock:
            self._release_locked()

    def _release_locked(self) -> None:
        state = self._state
        self._state = None
        if state is None:
            return
        state.tensor = torch.empty(0, dtype=torch.uint8)
        state.array = np.empty((0,), dtype=np.uint8)
        with suppress(Exception):
            state.mapping.close()
        with suppress(Exception):
            self._store.release_host_shared_region(state.handle)
        with suppress(Exception):
            self._store.unregister_region(state.region_id, force=True)


class DefaultTensorcastPageClient:
    def __init__(
        self,
        config: TensorcastHiCacheConfig,
        *,
        layout_id: str,
        engine_key_prefix: str,
    ) -> None:
        try:
            import tensorcast as tc
            from tensorcast.common.identity import build_byte_artifact_cgid
            from tensorcast.common.selection_contract import build_artifact_selection
            from tensorcast.proto.common.v1 import common_pb2
            from tensorcast.proto.daemon.v2 import store_daemon_pb2
            from tensorcast.types import HostSharedRegionClass, RegionMemoryKind
        except ImportError as e:
            raise ImportError(
                "Please install the `tensorcast` Python package "
                "(see https://github.com/tensorcast-ai/tensorcast) "
                "to use the Tensorcast HiCache storage backend."
            ) from e

        self._tc = tc
        self._common_pb2 = common_pb2
        self._store_daemon_pb2 = store_daemon_pb2
        self._build_artifact_selection = build_artifact_selection
        self._build_byte_artifact_cgid = build_byte_artifact_cgid
        self._config = config
        self._layout_id = _compact_cgid_segment(layout_id, prefix="ly")
        self._namespace = _compact_cgid_segment(config.namespace, prefix="ns")
        self._engine = _compact_cgid_segment(config.engine, prefix="en")
        self._model_id = str(config.model_id)
        self._model_version = str(config.model_version)
        self._engine_key_prefix = str(engine_key_prefix)
        self._artifact_id_cache: dict[str, str] = {}
        self._selection_cache: dict[str, object] = {}
        self._store = tc.Store(config.daemon_address)
        self._client = self._store._runtime.ensure_client()
        region_ttl_ms = int(config.staging_region_ttl_ms)
        self._put_staging = _HostSharedRegionManager(
            store=self._store,
            host_shared_region_class=HostSharedRegionClass.SCRATCH,
            region_memory_kind=RegionMemoryKind.HOST_SHARED,
            ttl_ms=region_ttl_ms,
            name="sglang_tensorcast_put_staging",
        )
        self._get_staging = _HostSharedRegionManager(
            store=self._store,
            host_shared_region_class=HostSharedRegionClass.SCRATCH,
            region_memory_kind=RegionMemoryKind.HOST_SHARED,
            ttl_ms=region_ttl_ms,
            name="sglang_tensorcast_get_staging",
        )
        logger.info(
            "Tensorcast page client configured verification_mode=%s namespace=%s engine=%s layout_id=%s model_id=%s model_version=%s",
            "BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY",
            self._namespace,
            self._engine,
            self._layout_id,
            self._model_id,
            self._model_version,
        )
        atexit.register(self.close)

    def close(self) -> None:
        self._put_staging.close()
        self._get_staging.close()

    def activate_stable_local_backing(self, region_id: str, *, slot_bytes: int) -> None:
        self._store.activate_stable_local_backing(
            region_id,
            slot_bytes=int(slot_bytes),
        )

    def artifact_id_for(self, logical_key: str) -> str:
        artifact_id = self._artifact_id_cache.get(logical_key)
        if artifact_id is not None:
            return artifact_id
        artifact_id = self._build_byte_artifact_cgid(
            namespace=self._namespace,
            engine=self._engine,
            model_id=self._model_id,
            model_version=self._model_version,
            layout_id=self._layout_id,
            engine_key=_engine_key_payload(self._engine_key_prefix, logical_key),
        )
        self._artifact_id_cache[logical_key] = artifact_id
        return artifact_id

    def batch_exists(self, logical_keys: list[str]) -> TensorcastBatchExistsResult:
        if not logical_keys:
            return TensorcastBatchExistsResult(existence_mask=(), rpc_elapsed_s=0.0)
        selections = [
            self._selection_for(self.artifact_id_for(key)) for key in logical_keys
        ]
        started_at = time.perf_counter()
        response = self._client.batch_exists(
            selections=selections,
            timeout_s=float(self._config.batch_exists_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - started_at
        outcome_by_artifact = {
            str(outcome.artifact_id): int(outcome.status)
            for outcome in response.outcomes
        }
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        miss_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_MISS)
        status_counts = collections.Counter(outcome_by_artifact.values())
        existence_mask: list[bool] = []
        for logical_key in logical_keys:
            artifact_id = self.artifact_id_for(logical_key)
            status = outcome_by_artifact.get(artifact_id, miss_status)
            existence_mask.append(status == ok_status)
        unexpected_statuses = {
            status: count
            for status, count in status_counts.items()
            if status not in {ok_status, miss_status}
        }
        if unexpected_statuses:
            status_summary = {
                self._store_daemon_pb2.BatchItemStatus.Name(status): count
                for status, count in unexpected_statuses.items()
            }
            logger.debug(
                "Tensorcast batch_exists unexpected statuses=%s first_key=%s first_artifact_id=%s",
                status_summary,
                logical_keys[0],
                self.artifact_id_for(logical_keys[0]),
            )
        return TensorcastBatchExistsResult(
            existence_mask=tuple(existence_mask),
            rpc_elapsed_s=rpc_elapsed_s,
        )

    def batch_put(
        self,
        logical_keys: list[str],
        pages: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        source_region_binding: "TensorcastHostRegionBinding | None" = None,
    ) -> TensorcastBatchTransferResult:
        if len(logical_keys) != len(pages):
            raise ValueError("logical_keys and pages must have the same length")
        if slot_tokens is not None and len(slot_tokens) != len(logical_keys):
            raise ValueError("slot_tokens and logical_keys must have the same length")
        if not logical_keys:
            return TensorcastBatchTransferResult(
                success_mask=(),
                adopted_duplicate_count=0,
                pack_elapsed_s=0.0,
                stage_copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
                host_fill_elapsed_s=0.0,
            )
        pack_started_at = time.perf_counter()
        packed_pages: list[tuple[str, str, torch.Tensor, int]] = []
        total_bytes = 0
        for logical_key, page in zip(logical_keys, pages, strict=True):
            artifact_id = self.artifact_id_for(logical_key)
            byte_length = int(page.numel()) * int(page.element_size())
            page_bytes = (
                page
                if source_region_binding is not None
                else _cpu_tensor_as_uint8_view(page)
            )
            packed_pages.append((logical_key, artifact_id, page_bytes, byte_length))
            total_bytes += byte_length
        pack_elapsed_s = time.perf_counter() - pack_started_at
        if source_region_binding is not None:
            if slot_tokens is None:
                raise ValueError(
                    "source_region_binding requires slot_tokens for direct host-slot publish"
                )
            return self._batch_put_from_resident_slots(
                logical_keys=logical_keys,
                packed_pages=packed_pages,
                pack_elapsed_s=pack_elapsed_s,
                slot_tokens=slot_tokens,
                source_region_binding=source_region_binding,
            )
        staging = self._put_staging.ensure_capacity(total_bytes)
        source_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = source_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = -1
        storage.storage_length = int(total_bytes)
        storage.mapping_base_offset = 0
        region_ref = storage.region_ref
        region_ref.region_id = staging.region_id
        region_ref.memory_kind = self._store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
        region_ref.device_id = -1
        region_ref.size_bytes = int(staging.capacity_bytes)
        items = []
        stage_started_at = time.perf_counter()
        cursor = 0
        for item_index, (_, artifact_id, page_bytes, byte_length) in enumerate(
            packed_pages
        ):
            staging.tensor[cursor : cursor + byte_length].copy_(
                page_bytes,
                non_blocking=False,
            )
            offset = source_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(cursor)
            offset.logical_length = int(byte_length)
            if slot_tokens is not None:
                offset.slot_index = int(slot_tokens[item_index].slot_index)
                offset.slot_generation = int(slot_tokens[item_index].slot_generation)
            item = self._store_daemon_pb2.BatchPutIfAbsentFromRegionItem(
                selection=self._selection_for(artifact_id),
                invariant=self._store_daemon_pb2.PutIfAbsentInvariant(
                    layout_id=self._layout_id,
                    byte_length=int(byte_length),
                    verification_mode=(
                        self._store_daemon_pb2.BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY
                    ),
                ),
            )
            items.append(item)
            cursor += byte_length
        stage_copy_elapsed_s = time.perf_counter() - stage_started_at
        rpc_started_at = time.perf_counter()
        response = self._client.batch_put_if_absent_from_region(
            items=items,
            source_layout=source_layout,
            pid=os.getpid(),
            device_uuid="",
            operation_id=uuid.uuid4().hex,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        success_mask: list[bool] = []
        adopted_duplicate_count = 0
        for _, artifact_id, _, _ in packed_pages:
            outcome = outcome_by_artifact.get(artifact_id)
            status = int(outcome.status) if outcome is not None else 0
            if status == ok_status:
                if (
                    outcome is not None
                    and str(outcome.message).strip().lower() == "joined"
                ):
                    adopted_duplicate_count += 1
                success_mask.append(True)
                continue
            success_mask.append(False)
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            invariant_mismatch_detected = False
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                if "invariant mismatch" in str(outcome.message).lower():
                    invariant_mismatch_detected = True
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            log_fn = logger.warning if invariant_mismatch_detected else logger.debug
            log_fn(
                "Tensorcast batch_put failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_pages[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=adopted_duplicate_count,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=stage_copy_elapsed_s,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=0.0,
        )

    def _batch_put_from_resident_slots(
        self,
        *,
        logical_keys: list[str],
        packed_pages: list[tuple[str, str, torch.Tensor, int]],
        pack_elapsed_s: float,
        slot_tokens: list[HostSharedPageSlotToken],
        source_region_binding: "TensorcastHostRegionBinding",
    ) -> TensorcastBatchTransferResult:
        source_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = source_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = -1
        storage.storage_length = int(source_region_binding.capacity_bytes)
        storage.mapping_base_offset = 0
        region_ref = storage.region_ref
        region_ref.region_id = str(source_region_binding.region_id)
        region_ref.memory_kind = self._store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
        region_ref.device_id = -1
        region_ref.size_bytes = int(source_region_binding.capacity_bytes)

        items = []
        page_bytes_per_slot: int | None = None
        for item_index, (_, artifact_id, _, byte_length) in enumerate(packed_pages):
            if page_bytes_per_slot is None:
                page_bytes_per_slot = int(byte_length)
            elif page_bytes_per_slot != int(byte_length):
                raise ValueError(
                    "resident-slot direct publish requires fixed byte length per page slot"
                )
            offset = source_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(slot_tokens[item_index].slot_index) * int(
                page_bytes_per_slot
            )
            offset.logical_length = int(byte_length)
            offset.slot_index = int(slot_tokens[item_index].slot_index)
            offset.slot_generation = int(slot_tokens[item_index].slot_generation)
            item = self._store_daemon_pb2.BatchPutIfAbsentFromRegionItem(
                selection=self._selection_for(artifact_id),
                invariant=self._store_daemon_pb2.PutIfAbsentInvariant(
                    layout_id=self._layout_id,
                    byte_length=int(byte_length),
                    verification_mode=(
                        self._store_daemon_pb2.BYTE_ARTIFACT_VERIFICATION_MODE_LAYOUT_AND_SIZE_ONLY
                    ),
                ),
            )
            items.append(item)

        rpc_started_at = time.perf_counter()
        response = self._client.batch_put_if_absent_from_region(
            items=items,
            source_layout=source_layout,
            pid=os.getpid(),
            device_uuid="",
            operation_id=uuid.uuid4().hex,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        success_mask: list[bool] = []
        adopted_duplicate_count = 0
        for _, artifact_id, _, _ in packed_pages:
            outcome = outcome_by_artifact.get(artifact_id)
            status = int(outcome.status) if outcome is not None else 0
            if status == ok_status:
                if (
                    outcome is not None
                    and str(outcome.message).strip().lower() == "joined"
                ):
                    adopted_duplicate_count += 1
                success_mask.append(True)
                continue
            success_mask.append(False)
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            invariant_mismatch_detected = False
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                if "invariant mismatch" in str(outcome.message).lower():
                    invariant_mismatch_detected = True
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            log_fn = logger.warning if invariant_mismatch_detected else logger.debug
            log_fn(
                "Tensorcast direct batch_put failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_pages[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=adopted_duplicate_count,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=0.0,
        )

    def batch_get_into(
        self,
        logical_keys: list[str],
        targets: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken] | None = None,
        target_region_binding: "TensorcastHostRegionBinding | None" = None,
    ) -> TensorcastBatchTransferResult:
        if len(logical_keys) != len(targets):
            raise ValueError("logical_keys and targets must have the same length")
        if slot_tokens is not None and len(slot_tokens) != len(logical_keys):
            raise ValueError("slot_tokens and logical_keys must have the same length")
        if not logical_keys:
            return TensorcastBatchTransferResult(
                success_mask=(),
                adopted_duplicate_count=0,
                pack_elapsed_s=0.0,
                stage_copy_elapsed_s=0.0,
                rpc_elapsed_s=0.0,
                host_fill_elapsed_s=0.0,
            )
        if target_region_binding is not None:
            if slot_tokens is None:
                raise ValueError(
                    "target_region_binding requires slot_tokens for direct host-slot fetch"
                )
            return self._batch_get_into_resident_slots(
                logical_keys=logical_keys,
                targets=targets,
                slot_tokens=slot_tokens,
                target_region_binding=target_region_binding,
            )
        pack_started_at = time.perf_counter()
        packed_targets: list[tuple[str, str, torch.Tensor, int]] = []
        total_bytes = 0
        for logical_key, target in zip(logical_keys, targets, strict=True):
            artifact_id = self.artifact_id_for(logical_key)
            target_bytes = _cpu_tensor_as_uint8_view(target)
            byte_length = int(target_bytes.numel())
            packed_targets.append((logical_key, artifact_id, target_bytes, byte_length))
            total_bytes += byte_length
        pack_elapsed_s = time.perf_counter() - pack_started_at
        staging = self._get_staging.ensure_capacity(total_bytes)
        target_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = target_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = -1
        storage.storage_length = int(total_bytes)
        storage.mapping_base_offset = 0
        region_ref = storage.region_ref
        region_ref.region_id = staging.region_id
        region_ref.memory_kind = self._store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
        region_ref.device_id = -1
        region_ref.size_bytes = int(staging.capacity_bytes)
        selections = []
        cursor = 0
        for item_index, (_, artifact_id, _, byte_length) in enumerate(packed_targets):
            selections.append(self._selection_for(artifact_id))
            offset = target_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(cursor)
            offset.logical_length = int(byte_length)
            if slot_tokens is not None:
                offset.slot_index = int(slot_tokens[item_index].slot_index)
                offset.slot_generation = int(slot_tokens[item_index].slot_generation)
            cursor += byte_length
        operation_id = uuid.uuid4().hex
        rpc_started_at = time.perf_counter()
        response = self._client.batch_get_into_region(
            selections=selections,
            target_layout=target_layout,
            pid=os.getpid(),
            device_uuid="",
            operation_id=operation_id,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        host_fill_started_at = time.perf_counter()
        cursor = 0
        success_mask: list[bool] = []
        stop_copying = False
        for _, artifact_id, target_bytes, byte_length in packed_targets:
            outcome = outcome_by_artifact.get(artifact_id)
            status = (
                int(outcome.status)
                if outcome is not None
                else int(self._store_daemon_pb2.BATCH_ITEM_STATUS_MISS)
            )
            if stop_copying or status != ok_status:
                stop_copying = True
                success_mask.append(False)
                cursor += byte_length
                continue
            target_bytes.copy_(
                staging.tensor[cursor : cursor + byte_length],
                non_blocking=False,
            )
            success_mask.append(True)
            cursor += byte_length
        host_fill_elapsed_s = time.perf_counter() - host_fill_started_at
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            logger.debug(
                "Tensorcast batch_get failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_targets[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=0,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=host_fill_elapsed_s,
            operation_id=operation_id,
        )

    def _batch_get_into_resident_slots(
        self,
        *,
        logical_keys: list[str],
        targets: list[torch.Tensor],
        slot_tokens: list[HostSharedPageSlotToken],
        target_region_binding: "TensorcastHostRegionBinding",
    ) -> TensorcastBatchTransferResult:
        pack_started_at = time.perf_counter()
        packed_targets: list[tuple[str, str, int]] = []
        page_bytes_per_slot: int | None = None
        for logical_key, target in zip(logical_keys, targets, strict=True):
            artifact_id = self.artifact_id_for(logical_key)
            target_bytes = _cpu_tensor_as_uint8_view(target)
            byte_length = int(target_bytes.numel())
            if page_bytes_per_slot is None:
                page_bytes_per_slot = int(byte_length)
            elif page_bytes_per_slot != int(byte_length):
                raise ValueError(
                    "resident-slot direct fetch requires fixed byte length per page slot"
                )
            packed_targets.append((logical_key, artifact_id, byte_length))
        pack_elapsed_s = time.perf_counter() - pack_started_at

        target_layout = self._store_daemon_pb2.TargetLayout(
            layout_kind=self._store_daemon_pb2.TargetLayout.LAYOUT_KIND_COALESCED_UNSPECIFIED,
            index_kind=self._store_daemon_pb2.TargetLayout.INDEX_KIND_CANONICAL_UNSPECIFIED,
            tensor_spec_kind=self._store_daemon_pb2.TargetLayout.TENSOR_SPEC_KIND_OFFSETS,
        )
        storage = target_layout.storages.add()
        storage.storage_id = "storage-0"
        storage.device_id = -1
        storage.storage_length = int(target_region_binding.capacity_bytes)
        storage.mapping_base_offset = 0
        region_ref = storage.region_ref
        region_ref.region_id = str(target_region_binding.region_id)
        region_ref.memory_kind = self._store_daemon_pb2.REGION_MEMORY_KIND_HOST_SHARED
        region_ref.device_id = -1
        region_ref.size_bytes = int(target_region_binding.capacity_bytes)

        selections = []
        for item_index, (_, artifact_id, byte_length) in enumerate(packed_targets):
            selections.append(self._selection_for(artifact_id))
            offset = target_layout.offsets.add()
            offset.name = artifact_id
            offset.storage_id = "storage-0"
            offset.storage_offset = int(slot_tokens[item_index].slot_index) * int(
                page_bytes_per_slot
            )
            offset.logical_length = int(byte_length)
            offset.slot_index = int(slot_tokens[item_index].slot_index)
            offset.slot_generation = int(slot_tokens[item_index].slot_generation)

        operation_id = uuid.uuid4().hex
        rpc_started_at = time.perf_counter()
        response = self._client.batch_get_into_region(
            selections=selections,
            target_layout=target_layout,
            pid=os.getpid(),
            device_uuid="",
            operation_id=operation_id,
            timeout_s=float(self._config.batch_transfer_timeout_s),
        )
        rpc_elapsed_s = time.perf_counter() - rpc_started_at
        ok_status = int(self._store_daemon_pb2.BATCH_ITEM_STATUS_OK)
        outcome_by_artifact = {
            str(outcome.artifact_id): outcome for outcome in response.outcomes
        }
        status_counts = collections.Counter(
            int(outcome.status) for outcome in response.outcomes
        )
        success_mask: list[bool] = []
        stop_copying = False
        for _, artifact_id, _ in packed_targets:
            outcome = outcome_by_artifact.get(artifact_id)
            status = (
                int(outcome.status)
                if outcome is not None
                else int(self._store_daemon_pb2.BATCH_ITEM_STATUS_MISS)
            )
            if stop_copying or status != ok_status:
                stop_copying = True
                success_mask.append(False)
                continue
            success_mask.append(True)
        failed_statuses = {
            self._store_daemon_pb2.BatchItemStatus.Name(status): count
            for status, count in status_counts.items()
            if status != ok_status
        }
        if failed_statuses:
            failure_messages = []
            for artifact_id, outcome in outcome_by_artifact.items():
                status = int(outcome.status)
                if status == ok_status:
                    continue
                failure_messages.append(
                    f"{artifact_id}:{self._store_daemon_pb2.BatchItemStatus.Name(status)}:{outcome.message}"
                )
                if len(failure_messages) >= 3:
                    break
            logger.debug(
                "Tensorcast direct batch_get failures statuses=%s first_key=%s first_artifact_id=%s samples=%s",
                failed_statuses,
                logical_keys[0],
                packed_targets[0][1],
                failure_messages,
            )
        return TensorcastBatchTransferResult(
            success_mask=tuple(success_mask),
            adopted_duplicate_count=0,
            pack_elapsed_s=pack_elapsed_s,
            stage_copy_elapsed_s=0.0,
            rpc_elapsed_s=rpc_elapsed_s,
            host_fill_elapsed_s=0.0,
            operation_id=operation_id,
        )

    def _selection_for(self, artifact_id: str):
        selection = self._selection_cache.get(artifact_id)
        if selection is not None:
            cloned = self._common_pb2.ArtifactSelection()
            cloned.CopyFrom(selection)
            return cloned
        selection = self._build_artifact_selection(
            artifact_id=artifact_id,
            canonical_index_bytes=b"",
            layout_index_bytes=None,
            view_spec=None,
            tensor_names=None,
        )
        self._selection_cache[artifact_id] = selection
        cloned = self._common_pb2.ArtifactSelection()
        cloned.CopyFrom(selection)
        return cloned
