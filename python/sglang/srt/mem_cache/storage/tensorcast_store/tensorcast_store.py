# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

"""TensorCast-backed synchronous FULL-page storage adapter."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from typing import Any, TypeVar

import torch
from tensorcast.api.store import (
    ByteArtifactKeyspace,
    ByteArtifactSpec,
    HostMemorySpan,
    RegionArtifactInputError,
    RegionArtifactTransfer,
    RegionBackedArtifactSession,
    RegionSessionFailedError,
    RegionSessionTerminatedError,
)

from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    HiCacheStorageExtraInfo,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.pool_host import HostKVCache
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
    TensorcastConfig,
    TensorcastHostTensorAllocator,
    TensorcastTransferMode,
    build_tensorcast_session_options,
    claim_tensorcast_store_session,
    normalize_tensorcast_config,
    resolve_tensorcast_model_id,
    terminate_tensorcast_store_session,
)

logger = logging.getLogger(__name__)

ARTIFACT_LAYOUT_SCHEMA_VERSION = "full-fragment-v1"
_ARTIFACT_LAYOUT_SCHEMA_TOKEN = "ff1"
_SUPPORTED_FULL_LAYOUTS = frozenset({"page_first", "page_first_direct"})
_FULL_LAYOUT_TOKENS = {
    "page_first": "pf",
    "page_first_direct": "pfd",
}
_DTYPE_LAYOUT_TOKENS = {
    "bfloat16": "bf16",
    "float16": "f16",
    "float32": "f32",
    "float64": "f64",
    "float8_e4m3fn": "f8e4m3fn",
    "float8_e4m3fnuz": "f8e4m3fnuz",
    "float8_e5m2": "f8e5m2",
    "uint8": "u8",
}
_ENGINE_KEY_DOMAIN = b"sglang-hicache-engine-key-v1\0"
_ResultT = TypeVar("_ResultT")


class FragmentComponent(str, Enum):
    """Logical contiguous components in one FULL page."""

    K = "k"
    V = "v"
    KV = "kv"


class _PoolFamily(str, Enum):
    MHA = "mha"
    MLA = "mla"


@dataclass(frozen=True, slots=True)
class FragmentSchema:
    """Immutable byte contract for one component of every logical page."""

    component: FragmentComponent
    byte_length: int


@dataclass(frozen=True, slots=True)
class PageFragment:
    """One caller-owned identity and optional host range in a page plan."""

    logical_page_index: int
    component: FragmentComponent
    engine_key: bytes
    host_address: int | None
    byte_length: int
    owner: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _RegisteredFullPool:
    pool: HostKVCache
    family: _PoolFamily
    fragment_schema: tuple[FragmentSchema, ...]
    roots: tuple[torch.Tensor, ...]
    keyspace: ByteArtifactKeyspace
    rank_suffix: str
    layout_id: str


class TensorcastStore(HiCacheStorage):
    """Adapt SGLang FULL pages to TensorCast byte-artifact Session calls."""

    def __init__(self, storage_config: HiCacheStorageConfig) -> None:
        source = storage_config.extra_config
        if source is None:
            raise ValueError(
                "TensorCast HiCache requires storage backend extra configuration"
            )
        tensorcast_config = normalize_tensorcast_config(source)
        session_options = build_tensorcast_session_options(tensorcast_config)
        _validate_rank_topology(storage_config)

        self._storage_config = storage_config
        self._tensorcast_config = tensorcast_config
        self._session_options = session_options
        self._registered: _RegisteredFullPool | None = None
        self._availability_lock = threading.Lock()
        self._disabled = False
        self._failure_logged = False

        # This must remain the final fallible/stateful constructor step so a local
        # validation failure cannot strand ownership in the process registry.
        self._session = claim_tensorcast_store_session(session_options, owner=self)

    @property
    def session(self) -> RegionBackedArtifactSession:
        return self._session

    def register_mem_pool_host(self, mem_pool_host: HostKVCache) -> None:
        registered = self._registered
        if registered is not None:
            if registered.pool is mem_pool_host:
                return
            raise ValueError(
                "TensorCast Store is already registered with a different HostPool"
            )

        candidate = _build_registered_full_pool(
            mem_pool_host,
            storage_config=self._storage_config,
            tensorcast_config=self._tensorcast_config,
        )
        _validate_transfer_mode_registration(
            candidate,
            config=self._tensorcast_config,
            session=self._session,
        )

        super().register_mem_pool_host(mem_pool_host)
        self._registered = candidate

    def exists(self, key: str) -> bool:
        return self.batch_exists([key]) == 1

    def batch_exists(
        self,
        keys: list[str],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> int:
        del extra_info
        if not keys:
            return 0
        if not self._adapter_is_available():
            return 0
        try:
            specs = self._build_artifact_specs(keys)
            result = self._session.batch_exists(specs)
            registered = self._require_registered()
            hit_pages = _leading_complete_page_count(
                result.existence_mask,
                page_count=len(keys),
                fragments_per_page=len(registered.fragment_schema),
            )
        except Exception as exc:
            self._disable_after_runtime_exception("batch_exists", exc)
            return 0
        return self._publish_if_available(hit_pages, fallback=0)

    def _build_page_fragments(self, keys: list[str]) -> tuple[PageFragment, ...]:
        registered = self._require_registered()
        return _expand_page_fragments(registered, keys)

    def _build_artifact_specs(self, keys: list[str]) -> tuple[ByteArtifactSpec, ...]:
        registered = self._require_registered()
        return _expand_artifact_specs(registered, keys)

    def _require_registered(self) -> _RegisteredFullPool:
        if self._registered is None:
            raise RuntimeError("TensorCast Store has no registered FULL HostPool")
        return self._registered

    def get(
        self,
        key: str,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> torch.Tensor | None:
        del key, target_location, target_sizes
        raise NotImplementedError("TensorCast does not support value-oriented get()")

    def batch_get(
        self,
        keys: list[str],
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> list[torch.Tensor | None] | int:
        del keys, target_locations, target_sizes
        raise NotImplementedError(
            "TensorCast does not support value-oriented batch_get()"
        )

    def set(
        self,
        key: str,
        value: Any | None = None,
        target_location: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        del key, value, target_location, target_sizes
        raise NotImplementedError("TensorCast does not support value-oriented set()")

    def batch_set(
        self,
        keys: list[str],
        values: Any | None = None,
        target_locations: Any | None = None,
        target_sizes: Any | None = None,
    ) -> bool:
        del keys, values, target_locations, target_sizes
        raise NotImplementedError(
            "TensorCast does not support value-oriented batch_set()"
        )

    def batch_get_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        del extra_info
        unavailable = [False] * len(keys)
        if not keys:
            return []
        if not self._adapter_is_available():
            return unavailable
        try:
            transfers = self._build_artifact_transfers(keys, host_indices)
            result = self._session.batch_get_into(transfers)
            registered = self._require_registered()
            page_mask = _fold_fragment_mask(
                result.success_mask,
                page_count=len(keys),
                fragments_per_page=len(registered.fragment_schema),
            )
            result_mask = list(_normalize_leading_page_mask(page_mask))
        except Exception as exc:
            self._disable_after_runtime_exception("batch_get_v1", exc)
            return unavailable
        return self._publish_if_available(result_mask, fallback=unavailable)

    def batch_set_v1(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> list[bool]:
        del extra_info
        unavailable = [False] * len(keys)
        if not keys:
            return []
        if not self._adapter_is_available():
            return unavailable
        try:
            transfers = self._build_artifact_transfers(keys, host_indices)
            result = self._session.batch_put_from(transfers)
            registered = self._require_registered()
            result_mask = list(
                _fold_fragment_mask(
                    result.success_mask,
                    page_count=len(keys),
                    fragments_per_page=len(registered.fragment_schema),
                )
            )
        except Exception as exc:
            self._disable_after_runtime_exception("batch_set_v1", exc)
            return unavailable
        return self._publish_if_available(result_mask, fallback=unavailable)

    def _adapter_is_available(self) -> bool:
        with self._availability_lock:
            return not self._disabled

    def _publish_if_available(
        self,
        result: _ResultT,
        *,
        fallback: _ResultT,
    ) -> _ResultT:
        with self._availability_lock:
            if self._disabled:
                return fallback
            return result

    def _disable_after_runtime_exception(
        self,
        operation: str,
        error: Exception,
    ) -> None:
        with self._availability_lock:
            if self._disabled:
                return
            self._disabled = True
            should_log = not self._failure_logged
            self._failure_logged = True

        if not should_log:
            return
        if isinstance(error, RegionSessionFailedError):
            failure = error.failure
            logger.error(
                "TensorCast L3 unavailable: category=session_failed "
                "exception_type=RegionSessionFailedError adapter_operation=%s "
                "failure_code=%s session_operation=%s operation_id=%s message=%s",
                operation,
                failure.code.value,
                failure.operation_kind.value,
                failure.operation_id,
                failure.message,
            )
            return
        if isinstance(error, RegionArtifactInputError):
            logger.error(
                "TensorCast L3 unavailable: category=adapter_input_failure "
                "exception_type=RegionArtifactInputError adapter_operation=%s "
                "message=%s",
                operation,
                error,
            )
            return
        if isinstance(error, RegionSessionTerminatedError):
            logger.error(
                "TensorCast L3 unavailable: category=unexpected_terminated "
                "exception_type=RegionSessionTerminatedError adapter_operation=%s "
                "message=%s",
                operation,
                error,
            )
            return
        logger.exception(
            "TensorCast L3 unavailable: category=adapter_exception "
            "exception_type=%s adapter_operation=%s message=%s",
            type(error).__name__,
            operation,
            error,
        )

    def close(self) -> None:
        with self._availability_lock:
            self._disabled = True
        terminate_tensorcast_store_session(owner=self)

    def _build_artifact_transfers(
        self,
        keys: list[str],
        host_indices: torch.Tensor,
    ) -> tuple[RegionArtifactTransfer, ...]:
        registered = self._require_registered()
        return _expand_artifact_transfers(registered, keys, host_indices)

    def batch_exists_v2(
        self,
        keys: list[str],
        pool_transfers: list[PoolTransfer] | None = None,
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> PoolTransferResult:
        del keys, pool_transfers, extra_info
        raise NotImplementedError("TensorCast initially supports FULL v1 only")

    def batch_get_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        del transfers, extra_info
        raise NotImplementedError("TensorCast initially supports FULL v1 only")

    def batch_set_v2(
        self,
        transfers: list[PoolTransfer],
        extra_info: HiCacheStorageExtraInfo | None = None,
    ) -> dict[str, list[bool]]:
        del transfers, extra_info
        raise NotImplementedError("TensorCast initially supports FULL v1 only")

    def clear(self) -> None:
        raise NotImplementedError("TensorCast has no namespace-wide clear operation")


def _build_registered_full_pool(
    mem_pool_host: HostKVCache,
    *,
    storage_config: HiCacheStorageConfig,
    tensorcast_config: TensorcastConfig,
) -> _RegisteredFullPool:
    family, components, roots = _select_full_pool_adapter(mem_pool_host)
    _validate_full_v1_scope(
        mem_pool_host,
        family=family,
        storage_config=storage_config,
    )
    fragment_schema = _sample_fragment_schema(mem_pool_host, components=components)
    _validate_roots(roots, pool_type=type(mem_pool_host).__name__)

    effective_model_id = resolve_tensorcast_model_id(
        tensorcast_config,
        storage_config.model_name,
    )
    rank_suffix = _build_rank_suffix(storage_config, family=family)
    layout_id = _build_layout_id(mem_pool_host, family=family)
    keyspace = ByteArtifactKeyspace(
        namespace=tensorcast_config.namespace,
        engine="sglang",
        model_id=effective_model_id,
        model_version=tensorcast_config.model_version,
        layout_id=layout_id,
    )
    return _RegisteredFullPool(
        pool=mem_pool_host,
        family=family,
        fragment_schema=fragment_schema,
        roots=roots,
        keyspace=keyspace,
        rank_suffix=rank_suffix,
        layout_id=layout_id,
    )


def _select_full_pool_adapter(
    mem_pool_host: HostKVCache,
) -> tuple[_PoolFamily, tuple[FragmentComponent, ...], tuple[torch.Tensor, ...]]:
    pool_type = type(mem_pool_host)
    if pool_type is AsymmetricMHATokenToKVPoolHost:
        asymmetric_pool = mem_pool_host
        return (
            _PoolFamily.MHA,
            (FragmentComponent.K, FragmentComponent.V),
            (asymmetric_pool.k_buffer, asymmetric_pool.v_buffer),
        )
    if pool_type is MHATokenToKVPoolHost:
        mha_pool = mem_pool_host
        return (
            _PoolFamily.MHA,
            (FragmentComponent.K, FragmentComponent.V),
            (mha_pool.kv_buffer, mha_pool.kv_buffer),
        )
    if pool_type is MLATokenToKVPoolHost:
        mla_pool = mem_pool_host
        return (
            _PoolFamily.MLA,
            (FragmentComponent.KV,),
            (mla_pool.kv_buffer,),
        )
    raise NotImplementedError(
        "TensorCast FULL v1 does not support HostPool type "
        f"{pool_type.__module__}.{pool_type.__qualname__}"
    )


def _validate_full_v1_scope(
    mem_pool_host: HostKVCache,
    *,
    family: _PoolFamily,
    storage_config: HiCacheStorageConfig,
) -> None:
    if mem_pool_host.layout not in _SUPPORTED_FULL_LAYOUTS:
        raise NotImplementedError(
            "TensorCast FULL v1 supports only page_first and page_first_direct, "
            f"got layout={mem_pool_host.layout!r}"
        )
    if mem_pool_host.mtp_draft_device_pools:
        raise NotImplementedError(
            "TensorCast FULL v1 does not support packed MTP draft pools"
        )
    if mem_pool_host.dcp_size != 1 or mem_pool_host.dcp_rank != 0:
        raise NotImplementedError(
            "TensorCast FULL v1 does not support DCP-aware HostPool layouts"
        )
    if storage_config.attn_cp_size != 1 or storage_config.attn_cp_rank != 0:
        raise NotImplementedError(
            "TensorCast FULL v1 requires attention context parallel size 1"
        )
    if storage_config.tp_lcm_size is not None:
        raise NotImplementedError("TensorCast FULL v1 does not support TP-LCM")
    if storage_config.should_split_heads:
        raise NotImplementedError("TensorCast FULL v1 does not support split heads")
    expected_mla = family is _PoolFamily.MLA
    if storage_config.is_mla_model is not expected_mla:
        raise ValueError(
            "TensorCast HostPool family conflicts with is_mla_model: "
            f"pool_family={family.value}, is_mla_model={storage_config.is_mla_model}"
        )
    if mem_pool_host.page_size <= 0 or mem_pool_host.page_num <= 0:
        raise ValueError(
            "TensorCast FULL HostPool requires positive page_size and page_num"
        )


def _sample_fragment_schema(
    mem_pool_host: HostKVCache,
    *,
    components: tuple[FragmentComponent, ...],
) -> tuple[FragmentSchema, ...]:
    sample_indices = torch.arange(mem_pool_host.page_size, dtype=torch.int64)
    pointers, byte_lengths = mem_pool_host.get_page_buffer_meta(sample_indices)
    expected_count = len(components)
    if len(pointers) != expected_count or len(byte_lengths) != expected_count:
        raise ValueError(
            "TensorCast FULL registration metadata must contain exactly one page: "
            f"expected_components={expected_count}, pointers={len(pointers)}, "
            f"byte_lengths={len(byte_lengths)}"
        )
    if any(type(pointer) is not int or pointer <= 0 for pointer in pointers):
        raise ValueError(
            "TensorCast FULL registration metadata contains a non-positive pointer"
        )
    if any(
        type(byte_length) is not int or byte_length <= 0 for byte_length in byte_lengths
    ):
        raise ValueError(
            "TensorCast FULL registration metadata contains an invalid byte length"
        )
    if type(mem_pool_host) is MHATokenToKVPoolHost and (
        byte_lengths[0] != byte_lengths[1]
    ):
        raise ValueError(
            "TensorCast standard MHA registration requires equal K/V byte lengths"
        )
    return tuple(
        FragmentSchema(component=component, byte_length=byte_length)
        for component, byte_length in zip(components, byte_lengths, strict=True)
    )


def _validate_roots(roots: tuple[torch.Tensor, ...], *, pool_type: str) -> None:
    for root in roots:
        if type(root) is not torch.Tensor:
            raise TypeError(
                f"TensorCast {pool_type} backing roots must be torch.Tensor objects"
            )
        if root.device.type != "cpu" or not root.is_contiguous():
            raise ValueError(
                f"TensorCast {pool_type} backing roots must be contiguous CPU tensors"
            )


def _validate_transfer_mode_registration(
    registered: _RegisteredFullPool,
    *,
    config: TensorcastConfig,
    session: RegionBackedArtifactSession,
) -> None:
    if config.transfer_mode is TensorcastTransferMode.ALLOCATOR:
        allocator = registered.pool.allocator
        if type(allocator) is not TensorcastHostTensorAllocator:
            raise ValueError(
                "TensorCast allocator mode requires TensorcastHostTensorAllocator "
                "backing for the registered HostPool"
            )
        if allocator.session is not session:
            raise ValueError(
                "TensorCast HostPool allocator and Store must use the same Session"
            )
        return

    maximum_logical_batch_pages = min(STORAGE_BATCH_SIZE, registered.pool.page_num)
    page_bytes = sum(fragment.byte_length for fragment in registered.fragment_schema)
    required_capacity_bytes = maximum_logical_batch_pages * page_bytes
    configured_capacity_bytes = config.scratch.capacity_bytes
    if configured_capacity_bytes < required_capacity_bytes:
        component_lengths = tuple(
            fragment.byte_length for fragment in registered.fragment_schema
        )
        raise ValueError(
            "TensorCast scratch capacity is insufficient: "
            f"configured_capacity_bytes={configured_capacity_bytes}, "
            f"required_capacity_bytes={required_capacity_bytes}, "
            f"maximum_logical_batch_pages={maximum_logical_batch_pages}, "
            f"page_bytes={page_bytes}, "
            f"pool_type={type(registered.pool).__name__}, "
            f"layout={registered.pool.layout}, "
            f"component_byte_lengths={component_lengths}"
        )


def _build_layout_id(mem_pool_host: HostKVCache, *, family: _PoolFamily) -> str:
    dtype_name = str(mem_pool_host.dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.")
    dtype_token = _DTYPE_LAYOUT_TOKENS.get(dtype_name, dtype_name)
    layout_token = _FULL_LAYOUT_TOKENS[mem_pool_host.layout]
    return (
        f"{_ARTIFACT_LAYOUT_SCHEMA_TOKEN}_{layout_token}_{dtype_token}_"
        f"p{mem_pool_host.page_size}_{family.value}"
    )


def _validate_rank_topology(storage_config: HiCacheStorageConfig) -> None:
    for rank_name, rank, size_name, size in (
        ("tp_rank", storage_config.tp_rank, "tp_size", storage_config.tp_size),
        ("pp_rank", storage_config.pp_rank, "pp_size", storage_config.pp_size),
    ):
        if size <= 0:
            raise ValueError(f"{size_name} must be positive, got {size}")
        if rank < 0 or rank >= size:
            raise ValueError(
                f"{rank_name} must be in [0, {size_name}), got "
                f"{rank_name}={rank}, {size_name}={size}"
            )


def _build_rank_suffix(
    storage_config: HiCacheStorageConfig,
    *,
    family: _PoolFamily,
) -> str:
    _validate_rank_topology(storage_config)
    if family is _PoolFamily.MLA:
        return f"pp{storage_config.pp_rank}of{storage_config.pp_size}"
    tp_suffix = f"tp{storage_config.tp_rank}of{storage_config.tp_size}"
    if storage_config.pp_size == 1:
        return tp_suffix
    return f"{tp_suffix}_pp{storage_config.pp_rank}of{storage_config.pp_size}"


def _build_engine_key(
    rank_suffix: str,
    logical_key: str,
    component: FragmentComponent,
) -> bytes:
    if type(logical_key) is not str:
        raise TypeError(
            "TensorCast logical page keys must be exact str values, got "
            f"{type(logical_key).__name__}"
        )
    component_suffix = (
        b"k" if component is FragmentComponent.KV else component.value.encode()
    )
    rank_bytes = rank_suffix.encode("ascii")
    logical_key_bytes = logical_key.encode("utf-8")
    payload = b"".join(
        (
            _ENGINE_KEY_DOMAIN,
            len(rank_bytes).to_bytes(4, byteorder="big"),
            rank_bytes,
            len(logical_key_bytes).to_bytes(8, byteorder="big"),
            logical_key_bytes,
            component_suffix,
        )
    )
    return sha256(payload).digest()


def _expand_page_fragments(
    registered: _RegisteredFullPool,
    keys: list[str],
) -> tuple[PageFragment, ...]:
    fragments = tuple(
        PageFragment(
            logical_page_index=page_index,
            component=schema.component,
            engine_key=_build_engine_key(
                registered.rank_suffix,
                logical_key,
                schema.component,
            ),
            host_address=None,
            byte_length=schema.byte_length,
            owner=None,
        )
        for page_index, logical_key in enumerate(keys)
        for schema in registered.fragment_schema
    )
    engine_keys = tuple(fragment.engine_key for fragment in fragments)
    if len(set(engine_keys)) != len(engine_keys):
        raise ValueError(
            "TensorCast logical keys must produce unique fragment identities "
            "within one batch"
        )
    return fragments


def _expand_artifact_specs(
    registered: _RegisteredFullPool,
    keys: list[str],
) -> tuple[ByteArtifactSpec, ...]:
    return tuple(
        ByteArtifactSpec(
            keyspace=registered.keyspace,
            engine_key=fragment.engine_key,
            byte_length=fragment.byte_length,
        )
        for fragment in _expand_page_fragments(registered, keys)
    )


def _expand_transfer_fragments(
    registered: _RegisteredFullPool,
    keys: list[str],
    host_indices: torch.Tensor,
) -> tuple[PageFragment, ...]:
    if type(host_indices) is not torch.Tensor:
        raise TypeError(
            "TensorCast FULL host_indices must be a torch.Tensor, got "
            f"{type(host_indices).__name__}"
        )
    if host_indices.ndim != 1:
        raise ValueError(
            "TensorCast FULL host_indices must be one-dimensional, got "
            f"shape={tuple(host_indices.shape)}"
        )
    expected_index_count = len(keys) * registered.pool.page_size
    if len(host_indices) != expected_index_count:
        raise ValueError(
            "TensorCast FULL logical-key/host-index count mismatch: "
            f"logical_pages={len(keys)}, page_size={registered.pool.page_size}, "
            f"expected_host_indices={expected_index_count}, "
            f"actual_host_indices={len(host_indices)}"
        )
    if not keys:
        return ()

    pointers, byte_lengths = registered.pool.get_page_buffer_meta(host_indices)
    fragments = _expand_page_fragments(registered, keys)
    expected_fragment_count = len(fragments)
    if (
        len(pointers) != expected_fragment_count
        or len(byte_lengths) != expected_fragment_count
    ):
        raise ValueError(
            "TensorCast FULL runtime metadata count mismatch: "
            f"expected_fragments={expected_fragment_count}, "
            f"pointers={len(pointers)}, byte_lengths={len(byte_lengths)}"
        )

    fragments_per_page = len(registered.fragment_schema)
    transfer_fragments: list[PageFragment] = []
    for fragment_index, (fragment, pointer, byte_length) in enumerate(
        zip(fragments, pointers, byte_lengths, strict=True)
    ):
        schema_index = fragment_index % fragments_per_page
        schema = registered.fragment_schema[schema_index]
        if type(pointer) is not int or pointer <= 0:
            raise ValueError(
                "TensorCast FULL runtime metadata contains a non-positive pointer: "
                f"fragment_index={fragment_index}, pointer={pointer!r}"
            )
        if type(byte_length) is not int or byte_length <= 0:
            raise ValueError(
                "TensorCast FULL runtime metadata contains an invalid byte length: "
                f"fragment_index={fragment_index}, byte_length={byte_length!r}"
            )
        if byte_length != schema.byte_length:
            raise ValueError(
                "TensorCast FULL runtime byte length differs from the registered "
                f"schema: fragment_index={fragment_index}, "
                f"component={schema.component.value}, expected={schema.byte_length}, "
                f"actual={byte_length}"
            )
        transfer_fragments.append(
            PageFragment(
                logical_page_index=fragment.logical_page_index,
                component=fragment.component,
                engine_key=fragment.engine_key,
                host_address=pointer,
                byte_length=byte_length,
                owner=registered.roots[schema_index],
            )
        )
    return tuple(transfer_fragments)


def _expand_artifact_transfers(
    registered: _RegisteredFullPool,
    keys: list[str],
    host_indices: torch.Tensor,
) -> tuple[RegionArtifactTransfer, ...]:
    transfers: list[RegionArtifactTransfer] = []
    for fragment in _expand_transfer_fragments(registered, keys, host_indices):
        if fragment.host_address is None or fragment.owner is None:
            raise RuntimeError("TensorCast transfer fragment has no owned host range")
        offset_bytes = fragment.host_address - int(fragment.owner.data_ptr())
        span = HostMemorySpan.from_tensor(
            fragment.owner,
            offset_bytes=offset_bytes,
            byte_length=fragment.byte_length,
        )
        transfers.append(
            RegionArtifactTransfer(
                artifact=ByteArtifactSpec(
                    keyspace=registered.keyspace,
                    engine_key=fragment.engine_key,
                    byte_length=fragment.byte_length,
                ),
                span=span,
            )
        )
    return tuple(transfers)


def _fold_fragment_mask(
    fragment_mask: tuple[bool, ...],
    *,
    page_count: int,
    fragments_per_page: int,
) -> tuple[bool, ...]:
    if fragments_per_page <= 0:
        raise ValueError("TensorCast fragments_per_page must be positive")
    expected_count = page_count * fragments_per_page
    if len(fragment_mask) != expected_count:
        raise ValueError(
            "TensorCast transfer result length does not match the logical page plan: "
            f"expected={expected_count}, actual={len(fragment_mask)}"
        )
    return tuple(
        all(
            fragment_mask[
                page_index * fragments_per_page : (page_index + 1) * fragments_per_page
            ]
        )
        for page_index in range(page_count)
    )


def _normalize_leading_page_mask(page_mask: tuple[bool, ...]) -> tuple[bool, ...]:
    prefix_complete = True
    normalized: list[bool] = []
    for page_complete in page_mask:
        prefix_complete = prefix_complete and page_complete
        normalized.append(prefix_complete)
    return tuple(normalized)


def _leading_complete_page_count(
    fragment_mask: tuple[bool, ...],
    *,
    page_count: int,
    fragments_per_page: int,
) -> int:
    page_mask = _fold_fragment_mask(
        fragment_mask,
        page_count=page_count,
        fragments_per_page=fragments_per_page,
    )
    prefix = 0
    for page_complete in page_mask:
        if not page_complete:
            break
        prefix += 1
    return prefix
