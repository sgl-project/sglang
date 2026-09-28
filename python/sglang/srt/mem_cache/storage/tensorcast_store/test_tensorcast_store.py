# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to SGLang project

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from tensorcast.api.store import (
    RegionArtifactExistsResult,
    RegionArtifactInputError,
    RegionArtifactTransferResult,
    RegionSessionFailedError,
    RegionSessionFailure,
    RegionSessionFailureCode,
    RegionSessionOperationKind,
    RegionSessionTerminatedError,
)

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    HiCacheStorage,
    HiCacheStorageConfig,
    PoolName,
)
from sglang.srt.mem_cache.pool_host import (
    HostKVCache,
    HostPoolGroup,
    HostTensorAllocator,
    PoolEntry,
)
from sglang.srt.mem_cache.pool_host.mha import (
    AsymmetricMHATokenToKVPoolHost,
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.mem_cache.storage.tensorcast_store.host_allocator import (
    TensorcastConfig,
    TensorcastHostTensorAllocator,
    normalize_tensorcast_config,
)
from sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store import (
    ARTIFACT_LAYOUT_SCHEMA_VERSION,
    FragmentComponent,
    TensorcastStore,
    _build_registered_pool,
    _expand_artifact_specs,
    _expand_transfer_fragments,
    _fold_fragment_mask,
    _leading_complete_page_count,
    _normalize_leading_page_mask,
    _validate_transfer_mode_registration,
)


class _FakeSession:
    def __init__(
        self,
        existence_mask: Sequence[bool] = (),
        *,
        get_mask: Sequence[bool] = (),
        put_mask: Sequence[bool] = (),
        exists_error: Exception | None = None,
        get_error: Exception | None = None,
        put_error: Exception | None = None,
    ) -> None:
        self.existence_mask = tuple(existence_mask)
        self.get_mask = tuple(get_mask)
        self.put_mask = tuple(put_mask)
        self.exists_calls: list[tuple[object, ...]] = []
        self.get_calls: list[tuple[object, ...]] = []
        self.put_calls: list[tuple[object, ...]] = []
        self.exists_error = exists_error
        self.get_error = get_error
        self.put_error = put_error
        self.terminate_calls = 0
        self.allocation_calls: list[
            tuple[tuple[int, ...], torch.dtype, str, torch.Tensor]
        ] = []

    def allocate_host_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        name: str,
    ) -> torch.Tensor:
        tensor = torch.empty(shape, dtype=dtype)
        self.allocation_calls.append((shape, dtype, name, tensor))
        return tensor

    def batch_exists(self, specs: Sequence[object]) -> RegionArtifactExistsResult:
        self.exists_calls.append(tuple(specs))
        if self.exists_error is not None:
            raise self.exists_error
        return RegionArtifactExistsResult(
            existence_mask=self.existence_mask,
            rpc_elapsed_s=0.0,
        )

    def batch_get_into(
        self, transfers: Sequence[object]
    ) -> RegionArtifactTransferResult:
        snapshot = tuple(transfers)
        self.get_calls.append(snapshot)
        if self.get_error is not None:
            raise self.get_error
        return RegionArtifactTransferResult(
            success_mask=self.get_mask,
            operation_id="fake-get" if snapshot else None,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
        )

    def batch_put_from(
        self, transfers: Sequence[object]
    ) -> RegionArtifactTransferResult:
        snapshot = tuple(transfers)
        self.put_calls.append(snapshot)
        if self.put_error is not None:
            raise self.put_error
        return RegionArtifactTransferResult(
            success_mask=self.put_mask,
            operation_id="fake-put" if snapshot else None,
            pack_elapsed_s=0.0,
            copy_elapsed_s=0.0,
            rpc_elapsed_s=0.0,
        )

    def terminate_process_session(self) -> None:
        self.terminate_calls += 1


def _wrapped_config(
    *,
    transfer_mode: str = "scratch",
    scratch_capacity: int = 4096,
    **overrides: object,
) -> dict[str, object]:
    tensorcast: dict[str, object] = {
        "daemon_address": "127.0.0.1:8073",
        "transfer_mode": transfer_mode,
        "scratch": {"capacity_bytes": scratch_capacity},
        "model_id": "model-a",
    }
    tensorcast.update(overrides)
    return {"tensorcast": tensorcast}


def _storage_config(
    *,
    family: str = "mha",
    extra_config: dict[str, object] | None = None,
    **overrides: object,
) -> HiCacheStorageConfig:
    values: dict[str, object] = {
        "tp_rank": 0,
        "tp_size": 1,
        "pp_rank": 0,
        "pp_size": 1,
        "attn_cp_rank": 0,
        "attn_cp_size": 1,
        "is_mla_model": family == "mla",
        "enable_storage_metrics": False,
        "is_page_first_layout": True,
        "model_name": "fallback-model",
        "tp_lcm_size": None,
        "should_split_heads": False,
        "extra_config": extra_config or _wrapped_config(),
    }
    values.update(overrides)
    return HiCacheStorageConfig(**values)  # type: ignore[arg-type]


def _make_pool(
    family: str,
    layout: str,
    *,
    page_num: int = 3,
    page_size: int = 2,
    dtype: torch.dtype = torch.float32,
    allocator: HostTensorAllocator | None = None,
    allocate_with_allocator: bool = False,
) -> HostKVCache:
    size = page_num * page_size
    layer_num = 1
    head_num = 1

    def allocate_root(shape: tuple[int, ...]) -> torch.Tensor:
        if not allocate_with_allocator:
            return torch.empty(shape, dtype=dtype)
        if allocator is None:
            raise ValueError("allocator-backed test pool requires an allocator")
        return allocator.allocate(shape, dtype, "cpu")

    if family == "mha":
        pool = object.__new__(MHATokenToKVPoolHost)
        head_dim = 2
        if layout == "page_first_direct":
            root = allocate_root(
                (2, page_num, layer_num, page_size, head_num, head_dim)
            )
        else:
            root = allocate_root((2, size, layer_num, head_num, head_dim))
        pool.kv_buffer = root
        pool.head_dim = head_dim
    elif family == "asymmetric_mha":
        pool = object.__new__(AsymmetricMHATokenToKVPoolHost)
        head_dim = 2
        v_head_dim = 3
        if layout == "page_first_direct":
            k_root = allocate_root((page_num, layer_num, page_size, head_num, head_dim))
            v_root = allocate_root(
                (page_num, layer_num, page_size, head_num, v_head_dim)
            )
        else:
            k_root = allocate_root((size, layer_num, head_num, head_dim))
            v_root = allocate_root((size, layer_num, head_num, v_head_dim))
        pool.kv_buffer = (k_root, v_root)
        pool.head_dim = head_dim
        pool.v_head_dim = v_head_dim
    elif family == "mla":
        pool = object.__new__(MLATokenToKVPoolHost)
        kv_cache_dim = 3
        if layout == "page_first_direct":
            root = allocate_root((page_num, layer_num, page_size, 1, kv_cache_dim))
        else:
            root = allocate_root((size, layer_num, 1, kv_cache_dim))
        pool.kv_buffer = root
        pool.kv_cache_dim = kv_cache_dim
    else:
        raise ValueError(f"unknown test pool family {family}")

    pool.layout = layout
    pool.page_num = page_num
    pool.page_size = page_size
    pool.size = size
    pool.layer_num = layer_num
    pool.head_num = head_num
    pool.dtype = dtype
    pool.mtp_draft_device_pools = ()
    pool.dcp_size = 1
    pool.dcp_rank = 0
    pool.allocator = allocator or HostTensorAllocator()
    return cast(HostKVCache, pool)


def _candidate(
    family: str,
    layout: str,
    *,
    pool: HostKVCache | None = None,
    storage_overrides: dict[str, object] | None = None,
    config: TensorcastConfig | None = None,
):
    normalized_family = "mla" if family == "mla" else "mha"
    storage_config = _storage_config(
        family=normalized_family,
        **(storage_overrides or {}),
    )
    return _build_registered_pool(
        pool or _make_pool(family, layout),
        storage_config=storage_config,
        tensorcast_config=config
        or normalize_tensorcast_config(storage_config.extra_config),
    )


@pytest.mark.parametrize("layout", ["page_first", "page_first_direct"])
@pytest.mark.parametrize(
    ("family", "components", "lengths"),
    [
        ("mha", (FragmentComponent.K, FragmentComponent.V), (16, 16)),
        (
            "asymmetric_mha",
            (FragmentComponent.K, FragmentComponent.V),
            (16, 24),
        ),
        ("mla", (FragmentComponent.KV,), (24,)),
    ],
)
def test_fragment_registration_matrix_and_root_retention(
    family: str,
    components: tuple[FragmentComponent, ...],
    lengths: tuple[int, ...],
    layout: str,
) -> None:
    pool = _make_pool(family, layout)
    registered = _candidate(family, layout, pool=pool)

    assert tuple(item.component for item in registered.fragment_schema) == components
    assert tuple(item.byte_length for item in registered.fragment_schema) == lengths
    if family == "mha":
        assert registered.roots[0] is pool.kv_buffer
        assert registered.roots[1] is pool.kv_buffer
    elif family == "asymmetric_mha":
        assert registered.roots == (pool.k_buffer, pool.v_buffer)
    else:
        assert registered.roots == (pool.kv_buffer,)


def test_fragment_registration_dispatch_is_exact_and_specific() -> None:
    asymmetric = _candidate("asymmetric_mha", "page_first")
    assert tuple(fragment.byte_length for fragment in asymmetric.fragment_schema) == (
        16,
        24,
    )

    class UnknownMhaPool(MHATokenToKVPoolHost):
        pass

    unknown = object.__new__(UnknownMhaPool)
    with pytest.raises(NotImplementedError, match="does not support HostPool type"):
        _candidate("mha", "page_first", pool=cast(HostKVCache, unknown))

    k_only = object.__new__(MHATokenToKOnlyPoolHost)
    with pytest.raises(NotImplementedError, match="does not support HostPool type"):
        _candidate("mha", "page_first", pool=cast(HostKVCache, k_only))


@pytest.mark.parametrize(
    ("pool_changes", "storage_changes", "message"),
    [
        ({"layout": "layer_first"}, {}, "supports only"),
        ({"layout": "page_head"}, {}, "supports only"),
        ({"mtp_draft_device_pools": (object(),)}, {}, "MTP"),
        ({"dcp_size": 2}, {}, "DCP"),
        ({}, {"attn_cp_size": 2}, "context parallel"),
        ({}, {"tp_lcm_size": 2}, "TP-LCM"),
        ({}, {"should_split_heads": True}, "split heads"),
    ],
)
def test_scope_rejects_unsupported_full_semantics(
    pool_changes: dict[str, object],
    storage_changes: dict[str, object],
    message: str,
) -> None:
    pool = _make_pool("mha", "page_first")
    for name, value in pool_changes.items():
        setattr(pool, name, value)
    with pytest.raises(NotImplementedError, match=message):
        _candidate(
            "mha",
            "page_first",
            pool=pool,
            storage_overrides=storage_changes,
        )


def test_scope_rejects_family_mismatch_and_invalid_page_geometry() -> None:
    pool = _make_pool("mha", "page_first")
    with pytest.raises(ValueError, match="conflicts with is_mla_model"):
        _candidate(
            "mha",
            "page_first",
            pool=pool,
            storage_overrides={"is_mla_model": True},
        )

    pool.page_num = 0
    with pytest.raises(ValueError, match="positive page_size and page_num"):
        _candidate("mha", "page_first", pool=pool)


def test_fragment_registration_rejects_invalid_sample_metadata() -> None:
    pool = _make_pool("mha", "page_first")
    pool.get_page_buffer_meta = lambda indices: ([1], [8])
    with pytest.raises(ValueError, match="exactly one page"):
        _candidate("mha", "page_first", pool=pool)

    pool.get_page_buffer_meta = lambda indices: ([1, 2], [8, 9])
    with pytest.raises(ValueError, match="equal K/V"):
        _candidate("mha", "page_first", pool=pool)


def test_registration_is_idempotent_only_for_same_pool() -> None:
    session = _FakeSession()
    source = _wrapped_config()
    store = _store_without_registry(source, session=session)
    pool = _make_pool("mha", "page_first")

    store.register_mem_pool_host(pool)
    registered = store._registered
    store.register_mem_pool_host(pool)
    assert store._registered is registered
    assert store.mem_pool_host is pool

    with pytest.raises(ValueError, match="different HostPool"):
        store.register_mem_pool_host(_make_pool("mha", "page_first"))


def test_registration_constructor_claim_is_final_local_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sglang.srt.runtime_context as runtime_context
    from sglang.srt.mem_cache.storage.tensorcast_store import tensorcast_store

    session = _FakeSession()
    claim_calls: list[object] = []

    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(world_rank=0, world_size=1),
    )

    def claim(options: object, *, owner: TensorcastStore) -> _FakeSession:
        assert owner._storage_config.model_name == "fallback-model"
        assert owner._registered is None
        assert owner._session_options is options
        claim_calls.append(owner)
        return session

    monkeypatch.setattr(tensorcast_store, "claim_tensorcast_store_session", claim)
    store = TensorcastStore(_storage_config())

    assert store.session is session
    assert claim_calls == [store]


def test_identity_inputs_are_canonical_and_transfer_mode_independent() -> None:
    source = _wrapped_config(
        namespace=" Tenant ",
        model_id=" Model/A ",
        model_version=" Release-1 ",
    )
    config = normalize_tensorcast_config(source)
    storage_overrides = {
        "tp_rank": 2,
        "tp_size": 4,
        "pp_rank": 1,
        "pp_size": 2,
    }
    registered = _candidate(
        "mha",
        "page_first_direct",
        storage_overrides=storage_overrides,
        config=config,
    )
    logical_key = "0123456789abcdef" * 4
    specs = _expand_artifact_specs(registered, [logical_key])

    assert registered.layout_id == "ff1_pfd_f32_p2_mha"
    assert registered.keyspace.namespace == "Tenant"
    assert registered.keyspace.model_id == "Model/A"
    assert registered.keyspace.model_version == "Release-1"
    assert registered.keyspace.engine == "sglang"
    assert specs[0].engine_key.hex() == (
        "0fb433b34c33e3a7ae1cfdf0c0b5017a58695d3fe63cde10e9d388572381aae1"
    )
    assert specs[1].engine_key.hex() == (
        "23e9888f2289481bd631f48f7518cba486e75bc06203c077403d4797ffb37adc"
    )
    assert len(specs[0].engine_key) == len(specs[1].engine_key) == 32
    assert tuple(spec.byte_length for spec in specs) == (16, 16)
    assert ARTIFACT_LAYOUT_SCHEMA_VERSION == "full-fragment-v1"

    scratch = _candidate("mha", "page_first", config=config)
    allocator_config = normalize_tensorcast_config(
        _wrapped_config(
            transfer_mode="allocator",
            namespace=" Tenant ",
            model_id=" Model/A ",
            model_version=" Release-1 ",
        )
    )
    allocator = _candidate("mha", "page_first", config=allocator_config)
    assert _expand_artifact_specs(scratch, ["same-key"]) == _expand_artifact_specs(
        allocator, ["same-key"]
    )


@pytest.mark.parametrize(
    ("dtype", "dtype_token"),
    [
        (torch.bfloat16, "bf16"),
        (torch.float16, "f16"),
        (torch.float32, "f32"),
        (torch.float64, "f64"),
        (torch.uint8, "u8"),
    ],
)
def test_identity_layout_uses_compact_dtype_tokens(
    dtype: torch.dtype,
    dtype_token: str,
) -> None:
    pool = _make_pool("mha", "page_first", dtype=dtype)
    registered = _candidate("mha", "page_first", pool=pool)

    assert registered.layout_id == f"ff1_pf_{dtype_token}_p2_mha"


def test_identity_mla_uses_pp_scope_and_historical_k_suffix() -> None:
    registered = _candidate(
        "mla",
        "page_first",
        storage_overrides={"tp_rank": 3, "tp_size": 8, "pp_rank": 1, "pp_size": 2},
    )
    specs = _expand_artifact_specs(registered, ["opaque"])

    assert len(specs) == 1
    assert specs[0].engine_key.hex() == (
        "5549b303b603995e8fe9873e7934566218131a9a3f547982cdd809cd6ca0ef27"
    )
    assert registered.layout_id == "ff1_pf_f32_p2_mla"


def test_identity_rejects_duplicate_logical_keys_and_non_string_keys() -> None:
    registered = _candidate("mha", "page_first")
    with pytest.raises(ValueError, match="unique fragment identities"):
        _expand_artifact_specs(registered, ["duplicate", "duplicate"])
    with pytest.raises(TypeError, match="exact str"):
        _expand_artifact_specs(registered, [cast(str, b"not-text")])


@pytest.mark.parametrize(
    ("mask", "expected"),
    [
        ((True, True, True, True, True, True), 3),
        ((False, True, True, True, True, True), 0),
        ((True, False, True, True, True, True), 0),
        ((True, True, False, True, True, True), 1),
        ((True, True, True, True, True, False), 2),
    ],
)
def test_exists_folds_fragments_to_leading_page_prefix(
    mask: tuple[bool, ...], expected: int
) -> None:
    session = _FakeSession(mask)
    store = _registered_store(session=session)

    assert store.batch_exists(["a", "b", "c"]) == expected
    assert len(session.exists_calls) == 1
    assert len(session.exists_calls[0]) == 6


def test_exists_single_item_and_empty_call_behavior() -> None:
    session = _FakeSession((True, True))
    store = _registered_store(session=session)

    assert store.exists("one") is True
    assert len(session.exists_calls) == 1
    session.existence_mask = (False, True)
    assert store.exists("one") is False
    assert len(session.exists_calls) == 2
    assert store.batch_exists([]) == 0
    assert len(session.exists_calls) == 2


def test_exists_rejects_result_length_mismatch() -> None:
    with pytest.raises(ValueError, match="result length"):
        _leading_complete_page_count((True,), page_count=1, fragments_per_page=2)


def test_scratch_capacity_accepts_exact_and_rejects_one_byte_short() -> None:
    session = _FakeSession()
    pool = _make_pool("asymmetric_mha", "page_first", page_num=3)
    exact = 3 * (16 + 24)
    exact_config = normalize_tensorcast_config(_wrapped_config(scratch_capacity=exact))
    registered = _candidate(
        "asymmetric_mha",
        "page_first",
        pool=pool,
        config=exact_config,
    )
    _validate_transfer_mode_registration(
        registered,
        config=exact_config,
        session=cast(object, session),
    )

    short_config = normalize_tensorcast_config(
        _wrapped_config(scratch_capacity=exact - 1)
    )
    with pytest.raises(ValueError) as error_info:
        _validate_transfer_mode_registration(
            registered,
            config=short_config,
            session=cast(object, session),
        )
    message = str(error_info.value)
    for expected_diagnostic in (
        f"configured_capacity_bytes={exact - 1}",
        f"required_capacity_bytes={exact}",
        "maximum_logical_batch_pages=3",
        "page_bytes=40",
        "pool_type=AsymmetricMHATokenToKVPoolHost",
        "layout=page_first",
        "component_byte_lengths=(16, 24)",
    ):
        assert expected_diagnostic in message


def test_scratch_capacity_caps_page_count_at_storage_batch_size() -> None:
    session = _FakeSession()
    pool = _make_pool("mla", "page_first", page_num=STORAGE_BATCH_SIZE + 1)
    required = STORAGE_BATCH_SIZE * 24
    config = normalize_tensorcast_config(_wrapped_config(scratch_capacity=required))
    registered = _candidate("mla", "page_first", pool=pool, config=config)

    _validate_transfer_mode_registration(
        registered,
        config=config,
        session=cast(object, session),
    )


@pytest.mark.parametrize(
    ("family", "layout", "fragments_per_page"),
    [
        ("mha", "page_first_direct", 2),
        ("mla", "page_first", 1),
    ],
)
def test_span_batch_get_v1_and_batch_set_v1_construct_ordered_transfers(
    family: str,
    layout: str,
    fragments_per_page: int,
) -> None:
    fragment_count = 2 * fragments_per_page
    session = _FakeSession(
        get_mask=(True,) * fragment_count,
        put_mask=(True,) * fragment_count,
    )
    pool = _make_pool(family, layout)
    source = _wrapped_config()
    store = _store_without_registry(source, session=session, family=family)
    store.register_mem_pool_host(pool)
    keys = ["page-a", "page-b"]
    host_indices = torch.tensor([2, 3, 4, 5], dtype=torch.int64)
    expected_pointers, expected_lengths = pool.get_page_buffer_meta(host_indices)

    assert store.batch_set_v1(keys, host_indices) == [True, True]
    assert store.batch_get_v1(keys, host_indices) == [True, True]
    assert len(session.put_calls) == len(session.get_calls) == 1

    for call in (session.put_calls[0], session.get_calls[0]):
        assert len(call) == fragment_count
        assert [transfer.span.address for transfer in call] == expected_pointers
        assert [transfer.span.byte_length for transfer in call] == expected_lengths
        assert [transfer.artifact.byte_length for transfer in call] == expected_lengths

    registered = store._require_registered()
    fragments = _expand_transfer_fragments(registered, keys, host_indices)
    assert [fragment.host_address for fragment in fragments] == expected_pointers
    for fragment_index, fragment in enumerate(fragments):
        schema_index = fragment_index % fragments_per_page
        root = registered.roots[schema_index]
        assert fragment.owner is root
        assert fragment.host_address is not None
        assert fragment.host_address - root.data_ptr() >= 0


def test_span_asymmetric_mha_retains_distinct_component_roots_and_lengths() -> None:
    session = _FakeSession(put_mask=(True, True))
    pool = _make_pool("asymmetric_mha", "page_first_direct")
    store = _store_without_registry(_wrapped_config(), session=session)
    store.register_mem_pool_host(pool)
    host_indices = torch.tensor([2, 3], dtype=torch.int64)

    assert store.batch_set_v1(["page"], host_indices) == [True]
    fragments = _expand_transfer_fragments(
        store._require_registered(), ["page"], host_indices
    )
    assert tuple(fragment.owner for fragment in fragments) == (
        pool.k_buffer,
        pool.v_buffer,
    )
    assert tuple(fragment.byte_length for fragment in fragments) == (16, 24)
    assert tuple(transfer.span.address for transfer in session.put_calls[0]) == tuple(
        fragment.host_address for fragment in fragments
    )


def test_folding_get_normalizes_tail_after_first_incomplete_page() -> None:
    session = _FakeSession(
        get_mask=(True, True, False, True, True, True),
        put_mask=(True, True, False, True, True, True),
    )
    store = _registered_store(session=session)
    keys = ["a", "b", "c"]
    host_indices = torch.arange(6, dtype=torch.int64)

    assert store.batch_set_v1(keys, host_indices) == [True, False, True]
    assert store.batch_get_v1(keys, host_indices) == [True, False, False]
    assert _fold_fragment_mask(
        session.get_mask,
        page_count=3,
        fragments_per_page=2,
    ) == (True, False, True)
    assert _normalize_leading_page_mask((True, False, True)) == (
        True,
        False,
        False,
    )


def test_batch_get_v1_and_batch_set_v1_empty_calls_skip_session_rpc() -> None:
    session = _FakeSession()
    store = _registered_store(session=session)
    empty_indices = torch.empty(0, dtype=torch.int64)

    assert store.batch_get_v1([], empty_indices) == []
    assert store.batch_set_v1([], empty_indices) == []
    assert session.get_calls == []
    assert session.put_calls == []


def test_failure_runtime_metadata_validation_disables_before_session_rpc() -> None:
    def make_case() -> tuple[_FakeSession, HostKVCache, TensorcastStore]:
        session = _FakeSession(put_mask=(True, True))
        pool = _make_pool("mha", "page_first")
        store = _store_without_registry(_wrapped_config(), session=session)
        store.register_mem_pool_host(pool)
        return session, pool, store

    session, _, store = make_case()
    assert store.batch_set_v1(["page"], torch.tensor([0], dtype=torch.int64)) == [False]
    assert store._disabled is True
    assert session.put_calls == []

    session, _, store = make_case()
    assert store.batch_set_v1(["page"], torch.tensor([[0, 1]], dtype=torch.int64)) == [
        False
    ]
    assert store._disabled is True
    assert session.put_calls == []

    session, pool, store = make_case()
    pool.get_page_buffer_meta = lambda indices: ([pool.kv_buffer.data_ptr()], [16])
    assert store.batch_set_v1(["page"], torch.tensor([0, 1], dtype=torch.int64)) == [
        False
    ]
    assert store._disabled is True
    assert session.put_calls == []

    session, pool, store = make_case()
    pool.get_page_buffer_meta = lambda indices: (
        [pool.kv_buffer.data_ptr(), pool.v_buffer.data_ptr()],
        [16, 15],
    )
    assert store.batch_set_v1(["page"], torch.tensor([0, 1], dtype=torch.int64)) == [
        False
    ]
    assert store._disabled is True
    assert session.put_calls == []


def test_scratch_and_allocator_build_identical_transfer_artifacts() -> None:
    keys = ["same-page"]
    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    scratch = _candidate("mha", "page_first")
    allocator_config = normalize_tensorcast_config(
        _wrapped_config(transfer_mode="allocator")
    )
    allocator = _candidate("mha", "page_first", config=allocator_config)

    scratch_fragments = _expand_transfer_fragments(scratch, keys, host_indices)
    allocator_fragments = _expand_transfer_fragments(allocator, keys, host_indices)
    assert tuple(fragment.engine_key for fragment in scratch_fragments) == tuple(
        fragment.engine_key for fragment in allocator_fragments
    )
    assert tuple(fragment.byte_length for fragment in scratch_fragments) == tuple(
        fragment.byte_length for fragment in allocator_fragments
    )


@pytest.mark.parametrize(
    ("family", "allocation_count", "fragments_per_page"),
    [
        ("mha", 1, 2),
        ("asymmetric_mha", 2, 2),
        ("mla", 1, 1),
    ],
)
def test_allocator_direct_no_copy_multi_region_transfer(
    family: str,
    allocation_count: int,
    fragments_per_page: int,
) -> None:
    fragment_count = 2 * fragments_per_page
    session = _FakeSession(
        get_mask=(True,) * fragment_count,
        put_mask=(True,) * fragment_count,
    )
    allocator = TensorcastHostTensorAllocator(cast(object, session))
    pool = _make_pool(
        family,
        "page_first_direct",
        allocator=allocator,
        allocate_with_allocator=True,
    )
    allocated_roots = tuple(call[3] for call in session.allocation_calls)
    assert len(allocated_roots) == allocation_count

    source = _wrapped_config(transfer_mode="allocator")
    store = _store_without_registry(
        source,
        session=session,
        family="mla" if family == "mla" else "mha",
    )
    store.register_mem_pool_host(pool)
    registered = store._require_registered()
    expected_roots = (
        (allocated_roots[0], allocated_roots[0]) if family == "mha" else allocated_roots
    )
    assert len(registered.roots) == len(expected_roots)
    assert all(
        registered_root is expected_root
        for registered_root, expected_root in zip(
            registered.roots, expected_roots, strict=True
        )
    )

    keys = ["allocator-page-0", "allocator-page-1"]
    host_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    allocation_calls_before_transfer = tuple(session.allocation_calls)

    assert store.batch_set_v1(keys, host_indices) == [True, True]
    assert store.batch_get_v1(keys, host_indices) == [True, True]
    assert tuple(session.allocation_calls) == allocation_calls_before_transfer
    assert len(session.put_calls) == len(session.get_calls) == 1

    fragments = _expand_transfer_fragments(registered, keys, host_indices)
    expected_owners = tuple(
        registered.roots[index % fragments_per_page] for index in range(fragment_count)
    )
    assert all(
        fragment.owner is expected_owner
        for fragment, expected_owner in zip(fragments, expected_owners, strict=True)
    )
    for call in (session.put_calls[0], session.get_calls[0]):
        assert tuple(transfer.span.address for transfer in call) == tuple(
            fragment.host_address for fragment in fragments
        )
        assert tuple(transfer.span.byte_length for transfer in call) == tuple(
            fragment.byte_length for fragment in fragments
        )

    if family == "asymmetric_mha":
        assert allocated_roots[0] is pool.k_buffer
        assert allocated_roots[1] is pool.v_buffer
        assert {fragment.owner.data_ptr() for fragment in fragments} == {
            pool.k_buffer.data_ptr(),
            pool.v_buffer.data_ptr(),
        }


def _public_runtime_error(
    error_kind: str,
    operation: str,
) -> Exception:
    if error_kind == "input":
        return RegionArtifactInputError("invalid runtime artifact input")
    if error_kind == "terminated":
        return RegionSessionTerminatedError("unexpected terminal Session")
    if error_kind != "failed":
        raise ValueError(f"unknown public error kind {error_kind!r}")
    operation_kind = {
        "exists": RegionSessionOperationKind.EXISTS,
        "get": RegionSessionOperationKind.GET_INTO,
        "put": RegionSessionOperationKind.PUT_FROM,
    }[operation]
    return RegionSessionFailedError(
        RegionSessionFailure(
            code=RegionSessionFailureCode.TRANSPORT,
            message="daemon transport failed",
            operation_kind=operation_kind,
            operation_id=f"{operation}-operation",
            occurred_at=datetime.now(timezone.utc),
        )
    )


def _session_with_runtime_error(
    operation: str,
    error: Exception,
) -> _FakeSession:
    if operation == "exists":
        return _FakeSession(exists_error=error)
    if operation == "get":
        return _FakeSession(get_error=error)
    if operation == "put":
        return _FakeSession(put_error=error)
    raise ValueError(f"unknown runtime operation {operation!r}")


def _invoke_runtime_operation(store: TensorcastStore, operation: str) -> object:
    if operation == "exists":
        return store.batch_exists(["page"])
    host_indices = torch.tensor([0, 1], dtype=torch.int64)
    if operation == "get":
        return store.batch_get_v1(["page"], host_indices)
    if operation == "put":
        return store.batch_set_v1(["page"], host_indices)
    raise ValueError(f"unknown runtime operation {operation!r}")


@pytest.mark.parametrize("operation", ["exists", "get", "put"])
@pytest.mark.parametrize("error_kind", ["input", "failed", "terminated"])
def test_failure_public_exceptions_disable_once_and_skip_future_session_calls(
    operation: str,
    error_kind: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = _session_with_runtime_error(
        operation,
        _public_runtime_error(error_kind, operation),
    )
    store = _registered_store(session=session)
    caplog.set_level(logging.ERROR, logger=TensorcastStore.__module__)

    expected = 0 if operation == "exists" else [False]
    assert _invoke_runtime_operation(store, operation) == expected
    assert store._disabled is True
    transition_records = [
        record
        for record in caplog.records
        if record.message.startswith("TensorCast L3 unavailable:")
    ]
    assert len(transition_records) == 1
    expected_category = {
        "input": "adapter_input_failure",
        "failed": "session_failed",
        "terminated": "unexpected_terminated",
    }[error_kind]
    assert f"category={expected_category}" in transition_records[0].message

    assert store.batch_exists(["later"]) == 0
    later_indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_get_v1(["later"], later_indices) == [False]
    assert store.batch_set_v1(["later"], later_indices) == [False]
    assert (
        len(session.exists_calls) + len(session.get_calls) + len(session.put_calls) == 1
    )
    assert (
        len(
            [
                record
                for record in caplog.records
                if record.message.startswith("TensorCast L3 unavailable:")
            ]
        )
        == 1
    )


def test_failure_unexpected_planner_span_session_and_folding_errors_are_contained(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger=TensorcastStore.__module__)

    planner_store = _registered_store(session=_FakeSession())
    monkeypatch.setattr(
        planner_store,
        "_build_artifact_specs",
        lambda keys: (_ for _ in ()).throw(RuntimeError("planner defect")),
    )
    assert planner_store.batch_exists(["page"]) == 0

    span_store = _registered_store(session=_FakeSession())

    def fail_span(
        cls: type[object],
        tensor: torch.Tensor,
        *,
        offset_bytes: int,
        byte_length: int,
    ) -> object:
        del cls, tensor, offset_bytes, byte_length
        raise RuntimeError("span construction defect")

    from sglang.srt.mem_cache.storage.tensorcast_store import tensorcast_store

    monkeypatch.setattr(
        tensorcast_store.HostMemorySpan,
        "from_tensor",
        classmethod(fail_span),
    )
    indices = torch.tensor([0, 1], dtype=torch.int64)
    assert span_store.batch_get_v1(["page"], indices) == [False]
    monkeypatch.undo()

    session_store = _registered_store(
        session=_FakeSession(put_error=RuntimeError("Session boundary defect"))
    )
    assert session_store.batch_set_v1(["page"], indices) == [False]

    folding_store = _registered_store(session=_FakeSession(get_mask=(True,)))
    assert folding_store.batch_get_v1(["page"], indices) == [False]

    transition_records = [
        record
        for record in caplog.records
        if record.message.startswith("TensorCast L3 unavailable:")
    ]
    assert len(transition_records) == 4
    assert all(
        "category=adapter_exception" in record.message for record in transition_records
    )


def test_failure_ordinary_item_misses_do_not_disable_adapter() -> None:
    session = _FakeSession(
        existence_mask=(False, False),
        get_mask=(False, False),
        put_mask=(False, False),
    )
    store = _registered_store(session=session)
    indices = torch.tensor([0, 1], dtype=torch.int64)

    assert store.batch_exists(["page"]) == 0
    assert store.batch_get_v1(["page"], indices) == [False]
    assert store.batch_set_v1(["page"], indices) == [False]
    assert store._disabled is False

    session.existence_mask = (True, True)
    session.get_mask = (True, True)
    session.put_mask = (True, True)
    assert store.batch_exists(["page"]) == 1
    assert store.batch_get_v1(["page"], indices) == [True]
    assert store.batch_set_v1(["page"], indices) == [True]


def test_concurrent_failure_publication_gate_suppresses_racing_success() -> None:
    class BlockingSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.exists_started = threading.Event()
            self.release_exists = threading.Event()

        def batch_exists(
            self,
            specs: Sequence[object],
        ) -> RegionArtifactExistsResult:
            self.exists_calls.append(tuple(specs))
            self.exists_started.set()
            if not self.release_exists.wait(timeout=5):
                raise RuntimeError("test did not release blocked exists")
            return RegionArtifactExistsResult(
                existence_mask=(True, True),
                rpc_elapsed_s=0.0,
            )

        def batch_put_from(
            self,
            transfers: Sequence[object],
        ) -> RegionArtifactTransferResult:
            self.put_calls.append(tuple(transfers))
            raise RegionArtifactInputError("disable while exists is in flight")

    session = BlockingSession()
    store = _registered_store(session=session)
    published: list[int] = []
    exists_thread = threading.Thread(
        target=lambda: published.append(store.batch_exists(["page"]))
    )
    exists_thread.start()
    assert session.exists_started.wait(timeout=5)

    indices = torch.tensor([0, 1], dtype=torch.int64)
    assert store.batch_set_v1(["page"], indices) == [False]
    session.release_exists.set()
    exists_thread.join(timeout=5)

    assert not exists_thread.is_alive()
    assert published == [0]
    assert len(session.exists_calls) == len(session.put_calls) == 1


def test_worker_callbacks_survive_get_and_backup_failures() -> None:
    indices = torch.tensor([0, 1], dtype=torch.int64)
    operation = SimpleNamespace(request_id="worker-operation")

    get_store = _registered_store(
        session=_FakeSession(get_error=RuntimeError("prefetch failure"))
    )
    get_controller = object.__new__(HiCacheController)
    get_controller.storage_backend = get_store
    assert (
        get_controller._page_get_zero_copy(
            operation,
            ["page"],
            indices,
        )
        == 0
    )

    put_store = _registered_store(
        session=_FakeSession(put_error=RuntimeError("backup failure"))
    )
    put_controller = object.__new__(HiCacheController)
    put_controller.storage_backend = put_store
    assert put_controller._page_set_zero_copy(["page"], indices) is False


def test_controller_tensorcast_factory_and_host_pool_group_use_v1_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory

    registry_entry = StorageBackendFactory._registry["tensorcast"]
    assert registry_entry["module_path"] == (
        "sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store"
    )
    assert registry_entry["class_name"] == "TensorcastStore"

    class FakeStorageBackend:
        def __init__(self) -> None:
            self.registered: list[HostKVCache] = []

        def register_mem_pool_host(self, pool: HostKVCache) -> None:
            self.registered.append(pool)

    backend = FakeStorageBackend()

    def create_backend(
        cls: type[StorageBackendFactory],
        backend_name: str,
        storage_config: object,
        mem_pool_host: HostKVCache,
        **kwargs: object,
    ) -> FakeStorageBackend:
        del cls, storage_config, mem_pool_host, kwargs
        assert backend_name == "tensorcast"
        return backend

    monkeypatch.setattr(
        StorageBackendFactory,
        "create_backend",
        classmethod(create_backend),
    )

    anchor = _make_pool("mha", "page_first")
    anchor.can_use_write_back_jit = False
    anchor.device = "cpu"
    group = HostPoolGroup(
        [
            PoolEntry(
                name=PoolName.KV,
                host_pool=anchor,
                device_pool=object(),
                layer_mapper=lambda layer_id: layer_id,
                is_primary_index_anchor=True,
            )
        ]
    )
    controller = object.__new__(HiCacheController)
    controller.enable_storage = False
    controller.storage_stop_event = threading.Event()
    controller.storage_host_pool = group.anchor_entry.host_pool
    controller.mem_pool_host = group
    controller.host_memory_mode = "cache"
    controller.page_size = anchor.page_size
    controller._stop_storage_threads = lambda: None
    controller._start_storage_threads = lambda: None
    controller._create_sync_groups = lambda: []
    controller._generate_storage_config = lambda model_name, extra_config: (
        SimpleNamespace(is_mla_model=False, tp_rank=0, extra_config=extra_config or {})
    )

    controller.attach_storage_backend(
        "tensorcast",
        model_name="model",
        storage_backend_extra_config={"tensorcast": {}},
    )

    assert backend.registered == [anchor]
    assert controller.page_get_func == controller._page_get_zero_copy
    assert controller.page_set_func == controller._page_set_zero_copy


def test_registration_allocator_requires_thin_allocator_and_same_session() -> None:
    session = _FakeSession()
    other_session = _FakeSession()
    config = normalize_tensorcast_config(_wrapped_config(transfer_mode="allocator"))
    pool = _make_pool(
        "mha",
        "page_first",
        allocator=TensorcastHostTensorAllocator(cast(object, session)),
    )
    registered = _candidate("mha", "page_first", pool=pool, config=config)

    _validate_transfer_mode_registration(
        registered,
        config=config,
        session=cast(object, session),
    )
    with pytest.raises(ValueError, match="same Session"):
        _validate_transfer_mode_registration(
            registered,
            config=config,
            session=cast(object, other_session),
        )

    pool.allocator = HostTensorAllocator()
    with pytest.raises(ValueError, match="TensorcastHostTensorAllocator"):
        _validate_transfer_mode_registration(
            registered,
            config=config,
            session=cast(object, session),
        )


def test_failure_containment_keeps_unsupported_value_v1_v2_and_clear_apis_explicit() -> (
    None
):
    store = _registered_store(session=_FakeSession())
    calls = (
        lambda: store.get("key"),
        lambda: store.batch_get(["key"]),
        lambda: store.set("key"),
        lambda: store.batch_set(["key"]),
        lambda: store.batch_exists_v2(["key"]),
        lambda: store.batch_get_v2([]),
        lambda: store.batch_set_v2([]),
        store.clear,
    )
    for call in calls:
        with pytest.raises(NotImplementedError):
            call()

    assert (
        TensorcastStore.register_mem_host_pool_v2
        is HiCacheStorage.register_mem_host_pool_v2
    )
    sidecar = object()
    store.register_mem_host_pool_v2(cast(HostKVCache, sidecar), PoolName.MAMBA)
    assert store.registered_pools == {PoolName.MAMBA: sidecar}


def _store_without_registry(
    source: dict[str, object],
    *,
    session: _FakeSession,
    family: str = "mha",
) -> TensorcastStore:
    store = object.__new__(TensorcastStore)
    store._storage_config = _storage_config(family=family, extra_config=source)
    store._tensorcast_config = normalize_tensorcast_config(source)
    store._registered = None
    store._session = cast(object, session)
    store._availability_lock = threading.Lock()
    store._disabled = False
    store._failure_logged = False
    return store


def _registered_store(*, session: _FakeSession) -> TensorcastStore:
    source = _wrapped_config()
    store = _store_without_registry(source, session=session)
    store.register_mem_pool_host(_make_pool("mha", "page_first"))
    return store
