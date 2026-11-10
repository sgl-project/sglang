"""Sparse L3 state writes with the actual Python and compiled Rust trees."""

import argparse
from array import array
from itertools import count
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.arg_groups.hicache_hook import (
    validate_hicache_mamba_l3_snapshot_interval,
)
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
    create_tree_core,
    select_tree_core_backend,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import index_slots, slot_of, tree_slots_of

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

FULL = FullComponent.component_type
MAMBA = MambaComponent.component_type


def _key(tokens, bigram=False):
    return RadixKey(array("q", tokens), is_bigram=bigram)


def _cache(backend, interval=16, page_size=1, bigram=False):
    # Real index ownership and tree commits; allocator/DMA/storage are CPU fixtures.
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.disable = False
    cache.enable_session_radix_cache = False
    cache.token_to_kv_pool_allocator = Mock()
    cache.enable_mamba_extra_buffer = True
    cache.buffer_pipeline = None
    cache.hicache_mamba_l3_snapshot_interval = interval
    cache.hicache_storage_pass_prefix_keys = True
    cache.sidecar_pool_specs = []
    cache.ongoing_backup = {}
    cache.insert_session_id = None
    cache.cache_controller = SimpleNamespace(write_storage=Mock(side_effect=count()))
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=SimpleNamespace(holds_state_slots=True),
        token_to_kv_pool_allocator=None,
        page_size=page_size,
        tree_slots=tree_slots_of((FULL, MAMBA), checkpoint_interval=4),
        enable_mamba_extra_buffer=True,
        is_eagle=bigram,
        tree_core_backend=backend,
    )
    with get_context().override_server_args(
        _mamba_cache_chunk_size=4,
        mamba_max_states_per_path=-1,
        hicache_mamba_l3_snapshot_interval=interval,
    ):
        cache.components = {
            FULL: FullComponent(cache, params, slot_of(params, FULL)),
            MAMBA: MambaComponent(cache, params, slot_of(params, MAMBA)),
        }
        index_slots(cache, cache.components)
        core = create_tree_core(
            select_tree_core_backend(params), params, cache.components
        )
        assert isinstance(
            core, RustUnifiedTreeCore if backend == "rust" else UnifiedTreeCore
        )
    cache.tree_core = core
    cache.enable_storage = True
    # Donated indices refer to no real pool. Consume release intents, not backups.
    cache._apply_cache_actions = lambda actions: actions.clear()
    return cache


def _insert(cache, length, slot=1, endpoint=False, tokens=None):
    tokens = list(range(length)) if tokens is None else tokens
    key = _key(tokens, cache.tree_core.is_eagle)
    return cache.insert(
        InsertParams(
            key=key,
            value=torch.arange(len(key), dtype=torch.int64),
            mamba_value=torch.tensor([slot], dtype=torch.int64),
            mamba_storage_prompt_boundary=endpoint,
        )
    ).last_device_node


def _backup(cache, node, pending=False):
    core = cache.tree_core
    full, transfers = core.build_backup_spec(node)
    for group in transfers.values():
        for transfer in group:
            transfer.host_indices = transfer.device_indices + 1000
    core.commit_backup(node, full + 1000, transfers)
    if pending:
        core.mark_write_through_pending([node], ack_id=91)


def _has_mamba(spec):
    return any(x.name == PoolName.MAMBA for xs in spec.comp_xfers.values() for x in xs)


def _write_and_ack(cache, node):
    cache.write_backup_storage(node)
    call = cache.cache_controller.write_storage.call_args
    for stored_node, receipt in cache.ongoing_backup.values():
        cache.tree_core.dec_host_lock_ref(stored_node, receipt)
    cache.ongoing_backup.clear()
    return call


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("page_size,bigram", [(1, False), (2, False), (2, True)])
def test_interval_keeps_existing_crossings_and_full_host_state(
    backend, page_size, bigram
):
    cache = _cache(backend, page_size=page_size, bigram=bigram)
    states = []
    selected = []
    # Offset nodes cross 16/32; no checkpoint is manufactured exactly at either grid.
    for i, end in enumerate([6, 12, 18, 24, 30, 36]):
        node = _insert(cache, end + int(bigram), slot=i + 1)
        _backup(cache, node)
        dense = cache.tree_core.build_storage_backup_spec(node, True)
        sparse = cache.tree_core.build_storage_backup_spec(node, True, 16)
        states.append((dense.comp_xfers[MAMBA][0].host_indices.clone(), node))
        assert torch.equal(sparse.host_value, dense.host_value)
        assert sparse.hash_value == dense.hash_value
        assert sparse.token_ids == dense.token_ids
        if _has_mamba(sparse):
            selected.append(end)
        # Filtering is read-only and does not discard the L2 snapshot.
        assert _has_mamba(cache.tree_core.build_storage_backup_spec(node, True))
    assert selected == [18, 36]
    for expected, node in states:
        dense = cache.tree_core.build_storage_backup_spec(node, False)
        torch.testing.assert_close(dense.comp_xfers[MAMBA][0].host_indices, expected)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_controller_filters_l3_but_keeps_kv_and_registered_state_sidecar(backend):
    cache = _cache(backend)
    nodes = [_insert(cache, end, slot=end) for end in [4, 8, 12, 16]]
    for node in nodes:
        _backup(cache, node)
    # A registered state sidecar uses exactly the selected Mamba source indices.
    cache.sidecar_pool_specs = [
        SimpleNamespace(
            pool_name="test_state_sidecar",
            indices_from_pool=PoolName.MAMBA,
            hit_policy="trailing_pages",
        )
    ]
    for node, expected in zip(nodes, [False, False, False, True], strict=True):
        call = _write_and_ack(cache, node)
        assert len(call.args[0]) > 0  # Full KV is written on every node.
        names = {x.name for x in call.kwargs["extra_pools"] or ()}
        assert (PoolName.MAMBA in names) is expected
        assert ("test_state_sidecar" in names) is expected


@pytest.mark.parametrize("backend", ["python", "rust"])
@pytest.mark.parametrize("pending", [False, True])
def test_existing_node_promotion_retries_once_after_host_ack(backend, pending):
    cache = _cache(backend)
    node = _insert(cache, 8)
    _backup(cache, node, pending=pending)
    assert not _has_mamba(cache.tree_core.build_storage_backup_spec(node, False, 16))
    assert _insert(cache, 8, slot=2, endpoint=True) == node
    writes = cache.cache_controller.write_storage.call_count
    assert writes == (0 if pending else 1)
    assert _has_mamba(cache.tree_core.build_storage_backup_spec(node, False, 16))
    _insert(cache, 8, slot=3, endpoint=True)
    assert cache.cache_controller.write_storage.call_count == writes
    if pending:
        cache.tree_core.finish_write_through([node], 91)
        call = _write_and_ack(cache, node)
        assert any(x.name == PoolName.MAMBA for x in call.kwargs["extra_pools"])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_new_endpoint_is_kept_when_its_l2_copy_becomes_available(backend):
    cache = _cache(backend)
    node = _insert(cache, 8, endpoint=True)
    assert cache.cache_controller.write_storage.call_count == 0
    _backup(cache, node)
    call = _write_and_ack(cache, node)
    assert any(x.name == PoolName.MAMBA for x in call.kwargs["extra_pools"])


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_endpoint_mark_stays_on_original_tail_after_split(backend):
    cache = _cache(backend, interval=64)
    tail = _insert(cache, 12, endpoint=True)
    _backup(cache, tail)
    _insert(cache, 12, slot=2, tokens=list(range(8)) + [90, 91, 92, 93])
    tail_spec = cache.tree_core.build_storage_backup_spec(tail, False, 64)
    assert _has_mamba(tail_spec)
    prefix = cache.tree_core.match_prefix(
        MatchPrefixParams(key=_key(list(range(8))))
    ).last_device_node
    assert prefix != tail
    assert not cache.tree_core.promote_mamba_storage_boundary(prefix)
    assert not cache.tree_core.promote_mamba_storage_boundary(
        cache.tree_core.root_node_handle(None)
    )


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_zero_keeps_all_writes_and_reset_drops_endpoint_marks(backend):
    cache = _cache(backend, interval=0)
    node = _insert(cache, 8, endpoint=True)
    _backup(cache, node)
    assert _has_mamba(cache.tree_core.build_storage_backup_spec(node, False))
    # With the policy off, the insertion did not install a preservation mark.
    assert not _has_mamba(cache.tree_core.build_storage_backup_spec(node, False, 64))
    assert cache.tree_core.promote_mamba_storage_boundary(node)
    cache.tree_core.reset()
    new = _insert(cache, 8, slot=2)
    _backup(cache, new)
    assert not _has_mamba(cache.tree_core.build_storage_backup_spec(new, False, 64))


@pytest.mark.parametrize(
    "changes",
    [
        {"hicache_mamba_l3_snapshot_interval": -1},
        {"hicache_mamba_l3_snapshot_interval": True},
        {"hicache_mamba_l3_snapshot_interval": 32769},
        {"mamba_track_interval": 0},
        {"mamba_track_interval": True},
        {"enable_hierarchical_cache": False},
        {"hicache_storage_backend": None},
        {"hicache_host_memory_mode": "buffer_only"},
    ],
)
def test_invalid_interval_configuration_is_rejected(changes):
    cfg = dict(
        hicache_mamba_l3_snapshot_interval=32768,
        mamba_track_interval=1024,
        enable_hierarchical_cache=True,
        hicache_storage_backend="mooncake",
        hicache_host_memory_mode="cache",
    )
    cfg.update(changes)
    with pytest.raises(ValueError):
        validate_hicache_mamba_l3_snapshot_interval(SimpleNamespace(**cfg))


def test_cli_registers_independent_l3_interval():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model-path", "unused", "--hicache-mamba-l3-snapshot-interval", "32768"]
    )
    assert args.hicache_mamba_l3_snapshot_interval == 32768
    validate_hicache_mamba_l3_snapshot_interval(
        SimpleNamespace(hicache_mamba_l3_snapshot_interval=0)
    )


@pytest.mark.parametrize(
    "finished,length,expected", [(False, 4, False), (False, 8, True), (True, 4, True)]
)
def test_prepare_labels_prompt_and_request_endpoints(finished, length, expected):
    component = MambaComponent.__new__(MambaComponent)
    component.cache = SimpleNamespace(
        hicache_mamba_l3_snapshot_interval=16, enable_mamba_extra_buffer=True
    )
    component._select_finished_checkpoint = lambda *_: None
    req = SimpleNamespace(
        origin_input_ids=list(range(8)),
        kv=SimpleNamespace(mamba_last_track_seqlen=None),
    )
    params = InsertParams()
    assert component.prepare_for_caching_req(req, params, length, finished) == 0
    assert params.mamba_storage_prompt_boundary is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
