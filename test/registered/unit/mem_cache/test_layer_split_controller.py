"""Community ACK integration and LayerSplit startup contracts.

Transport, byte layout and storage result masks are exercised in the transfer
and Mooncake tests; these cases focus on the controller/scheduler boundary.
"""

import threading
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from sglang.srt.arg_groups.hicache_hook import validate_layer_split_storage
from sglang.srt.managers.cache_controller import (
    HiCacheController as BaseHiCacheController,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.layer_split.layer_split_config import StagingBufferConfig
from sglang.srt.mem_cache.layer_split.layer_split_engine import LayerSplitTransferEngine
from sglang.srt.mem_cache.layer_split.layer_split_host_view import LayerSplitHostView
from sglang.srt.mem_cache.layer_split.layer_split_utils import (
    PREFETCH,
    SharedWindowPGPool,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def controller():
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.page_size = 2
    cc.storage_stop_event = threading.Event()
    cc.prefetch_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.prefetch_buffer = Queue()
    cc.prefetch_sync_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.backup_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.append_host_mem_release = mock.Mock()
    cc.staging_engine = LayerSplitTransferEngine(
        0,
        controller=cc,
        staging_buffer_config=StagingBufferConfig(
            page_size=2,
            shard_size=1,
            window_size=8,
            backup_max_outstanding_operations=2,
        ),
    )
    cc.staging_engine.stages[PREFETCH] = SimpleNamespace(
        active=False, io=None, works=[]
    )
    cc.staging_engine.pg_pool = mock.Mock(spec=["abort", "assert_idle", "close"])
    return cc


def operation(pages=5):
    op = PrefetchOperation(
        CacheRequestHandle("request", 0),
        [1] * (pages * 2),
        pool_transfers=[PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)],
    )
    op.hash_value = [str(i) for i in range(pages)]
    op.host_indices = torch.arange(pages * 2)
    op.storage_hit_count = pages * 2
    return op


def test_storage_coordination_follows_runtime_engine_replacement():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.cache_controller = controller()
    original = cache.cache_controller.staging_engine
    assert cache.storage_coordination is original
    cache.cache_controller.staging_engine = None
    assert cache.storage_coordination is None
    replacement = object()
    cache.cache_controller.staging_engine = replacement
    assert cache.storage_coordination is replacement


@pytest.mark.parametrize("short", [False, True])
def test_native_worker_publishes_fixed_ack_schedule_and_leaves_progress_to_scheduler(
    short,
):
    cc, op = controller(), operation()
    cc.prefetch_buffer.put(op)

    def run_window(plan, window, indices, *, stop_requested):
        assert plan.op_id == op.id
        if short or window.index == len(plan.windows()) - 1:
            cc.storage_stop_event.set()
        return (1 if short else window.page_count), False

    with (
        mock.patch.object(cc, "_page_transfer", side_effect=cc.staging_engine.prefetch),
        mock.patch.object(cc.staging_engine, "prefetch_window", side_effect=run_window),
    ):
        cc.prefetch_io_aux_func()
    assert op.completed_tokens == 0  # worker must never race ACK publication
    assert not op.pool_transfers_done
    acks = [cc.prefetch_sync_queue.get_nowait() for _ in range(2)]
    assert acks[0].completed_tokens == (2 if short else 10)
    assert acks[0].pool_hits == {}
    assert acks[1].completed_req is True
    assert cc.prefetch_sync_queue.empty()
    assert not cc.staging_engine.stages[PREFETCH].active
    # Actual ACK reduction, scheduler consumption and cancellation/tail release
    # are covered together by the real-Gloo test in test_layer_split_transfers.
    cc.append_host_mem_release.assert_not_called()


@pytest.mark.parametrize("agreed", [0, 4, 10])
def test_allocation_agreement_precedes_submission_and_rolls_back_tail(agreed):
    cc, op = controller(), operation()
    cc.mem_pool_host = SimpleNamespace(free=mock.Mock())
    indices = op.host_indices
    with (
        mock.patch.object(cc, "_allocate_storage_hit", return_value=(indices, 10)),
        mock.patch.object(cc, "free_prefetch_host_buffers") as rollback,
        mock.patch.object(
            cc.staging_engine, "align_prefetch_allocation", return_value=agreed
        ),
    ):
        result, count = cc.allocate_storage_hit(
            op, 10, allow_partial=True, min_tokens=2, evict_host=mock.Mock()
        )
    assert count == agreed
    if agreed == 0:
        assert result is None
        rollback.assert_called_once_with(op, indices)
    else:
        assert torch.equal(result, indices[:agreed])
        rollback.assert_not_called()
    assert cc.prefetch_buffer.empty()
    if 0 < agreed < 10:
        assert torch.equal(cc.mem_pool_host.free.call_args.args[0], indices[agreed:])


def test_failed_local_allocation_still_participates_in_agreement():
    cc, op = controller(), operation()
    with (
        mock.patch.object(cc, "_allocate_storage_hit", return_value=(None, 10)),
        mock.patch.object(
            cc.staging_engine, "align_prefetch_allocation", return_value=0
        ) as agreement,
    ):
        assert cc.allocate_storage_hit(
            op, 10, allow_partial=True, min_tokens=2, evict_host=mock.Mock()
        ) == (None, 0)
    agreement.assert_called_once_with(op, 0)


@pytest.mark.parametrize(
    "local_length,agreed_length,mismatch",
    [(10, 4, False), (10, 0, False), (0, 0, False), (10, 10, True)],
)
def test_single_allocation_agrees_length_and_identity(
    local_length, agreed_length, mismatch
):
    cc, op = controller(), operation()
    engine = cc.staging_engine
    digest = engine._identity([op.request_id, len(op.hash_value), *op.hash_value])

    def reduce(states, reduction):
        assert reduction == torch.distributed.ReduceOp.MIN
        assert states.tolist() == [local_length, digest, -digest]
        states[0] = agreed_length
        if mismatch:
            states[2] = -digest - 1

    with mock.patch.object(engine, "_reduce_control", side_effect=reduce) as agreement:
        if mismatch:
            with pytest.raises(RuntimeError, match="identities differ"):
                engine.align_prefetch_allocation(op, local_length)
        else:
            assert engine.align_prefetch_allocation(op, local_length) == agreed_length
    agreement.assert_called_once()


def test_backup_uses_native_fifo_and_cap_waits_for_scheduler_retirement():
    cc = controller()
    ids = [cc.write_storage(torch.arange(2), [1, 2], [name]) for name in ("a", "b")]
    assert cc.write_storage(torch.arange(2), [1, 2], ["overflow"]) is None
    first = cc.backup_queue.get_nowait()
    cc.ack_backup_queue.put(first)
    assert cc.write_storage(torch.arange(2), [1, 2], ["still-full"]) is None
    assert first.id == ids[0]
    assert cc.backup_queue.get_nowait().id == ids[1]
    cc.staging_engine.retire_backup(first.id)
    assert cc.write_storage(torch.arange(2), [1, 2], ["c"]) is not None


def test_busy_backup_prevents_reset_or_detach():
    cc = controller()
    cc.write_storage(torch.arange(2), [1, 2], ["a"])
    with pytest.raises(RuntimeError, match="completions are consumed"):
        cc.staging_engine.reset()
    with pytest.raises(RuntimeError, match="completions are consumed"):
        cc.staging_engine.detach()
    cc.staging_engine.pg_pool.close.assert_not_called()


def test_runtime_attach_rejects_a_backend_without_layer_split_support():
    cc = controller()
    cc._layer_split_direct = True
    cc.host_memory_mode = "cache"
    with pytest.raises(ValueError, match="requires mooncake"):
        cc.attach_storage_backend("file")


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("live", [(0, 1, 2, 3, 4), (0, 3), (4,)])
def test_host_view_keeps_kv_and_compact_indexer_geometry_separate(rank, live):
    from test_layer_split_mooncake import pools

    from sglang.srt.layers.cp.utils import get_layer_shard_range

    host = pools(rank, 2, 5)
    device = host.entry_map[PoolName.KV].device_pool
    device.host_pool_decls = lambda: (
        SimpleNamespace(pool_name=PoolName.KV, owned_device_layers=None),
        SimpleNamespace(pool_name=PoolName.INDEXER, owned_device_layers=live),
    )
    lo, hi = get_layer_shard_range(rank, 2, 5)
    count = sum(lo <= layer < hi for layer in live)
    idx = host.get_pool(PoolName.INDEXER)
    idx.layer_num = idx.target_layer_num = count
    idx.index_k_with_scale_buffer = torch.zeros((count, 4, 6), dtype=torch.uint8)
    view = LayerSplitHostView(host)
    assert view.component_layer_counts["target"] == (3, 2)
    assert sum(view.component_layer_counts["indexer"]) == len(live)
    assert view.shard_views["indexer"].owned_layers == count
    data = torch.arange(count * 6, dtype=torch.uint8)
    view.shard_views["indexer"].write_page(2, data)
    result = torch.empty_like(data)
    view.shard_views["indexer"].read_page(2, result)
    assert torch.equal(data, result)


def test_startup_initializes_two_buffers_before_native_workers():
    from test_layer_split_mooncake import pools

    cc = controller()
    cc.staging_engine = None
    cc._staging_buffer_config = StagingBufferConfig(window_size=8)
    cc.mem_pool_host = pools(0, 2, 5)
    cc.storage_backend = mock.Mock(spec=["register_mem_host_pool_v2"])
    cc.attn_cp_group = object()
    cc.pp_prefetch_command_group = None
    cc.backup_skip = True
    with (
        mock.patch.object(LayerSplitTransferEngine, "_reduce_control"),
        mock.patch.object(
            torch.distributed, "get_process_group_ranks", return_value=[0, 1]
        ),
        mock.patch.object(SharedWindowPGPool, "attach") as attach,
        mock.patch.object(SharedWindowPGPool, "close"),
        mock.patch.object(BaseHiCacheController, "_start_storage_threads") as native,
    ):
        cc._start_storage_threads()
        attach.assert_called_once()
        native.assert_called_once()
        assert cc.storage_backend.register_mem_host_pool_v2.call_count == 4
        assert len(cc.staging_engine.stages) == 2
        assert not cc.backup_skip
        cc.staging_engine.detach()


@pytest.mark.parametrize("staged", [True, False])
def test_storage_configuration_preserves_native_fields(staged):
    from test_layer_split_mooncake import pools

    cc = controller()
    cc.mem_pool_host = pools(1, 2, 5)
    cc._staging_buffer_config = StagingBufferConfig() if staged else None
    cc._layer_split_direct = not staged
    native = HiCacheStorageConfig(
        tp_rank=0,
        tp_size=1,
        pp_rank=0,
        pp_size=1,
        attn_cp_rank=1,
        attn_cp_size=2,
        is_mla_model=True,
        enable_storage_metrics=False,
        is_page_first_layout=False,
        model_name="model",
        dp_rank=3,
    )
    with mock.patch.object(
        BaseHiCacheController, "_generate_storage_config", return_value=native
    ):
        result = cc._generate_storage_config()
    assert result.dp_rank == 3 and result.model_name == "model"
    assert native.layer_shard is None and native.external_buffer_pools == ()
    if staged:
        assert result.external_buffer_pools == ("kv", "indexer")
    else:
        assert (result.layer_shard.start, result.layer_shard.end) == (3, 5)


def config(**changes):
    values = dict(
        enable_hicache_layer_split_staging=True,
        enable_dsa_cache_layer_split=True,
        enable_hierarchical_cache=True,
        hicache_storage_backend="mooncake",
        hicache_host_memory_mode="cache",
        hicache_mem_layout="layer_first",
        hicache_io_backend="kernel",
        pp_size=1,
        disaggregation_mode="prefill",
        enable_unified_memory=False,
        enable_unified_cache_external_linker=False,
        speculative_algorithm=None,
        speculative_draft_model_path=None,
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("backend", ["mooncake", "flashkv"])
def test_layer_split_backend_validation_keeps_extension_name(backend):
    validate_layer_split_storage(config(hicache_storage_backend=backend))


@pytest.mark.parametrize(
    "field,value",
    [
        ("enable_dsa_cache_layer_split", False),
        ("enable_hierarchical_cache", False),
        ("hicache_host_memory_mode", "buffer_only"),
        ("hicache_storage_backend", "file"),
        ("hicache_mem_layout", "page_first"),
        ("hicache_io_backend", "unknown"),
        ("pp_size", 2),
        ("disaggregation_mode", "decode"),
        ("enable_unified_memory", True),
        ("enable_unified_cache_external_linker", True),
        ("speculative_algorithm", "EAGLE"),
        ("speculative_draft_model_path", "draft"),
    ],
)
def test_layer_split_rejects_unsupported_geometry_and_lifecycle(field, value):
    with pytest.raises(ValueError):
        validate_layer_split_storage(config(**{field: value}))
