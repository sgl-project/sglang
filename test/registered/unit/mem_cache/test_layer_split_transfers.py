"""Engine transfer contracts; real CPU Gloo and in-memory L3."""

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import timedelta
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.layer_split.layer_split_config import StagingBufferConfig
from sglang.srt.mem_cache.layer_split.layer_split_engine import (
    LayerSplitTransferEngine,
    StagingBackupOp,
    StagingPrefetchOp,
)
from sglang.srt.mem_cache.layer_split.layer_split_host_view import LayerSplitShardView
from sglang.srt.mem_cache.layer_split.layer_split_plan import EXCHANGE_COMPONENTS
from sglang.srt.mem_cache.layer_split.layer_split_staging_pool import FixedStageBuffer
from sglang.srt.mem_cache.layer_split.layer_split_utils import (
    BACKUP,
    PREFETCH,
    SharedWindowPGPool,
    WindowJob,
    owned_layer_range,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="base-a-test-cpu")


def _prefetch(engine, plan, indices, **kwargs):
    return engine.prefetch_run(StagingPrefetchOp(plan, host_indices=indices), **kwargs)


def _backup(engine, plan, indices):
    return engine.backup_run(StagingBackupOp(plan), indices)


def _config(shards=2, **kwargs):
    return StagingBufferConfig(
        page_size=2,
        shard_size=shards,
        window_size=8,
        requested_exchange_group_count=2,
        get_timeout_s=10,
        set_timeout_s=10,
        window_agreement_timeout_s=20,
        **kwargs,
    )


def _views(rank, shards, layers, pages, indexer_layers=None):
    lo, hi = owned_layer_range(rank, shards, layers)
    indexer_layers = range(layers) if indexer_layers is None else indexer_layers
    indexer_owned = tuple(i for i in indexer_layers if lo <= i < hi)
    capacity = -(-layers // shards)
    common = dict(layout="layer_first", layer_num=capacity, target_layer_num=capacity)
    pools = {
        "target": SimpleNamespace(
            **common,
            token_stride_size=4,
            kv_buffer=torch.zeros((capacity, 2 * pages, 1, 2), dtype=torch.float16),
        ),
        "indexer": SimpleNamespace(
            layout="layer_first",
            layer_num=len(indexer_owned),
            target_layer_num=len(indexer_owned),
            indexer_page_stride_size=6,
            indexer_dtype=torch.uint8,
            index_k_with_scale_buffer=torch.zeros(
                (len(indexer_owned), pages, 6), dtype=torch.uint8
            ),
        ),
    }
    views = {
        c: LayerSplitShardView(
            pools[c],
            c,
            owned_layers=hi - lo if c == "target" else len(indexer_owned),
            capacity_layers=pools[c].layer_num,
            page_size=2,
        )
        for c in EXCHANGE_COMPONENTS
    }
    views["target"].test_layer_ids = tuple(range(lo, hi))
    views["indexer"].test_layer_ids = indexer_owned
    return views


class _Storage:
    def __init__(self, objects):
        self.objects = objects
        self.pools = {}
        self.fail_get = False
        self.get_gate = None

    def register_mem_host_pool_v2(self, pool, name):
        self.pools[name] = pool

    def batch_set_v2(self, transfers):
        for transfer in transfers:
            slab = self.pools[transfer.physical_pool_name]
            for key, position in zip(transfer.keys, transfer.host_indices):
                self.objects[(str(transfer.name), key)] = (
                    slab.buffer[int(position)].numpy().tobytes()
                )
        return {t.name: [True] * len(t.keys) for t in transfers}

    def batch_get_v2(self, transfers):
        if self.get_gate is not None:
            assert self.get_gate.wait(10)
        if self.fail_get:
            self.fail_get = False
            raise ConnectionError("settled GET disconnect")
        result = {}
        for transfer in transfers:
            slab = self.pools[transfer.physical_pool_name]
            mask = []
            for key, position in zip(transfer.keys, transfer.host_indices):
                value = self.objects.get((str(transfer.name), key))
                mask.append(value is not None)
                if value is not None:
                    data = torch.frombuffer(bytearray(value), dtype=torch.uint8)
                    slab.buffer[int(position)].copy_(
                        data.reshape_as(slab.buffer[int(position)])
                    )
            result[transfer.name] = mask
        return result


def test_fixed_buffers_have_independent_memory_and_expected_capacity():
    config = _config()
    counts = [owned_layer_range(r, 2, 5) for r in range(2)]
    counts = [hi - lo for lo, hi in counts]
    prefetch = FixedStageBuffer(
        PREFETCH,
        config,
        {c: counts for c in EXCHANGE_COMPONENTS},
        {"target": 4, "indexer": 3},
        0,
        False,
    )
    backup = FixedStageBuffer(
        BACKUP,
        config,
        {c: counts for c in EXCHANGE_COMPONENTS},
        {"target": 4, "indexer": 3},
        0,
        False,
    )
    assert prefetch.max_rounds == 2
    for component in EXCHANGE_COMPONENTS:
        assert (
            prefetch.components[component].buffer.data_ptr()
            != backup.components[component].buffer.data_ptr()
        )
        assert (
            prefetch.scratch[component].data_ptr()
            != backup.scratch[component].data_ptr()
        )
    for buffer in (prefetch, backup):
        assert buffer.components["target"].buffer.shape == (2, 5, 2, 4)


@pytest.mark.parametrize("outcome", ["complete", "stop", "short_get"])
def test_prefetch_finishes_l2_copy_before_starting_next_window(outcome):
    config = _config(shards=1)
    views = _views(0, 1, 3, 9)
    storage = _Storage({})
    engine = LayerSplitTransferEngine(
        storage_backend=storage,
        rank=0,
        staging_buffer_config=config,
        layer_count=3,
        exchange_ranks=[0],
        l2_pools=views,
        pin_memory=False,
    )
    with mock.patch.object(engine.pg_pool, "attach"):
        engine.attach()
    for component, name in (("target", PoolName.KV), ("indexer", PoolName.INDEXER)):
        size = engine.stages[PREFETCH].buffer.components[component].buffer[0].numel()
        for page in range(9):
            storage.objects[(str(name), str(page))] = bytes([page + 1]) * size
    if outcome == "short_get":
        del storage.objects[(str(PoolName.INDEXER), "5")]
    plan = engine.build_plan(
        op_id=1, request_id="serial", page_hashes=list(map(str, range(9)))
    )
    calls = []
    start_io, copy_shards = engine._start_io, engine._copy_shards

    def get(lane, plan, window, owned):
        calls.append(("get", window.index))
        return start_io(lane, plan, window, owned)

    def communicate(lane, plan, window, rounds):
        calls.append(("exchange", window.index))
        buffer = engine.stages[lane].buffer
        for component in EXCHANGE_COMPONENTS:
            for exchange_round in rounds:
                buffer.scratch[component][exchange_round.index, 0].copy_(
                    buffer.components[component]
                    .buffer[exchange_round.index]
                    .reshape(-1)
                )

    def unpack(lane, window, rounds, indices):
        result = copy_shards(lane, window, rounds, indices)
        calls.append(("unpack", window.index))
        return result

    try:
        with (
            mock.patch("psutil.Process.send_signal") as shutdown,
            mock.patch.object(engine, "_start_io", side_effect=get),
            mock.patch.object(engine, "_communicate", side_effect=communicate),
            mock.patch.object(engine, "_copy_shards", side_effect=unpack),
            mock.patch.object(
                engine.pg_pool,
                "submit",
                side_effect=lambda lane, job: job.execute(job.values()),
            ),
        ):
            result = _prefetch(
                engine, plan, torch.arange(18), stop_requested=lambda: outcome == "stop"
            )
            shutdown.assert_not_called()
        windows, published = {"complete": (3, 9), "stop": (1, 4), "short_get": (2, 5)}[
            outcome
        ]
        assert calls == [
            (kind, window)
            for window in range(windows)
            for kind in ("get", "exchange", "unpack")
        ]
        assert result.published_pages == published
        assert not engine.stages[PREFETCH].active
        for view in views.values():
            actual = torch.empty(
                view.owned_layers * 2 * view.row_bytes, dtype=torch.uint8
            )
            for page in range(9):
                view.read_page(page * 2, actual)
                assert (actual == (page + 1 if page < published else 0)).all()
    finally:
        for stage in engine.stages.values():
            stage.executor.shutdown(wait=True)


@pytest.mark.parametrize("failed_pages", [{1}, {0, 1, 2, 3}])
def test_shared_backup_read_failures_skip_pages_and_continue_later_windows(
    failed_pages,
):
    config = _config(shards=1)
    views = _views(0, 1, 3, 9)
    storage = _Storage({})
    engine = LayerSplitTransferEngine(
        storage_backend=storage,
        rank=0,
        staging_buffer_config=config,
        layer_count=3,
        exchange_ranks=[0],
        l2_pools=views,
        pin_memory=False,
    )
    with mock.patch.object(engine.pg_pool, "attach"):
        engine.attach()
    for component, view in views.items():
        for page in range(9):
            view.write_page(page * 2, _payload(view, 0, page, component))
    real_read = views["indexer"].read_page
    communicated = []

    def read(token_base, destination):
        if token_base // 2 in failed_pages:
            raise OSError("unreadable L2 page")
        real_read(token_base, destination)

    def gather(lane, plan, window, rounds):
        communicated.append(window.index)
        buffer = engine.stages[lane].buffer
        for component in EXCHANGE_COMPONENTS:
            for exchange_round in rounds:
                buffer.components[component].buffer[exchange_round.index].reshape(
                    -1
                ).copy_(buffer.scratch[component][exchange_round.index, 0])

    try:
        with (
            mock.patch("psutil.Process.send_signal") as shutdown,
            mock.patch.object(views["indexer"], "read_page", side_effect=read),
            mock.patch.object(engine, "_communicate", side_effect=gather),
            mock.patch.object(
                engine.pg_pool,
                "submit",
                side_effect=lambda lane, job: job.execute(job.values()),
            ),
        ):
            plan = engine.build_plan(
                op_id=1, request_id="read-failure", page_hashes=list(map(str, range(9)))
            )
            result = _backup(engine, plan, torch.arange(18))
        shutdown.assert_not_called()
        assert (result.written_pages, result.skipped_pages) == (
            9 - len(failed_pages),
            len(failed_pages),
        )
        assert communicated == ([1, 2] if len(failed_pages) == 4 else [0, 1, 2])
        for component, name in (("target", PoolName.KV), ("indexer", PoolName.INDEXER)):
            for page in range(9):
                key = (str(name), str(page))
                if page in failed_pages:
                    assert key not in storage.objects
                else:
                    assert (
                        storage.objects[key]
                        == _payload(views[component], 0, page, component)
                        .numpy()
                        .tobytes()
                    )
        assert not engine.stages[BACKUP].active
    finally:
        for stage in engine.stages.values():
            stage.executor.shutdown(wait=True)


@pytest.mark.parametrize(
    "error", [OSError("IO"), ConnectionError("connection"), TimeoutError("SDK timeout")]
)
def test_settled_io_failure_is_a_miss_but_live_timeout_is_not(error):
    engine = LayerSplitTransferEngine.__new__(LayerSplitTransferEngine)
    ended = Future()
    ended.set_exception(error)
    assert engine._finish_io((ended, 0), 3) == [False] * 3
    live = Future()
    with pytest.raises(TimeoutError, match="did not settle"):
        engine._finish_io((live, 0), 3)
    assert not live.cancelled()


def test_malformed_backend_result_is_not_a_storage_miss():
    engine = LayerSplitTransferEngine.__new__(LayerSplitTransferEngine)
    future = Future()
    future.set_result({PoolName.KV: [True]})
    with pytest.raises(ValueError, match="INDEXER"):
        engine._finish_io((future, 0), 1)


@pytest.mark.parametrize("incomplete", [PoolName.KV, PoolName.INDEXER])
@pytest.mark.parametrize("short_mask", [False, True])
def test_shared_io_requires_both_component_masks(incomplete, short_mask):
    engine = LayerSplitTransferEngine.__new__(LayerSplitTransferEngine)
    results = {PoolName.KV: [True] * 3, PoolName.INDEXER: [True] * 3}
    results[incomplete] = [True] if short_mask else [True, False, True]
    future = Future()
    future.set_result(results)
    mask = engine._finish_io((future, 0), 3)
    assert mask == ([True, False, False] if short_mask else [True, False, True])
    assert mask.index(False) == 1  # Prefetch publishes only the leading run.
    assert sum(mask) == (1 if short_mask else 2)  # Backup may retain later pages.


@pytest.mark.parametrize("failure", ["missing_handle", "wait_error", "timeout"])
def test_shared_all_to_all_failure_retains_buffers_and_never_sets_l3(failure):
    storage = _Storage({})
    engine = LayerSplitTransferEngine(
        storage_backend=storage,
        rank=0,
        staging_buffer_config=_config(shards=1),
        layer_count=3,
        exchange_ranks=[0],
        l2_pools=_views(0, 1, 3, 4),
        pin_memory=False,
    )
    with mock.patch.object(engine.pg_pool, "attach"):
        engine.attach()
    engine.pg_pool.data_groups = [object()]
    work = (
        None
        if failure == "missing_handle"
        else SimpleNamespace(
            is_completed=lambda: failure != "timeout",
            wait=mock.Mock(
                side_effect=RuntimeError("gloo failed")
                if failure == "wait_error"
                else None
            ),
        )
    )
    error = {
        "missing_handle": AttributeError,
        "wait_error": RuntimeError,
        "timeout": TimeoutError,
    }[failure]
    try:
        with (
            mock.patch("psutil.Process.send_signal") as shutdown,
            mock.patch.object(
                engine.pg_pool,
                "submit",
                side_effect=lambda lane, job: job.execute(job.values()),
            ),
            mock.patch.object(dist, "all_to_all_single", return_value=work),
            mock.patch(
                "sglang.srt.mem_cache.layer_split.layer_split_engine.time.monotonic",
                side_effect=[0, 0, 999],
            ),
        ):
            plan = engine.build_plan(
                op_id=1, request_id="failed-work", page_hashes=["a"]
            )
            with pytest.raises(error):
                _backup(engine, plan, torch.arange(2))
        shutdown.assert_called_once()
        assert not storage.objects
        assert engine.backup_ops_run == 0
        stage = engine.stages[BACKUP]
        assert stage.active and len(stage.works) == 2
        assert all(recv.numel() and send.numel() for _, recv, send, _ in stage.works)
        with pytest.raises(RuntimeError, match="active operations"):
            engine.detach()
    finally:
        for stage in engine.stages.values():
            stage.executor.shutdown(wait=True)


def test_component_slab_exposes_exact_page_addresses_to_storage():
    from sglang.srt.mem_cache.layer_split.layer_split_staging_pool import (
        StagingComponent,
    )

    slab = StagingComponent(
        name="test",
        num_pages=4,
        layer_count=3,
        page_size=2,
        row_bytes=4,
        pin_memory=False,
    )
    pointers, sizes = slab.get_page_buffer_meta([0, 2])
    assert sizes == [24, 24]
    assert pointers == [slab.buffer.data_ptr(), slab.buffer.data_ptr() + 48]
    assert slab.get_hybrid_pool_buffer()[0] is slab.buffer
    for invalid in (-1, 4):
        with pytest.raises(ValueError, match="out of range"):
            slab.get_page_buffer_meta([invalid])


def test_shared_scheduler_selects_global_readiness_not_local_arrival_order():
    pool = SharedWindowPGPool(_config(), [0, 1])
    executed = []
    prefetch = WindowJob(11, lambda: [2, 1], lambda v: executed.append(PREFETCH) or v)
    backup = WindowJob(22, lambda: [1, 1], lambda v: executed.append(BACKUP) or v)
    pool._jobs[:] = [prefetch, backup]
    # Both are ready locally, but only backup is ready at every peer. The
    # preferred lane must not override that globally reduced readiness result.
    responses = [
        [0, 0, -11, 1, 22, -22, 0, 1],
        [1, 1],
        [1, 11, -11, 0, 0, 0, 0, 1],
        [2, 1],
        [0, 0, 0, 0, 0, 0, 1, 1],
    ]
    with mock.patch.object(pool, "_min", side_effect=responses):
        pool._dispatch()
    assert executed == [BACKUP, PREFETCH]
    assert prefetch.result.result() == [2, 1]
    assert backup.result.result() == [1, 1]
    assert pool._jobs == [None, None]
    assert pool._completion.result() is None


@pytest.mark.parametrize("failure", ["identity", "transport", "peer_abort"])
@pytest.mark.parametrize("lane", [PREFETCH, BACKUP])
def test_shared_scheduler_fails_waiters_without_executing_wrong_data(failure, lane):
    pool = SharedWindowPGPool(_config(), [0, 1])
    execute = mock.Mock()
    job = WindowJob(11, lambda: [2, 1], execute)
    pool._jobs[lane] = job
    response = [0, 0, 0, 0, 0, 0, 0, int(failure != "peer_abort")]
    response[3 * lane : 3 * lane + 3] = [
        1,
        11,
        -12 if failure == "identity" else -11,
    ]
    reducer = mock.Mock(return_value=response)
    if failure == "transport":
        reducer.side_effect = RuntimeError("control transport failed")
    with (
        mock.patch.object(pool, "_min", reducer),
        mock.patch("psutil.Process.send_signal") as signal,
    ):
        pool._dispatch()
    execute.assert_not_called()
    signal.assert_called_once()
    with pytest.raises(RuntimeError):
        job.result.result()
    with pytest.raises(RuntimeError):
        pool.submit(BACKUP, WindowJob(22, lambda: [], execute))


def test_failed_communication_keeps_fixed_buffer_and_work_references():
    engine = LayerSplitTransferEngine(0, staging_buffer_config=_config())
    stage = SimpleNamespace(active=False, io=None, works=[object()])
    retained = stage.works[0]
    engine.stages = {PREFETCH: stage}
    engine.pg_pool = mock.Mock(spec=["abort", "close", "assert_idle"])
    plan = engine.build_plan(op_id=1, request_id="fatal", page_hashes=["a"])
    op = StagingPrefetchOp(plan)
    with mock.patch("psutil.Process.send_signal") as signal:
        with pytest.raises(RuntimeError, match="collective"):
            with mock.patch.object(
                engine, "prefetch_window", side_effect=RuntimeError("collective failed")
            ):
                engine.prefetch_run(op)
    signal.assert_called_once()
    engine.pg_pool.abort.assert_called_once()
    assert stage.active and stage.works == [retained]
    for cleanup in (engine.reset, engine.detach):
        with pytest.raises(RuntimeError, match="active operations"):
            cleanup()
    engine.pg_pool.close.assert_not_called()


def _payload(view, layer_start, page, component):
    result = torch.empty((view.owned_layers, 2, view.row_bytes), dtype=torch.uint8)
    for layer, global_layer in enumerate(view.test_layer_ids):
        result[layer].fill_(
            (11 * global_layer + 13 * page + 70 * (component == "indexer")) % 251
        )
    return result


def _native_prefetch_ack_round_trip(engine, keys, indices, views, start, outcome):
    """Run the native worker/ACK pipeline over real Gloo and in-memory L3."""
    ack_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=20))
    cc = HybridCacheController.__new__(HybridCacheController)
    cc.storage_stop_event = threading.Event()
    cc.prefetch_buffer = Queue()
    cc.prefetch_sync_queue = Queue()
    cc.ack_prefetch_queue = Queue()
    cc.prefetch_hit_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.extra_host_mem_release_queues = {}
    cc.prefetch_completion_sync_groups = [ack_group]
    cc.mem_pool_host = SimpleNamespace(page_size=2, entry_map={})
    cc._page_transfer = engine.prefetch
    previous_controller = engine._controller
    engine._controller = cc
    op = PrefetchOperation(
        CacheRequestHandle(f"native-ack-{outcome}", 0),
        [1] * len(indices),
        pool_transfers=[PoolTransfer(PoolName.INDEXER, indices_from_pool=PoolName.KV)],
    )
    op.hash_value = keys
    op.host_indices = indices
    op.storage_hit_count = len(indices)
    engine.prepare_prefetch(op)
    if outcome == "cancelled" and engine.rank == 0:
        op.mark_terminate()
    engine.storage_backend.fail_get = outcome == "failed_get" and engine.rank == 0
    for component, view in views.items():
        for page in range(len(keys)):
            sentinel = torch.full_like(_payload(view, start, page, component), 0xED)
            view.write_page(int(indices[2 * page]), sentinel)
    workers = [
        threading.Thread(target=cc.prefetch_io_aux_func, daemon=True),
        threading.Thread(target=cc.prefetch_sync_thread_func, daemon=True),
    ]
    try:
        for worker in workers:
            worker.start()
        cc.prefetch_buffer.put(op)
        acks = [cc.ack_prefetch_queue.get(timeout=25) for _ in range(2)]
        copied_pages = (
            0
            if outcome == "failed_get"
            else engine.staging_buffer_config.pages_per_window
            if outcome == "cancelled"
            else len(keys)
        )
        assert acks[0].completed_tokens == copied_pages * 2
        assert acks[0].pool_hits == {pool.value: 0 for pool in PoolName}
        assert acks[1].completed_req is True
        assert acks[1].completed_tokens is None
        assert cc.ack_prefetch_queue.empty()
        assert op.completed_tokens == 0 and not op.pool_transfers_done
        assert not engine.stages[PREFETCH].active
        for component, view in views.items():
            for page in range(len(keys)):
                expected = _payload(view, start, page, component)
                if page >= copied_pages:
                    expected.fill_(0xED)
                actual = torch.empty_like(expected)
                view.read_page(int(indices[2 * page]), actual)
                assert torch.equal(actual, expected)
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.cache_controller = cc
        cache.tree_core = SimpleNamespace(page_size=2)
        cache.host_memory_mode = "cache"
        cache.ongoing_prefetch = (
            {} if outcome == "cancelled" else {op.handle: SimpleNamespace(operation=op)}
        )
        cache._handle_prefetch_result = mock.Mock()
        for ack in acks:
            cc.ack_prefetch_queue.put(ack)
        cache._drain_storage_control_queues_impl(0, 2, 0, 0, {}, False)
        if outcome == "cancelled":
            assert op.completed_tokens == 0
            cache._handle_prefetch_result.assert_not_called()
        else:
            assert op.completed_tokens == copied_pages * 2 and op.pool_transfers_done
            # KV-derived sidecars are part of the KV progress, not extra pool hits.
            assert cache._check_hybrid_prefetch_result(op.handle, op, keys, indices)
            cache._handle_prefetch_result.assert_called_once_with(op)
        released_pages = list(cc.host_mem_release_queue.queue)
        released = torch.cat(released_pages) if released_pages else indices[:0]
        assert torch.equal(released, indices[op.completed_tokens :])
    finally:
        cc.storage_stop_event.set()
        cc.prefetch_buffer.put(None)
        cc.prefetch_sync_queue.put(None)
        for worker in workers:
            worker.join(5)
            assert not worker.is_alive()
        engine._controller = previous_controller
        dist.destroy_process_group(ack_group)


def _worker(
    rank, rendezvous, objects, shards, layers, pages, indexer_layers, native_ack
):
    torch.set_num_threads(1)
    signal_patch = mock.patch("psutil.Process.send_signal")
    signals = signal_patch.start()
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=shards,
        timeout=timedelta(seconds=30),
    )
    engine = None
    try:
        indexer_layers = (
            tuple(range(layers)) if indexer_layers is None else indexer_layers
        )
        views = _views(rank, shards, layers, 2 * pages, indexer_layers)
        component_layers = {"target": tuple(range(layers)), "indexer": indexer_layers}
        counts = {
            c: tuple(
                sum(lo <= layer < hi for layer in layer_ids)
                for lo, hi in (
                    owned_layer_range(r, shards, layers) for r in range(shards)
                )
            )
            for c, layer_ids in component_layers.items()
        }
        storage = _Storage(objects)
        config = _config(shards)
        engine = LayerSplitTransferEngine(
            storage_backend=storage,
            rank=rank,
            staging_buffer_config=config,
            layer_count=layers,
            exchange_ranks=list(range(shards)),
            l2_pools=views,
            pin_memory=False,
            component_layer_counts=counts,
        )
        engine.attach()
        assert len(engine.pg_pool.data_groups) == config.exchange_group_count
        assert len(storage.pools) == 4  # two directions, two components; no key packing
        buffers = {
            lane: {
                c: stage.buffer.components[c].buffer.data_ptr()
                for c in EXCHANGE_COMPONENTS
            }
            for lane, stage in engine.stages.items()
        }
        start, _ = owned_layer_range(rank, shards, layers)
        for component, view in views.items():
            for page in range(pages):
                view.write_page(page * 2, _payload(view, start, page, component))
        indices = torch.arange(pages * 2)
        keys = [f"page-{p}" for p in range(pages)]

        def build(op, hashes):
            return engine.build_plan(
                # Local operation counters may diverge; request/window identity
                # must still pair the same shared-PG job on every rank.
                op_id=op + rank * 1000,
                request_id=f"req-{op}",
                page_hashes=hashes,
            )

        result = _backup(engine, build(1, keys), indices)
        count = torch.tensor([result.written_pages])
        dist.all_reduce(count, dist.ReduceOp.SUM)
        assert count.item() == pages
        assert (
            len(objects[(str(PoolName.INDEXER), keys[0])])
            == len(indexer_layers) * 2 * 3
        )
        dist.barrier()

        # Both directions now use the SAME data group objects. Hold the local
        # preferred lane on opposite ranks, then release both: only readiness
        # agreement, not local arrival order, may choose the communication order.
        selected = []
        original = engine._communicate

        def trace(lane, plan, window, rounds):
            selected.append((lane, window.index))
            return original(lane, plan, window, rounds)

        engine._communicate = trace
        gate = threading.Event()
        if rank % 2 == 0:
            storage.get_gate = gate
        timer = threading.Timer(0.1, gate.set)
        timer.start()

        def write():
            if rank % 2:
                assert gate.wait(10)
            return _backup(engine, build(2, [f"copy-{k}" for k in keys]), indices)

        with ThreadPoolExecutor(max_workers=2) as workers:
            if rank % 2:
                read = workers.submit(
                    _prefetch, engine, build(3, keys), indices + pages * 2
                )
                written = workers.submit(write)
            else:
                written = workers.submit(write)
                read = workers.submit(
                    _prefetch, engine, build(3, keys), indices + pages * 2
                )
            assert read.result(25).published_pages == pages
            written.result(25)
        timer.join()
        storage.get_gate = None
        schedules = [None] * shards
        dist.all_gather_object(schedules, selected)
        assert all(schedule == schedules[0] for schedule in schedules)
        assert {lane for lane, _ in selected} == {PREFETCH, BACKUP}
        for component, view in views.items():
            for page in range(pages):
                actual = torch.empty_like(_payload(view, start, page, component))
                view.read_page((pages + page) * 2, actual)
                assert torch.equal(actual, _payload(view, start, page, component))

        # A tail window, a missing component, and one rank requesting stop all
        # use the same scheduler and data group pool, with no empty-round calls.
        missing = (str(PoolName.INDEXER), keys[2])
        saved = objects[missing] if rank == 0 else None
        if rank == 0:
            del objects[missing]
        dist.barrier()
        short = _prefetch(engine, build(4, keys), indices + pages * 2)
        assert short.truncation_reason == "get"
        assert 0 <= short.published_pages <= 2
        dist.barrier()
        if rank == 0:
            objects[missing] = saved
        dist.barrier()
        stopped = _prefetch(
            engine,
            build(5, keys),
            indices + pages * 2,
            stop_requested=lambda: rank == 0,
        )
        assert stopped.published_pages == 4 and stopped.truncation_reason == "aborted"
        storage.fail_get = rank == 0
        failed = _prefetch(engine, build(6, keys), indices + pages * 2)
        assert failed.published_pages == 0
        assert (
            _prefetch(engine, build(7, keys), indices + pages * 2).published_pages
            == pages
        )
        before = len(selected)
        assert _prefetch(engine, build(8, []), indices).published_pages == 0
        assert _backup(engine, build(9, []), indices).written_pages == 0
        assert len(selected) == before
        # Layout/byte checks run for every shape. Exercise the native ACK
        # lifecycle once, with multiple ranks and a rank owning no indexer layers.
        if native_ack:
            for outcome in ("complete", "failed_get", "cancelled"):
                _native_prefetch_ack_round_trip(
                    engine, keys, indices + pages * 2, views, start, outcome
                )
        # Reset needs no registry/epoch: idle fixed buffers and PGs are reused.
        data_groups = tuple(engine.pg_pool.data_groups)
        engine.reset()
        assert tuple(engine.pg_pool.data_groups) == data_groups
        assert (
            _prefetch(engine, build(10, keys), indices + pages * 2).published_pages
            == pages
        )
        for lane, stage in engine.stages.items():
            assert not stage.active and not stage.works and stage.io is None
            assert {
                c: stage.buffer.components[c].buffer.data_ptr()
                for c in EXCHANGE_COMPONENTS
            } == buffers[lane]
        signals.assert_not_called()
        communication_thread = engine.pg_pool._thread
        engine.detach()
        assert not communication_thread.is_alive()
    finally:
        signal_patch.stop()
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize(
    "shards,layers,pages,indexer_layers,native_ack",
    [
        (1, 3, 9, None, False),
        (2, 5, 9, None, False),
        (3, 2, 5, None, False),
        (2, 5, 7, (0, 3, 4), False),
        (2, 5, 7, (4,), True),
    ],
)
def test_shared_pg_round_trip(
    tmp_path, shards, layers, pages, indexer_layers, native_ack
):
    with mp.Manager() as manager:
        mp.spawn(
            _worker,
            args=(
                f"file://{tmp_path / 'rendezvous'}",
                manager.dict(),
                shards,
                layers,
                pages,
                indexer_layers,
                native_ack,
            ),
            nprocs=shards,
            join=True,
        )
