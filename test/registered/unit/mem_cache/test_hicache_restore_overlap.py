from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers.cache_controller import HiCacheController, LayerDoneCounter
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _make_dma_controller(
    forward: torch.cuda.Stream, backward_overlap: bool, allocator_device: str = "cuda"
) -> HiCacheController:
    device = SimpleNamespace(data=torch.zeros((2, 9), device="cuda"))
    host_data = torch.full((2, 9), 33.0, pin_memory=True)

    def load_layer(pool, host_indices, device_indices, layer, backend, is_draft=False):
        start = int(device_indices[0])
        pool.data[layer, start : start + len(device_indices)].copy_(
            host_data[layer, : len(host_indices)], non_blocking=True
        )

    cache = object.__new__(HiCacheController)
    cache.io_backend = "direct"
    cache.device = "cuda"
    cache.mem_pool_host = SimpleNamespace(
        layout="page_first_direct",
        size_per_token=8,
        layer_num=2,
        prepare_transfer_indices=lambda host, device, _backend: (host, device),
        load_to_device_per_layer_physical=load_layer,
    )
    cache.mem_pool_device = device
    cache.mem_pool_device_allocator = TokenToKVPoolAllocator(
        8, torch.float32, allocator_device, None, False
    )
    # Use ordinary pinned DMA to isolate controller/engine stream ordering.
    cache.l2_transfer_engine = L2TransferEngine("direct")
    cache.layer_done_counter = LayerDoneCounter(2)
    cache.layer_done_counter.init_graph_events()
    cache.transfer_layer_id_max = 2
    cache.load_queue = []
    cache.ack_load_queue = []
    cache.set_load_fence_stream(forward, restore_backward_overlap=backward_overlap)
    allocator = cache.mem_pool_device_allocator
    assert not allocator.needs_restore_fence
    # Initialize the submission path before checking that its fence is asynchronous.
    allocator.free(allocator.alloc(8))
    cache.load(torch.arange(2))
    cache.start_loading()
    cache.ack_load_queue.pop().finish_event.synchronize()
    allocator.clear()
    torch.cuda._sleep(1)
    device.data.fill_(7)
    return cache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("backward_overlap", [True, False])
@pytest.mark.parametrize("reset_at_launch", [False, True])
@pytest.mark.parametrize("recycle_destination", [False, True])
def test_controller_dma_fences_recycled_pages_without_blocking_cpu(
    backward_overlap, reset_at_launch, recycle_destination
):
    forward = torch.cuda.Stream()
    cache = _make_dma_controller(forward, backward_overlap)
    device = cache.mem_pool_device
    allocator = cache.mem_pool_device_allocator
    previous_output = torch.empty((2, 2), device="cuda")
    previous_output.copy_(device.data[:, 1:3])
    previous_pages = allocator.alloc(8 if recycle_destination else 2)
    torch.cuda.synchronize()

    # Registration also protects startup work before the scheduler's first launch.
    if reset_at_launch:
        cache.begin_forward()
    with torch.cuda.stream(forward):
        torch.cuda._sleep(400_000_000)
        previous_output.copy_(device.data[:, 1:3])
        previous_done = forward.record_event()
    if recycle_destination:
        # Logical eviction can precede the old forward's last physical access.
        allocator.free(previous_pages)
    destination = cache.load(torch.arange(2))
    assert allocator.needs_restore_fence is (backward_overlap and recycle_destination)
    cache.start_loading()
    assert not previous_done.query(), "Restore submission blocked the CPU on forward"
    cache.ack_load_queue[0].finish_event.synchronize()
    overlaps = backward_overlap and not recycle_destination
    assert previous_done.query() is (not overlaps)
    assert torch.all(device.data[:, destination].cpu() == 33)
    forward.synchronize()
    assert torch.all(previous_output.cpu() == 7)


@pytest.mark.parametrize("recycled_pool", [None, "full", "swa", "mamba", "all"])
def test_merged_restore_fences_once_after_allocation(
    monkeypatch: pytest.MonkeyPatch, recycled_pool: str | None
):
    from sglang.srt.managers import cache_controller
    from sglang.srt.mem_cache.allocator.mamba import MambaSlotAllocator
    from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from sglang.srt.mem_cache.pool_host import PoolEntry

    allocators = {
        "full": TokenToKVPoolAllocator(8, torch.float32, "cpu", None, False),
        "swa": TokenToKVPoolAllocator(8, torch.float32, "cpu", None, False),
        "mamba": MambaSlotAllocator(8, "cpu"),
    }
    entries = [
        PoolEntry(
            name,
            SimpleNamespace(size_per_token=8),
            object(),
            layer_mapper=lambda layer: layer,
            device_alloc_fn=allocator.alloc,
            device_free_fn=allocator.free,
        )
        for name, allocator in zip(
            (PoolName.KV, PoolName.SWA, PoolName.MAMBA),
            allocators.values(),
            strict=True,
        )
    ]
    cache = object.__new__(HybridCacheController)
    cache.mem_pool_device_allocator = allocators["full"]
    cache.mem_pool_host = SimpleNamespace(
        layout="page_first_direct",
        anchor_entry=entries[0],
        entry_map={e.name: e for e in entries},
    )
    cache.io_backend = "direct"
    cache.load_queue = []
    cache.ack_load_queue = []
    cache.layer_done_counter = SimpleNamespace(
        update_producer=Mock(return_value=0),
        events=[Mock()],
        block_ends=(1, 2),
        graph_enabled=True,
    )
    cache.transfer_layer_id_max = 2
    cache.load_fence_stream = Mock()
    cache.restore_backward_overlap = True
    cache.restore_allocators = list(allocators.values())
    cache.l2_transfer_engine = engine = Mock()
    monkeypatch.setattr(cache_controller, "check_cuda_graph_backend", lambda *_: True)
    cache.begin_forward()
    assert cache.start_loading() == -1
    for name, allocator in allocators.items():
        if recycled_pool in (name, "all"):
            allocator.free(allocator.alloc(8))
    for node_id in range(3):
        assert (
            cache.load(
                torch.tensor([node_id]),
                node_id=node_id,
                extra_pools=[
                    PoolTransfer(name=name, host_indices=torch.tensor([node_id]))
                    for name in (PoolName.SWA, PoolName.MAMBA)
                ],
            )
            is not None
        )
    # Allocation and queueing only update native ownership; submission owns the fence.
    assert not engine.mock_calls
    assert cache.start_loading() == 0
    expected = ["submit_host_to_device"]
    if recycled_pool is not None:
        expected.insert(0, "host_to_device_stream.wait_stream")
        engine.host_to_device_stream.wait_stream.assert_called_once_with(
            cache.load_fence_stream
        )
    assert [call[0] for call in engine.mock_calls] == expected
    transfers = engine.submit_host_to_device.call_args.args[0]
    assert len(transfers) == 3
    assert all(len(transfer.device_indices) == 3 for transfer in transfers)
    assert cache.ack_load_queue[0].node_ids == [0, 1, 2]
    assert cache.start_loading() == -1
    assert [call[0] for call in engine.mock_calls] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_next_prefill_war_barrier_orders_older_frees_before_restore():
    forward, scheduler = torch.cuda.Stream(), torch.cuda.Stream()
    # CPU indices avoid a D2H sync that could hide a missing DMA start-event wait.
    cache = _make_dma_controller(forward, True, allocator_device="cpu")
    allocator = cache.mem_pool_device_allocator
    device = cache.mem_pool_device
    old_pages = allocator.alloc(8)
    old_output = torch.empty((2, 2), device="cuda")
    old_output.copy_(device.data[:, 1:3])
    torch.cuda.synchronize()

    cache.begin_forward()
    with torch.cuda.stream(forward):
        torch.cuda._sleep(400_000_000)
        old_output.copy_(device.data[:, 1:3])
        first_done = forward.record_event()
    allocator.free(old_pages)

    cache.begin_forward()
    with torch.cuda.stream(forward):
        # B's early metadata fence follows all of A, but precedes B's compute.
        shared_reads_done = forward.record_event()
        torch.cuda._sleep(800_000_000)
        second_done = forward.record_event()
    with torch.cuda.stream(scheduler):
        scheduler.wait_event(shared_reads_done)
        destination = cache.load(torch.arange(2))
        assert not allocator.needs_restore_fence
        cache.start_loading()
    assert not first_done.query(), "Restore submission waited for the older forward"
    cache.ack_load_queue[0].finish_event.synchronize()
    assert first_done.query(), "DMA bypassed the scheduler's WAR dependency"
    assert not second_done.query(), "DMA waited for the entire previous prefill"
    assert torch.all(old_output.cpu() == 7)
    assert torch.all(device.data[:, destination].cpu() == 33)
    forward.synchronize()


@pytest.mark.parametrize(
    "forward_mode",
    [
        ForwardMode.EXTEND,
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.MIXED,
        ForwardMode.IDLE,
    ],
)
def test_scheduler_resets_restore_tracking_at_every_forward_launch(forward_mode):
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.runtime_context import get_context

    controller = object.__new__(HiCacheController)
    allocator = TokenToKVPoolAllocator(8, torch.float32, "cpu", None, False)
    controller.restore_allocators = [allocator]
    controller.begin_forward()
    controller.begin_forward = Mock(wraps=controller.begin_forward)
    pages = allocator.alloc(8)
    allocator.free(pages)
    pages = allocator.alloc(8)
    assert allocator.needs_restore_fence
    scheduler = object.__new__(Scheduler)
    scheduler.tree_cache = SimpleNamespace(cache_controller=controller)
    scheduler.gracefully_exit = scheduler._engine_paused = False
    scheduler.last_batch = None
    scheduler.running_batch = Mock()
    scheduler.is_generation = False
    scheduler.ingest_requests = Mock()
    scheduler.is_disable_overlap_for_batch = Mock(return_value=False)
    scheduler._apply_war_barrier = Mock()
    scheduler._begin_hicache_events = Mock(return_value=False)
    scheduler.process_batch_result = Mock()
    scheduler.invariant_checker = Mock()
    previous = Mock(forward_mode=forward_mode)
    following = Mock(forward_mode=ForwardMode.EXTEND)
    batches = iter([previous, following])

    def next_batch(**kwargs):
        batch = next(batches)
        # Selection submits restores while the previous allocation interval is intact.
        assert allocator.needs_restore_fence
        return SimpleNamespace(
            batch_to_run=batch, running_batch=scheduler.running_batch
        )

    def run_batch(batch):
        nonlocal pages
        assert not allocator.needs_restore_fence
        allocator.free(pages)
        pages = allocator.alloc(8)
        assert allocator.needs_restore_fence
        scheduler.gracefully_exit = batch is following
        return Mock()

    scheduler.get_next_batch_to_run = next_batch
    scheduler.run_batch = run_batch
    with get_context().override_server_args(hicache_restore_backward_overlap=True):
        scheduler.event_loop_overlap()
    assert controller.begin_forward.call_count == 2
    controller.begin_forward.assert_called_with()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("delayed_dma", [False, True])
def test_early_dma_preserves_old_graph_readiness_and_next_replay(delayed_dma):
    counter = LayerDoneCounter(2)
    counter.init_graph_events()
    forward = torch.cuda.Stream()
    dma = torch.cuda.Stream()
    previous_input = torch.full((2,), 7.0, device="cuda")
    restored = torch.zeros(2, device="cuda")
    previous_output = torch.empty_like(previous_input)
    next_output = torch.empty_like(previous_input)
    host = torch.full((2,), 33.0, pin_memory=True)
    previous_graph = torch.cuda.CUDAGraph()
    next_graph = torch.cuda.CUDAGraph()
    torch.cuda._sleep(1)
    torch.cuda.synchronize()

    with torch.cuda.graph(previous_graph, stream=forward):
        counter.wait_until(0)
        previous_output[0].copy_(previous_input[0])
        torch.cuda._sleep(400_000_000)
        counter.wait_until(1)
        previous_output[1].copy_(previous_input[1])
    with torch.cuda.graph(next_graph, stream=forward):
        counter.wait_until(0)
        next_output[0].copy_(restored[0])
        counter.wait_until(1)
        next_output[1].copy_(restored[1])
    torch.cuda.synchronize()

    # Reuse the captured event handles across successive generations.
    for _ in range(6):
        with torch.cuda.stream(forward):
            if delayed_dma:
                torch.cuda._sleep(100_000_000)
            previous_graph.replay()
            previous_done = forward.record_event()
        producer = counter.events[counter.update_producer()]
        with torch.cuda.stream(dma):
            for layer in range(2):
                if delayed_dma and layer == 1:
                    torch.cuda._sleep(800_000_000)
                restored[layer : layer + 1].copy_(
                    host[layer : layer + 1], non_blocking=True
                )
                producer.complete(layer)
            producer.finish_event = dma.record_event()
        with torch.cuda.stream(forward):
            next_graph.replay()
        if delayed_dma:
            previous_done.synchronize()
            assert not producer.finish_event.query(), (
                "Old replay waited for the next restore"
            )
        else:
            dma.synchronize()
            assert not previous_done.query(), "DMA was serialized behind the old graph"
        forward.synchronize()
        assert previous_output.cpu().tolist() == [7, 7]
        assert next_output.cpu().tolist() == [33, 33]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
@pytest.mark.parametrize("has_swa", [False, True])
@pytest.mark.parametrize("backward_overlap", [False, True])
def test_controller_tracks_native_full_and_swa_allocators(has_swa, backward_overlap):
    from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator

    full = TokenToKVPoolAllocator(8, torch.float32, "cuda", None, False)
    swa = TokenToKVPoolAllocator(8, torch.float32, "cuda", None, True)
    allocator = full
    if has_swa:
        allocator = object.__new__(SWATokenToKVPoolAllocator)
        allocator.full_attn_allocator = full
        allocator.swa_attn_allocator = swa
    controller = object.__new__(HybridCacheController)
    controller.io_backend = "direct"
    controller.mem_pool_host = None
    controller.mem_pool_device_allocator = allocator
    controller.set_load_fence_stream(torch.cuda.Stream(), backward_overlap)
    assert not full.needs_restore_fence
    assert not swa.needs_restore_fence
    controller.begin_forward()
    for pool, tracked in (
        (full, backward_overlap),
        (swa, has_swa and backward_overlap),
    ):
        released = pool.alloc(2)
        pool.free(released)
        assert not pool.needs_restore_fence
        assert pool.alloc(8) is not None
        assert pool.needs_restore_fence is tracked
    controller.begin_forward()
    assert not full.needs_restore_fence
    assert not swa.needs_restore_fence
