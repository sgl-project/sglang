import tempfile
import time
import unittest
import weakref
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.srt.managers.cache_controller import LayerDoneCounter
from sglang.srt.mem_cache.mla_host_dedup import (
    MLAHostDedupBroadcaster,
    MLAHostDedupContext,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-b", runner_config="2-gpu-large")


def _run_layerwise_rank(rank, init_method, with_indexer, layer_group=1, world=2):
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world,
        init_method=init_method,
        timeout=timedelta(seconds=60),
    )
    group = dist.new_group(
        list(range(world)), backend="nccl", timeout=timedelta(seconds=60)
    )

    class DSAPool(SimpleNamespace):
        pass

    layers = 7 if layer_group > 1 else 3
    capacity = world * 16 + 16
    pool = (DSAPool if with_indexer else SimpleNamespace)(
        layer_num=layers,
        device=device,
        kv_cache_dim=8,
        kv_buffer=[
            torch.zeros(
                (capacity, 1, 8),
                dtype=torch.bfloat16 if with_indexer else torch.float32,
                device=device,
            )
            for _ in range(layers)
        ],
        host_transfer_translate=lambda indices: (
            ((14 - indices // 2 * 2 + indices % 2) if rank % 2 else indices) + rank * 16
        ),
    )
    if with_indexer:
        pool.page_size = 2
        pool.index_k_with_scale_buffer = [
            torch.zeros(
                (0 if layer in (0, 3, 6) else capacity // 2, 2, 4),
                dtype=torch.uint8,
                device=device,
            )
            for layer in range(layers)
        ]
    with (
        patch(
            "sglang.srt.mem_cache.mla_host_dedup.mla_dedup_rank_and_size",
            return_value=(rank, world),
        ),
        patch("sglang.srt.mem_cache.mla_host_dedup.DSATokenToKVPool", DSAPool),
    ):
        broadcaster = MLAHostDedupBroadcaster(
            pool, group, src_global_rank=0, layerwise_group_size=layer_group
        )
    broadcaster._warmup_group()
    logical = torch.arange(16, device=device)
    local = pool.host_transfer_translate(logical)
    prepared = broadcaster.prepare_broadcast(logical, torch.cuda.current_stream())
    broadcaster.broadcast_loaded_layer(0, prepared)
    dist.all_reduce(pool.kv_buffer[0].index_select(0, local).sum())
    torch.cuda.synchronize()

    context = MLAHostDedupContext(broadcaster, None, None, layerwise_overlap=True)
    counter = LayerDoneCounter(
        layers, context.broadcast_ready_layers, context.has_pending_layerwise_load
    )
    pool.layer_transfer_counter = counter
    forward_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    load_stream = torch.cuda.Stream()
    # Keep pinned allocations outside the measured work, matching real host
    # pools that are allocated before restore.
    sources = (
        [
            torch.empty((16, 1, 8), dtype=pool.kv_buffer[0].dtype, pin_memory=True)
            for _ in range(layers)
        ]
        if rank == 0
        else None
    )
    index_sources = (
        [
            torch.empty((8, 2, 4), dtype=torch.uint8, pin_memory=True)
            for _ in range(layers)
        ]
        if with_indexer and rank == 0
        else None
    )
    for round_id in range(25):
        for buffer in pool.kv_buffer:
            buffer.zero_()
        if with_indexer:
            for buffer in pool.index_k_with_scale_buffer:
                buffer.zero_()
        torch.cuda.synchronize()
        if sources is not None:
            for layer, source in enumerate(sources):
                source.copy_(
                    (torch.arange(16) + round_id + layer * 16)[:, None, None].expand(
                        16, 1, 8
                    )
                )
        if index_sources is not None:
            for layer, source in enumerate(index_sources):
                source.copy_(
                    (torch.arange(8) + round_id + layer + 20)[:, None, None].expand(
                        8, 2, 4
                    )
                )
        slot = counter.update_producer()
        counter.set_consumer(slot)
        context.begin_layerwise_load(slot, logical, torch.cuda.Event())
        assert counter.needs_layerwise_restore()
        assert not context.is_layerwise_load_done(slot)
        start = torch.cuda.Event(enable_timing=True)
        first_done = torch.cuda.Event(enable_timing=True)
        last_loaded = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.stream(load_stream):
            load_stream.wait_event(start)
            for layer in range(layers):
                if rank == 0:
                    pool.kv_buffer[layer][:16].copy_(sources[layer], non_blocking=True)
                    if with_indexer and pool.index_k_with_scale_buffer[layer].numel():
                        pool.index_k_with_scale_buffer[layer][:8].copy_(
                            index_sources[layer], non_blocking=True
                        )
                counter.events[slot].complete(layer)
                if layer == layer_group - 1 and 2 <= round_id < 5:
                    # Delay later copies so the test can detect a whole-load wait.
                    torch.cuda._sleep(100_000_000)
            last_loaded.record()
        previous_compute = None
        checks = []
        for layer in range(layers):
            stream = forward_streams[layer % 2]
            with torch.cuda.stream(stream):
                if previous_compute is not None:
                    stream.wait_event(previous_compute)
                counter.wait_until(layer)
                selected = pool.kv_buffer[layer].index_select(0, local)
                total = selected.float().sum()
                # Model TP communication follows dedup on a separate communicator.
                dist.all_reduce(total)
                if layer == 0:
                    first_done.record()
                expected_total = world * 8 * (120 + 16 * (round_id + layer * 16))
                expected = (
                    (logical + round_id + layer * 16)[:, None, None]
                    .expand(16, 1, 8)
                    .to(pool.kv_buffer[layer].dtype)
                )
                checks.append((total, expected_total, selected, expected))
                if with_indexer and pool.index_k_with_scale_buffer[layer].numel():
                    selected_idx = pool.index_k_with_scale_buffer[layer].index_select(
                        0, local[::2] // 2
                    )
                    index_total = selected_idx.sum()
                    dist.all_reduce(index_total)
                    expected_index_total = (
                        world * 8 * (28 + 8 * (round_id + layer + 20))
                    )
                    expected_idx = (
                        (torch.arange(8, device=device) + round_id + layer + 20)[
                            :, None, None
                        ]
                        .expand(8, 2, 4)
                        .to(torch.uint8)
                    )
                    checks.append(
                        (index_total, expected_index_total, selected_idx, expected_idx)
                    )
                previous_compute = torch.cuda.Event()
                previous_compute.record()
        torch.cuda.current_stream().wait_event(previous_compute)
        counter.finish_consumer()
        assert not counter.needs_layerwise_restore()
        torch.cuda.synchronize()
        assert context.is_layerwise_load_done(slot)
        for total, expected_total, actual, expected in checks:
            assert total.item() == expected_total
            assert torch.equal(actual, expected)
        # Warm the real BF16/byte gather, scatter and model collectives before
        # testing overlap; CUDA's first kernel load may synchronize streams.
        if 2 <= round_id < 5:
            first_ms = start.elapsed_time(first_done)
            last_ms = start.elapsed_time(last_loaded)
            assert first_ms < last_ms, (
                f"rank={rank}, round={round_id}: first layer {first_ms:.3f} ms, "
                f"last copy {last_ms:.3f} ms"
            )
        context.release_layerwise_load(slot)
    if with_indexer and layer_group > 1:
        # Two outstanding slots use the same scratch on different streams.
        # Delay A so an unfenced B could overwrite its pending broadcast.
        slots = [counter.update_producer(), counter.update_producer()]
        parts = [logical[:8], logical[8:]]
        for slot, indices in zip(slots, parts):
            for layer in range(layers):
                counter.events[slot].complete(layer)
            context.begin_layerwise_load(slot, indices, torch.cuda.Event())
        snapshots = []
        for part, first in ((0, 0), (1, 0), (0, 4), (1, 4)):
            with torch.cuda.stream(forward_streams[part]):
                if part == 0 and first == 0:
                    torch.cuda._sleep(10_000_000)
                context.broadcast_ready_layers(slots[part], first)
                indices = pool.host_transfer_translate(parts[part])
                for layer in range(first, min(first + 4, layers)):
                    snapshots.append(
                        (
                            part,
                            layer,
                            pool.kv_buffer[layer].index_select(0, indices),
                            None,
                        )
                    )
                    if pool.index_k_with_scale_buffer[layer].numel():
                        snapshots.append(
                            (
                                part,
                                layer,
                                None,
                                pool.index_k_with_scale_buffer[layer].index_select(
                                    0, indices[::2] // 2
                                ),
                            )
                        )
        torch.cuda.synchronize()
        for part, layer, kv, indexer in snapshots:
            if kv is not None:
                expected = (
                    (parts[part] + 24 + layer * 16)[:, None, None]
                    .expand(8, 1, 8)
                    .to(kv.dtype)
                )
                assert torch.equal(kv, expected)
            if indexer is not None:
                expected = (
                    (torch.arange(4, device=device) + part * 4 + 24 + layer + 20)[
                        :, None, None
                    ]
                    .expand(4, 2, 4)
                    .to(torch.uint8)
                )
                assert torch.equal(indexer, expected)
        for slot in slots:
            assert context.is_layerwise_load_done(slot)
            context.release_layerwise_load(slot)
        # The last group changes stream and removes the pending load while its
        # CUDA index tensors are still in use. Allocations on their first stream
        # must not recycle that storage until the delayed last group finishes.
        slot = counter.update_producer()
        for layer in range(layers):
            counter.events[slot].complete(layer)
        context.begin_layerwise_load(slot, logical, torch.cuda.Event())
        with torch.cuda.stream(forward_streams[0]):
            context.broadcast_ready_layers(slot, 0)
            references = [
                weakref.ref(tensor) for tensor in context.pending_loads[slot].prepared
            ]
        tail_snapshots = []
        with torch.cuda.stream(forward_streams[1]):
            torch.cuda._sleep(100_000_000)
            context.broadcast_ready_layers(slot, 4)
            for layer in range(4, layers):
                tail_snapshots.append(
                    (layer, pool.kv_buffer[layer].index_select(0, local), None)
                )
                if pool.index_k_with_scale_buffer[layer].numel():
                    tail_snapshots.append(
                        (
                            layer,
                            None,
                            pool.index_k_with_scale_buffer[layer].index_select(
                                0, local[::2] // 2
                            ),
                        )
                    )
        assert slot not in context.pending_loads
        assert all(reference() is None for reference in references)
        with torch.cuda.stream(forward_streams[0]):
            pressure = [
                torch.zeros((n,), dtype=torch.int64, device=device)
                for _ in range(256)
                for n in (16, 8)
            ]
        torch.cuda.synchronize()
        for layer, kv, indexer in tail_snapshots:
            if kv is not None:
                expected = (
                    (logical + 24 + layer * 16)[:, None, None]
                    .expand(16, 1, 8)
                    .to(kv.dtype)
                )
                assert torch.equal(kv, expected)
            if indexer is not None:
                expected = (
                    (torch.arange(8, device=device) + 24 + layer + 20)[:, None, None]
                    .expand(8, 2, 4)
                    .to(torch.uint8)
                )
                assert torch.equal(indexer, expected)
        del pressure
        assert context.is_layerwise_load_done(slot)
        context.release_layerwise_load(slot)
    context.destroy()
    dist.destroy_process_group()


class TestMLAHostDedupLayerwiseCUDA(CustomTestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "Requires two CUDA GPUs")
    def test_layerwise_restore_interleaves_with_forward_collectives(self):
        for with_indexer, layer_group in ((False, 1), (True, 1), (True, 4)):
            with self.subTest(with_indexer=with_indexer, layer_group=layer_group):
                self._check_layerwise_restore(with_indexer, layer_group)

    def _check_layerwise_restore(self, with_indexer, layer_group):
        with tempfile.TemporaryDirectory() as directory:
            processes = mp.spawn(
                _run_layerwise_rank,
                args=(f"file://{directory}/group", with_indexer, layer_group),
                nprocs=2,
                join=False,
            )
            try:
                deadline = time.monotonic() + 90
                while not processes.join(timeout=max(0, deadline - time.monotonic())):
                    if time.monotonic() >= deadline:
                        self.fail(
                            "Layer-wise TP2 test did not finish within 90 seconds"
                        )
            finally:
                for process in processes.processes:
                    if process.is_alive():
                        process.terminate()
                    process.join(timeout=10)


if __name__ == "__main__":
    unittest.main()
