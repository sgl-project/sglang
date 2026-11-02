"""Page-wise reads use the ordinary session/range API and publish atomically."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import LayerWiseLoadCounter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _linker(enabled=True, threshold=1):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.page_wise_load = enabled
    linker.page_wise_load_threshold = threshold
    linker.num_layers = 2
    events = []
    store = Mock()
    store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
    store.batch_get_session_end.side_effect = lambda keys: events.append(("end", keys))

    def read(keys, ptrs, sizes, offsets):
        events.append(("read", keys))
        return [sum(row) for row in sizes]

    store.batch_get_into_multi_buffer_ranges.side_effect = read
    linker.storage = SimpleNamespace(
        store=store,
        _tag_keys=lambda keys: keys,
        _get_hybrid_page_component_keys=lambda keys, transfer: (
            [f"{transfer.name.value}:{key}" for key in keys],
            1,
        ),
    )
    linker.pools = {
        PoolName.KV: SimpleNamespace(
            prepare_locations=lambda indices: indices.tolist(),
            get_prepared_layer_range_meta=lambda locations, layer: (
                [[100 + i * 100 + layer * 4] for i in locations],
                [[4 if layer == 0 else 8] for _ in locations],
                [[layer * 4] for _ in locations],
            ),
        ),
        PoolName.INDEXER: SimpleNamespace(
            prepare_locations=lambda indices: indices.tolist(),
            get_prepared_layer_range_meta=lambda locations, layer: (
                ([[300, 304]], [[2, 2]], [[0, 2]]) if layer == 1 else None
            ),
        ),
    }
    linker.layer_done_counter = LayerWiseLoadCounter(2)
    index = linker.layer_done_counter.update_producer()
    complete = linker.layer_done_counter.complete

    def publish(index, layer):
        events.append(("ready", layer))
        complete(index, layer)

    linker.layer_done_counter.complete = publish
    transfers = [
        [
            PoolTransfer(
                name=PoolName.KV, keys=["a", "b"], host_indices=torch.tensor([0, 1])
            ),
            PoolTransfer(
                name=PoolName.INDEXER, keys=["i"], host_indices=torch.tensor([0])
            ),
        ]
    ]
    return linker, store, index, transfers, events


class TestMooncakePageWise(unittest.TestCase):
    def test_packed_and_sparse_pools_share_one_read(self):
        linker, store, index, transfers, events = _linker()
        linker.load_layer_wise(index, transfers)

        store.batch_get_into_multi_buffer_ranges.assert_called_once_with(
            ["kv:a", "kv:b", "indexer:i"],
            [[100, 104], [200, 204], [300, 304]],
            [[4, 8], [4, 8], [2, 2]],
            [[0, 4], [0, 4], [0, 2]],
        )
        self.assertEqual(
            [event[0] for event in events], ["read", "end", "end", "ready", "ready"]
        )
        self.assertTrue(linker.layer_done_counter.finish(index))

    def test_disabled_and_below_threshold_keep_layer_reads(self):
        for enabled, threshold in ((False, 1), (True, 4)):
            with self.subTest(enabled=enabled, threshold=threshold):
                linker, store, index, transfers, _ = _linker(enabled, threshold)
                linker.load_layer_wise(index, transfers)
                self.assertEqual(store.batch_get_into_multi_buffer_ranges.call_count, 3)
                self.assertTrue(linker.layer_done_counter.finish(index))

    def test_short_error_or_partial_read_publishes_no_layer(self):
        for result in (None, -707, [12], [12, -707, 4]):
            with self.subTest(result=result):
                linker, store, index, transfers, events = _linker()
                store.batch_get_into_multi_buffer_ranges.side_effect = None
                store.batch_get_into_multi_buffer_ranges.return_value = result
                linker.load_layer_wise(index, transfers)
                self.assertFalse(linker.layer_done_counter.finish(index))
                self.assertNotIn("ready", [event[0] for event in events])
                self.assertEqual(store.batch_get_session_end.call_count, 2)

    def test_range_exception_and_cleanup_exception_publish_failure(self):
        for method in ("batch_get_into_multi_buffer_ranges", "batch_get_session_end"):
            with self.subTest(method=method):
                linker, store, index, transfers, events = _linker()
                getattr(store, method).side_effect = RuntimeError("injected failure")
                linker.load_layer_wise(index, transfers)
                self.assertFalse(linker.layer_done_counter.finish(index))
                self.assertNotIn("ready", [event[0] for event in events])
                self.assertEqual(store.batch_get_session_end.call_count, 2)

    def test_partial_session_start_is_cleaned_up(self):
        linker, store, index, transfers, events = _linker()
        store.batch_get_session_start.return_value = [0, -707]
        store.batch_get_session_start.side_effect = None
        linker.load_layer_wise(index, transfers)
        self.assertFalse(linker.layer_done_counter.finish(index))
        store.batch_get_session_end.assert_called_once_with(["kv:a", "kv:b"])
        store.batch_get_into_multi_buffer_ranges.assert_not_called()

    def test_invalid_range_count_fails_before_read(self):
        linker, store, index, transfers, events = _linker()
        linker.pools[PoolName.KV].get_prepared_layer_range_meta = (
            lambda locations, layer: ([[100]], [[4]], [[0]])
        )
        linker.load_layer_wise(index, transfers)
        self.assertFalse(linker.layer_done_counter.finish(index))
        store.batch_get_into_multi_buffer_ranges.assert_not_called()
        self.assertEqual(store.batch_get_session_end.call_count, 2)


if __name__ == "__main__":
    unittest.main()
