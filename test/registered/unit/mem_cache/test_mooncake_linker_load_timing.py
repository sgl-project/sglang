"""The Mooncake linker logs each request's load time and bytes on the timing rank."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.storage.mooncake_store import mooncake_direct_linker
from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LOGGER = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker"
# Bytes one key moves per layer, by pool.
KEY_BYTES = {PoolName.KV: 100, PoolName.SWA: 10}


class _Pool:
    def __init__(self, name):
        self.name = name

    def prepare_locations(self, host_indices):
        return list(host_indices)

    def get_prepared_layer_range_meta(self, locations, layer):
        size = KEY_BYTES[self.name]
        return [[0]] * len(locations), [[size]] * len(locations), [[0]] * len(locations)


def _transfer(name, pages):
    return SimpleNamespace(
        name=name, keys=[f"{name}-{i}" for i in range(pages)], host_indices=range(pages)
    )


def _linker(log_load_timing):
    linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
    linker.log_load_timing = log_load_timing
    linker.num_layers = 2
    linker.pools = {name: _Pool(name) for name in KEY_BYTES}
    linker.layer_done_counter = MagicMock()
    store = MagicMock()
    store.batch_get_session_start.side_effect = lambda keys: [0] * len(keys)
    store.batch_get_into_multi_buffer_ranges.side_effect = (
        lambda keys, ptrs, sizes, offsets: [sum(item) for item in sizes]
    )
    linker.storage = SimpleNamespace(
        store=store,
        _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
        _tag_keys=lambda keys: keys,
    )
    return linker


class TestMooncakeLinkerLoadTiming(CustomTestCase):
    def test_each_request_logs_its_own_bytes_after_all_layers_land(self):
        linker = _linker(log_load_timing=True)
        requests = [
            [_transfer(PoolName.KV, 2), _transfer(PoolName.SWA, 1)],
            [_transfer(PoolName.KV, 3)],
        ]
        with self.assertLogs(LOGGER, level="INFO") as logs:
            linker.load_layer_wise(7, requests, [("a", 0.0), ("b", None)])
        lines = [line for line in logs.output if "linker load" in line]
        self.assertEqual(len(lines), 2)
        # Two layers: (2 x 100 + 1 x 10) x 2 and 3 x 100 x 2.
        self.assertIn("rid=a pages=2 bytes=420 ", lines[0])
        self.assertIn("rid=b pages=3 bytes=600 ", lines[1])
        self.assertIn("batch_reqs=2", lines[0])
        self.assertIn("wait=0ms", lines[1])  # Unknown queue time counts from start.
        linker.layer_done_counter.complete.assert_any_call(7, 1)
        linker.layer_done_counter.fail.assert_not_called()

    def test_disabled_rank_and_failed_loads_log_nothing(self):
        for log_load_timing, fail in [(False, False), (True, True)]:
            with self.subTest(log_load_timing=log_load_timing, fail=fail):
                linker = _linker(log_load_timing)
                if fail:
                    linker.storage.store.batch_get_into_multi_buffer_ranges.side_effect = (
                        lambda keys, ptrs, sizes, offsets: [0] * len(keys)
                    )
                with patch.object(mooncake_direct_linker.logger, "info") as info:
                    with patch.object(mooncake_direct_linker.logger, "exception"):
                        linker.load_layer_wise(
                            1, [[_transfer(PoolName.KV, 1)]], [("a", 0.0)]
                        )
                info.assert_not_called()
                self.assertEqual(linker.layer_done_counter.fail.called, fail)

    def test_queue_time_follows_the_request_into_the_load_thread(self):
        linker = _linker(log_load_timing=True)
        linker.pool_group = SimpleNamespace(
            resolve_transfers=lambda transfers, **kwargs: transfers
        )
        linker.pending_loads = {}
        linker.load_queued_at = {}
        linker.mla_broadcaster = None
        linker.gc_frozen = True
        linker.load_queue = queue.Queue()
        linker.broadcast_loads = {}
        linker.stats = {"load": 0}
        linker.layer_done_counter.update_producer.return_value = 3
        self.assertTrue(linker.load("kept", [_transfer(PoolName.KV, 1)]))
        self.assertTrue(linker.load("dropped", [_transfer(PoolName.KV, 1)]))
        self.assertTrue(linker.cancel_queued_load("dropped"))
        with patch.object(mooncake_direct_linker, "device_module", MagicMock()):
            self.assertEqual(linker.start_layer_wise_loading(), 3)
        index, pending, _, queued_at = linker.load_queue.get_nowait()
        self.assertEqual(
            (index, list(pending), list(queued_at)), (3, ["kept"], ["kept"])
        )
        self.assertEqual(linker.load_queued_at, {})


if __name__ == "__main__":
    unittest.main()
