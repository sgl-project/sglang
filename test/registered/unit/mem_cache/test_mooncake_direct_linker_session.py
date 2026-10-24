"""Regression tests for concurrent Mooncake load sessions."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker import (
    MooncakeDirectLinker,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLayerWiseLoadSessions(CustomTestCase):
    def setUp(self):
        self.linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.store = Mock()
        self.linker.storage = SimpleNamespace(
            store=self.store,
            _get_hybrid_page_component_keys=lambda keys, transfer: (keys, None),
            _tag_keys=lambda keys: keys,
        )
        self.linker.pools = {
            name: Mock(get_prepared_layer_range_meta=Mock(return_value=None))
            for name in ("kv", "index")
        }
        self.linker.num_layers = 1
        self.linker.layer_done_counter = Mock()
        self.linker.session_executor = ThreadPoolExecutor(max_workers=2)
        self.addCleanup(self.linker.session_executor.shutdown)
        self.transfers = [
            [
                SimpleNamespace(name=name, keys=[f"{name}-{row}"], host_indices=[row])
                for name in self.linker.pools
            ]
            for row in (4, 0)
        ]

    def test_concurrent_starts_before_fence_and_ordered_ranges_after(self):
        both_started = threading.Barrier(2, timeout=5)
        fence = Mock()

        def start(keys):
            fence.synchronize.assert_not_called()
            both_started.wait()
            return [0] * len(keys)

        def locations(indices):
            fence.synchronize.assert_called_once_with()
            return indices

        self.store.batch_get_session_start.side_effect = start
        self.store.batch_get_into_multi_buffer_ranges.return_value = [3, 3]
        for pool in self.linker.pools.values():
            pool.prepare_locations.side_effect = locations
            pool.get_prepared_layer_range_meta.side_effect = lambda rows, layer: (
                [[100 + int(row) * 3] for row in rows],
                [[3], [3]],
                [[0], [0]],
            )
        self.linker.load_layer_wise(7, self.transfers, fence)

        self.linker.layer_done_counter.fail.assert_not_called()
        self.linker.layer_done_counter.complete.assert_called_once_with(7, 0)
        self.assertEqual(
            self.store.batch_get_into_multi_buffer_ranges.call_args_list,
            [
                call([f"{name}-4", f"{name}-0"], [[112], [100]], [[3], [3]], [[0], [0]])
                for name in self.linker.pools
            ],
        )
        self.store.batch_get_session_end.assert_has_calls(
            [call([f"{name}-4", f"{name}-0"]) for name in self.linker.pools],
            any_order=True,
        )

    def test_failed_start_ends_successful_sessions(self):
        self.store.batch_get_session_start.side_effect = lambda keys: (
            [1, 1] if keys[0].startswith("index") else [0, 0]
        )
        with self.assertLogs(MooncakeDirectLinker.__module__, level="ERROR"):
            self.linker.load_layer_wise(7, self.transfers)

        self.linker.layer_done_counter.complete.assert_not_called()
        self.linker.layer_done_counter.fail.assert_called_once()
        self.assertIn(
            "get session start failed",
            str(self.linker.layer_done_counter.fail.call_args.args[1]),
        )
        self.store.batch_get_into_multi_buffer_ranges.assert_not_called()
        self.store.batch_get_session_end.assert_called_once_with(["kv-4", "kv-0"])


if __name__ == "__main__":
    unittest.main()
