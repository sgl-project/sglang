"""Regression tests for batched Mooncake load sessions."""

import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

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
        self.linker.load_layer_group = 1
        self.linker.load_threads = 1
        self.linker.load_io_executor = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(self.linker.load_io_executor.shutdown)
        self.linker.layer_done_counter = Mock()
        self.transfers = [
            [
                SimpleNamespace(name=name, keys=[f"{name}-{row}"], host_indices=[row])
                for name in self.linker.pools
            ]
            for row in (4, 0)
        ]

    def test_batched_start_and_ordered_ranges_after_fence(self):
        fence = Mock()

        def start(keys):
            fence.synchronize.assert_called_once_with()
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
        keys = ["kv-4", "kv-0", "index-4", "index-0"]
        self.store.batch_get_session_start.assert_called_once_with(keys)
        self.store.batch_get_session_end.assert_called_once_with(keys)

    def test_failed_start_ends_successful_sessions(self):
        # One key fails while another key in the same pool succeeds.
        self.store.batch_get_session_start.return_value = [0, -1, 0, 0]
        with self.assertLogs(MooncakeDirectLinker.__module__, level="ERROR"):
            self.linker.load_layer_wise(7, self.transfers)

        self.linker.layer_done_counter.complete.assert_not_called()
        self.linker.layer_done_counter.fail.assert_called_once()
        self.assertIn(
            "get session start failed",
            str(self.linker.layer_done_counter.fail.call_args.args[1]),
        )
        self.store.batch_get_into_multi_buffer_ranges.assert_not_called()
        self.store.batch_get_session_end.assert_called_once_with(
            ["kv-4", "kv-0", "index-4", "index-0"]
        )


class TestBufferRegistrationLifecycle(CustomTestCase):
    def setUp(self):
        self.linker = MooncakeDirectLinker.__new__(MooncakeDirectLinker)
        self.store = Mock(register_buffer=Mock(return_value=0))
        self.store.unregister_buffer.return_value = 0
        self.linker.storage = SimpleNamespace(store=self.store, close=Mock())
        self.linker._registered_buffers = {}
        self.linker._workers_stopped = False
        self.linker._closed = False
        self.linker.mla_broadcaster = None
        self.linker.stats = {}

    @staticmethod
    def buffer(address, size):
        storage = SimpleNamespace(data_ptr=lambda: address, nbytes=lambda: size)
        return SimpleNamespace(untyped_storage=lambda: storage)

    def test_shared_storage_is_registered_once_and_retained_until_unregister(self):
        buffer = self.buffer(0x1000, 64)
        self.linker.pools = {
            "kv": Mock(get_hybrid_pool_buffer=Mock(return_value=[buffer, buffer])),
            "side": Mock(get_hybrid_pool_buffer=Mock(return_value=[buffer])),
        }

        self.linker.register_buffers()
        self.linker.register_buffers()
        self.store.register_buffer.assert_called_once_with(0x1000, 64)
        self.assertIs(
            self.linker._registered_buffers[(0x1000, 64)], buffer.untyped_storage()
        )

        self.linker.unregister_buffers()
        self.linker.unregister_buffers()
        self.store.unregister_buffer.assert_called_once_with(0x1000)
        self.assertEqual(self.linker._registered_buffers, {})

    def test_registration_failure_rolls_back_prior_successes(self):
        self.linker.pools = {
            "kv": Mock(
                get_hybrid_pool_buffer=Mock(
                    return_value=[self.buffer(0x1000, 64), self.buffer(0x2000, 128)]
                )
            )
        }
        self.store.register_buffer.side_effect = [0, -1]

        with self.assertRaisesRegex(RuntimeError, "Failed to register GPU KV buffer"):
            self.linker.register_buffers()

        self.assertEqual(
            self.store.register_buffer.call_args_list,
            [call(0x1000, 64), call(0x2000, 128)],
        )
        self.store.unregister_buffer.assert_called_once_with(0x1000)
        self.assertEqual(self.linker._registered_buffers, {})

    def test_close_drains_transfers_and_stops_workers_before_unregister(self):
        self.linker._registered_buffers[(0x1000, 64)] = object()
        self.linker.pending_loads = {}
        self.linker.broadcast_loads = {}
        self.linker.offload_results = Queue()
        self.linker.layer_done_counter = Mock()
        self.linker.load_queue = Queue()
        self.linker.offload_queue = Queue()
        self.linker.load_io_executor = Mock()
        started = threading.Event()
        release = threading.Event()
        unregistered = threading.Event()

        def worker(queue):
            while True:
                task = queue.get()
                try:
                    if task is None:
                        return
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("Transfer was not released")
                finally:
                    queue.task_done()

        self.linker.load_thread = threading.Thread(
            target=worker, args=(self.linker.load_queue,)
        )
        self.linker.offload_thread = threading.Thread(
            target=worker, args=(self.linker.offload_queue,)
        )
        self.linker.load_thread.start()
        self.linker.offload_thread.start()
        self.linker.offload_queue.put("in-flight")

        def unregister(address):
            self.assertFalse(self.linker.load_thread.is_alive())
            self.assertFalse(self.linker.offload_thread.is_alive())
            self.linker.load_io_executor.shutdown.assert_called_once_with()
            unregistered.set()
            return 0

        self.store.unregister_buffer.side_effect = unregister
        with ThreadPoolExecutor(1) as closer:
            self.assertTrue(started.wait(5))
            closing = closer.submit(self.linker.close)
            try:
                self.assertFalse(unregistered.wait(0.1))
                self.assertFalse(closing.done())
            finally:
                release.set()
            closing.result(timeout=5)

        self.assertTrue(unregistered.is_set())
        self.linker.storage.close.assert_called_once_with()
        self.assertTrue(self.linker._closed)

    def test_failed_unregister_retains_allocation_and_close_can_retry(self):
        first, second = object(), object()
        self.linker._registered_buffers = {(0x1000, 64): first, (0x2000, 128): second}
        self.linker.reset = Mock()
        self.linker.load_queue = Mock()
        self.linker.offload_queue = Mock()
        self.linker.load_thread = Mock()
        self.linker.offload_thread = Mock()
        self.linker.load_io_executor = Mock()
        self.store.unregister_buffer.side_effect = [0, -1, 0]

        with self.assertRaisesRegex(RuntimeError, "Failed to unregister GPU KV buffer"):
            self.linker.close()
        self.assertEqual(self.linker._registered_buffers, {(0x2000, 128): second})
        self.linker.storage.close.assert_not_called()
        self.assertFalse(self.linker._closed)

        self.linker.close()
        self.linker.close()
        self.assertEqual(self.linker._registered_buffers, {})
        self.assertEqual(
            self.store.unregister_buffer.call_args_list,
            [call(0x1000), call(0x2000), call(0x2000)],
        )
        self.linker.reset.assert_called_once_with()
        self.linker.load_queue.put.assert_called_once_with(None)
        self.linker.offload_queue.put.assert_called_once_with(None)
        self.linker.load_thread.join.assert_called_once_with()
        self.linker.offload_thread.join.assert_called_once_with()
        self.linker.load_io_executor.shutdown.assert_called_once_with()
        self.linker.storage.close.assert_called_once_with()


class TestPersistentRegistrationMemorySaver(CustomTestCase):
    def test_guard_uses_effective_config_only_for_dummy_clients(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
            MooncakeStoreConfig,
        )

        # Explicit false must override env=1; embedded registration is unrelated
        # to the Dummy CUDA IPC option, including when the option is true.
        cases = (
            (True, None, "1", True, True),
            (True, True, "0", True, True),
            (True, False, "1", True, False),
            (True, None, "0", True, False),
            (True, True, "1", False, False),
            (False, None, "1", True, False),
            (False, True, "1", True, False),
        )

        class ReachedPoolSetup(Exception):
            pass

        for standalone, configured, env_value, memory_saver, rejected in cases:
            with self.subTest(
                standalone=standalone,
                configured=configured,
                env_value=env_value,
                memory_saver=memory_saver,
            ):
                extra_config = {
                    "client_server_address": "127.0.0.1:50052",
                    "standalone_storage": standalone,
                    "cuda_ipc_persistent_registration": configured,
                }
                params = SimpleNamespace(
                    page_size=256, token_to_kv_pool_allocator=Mock()
                )
                with (
                    patch.dict(
                        os.environ,
                        {"MC_CUDA_IPC_PERSISTENT_REGISTRATION": env_value},
                    ),
                    patch(
                        f"{MooncakeDirectLinker.__module__}.get_exec",
                        return_value=SimpleNamespace(
                            features=SimpleNamespace(enable_memory_saver=memory_saver)
                        ),
                    ),
                    patch(
                        f"{MooncakeDirectLinker.__module__}.resolve_hybrid_device_pool_group",
                        side_effect=ReachedPoolSetup,
                    ) as pool_setup,
                    patch(
                        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.MooncakeStore"
                    ) as store_setup,
                    patch.object(
                        MooncakeStoreConfig, "load", wraps=MooncakeStoreConfig.load
                    ) as load_config,
                ):
                    expected = ValueError if rejected else ReachedPoolSetup
                    with self.assertRaises(expected) as error:
                        MooncakeDirectLinker(
                            None, params, components=(), extra_config=extra_config
                        )
                    if rejected:
                        self.assertIn(
                            "requires a fixed GPU KV pool", str(error.exception)
                        )
                        pool_setup.assert_not_called()
                    else:
                        pool_setup.assert_called_once()
                    store_setup.assert_not_called()
                    load_config.assert_called_once_with(extra_config)


if __name__ == "__main__":
    unittest.main()
