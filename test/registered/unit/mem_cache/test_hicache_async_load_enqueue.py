"""Unit tests for the async HiCache load enqueue.

With SGLANG_ENABLE_HICACHE_ASYNC_LOAD_ENQUEUE, start_loading hands the
host-to-device submission to a loader thread. There is no device here, so the
device module of the controller and of the transfer engine is replaced by a
fake that logs every stream and event call with the calling thread.
"""

import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers import cache_controller
from sglang.srt.managers.cache_controller import (
    CacheOperation,
    HiCacheController,
    LayerDoneCounter,
)
from sglang.srt.mem_cache import l2_transfer
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.l2_transfer import L2Transfer, L2TransferEngine
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LOADER = "hicache-load-enqueue"
TIMEOUT = 10.0


def _thread() -> str:
    return threading.current_thread().name


class FakeEvent:
    def __init__(self, log, enable_timing=False):
        self._log = log
        self.recorded_on = None
        self.recorded_stream = None

    def record(self, stream=None):
        self.recorded_on = _thread()
        self.recorded_stream = stream
        self._log.append(("record", self, _thread()))

    def wait(self, stream=None):
        self._log.append(("event_wait", self, stream, _thread()))

    def query(self):
        return True

    def synchronize(self):
        pass


class FakeStream:
    def __init__(self, name, log):
        self.name = name
        self._log = log

    def wait_event(self, event):
        self._log.append(("stream_wait_event", self, event, _thread()))

    def wait_stream(self, stream):
        self._log.append(("stream_wait_stream", self, stream, _thread()))

    def synchronize(self):
        pass

    def __repr__(self):
        return f"<FakeStream {self.name}>"


class _StreamContext:
    def __init__(self, module, stream):
        self._module = module
        self._stream = stream

    def __enter__(self):
        self._module.log.append(("stream_enter", self._stream, _thread()))
        self._previous = self._module.current_stream()
        self._module.local.stream = self._stream
        return self._stream

    def __exit__(self, *exc):
        self._module.local.stream = self._previous
        self._module.log.append(("stream_exit", self._stream, _thread()))
        return False


class _CurrentStream(threading.local):
    stream = None


class FakeDeviceModule:
    """The part of torch.cuda the load path uses, with an ordered call log."""

    __name__ = "fake_device_module"

    def __init__(self):
        self.log = []
        self.set_device_calls = []
        self.device_index = 7
        self.local = _CurrentStream()
        self.default_stream = FakeStream("default", self.log)
        self._stream_count = 0

    def Event(self, enable_timing=False):
        return FakeEvent(self.log, enable_timing)

    def Stream(self):
        self._stream_count += 1
        return FakeStream(f"stream{self._stream_count}", self.log)

    def stream(self, stream):
        return _StreamContext(self, stream)

    def current_stream(self):
        return self.local.stream or self.default_stream

    def current_device(self):
        return self.device_index

    def set_device(self, index):
        self.set_device_calls.append((index, _thread()))


class FakeHostPool:
    """Records each per-layer submission; can block or fail mid-burst."""

    layout = "page_first_direct"
    size_per_token = 16

    def __init__(self, layer_num, log, name="kv"):
        self.layer_num = layer_num
        self.name = name
        self._log = log
        self.gate = None
        self.fail_at_layer = None
        self.failure = RuntimeError("injected per-layer load failure")

    def prepare_transfer_indices(self, host_indices, device_indices, io_backend):
        return host_indices, device_indices

    def load_to_device_per_layer_physical(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
        is_draft=False,
    ):
        if self.gate is not None:
            self.gate.wait()
        if self.fail_at_layer == layer_id:
            raise self.failure
        self._log.append(("load_layer", layer_id, _thread(), self.name))


class FakeHostPoolGroup:
    """An anchor KV pool plus one sidecar pool."""

    layout = "page_first_direct"

    def __init__(self, layer_num, log):
        self.anchor = FakeHostPool(layer_num, log, name="kv")
        self.sidecar = FakeHostPool(layer_num, log, name="indexer")
        self.anchor_entry = self._entry(PoolName.KV, self.anchor)
        self.entry_map = {
            PoolName.KV: self.anchor_entry,
            PoolName.INDEXER: self._entry(PoolName.INDEXER, self.sidecar),
        }

    @staticmethod
    def _entry(name, host_pool):
        return SimpleNamespace(
            name=name,
            host_pool=host_pool,
            device_pool=object(),
            layer_mapper=None,
            device_translate=None,
            packed_draft_device_pools=(),
        )


class AsyncLoadEnqueueTestBase(CustomTestCase):
    layer_num = 3

    def setUp(self):
        super().setUp()
        self.device_module = FakeDeviceModule()
        for module in (cache_controller, l2_transfer):
            patcher = mock.patch.object(module, "device_module", self.device_module)
            patcher.start()
            self.addCleanup(patcher.stop)
        # Process-cached; would carry a verdict taken on the real module.
        l2_transfer._timing_events_supported.cache_clear()
        self.addCleanup(l2_transfer._timing_events_supported.cache_clear)

    @property
    def log(self):
        return self.device_module.log

    def make_controller(self, async_enabled, cls=HiCacheController, host_pool=None):
        """A controller carrying only what the load path reads."""
        controller = cls.__new__(cls)
        controller.transfer_layer_id_max = self.layer_num
        controller.io_backend = "direct"
        controller.device = "cpu"
        controller.is_kv_sharded = False
        controller.kv_storage = None
        controller.mem_pool_device = object()
        controller.mem_pool_host = host_pool or FakeHostPool(self.layer_num, self.log)
        controller.load_queue = []
        controller.ack_load_queue = []
        controller.load_fence_stream = None
        controller.l2_transfer_engine = L2TransferEngine("direct")
        controller.async_load_enqueue = async_enabled
        controller.layer_done_counter = LayerDoneCounter(
            self.layer_num, async_enqueue=async_enabled
        )
        controller._init_load_enqueue_thread()
        self.addCleanup(self._shutdown, controller)

        real_move_indices = controller.move_indices

        def logging_move_indices(host_indices, device_indices):
            self.log.append(("move_indices", self.device_module.current_stream()))
            return real_move_indices(host_indices, device_indices)

        controller.move_indices = logging_move_indices
        return controller

    def _shutdown(self, controller):
        for pool in self.host_pools(controller):
            if pool.gate is not None:
                pool.gate.set()
        controller.stop_load_enqueue_thread()

    @staticmethod
    def host_pools(controller):
        host = controller.mem_pool_host
        if isinstance(host, FakeHostPoolGroup):
            return [host.anchor, host.sidecar]
        return [host]

    def gate_loads(self, controller):
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        return gate

    def queue_burst(self, controller, node_id, num_tokens=2):
        controller.load_queue.append(
            CacheOperation(
                torch.arange(num_tokens, dtype=torch.int64),
                torch.arange(num_tokens, dtype=torch.int64),
                node_id,
            )
        )

    def entries(self, kind):
        return [entry for entry in self.log if entry[0] == kind]

    def run_in_thread(self, fn):
        """Start fn on a thread; the returned event is set once it returns."""
        done = threading.Event()

        def runner():
            fn()
            done.set()

        threading.Thread(target=runner, daemon=True).start()
        return done


class TestInlinePath(AsyncLoadEnqueueTestBase):
    def test_gate_off_keeps_the_inline_call_sequence(self):
        controller = self.make_controller(async_enabled=False)
        fence = FakeStream("forward", self.log)
        controller.load_fence_stream = fence
        self.queue_burst(controller, node_id=11)

        producer_id = controller.start_loading()

        self.assertIsNone(controller.load_enqueue_thread)
        self.assertEqual(self.device_module.set_device_calls, [])
        ack = controller.ack_load_queue[0]
        slot = controller.layer_done_counter.events[producer_id]
        engine = controller.l2_transfer_engine
        index, h2d = engine.index_stream, engine.host_to_device_stream
        me = _thread()
        prepared = self.entries("record")[1][1]
        expected = [
            ("move_indices", self.device_module.default_stream),
            ("record", slot.start_event, me),
            ("stream_wait_stream", h2d, fence, me),
            ("stream_enter", index, me),
            ("event_wait", slot.start_event, index, me),
            ("record", prepared, me),
            ("stream_exit", index, me),
            ("stream_enter", h2d, me),
            ("event_wait", prepared, h2d, me),
            ("record", ack.start_event, me),
        ]
        for layer in range(self.layer_num):
            expected.append(("load_layer", layer, me, "kv"))
            expected.append(("record", slot.load_events[layer], me))
        expected.append(("record", ack.finish_event, me))
        expected.append(("stream_exit", h2d, me))
        self.assertEqual(self.log, expected)
        self.assertEqual(ack.node_ids, [11])
        self.assertEqual(ack.num_tokens, 2)

    def test_unified_storage_keeps_the_inline_path(self):
        # Its page moves are ordered against a load only on the scheduler thread.
        controller = HiCacheController.__new__(HiCacheController)
        with envs.SGLANG_ENABLE_HICACHE_ASYNC_LOAD_ENQUEUE.override(True):
            controller.kv_storage = None
            self.assertTrue(controller._resolve_async_load_enqueue())
            controller.kv_storage = object()
            with self.assertLogs(cache_controller.logger, level="WARNING"):
                self.assertFalse(controller._resolve_async_load_enqueue())


class TestAsyncEnqueue(AsyncLoadEnqueueTestBase):
    def test_hand_over_returns_before_any_layer_is_submitted(self):
        controller = self.make_controller(async_enabled=True)
        gate = self.gate_loads(controller)
        self.queue_burst(controller, node_id=21)

        producer_id = controller.start_loading()
        slot = controller.layer_done_counter.events[producer_id]

        self.assertEqual(producer_id, 0)
        self.assertEqual(controller.load_queue, [])
        self.assertEqual(slot.start_event.recorded_on, _thread())
        self.assertEqual(self.entries("load_layer"), [])
        self.assertEqual(controller.ack_load_queue, [])

        gate.set()
        controller._drain_load_enqueue()

        self.assertEqual(
            [entry[2] for entry in self.entries("load_layer")],
            [LOADER] * self.layer_num,
        )
        self.assertEqual(
            self.device_module.set_device_calls,
            [(self.device_module.device_index, LOADER)],
        )
        self.assertEqual([ack.node_ids for ack in controller.ack_load_queue], [[21]])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 2)

    def test_consumer_blocks_until_its_layer_is_recorded(self):
        controller = self.make_controller(async_enabled=True)
        gate = self.gate_loads(controller)
        self.queue_burst(controller, node_id=22)
        producer_id = controller.start_loading()
        controller.layer_done_counter.set_consumer(producer_id)

        # A device event not recorded yet does not block: the wait must.
        waited = self.run_in_thread(lambda: controller.layer_done_counter.wait_until(0))
        self.assertFalse(waited.wait(0.3))
        self.assertEqual(self.entries("stream_wait_event"), [])

        gate.set()
        self.assertTrue(waited.wait(TIMEOUT))
        slot = controller.layer_done_counter.events[producer_id]
        self.assertEqual(self.entries("stream_wait_event")[0][2], slot.load_events[0])
        self.assertEqual(slot.load_events[0].recorded_on, LOADER)

    def test_bursts_and_acks_keep_submission_order(self):
        controller = self.make_controller(async_enabled=True)
        gate = self.gate_loads(controller)
        self.queue_burst(controller, node_id=31)
        first = controller.start_loading()
        self.queue_burst(controller, node_id=32)
        second = controller.start_loading()
        gate.set()
        controller._drain_load_enqueue()

        self.assertEqual([first, second], [0, 1])
        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue], [[31], [32]]
        )
        self.assertEqual(
            [entry[1] for entry in self.entries("load_layer")],
            list(range(self.layer_num)) * 2,
        )

    def test_slot_reuse_waits_for_the_queued_burst(self):
        controller = self.make_controller(async_enabled=True)
        gate = self.gate_loads(controller)
        for node_id in (51, 52, 53):
            self.queue_burst(controller, node_id=node_id)
            controller.start_loading()

        self.queue_burst(controller, node_id=54)
        reused = []
        done = self.run_in_thread(lambda: reused.append(controller.start_loading()))
        self.assertFalse(done.wait(0.3), "slot 0 was reused while its burst is queued")

        gate.set()
        self.assertTrue(done.wait(TIMEOUT))
        self.assertEqual(reused, [0])
        controller._drain_load_enqueue()
        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue],
            [[51], [52], [53], [54]],
        )

    def test_queued_bursts_count_as_unreaped(self):
        # Every rank must count the same, whenever its loader appends the ack.
        controller = self.make_controller(async_enabled=True)
        gate = self.gate_loads(controller)
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = controller
        for node_id in (61, 62, 63):
            self.queue_burst(controller, node_id=node_id)
            controller.start_loading()

        self.assertEqual(controller.ack_load_queue, [])
        self.assertEqual(controller.num_unreaped_load_bursts(), 3)
        self.assertFalse(cache.has_free_load_back_slot())

        gate.set()
        controller._drain_load_enqueue()
        self.assertEqual(controller.num_unreaped_load_bursts(), 3)
        controller.ack_load_queue.pop(0)
        self.assertTrue(cache.has_free_load_back_slot())

    def test_reset_drains_the_loader_before_clearing_acks(self):
        controller = self.make_controller(async_enabled=True)
        controller.write_queue = []
        controller.ack_write_queue = []
        controller.enable_storage = False
        controller.storage_stop_event = threading.Event()
        controller._stop_storage_threads = lambda: None
        gate = self.gate_loads(controller)
        self.queue_burst(controller, node_id=64)
        controller.start_loading()

        threading.Timer(0.1, gate.set).start()
        controller.reset()

        self.assertEqual(len(self.entries("load_layer")), self.layer_num)
        self.assertEqual(controller.ack_load_queue, [])
        self.assertEqual(controller.num_unreaped_load_bursts(), 0)

    def test_fence_is_captured_at_hand_over_and_waited_by_the_loader(self):
        controller = self.make_controller(async_enabled=True)
        fence = FakeStream("forward", self.log)
        controller.load_fence_stream = fence
        gate = self.gate_loads(controller)
        self.queue_burst(controller, node_id=65)

        controller.start_loading()

        fence_events = [
            entry[1]
            for entry in self.entries("record")
            if entry[1].recorded_stream is fence
        ]
        self.assertEqual(len(fence_events), 1)
        self.assertEqual(fence_events[0].recorded_on, _thread())

        gate.set()
        controller._drain_load_enqueue()

        h2d = controller.l2_transfer_engine.host_to_device_stream
        self.assertEqual(self.entries("stream_wait_stream"), [])
        fence_wait = ("event_wait", fence_events[0], h2d, LOADER)
        self.assertLess(
            self.log.index(fence_wait), self.log.index(self.entries("load_layer")[0])
        )

    def test_index_move_runs_behind_the_start_event(self):
        # op.device_indices is written on the scheduler's stream: the loader's
        # host read of it has no other ordering against that stream.
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=66)

        producer_id = controller.start_loading()
        controller._drain_load_enqueue()

        index = controller.l2_transfer_engine.index_stream
        start_event = controller.layer_done_counter.events[producer_id].start_event
        self.assertEqual(self.entries("move_indices"), [("move_indices", index)])
        self.assertLess(
            self.log.index(("event_wait", start_event, index, LOADER)),
            self.log.index(("move_indices", index)),
        )

    def test_hybrid_burst_loads_every_pool_on_the_loader(self):
        controller = self.make_controller(
            async_enabled=True,
            cls=HybridCacheController,
            host_pool=FakeHostPoolGroup(self.layer_num, self.log),
        )
        controller.load_queue.append(
            CacheOperation(
                torch.arange(2, dtype=torch.int64),
                torch.arange(2, dtype=torch.int64),
                71,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.INDEXER,
                        host_indices=torch.arange(2, dtype=torch.int64) + 100,
                        device_indices=torch.arange(2, dtype=torch.int64) + 200,
                    )
                ],
            )
        )

        controller.start_loading()
        controller._drain_load_enqueue()

        self.assertEqual(
            [entry[1:] for entry in self.entries("load_layer")],
            [
                (layer, LOADER, name)
                for layer in range(self.layer_num)
                for name in ("kv", "indexer")
            ],
        )
        self.assertEqual(
            controller.ack_load_queue[0].num_tokens_by_pool,
            {str(PoolName.KV): 2, str(PoolName.INDEXER): 2},
        )


class TestLoaderFailure(AsyncLoadEnqueueTestBase):
    def test_failed_burst_releases_its_consumer_and_its_ack(self):
        controller = self.make_controller(async_enabled=True)
        controller.mem_pool_host.fail_at_layer = 1
        self.queue_burst(controller, node_id=81)

        with self.assertLogs(cache_controller.logger, level="ERROR") as logs:
            producer_id = controller.start_loading()
            controller._drain_load_enqueue()

        self.assertIn("81", "\n".join(logs.output))
        self.assertIs(controller.load_enqueue_error, controller.mem_pool_host.failure)
        controller.layer_done_counter.set_consumer(producer_id)
        waited = self.run_in_thread(
            lambda: controller.layer_done_counter.wait_until(self.layer_num - 1)
        )
        self.assertTrue(
            waited.wait(TIMEOUT), "a failed burst left its consumer blocked"
        )
        self.assertEqual([ack.node_ids for ack in controller.ack_load_queue], [[81]])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 0)

        controller.mem_pool_host.fail_at_layer = None
        self.queue_burst(controller, node_id=82)
        self.assertEqual(controller.start_loading(), 1)
        controller._drain_load_enqueue()
        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue], [[81], [82]]
        )
        self.assertEqual(controller.num_unreaped_load_bursts(), 2)


class TestStopLoaderThread(AsyncLoadEnqueueTestBase):
    def test_stop_drains_then_enqueues_inline(self):
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=91)
        controller.start_loading()

        controller.stop_load_enqueue_thread()

        self.assertEqual([ack.node_ids for ack in controller.ack_load_queue], [[91]])
        self.assertNotIn(LOADER, [t.name for t in threading.enumerate()])

        # The slots keep their handshake: a full rotation must not park.
        def rotate():
            for node_id in (92, 93, 94):
                self.queue_burst(controller, node_id=node_id)
                controller.start_loading()

        self.assertTrue(self.run_in_thread(rotate).wait(TIMEOUT))
        self.assertEqual(len(controller.ack_load_queue), 4)
        self.assertNotIn(
            LOADER, [entry[2] for entry in self.entries("load_layer")[self.layer_num :]]
        )


class TestLoadBackEventDone(AsyncLoadEnqueueTestBase):
    def test_queued_slot_is_not_done_until_enqueued(self):
        # Until then the slot's finish_event is the previous rotation's.
        counter = LayerDoneCounter(self.layer_num, async_enqueue=True)
        cache = object.__new__(UnifiedRadixCache)
        cache.cache_controller = SimpleNamespace(layer_done_counter=counter)

        slot = counter.update_producer()
        self.assertFalse(cache.is_load_back_event_done(slot))
        counter.events[slot].mark_enqueue_done()
        self.assertTrue(cache.is_load_back_event_done(slot))


class TestHybridCounterOverride(AsyncLoadEnqueueTestBase):
    def test_transfer_layer_override_keeps_the_handshake(self):
        device_pool = SimpleNamespace(
            device="cpu",
            layer_num=3,
            layer_router=SimpleNamespace(wrapped_view=lambda: None),
            register_layer_transfer_counter=lambda counter: None,
        )
        host_pool = SimpleNamespace(layout="page_first_direct", layer_num=3)
        host_pool.anchor_entry = SimpleNamespace(name=PoolName.KV, host_pool=host_pool)
        host_pool.entry_map = {}
        with envs.SGLANG_ENABLE_HICACHE_ASYNC_LOAD_ENQUEUE.override(True):
            controller = HybridCacheController(
                token_to_kv_pool_allocator=SimpleNamespace(
                    get_kvcache=lambda: device_pool
                ),
                mem_pool_host=host_pool,
                page_size=1,
                tp_group=None,
                load_cache_event=threading.Event(),
                transfer_layer_id_max=5,
            )
        self.addCleanup(controller.stop_load_enqueue_thread)

        self.assertIsNotNone(controller.load_enqueue_thread)
        slot = controller.layer_done_counter.update_producer()
        self.assertFalse(controller.layer_done_counter.events[slot].is_enqueue_done())


class TestEngineSharedByTwoThreads(AsyncLoadEnqueueTestBase):
    def _transfer(self):
        indices = torch.arange(2, dtype=torch.int64)
        return L2Transfer(
            host_pool=FakeHostPool(self.layer_num, self.log),
            device_pool=object(),
            host_indices=indices,
            device_indices=indices,
        )

    def _submit_on_another_thread(self, engine):
        done = self.run_in_thread(
            lambda: engine.submit_host_to_device(
                [self._transfer()], transfer_layer_id_max=self.layer_num
            )
        )
        self.assertTrue(done.wait(TIMEOUT))

    def test_on_current_stream_does_not_redirect_another_thread(self):
        engine = L2TransferEngine("direct")
        with engine.on_current_stream():
            self._submit_on_another_thread(engine)

        entered = {entry[1] for entry in self.entries("stream_enter")}
        self.assertEqual(entered, {engine.index_stream, engine.host_to_device_stream})

    def test_own_stream_submissions_do_not_overlap(self):
        # A host layout keeps the last released finish event per stream, so two
        # threads must release in the order they recorded on it.
        engine = L2TransferEngine("direct")
        first = self._transfer()
        first.host_pool.gate = threading.Event()
        self.run_in_thread(
            lambda: engine.submit_host_to_device(
                [first], transfer_layer_id_max=self.layer_num
            )
        )
        second = self.run_in_thread(
            lambda: engine.submit_host_to_device(
                [self._transfer()], transfer_layer_id_max=self.layer_num
            )
        )
        self.assertFalse(second.wait(0.3))
        with engine.on_current_stream():
            engine.submit_host_to_device(
                [self._transfer()], transfer_layer_id_max=self.layer_num
            )

        first.host_pool.gate.set()
        self.assertTrue(second.wait(TIMEOUT))


if __name__ == "__main__":
    unittest.main()
