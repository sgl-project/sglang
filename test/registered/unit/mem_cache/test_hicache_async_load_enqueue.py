"""Unit tests for the async HiCache load-enqueue split.

``start_loading`` used to submit every layer's host->device copy inline on the
scheduler thread, which is hundreds of milliseconds of descriptor construction
with io_backend='direct' -- long enough that the copies were finished before
the forward ever launched, so the layer-wise pipeline overlapped nothing.
``SGLANG_HICACHE_ASYNC_LOAD_ENQUEUE`` keeps the producer-slot hand-out on the
scheduler thread and moves the enqueue loop to a dedicated worker.

There is no device here, so the ``device_module`` globals of the controller and
of the transfer engine are swapped for a fake that records the exact
stream/event call sequence together with the thread that made each call; the
KV pools are duck-typed for the same reason. The index tensors are real CPU
tensors, because ``move_indices`` and ``merge_ops`` are the production ones.
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
from sglang.srt.mem_cache.l2_transfer import L2TransferEngine
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

LOADER_THREAD_NAME = "hicache-load-enqueue"
JOIN_TIMEOUT = 10.0


def _thread() -> str:
    return threading.current_thread().name


class FakeEvent:
    def __init__(self, log, enable_timing=False):
        self._log = log
        self.enable_timing = enable_timing
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

    def elapsed_time(self, other):
        return 0.0


class FakeStream:
    def __init__(self, name, log=None):
        self.name = name
        self._log = log if log is not None else []

    def wait_event(self, event):
        self._log.append(("stream_wait_event", self, event, _thread()))

    def wait_stream(self, stream):
        self._log.append(("stream_wait_stream", self, stream, _thread()))

    def __repr__(self):
        return f"<FakeStream {self.name}>"


class _StreamContext:
    def __init__(self, module, stream):
        self._module = module
        self._stream = stream

    def __enter__(self):
        self._module.log.append(("stream_enter", self._stream, _thread()))
        self._previous = self._module.current_stream()
        self._module._local.stream = self._stream
        return self._stream

    def __exit__(self, *exc):
        self._module._local.stream = self._previous
        self._module.log.append(("stream_exit", self._stream, _thread()))
        return False


class FakeDeviceModule:
    """Enough of torch.cuda for the load path, with an ordered call log."""

    __name__ = "fake_device_module"

    def __init__(self):
        self.log = []
        self.set_device_calls = []
        self.device_index = 7
        self._local = threading.local()
        self._default_stream = FakeStream("default", self.log)
        self._stream_count = 0

    def Event(self, enable_timing=False):
        return FakeEvent(self.log, enable_timing)

    def Stream(self):
        self._stream_count += 1
        return FakeStream(f"stream{self._stream_count}", self.log)

    def stream(self, stream):
        return _StreamContext(self, stream)

    def current_stream(self):
        return getattr(self._local, "stream", self._default_stream)

    def current_device(self):
        return self.device_index

    def set_device(self, index):
        self.set_device_calls.append((index, _thread()))


class FakeHostPool:
    """Duck-typed host pool: records each per-layer submission, and can be told
    to block or to fail so a test can observe a burst mid-flight."""

    layout = "page_first_direct"
    size_per_token = 16

    def __init__(self, layer_num, log, name="kv"):
        self.layer_num = layer_num
        self.name = name
        self._log = log
        self.gate = None
        self.fail_at_layer = None
        self.failure = RuntimeError("injected load_to_device_per_layer failure")

    def load_to_device_per_layer(
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


class AsyncLoadEnqueueTestBase(CustomTestCase):
    layer_num = 3

    def setUp(self):
        super().setUp()
        self.device_module = FakeDeviceModule()
        for module in (cache_controller, l2_transfer):
            patcher = mock.patch.object(module, "device_module", self.device_module)
            patcher.start()
            self.addCleanup(patcher.stop)
        # The support probe is process-cached and would otherwise carry a
        # verdict taken against the real device module.
        l2_transfer._timing_events_supported.cache_clear()
        self.addCleanup(l2_transfer._timing_events_supported.cache_clear)

    @property
    def log(self):
        return self.device_module.log

    def _wire(self, controller, async_enabled: bool, mem_pool_host):
        controller.layer_num = self.layer_num
        controller.io_backend = "direct"
        controller.device = "cpu"
        controller.mem_pool_device = object()
        controller.mem_pool_host = mem_pool_host
        controller.load_queue = []
        controller.ack_load_queue = []
        controller.load_fence_stream = None
        controller.l2_transfer_engine = L2TransferEngine("direct")
        controller.async_load_enqueue = async_enabled
        controller.layer_done_counter = LayerDoneCounter(
            self.layer_num, async_enqueue=async_enabled
        )
        controller._init_load_enqueue_thread(async_enabled)
        self.addCleanup(self._shutdown, controller)

        real_move_indices = controller.move_indices

        def logging_move_indices(host_indices, device_indices):
            self.log.append(("move_indices", _thread()))
            return real_move_indices(host_indices, device_indices)

        controller.move_indices = logging_move_indices
        return controller

    def make_controller(self, async_enabled: bool) -> HiCacheController:
        """A controller carrying only what the load path touches.

        __init__ wants a real allocator, host pool and process group; the load
        path wants none of that, so the fields are set directly and the one
        initializer under test is called explicitly.
        """
        controller = HiCacheController.__new__(HiCacheController)
        return self._wire(
            controller, async_enabled, FakeHostPool(self.layer_num, self.log)
        )

    def _shutdown(self, controller):
        gate = getattr(controller.mem_pool_host, "gate", None)
        if gate is not None:
            gate.set()
        controller.stop_load_enqueue_thread()

    @staticmethod
    def h2d(controller):
        return controller.l2_transfer_engine.host_to_device_stream

    def queue_burst(self, controller, node_id, num_tokens=2):
        controller.load_queue.append(
            CacheOperation(
                torch.arange(num_tokens, dtype=torch.int64),
                torch.arange(num_tokens, dtype=torch.int64),
                node_id,
            )
        )

    def loader_threads(self):
        return [t for t in threading.enumerate() if t.name == LOADER_THREAD_NAME]

    def entries(self, kind):
        return [entry for entry in self.log if entry[0] == kind]

    def _wait_for(self, predicate, timeout=JOIN_TIMEOUT):
        tick = threading.Event()
        waited = 0.0
        while waited < timeout:
            if predicate():
                return
            tick.wait(0.01)
            waited += 0.01
        self.fail("timed out waiting for the loader thread to make progress")

    def _run_with_timeout(self, fn, message, timeout=JOIN_TIMEOUT):
        done = threading.Event()
        failure = []

        def runner():
            try:
                fn()
            except BaseException as e:  # surfaced below; a dead thread reads as a hang
                failure.append(e)
            else:
                done.set()

        worker = threading.Thread(target=runner, daemon=True)
        worker.start()
        worker.join(timeout=timeout)
        if failure:
            raise failure[0]
        self.assertTrue(done.is_set(), message)


class TestSyncPathUnchanged(AsyncLoadEnqueueTestBase):
    def test_gate_off_creates_no_thread(self):
        controller = self.make_controller(async_enabled=False)
        self.assertIsNone(controller.load_enqueue_queue)
        self.assertIsNone(controller.load_enqueue_thread)
        self.assertEqual(self.loader_threads(), [])
        self.assertEqual(self.device_module.set_device_calls, [])

    def test_gate_off_call_sequence_is_the_pre_split_one(self):
        controller = self.make_controller(async_enabled=False)
        self.queue_burst(controller, node_id=11)

        producer_id = controller.start_loading()

        self.assertEqual(producer_id, 0)
        self.assertEqual(len(controller.ack_load_queue), 1)
        ack = controller.ack_load_queue[0]
        producer_event = controller.layer_done_counter.events[producer_id]
        h2d = self.h2d(controller)
        me = threading.current_thread().name

        expected = [
            ("move_indices", me),
            ("record", producer_event.start_event, me),
            ("stream_enter", h2d, me),
            ("event_wait", producer_event.start_event, h2d, me),
            ("record", ack.start_event, me),
        ]
        for layer in range(self.layer_num):
            expected.append(("load_layer", layer, me, "kv"))
            expected.append(("record", producer_event.load_events[layer], me))
        expected.append(("record", ack.finish_event, me))
        expected.append(("stream_exit", h2d, me))

        self.assertEqual(self.log, expected)
        self.assertEqual(ack.node_ids, [11])
        self.assertEqual(ack.num_tokens, 2)
        self.assertEqual(ack.num_tokens_by_pool, {PoolName.KV.value: 2})
        self.assertEqual(ack.num_bytes, 2 * FakeHostPool.size_per_token)
        self.assertEqual(controller.load_queue, [])

    def test_gate_off_empty_queue_returns_minus_one(self):
        controller = self.make_controller(async_enabled=False)
        self.assertEqual(controller.start_loading(), -1)
        self.assertEqual(self.log, [])

    def test_gate_off_fence_still_waits_on_the_forward_stream_inline(self):
        controller = self.make_controller(async_enabled=False)
        fence = FakeStream("forward", self.log)
        controller.load_fence_stream = fence
        self.queue_burst(controller, node_id=12)

        controller.start_loading()

        me = threading.current_thread().name
        self.assertEqual(
            self.entries("stream_wait_stream"),
            [("stream_wait_stream", self.h2d(controller), fence, me)],
        )
        kinds = [entry[0] for entry in self.log]
        self.assertLess(kinds.index("stream_wait_stream"), kinds.index("load_layer"))


class TestAsyncEnqueue(AsyncLoadEnqueueTestBase):
    def test_producer_is_synchronous_and_the_enqueue_is_not(self):
        controller = self.make_controller(async_enabled=True)
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        self.queue_burst(controller, node_id=21)

        producer_id = controller.start_loading()
        producer_event = controller.layer_done_counter.events[producer_id]
        me = threading.current_thread().name

        # Synchronous half: the slot and the compute-stream start point,
        # nothing more -- the batch is created with this id as
        # hicache_consumer_index.
        self.assertEqual(producer_id, 0)
        self.assertEqual(controller.load_queue, [])
        self.assertEqual(producer_event.start_event.recorded_on, me)
        self.assertEqual(self.entries("load_layer"), [])
        self.assertEqual(controller.ack_load_queue, [])

        gate.set()
        controller._drain_load_enqueue()

        # Asynchronous half: every bit of it on the loader thread.
        self.assertEqual(
            [entry[2] for entry in self.entries("load_layer")],
            [LOADER_THREAD_NAME] * self.layer_num,
        )
        self.assertEqual(
            [entry[1] for entry in self.entries("move_indices")],
            [LOADER_THREAD_NAME],
        )
        # Two H2D stream contexts on the async path: the short one that orders
        # the index move behind start_event, then the transfer body.
        self.assertEqual(
            [entry[2] for entry in self.entries("stream_enter")],
            [LOADER_THREAD_NAME] * 2,
        )
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [21])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 2)

    def test_loader_thread_binds_to_the_scheduler_device(self):
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=22)
        controller.start_loading()
        controller._drain_load_enqueue()
        self.assertEqual(
            self.device_module.set_device_calls,
            [(self.device_module.device_index, LOADER_THREAD_NAME)],
        )

    def test_ack_is_published_only_after_the_last_layer(self):
        controller = self.make_controller(async_enabled=True)
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        self.queue_burst(controller, node_id=23)
        controller.start_loading()

        # The worker is parked inside the per-layer loop, past move_indices
        # and inside the H2D stream context, and the ack must not be visible.
        self._wait_for(lambda: self.entries("stream_enter"))
        self.assertEqual(controller.ack_load_queue, [])

        gate.set()
        controller._drain_load_enqueue()
        self.assertEqual(len(controller.ack_load_queue), 1)

    def test_every_layer_completes_so_a_consumer_cannot_hang(self):
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=24)
        producer_id = controller.start_loading()
        controller.layer_done_counter.set_consumer(producer_id)

        # wait_until blocks on the CPU-side handshake before touching the
        # device event, so a layer that was never submitted would park this
        # thread.
        self._run_with_timeout(
            lambda: [
                controller.layer_done_counter.wait_until(layer)
                for layer in range(self.layer_num)
            ],
            "consumer hung waiting for a layer that was never enqueued",
        )

        producer_event = controller.layer_done_counter.events[producer_id]
        for layer in range(self.layer_num):
            self.assertEqual(
                producer_event.load_events[layer].recorded_on, LOADER_THREAD_NAME
            )

    def test_bursts_complete_in_submission_order(self):
        controller = self.make_controller(async_enabled=True)
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        self.queue_burst(controller, node_id=31)
        first = controller.start_loading()
        self.queue_burst(controller, node_id=32)
        second = controller.start_loading()

        self.assertEqual([first, second], [0, 1])

        gate.set()
        controller._drain_load_enqueue()

        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue], [[31], [32]]
        )
        # One burst's layers never interleave with the next one's: a single
        # worker keeps the H2D stream, the slot rotation and the acks in one
        # order.
        self.assertEqual(
            [entry[1] for entry in self.entries("load_layer")],
            list(range(self.layer_num)) * 2,
        )

    def test_merged_burst_carries_every_node_id(self):
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=41)
        self.queue_burst(controller, node_id=42)
        controller.start_loading()
        controller._drain_load_enqueue()
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [41, 42])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 4)

    def test_slot_reuse_waits_for_the_queued_burst(self):
        controller = self.make_controller(async_enabled=True)
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        # Fill the whole rotation, then ask for the slot the first burst holds.
        for node_id in (51, 52, 53):
            self.queue_burst(controller, node_id=node_id)
            controller.start_loading()

        self.queue_burst(controller, node_id=54)
        reused = []
        blocked = threading.Thread(
            target=lambda: reused.append(controller.start_loading())
        )
        blocked.start()
        blocked.join(timeout=0.5)
        self.assertTrue(
            blocked.is_alive(),
            "slot 0 was handed out again while its burst was still queued",
        )

        gate.set()
        blocked.join(timeout=JOIN_TIMEOUT)
        self.assertFalse(blocked.is_alive())
        self.assertEqual(reused, [0])
        controller._drain_load_enqueue()
        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue],
            [[51], [52], [53], [54]],
        )

    def test_reset_does_not_resurrect_a_drained_ack(self):
        controller = self.make_controller(async_enabled=True)
        controller.write_queue = []
        controller.ack_write_queue = []
        controller.enable_storage = False
        controller.storage_stop_event = threading.Event()
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        self.queue_burst(controller, node_id=61)
        controller.start_loading()

        gate.set()
        controller.reset()
        self.assertEqual(controller.ack_load_queue, [])

    def test_fence_is_captured_at_hand_over_and_waited_on_the_loader(self):
        controller = self.make_controller(async_enabled=True)
        fence = FakeStream("forward", self.log)
        controller.load_fence_stream = fence
        gate = threading.Event()
        controller.mem_pool_host.gate = gate
        self.queue_burst(controller, node_id=62)
        me = threading.current_thread().name

        controller.start_loading()

        # The fence point is taken on the scheduler thread, before the burst
        # is handed over, so it is the forward stream *now* and not wherever
        # that stream is once the loader thread gets to the burst.
        fence_records = [
            entry
            for entry in self.entries("record")
            if entry[1].recorded_stream is fence
        ]
        self.assertEqual(len(fence_records), 1)
        fence_event = fence_records[0][1]
        self.assertEqual(fence_event.recorded_on, me)

        gate.set()
        controller._drain_load_enqueue()

        h2d = self.h2d(controller)
        self.assertEqual(self.entries("stream_wait_stream"), [])
        fence_waits = [
            entry for entry in self.entries("event_wait") if entry[1] is fence_event
        ]
        self.assertEqual(
            fence_waits, [("event_wait", fence_event, h2d, LOADER_THREAD_NAME)]
        )
        wait_at = self.log.index(fence_waits[0])
        move_at = self.log.index(self.entries("move_indices")[0])
        load_at = self.log.index(self.entries("load_layer")[0])
        self.assertLess(move_at, wait_at)
        self.assertLess(wait_at, load_at)


class UnsettledDeviceIndices:
    """Stands in for the allocator's device index tensor.

    Its *values* are written by kernels on the compute stream, so the blocking
    ``.cpu()`` that ends ``move_indices`` for the 'direct' backends only reads
    real indices once the reader is ordered behind
    ``producer_event.start_event``. Read any earlier and it hands back garbage
    -- which in production becomes wild device addresses in the batched copy.
    """

    GARBAGE = -(2**40)

    def __init__(self, values, is_settled):
        self._values = torch.tensor(values, dtype=torch.int64)
        self._is_settled = is_settled
        self.reads = []

    def cpu(self):
        out = (
            self._values.clone()
            if self._is_settled()
            else torch.full_like(self._values, self.GARBAGE)
        )
        self.reads.append(out.tolist())
        return out

    def __len__(self):
        return self._values.numel()


class TestAsyncOrdersTheIndexMove(AsyncLoadEnqueueTestBase):
    """The cross-stream race the split introduces, and the ordering that fixes it.

    ``op.device_indices`` comes out of the allocator, and its values are
    produced on the *compute* stream. Inline (gate off) the burst runs on the
    scheduler thread, so the index move's D2H is implicitly behind those
    kernels. On the loader thread the current stream is the default one and
    nothing orders it -- the burst has to enter the H2D stream and wait
    ``start_event`` before it reads.
    """

    def record_move_stream(self, controller):
        """Capture the fake current stream at every move_indices call."""
        moved_on = []
        wrapped = controller.move_indices

        def recording_move_indices(host_indices, device_indices):
            moved_on.append(self.device_module.current_stream())
            return wrapped(host_indices, device_indices)

        controller.move_indices = recording_move_indices
        return moved_on

    def index_of(self, predicate, message):
        for position, entry in enumerate(self.log):
            if predicate(entry):
                return position
        self.fail(message)

    def test_move_runs_inside_the_h2d_stream_after_the_start_event_wait(self):
        controller = self.make_controller(async_enabled=True)
        moved_on = self.record_move_stream(controller)
        self.queue_burst(controller, node_id=101)

        producer_id = controller.start_loading()
        controller._drain_load_enqueue()
        start_event = controller.layer_done_counter.events[producer_id].start_event
        h2d = self.h2d(controller)

        # The move saw the H2D stream as the current stream, not the default.
        self.assertEqual(moved_on, [h2d])

        enter_at = self.index_of(
            lambda e: e[0] == "stream_enter" and e[1] is h2d,
            "the burst never entered the H2D stream",
        )
        wait_at = self.index_of(
            lambda e: e[0] == "event_wait" and e[1] is start_event and e[2] is h2d,
            "the H2D stream never waited on the producer's start_event",
        )
        move_at = self.index_of(
            lambda e: e[0] == "move_indices", "move_indices never ran"
        )
        load_at = self.index_of(
            lambda e: e[0] == "load_layer", "no layer was ever submitted"
        )
        self.assertLess(enter_at, wait_at)
        self.assertLess(wait_at, move_at)
        self.assertLess(move_at, load_at)

    def test_gate_off_still_moves_on_the_scheduler_stream(self):
        # The inline path is unchanged: the scheduler thread's current stream
        # is the compute one, which already orders the D2H, and start_event is
        # recorded after the move exactly as it always was.
        controller = self.make_controller(async_enabled=False)
        moved_on = self.record_move_stream(controller)
        self.queue_burst(controller, node_id=102)

        controller.start_loading()

        self.assertEqual(moved_on, [self.device_module._default_stream])
        kinds = [entry[0] for entry in self.log]
        self.assertLess(kinds.index("move_indices"), kinds.index("stream_enter"))

    def test_unsettled_indices_are_never_read_before_the_wait(self):
        controller = self.make_controller(async_enabled=True)
        h2d = self.h2d(controller)
        # First burst on a fresh controller, so the slot is 0; naming the event
        # up front keeps the predicate free of any race with the loader thread.
        start_event = controller.layer_done_counter.events[0].start_event

        def h2d_waited_on_start():
            return any(
                entry[0] == "event_wait" and entry[1] is start_event and entry[2] is h2d
                for entry in self.log
            )

        # Guard: the sentinel really does bite, so a green run below is not
        # vacuous -- an unordered read of these indices returns garbage.
        probe = UnsettledDeviceIndices([5, 6], h2d_waited_on_start)
        self.assertEqual(probe.cpu().tolist(), [probe.GARBAGE] * 2)

        device_indices = UnsettledDeviceIndices([5, 6], h2d_waited_on_start)
        controller.load_queue.append(
            CacheOperation(torch.arange(2, dtype=torch.int64), device_indices, 103)
        )

        producer_id = controller.start_loading()
        controller._drain_load_enqueue()

        self.assertEqual(producer_id, 0)
        self.assertEqual(device_indices.reads, [[5, 6]])
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [103])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 2)


class TestLoaderThreadFailure(AsyncLoadEnqueueTestBase):
    def test_failure_is_logged_and_releases_every_waiter(self):
        controller = self.make_controller(async_enabled=True)
        controller.mem_pool_host.fail_at_layer = 1
        self.queue_burst(controller, node_id=71)

        with self.assertLogs(cache_controller.logger, level="ERROR") as logs:
            producer_id = controller.start_loading()
            controller._drain_load_enqueue()

        self.assertIn("71", "\n".join(logs.output))
        self.assertIs(controller.load_enqueue_error, controller.mem_pool_host.failure)

        # A forward already parked on this producer cannot be interrupted, so
        # the abort path has to release it rather than leave it waiting.
        producer_event = controller.layer_done_counter.events[producer_id]
        controller.layer_done_counter.set_consumer(producer_id)
        self._run_with_timeout(
            lambda: [
                controller.layer_done_counter.wait_until(layer)
                for layer in range(self.layer_num)
            ],
            "a failed burst left the consumer blocked",
        )
        for layer in range(self.layer_num):
            self.assertEqual(
                producer_event.load_events[layer].recorded_on, LOADER_THREAD_NAME
            )

        # And the ack still lands, or loading_check would never reach the
        # bursts queued behind this one.
        self.assertEqual(len(controller.ack_load_queue), 1)
        self.assertEqual(controller.ack_load_queue[0].node_ids, [71])
        self.assertEqual(controller.ack_load_queue[0].num_tokens, 0)

    def test_controller_still_serves_the_next_burst(self):
        controller = self.make_controller(async_enabled=True)
        controller.mem_pool_host.fail_at_layer = 0
        self.queue_burst(controller, node_id=81)
        with self.assertLogs(cache_controller.logger, level="ERROR"):
            controller.start_loading()
            controller._drain_load_enqueue()

        controller.mem_pool_host.fail_at_layer = None
        self.queue_burst(controller, node_id=82)
        second = controller.start_loading()
        controller._drain_load_enqueue()

        self.assertEqual(second, 1)
        self.assertTrue(controller.load_enqueue_thread.is_alive())
        self.assertEqual(
            [ack.node_ids for ack in controller.ack_load_queue], [[81], [82]]
        )
        self.assertEqual(
            [entry[1] for entry in self.entries("load_layer")],
            list(range(self.layer_num)),
        )


class TestShutdown(AsyncLoadEnqueueTestBase):
    def test_thread_is_a_daemon(self):
        controller = self.make_controller(async_enabled=True)
        self.assertTrue(controller.load_enqueue_thread.daemon)
        self.assertEqual(len(self.loader_threads()), 1)

    def test_stop_joins_and_falls_back_to_the_inline_path(self):
        controller = self.make_controller(async_enabled=True)
        self.queue_burst(controller, node_id=91)
        controller.start_loading()

        with self.assertLogs(cache_controller.logger, level="INFO") as logs:
            controller.stop_load_enqueue_thread()
        self.assertIn(
            "HiCache async load enqueue: downgraded to inline (HiCacheController)",
            "\n".join(logs.output),
        )

        self.assertIsNone(controller.load_enqueue_thread)
        self.assertIsNone(controller.load_enqueue_queue)
        self.assertFalse(controller.async_load_enqueue)
        self.assertEqual(self.loader_threads(), [])
        # The queued burst was drained on the way out, not dropped.
        self.assertEqual(len(controller.ack_load_queue), 1)

        # Dropping the handshake with the thread is what keeps the rotation
        # from parking: nothing on the inline path hands _enqueue_done back.
        self.queue_burst(controller, node_id=92)
        self._run_with_timeout(
            controller.start_loading,
            "start_loading blocked after the loader thread was stopped",
        )
        self.assertEqual(len(controller.ack_load_queue), 2)
        self.assertNotIn(
            LOADER_THREAD_NAME,
            [entry[2] for entry in self.entries("load_layer")[-self.layer_num :]],
        )

    def test_stop_is_idempotent(self):
        controller = self.make_controller(async_enabled=True)
        controller.stop_load_enqueue_thread()
        controller.stop_load_enqueue_thread()
        self.assertEqual(self.loader_threads(), [])


class TestLoadBackEventDone(AsyncLoadEnqueueTestBase):
    """``is_load_back_event_done`` must not trust a slot's finish_event while
    the burst that holds the slot is still sitting in the loader queue."""

    def _stub(self, counter):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        stub = object.__new__(HiRadixCache)
        stub.cache_controller = SimpleNamespace(layer_done_counter=counter)
        stub.loading_check = mock.MagicMock()
        return stub

    def test_queued_slot_is_not_done_until_enqueued(self):
        counter = LayerDoneCounter(self.layer_num, async_enqueue=True)
        stub = self._stub(counter)
        self.assertTrue(stub.is_load_back_event_done(-1))

        slot = counter.update_producer()
        self.assertFalse(stub.is_load_back_event_done(slot))
        stub.loading_check.assert_not_called()

        counter.events[slot].mark_enqueue_done()
        self.assertTrue(stub.is_load_back_event_done(slot))
        stub.loading_check.assert_called_once()

    def test_gate_off_only_consults_the_device_event(self):
        counter = LayerDoneCounter(self.layer_num)
        stub = self._stub(counter)
        slot = counter.update_producer()
        self.assertTrue(stub.is_load_back_event_done(slot))


class FakeHostPoolGroup:
    """Duck-typed HostPoolGroup: an anchor KV pool plus one sidecar pool."""

    layout = "page_first_direct"

    def __init__(self, layer_num, log):
        self.anchor = FakeHostPool(layer_num, log, name="kv")
        self.sidecar = FakeHostPool(layer_num, log, name="indexer")
        self.anchor_entry = SimpleNamespace(
            name=PoolName.KV,
            host_pool=self.anchor,
            device_pool=object(),
            layer_mapper=None,
            packed_draft_device_pools=(),
        )
        sidecar_entry = SimpleNamespace(
            name=PoolName.INDEXER,
            host_pool=self.sidecar,
            device_pool=object(),
            layer_mapper=None,
            packed_draft_device_pools=(),
        )
        self.entry_map = {
            PoolName.KV: self.anchor_entry,
            PoolName.INDEXER: sidecar_entry,
        }

    @property
    def gate(self):
        return self.anchor.gate

    @gate.setter
    def gate(self, value):
        self.anchor.gate = value


class TestHybridBurst(AsyncLoadEnqueueTestBase):
    """The hybrid controller inherits the split: its sidecar pool transfers
    ride the same burst on the loader thread."""

    def make_hybrid_controller(self, async_enabled: bool) -> HybridCacheController:
        controller = HybridCacheController.__new__(HybridCacheController)
        return self._wire(
            controller, async_enabled, FakeHostPoolGroup(self.layer_num, self.log)
        )

    def queue_hybrid_burst(self, controller, node_id, num_tokens=2):
        controller.load_queue.append(
            CacheOperation(
                torch.arange(num_tokens, dtype=torch.int64),
                torch.arange(num_tokens, dtype=torch.int64),
                node_id,
                pool_transfers=[
                    PoolTransfer(
                        name=PoolName.INDEXER,
                        host_indices=torch.arange(num_tokens, dtype=torch.int64) + 100,
                        device_indices=torch.arange(num_tokens, dtype=torch.int64)
                        + 200,
                    )
                ],
            )
        )

    def _expected_layers(self):
        expected = []
        for layer in range(self.layer_num):
            expected.extend([(layer, "kv"), (layer, "indexer")])
        return expected

    def test_gate_off_hybrid_burst_runs_inline(self):
        controller = self.make_hybrid_controller(async_enabled=False)
        self.queue_hybrid_burst(controller, node_id=201)
        me = threading.current_thread().name

        controller.start_loading()

        self.assertEqual(
            [(entry[1], entry[3]) for entry in self.entries("load_layer")],
            self._expected_layers(),
        )
        self.assertEqual({entry[2] for entry in self.entries("load_layer")}, {me})
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.node_ids, [201])
        self.assertEqual(
            ack.num_tokens_by_pool,
            {PoolName.KV.value: 2, PoolName.INDEXER.value: 2},
        )

    def test_gate_on_hybrid_burst_runs_on_the_loader_thread(self):
        controller = self.make_hybrid_controller(async_enabled=True)
        self.queue_hybrid_burst(controller, node_id=202)

        producer_id = controller.start_loading()
        self.assertEqual(producer_id, 0)
        controller._drain_load_enqueue()

        self.assertEqual(
            [(entry[1], entry[3]) for entry in self.entries("load_layer")],
            self._expected_layers(),
        )
        self.assertEqual(
            {entry[2] for entry in self.entries("load_layer")}, {LOADER_THREAD_NAME}
        )
        # Both index moves (KV and sidecar) happen behind the start_event wait.
        self.assertEqual(
            [entry[1] for entry in self.entries("move_indices")],
            [LOADER_THREAD_NAME] * 2,
        )
        ack = controller.ack_load_queue[0]
        self.assertEqual(ack.node_ids, [202])
        self.assertEqual(
            ack.num_tokens_by_pool,
            {PoolName.KV.value: 2, PoolName.INDEXER.value: 2},
        )

    def test_failed_hybrid_burst_still_releases_and_acks(self):
        controller = self.make_hybrid_controller(async_enabled=True)
        controller.mem_pool_host.sidecar.fail_at_layer = 1
        self.queue_hybrid_burst(controller, node_id=203)

        with self.assertLogs(cache_controller.logger, level="ERROR"):
            producer_id = controller.start_loading()
            controller._drain_load_enqueue()

        controller.layer_done_counter.set_consumer(producer_id)
        self._run_with_timeout(
            lambda: [
                controller.layer_done_counter.wait_until(layer)
                for layer in range(self.layer_num)
            ],
            "a failed hybrid burst left the consumer blocked",
        )
        self.assertEqual([ack.node_ids for ack in controller.ack_load_queue], [[203]])


class FakeDeviceKVPool:
    device = "cpu"
    layer_num = 3

    def register_layer_transfer_counter(self, counter):
        self.counter = counter


class FakeAllocator:
    def __init__(self):
        self.pool = FakeDeviceKVPool()

    def get_kvcache(self):
        return self.pool


class FakeInitHostPool:
    layout = "page_first_direct"
    layer_num = 3
    entry_map = {}
    entries = None

    def __init__(self):
        self.anchor_entry = SimpleNamespace(name=PoolName.KV, host_pool=self)


class TestActivationSignature(AsyncLoadEnqueueTestBase):
    """The startup log has to describe what the process will actually do."""

    def _build(self, cls, **kwargs):
        controller = cls(
            token_to_kv_pool_allocator=FakeAllocator(),
            mem_pool_host=FakeInitHostPool(),
            page_size=1,
            tp_group=None,
            load_cache_event=threading.Event(),
            **kwargs,
        )
        self.addCleanup(controller.stop_load_enqueue_thread)
        return controller

    def _gate_on(self):
        patcher = mock.patch.object(
            envs.SGLANG_HICACHE_ASYNC_LOAD_ENQUEUE, "get", lambda: True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_enabled_line_names_the_concrete_class(self):
        self._gate_on()
        with self.assertLogs(cache_controller.logger, level="INFO") as logs:
            controller = self._build(HybridCacheController)
        self.assertIn(
            "HiCache async load enqueue: enabled (HybridCacheController)",
            "\n".join(logs.output),
        )
        # And the claim is true: the worker is still there after __init__.
        self.assertTrue(controller.async_load_enqueue)
        self.assertIsNotNone(controller.load_enqueue_thread)

    def test_base_line_names_the_base_class(self):
        self._gate_on()
        with self.assertLogs(cache_controller.logger, level="INFO") as logs:
            self._build(HiCacheController)
        self.assertIn(
            "HiCache async load enqueue: enabled (HiCacheController)",
            "\n".join(logs.output),
        )

    def test_a_transfer_layer_override_keeps_the_handshake(self):
        # The replacement LayerDoneCounter has to carry async_enqueue, or the
        # forward would wait on device events the loader has not recorded yet.
        self._gate_on()
        controller = self._build(HybridCacheController, transfer_layer_num=5)
        self.assertEqual(controller.layer_num, 5)
        for event in controller.layer_done_counter.events:
            self.assertIsNotNone(event._recorded)
            self.assertEqual(len(event._recorded), 5)

    def test_gate_off_says_disabled(self):
        with self.assertLogs(cache_controller.logger, level="INFO") as logs:
            controller = self._build(HybridCacheController)
        self.assertIn(
            "HiCache async load enqueue: disabled (HybridCacheController)",
            "\n".join(logs.output),
        )
        self.assertIsNone(controller.load_enqueue_thread)


if __name__ == "__main__":
    unittest.main()
