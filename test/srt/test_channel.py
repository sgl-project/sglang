"""Unit tests for ``sglang.srt.managers.channel``.

These tests are intentionally pure-Python (no GPU, no model) so they
run on any 3.12+ interpreter, free-threaded or not.
"""

from __future__ import annotations

import asyncio
import queue
import unittest

import zmq

from sglang.srt.managers.channel import (
    AsyncChannelPair,
    AsyncQueueReceiver,
    ChannelHub,
    QueueDealer,
    QueueReceiver,
    QueueSender,
    SyncChannelPair,
)


class TestQueueSender(unittest.TestCase):
    def test_send_pyobj_enqueues_synchronously(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        sender = QueueSender(q)
        sender.send_pyobj({"x": 1})
        self.assertEqual(q.get_nowait(), {"x": 1})

    def test_send_pyobj_return_is_awaitable(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        sender = QueueSender(q)

        async def go():
            await sender.send_pyobj("hello")

        asyncio.run(go())
        self.assertEqual(q.get_nowait(), "hello")


class TestQueueReceiver(unittest.TestCase):
    def test_blocking_recv(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        q.put(42)
        receiver = QueueReceiver(q)
        self.assertEqual(receiver.recv_pyobj(), 42)

    def test_noblock_raises_zmq_again_on_empty(self):
        receiver = QueueReceiver(queue.SimpleQueue())
        with self.assertRaises(zmq.Again) as cm:
            receiver.recv_pyobj(flags=zmq.NOBLOCK)
        # Must be a real zmq.Again, not just any ZMQError — code paths
        # that key off errno==EAGAIN must still work.
        self.assertIsInstance(cm.exception, zmq.ZMQError)
        self.assertEqual(cm.exception.errno, zmq.EAGAIN)

    def test_noblock_returns_value_when_available(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        q.put("hi")
        receiver = QueueReceiver(q)
        self.assertEqual(receiver.recv_pyobj(flags=zmq.NOBLOCK), "hi")


class TestAsyncQueueReceiver(unittest.TestCase):
    def test_async_recv(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        receiver = AsyncQueueReceiver(q)

        async def go():
            q.put("payload")
            return await receiver.recv_pyobj()

        try:
            self.assertEqual(asyncio.run(go()), "payload")
        finally:
            receiver.close()

    def test_close_stops_bridge_thread(self):
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        receiver = AsyncQueueReceiver(q)

        async def warmup():
            # Force bridge thread to spin up.
            q.put("ping")
            return await receiver.recv_pyobj()

        asyncio.run(warmup())
        bridge = receiver._bridge_thread
        self.assertIsNotNone(bridge)
        self.assertTrue(bridge.is_alive())

        receiver.close(timeout=2.0)
        self.assertFalse(bridge.is_alive())

    def test_close_is_idempotent(self):
        receiver = AsyncQueueReceiver(queue.SimpleQueue())
        receiver.close()
        receiver.close()  # must not raise

    def test_bridge_survives_loop_close(self):
        """If the event loop closes mid-flight, the bridge must exit cleanly
        (not raise PytestUnhandledThreadException / leak a thread).

        We trigger this by closing the loop while the bridge is parked on
        ``sync_q.get()``, then pushing one item to wake it.
        """
        q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        receiver = AsyncQueueReceiver(q)

        async def warmup():
            q.put("warmup")
            return await receiver.recv_pyobj()

        asyncio.run(warmup())
        bridge = receiver._bridge_thread
        self.assertIsNotNone(bridge)

        # The loop is closed at the end of asyncio.run; bridge is now
        # parked on sync_q.get(). Push an item — call_soon_threadsafe will
        # raise RuntimeError, and the bridge must swallow it and exit.
        q.put("post-close")
        bridge.join(timeout=2.0)
        self.assertFalse(
            bridge.is_alive(),
            "bridge thread must exit when the asyncio loop is closed",
        )

        receiver.close()


class TestChannelPairs(unittest.TestCase):
    def test_sync_pair_round_trip(self):
        pair = SyncChannelPair()
        pair.sender.send_pyobj({"k": "v"})
        self.assertEqual(pair.receiver.recv_pyobj(), {"k": "v"})

    def test_async_pair_round_trip(self):
        pair = AsyncChannelPair()

        async def go():
            pair.sender.send_pyobj("ok")
            return await pair.receiver.recv_pyobj()

        try:
            self.assertEqual(asyncio.run(go()), "ok")
        finally:
            pair.receiver.close()

    def test_sync_pair_exposes_no_async_receiver(self):
        """1:1 invariant: a SyncChannelPair has exactly one sender and
        one (sync) receiver. There is no async receiver attribute that
        could split items nondeterministically.
        """
        pair = SyncChannelPair()
        self.assertFalse(hasattr(pair, "async_receiver"))
        self.assertIsInstance(pair.receiver, QueueReceiver)

    def test_async_pair_exposes_no_sync_receiver(self):
        pair = AsyncChannelPair()
        self.assertIsInstance(pair.receiver, AsyncQueueReceiver)
        # `receiver` is the only consumer; no sync receiver on the side.
        public_attrs = {a for a in dir(pair) if not a.startswith("_")}
        self.assertEqual(public_attrs, {"sender", "receiver"})


class TestQueueDealer(unittest.TestCase):
    def test_send_is_a_sink(self):
        """The dealer's send side must drop payloads, not enqueue them.

        In threaded mode no peer drives the dealer; if send_pyobj queued
        responses, the queue would grow without bound (slow memory leak).
        """
        recv_q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        dealer = QueueDealer(recv_q)

        # Send 10k payloads — none of them should be observable anywhere.
        for i in range(10_000):
            dealer.send_pyobj({"resp": i})

        # Recv side stays empty — sends did not bleed in either.
        with self.assertRaises(zmq.Again):
            dealer.recv_pyobj(flags=zmq.NOBLOCK)

    def test_recv_reads_recv_queue(self):
        recv_q: "queue.SimpleQueue[object]" = queue.SimpleQueue()
        dealer = QueueDealer(recv_q)
        recv_q.put({"req": 2})
        self.assertEqual(dealer.recv_pyobj(), {"req": 2})

    def test_send_pyobj_return_is_awaitable(self):
        """``await sender.send_pyobj(...)`` must keep working on the sink."""
        dealer = QueueDealer(queue.SimpleQueue())

        async def go():
            await dealer.send_pyobj("dropped")

        asyncio.run(go())  # must not raise


class TestChannelHub(unittest.TestCase):
    def test_endpoints_wired(self):
        hub = ChannelHub()
        try:
            # Each leg of the topology is its own pair (no aliasing).
            self.assertIsInstance(hub.tokenizer_to_scheduler, SyncChannelPair)
            self.assertIsInstance(hub.scheduler_to_detokenizer, SyncChannelPair)
            self.assertIsInstance(hub.detokenizer_to_tokenizer, AsyncChannelPair)
            self.assertIsInstance(hub.rpc_dealer, QueueDealer)

            self.assertIsNot(hub.tokenizer_to_scheduler, hub.scheduler_to_detokenizer)
            self.assertIsNot(hub.tokenizer_to_scheduler, hub.detokenizer_to_tokenizer)
            self.assertIsNot(hub.scheduler_to_detokenizer, hub.detokenizer_to_tokenizer)
        finally:
            hub.close()

    def test_rpc_dealer_is_empty_by_default(self):
        """Threaded mode doesn't drive RPC, so the dealer's recv side must
        raise Again on NOBLOCK — exactly what request_receiver expects.
        """
        hub = ChannelHub()
        try:
            with self.assertRaises(zmq.Again):
                hub.rpc_dealer.recv_pyobj(flags=zmq.NOBLOCK)
        finally:
            hub.close()

    def test_round_trip_tokenizer_to_scheduler(self):
        hub = ChannelHub()
        try:
            hub.tokenizer_to_scheduler.sender.send_pyobj({"req": 1})
            self.assertEqual(
                hub.tokenizer_to_scheduler.receiver.recv_pyobj(),
                {"req": 1},
            )
        finally:
            hub.close()

    def test_round_trip_detokenizer_to_tokenizer(self):
        hub = ChannelHub()

        async def go():
            hub.detokenizer_to_tokenizer.sender.send_pyobj("output-1")
            return await hub.detokenizer_to_tokenizer.receiver.recv_pyobj()

        try:
            self.assertEqual(asyncio.run(go()), "output-1")
        finally:
            hub.close()

    def test_close_is_idempotent(self):
        hub = ChannelHub()
        hub.close()
        hub.close()


class TestEnableThreadedEngineIdempotent(unittest.TestCase):
    """``enable_threaded_engine`` must be safe to call multiple times.

    The function has irreversible side effects on the running process
    (``mp.set_start_method("spawn", force=True)``, mutating
    ``sglang.srt.managers.mm_utils._is_default_tensor_transport``, etc.),
    so calling it for real here would pollute every test that runs
    after this one in the same pytest worker. We patch the side-effect
    targets so the body executes but no global state actually changes.
    """

    def test_idempotent(self):
        try:
            from sglang.srt.entrypoints import engine_threaded
            from sglang.srt.server_args import ServerArgs
        except ImportError as e:
            self.skipTest(f"sglang not importable in this env: {e}")

        from unittest import mock

        # Snapshot anything we are going to touch so we can also assert
        # it was not mutated by the patched calls.
        sa = ServerArgs(model_path="dummy")
        sa.disable_piecewise_cuda_graph = True  # already what we'd force
        engine_threaded._patches_applied = False

        with mock.patch.object(
            engine_threaded.mp, "set_start_method"
        ) as set_method, mock.patch(
            "sglang.srt.managers.mm_utils._is_default_tensor_transport",
            create=True,
            new=False,
        ):
            engine_threaded.enable_threaded_engine(sa)
            self.assertTrue(engine_threaded._patches_applied)
            first_calls = set_method.call_count

            # Second call must be a no-op — no extra mp.set_start_method,
            # no exception, flag still True.
            engine_threaded.enable_threaded_engine(sa)
            self.assertTrue(engine_threaded._patches_applied)
            self.assertEqual(set_method.call_count, first_calls)

        # Reset so this test does not leak state to subsequent tests.
        engine_threaded._patches_applied = False


if __name__ == "__main__":
    unittest.main()
