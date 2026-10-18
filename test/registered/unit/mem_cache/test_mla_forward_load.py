import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import cache_controller as cc
from sglang.srt.mem_cache.l2_transfer import TransferCompletion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class Event:
    def __init__(self, **kwargs):
        self.ready = True

    def record(self):
        self.ready = False

    def query(self):
        return self.ready

    def wait(self, stream):
        stream.wait_event(self)

    def synchronize(self):
        self.ready = True


class Stream:
    def __init__(self):
        self.waits = []

    def wait_event(self, event):
        self.waits.append(event)

    def wait_stream(self, stream):
        self.waits.append(stream)

    def synchronize(self):
        raise AssertionError("Forward mode must not drain the load stream")


class Device:
    Event = Event
    active = None

    @classmethod
    @contextmanager
    def stream(cls, stream):
        previous, cls.active = cls.active, stream
        try:
            yield
        finally:
            cls.active = previous

    @classmethod
    def current_stream(cls):
        return cls.active

    @staticmethod
    def is_current_stream_capturing():
        return False


class TestMLAForwardLoad(unittest.TestCase):
    @contextmanager
    def controller(self, is_source):
        operations = []
        load, forward, fence = Stream(), Stream(), Stream()
        Device.active = forward
        args = SimpleNamespace(
            disable_overlap_schedule=False,
            disaggregation_mode="null",
            pp_size=1,
            enable_unified_memory=False,
        )
        controller = cc.HiCacheController.__new__(cc.HiCacheController)
        controller.transfer_layer_id_max = 3
        controller.enable_storage = False
        controller.load_fence_stream = fence
        controller.ack_load_queue = []
        controller.mem_pool_device_allocator = Mock()
        controller._prepare_mla_source_load = Mock(return_value=["target"])
        controller._prepare_draft_load = Mock(return_value=["draft"])
        controller._mla_broadcast_transfer = lambda op: op
        controller._num_tokens_by_pool = lambda op: {}
        controller._mla_transfer_num_bytes = lambda op: 128
        controller._load_mla_transfers_layer = lambda transfers, k: operations.append(
            ("h2d", tuple(transfers), k, Device.active)
        )
        controller.mla_dedup = SimpleNamespace(
            broadcaster=SimpleNamespace(
                is_src=is_source,
                layer_num=3,
                prepare_broadcast=lambda indices, stream: (indices, None),
                broadcast_loaded_layer=lambda k, plan: operations.append(
                    ("broadcast", (), k, Device.active)
                ),
            )
        )

        @contextmanager
        def submission(transfers, *, start_event):
            self.assertIs(Device.active, load)
            completion = TransferCompletion(Event(), Event(), False)
            yield transfers, completion
            completion.finish_event.record()

        controller.l2_transfer_engine = SimpleNamespace(
            host_to_device_stream=load,
            host_to_device_submission=submission,
            resolve_device_indices=lambda op: op.device_indices,
        )
        with (
            patch.object(cc, "device_module", Device),
            patch("sglang.srt.runtime_context.get_server_args", return_value=args),
            patch.dict("os.environ", {"SGLANG_MLA_DEDUP_LOAD_MODE": "forward"}),
        ):
            controller.layer_done_counter = cc.LayerDoneCounter(3)
            yield controller, operations, load, forward, fence, args

    def test_source_and_peer_prefetch_then_broadcast(self):
        for source in (True, False):
            with self.subTest(source=source), self.controller(source) as setup:
                controller, operations, load, forward, fence, _ = setup
                counter = controller.layer_done_counter
                slot = counter.update_producer()
                indices = torch.arange(4)
                controller._start_loading_mla(
                    slot, cc.CacheOperation(indices, indices, 123)
                )
                state = controller._mla_forward_loads[slot]
                self.assertIn(fence, load.waits)
                self.assertFalse(counter.events[slot].finish_event.query())
                self.assertEqual(controller.ack_load_queue, [])
                controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_not_called()
                self.assertTrue(all(kind == "h2d" for kind, *_ in operations))
                self.assertEqual(
                    sum(ts == ("target",) for _, ts, _, _ in operations),
                    3 if source else 0,
                )
                self.assertEqual(sum(ts == ("draft",) for _, ts, _, _ in operations), 3)

                counter.set_consumer(slot)
                handle = counter.bind_forward_load()
                handle.before_layer(0)
                counter.wait_until(0)
                self.assertEqual(sum(kind == "broadcast" for kind, *_ in operations), 1)
                self.assertEqual(controller.ack_load_queue, [])
                handle.before_layer(2)
                self.assertTrue(
                    all(
                        s is forward
                        for kind, _, _, s in operations
                        if kind == "broadcast"
                    )
                )
                self.assertTrue(
                    all(event in forward.waits for event in state["copy_ready"])
                )
                ack = controller.ack_load_queue[0]
                self.assertIs(ack.finish_event, state["finish"])
                self.assertIsNot(ack.finish_event, state["completion"].finish_event)
                self.assertFalse(ack.finish_event.query())
                self.assertEqual(ack.node_ids, [123])
                controller.mem_pool_device_allocator.set_hicache_transfer_done_event.assert_called_once_with(
                    (id(controller), "load"), ack.finish_event
                )
                self.assertIsNone(counter.bind_forward_load())

    def test_queued_batches_order_and_recycled_handles(self):
        with self.controller(True) as setup:
            controller, _, _, _, _, _ = setup
            counter = controller.layer_done_counter
            indices = torch.arange(4)
            handles = []
            for node in (10, 11):
                slot = counter.update_producer()
                controller._start_loading_mla(
                    slot, cc.CacheOperation(indices, indices, node)
                )
                counter.set_consumer(slot)
                handles.append(counter.bind_forward_load())
            with self.assertRaisesRegex(RuntimeError, "submission order"):
                handles[1].before_layer(0)
            with self.assertRaisesRegex(RuntimeError, "unfinished"):
                counter.reset()
            handles[0].before_layer(2)
            self.assertEqual(counter.consumer_index, 1)
            handles[1].before_layer(2)
            self.assertEqual(
                [a.node_ids for a in controller.ack_load_queue], [[10], [11]]
            )
            counter.producer_index = 2
            with self.assertRaisesRegex(AssertionError, "ready before being reused"):
                counter.update_producer()
            self.assertEqual(counter.producer_index, 2)
            for event in counter.events[0].load_events:
                event.ready = True
            counter.update_producer()
            with self.assertRaisesRegex(RuntimeError, "recycled slot"):
                handles[0].before_layer(0)

    def test_unsupported_paths_fail_before_submission(self):
        with self.controller(True) as setup:
            controller, operations, _, _, _, args = setup
            indices = torch.arange(4)
            op = cc.CacheOperation(indices, indices, 10)
            slot = controller.layer_done_counter.update_producer()
            args.enable_unified_memory = True
            with self.assertRaisesRegex(ValueError, "movable device pages"):
                controller._start_loading_mla(slot, op)
            args.enable_unified_memory = False
            for name, value in (("pp_size", 2), ("disaggregation_mode", "prefill")):
                previous = getattr(args, name)
                setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, "non-PD, PP1, L2-only"):
                    controller._start_loading_mla(slot, op)
                setattr(args, name, previous)
            controller.enable_storage = True
            with self.assertRaisesRegex(ValueError, "L2-only"):
                controller._start_loading_mla(slot, op)
            self.assertEqual(operations, [])
            self.assertFalse(controller.layer_done_counter.events[slot].forward_pending)


if __name__ == "__main__":
    unittest.main()
