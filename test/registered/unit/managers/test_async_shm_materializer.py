from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components.request_receiver import (
    SchedulerRequestReceiver,
)
from sglang.srt.models.unlimited_ocr_v2_support.async_shm import (
    AsyncShmMaterializer,
    ShmPointerMMData,
    create_async_shm_materializer,
    estimate_shm_bytes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakePointer:
    def __init__(self, name, shape=(4,), itemsize=4):
        self.shm_name = name
        self.shape = shape
        self.dtype = SimpleNamespace(itemsize=itemsize)


class AsyncShmMaterializerTest(CustomTestCase):
    def _value(self, name):
        item = SimpleNamespace(
            feature=FakePointer(name),
            precomputed_embeddings=None,
            model_specific_data={},
        )
        return SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=[item]))

    def _coordinator(self, **overrides):
        self.enterContext(
            envs.SGLANG_UOCR_ASYNC_SHM_MAX_MB.override(overrides.get("mb", 1))
        )
        coordinator = AsyncShmMaterializer(
            max_recv_per_poll=2,
            discard=Mock(),
            max_items=overrides.get("items", 4),
            workers=2,
        )
        self.addCleanup(coordinator.shutdown)
        return coordinator

    def test_fifo_holds_later_completion_and_control(self):
        coordinator = self._coordinator()
        first = self._value("first")
        second = self._value("second")
        control = object()
        first_started = threading.Event()
        release_first = threading.Event()
        second_done = threading.Event()

        def finalize(values):
            if values[0] is first:
                first_started.set()
                release_first.wait(2)
            elif values[0] is second:
                second_done.set()

        coordinator.reserve_receive(extra_items=1)
        coordinator.submit(
            [first, second, control],
            finalize=finalize,
            has_shm=lambda values: isinstance(values[0], SimpleNamespace),
        )
        self.assertTrue(first_started.wait(2))
        self.assertTrue(second_done.wait(2))
        self.assertEqual(coordinator.pop_ready(), [])
        release_first.set()
        coordinator.entries[0].future.result(2)
        self.assertEqual(coordinator.pop_ready(), [first, second, control])

    def test_byte_fallback_and_receive_admission(self):
        coordinator = self._coordinator(items=4)
        value = self._value("large")
        value.mm_inputs.mm_items[0].feature.shape = (300_000,)
        finalize = Mock()
        coordinator.reserve_receive()
        coordinator.submit([value], finalize=finalize, has_shm=lambda _values: True)
        finalize.assert_called_once_with([value])
        self.assertEqual(coordinator.pop_ready(), [value])
        coordinator.pop_ready()
        coordinator.reserve_receive()
        coordinator.submit([object()], finalize=Mock(), has_shm=lambda _values: False)
        self.assertFalse(coordinator.reserve_receive(extra_items=2))
        self.assertTrue(coordinator.reserve_receive(extra_items=1))
        coordinator.cancel_receive()

    def test_wait_for_progress_tracks_only_the_fifo_head(self):
        coordinator = self._coordinator()
        first = self._value("first")
        second = self._value("second")
        first_started = threading.Event()
        release_first = threading.Event()
        second_done = threading.Event()

        def finalize(values):
            if values[0] is first:
                first_started.set()
                release_first.wait(2)
            else:
                second_done.set()

        coordinator.reserve_receive()
        coordinator.submit(
            [first, second],
            finalize=finalize,
            has_shm=lambda _values: True,
        )
        self.assertTrue(first_started.wait(2))
        self.assertTrue(second_done.wait(2))

        with patch(
            "sglang.srt.models.unlimited_ocr_v2_support.async_shm.wait"
        ) as wait_mock:
            coordinator.wait_for_progress()
        wait_mock.assert_called_once_with(
            (coordinator.entries[0].future,), timeout=0.001
        )
        self.assertEqual(coordinator.pop_ready(), [])

        release_first.set()
        coordinator.entries[0].future.result(2)
        coordinator.wait_for_progress()
        self.assertEqual(coordinator.pop_ready(), [first, second])

    def test_unexpected_failure_discards_at_fifo_head(self):
        coordinator = self._coordinator()
        value = self._value("broken")
        coordinator.reserve_receive()
        coordinator.submit(
            [value],
            finalize=Mock(side_effect=ValueError("broken")),
            has_shm=lambda _values: True,
        )
        deadline = time.monotonic() + 2
        while not coordinator.entries[0].future.done():
            self.assertLess(time.monotonic(), deadline)
            time.sleep(0.001)
        with self.assertRaisesRegex(ValueError, "broken"):
            coordinator.pop_ready()
        coordinator.discard.assert_called_once_with(value)

    def test_shutdown_waits_and_cancels_queued_work(self):
        self.enterContext(envs.SGLANG_UOCR_ASYNC_SHM_MAX_MB.override(1))
        coordinator = AsyncShmMaterializer(
            max_recv_per_poll=2, discard=Mock(), max_items=4, workers=1
        )
        running = self._value("running")
        queued = self._value("queued")
        started = threading.Event()
        release = threading.Event()
        coordinator.reserve_receive()

        def finalize(values):
            if values[0] is running:
                started.set()
                release.wait(2)

        coordinator.submit(
            [running, queued], finalize=finalize, has_shm=lambda _values: True
        )
        self.assertTrue(started.wait(2))
        closer = threading.Thread(target=coordinator.shutdown)
        closer.start()
        self.assertTrue(closer.is_alive())
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        coordinator.discard.assert_called_once_with(queued)
        self.assertEqual(coordinator.stats().pending_entries, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            coordinator.submit(
                [object()], finalize=Mock(), has_shm=lambda _values: False
            )

    def test_supported_configuration_and_fallbacks(self):
        kwargs = dict(
            max_recv_per_poll=2,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            nnodes=1,
            mm_feature_transport="cpu",
            scripted_runtime=False,
            rust_server=False,
            is_unlimited_ocr=True,
        )
        coordinator = create_async_shm_materializer(**kwargs)
        self.assertIsInstance(coordinator, AsyncShmMaterializer)
        self.assertEqual(coordinator.max_recv_per_poll, 2)
        coordinator.shutdown()

        with envs.SGLANG_ENABLE_UOCR_ASYNC_SHM_MATERIALIZATION.override(False):
            self.assertIsNone(create_async_shm_materializer(**kwargs))

        unsupported = (
            {"is_unlimited_ocr": False},
            {"max_recv_per_poll": 0},
            {"max_recv_per_poll": 65},
            {"tp_size": 2},
            {"pp_size": 2},
            {"dp_size": 2},
            {"attn_cp_size": 2},
            {"nnodes": 2},
            {"mm_feature_transport": None},
            {"scripted_runtime": True},
            {"rust_server": True},
        )
        for override in unsupported:
            with self.subTest(override=override):
                self.assertIsNone(
                    create_async_shm_materializer(**{**kwargs, **override})
                )

    def test_unbounded_scheduler_polls_get_a_bounded_receive_batch(self):
        """The scheduler default (-1) receives without limit, which no queue
        reservation can cover; the materializer must bound its own polls and
        leave room for local aborts beside entries still pending."""
        coordinator = create_async_shm_materializer(
            max_recv_per_poll=-1,
            tp_size=1,
            pp_size=1,
            dp_size=1,
            attn_cp_size=1,
            nnodes=1,
            mm_feature_transport="cpu",
            scripted_runtime=False,
            rust_server=False,
            is_unlimited_ocr=True,
        )
        self.addCleanup(coordinator.shutdown)
        self.assertGreater(coordinator.max_recv_per_poll, 0)
        self.assertLess(coordinator.max_recv_per_poll, coordinator.max_items)
        self.assertTrue(coordinator.reserve_receive(extra_items=1))
        coordinator.cancel_receive()

        # The receiver stops pulling at the materializer's batch, and only then.
        limit_reached = SchedulerRequestReceiver.recv_limit_reached
        receiver = SimpleNamespace(
            max_recv_per_poll=-1, async_shm_materializer=coordinator
        )
        limit = coordinator.max_recv_per_poll
        self.assertFalse(limit_reached(receiver, limit - 1))
        self.assertTrue(limit_reached(receiver, limit))
        receiver.async_shm_materializer = None
        self.assertFalse(limit_reached(receiver, 10_000))

    def test_estimate_deduplicates_pointer_name(self):
        pointer = object.__new__(ShmPointerMMData)
        pointer.shm_name = "same"
        pointer.shape = (4,)
        pointer.dtype = SimpleNamespace(itemsize=4)
        pointer._shm_handle = None
        pointer.tensor = None
        pointer._materialization_error = None
        pointer.close_and_unlink = Mock()
        item = SimpleNamespace(
            feature=pointer,
            precomputed_embeddings=pointer,
            model_specific_data={},
        )
        value = SimpleNamespace(mm_inputs=SimpleNamespace(mm_items=[item]))
        with (
            patch(
                "sglang.srt.models.unlimited_ocr_v2_support.async_shm.TokenizedGenerateReqInput",
                SimpleNamespace,
            ),
            patch(
                "sglang.srt.models.unlimited_ocr_v2_support.async_shm.MultimodalProcessorOutput",
                SimpleNamespace,
            ),
        ):
            self.assertEqual(estimate_shm_bytes(value), 16)
        pointer._materialization_error = "test pointer"


if __name__ == "__main__":
    import unittest

    unittest.main()
