"""DSv41 native work must outlive cancellation of its preprocessing waiter."""

import asyncio
import contextvars
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import torch

import sglang.srt.multimodal.processors.deepseek_v41 as dsv41
from sglang.srt.managers.multimodal_preprocessing_admission import (
    MultimodalPreprocessingAdmission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_TIMEOUT = 5
_REQUEST_CONTEXT = contextvars.ContextVar("dsv41_admission_test_context", default=None)


class _NativeFailure(BaseException):
    pass


class TestDeepseekV41PreprocessingAdmission(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.loop.set_default_executor(self.executor)
        self.release_events = []

    async def asyncTearDown(self):
        for event in self.release_events:
            event.set()
        # Drain the same executor, including wrappers whose asyncio waiter exited.
        await self._drain_executor()

    async def _drain_executor(self):
        await asyncio.wait_for(
            self.loop.run_in_executor(self.executor, lambda: None), _TIMEOUT
        )

    def _release_event(self):
        event = threading.Event()
        self.release_events.append(event)
        return event

    @contextmanager
    def _processor(self, backend, native):
        # Skip model/config construction, but execute the real processor entry,
        # output containers, thread bridge, and admission implementation.
        processor = object.__new__(dsv41.DeepseekV41ImageProcessor)
        processor.preprocess_backend = backend
        processor.cpu_patchify = native
        processor.hf_config = SimpleNamespace()
        processor.image_token_id = 42
        processor.mm_tokens = object()
        processor.mm_feature_transport = "cpu"
        image = object()
        processor.load_mm_data = AsyncMock(return_value=SimpleNamespace(images=[image]))
        # Transport/SHM is outside the native preparation lifetime under test.
        processor._prepare_mm_items_for_transport = lambda items: items
        with patch.object(dsv41, "prepare_image_gpu", native):
            yield processor, image

    def _prepared_image(self, backend):
        patches = torch.zeros((1, 3), dtype=torch.bfloat16)
        if backend == "gpu":
            plan = {"height": 16, "width": 16, "patch_size": 16}
            return patches, (patches, plan, 1, 1)
        return patches, (patches, 1, 1, 1, 1)

    async def _process(self, processor):
        return await processor.process_mm_data_async(
            image_data=["fixture"], input_text=[7, 42, 9], request_obj=None
        )

    async def _owned_process(self, admission, processor):
        lease = admission.acquire(1)
        try:
            with lease.activate():
                return await self._process(processor)
        finally:
            lease.release()

    def _assert_reusable(self, admission):
        self.assertEqual(admission.inflight_items, 0)
        next_lease = admission.acquire(1)
        next_lease.release()
        self.assertEqual(admission.inflight_items, 0)

    async def test_completed_prepare_preserves_result_context_and_owner_reservation(
        self,
    ):
        for backend in ("cpu", "gpu"):
            with self.subTest(backend=backend):
                admission = MultimodalPreprocessingAdmission(1)
                lease = admission.acquire(1)
                patches, prepared = self._prepared_image(backend)
                seen = []

                def native(image, config, *, seen=seen, prepared=prepared):
                    seen.append(
                        (image, config, _REQUEST_CONTEXT.get(), threading.get_ident())
                    )
                    return prepared

                token = _REQUEST_CONTEXT.set("request context")
                try:
                    with self._processor(backend, native) as (processor, image):
                        with lease.activate():
                            output = await self._process(processor)
                        self.assertEqual(
                            seen[0][:3], (image, processor.hf_config, "request context")
                        )
                        self.assertNotEqual(seen[0][3], threading.get_ident())
                        self.assertIs(output.mm_items[0].feature, patches)
                        self.assertEqual(output.input_ids[0], 7)
                        self.assertEqual(output.input_ids[-1], 9)
                        self.assertEqual(
                            output.mm_items[0].model_specific_data["n_vit_h"], 1
                        )
                        self.assertEqual(
                            dsv41.GPU_PLAN_KEY
                            in output.mm_items[0].model_specific_data,
                            backend == "gpu",
                        )
                    # Native completion alone cannot release a still-owned lease.
                    self.assertEqual(admission.inflight_items, 1)
                    self.assertIsNone(admission.try_acquire(1))
                finally:
                    _REQUEST_CONTEXT.reset(token)
                    lease.release()
                self._assert_reusable(admission)

    async def test_running_prepare_cancel_retains_capacity_until_native_finishes(self):
        for backend in ("cpu", "gpu"):
            with self.subTest(backend=backend):
                admission = MultimodalPreprocessingAdmission(1)
                started = asyncio.Event()
                release = self._release_event()
                _, prepared = self._prepared_image(backend)

                def native(
                    image,
                    config,
                    *,
                    started=started,
                    release=release,
                    prepared=prepared,
                ):
                    self.loop.call_soon_threadsafe(started.set)
                    if not release.wait(_TIMEOUT):
                        raise TimeoutError("test did not release native preparation")
                    return prepared

                with self._processor(backend, native) as (processor, _):
                    task = asyncio.create_task(
                        self._owned_process(admission, processor)
                    )
                    try:
                        await asyncio.wait_for(started.wait(), _TIMEOUT)
                        task.cancel()
                        with self.assertRaises(asyncio.CancelledError):
                            await task
                        # Owner has exited; the real native body still owns capacity.
                        self.assertEqual(admission.inflight_items, 1)
                        self.assertIsNone(admission.try_acquire(1))
                    finally:
                        release.set()
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        await self._drain_executor()
                self._assert_reusable(admission)

    async def test_cancelled_queued_wrapper_skips_native_even_if_it_runs_later(self):
        admission = MultimodalPreprocessingAdmission(1)
        blocker_started = asyncio.Event()
        unblock_executor = self._release_event()
        submitted = asyncio.Event()
        wrapper_ran = threading.Event()
        native_ran = threading.Event()
        pending_threads = []

        def occupy_worker():
            self.loop.call_soon_threadsafe(blocker_started.set)
            if not unblock_executor.wait(_TIMEOUT):
                raise TimeoutError("test did not release executor")

        blocker = self.executor.submit(occupy_worker)
        await asyncio.wait_for(blocker_started.wait(), _TIMEOUT)
        original_to_thread = asyncio.to_thread
        original_submit = self.executor.submit

        def observe_submit(*args, **kwargs):
            future = original_submit(*args, **kwargs)
            submitted.set()
            return future

        async def delayed_to_thread(func, *args, **kwargs):
            def late_wrapper():
                wrapper_ran.set()
                return func(*args, **kwargs)

            thread_task = asyncio.create_task(original_to_thread(late_wrapper))
            pending_threads.append(thread_task)
            # Force the race where queued executor work survives its waiter.
            # Shield exists only in this fixture, never in the implementation.
            return await asyncio.shield(thread_task)

        def native(image, config):
            native_ran.set()
            return self._prepared_image("cpu")[1]

        with (
            self._processor("cpu", native) as (processor, _),
            patch.object(self.executor, "submit", side_effect=observe_submit),
            patch.object(asyncio, "to_thread", delayed_to_thread),
        ):
            task = asyncio.create_task(self._owned_process(admission, processor))
            try:
                await asyncio.wait_for(submitted.wait(), _TIMEOUT)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self._assert_reusable(admission)
            finally:
                unblock_executor.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                await asyncio.wait_for(
                    asyncio.gather(*pending_threads, return_exceptions=True), _TIMEOUT
                )
                await asyncio.wrap_future(blocker)
        self.assertTrue(wrapper_ran.is_set())
        self.assertFalse(native_ran.is_set())
        self._assert_reusable(admission)

    async def test_native_exception_propagates_and_releases_capacity(self):
        for error in (ValueError("invalid image"), _NativeFailure("native failure")):
            with self.subTest(exception=type(error).__name__):
                admission = MultimodalPreprocessingAdmission(1)

                def native(image, config, *, error=error):
                    raise error

                with (
                    self._processor("cpu", native) as (processor, _),
                    self.assertRaises(type(error)) as raised,
                ):
                    await self._owned_process(admission, processor)
                self.assertIs(raised.exception, error)
                self._assert_reusable(admission)

    async def test_submit_failure_does_not_leave_a_tracked_reservation(self):
        admission = MultimodalPreprocessingAdmission(1)
        error = RuntimeError("executor unavailable")
        native_ran = threading.Event()

        def native(image, config):
            native_ran.set()

        with (
            self._processor("cpu", native) as (processor, _),
            patch.object(self.executor, "submit", side_effect=error),
            self.assertRaises(RuntimeError) as raised,
        ):
            await self._owned_process(admission, processor)
        self.assertIs(raised.exception, error)
        self.assertFalse(native_ran.is_set())
        self._assert_reusable(admission)


if __name__ == "__main__":
    unittest.main()
