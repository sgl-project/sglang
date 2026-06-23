"""Regression test for Qwen-VL video preprocessing blocking the event loop.

`preprocess_video` used to be an ``async def`` whose body was fully synchronous
(frame decode/sample, resize, tensor build), so awaiting it ran the CPU-bound
work inline on the asyncio event loop and stalled the API server while a
request's video was processed. The fix keeps the public coroutine API but
offloads the synchronous implementation to a worker thread. See issue #28247.
"""

import asyncio
import inspect
import threading

from sglang.srt.multimodal.processors import qwen_vl
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwenVLPreprocessVideoNonBlocking(CustomTestCase):
    def test_preprocess_video_is_coroutine_function(self):
        # Lock the public API shape: callers (qwen_vl processor, encode_server)
        # await preprocess_video(), so it must stay an async def.
        self.assertTrue(inspect.iscoroutinefunction(qwen_vl.preprocess_video))

    def test_preprocess_video_runs_off_the_event_loop(self):
        loop_thread_id = threading.get_ident()
        captured = {}

        original_impl = qwen_vl._preprocess_video_impl

        def spy_impl(vr, **kwargs):
            captured["thread_id"] = threading.get_ident()
            return ("sentinel", None)

        qwen_vl._preprocess_video_impl = spy_impl
        try:

            async def run():
                return await qwen_vl.preprocess_video("dummy-video")

            result = asyncio.run(run())
        finally:
            qwen_vl._preprocess_video_impl = original_impl

        # The wrapper must delegate to the sync impl...
        self.assertEqual(result, ("sentinel", None))
        self.assertIn("thread_id", captured)
        # ...and run it on a worker thread, not the event-loop thread.
        self.assertNotEqual(captured["thread_id"], loop_thread_id)

    def test_combine_lock_serializes_concurrent_offloads(self):
        # process_and_combine_mm_data is offloaded to a worker thread but
        # touches the shared, non-reentrant HF tokenizer; _combine_lock() must
        # serialize concurrent offloads so they never run in parallel (the Rust
        # fast tokenizer raises "Already borrowed" otherwise). See issue #28247.
        import time

        proc = object.__new__(qwen_vl.QwenVLImageProcessor)
        proc._mm_combine_lock = None

        state = {"inside": 0, "max_inside": 0}

        def fake_combine():
            state["inside"] += 1
            state["max_inside"] = max(state["max_inside"], state["inside"])
            time.sleep(0.01)
            state["inside"] -= 1
            return "ok"

        async def one_request():
            async with proc._combine_lock():
                return await asyncio.to_thread(fake_combine)

        async def run():
            return await asyncio.gather(*(one_request() for _ in range(16)))

        results = asyncio.run(run())
        self.assertEqual(results, ["ok"] * 16)
        # Never more than one offloaded combine ran at a time.
        self.assertEqual(state["max_inside"], 1)
        # Lazy init produced a single shared asyncio.Lock.
        self.assertIsInstance(proc._mm_combine_lock, asyncio.Lock)


if __name__ == "__main__":
    import unittest

    unittest.main()
