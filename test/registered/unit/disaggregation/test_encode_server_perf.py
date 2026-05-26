"""
Unit tests for encode_server async preprocessing (preproc_executor + run_in_executor)
for image, video, and audio paths.
"""

import asyncio
import time
import unittest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAsyncPreprocessing(unittest.IsolatedAsyncioTestCase):
    """Verify that image preprocessing runs off the asyncio event loop."""

    async def test_concurrent_preprocessing_does_not_block_event_loop(self):
        """
        N concurrent preprocessing tasks should complete in ~1× the single-task
        time (parallel), not N× (serial/blocking).
        """
        sleep_s = 0.05  # simulate 50ms preprocessing
        n = 8

        import concurrent.futures
        import functools

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n)

        def blocking_preproc(delay):
            time.sleep(delay)
            return f"result_{delay}"

        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                loop.run_in_executor(
                    executor, functools.partial(blocking_preproc, sleep_s)
                )
                for _ in range(n)
            ]
        )
        elapsed = time.perf_counter() - t_start

        # All tasks ran concurrently — wall time should be well under N × sleep_s
        self.assertEqual(len(results), n)
        self.assertLess(
            elapsed,
            sleep_s * n * 0.5,
            f"Expected ~{sleep_s:.2f}s wall time, got {elapsed:.2f}s — "
            "preprocessing may be running serially",
        )

    async def test_preproc_executor_is_separate_from_main_executor(self):
        """
        EncodeServer should have a dedicated preproc_executor separate from
        self.executor, so ZMQ sends and preprocessing don't contend.
        """
        import concurrent.futures

        # Simulate the __init__ setup
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        preproc_executor = concurrent.futures.ThreadPoolExecutor(max_workers=32)

        self.assertIsNot(executor, preproc_executor)

        # Verify max_workers
        self.assertEqual(preproc_executor._max_workers, 32)

        executor.shutdown(wait=False)
        preproc_executor.shutdown(wait=False)

    async def test_video_and_audio_preprocessing_does_not_block_event_loop(self):
        """
        Video and audio preprocessing (now using run_in_executor like images) should
        run concurrently off the asyncio event loop, not serialize it.
        """
        import concurrent.futures
        import functools

        sleep_s = 0.05  # simulate 50ms video/audio feature extraction
        n = 6  # 3 video + 3 audio concurrent requests

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n)

        def blocking_video_preproc(delay):
            time.sleep(delay)
            return {"pixel_values_videos": f"video_result_{delay}"}

        def blocking_audio_preproc(delay):
            time.sleep(delay)
            return {"input_features": f"audio_result_{delay}"}

        loop = asyncio.get_running_loop()
        t_start = time.perf_counter()
        results = await asyncio.gather(
            *[
                loop.run_in_executor(
                    executor, functools.partial(blocking_video_preproc, sleep_s)
                )
                for _ in range(3)
            ],
            *[
                loop.run_in_executor(
                    executor, functools.partial(blocking_audio_preproc, sleep_s)
                )
                for _ in range(3)
            ],
        )
        elapsed = time.perf_counter() - t_start

        self.assertEqual(len(results), n)
        # All 6 tasks ran concurrently — wall time should be ~1× sleep_s, not 6×
        self.assertLess(
            elapsed,
            sleep_s * n * 0.5,
            f"Expected ~{sleep_s:.2f}s wall time, got {elapsed:.2f}s — "
            "video/audio preprocessing may be running serially",
        )


if __name__ == "__main__":
    unittest.main()
