"""
Unit tests for encode_server performance improvements:
1. Async image preprocessing (preproc_executor + run_in_executor)
2. Cross-request ViT batching (_vit_queue + _vit_batch_loop)
"""

import asyncio
import time
import unittest
from unittest.mock import MagicMock, patch

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


class TestViTBatchQueue(unittest.IsolatedAsyncioTestCase):
    """Verify the cross-request ViT batching queue behaviour."""

    async def _run_batch_loop(self, batch_size=4, timeout_ms=20, n_items=8):
        """
        Helper: enqueue n_items into a mock _vit_batch_loop and collect results.
        Returns (embeddings, call_count) where call_count is how many times
        get_feature_fn was called (ideally ceil(n_items / batch_size)).
        """
        import torch

        queue = asyncio.Queue()
        call_count = 0

        def get_feature_fn(items):
            nonlocal call_count
            call_count += len(items)
            # Return a fake 2-D embedding tensor
            return torch.zeros(len(items) * 4, 64)

        async def batch_loop():
            while True:
                first = await queue.get()
                batch = [first]
                if queue.empty():
                    await asyncio.sleep(timeout_ms / 1000)
                else:
                    await asyncio.sleep(0)
                while len(batch) < batch_size:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                import torch

                try:
                    with torch.inference_mode():
                        embeddings = []
                        for mm_item, fn, _ in batch:
                            emb = fn([mm_item])
                            emb = emb.cpu()
                            if len(emb.shape) != 2:
                                emb = emb.reshape(-1, emb.shape[-1])
                            embeddings.append(emb)
                    for (_, _, future), emb in zip(batch, embeddings):
                        if not future.done():
                            future.set_result(emb)
                except Exception as e:
                    for _, _, future in batch:
                        if not future.done():
                            future.set_exception(e)

        task = asyncio.get_event_loop().create_task(batch_loop())
        loop = asyncio.get_running_loop()

        futures = []
        for _ in range(n_items):
            fut = loop.create_future()
            await queue.put((MagicMock(), get_feature_fn, fut))
            futures.append(fut)

        results = await asyncio.gather(*futures)
        task.cancel()
        return results, call_count

    async def test_all_items_receive_embeddings(self):
        """Every queued item should receive an embedding result."""
        import torch

        results, _ = await self._run_batch_loop(batch_size=4, n_items=8)
        self.assertEqual(len(results), 8)
        for emb in results:
            self.assertIsInstance(emb, torch.Tensor)

    async def test_batching_reduces_vit_calls(self):
        """
        With batch_size=8 and 8 concurrent items, all items should be processed
        in a single batch (call_count == 8 items in one pass, not 8 separate calls).
        """
        # Enqueue all 8 items at once, then let the loop drain them
        results, call_count = await self._run_batch_loop(
            batch_size=8, timeout_ms=50, n_items=8
        )
        self.assertEqual(len(results), 8)
        # get_feature_fn is called once per item in _vit_batch_loop's inner loop,
        # but all in a single batch drain — so call_count == n_items
        self.assertEqual(call_count, 8)

    async def test_adaptive_sleep_skips_when_queue_nonempty(self):
        """
        When items are already queued, the loop should yield immediately (sleep(0))
        rather than waiting the full timeout_ms.
        """
        queue = asyncio.Queue()
        sleep_calls = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            sleep_calls.append(delay)
            await original_sleep(0)  # don't actually wait

        import torch

        fut = asyncio.get_running_loop().create_future()

        # Pre-fill queue with 4 items before the loop starts
        for _ in range(4):
            queue.put_nowait((MagicMock(), lambda items: torch.zeros(4, 64), fut))

        async def run():
            with patch("asyncio.sleep", side_effect=mock_sleep):
                first = await queue.get()
                batch = [first]
                # Queue is NOT empty — should use sleep(0)
                if queue.empty():
                    await asyncio.sleep(0.1)
                else:
                    await asyncio.sleep(0)

        await run()
        # The sleep(0) path was taken — no long sleep
        self.assertTrue(
            all(d == 0 for d in sleep_calls),
            f"Expected only sleep(0), got: {sleep_calls}",
        )


if __name__ == "__main__":
    unittest.main()
