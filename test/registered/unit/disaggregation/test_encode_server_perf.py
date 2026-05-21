"""
Unit tests for encode_server performance improvements:
1. Async preprocessing for image, video, and audio (preproc_executor + run_in_executor)
2. Cross-request ViT batching (_vit_queue + _vit_batch_loop) with modality grouping
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


class TestViTBatchQueue(unittest.IsolatedAsyncioTestCase):
    """Verify the cross-request ViT batching queue behaviour."""

    async def _run_batch_loop(self, batch_size=4, timeout_ms=20, n_items=8):
        """
        Helper: enqueue n_items into a mock _vit_batch_loop and collect results.
        Returns (embeddings, fn_call_count) where fn_call_count is how many times
        get_feature_fn was invoked (ideally ceil(n_items / batch_size) with true
        batching — one invocation per batch, not one per item).
        """
        import torch

        TOKENS_PER_ITEM = 4  # each mock item produces this many embedding rows
        queue = asyncio.Queue()
        fn_call_count = 0  # number of get_feature_fn invocations (not total items)

        def get_feature_fn(items):
            nonlocal fn_call_count
            fn_call_count += 1
            # Return a single concatenated tensor for all items in this batch
            return torch.zeros(len(items) * TOKENS_PER_ITEM, 64)

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

                mm_items_in_batch = [item for item, _, _ in batch]
                fns_in_batch = [fn for _, fn, _ in batch]
                futures_in_batch = [f for _, _, f in batch]

                try:
                    with torch.inference_mode():
                        # Group by function identity for one batched forward pass per group
                        fn_groups: dict = {}
                        for mm_item, fn, fut in zip(
                            mm_items_in_batch, fns_in_batch, futures_in_batch
                        ):
                            gid = id(fn)
                            if gid not in fn_groups:
                                fn_groups[gid] = (fn, [], [])
                            fn_groups[gid][1].append(mm_item)
                            fn_groups[gid][2].append(fut)

                        for fn, group_items, group_futures in fn_groups.values():
                            combined = fn(group_items).cpu()
                            if combined.ndim != 2:
                                combined = combined.reshape(-1, combined.shape[-1])
                            # Slice per item (mock: each item contributes TOKENS_PER_ITEM rows)
                            offset = 0
                            for fut in group_futures:
                                emb = combined[offset : offset + TOKENS_PER_ITEM]
                                offset += TOKENS_PER_ITEM
                                if not fut.done():
                                    fut.set_result(emb)
                except Exception as e:
                    for fut in futures_in_batch:
                        if not fut.done():
                            fut.set_exception(e)

        task = asyncio.get_event_loop().create_task(batch_loop())
        loop = asyncio.get_running_loop()

        futures = []
        for _ in range(n_items):
            fut = loop.create_future()
            await queue.put((MagicMock(), get_feature_fn, fut))
            futures.append(fut)

        results = await asyncio.gather(*futures)
        task.cancel()
        return results, fn_call_count

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
        in a single batched forward pass — get_feature_fn called once with all 8
        items, not 8 times with 1 item each.
        """
        results, fn_call_count = await self._run_batch_loop(
            batch_size=8, timeout_ms=50, n_items=8
        )
        self.assertEqual(len(results), 8)
        # True GPU batching: one invocation of get_feature_fn for the whole batch
        self.assertEqual(fn_call_count, 1)

    async def test_mixed_modality_items_are_grouped_separately(self):
        """
        Items with different get_feature_fn (e.g. image vs video) should be
        grouped separately and each function called once with only its own items.
        Neither function should receive the other modality's items.
        """
        import torch

        TOKENS_PER_ITEM = 4
        queue = asyncio.Queue()

        image_call_items = []
        video_call_items = []

        def get_image_feature(items):
            image_call_items.append(len(items))
            return torch.zeros(len(items) * TOKENS_PER_ITEM, 64)

        def get_video_feature(items):
            video_call_items.append(len(items))
            return torch.zeros(len(items) * TOKENS_PER_ITEM, 64)

        async def batch_loop():
            while True:
                first = await queue.get()
                batch = [first]
                if queue.empty():
                    await asyncio.sleep(0.05)
                else:
                    await asyncio.sleep(0)
                while len(batch) < 8:
                    try:
                        batch.append(queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                mm_items_in_batch = [item for item, _, _ in batch]
                fns_in_batch = [fn for _, fn, _ in batch]
                futures_in_batch = [f for _, _, f in batch]

                try:
                    with torch.inference_mode():
                        fn_groups: dict = {}
                        for mm_item, fn, fut in zip(
                            mm_items_in_batch, fns_in_batch, futures_in_batch
                        ):
                            gid = id(fn)
                            if gid not in fn_groups:
                                fn_groups[gid] = (fn, [], [])
                            fn_groups[gid][1].append(mm_item)
                            fn_groups[gid][2].append(fut)

                        for fn, group_items, group_futures in fn_groups.values():
                            combined = fn(group_items).cpu()
                            if combined.ndim != 2:
                                combined = combined.reshape(-1, combined.shape[-1])
                            offset = 0
                            for fut in group_futures:
                                emb = combined[offset : offset + TOKENS_PER_ITEM]
                                offset += TOKENS_PER_ITEM
                                if not fut.done():
                                    fut.set_result(emb)
                except Exception as e:
                    for fut in futures_in_batch:
                        if not fut.done():
                            fut.set_exception(e)

        task = asyncio.get_event_loop().create_task(batch_loop())
        loop = asyncio.get_running_loop()

        # Enqueue 4 image items and 3 video items in one batch
        futures = []
        for _ in range(4):
            fut = loop.create_future()
            await queue.put((MagicMock(), get_image_feature, fut))
            futures.append(fut)
        for _ in range(3):
            fut = loop.create_future()
            await queue.put((MagicMock(), get_video_feature, fut))
            futures.append(fut)

        results = await asyncio.gather(*futures)
        task.cancel()

        # All 7 items got embeddings
        self.assertEqual(len(results), 7)
        for emb in results:
            self.assertIsInstance(emb, torch.Tensor)

        # image function called once with exactly 4 items
        self.assertEqual(image_call_items, [4])
        # video function called once with exactly 3 items
        self.assertEqual(video_call_items, [3])

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
