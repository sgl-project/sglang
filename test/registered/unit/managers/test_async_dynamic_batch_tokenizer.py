import asyncio
import threading
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCancelledDynamicTokenization(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "active": 1, "live": 2, "one": 3, "two": 4},
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        self.real_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="[UNK]"
        )
        self.calls = []
        self.started = asyncio.get_running_loop().create_future()
        self.release = threading.Event()
        loop = asyncio.get_running_loop()

        def encode(texts, **kwargs):
            self.calls.append(texts)
            if texts == "active":
                loop.call_soon_threadsafe(self.started.set_result, None)
                if not self.release.wait(5):
                    raise TimeoutError("active tokenization was not released")
            return self.real_tokenizer(texts, **kwargs)

        self.batcher = AsyncDynamicbatchTokenizer(
            encode, max_batch_size=5, batch_wait_timeout_s=1
        )

    async def asyncTearDown(self):
        self.release.set()
        if self.batcher._batcher_task is not None:
            self.batcher._batcher_task.cancel()
            await asyncio.gather(self.batcher._batcher_task, return_exceptions=True)
        self.batcher._executor.shutdown(wait=True)

    async def check_queued_cancellation(self, mixed_live_kwargs=False):
        active = asyncio.create_task(self.batcher.encode("active"))
        await asyncio.wait_for(self.started, timeout=3)
        prompts = [
            "cancel first",
            "live one",
            "cancel middle",
            "live two",
            "cancel last",
        ]
        kwargs = [
            {"return_token_type_ids": True},
            {},
            {},
            {"return_token_type_ids": True} if mixed_live_kwargs else {},
            {},
        ]
        queued = [
            asyncio.create_task(self.batcher.encode(prompt, **kw))
            for prompt, kw in zip(prompts, kwargs)
        ]
        # Tasks above are scheduled before this barrier, so all have enqueued
        # behind the active encoder without a sleep-based timing assumption.
        enqueued = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(enqueued.set_result, None)
        await enqueued
        self.assertEqual(self.batcher._queue.qsize(), len(queued))
        for index in [0, 2, 4]:
            queued[index].cancel()
        cancelled = await asyncio.gather(
            *(queued[index] for index in [0, 2, 4]), return_exceptions=True
        )
        self.assertTrue(
            all(isinstance(result, asyncio.CancelledError) for result in cancelled)
        )
        self.release.set()

        results = await asyncio.wait_for(
            asyncio.gather(active, queued[1], queued[3]), timeout=3
        )
        self.assertEqual(results[0], self.real_tokenizer("active"))
        self.assertEqual(results[1], self.real_tokenizer(prompts[1], **kwargs[1]))
        self.assertEqual(results[2], self.real_tokenizer(prompts[3], **kwargs[3]))
        self.assertEqual(
            self.calls,
            ["active", "live one", "live two"]
            if mixed_live_kwargs
            else ["active", ["live one", "live two"]],
        )

    async def test_cancelled_prompts_are_excluded_before_batching(self):
        await self.check_queued_cancellation()

    async def test_live_requests_with_different_kwargs_keep_their_results(self):
        await self.check_queued_cancellation(mixed_live_kwargs=True)

    async def test_all_cancelled_batches_do_not_invoke_tokenizer(self):
        for size in [1, 3]:
            with self.subTest(size=size):
                futures = [
                    asyncio.get_running_loop().create_future() for _ in range(size)
                ]
                for future in futures:
                    future.cancel()
                await asyncio.wait_for(
                    self.batcher._process_dynamic_batch(
                        ["cancelled"] * size, [{} for _ in range(size)], futures
                    ),
                    timeout=3,
                )
                self.assertEqual(self.calls, [])

    async def test_live_request_after_all_cancelled_batch_completes(self):
        active = asyncio.create_task(self.batcher.encode("active"))
        await asyncio.wait_for(self.started, timeout=3)
        queued = [
            asyncio.create_task(self.batcher.encode("cancelled")) for _ in range(5)
        ]
        enqueued = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_soon(enqueued.set_result, None)
        await enqueued
        self.assertEqual(self.batcher._queue.qsize(), len(queued))
        for task in queued:
            task.cancel()
        await asyncio.gather(*queued, return_exceptions=True)
        self.release.set()
        await asyncio.wait_for(active, timeout=3)

        result = await asyncio.wait_for(self.batcher.encode("live"), timeout=3)
        self.assertEqual(result, self.real_tokenizer("live"))
        self.assertEqual(self.calls, ["active", "live"])


if __name__ == "__main__":
    unittest.main()
