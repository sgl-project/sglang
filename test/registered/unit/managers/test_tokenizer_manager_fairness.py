import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (  # noqa: E402
    BatchEmbeddingOutput,
    BatchStrOutput,
    BatchTokenIDOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTokenizerReceiveFairness(unittest.IsolatedAsyncioTestCase):
    async def test_ready_messages_yield_to_waiters(self):
        for message_type in (
            BatchStrOutput,
            BatchEmbeddingOutput,
            BatchTokenIDOutput,
            object,
        ):
            with self.subTest(message_type=message_type.__name__):
                queue = asyncio.Queue()
                messages = [Mock(spec=message_type) for _ in range(32)]
                for message in messages:
                    queue.put_nowait(message)

                processed = []
                notified = asyncio.Event()
                drained = asyncio.Event()
                observed_counts = []

                def process(message):
                    processed.append(message)
                    notified.set()
                    if len(processed) == len(messages):
                        drained.set()

                async def receive(_socket):
                    # Like a buffered ZMQ receive, a nonempty queue does not suspend.
                    return await queue.get()

                async def waiter():
                    await notified.wait()
                    observed_counts.append(len(processed))

                manager = TokenizerManager.__new__(TokenizerManager)
                manager.recv_from_detokenizer = queue
                manager.soft_watchdog = MagicMock()
                # Small or skipped batches need not yield inside the handler.
                manager._handle_batch_output = AsyncMock(side_effect=process)
                manager._result_dispatcher = Mock(side_effect=process)

                with patch(
                    "sglang.srt.managers.tokenizer_manager.async_sock_recv", receive
                ):
                    waiting = asyncio.create_task(waiter())
                    receiving = asyncio.create_task(manager.handle_loop())
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(waiting, drained.wait()), timeout=5
                        )
                        self.assertEqual(processed, messages)
                        self.assertGreater(observed_counts[0], 0)
                        self.assertLess(
                            observed_counts[0],
                            len(messages),
                            "receive loop drained every message before running a waiter",
                        )
                    finally:
                        receiving.cancel()
                        waiting.cancel()
                        await asyncio.gather(receiving, waiting, return_exceptions=True)


if __name__ == "__main__":
    unittest.main()
