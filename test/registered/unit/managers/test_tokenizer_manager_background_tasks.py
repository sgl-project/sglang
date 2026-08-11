"""Regression tests for TokenizerManager background-task ownership."""

import asyncio
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTokenizerManagerBackgroundTasks(CustomTestCase):
    def test_task_is_retained_until_completion(self):
        async def run_test():
            started = asyncio.Event()
            finish = asyncio.Event()

            async def background_job():
                started.set()
                await finish.wait()

            manager = TokenizerManager.__new__(TokenizerManager)
            manager.asyncio_tasks = set()
            task = manager._create_background_task(background_job())

            await started.wait()
            self.assertIn(task, manager.asyncio_tasks)

            finish.set()
            await task
            await asyncio.sleep(0)
            self.assertNotIn(task, manager.asyncio_tasks)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
