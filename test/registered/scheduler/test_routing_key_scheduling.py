import asyncio
import os
import time
import unittest

import aiohttp

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)
register_amd_ci(est_time=120, suite="nightly-amd-1-gpu", nightly=True)


class TestRoutingKeyScheduling(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["SGLANG_ROUTING_KEY_POLICY_DEBUG_LOG"] = "1"

        cls.model = "Qwen/Qwen3-0.6B"
        cls.base_url = DEFAULT_URL_FOR_TEST

        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--max-running-requests",
                "3",
                "--schedule-policy",
                "routing-key",
            ),
            return_stdout_stderr=(cls.stdout, cls.stderr),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()
        os.remove(STDOUT_FILENAME)
        os.remove(STDERR_FILENAME)

    def test_routing_key_scheduling_order(self):
        """Verify requests with matching routing keys are prioritized.

        Test strategy:
        1. First send 2 long-running key_a requests to occupy running batch
        2. Then send 10 key_a and 10 key_b short requests concurrently
        3. With max_running_requests=3, key_a requests should be prioritized
           because running batch has 2 key_a requests
        4. Verify key_a requests finish before key_b requests on average
        """
        asyncio.run(self._test_routing_key_scheduling_order())

    async def _test_routing_key_scheduling_order(self):
        long_running_tasks = [
            asyncio.create_task(self._send_chat_request("key_a", 20000)),
            asyncio.create_task(self._send_chat_request("key_a", 20000)),
        ]

        await asyncio.sleep(2.0)

        short_tasks = []
        for _ in range(10):
            short_tasks.append(
                asyncio.create_task(self._send_chat_request("key_a", 10))
            )
            short_tasks.append(
                asyncio.create_task(self._send_chat_request("key_b", 10))
            )

        all_short_results = await asyncio.gather(*short_tasks)
        await asyncio.gather(*long_running_tasks)

        key_a_latencies = [lat for key, lat in all_short_results if key == "key_a"]
        key_b_latencies = [lat for key, lat in all_short_results if key == "key_b"]

        avg_key_a = sum(key_a_latencies) / len(key_a_latencies)
        avg_key_b = sum(key_b_latencies) / len(key_b_latencies)

        print(f"Average key_a latency: {avg_key_a:.3f}s")
        print(f"Average key_b latency: {avg_key_b:.3f}s")

        self.assertLess(
            avg_key_a,
            avg_key_b,
            f"key_a requests (avg={avg_key_a:.3f}s) should finish before key_b (avg={avg_key_b:.3f}s)",
        )

    async def _send_chat_request(self, routing_key: str, max_tokens: int):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        headers = {"x-smg-routing-key": routing_key}
        start_time = time.perf_counter()
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                await resp.json()
        latency = time.perf_counter() - start_time
        return routing_key, latency


if __name__ == "__main__":
    unittest.main()
