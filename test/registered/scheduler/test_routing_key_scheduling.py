import ast
import asyncio
import os
import unittest
from pathlib import Path

import aiohttp

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
    register_ppu_ci,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=50, stage="nightly", runner_config="1-gpu-large")
register_ppu_ci(est_time=54, suite="nightly-1-ppu", nightly=True)
register_amd_ci(est_time=120, suite="nightly-amd-1-gpu", nightly=True)
register_cpu_ci(est_time=184, suite="base-c-test-cpu")


def _has_valid_routing_key_order(log_text):
    running_counts = None
    for line in log_text.splitlines():
        if "routing_key_counts=" in line:
            try:
                raw_counts = line.split("routing_key_counts=", 1)[1].split(
                    ", waiting_keys_before=", 1
                )[0]
                running_counts = ast.literal_eval(raw_counts)
            except (SyntaxError, ValueError):
                running_counts = None
            continue
        if "waiting_keys_after=" not in line or not running_counts:
            continue
        try:
            keys = ast.literal_eval(line.split("waiting_keys_after=", 1)[1])
        except (SyntaxError, ValueError):
            continue
        if not isinstance(keys, list):
            continue
        active_positions = [i for i, key in enumerate(keys) if key in running_counts]
        other_positions = [i for i, key in enumerate(keys) if key not in running_counts]
        if not active_positions or not other_positions:
            continue
        active_counts = [running_counts[key] for key in keys if key in running_counts]
        if max(active_positions) < min(other_positions) and active_counts == sorted(
            active_counts, reverse=True
        ):
            return True
    return False


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
        3. Inspect the scheduler's routing-key debug logs
        4. Verify requests sharing a running batch's routing key are ordered
           ahead of requests with other routing keys
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

        await asyncio.gather(*short_tasks)
        await asyncio.gather(*long_running_tasks)

        self.stdout.flush()
        self.stderr.flush()
        log_text = "\n".join(
            Path(filename).read_text()
            for filename in (STDOUT_FILENAME, STDERR_FILENAME)
        )
        self.assertTrue(
            _has_valid_routing_key_order(log_text),
            "Expected requests sharing a running batch's routing key to be scheduled before other routing keys",
        )

    async def _send_chat_request(self, routing_key: str, max_tokens: int):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        headers = {"x-smg-routing-key": routing_key}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            ) as resp:
                await resp.json()


if __name__ == "__main__":
    unittest.main()
