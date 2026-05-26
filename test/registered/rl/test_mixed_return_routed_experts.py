import asyncio
import unittest

import aiohttp

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


class TestMixedReturnRoutedExperts(CustomTestCase):
    @classmethod
    def _run_server_case(cls, other_args):
        process = popen_launch_server(
            DEFAULT_ENABLE_ROUTED_EXPERTS_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                2,
                "--enable-return-routed-experts",
                "--disable-cuda-graph",
                *other_args,
            ],
        )
        try:
            return asyncio.run(cls._send_mixed_batch())
        finally:
            kill_process_tree(process.pid)

    @classmethod
    async def _send_mixed_batch(cls):
        payload_no_rr = {
            "text": "The quick brown fox jumps over the lazy dog.",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "return_routed_experts": False,
        }
        payload_with_rr = {
            "text": "The quick brown fox jumps over the lazy dog.",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 16,
                "ignore_eos": True,
            },
            "return_routed_experts": True,
        }

        async with aiohttp.ClientSession() as session:
            return await asyncio.gather(
                cls._post_generate(session, payload_no_rr),
                cls._post_generate(session, payload_with_rr),
            )

    @staticmethod
    async def _post_generate(session, payload):
        async with session.post(
            f"{DEFAULT_URL_FOR_TEST}/generate", json=payload
        ) as response:
            body = await response.json()
            if response.status != 200:
                raise AssertionError(f"HTTP {response.status}: {body}")
            if "error" in body:
                raise AssertionError(f"generate returned error: {body['error']}")
            return body

    def _assert_mixed_batch_result(self, responses):
        no_rr, with_rr = responses
        self.assertNotIn("routed_experts", no_rr.get("meta_info", {}))
        self.assertIn("routed_experts", with_rr.get("meta_info", {}))

    def test_mixed_return_routed_experts_single_tokenizer(self):
        responses = self._run_server_case([])
        self._assert_mixed_batch_result(responses)

    def test_mixed_return_routed_experts_multi_tokenizer(self):
        responses = self._run_server_case(["--tokenizer-worker-num", 2])
        self._assert_mixed_batch_result(responses)


if __name__ == "__main__":
    unittest.main()
