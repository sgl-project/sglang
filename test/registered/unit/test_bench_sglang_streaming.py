"""Native benchmark timing follows generated tokens, including empty text."""

import json
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch

from sglang.benchmark import serving
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNativeStreamingTiming(unittest.IsolatedAsyncioTestCase):
    async def generate(self, frames):
        clock = [100.0]

        async def content():
            for timestamp, frame in frames:
                clock[0] = timestamp
                body = frame if isinstance(frame, str) else json.dumps(frame)
                yield ("data: " + body).encode()

        @asynccontextmanager
        async def response(**_kwargs):
            yield SimpleNamespace(status=200, content=content())

        @asynccontextmanager
        async def session():
            yield SimpleNamespace(post=response)

        args = SimpleNamespace(
            temperature=0,
            top_p=1,
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            cache_report=True,
        )
        request = serving.RequestFuncInput(
            prompt=[1, 2],
            api_url="http://unused/generate",
            prompt_len=2,
            output_len=99,
            model="test",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        with (
            patch.object(serving, "args", args, create=True),
            patch.object(serving, "_create_bench_client_session", session),
            patch.object(serving, "get_request_headers", return_value={}),
            patch.object(serving.time, "perf_counter", side_effect=lambda: clock[0]),
        ):
            result = await serving.async_request_sglang_generate(request)
        self.assertTrue(result.success, result.error)
        return result

    async def test_all_empty_text_still_records_first_token_and_itls(self):
        result = await self.generate(
            [
                (100.1, {"text": "", "meta_info": {"completion_tokens": 1}}),
                (100.2, {"text": "", "meta_info": {"completion_tokens": 2}}),
                (100.3, {"text": "", "meta_info": {"completion_tokens": 3}}),
                (100.4, "[DONE]"),
            ]
        )
        self.assertEqual(result.output_len, 3)
        self.assertEqual(result.generated_text, "")
        self.assertAlmostEqual(result.ttft, 0.1)
        self.assertAlmostEqual(result.latency, 0.4)
        self.assertEqual(len(result.itl), 2)
        self.assertAlmostEqual(sum(result.itl), 0.2)

    async def test_empty_prefix_does_not_delay_first_token_until_visible_text(self):
        result = await self.generate(
            [
                (100.1, {"text": "", "meta_info": {"completion_tokens": 1}}),
                (100.2, {"text": "", "meta_info": {"completion_tokens": 2}}),
                (100.3, {"text": "visible", "meta_info": {"completion_tokens": 3}}),
                (100.4, "[DONE]"),
            ]
        )
        self.assertAlmostEqual(result.ttft, 0.1)
        self.assertEqual(len(result.itl), 2)
        self.assertEqual(result.generated_text, "visible")

    async def test_batched_tokens_and_final_usage_do_not_double_count(self):
        result = await self.generate(
            [
                (100.1, {"meta_info": {"completion_tokens": 0}}),
                (100.2, {"text": "a", "meta_info": {"completion_tokens": 2}}),
                (100.5, {"text": "ab", "meta_info": {"completion_tokens": 5}}),
                (
                    100.6,
                    {
                        "text": "abc",
                        "meta_info": {"completion_tokens": 5, "cached_tokens": 2},
                    },
                ),
                (
                    100.65,
                    {
                        "text": "",
                        "meta_info": {"completion_tokens": 5, "cached_tokens": 2},
                    },
                ),
                (100.7, "[DONE]"),
            ]
        )
        self.assertEqual(result.output_len, 5)
        self.assertEqual(result.generated_text, "abc")
        self.assertEqual(result.cached_tokens, 2)
        self.assertAlmostEqual(result.ttft, 0.2)
        self.assertAlmostEqual(result.latency, 0.7)
        self.assertEqual(len(result.itl), 3)
        for interval in result.itl:
            self.assertAlmostEqual(interval, 0.1)

    async def test_no_tokens_does_not_report_requested_output_length(self):
        result = await self.generate(
            [
                (100.1, {"text": "", "meta_info": {"completion_tokens": 0}}),
                (100.2, "[DONE]"),
            ]
        )
        self.assertEqual(result.output_len, 0)
        self.assertEqual(result.ttft, 0)
        self.assertEqual(result.itl, [])


if __name__ == "__main__":
    unittest.main()
