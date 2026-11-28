"""Regression tests for native /generate token accounting, without a model."""

import asyncio
import json
import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.benchmark import serving
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _chunk(tokens, text="", **meta):
    return {"text": text, "meta_info": {"completion_tokens": tokens, **meta}}


class TestBenchServingNativeStream(CustomTestCase):
    def _run(self, events, requested_output_len=1024):
        # Each event supplies its receive time. No sleeps or real model are needed.
        clock = MagicMock(return_value=0.0)

        async def content():
            for timestamp, data in events:
                clock.return_value = timestamp
                payload = data if isinstance(data, str) else json.dumps(data)
                yield f"data: {payload}\n\n".encode()

        response = SimpleNamespace(status=200, content=content())
        response_context = MagicMock()
        response_context.__aenter__.return_value = response
        session = MagicMock()
        session.post.return_value = response_context
        session_context = MagicMock()
        session_context.__aenter__.return_value = session
        args = Namespace(
            temperature=0,
            top_p=1,
            disable_ignore_eos=False,
            disable_stream=False,
            return_logprob=False,
            return_routed_experts=False,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=None,
            header=None,
            cache_report=False,
        )
        request = serving.RequestFuncInput(
            prompt="hello",
            api_url="http://localhost/generate",
            prompt_len=1,
            output_len=requested_output_len,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        with (
            patch.object(serving, "args", args, create=True),
            patch.object(
                serving, "_create_bench_client_session", return_value=session_context
            ),
            patch.object(serving.time, "perf_counter", clock),
        ):
            return asyncio.run(serving.async_request_sglang_generate(request))

    def _metrics(self, outputs, duration=5.0):
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: text.split())
        return serving.calculate_metrics(
            input_requests=None,
            outputs=outputs,
            dur_s=duration,
            tokenizer=tokenizer,
            backend="sglang",
        )[0]

    def test_empty_text_records_tokens_ttft_itl_and_tpot(self):
        # Zero-token and duplicate chunks must not start or reset token timing.
        out = self._run(
            [
                (0.2, _chunk(0)),
                (1.0, _chunk(1)),
                (1.5, _chunk(1)),
                (3.0, _chunk(3)),
                (5.0, _chunk(5, spec_accept_length=4.5)),
                (5.0, _chunk(5)),
                (5.0, "[DONE]"),
            ]
        )
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.generated_text, "")
        self.assertEqual(out.output_len, 5)
        self.assertEqual(out.ttft, 1.0)
        self.assertEqual(out.itl, [1.0] * 4)
        self.assertEqual(out.spec_accept_length, 4.5)
        metrics = self._metrics([out])
        self.assertEqual(metrics.mean_ttft_ms, 1000.0)
        self.assertEqual(metrics.mean_tpot_ms, 1000.0)
        self.assertEqual(metrics.mean_itl_ms, 1000.0)
        self.assertEqual(metrics.output_throughput, 1.0)

    def test_visible_text_after_special_tokens_keeps_first_token_time(self):
        out = self._run([(1.0, _chunk(1)), (2.0, _chunk(2)), (3.0, _chunk(3, "hello"))])
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.generated_text, "hello")
        self.assertEqual(out.ttft, 1.0)
        self.assertEqual(out.itl, [1.0, 1.0])

    def test_duplicate_visible_text_does_not_reset_itl(self):
        out = self._run(
            [
                (1.0, _chunk(1, "hello")),
                (2.0, _chunk(1, "hello")),
                (3.0, _chunk(2, "hello world")),
            ]
        )
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.generated_text, "hello world")
        self.assertEqual(out.itl, [2.0])

    def test_speculative_chunks_use_token_count_increments(self):
        out = self._run([(1.0, _chunk(6)), (2.2, _chunk(12))])
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.output_len, 12)
        self.assertEqual(out.ttft, 1.0)
        self.assertEqual(len(out.itl), 6)
        for itl in out.itl:
            self.assertAlmostEqual(itl, 0.2)

    def test_usage_only_chunk_updates_count_without_creating_token_event(self):
        out = self._run(
            [
                (1.0, _chunk(1, "hello")),
                (2.0, _chunk(2, "hello world")),
                (3.0, {"meta_info": {"completion_tokens": 3}}),
            ]
        )
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.output_len, 3)
        self.assertEqual(out.generated_text, "hello world")
        self.assertEqual(out.itl, [1.0])

    def test_output_ids_without_text_can_record_token_events(self):
        out = self._run(
            [
                (1.0, {"output_ids": [0], "meta_info": {"completion_tokens": 1}}),
                (2.0, {"output_ids": [0, 0], "meta_info": {"completion_tokens": 2}}),
            ]
        )
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.output_len, 2)
        self.assertEqual(out.ttft, 1.0)
        self.assertEqual(out.itl, [1.0])

    def test_empty_terminal_chunk_preserves_previously_received_text(self):
        out = self._run(
            [
                (1.0, _chunk(1, "hello")),
                (2.0, _chunk(2, "hello world")),
                (3.0, _chunk(2)),
            ]
        )
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.generated_text, "hello world")
        self.assertEqual(out.output_len, 2)
        self.assertEqual(out.itl, [1.0])

    def test_usage_only_response_does_not_invent_token_timestamps(self):
        out = self._run([(2.0, {"meta_info": {"completion_tokens": 3}})])
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.output_len, 3)
        self.assertEqual(out.ttft, 0.0)
        self.assertEqual(out.itl, [])
        with self.assertWarnsRegex(UserWarning, "1.*first-token"):
            metrics = self._metrics([out])
        self.assertEqual(metrics.total_output, 3)
        self.assertEqual(metrics.max_output_tokens_per_s, 0.0)

    def test_zero_output_does_not_use_requested_length_or_start_timing(self):
        out = self._run([(1.0, _chunk(0)), (2.0, "[DONE]")])
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.output_len, 0)
        self.assertEqual(out.ttft, 0.0)
        self.assertEqual(out.itl, [])
        self.assertEqual(self._metrics([out]).max_output_tokens_per_s, 0.0)

    def test_missing_completion_count_fails_instead_of_using_requested_length(self):
        out = self._run([(1.0, {"text": "hello"}), (2.0, "[DONE]")])
        self.assertFalse(out.success)
        self.assertEqual(out.output_len, 0)
        self.assertIn("completion_tokens", out.error)

    def test_unmeasured_ttft_does_not_corrupt_latency_or_peak_metrics(self):
        measured = serving.RequestFuncOutput(
            success=True,
            output_len=4,
            ttft=1.0,
            latency=4.0,
            itl=[1.0, 1.0, 1.0],
        )
        unmeasured = serving.RequestFuncOutput(success=True, output_len=4, latency=9.0)
        with self.assertWarnsRegex(UserWarning, "2.*first-token"):
            metrics = self._metrics([measured, unmeasured, unmeasured], duration=10.0)
        self.assertEqual(metrics.mean_ttft_ms, 1000.0)
        self.assertEqual(metrics.mean_tpot_ms, 1000.0)
        self.assertEqual(metrics.max_output_tokens_per_s, 1.0)
        self.assertEqual(metrics.completed, 3)
        self.assertEqual(metrics.total_output, 12)
        self.assertEqual(metrics.output_throughput, 1.2)
        self.assertEqual(metrics.max_concurrent_requests, 3)


if __name__ == "__main__":
    unittest.main()
