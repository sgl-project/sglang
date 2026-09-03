"""Timing-accounting invariants for the benchmark serving stream loop.

Pins the client-side TTFT/ITL/latency arithmetic of
``sglang/benchmark/serving.py`` and documents, as tested behavior, that
ITL is recorded per stream *event*, not per token: an event carrying
several tokens contributes a single ITL sample, so ``len(itl)`` can be
smaller than ``output_len - 1`` and mean(ITL) can exceed TPOT for the
very same request because the two use different denominators.
Context: sgl-project/sglang#3050.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.benchmark import serving
from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    calculate_metrics,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-c-test-cpu")


class _FakeTime:
    """``perf_counter()`` returns pre-scripted ticks in call order."""

    def __init__(self, ticks):
        self._ticks = list(ticks)

    def perf_counter(self):
        return self._ticks.pop(0)


def _sse_event(delta_content="", usage=None):
    event = {"choices": [{"delta": {"content": delta_content}}]}
    if usage is not None:
        event["usage"] = usage
    return f"data: {json.dumps(event)}".encode("utf-8")


class _FakeContent:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        raise StopAsyncIteration


class _FakeResponse:
    def __init__(self, chunks, response_json=None):
        self.status = 200
        self.content = _FakeContent(chunks)
        self._response_json = response_json

    async def json(self):
        return self._response_json


class _PostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, response):
        self._response = response

    def post(self, url=None, json=None, headers=None):
        return _PostContext(self._response)


class _SessionContext:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


class _StaticTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text)


class TestChatCompletionsTimingInvariants(CustomTestCase):
    def setUp(self):
        # ``args`` is a module global only assigned in main(); it may be
        # absent at import time, so save-or-None and delete on teardown.
        self._orig_args = getattr(serving, "args", None)
        self._orig_time = serving.time
        self._orig_session_factory = serving._create_bench_client_session
        serving.args = SimpleNamespace(
            print_requests=False,
            disable_stream=False,
            cache_report=False,
            disable_ignore_eos=True,
            return_logprob=False,
            top_logprobs_num=0,
        )

    def tearDown(self):
        if self._orig_args is None:
            if hasattr(serving, "args"):
                del serving.args
        else:
            serving.args = self._orig_args
        serving.time = self._orig_time
        serving._create_bench_client_session = self._orig_session_factory

    @staticmethod
    def _request():
        return RequestFuncInput(
            prompt=[{"role": "user", "content": "hi"}],
            api_url="http://server/v1/chat/completions",
            prompt_len=1,
            output_len=5,
            model="test-model",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )

    def _run_streaming(self):
        # Three token-carrying events: "a" at t=0.1, "bb" (two tokens in
        # one event) at t=0.2, "c" at t=0.35; [DONE] at t=0.4. Usage
        # reports 5 completion tokens. Ticks: st, then (latency, timestamp)
        # per event, then latency for [DONE].
        serving.time = _FakeTime([0.0, 0.100, 0.100, 0.200, 0.200, 0.350, 0.350, 0.400])
        serving._create_bench_client_session = lambda: _SessionContext(
            _FakeSession(
                _FakeResponse(
                    [
                        _sse_event("a"),
                        _sse_event("bb"),
                        _sse_event("c", usage={"completion_tokens": 5}),
                        b"data: [DONE]",
                    ]
                )
            )
        )
        return asyncio.run(async_request_openai_chat_completions(self._request()))

    def test_ttft_latency_and_itl_values(self):
        output = self._run_streaming()
        self.assertTrue(output.success)
        self.assertAlmostEqual(output.ttft, 0.100)
        self.assertAlmostEqual(output.latency, 0.400)
        self.assertEqual(len(output.itl), 2)
        self.assertAlmostEqual(output.itl[0], 0.100)
        self.assertAlmostEqual(output.itl[1], 0.150)
        self.assertEqual(output.output_len, 5)
        self.assertEqual(output.generated_text, "abbc")

    def test_sum_itl_plus_done_tail_equals_decode_span(self):
        output = self._run_streaming()
        done_tail = 0.400 - 0.350  # [DONE] arrives after the last token
        self.assertAlmostEqual(
            sum(output.itl) + done_tail, output.latency - output.ttft
        )

    def test_itl_is_recorded_per_event_not_per_token(self):
        output = self._run_streaming()
        # One ITL sample per event gap, regardless of tokens per event:
        # 3 events -> 2 samples, while usage reports 5 output tokens.
        self.assertEqual(len(output.itl), 2)
        self.assertLess(len(output.itl), output.output_len - 1)

    def test_tpot_and_mean_itl_use_different_denominators(self):
        output = self._run_streaming()
        metrics, _ = calculate_metrics(
            None,
            [output],
            1.0,
            _StaticTokenizer(),
            "openai-chat",
        )
        # TPOT = (latency - ttft) / (output_len - 1) = 0.3 / 4 = 75 ms.
        self.assertAlmostEqual(metrics.mean_tpot_ms / 1000.0, 0.075)
        # Mean ITL averages the two event gaps = 125 ms — larger than TPOT
        # for the same request because ITL counts events, not tokens.
        self.assertAlmostEqual(metrics.mean_itl_ms / 1000.0, 0.125)
        self.assertGreater(metrics.mean_itl_ms, metrics.mean_tpot_ms)

    def test_non_streaming_reports_ttft_equal_to_latency(self):
        serving.time = _FakeTime([0.0, 1.0])
        serving._create_bench_client_session = lambda: _SessionContext(
            _FakeSession(
                _FakeResponse(
                    [],
                    response_json={
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"completion_tokens": 2},
                    },
                )
            )
        )
        serving.args.disable_stream = True
        output = asyncio.run(async_request_openai_chat_completions(self._request()))
        self.assertTrue(output.success)
        self.assertAlmostEqual(output.ttft, output.latency)
        self.assertAlmostEqual(output.latency, 1.0)
        self.assertEqual(output.output_len, 2)
        self.assertEqual(output.generated_text, "hello")


if __name__ == "__main__":
    unittest.main()
