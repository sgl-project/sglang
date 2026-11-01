"""Encode timing must exclude rendering/queueing and survive process aggregation.

These CPU tests exercise the serving hooks and exported histogram, rather than
asserting that a mocked observer was called. Moving a timer outside the encode,
dividing batch time, or recording cache hits silently changes dashboard meaning.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from collections import OrderedDict
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    multiprocess,
)

from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()  # CPU tests never call GPU kernels.

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.frontend_request_timing import FrontendRequestTiming
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.observability.tokenizer_encode_metrics import measure_tokenizer_encode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_METRIC = "sglang:tokenizer_encode_seconds"
_CLOCK = "sglang.srt.observability.tokenizer_encode_metrics.time.perf_counter"


class _Clock:
    value = 0.0

    def __call__(self):
        return self.value


class _Tokenizer:
    is_fast = True
    chat_template = "test template"

    def __init__(self, clock):
        self.clock = clock

    def encode(self, text, **kwargs):
        self.clock.value += 0.02
        if text == "invalid":
            raise ValueError("invalid input")
        return [len(text)]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text, **kwargs) for text in texts]}

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        self.clock.value += 1.0
        return self.encode("rendered") if tokenize else "rendered"

    def decode(self, ids):
        self.clock.value += 2.0
        return "decoded"


class TestTokenizerEncodeMetrics(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.enterContext(published_topology())
        self.registry = CollectorRegistry()

        class IsolatedCollector(TokenizerMetricsCollector):
            _counter_cls = partial(Counter, registry=self.registry)
            _gauge_cls = partial(Gauge, registry=self.registry)
            _histogram_cls = partial(Histogram, registry=self.registry)

        self.collector = IsolatedCollector(
            labels={"model_name": "test", "engine_type": "unified"}
        )
        self.clock = _Clock()
        self.tokenizer = _Tokenizer(self.clock)

    def sample(self, suffix, *, source="manager", input_type="single", **extra):
        return self.registry.get_sample_value(
            _METRIC + suffix,
            {
                "model_name": "test",
                "engine_type": "unified",
                "source": source,
                "input_type": input_type,
                "operation": "encode",
                "result": "success",
                **extra,
            },
        )

    def manager(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = self.tokenizer
        manager.metrics_collector = self.collector
        manager.async_dynamic_batch_tokenizer = None
        manager.model_config = SimpleNamespace(is_embedding_gemma=False)
        return manager

    def test_chat_excludes_render_decode_and_cache_hits(self):
        server = SimpleNamespace(
            tokenizer_manager=self.manager(),
            template_manager=SimpleNamespace(python_chat_template=None),
            _prompt_text_round_trip_is_lossy=False,
            _chat_template_cache=OrderedDict(),
        )
        args = dict(tools=None, template_kwargs={}, encode_kwargs={}, use_cache=True)
        messages = [{"role": "user", "content": "hello"}]
        with patch(_CLOCK, self.clock):
            first = OpenAIServingChat._render_and_encode_chat_template(
                server, messages, **args
            )
            second = OpenAIServingChat._render_and_encode_chat_template(
                server, messages, **args
            )
        self.assertEqual(first, ([8], "decoded"))
        self.assertEqual(first, second)
        self.assertEqual(self.sample("_count", source="chat"), 1)
        self.assertAlmostEqual(self.sample("_sum", source="chat"), 0.02)
        self.assertGreater(self.clock.value, 3.0)

    def test_frontend_submission_records_one_sample_per_logical_request(self):
        timing = FrontendRequestTiming(started_at=10.0)
        timing.add_span("text", 10.1, 10.3)
        timing.add_span("multimodal", 10.4, 10.8)
        manager = self.manager()
        manager.enable_metrics = True
        manager.rid_to_state = {
            "request": SimpleNamespace(
                obj=SimpleNamespace(_frontend_request_timing=timing)
            ),
            "warmup": SimpleNamespace(
                obj=SimpleNamespace(_frontend_request_timing=None)
            ),
        }
        manager._observe_frontend_request("warmup", 10.9)
        manager._observe_frontend_request("request", 11.0)
        manager._observe_frontend_request("request", 12.0)
        labels = {"model_name": "test", "engine_type": "unified", "stage": "total"}
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:frontend_request_stage_seconds_count", labels
            ),
            1,
        )
        self.assertEqual(
            self.registry.get_sample_value(
                "sglang:frontend_request_stage_seconds_sum", labels
            ),
            1,
        )
        labels["stage"] = "text"
        self.assertAlmostEqual(
            self.registry.get_sample_value(
                "sglang:frontend_request_stage_seconds_sum", labels
            ),
            0.2,
        )

    def test_manager_batch_counts_calls_without_dividing_duration(self):
        with patch(_CLOCK, self.clock):
            ids, types = asyncio.run(
                self.manager()._tokenize_texts(["first", "second"])
            )
        self.assertEqual(ids, [[5], [6]])
        self.assertIsNone(types)
        self.assertEqual(self.sample("_count", input_type="batch"), 1)
        self.assertAlmostEqual(self.sample("_sum", input_type="batch"), 0.04)
        self.assertIsNone(self.sample("_count"))

    def test_fused_template_encoding_has_a_separate_operation(self):
        server = SimpleNamespace(
            tokenizer_manager=self.manager(), _prompt_text_round_trip_is_lossy=True
        )
        with patch(_CLOCK, self.clock):
            ids, _ = OpenAIServingChat._render_and_encode_chat_template(
                server,
                [{"role": "user", "content": "hello"}],
                tools=None,
                template_kwargs={},
                encode_kwargs={},
                use_cache=False,
            )
        self.assertEqual(ids, [8])
        self.assertEqual(
            self.sample("_count", source="chat", operation="chat_template"), 1
        )
        self.assertAlmostEqual(
            self.sample("_sum", source="chat", operation="chat_template"), 1.02
        )
        self.assertIsNone(self.sample("_count", source="chat"))

    def test_encode_error_keeps_the_exception_and_its_own_population(self):
        self.tokenizer.is_fast = False
        with (
            patch(_CLOCK, self.clock),
            self.assertRaisesRegex(ValueError, "invalid input"),
        ):
            asyncio.run(self.manager()._tokenize_texts("invalid"))
        self.assertEqual(self.sample("_count", result="error"), 1)
        self.assertAlmostEqual(self.sample("_sum", result="error"), 0.02)
        self.assertIsNone(self.sample("_count"))

    def test_metrics_disabled_does_not_read_the_clock(self):
        with patch(_CLOCK, side_effect=AssertionError("disabled timer read clock")):
            with measure_tokenizer_encode(None, source="manager"):
                ids = self.tokenizer.encode("input")
        self.assertEqual(ids, [5])
        self.assertIsNone(self.sample("_count"))

    def test_async_timer_starts_in_executor_after_queue_wait(self):
        entered, release = threading.Event(), threading.Event()
        batcher = AsyncDynamicbatchTokenizer(self.tokenizer)
        batcher.metrics_collector = self.collector

        def block_executor():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release executor")

        async def run():
            loop = asyncio.get_running_loop()
            blocker = loop.run_in_executor(batcher._executor, block_executor)
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            futures = [loop.create_future(), loop.create_future()]
            pending = asyncio.create_task(
                batcher._process_dynamic_batch(["first", "second"], [{}, {}], futures)
            )
            await asyncio.sleep(0)
            self.clock.value += 1.0
            release.set()
            await blocker
            await pending
            return [future.result() for future in futures]

        try:
            with patch(_CLOCK, self.clock):
                result = asyncio.run(run())
        finally:
            release.set()
            batcher._executor.shutdown(wait=True)
        self.assertEqual(result, [{"input_ids": [5]}, {"input_ids": [6]}])
        self.assertEqual(self.sample("_count", input_type="batch"), 1)
        self.assertAlmostEqual(self.sample("_sum", input_type="batch"), 0.04)

    def test_multiprocess_histogram_merges_without_a_scheduler(self):
        code = """
import sys
from sglang.test.test_utils import maybe_stub_sgl_kernel, published_topology
maybe_stub_sgl_kernel()
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
with published_topology():
    collector = TokenizerMetricsCollector(labels={"model_name": "test", "engine_type": "unified"})
    collector.observe_tokenizer_encode(float(sys.argv[1]), source="chat", input_type="single", operation="encode", result="success")
"""
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory}
            for duration in (0.02, 0.03):
                subprocess.run(
                    [sys.executable, "-c", code, str(duration)],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry, path=directory)
            labels = {
                "model_name": "test",
                "engine_type": "unified",
                "source": "chat",
                "input_type": "single",
                "operation": "encode",
                "result": "success",
            }
            self.assertEqual(registry.get_sample_value(_METRIC + "_count", labels), 2)
            self.assertAlmostEqual(
                registry.get_sample_value(_METRIC + "_sum", labels), 0.05
            )
            self.assertEqual(
                registry.get_sample_value(
                    _METRIC + "_bucket", {**labels, "le": "0.025"}
                ),
                1,
            )


if __name__ == "__main__":
    unittest.main()
