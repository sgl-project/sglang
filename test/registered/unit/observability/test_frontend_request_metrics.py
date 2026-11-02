"""Exercise request timing at serving hooks and the exported histogram."""

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

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, multiprocess
from sglang.test.test_utils import (
    CustomTestCase,
    maybe_stub_sgl_kernel,
    published_topology,
)

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.async_dynamic_batch_tokenizer import AsyncDynamicbatchTokenizer
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.observability.frontend_request_timing import (
    FrontendRequestTiming,
    bind_frontend_timing,
)
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

_METRIC = "sglang:frontend_request_stage_seconds"
_CLOCK = "sglang.srt.observability.frontend_request_timing.time.perf_counter"


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
        return [len(text)]

    def __call__(self, texts, **kwargs):
        return {"input_ids": [self.encode(text, **kwargs) for text in texts]}

    def apply_chat_template(self, messages, *, tokenize, **kwargs):
        self.clock.value += 1.0
        return self.encode("rendered") if tokenize else "rendered"

    def decode(self, ids):
        self.clock.value += 2.0
        return "decoded"


class TestFrontendRequestMetrics(CustomTestCase):
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

    def manager(self):
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = self.tokenizer
        manager.metrics_collector = self.collector
        manager.async_dynamic_batch_tokenizer = None
        manager.model_config = SimpleNamespace(is_embedding_gemma=False)
        return manager

    def test_only_three_stages_and_one_sample_at_first_submission(self):
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
        for stage, duration in [("total", 1.0), ("text", 0.2), ("multimodal", 0.4)]:
            labels = {"model_name": "test", "engine_type": "unified", "stage": stage}
            self.assertEqual(
                self.registry.get_sample_value(_METRIC + "_count", labels), 1
            )
            self.assertAlmostEqual(
                self.registry.get_sample_value(_METRIC + "_sum", labels), duration
            )
        stages = {
            s.labels["stage"]
            for m in self.registry.collect()
            for s in m.samples
            if s.name == _METRIC + "_count"
        }
        self.assertEqual(stages, {"total", "text", "multimodal"})
        self.assertNotIn(
            "sglang:tokenizer_encode_seconds", {m.name for m in self.registry.collect()}
        )

    def test_chat_encoding_excludes_separate_render_decode_and_cache_hits(self):
        server = SimpleNamespace(
            tokenizer_manager=self.manager(),
            template_manager=SimpleNamespace(python_chat_template=None),
            _prompt_text_round_trip_is_lossy=False,
            _chat_template_cache=OrderedDict(),
        )
        timing = FrontendRequestTiming(started_at=0.0)
        args = dict(tools=None, template_kwargs={}, encode_kwargs={}, use_cache=True)
        messages = [{"role": "user", "content": "hello"}]
        with patch(_CLOCK, self.clock), bind_frontend_timing(timing):
            first = OpenAIServingChat._render_and_encode_chat_template(
                server, messages, **args
            )
            second = OpenAIServingChat._render_and_encode_chat_template(
                server, messages, **args
            )
        self.assertEqual(first, ([8], "decoded"))
        self.assertEqual(first, second)
        self.assertAlmostEqual(timing.snapshot(self.clock.value)["text"], 0.02)
        self.assertGreater(self.clock.value, 3.0)

    def test_fused_template_remains_text_without_extra_stage(self):
        server = SimpleNamespace(
            tokenizer_manager=self.manager(), _prompt_text_round_trip_is_lossy=True
        )
        timing = FrontendRequestTiming(started_at=0.0)
        with patch(_CLOCK, self.clock), bind_frontend_timing(timing):
            ids, _ = OpenAIServingChat._render_and_encode_chat_template(
                server,
                [{"role": "user", "content": "hello"}],
                tools=None,
                template_kwargs={},
                encode_kwargs={},
                use_cache=False,
            )
        self.assertEqual(ids, [8])
        self.assertAlmostEqual(timing.snapshot(self.clock.value)["text"], 1.02)
        self.assertNotIn("chat_template", timing.snapshot(self.clock.value))

    def test_async_batch_attributes_execution_to_each_request_without_queue_wait(self):
        entered, release = threading.Event(), threading.Event()
        batcher = AsyncDynamicbatchTokenizer(self.tokenizer)
        timings = [FrontendRequestTiming(0.0), FrontendRequestTiming(0.0)]

        def block_executor():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test did not release executor")

        async def run():
            loop = asyncio.get_running_loop()
            blocker = loop.run_in_executor(batcher._executor, block_executor)
            self.assertTrue(await asyncio.to_thread(entered.wait, 5))
            futures = [loop.create_future(), loop.create_future()]
            for future, timing in zip(futures, timings):
                future._frontend_timing = timing
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
        for timing in timings:
            self.assertAlmostEqual(timing.snapshot(self.clock.value)["text"], 0.04)

    def test_request_histogram_merges_across_processes(self):
        code = """
import sys
from sglang.test.test_utils import maybe_stub_sgl_kernel, published_topology
maybe_stub_sgl_kernel()
from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
with published_topology():
    collector = TokenizerMetricsCollector(labels={"model_name": "test", "engine_type": "unified"})
    collector.observe_frontend_request_timing({"total": float(sys.argv[1]), "text": 0.01})
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
            for stage, expected in [
                ("total", 0.05),
                ("text", 0.02),
                ("multimodal", 0.0),
            ]:
                labels = {
                    "model_name": "test",
                    "engine_type": "unified",
                    "stage": stage,
                }
                self.assertEqual(
                    registry.get_sample_value(_METRIC + "_count", labels), 2
                )
                self.assertAlmostEqual(
                    registry.get_sample_value(_METRIC + "_sum", labels), expected
                )


if __name__ == "__main__":
    unittest.main()
