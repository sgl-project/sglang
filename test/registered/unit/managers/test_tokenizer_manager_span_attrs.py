"""Unit tests for the OpenTelemetry attributes on a request's root span."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import BatchEmbeddingOutput  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.srt.observability.trace import SpanAttributes  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _span_attrs(recv_obj):
    manager = SimpleNamespace(enable_trace=True, served_model_name="test-model")
    state = SimpleNamespace(
        obj=SimpleNamespace(rid="req-1", sampling_params={"max_new_tokens": 16})
    )
    stats = SimpleNamespace(prompt_tokens=120, cached_tokens=96)
    return TokenizerManager.convert_to_span_attrs(manager, state, recv_obj, 0, stats)


class TestSpanAttrs(CustomTestCase):
    def test_generate_uses_current_genai_names(self):
        attrs = _span_attrs(
            SimpleNamespace(completion_tokens=[7], finished_reasons=[{"type": "stop"}])
        )
        self.assertEqual(attrs[SpanAttributes.GEN_AI_PROVIDER_NAME], "sglang")
        self.assertEqual(attrs["gen_ai.usage.input_tokens"], 120)
        self.assertEqual(attrs["gen_ai.usage.output_tokens"], 7)
        self.assertEqual(attrs["gen_ai.usage.cache_read.input_tokens"], 96)

    def test_generate_keeps_deprecated_names(self):
        attrs = _span_attrs(
            SimpleNamespace(completion_tokens=[7], finished_reasons=[{"type": "stop"}])
        )
        self.assertEqual(attrs["gen_ai.usage.prompt_tokens"], 120)
        self.assertEqual(attrs["gen_ai.usage.completion_tokens"], 7)
        self.assertEqual(attrs["gen_ai.usage.cached_tokens"], 96)

    def test_embedding_has_no_output_tokens(self):
        recv_obj = MagicMock(spec=BatchEmbeddingOutput)
        recv_obj.finished_reasons = [None]
        attrs = _span_attrs(recv_obj)
        self.assertEqual(attrs["gen_ai.usage.input_tokens"], 120)
        self.assertNotIn("gen_ai.usage.output_tokens", attrs)
        self.assertNotIn("gen_ai.usage.completion_tokens", attrs)

    def test_nothing_when_tracing_is_off(self):
        manager = SimpleNamespace(enable_trace=False, served_model_name="test-model")
        attrs = TokenizerManager.convert_to_span_attrs(
            manager, None, None, 0, SimpleNamespace()
        )
        self.assertEqual(attrs, {})


if __name__ == "__main__":
    unittest.main()
