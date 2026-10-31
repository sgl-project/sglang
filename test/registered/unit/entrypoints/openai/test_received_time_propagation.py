"""RED tests for F-SCORE: received_time propagation for /v1/score and /v1/responses.

CPU-only. The serving layer's heavy preprocessing/tokenization is stubbed; the
tests capture the ``received_time`` stamped onto the internal
GenerateReqInput/EmbeddingReqInput (or forwarded into score_request).

Control groups:
- /v1/embeddings: OpenAIServingBase.handle_request already stamps it
  (serving_base.py: L98) and reaches generate_request directly.
- /v1/responses uses the same stub-and-drive pattern as the existing
  test_multimodal_create_responses_sends_text_and_media_to_engine.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingRequest,
    MessageProcessingResult,
    ResponsesRequest,
    ScoringRequest,
)
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.entrypoints.openai.serving_score import OpenAIServingScore
from sglang.srt.observability.req_time_stats import monotonic_time
from sglang.srt.runtime_context import publish, reset_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

register_cpu_ci(est_time=9, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="stage-b-test-cpu-intel")


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class _StubTemplateManager:
    jinja_template_content_format = None
    chat_template_name = "llama-3"
    completion_template_name = None
    reasoning_config = None
    force_reasoning = False
    jinja_template_may_reorder_tool_results = False


class _StubTokenizerManager:
    """Records received_time seen by the engine-facing call."""

    served_model_name = "x"
    lora_registry = None

    def __init__(self):
        self.model_config = MagicMock()
        self.model_config.context_len = 4096
        self.model_config.is_multimodal = False
        self.model_config.get_default_sampling_params.return_value = {}
        self.model_config.hf_config = MagicMock(architectures=["LlamaForCausalLM"])
        self.server_args = MagicMock(
            enable_cache_report=False,
            reasoning_parser=None,
            tool_call_parser=None,
            stream_response_default_include_usage=False,
            tokenizer_metrics_allowed_custom_labels=None,
            incremental_streaming_output=False,
        )
        self.tokenizer = MagicMock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.num_reserved_tokens = 0
        self.captured = []
        self.request_logger = MagicMock(log_requests=False)

    def config_value(self, name: str):
        return getattr(self.server_args, name)

    async def generate_request(self, obj, raw_request=None):
        self.captured.append(("generate", getattr(obj, "received_time", None)))
        yield {"text": "", "meta_info": {}}
        return

    async def score_request(self, **kwargs):
        self.captured.append(("score", kwargs.get("received_time")))
        return SimpleNamespace(
            scores=[], pooled_hidden_states=None, token_logprobs=None, prompt_tokens=0
        )

    def _resolve_score_extraction_token_id(self, token):
        return None


class ReceivedTimePropagationTest(CustomTestCase):
    def setUp(self):
        super().setUp()
        from sglang.srt.server_args import ServerArgs

        publish(ServerArgs(model_path="dummy"), role="test")
        self.addCleanup(reset_context)
        self.tm = _StubTokenizerManager()
        self.raw = MagicMock()
        self.raw.headers = {}

    def _entry_time(self):
        return monotonic_time()

    def test_embedding_control_stamps_received_time(self):
        # Control: embedding goes through handle_request and must already
        # carry an entry timestamp (existing behavior, serving_base.py:98).
        serving = OpenAIServingEmbedding(self.tm, _StubTemplateManager())
        before = self._entry_time()
        req = EmbeddingRequest(model="x", input="hi")
        _run(serving.handle_request(req, self.raw))
        self.assertEqual(len(self.tm.captured), 1)
        kind, ts = self.tm.captured[0]
        self.assertEqual(kind, "generate")
        self.assertIsNotNone(ts)
        self.assertGreaterEqual(ts, before)

    def test_score_forwards_entry_received_time(self):
        # /v1/score: handle_request must stamp ScoringRequest at entry and the
        # serving layer must forward it into score_request (RED before fix).
        serving = OpenAIServingScore(self.tm)
        before = self._entry_time()
        req = ScoringRequest(model="x", query="q", items=["i"])
        _run(serving.handle_request(req, self.raw))
        self.assertEqual(len(self.tm.captured), 1)
        kind, ts = self.tm.captured[0]
        self.assertEqual(kind, "score")
        self.assertIsNotNone(
            ts, "score path must forward entry received_time to score_request"
        )
        self.assertGreaterEqual(ts, before)

    def test_responses_stamps_internal_request_from_entry(self):
        # /v1/responses bypasses handle_request; create_responses must take a
        # timestamp at entry and stamp the internal GenerateReqInput.
        serving = OpenAIServingResponses(self.tm, _StubTemplateManager())
        serving.served_model_name = ["x"]
        serving.enable_response_store = False
        serving.tool_server = MagicMock()
        serving._process_messages = Mock(
            return_value=MessageProcessingResult(
                prompt="hi",
                prompt_ids=[1, 2, 3],
                image_data=None,
                audio_data=None,
                video_data=None,
                modalities=None,
                stop=[],
            )
        )

        captured = {}

        async def fake_generate(
            request_id,
            request_prompt,
            adapted_request,
            sampling_params,
            context,
            **kwargs,
        ):
            captured["received_time"] = adapted_request.received_time
            context.append_output(
                {
                    "text": "ok",
                    "meta_info": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "cached_tokens": 0,
                    },
                }
            )
            yield context

        serving._generate_with_builtin_tools = fake_generate
        before = self._entry_time()
        req = ResponsesRequest(model="x", input="hi", stream=False, store=False)
        _run(serving.create_responses(req, self.raw))
        self.assertIsNotNone(
            captured.get("received_time"),
            "responses internal GenerateReqInput got no entry received_time",
        )
        self.assertGreaterEqual(captured["received_time"], before)


if __name__ == "__main__":
    unittest.main()
