"""Exercise Anthropic effort translation through real chat-template rendering.

No model is loaded: only token generation is stubbed. The HTTP, Anthropic,
OpenAI, Hugging Face tokenizer, and Jinja validation paths are real.
"""

import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat  # noqa: E402
from sglang.srt.runtime_context import get_context  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestAnthropicEffortHTTP(CustomTestCase):
    def setUp(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(
                WordLevel({"[UNK]": 0, "hi": 1}, unk_token="[UNK]")
            ),
            unk_token="[UNK]",
            chat_template=(
                "{% if reasoning_effort is defined and "
                "reasoning_effort not in ['xhigh', 'medium', 'low'] %}"
                "{{ raise_exception('Unexpected reasoning effort ' + reasoning_effort"
                " + '. Supported types are xhigh, medium, and low.') }}"
                "{% endif %}{{ messages[0]['content'] }}"
            ),
        )
        config = get_context().override_server_args(
            enable_cache_report=False,
            tool_call_parser=None,
            reasoning_parser=None,
            default_chat_template_kwargs=None,
            context_length=2048,
            allow_auto_truncate=False,
            enable_lora=False,
            stream_response_default_include_usage=False,
            incremental_streaming_output=False,
        )
        server_args = config.install()
        self.addCleanup(config.restore)
        self.generated_requests = []

        async def generate_request(request, *args, **kwargs):
            self.generated_requests.append(request)
            yield {
                "text": "ok",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "weight_version": "default",
                    "finish_reason": {"type": "stop", "matched": None},
                },
                "index": 0,
            }

        tokenizer_manager = SimpleNamespace(
            tokenizer=tokenizer,
            server_args=server_args,
            model_config=SimpleNamespace(
                is_multimodal=False,
                hf_config=SimpleNamespace(
                    model_type="qwen3", architectures=["Qwen3ForCausalLM"]
                ),
                get_default_sampling_params=lambda: {},
            ),
            config_value=lambda name: getattr(server_args, name),
            model_path="test-model",
            served_model_name="test-model",
            preferred_sampling_params={},
            generate_request=generate_request,
            create_abort_task=lambda _: None,
            request_logger=SimpleNamespace(log_requests=False, log_requests_level=0),
        )
        template_manager = SimpleNamespace(
            chat_template_name=None,
            jinja_template_content_format="string",
            completion_template_name=None,
            reasoning_config=None,
            force_reasoning=False,
            jinja_template_may_reorder_tool_results=False,
        )
        serving = AnthropicServing(
            OpenAIServingChat(tokenizer_manager, template_manager)
        )
        app = FastAPI()

        @app.post("/v1/messages")
        async def messages(request: AnthropicMessagesRequest, raw_request: Request):
            return await serving.handle_messages(request, raw_request)

        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def _request(self, effort, stream):
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 16,
            "stream": stream,
        }
        if effort is not None:
            payload["output_config"] = {"effort": effort}
        return self.client.post("/v1/messages", json=payload)

    def test_template_supported_effort_returns_success(self):
        for stream in (False, True):
            for effort in (None, "low", "medium", "xhigh"):
                with self.subTest(stream=stream, effort=effort):
                    before = len(self.generated_requests)
                    response = self._request(effort, stream)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(len(self.generated_requests), before + 1)
                    if stream:
                        self.assertIn(
                            "text/event-stream", response.headers["content-type"]
                        )
                        events = [
                            json.loads(line[6:])
                            for line in response.text.splitlines()
                            if line.startswith("data: ")
                        ]
                        self.assertNotIn("error", [event["type"] for event in events])
                        self.assertEqual(events[-1]["type"], "message_stop")
                        self.assertEqual(
                            [
                                event["delta"]["text"]
                                for event in events
                                if event["type"] == "content_block_delta"
                            ],
                            ["ok"],
                        )
                    else:
                        self.assertEqual(
                            response.json()["content"], [{"type": "text", "text": "ok"}]
                        )
                        self.assertEqual(response.json()["stop_reason"], "end_turn")

    def test_template_rejected_effort_returns_400_before_generation(self):
        for stream in (False, True):
            for effort in ("high", "max"):
                with self.subTest(stream=stream, effort=effort):
                    response = self._request(effort, stream)
                    self.assertEqual(response.status_code, 400, response.text)
                    self.assertEqual(
                        response.json(),
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": (
                                    f"Unexpected reasoning effort {effort}. "
                                    "Supported types are xhigh, medium, and low."
                                ),
                            },
                        },
                    )
                    self.assertEqual(self.generated_requests, [])


if __name__ == "__main__":
    unittest.main()
