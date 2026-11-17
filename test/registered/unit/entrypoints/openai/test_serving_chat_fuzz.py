"""Malformed /v1/chat/completions bodies get a 4xx through the real HTTP route, and
a valid request still succeeds after each one. The model is replaced by a fake
TokenizerManager that runs the request checks done before the scheduler, and a
real tokenizer renders a jinja chat template."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import models as tokenizer_models
from tokenizers import pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.srt.entrypoints import http_server
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.request_fuzz_utils import (
    CHAT_ACCEPTED_CASES,
    CHAT_KNOWN_FAILURES,
    CHAT_VALUE_CASES,
    SEED,
    chat_nested_type_cases,
    type_cases,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

VOCAB_SIZE = 32000
VALID_BODY = {
    "model": "m",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 1,
    "temperature": 0,
}
CASES = (
    type_cases(ChatCompletionRequest)
    + chat_nested_type_cases()
    + [c for c in CHAT_VALUE_CASES if not c.needs_server]
)
# Renders roles, content and tools, like the templates of the CI models.
CHAT_TEMPLATE = (
    "{% if tools %}{{ tools | tojson }}{% endif %}"
    "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}{% endfor %}"
    "{% if add_generation_prompt %}<assistant>{% endif %}"
)


def _tokenizer():
    backend = BackendTokenizer(
        tokenizer_models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>"
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    return tokenizer


class _FakeTokenizerManager:
    def __init__(self):
        self.tokenizer = _tokenizer()
        self.model_config = Mock(
            is_multimodal=False, vocab_size=VOCAB_SIZE, context_len=4096
        )
        self.model_config.get_default_sampling_params.return_value = {}
        self.model_config.hf_config = Mock(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        self.server_args = Mock(
            enable_cache_report=False,
            reasoning_parser=None,
            tool_call_parser=None,
            stream_response_default_include_usage=False,
            incremental_streaming_output=False,
            grammar_backend="xgrammar",
            tokenizer_metrics_allowed_custom_labels=None,
        )
        self.served_model_name = "m"
        self.model_path = "m"
        self.lora_registry = None
        self.request_logger = Mock(log_requests=False)
        self.create_abort_task = Mock()

    def config_value(self, name: str):
        return getattr(self.server_args, name)

    async def generate_request(self, obj, request=None):
        obj.normalize_batch_and_arguments()
        params = SamplingParams(**obj.sampling_params)
        params.normalize(self.tokenizer)
        params.verify(VOCAB_SIZE)
        yield {
            "text": "ok",
            "meta_info": {
                "id": "chatcmpl-fuzz",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "length", "length": 1},
                "weight_version": "default",
            },
            "index": 0,
        }


class _TemplateManager:
    # The jinja path, which serves models that ship a chat template.
    chat_template_name = None
    jinja_template_content_format = "string"
    completion_template_name = None
    reasoning_config = None
    force_reasoning = False
    jinja_template_may_reorder_tool_results = False
    python_chat_template = None


class TestChatCompletionsRequestFuzz(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        state = http_server.app.state
        saved = getattr(state, "openai_serving_chat", None)
        self.addCleanup(setattr, state, "openai_serving_chat", saved)
        state.openai_serving_chat = OpenAIServingChat(
            _FakeTokenizerManager(), _TemplateManager()
        )
        # An exception that escapes the app is a 500 to a real client, not a test error.
        self.client = TestClient(http_server.app, raise_server_exceptions=False)

    def _post(self, overrides):
        return self.client.post(
            "/v1/chat/completions", json={**VALID_BODY, **overrides}
        )

    def _assert_still_serving(self):
        valid = self._post({})
        self.assertEqual(valid.status_code, 200, valid.text[:300])
        self.assertEqual(valid.json()["choices"][0]["message"]["content"], "ok")

    def test_malformed_requests_get_4xx_and_server_keeps_serving(self):
        for case in CASES:
            with self.subTest(field=case.field, case=case.case_id, seed=SEED):
                response = self._post(case.overrides)
                self._assert_still_serving()

                rejected = 400 <= response.status_code < 500
                fix = CHAT_KNOWN_FAILURES.get((case.field, case.case_id))
                if fix is not None and not rejected:
                    # Skip rather than fail, so the fix landing cannot break CI.
                    self.skipTest(f"known failure until {fix} merges")
                self.assertTrue(
                    rejected,
                    f"{case.overrides!r}: {response.status_code} {response.text[:300]}",
                )

    def test_values_accepted_on_purpose_still_succeed(self):
        for case in CHAT_ACCEPTED_CASES:
            with self.subTest(field=case.field, case=case.case_id):
                response = self._post(case.overrides)
                self.assertEqual(response.status_code, 200, response.text[:300])


if __name__ == "__main__":
    unittest.main()
