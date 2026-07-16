"""Unit tests for served-model validation on the OpenAI serving layer.

These exercise ``OpenAIServingBase.validate_served_model`` and its wiring into
``handle_request`` (chat / completions) and ``create_responses`` (responses).

Note: the "valid" cases assert that ``validate_served_model`` returns ``None``
(i.e. the request is allowed to proceed and is not rejected with a 404).
Asserting a full HTTP 200 generation would require running the model pipeline,
which needs a GPU / engine and is out of scope for a CPU unit test.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi import Request

# Importing utils first installs the CPU stubs (sgl_kernel + torch.compile)
# needed before the serving modules are imported.
from utils import MockTemplateManager

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ResponsesRequest,
    V1RerankReqInput,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SERVED_MODEL = "served-model"
LORA_ADAPTER = "my-lora"


def _make_tokenizer_manager(*, enable_lora=False, adapters=()):
    tm = Mock()
    tm.model_config = Mock(is_multimodal=False, context_len=4096)
    tm.model_config.get_default_sampling_params.return_value = {}
    tm.model_config.hf_config = Mock(
        model_type="llama", architectures=["LlamaForCausalLM"]
    )
    tm.server_args = Mock(
        enable_cache_report=False,
        reasoning_parser=None,
        tool_call_parser=None,
        default_chat_template_kwargs=None,
        stream_response_default_include_usage=False,
        tokenizer_metrics_allowed_custom_labels=None,
        incremental_streaming_output=False,
        enable_lora=enable_lora,
    )
    tm.served_model_name = SERVED_MODEL
    tm.tokenizer = Mock()

    lora_registry = Mock()
    lora_registry.get_all_adapters.return_value = {
        name: SimpleNamespace(lora_name=name, lora_path=f"/loras/{name}")
        for name in adapters
    }
    tm.lora_registry = lora_registry
    return tm


class ValidateServedModelTestCase(unittest.TestCase):
    def setUp(self):
        self.tm = _make_tokenizer_manager()
        self.template_manager = MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.completion = OpenAIServingCompletion(self.tm, self.template_manager)
        self.responses = OpenAIServingResponses(self.tm, self.template_manager)
        self.raw_request = Mock(spec=Request)
        self.raw_request.headers = {}

    @staticmethod
    def _error_body(response):
        return json.loads(response.body)["error"]

    # (a) unknown model -> 404 model_not_found for chat / completions / responses
    def test_unknown_model_chat_returns_404(self):
        request = ChatCompletionRequest(
            model="ghost-model",
            messages=[{"role": "user", "content": "hi"}],
        )
        result = asyncio.run(self.chat.handle_request(request, self.raw_request))
        self.assertEqual(result.status_code, 404)
        self.assertEqual(self._error_body(result)["code"], "model_not_found")

    def test_unknown_model_completion_returns_404(self):
        request = CompletionRequest(model="ghost-model", prompt="hi")
        result = asyncio.run(self.completion.handle_request(request, self.raw_request))
        self.assertEqual(result.status_code, 404)
        self.assertEqual(self._error_body(result)["code"], "model_not_found")

    def test_unknown_model_responses_returns_404(self):
        request = ResponsesRequest(model="ghost-model", input="hi", store=False)
        result = asyncio.run(self.responses.create_responses(request, raw_request=None))
        self.assertEqual(result.status_code, 404)
        self.assertEqual(self._error_body(result)["code"], "model_not_found")

    # (b) the real served model name is accepted
    def test_served_model_name_is_valid(self):
        request = ChatCompletionRequest(
            model=SERVED_MODEL,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNone(self.chat.validate_served_model(request))

    # (c) an omitted model (default sentinel / None) is accepted
    def test_default_sentinel_model_is_valid(self):
        # CompletionRequest omits ``model`` -> defaults to the sentinel value.
        completion_request = CompletionRequest(prompt="hi")
        self.assertIsNone(self.completion.validate_served_model(completion_request))
        # ResponsesRequest leaves ``model`` as None when omitted.
        responses_request = ResponsesRequest(input="hi", store=False)
        self.assertIsNone(self.responses.validate_served_model(responses_request))

    # (d) a loaded LoRA adapter name is accepted
    def test_loaded_lora_adapter_is_accepted(self):
        tm = _make_tokenizer_manager(enable_lora=True, adapters=(LORA_ADAPTER,))
        chat = OpenAIServingChat(tm, self.template_manager)

        bare_adapter = ChatCompletionRequest(
            model=LORA_ADAPTER,
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNone(chat.validate_served_model(bare_adapter))

        # "base-model:adapter" colon syntax is accepted via the base model.
        colon_syntax = ChatCompletionRequest(
            model=f"{SERVED_MODEL}:{LORA_ADAPTER}",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNone(chat.validate_served_model(colon_syntax))

        # An adapter that is not loaded is still rejected.
        not_loaded = ChatCompletionRequest(
            model="not-loaded",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertIsNotNone(chat.validate_served_model(not_loaded))

    # (e) a rerank request has no ``model`` field and is unaffected
    def test_rerank_request_is_unaffected(self):
        request = V1RerankReqInput(query="q", documents=["d1", "d2"])
        self.assertIsNone(self.chat.validate_served_model(request))


if __name__ == "__main__":
    unittest.main()
