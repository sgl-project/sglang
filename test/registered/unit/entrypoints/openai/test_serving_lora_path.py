"""The adapter named by ``model="base:adapter"``, or by an explicit
``lora_path``, must reach the engine as ``GenerateReqInput.lora_path`` on both
OpenAI endpoints; a dropped value silently serves the request on the base model."""

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from unittest.mock import Mock, patch

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    MessageProcessingResult,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SERVED_MODEL_NAME = "test-model"

# name -> (request model, explicit lora_path, lora_path the engine must receive)
CASES = {
    "suffix on the served name": (f"{SERVED_MODEL_NAME}:adapter", None, "adapter"),
    "explicit lora_path": (SERVED_MODEL_NAME, "adapter", "adapter"),
    "suffix wins over explicit": (f"{SERVED_MODEL_NAME}:adapter", "other", "adapter"),
    "no adapter": (SERVED_MODEL_NAME, None, None),
}

PROCESSED_MESSAGES = MessageProcessingResult(
    prompt="rendered",
    prompt_ids=[1, 2, 3],
    image_data=None,
    audio_data=None,
    video_data=None,
    modalities=[],
    stop=[],
)


class _MockTokenizerManager:
    """The manager attributes the two serving constructors and conversions read."""

    def __init__(self):
        self.server_args = Mock(
            model_path="dummy",
            revision=None,
            enable_cache_report=False,
            tool_call_parser=None,
            reasoning_parser=None,
            default_chat_template_kwargs=None,
        )
        self.model_path = self.server_args.model_path
        self.served_model_name = SERVED_MODEL_NAME
        self.model_config = Mock(is_multimodal=False)
        self.model_config.get_default_sampling_params.return_value = {}
        self.model_config.hf_config = Mock(
            model_type="llama", architectures=["LlamaForCausalLM"]
        )
        self.model_config.hf_config.to_dict.return_value = {}
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "decoded"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

    def config_value(self, name):
        return getattr(self.server_args, name)


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name = "llama-3"
        self.jinja_template_content_format = None
        self.completion_template_name = None
        self.reasoning_config = None
        self.force_reasoning = False
        self.jinja_template_may_reorder_tool_results = False
        self.python_chat_template = None


class TestServingLoRAPath(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.tokenizer_manager = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()

    def test_chat_completion_carries_resolved_lora_path(self):
        chat = OpenAIServingChat(self.tokenizer_manager, self.template_manager)
        for name, (model, explicit, expected) in CASES.items():
            with self.subTest(name):
                request = ChatCompletionRequest(
                    model=model,
                    messages=[{"role": "user", "content": "Hi"}],
                    lora_path=explicit,
                    max_tokens=1,
                )
                with patch.object(
                    chat, "_process_messages", return_value=PROCESSED_MESSAGES
                ):
                    adapted, _ = chat._convert_to_internal_request(request)
                self.assertEqual(adapted.lora_path, expected)

    def test_completion_carries_resolved_lora_path(self):
        completion = OpenAIServingCompletion(
            self.tokenizer_manager, self.template_manager
        )
        for name, (model, explicit, expected) in CASES.items():
            with self.subTest(name):
                request = CompletionRequest(
                    model=model, prompt="Hi", lora_path=explicit, max_tokens=1
                )
                adapted, _ = completion._convert_to_internal_request(request)
                self.assertEqual(adapted.lora_path, expected)


if __name__ == "__main__":
    unittest.main()
