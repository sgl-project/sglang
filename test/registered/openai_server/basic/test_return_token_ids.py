"""
Unit tests for the return_prompt_token_ids feature in ChatCompletion endpoint.

Tests that:
1. Protocol models correctly handle return_prompt_token_ids / prompt_token_ids fields
2. Request conversion passes return_prompt_token_ids flag through
3. Non-streaming response includes prompt_token_ids
4. Fields are omitted from JSON when return_prompt_token_ids is False (default)

Run with:
    python -m pytest test/registered/openai_server/basic/test_return_token_ids.py -v
or:
    python test/registered/openai_server/basic/test_return_token_ids.py -v
"""

import json
import sys
import unittest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock out heavy GPU dependencies so tests run on CPU-only machines.
# We install a MagicMock for every missing module in the import chain.
# ---------------------------------------------------------------------------

_GPU_MODULES = [
    # PyTorch
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.parameter",
    "torch.cuda", "torch.distributed", "torch.library", "torch.utils",
    "torch.utils.checkpoint", "torch.fx", "torch.profiler",
    "torch.autograd", "torch.amp", "torch.optim",
    # Triton
    "triton", "triton.language", "triton.runtime",
    # SGLang kernel / vLLM / transformers
    "sgl_kernel", "vllm", "vllm.config", "vllm.model_executor",
    "transformers", "transformers.models", "outlines",
    # CUDA-specific
    "cuda", "cupy", "numba",
    # Packaging
    "packaging", "packaging.version",
]

_mock_cache = {}

for mod_name in _GPU_MODULES:
    if mod_name not in sys.modules:
        mock = MagicMock()
        sys.modules[mod_name] = mock
        _mock_cache[mod_name] = mock

# ---------------------------------------------------------------------------
# Now safe to import sglang modules
# ---------------------------------------------------------------------------

import asyncio
from typing import List, Optional

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)

# These may fail on CPU-only if the import chain hits something we missed.
# We protect with try/except and skip the dependent tests.
_HAS_IO_STRUCT = False
_HAS_SERVING_CHAT = False
_HAS_TOKENIZER_MANAGER = False

try:
    from sglang.srt.managers.io_struct import GenerateReqInput
    _HAS_IO_STRUCT = True
except Exception:
    pass

try:
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    from sglang.srt.entrypoints.openai.protocol import MessageProcessingResult
    _HAS_SERVING_CHAT = True
except Exception:
    pass

try:
    from sglang.srt.managers.tokenizer_manager import ReqState
    _HAS_TOKENIZER_MANAGER = True
except Exception:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MOCK_PROMPT_TOKEN_IDS = [128000, 882, 1234, 5678, 9012]
MOCK_OUTPUT_TOKEN_IDS = [100, 200, 300]


# ===========================================================================
# 1. Protocol Tests — pure Pydantic model serialization (always runnable)
# ===========================================================================


class TestReturnTokenIdsProtocol(unittest.TestCase):
    """Test protocol model fields for return_prompt_token_ids."""

    # --- Request ---

    def test_request_default_false(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
        )
        self.assertFalse(req.return_prompt_token_ids)

    def test_request_explicit_true(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            return_prompt_token_ids=True,
        )
        self.assertTrue(req.return_prompt_token_ids)

    # --- Response (non-streaming) ---

    def test_choice_omits_prompt_token_ids_when_none(self):
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="hi"),
            finish_reason="stop",
        )
        data = choice.model_dump()
        self.assertNotIn("prompt_token_ids", data)

    def test_choice_includes_prompt_token_ids_when_set(self):
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="hi"),
            finish_reason="stop",
            prompt_token_ids=[1, 2, 3],
        )
        data = choice.model_dump()
        self.assertIn("prompt_token_ids", data)
        self.assertEqual(data["prompt_token_ids"], [1, 2, 3])

    # --- Full JSON round-trip ---

    def test_full_response_json_with_prompt_token_ids(self):
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="hello"),
            finish_reason="stop",
            prompt_token_ids=MOCK_PROMPT_TOKEN_IDS,
        )
        resp = ChatCompletionResponse(
            id="test-id",
            model="test",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        data = json.loads(resp.model_dump_json())
        self.assertEqual(data["choices"][0]["prompt_token_ids"], MOCK_PROMPT_TOKEN_IDS)

    def test_full_response_json_without_token_ids(self):
        choice = ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="hello"),
            finish_reason="stop",
        )
        resp = ChatCompletionResponse(
            id="test-id",
            model="test",
            choices=[choice],
            usage=UsageInfo(prompt_tokens=5, completion_tokens=3, total_tokens=8),
        )
        data = json.loads(resp.model_dump_json())
        self.assertNotIn("prompt_token_ids", data["choices"][0])


# ===========================================================================
# 2. GenerateReqInput Tests
# ===========================================================================


@unittest.skipUnless(_HAS_IO_STRUCT, "io_struct import requires GPU deps")
class TestReturnTokenIdsIOStruct(unittest.TestCase):
    """Test GenerateReqInput return_prompt_token_ids field."""

    def test_default_false(self):
        req = GenerateReqInput(text="hello")
        self.assertFalse(req.return_prompt_token_ids)

    def test_explicit_true(self):
        req = GenerateReqInput(text="hello", return_prompt_token_ids=True)
        self.assertTrue(req.return_prompt_token_ids)

    def test_does_not_affect_logprob_fields(self):
        req = GenerateReqInput(
            text="hello",
            return_prompt_token_ids=True,
            return_logprob=False,
            logprob_start_len=-1,
        )
        self.assertTrue(req.return_prompt_token_ids)
        self.assertFalse(req.return_logprob)
        self.assertEqual(req.logprob_start_len, -1)


# ===========================================================================
# 3. Request Conversion Tests
# ===========================================================================


@unittest.skipUnless(
    _HAS_SERVING_CHAT, "OpenAIServingChat import requires GPU deps"
)
class TestReturnTokenIdsRequestConversion(unittest.TestCase):
    """Test that return_prompt_token_ids flows through _convert_to_internal_request."""

    def setUp(self):
        from unittest.mock import Mock, patch

        tm = Mock()
        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser=None,
            reasoning_parser=None,
        )
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        tm.model_config.hf_config = mock_hf_config
        tm.chat_template_name = "llama-3"
        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3]
        tm.tokenizer.chat_template = None
        tm.tokenizer.bos_token_id = 1

        template_mgr = Mock()
        template_mgr.chat_template_name = "llama-3"
        template_mgr.jinja_template_content_format = None
        template_mgr.completion_template_name = None
        template_mgr.force_reasoning = False

        self.chat = OpenAIServingChat(tm, template_mgr)

    def _convert(self, return_prompt_token_ids: bool):
        from unittest.mock import patch

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            return_prompt_token_ids=return_prompt_token_ids,
        )
        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = MessageProcessingResult(
                "Test prompt", [1, 2, 3], None, None, [], ["</s>"], None,
            )
            adapted, _ = self.chat._convert_to_internal_request(req)
        return adapted

    def test_flag_passed_when_true(self):
        adapted = self._convert(return_prompt_token_ids=True)
        self.assertIsInstance(adapted, GenerateReqInput)
        self.assertTrue(adapted.return_prompt_token_ids)

    def test_flag_passed_when_false(self):
        adapted = self._convert(return_prompt_token_ids=False)
        self.assertIsInstance(adapted, GenerateReqInput)
        self.assertFalse(adapted.return_prompt_token_ids)

    def test_logprob_not_affected(self):
        adapted = self._convert(return_prompt_token_ids=True)
        self.assertFalse(adapted.return_logprob)
        self.assertEqual(adapted.logprob_start_len, -1)

    def test_stream_with_return_prompt_token_ids_raises(self):
        """return_prompt_token_ids=True + stream=True should raise ValueError."""
        from unittest.mock import patch

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            return_prompt_token_ids=True,
            stream=True,
        )
        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = MessageProcessingResult(
                "Test prompt", [1, 2, 3], None, None, [], ["</s>"], None,
            )
            with self.assertRaises(ValueError):
                self.chat._convert_to_internal_request(req)


# ===========================================================================
# 4. Response Building Tests
# ===========================================================================


@unittest.skipUnless(
    _HAS_SERVING_CHAT, "OpenAIServingChat import requires GPU deps"
)
class TestReturnTokenIdsResponseBuilding(unittest.TestCase):
    """Test _build_chat_response includes prompt_token_ids when requested."""

    def setUp(self):
        from unittest.mock import Mock

        tm = Mock()
        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser=None,
            reasoning_parser=None,
        )
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        tm.model_config.hf_config = mock_hf_config
        tm.chat_template_name = "llama-3"
        tm.tokenizer = Mock()
        tm.tokenizer.chat_template = None
        tm.tokenizer.bos_token_id = 1

        template_mgr = Mock()
        template_mgr.chat_template_name = "llama-3"
        template_mgr.jinja_template_content_format = None
        template_mgr.completion_template_name = None
        template_mgr.force_reasoning = False

        self.chat = OpenAIServingChat(tm, template_mgr)

    def _make_ret(
        self,
        include_prompt_token_ids: bool = False,
    ):
        ret = {
            "text": "Test response",
            "output_ids": MOCK_OUTPUT_TOKEN_IDS,
            "meta_info": {
                "id": "chatcmpl-test",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "weight_version": "default",
            },
        }
        if include_prompt_token_ids:
            ret["prompt_token_ids"] = MOCK_PROMPT_TOKEN_IDS
        return ret

    def test_with_return_prompt_token_ids(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            return_prompt_token_ids=True,
        )
        ret = [self._make_ret(include_prompt_token_ids=True)]
        response = self.chat._build_chat_response(req, ret, created=0)

        self.assertIsInstance(response, ChatCompletionResponse)
        self.assertEqual(response.choices[0].prompt_token_ids, MOCK_PROMPT_TOKEN_IDS)

    def test_without_return_prompt_token_ids(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
        )
        ret = [self._make_ret(include_prompt_token_ids=False)]
        response = self.chat._build_chat_response(req, ret, created=0)

        self.assertIsNone(response.choices[0].prompt_token_ids)

        data = json.loads(response.model_dump_json())
        self.assertNotIn("prompt_token_ids", data["choices"][0])

    def test_json_round_trip(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            return_prompt_token_ids=True,
        )
        ret = [self._make_ret(include_prompt_token_ids=True)]
        response = self.chat._build_chat_response(req, ret, created=0)

        data = json.loads(response.model_dump_json())
        self.assertEqual(data["choices"][0]["prompt_token_ids"], MOCK_PROMPT_TOKEN_IDS)


# ===========================================================================
# 5. ReqState Tests
# ===========================================================================


@unittest.skipUnless(
    _HAS_TOKENIZER_MANAGER, "tokenizer_manager import requires GPU deps"
)
class TestReturnTokenIdsReqState(unittest.TestCase):
    """Test that ReqState stores prompt_token_ids correctly."""

    def test_reqstate_default_none(self):
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=MagicMock(),
            time_stats=MagicMock(),
        )
        self.assertIsNone(state.prompt_token_ids)

    def test_reqstate_stores_prompt_token_ids(self):
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=MagicMock(),
            time_stats=MagicMock(),
        )
        state.prompt_token_ids = MOCK_PROMPT_TOKEN_IDS
        self.assertEqual(state.prompt_token_ids, MOCK_PROMPT_TOKEN_IDS)


if __name__ == "__main__":
    unittest.main()
