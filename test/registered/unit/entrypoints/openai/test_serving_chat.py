"""
Unit-tests for OpenAIServingChat -- rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

from sglang.test.test_utils import CustomTestCase, enter_override, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import gc
import json
import unittest
import uuid
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request
from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import models as tokenizer_models
from tokenizers import pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import _compile_jinja_template

from sglang.srt.entrypoints.openai.chat_encoding import resolve_chat_encoding_spec
from sglang.srt.entrypoints.openai.parallel_prompt_encode import encode_prompt_segments
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    MessageProcessingResult,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.entrypoints.openai.serving_chat import (
    OpenAIServingChat,
    normalize_tool_content,
)
from sglang.srt.environ import envs
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik3_format import TOOLS_CLOSE, TOOLS_OPEN
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.jinja_template_utils import (
    jinja_template_may_reorder_tool_results,
)
from sglang.srt.parser.python_chat_template import GLM53, install_python_chat_template
from sglang.srt.parser.response_template import (
    GRAMMAR_AFTER_REASONING,
    GRAMMAR_FROM_START,
    ResponseTemplateReasoningDetector,
    ResponseTemplateToolDetector,
)
from sglang.srt.parser.template_detection import ReasoningToggleConfig
from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.sampling.sampling_params import (
    REQUEST_REASONING_END_TOKEN_IDS_KEY,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=13, suite="base-a-test-cpu")

# Every spec resolve_chat_encoding_spec can return; pinned by the guard below.
_ALL_CHAT_ENCODING_SPECS = ("dsv41", "dsv4", "dsv32", "inkling", "kimi_k3")

_RESPONSE_TEMPLATE = {
    "start_anchor": "<assistant>",
    "fields": {
        "content": {"content": "text"},
        "tool_calls": {
            "open_pattern": r"<call:(?P<name>\w+)>",
            "close": "</call>",
            "content": "json",
            "repeats": True,
            "transform": {
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}",
                },
            },
        },
    },
}
# Every field is delimited, so the template gives no field to text outside them.
_DELIMITED_RESPONSE_TEMPLATE = {
    "start_anchor": "<assistant>",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>"},
        "content": {"open": "<answer>", "close": "</answer>"},
        "tool_calls": _RESPONSE_TEMPLATE["fields"]["tool_calls"],
    },
}
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
        },
    },
}


def _spec_result(index):
    return {
        "text": f"choice-{index}",
        "meta_info": {
            "id": "chatcmpl-spec-test",
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "cached_tokens": 0,
            "finish_reason": {"type": "stop"},
            "weight_version": "default",
            "spec_accept_rate": 0.5,
            "spec_accept_length": 2.0,
            "spec_cap_length": index + 1.0,
            "spec_block_accept_length": index + 0.5,
            "spec_num_correct_drafts": 1,
            "spec_num_proposed_drafts": 2,
            "spec_verify_ct": 1,
            "spec_correct_drafts_histogram": [0, 1],
            "spec_cap_lens_histogram": [index, 1],
        },
        "index": index,
    }


_DSV4_PREVIEW_ENCODER = 'REASONING_EFFORT_MAX = "preview"\n'
_DSV4_OFFICIAL_ENCODER = (
    "REASONING_EFFORT_PROMPTS: Dict[str, str] = "
    '{"low": "", "high": "h", "max": "m"}\n'
    'DEFAULT_REASONING_EFFORT = "low"\n'
)

_TOOL_RESULT_REORDER_TEMPLATE = """
{%- for assistant in messages
    if assistant.role == 'assistant' and assistant.tool_calls -%}
    {%- for tool_call in assistant.tool_calls -%}
        {%- for result in messages
            if result.role == 'tool' and result.tool_call_id == tool_call.id -%}
            {%- for part in result.content -%}
                {%- if part.type == 'text' -%}
                    {{- part.text -}}
                {%- else -%}
                    {{- '<' + part.type + '>' -}}
                {%- endif -%}
            {%- endfor -%}
        {%- endfor -%}
    {%- endfor -%}
{%- endfor -%}
"""


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            model_path="deepseek-ai/DeepSeek-V4-Flash",
            revision=None,
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
            default_chat_template_kwargs=None,
            return_input_ids=False,
            return_output_ids=False,
            incremental_streaming_output=False,
        )
        self.model_path = self.server_args.model_path
        # The manager tracks the served name itself; a weight update rewrites it.
        self.served_model_name = "test-model"
        # Stands in for the context's resolved leaves: an override replaces the
        # field's one live value, the seed stays on server_args.
        self._config_overrides = {}

        # Mock hf_config for _resolve_chat_encoding_spec check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        mock_hf_config.to_dict.return_value = {}
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "weight_version": "test-version",
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()
        self.request_logger = Mock(log_requests=False, log_requests_level=0)

    def config_value(self, name: str):
        """The value in effect for one config field."""
        if name in self._config_overrides:
            return self._config_overrides[name]
        return getattr(self.server_args, name)


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False
        self.jinja_template_may_reorder_tool_results = False
        self.python_chat_template = None


class TestChatTemplateCache(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", default_chat_template_kwargs=None),
            role="tokenizer",
        )
        self.tokenizer_manager = _MockTokenizerManager()
        self.chat = OpenAIServingChat(
            self.tokenizer_manager,
            _MockTemplateManager(),
        )
        self.tokenizer_manager.tokenizer.apply_chat_template.return_value = "rendered"
        self.tokenizer_manager.tokenizer.encode.return_value = [11, 12]
        self.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.tokenizer_manager.tokenizer.reset_mock()

    def _render(self, **overrides):
        kwargs = {
            "messages": [{"role": "user", "content": "same text prefix"}],
            "tools": None,
            "template_kwargs": {"enable_thinking": False},
            "encode_kwargs": {"add_special_tokens": False},
            "use_cache": True,
        }
        kwargs.update(overrides)
        return self.chat._render_and_encode_chat_template(**kwargs)

    def test_cache_hit_reuses_render_encode_and_returns_an_owned_id_list(self):
        first = self._render()
        first[0].append(99)
        second = self._render()

        self.assertEqual(second, ([11, 12], "decoded"))
        self.tokenizer_manager.tokenizer.apply_chat_template.assert_called_once()
        self.tokenizer_manager.tokenizer.encode.assert_called_once()
        self.tokenizer_manager.tokenizer.decode.assert_called_once()

    def test_cache_key_includes_template_and_encode_options(self):
        self._render()
        self._render(template_kwargs={"enable_thinking": True})
        self._render(encode_kwargs={"add_special_tokens": True})

        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count,
            3,
        )
        self.assertEqual(self.tokenizer_manager.tokenizer.encode.call_count, 3)

    def test_cache_key_tracks_tokenizer_chat_template_updates(self):
        self.tokenizer_manager.tokenizer.chat_template = "template-v1"
        self._render()
        self.tokenizer_manager.tokenizer.chat_template = "template-v2"
        self._render()

        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count,
            2,
        )

    def test_cache_key_distinguishes_dict_key_order(self):
        """A tool schema sent in a different key order must not reuse the other order's render."""
        sorted_tools = [{"function": {"name": "f", "parameters": {"a": 1, "b": 2}}}]
        sent_tools = [{"function": {"name": "f", "parameters": {"b": 2, "a": 1}}}]
        self._render(tools=sorted_tools)
        self._render(tools=sent_tools)

        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count,
            2,
        )

    def test_non_serializable_input_bypasses_cache(self):
        messages = [{"role": "user", "content": object()}]
        self._render(messages=messages)
        self._render(messages=messages)

        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count,
            2,
        )
        self.assertEqual(self.tokenizer_manager.tokenizer.encode.call_count, 2)
        self.tokenizer_manager.tokenizer.decode.assert_not_called()


class TestPythonChatTemplateSegments(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", default_chat_template_kwargs=None),
            role="tokenizer",
        )
        backend = BackendTokenizer(
            tokenizer_models.WordLevel(vocab={"[UNK]": 0}, unk_token="[UNK]")
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", eos_token="<|endoftext|>"
        )
        self.tokenizer.add_tokens(
            ["hello", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>"]
        )
        install_python_chat_template(self.tokenizer, GLM53)
        tokenizer_manager = _MockTokenizerManager()
        tokenizer_manager.tokenizer = self.tokenizer
        template_manager = _MockTemplateManager()
        template_manager.python_chat_template = GLM53
        self.chat = OpenAIServingChat(tokenizer_manager, template_manager)
        self.messages = [{"role": "user", "content": "hello there"}]

    def _render(self, template_kwargs):
        return self.chat._render_and_encode_chat_template(
            self.messages,
            tools=None,
            template_kwargs=template_kwargs,
            encode_kwargs={},
            use_cache=False,
        )[0]

    def test_template_variables_encode_the_rendered_segments(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.encode_prompt_segments",
            wraps=encode_prompt_segments,
        ) as encode:
            ids = self._render({"reasoning_effort": "low"})
            # An apply_chat_template argument goes through apply_chat_template.
            documents_ids = self._render(
                {"reasoning_effort": "low", "documents": [{"title": "t", "text": "x"}]}
            )
        self.assertEqual(encode.call_count, 1)
        rendered = self.tokenizer.apply_chat_template(
            self.messages,
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort="low",
        )
        self.assertEqual(ids, self.tokenizer.encode(rendered))
        self.assertEqual(documents_ids, ids)


class ServingChatTestCase(CustomTestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        # The serving layer reads its config from the bags, so the fixture has
        # to publish one rather than hang the values off a mock manager.
        reset_context()
        self.addCleanup(reset_context)
        # Tests drive coroutines through get_or_create_event_loop(), which
        # creates a fresh loop per call and leaves the previous one unclosed.
        # Finalize those loops here, between tests: if the cyclic GC collects
        # one mid-import, its ResourceWarning imports tracemalloc while the
        # outer import still holds the module-lock bookkeeping, which raises
        # KeyError from importlib._bootstrap on Python < 3.12.
        self.addCleanup(self._close_event_loops)
        publish(
            ServerArgs(
                model_path="dummy",
                revision=None,
                enable_cache_report=False,
                tool_call_parser="hermes",
                reasoning_parser=None,
                stream_response_default_include_usage=False,
                default_chat_template_kwargs=None,
            ),
            role="tokenizer",
        )
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.tm.tokenizer.reset_mock()

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            session_id="session-1",
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    def test_undeclared_tools_run_the_tool_parser_only_for_the_m3_parser(self):
        """With no tools declared the M3 model can still call a tool; only its parser runs anyway."""
        user = [{"role": "user", "content": "Hi?"}]
        undeclared = (
            ChatCompletionRequest(model="x", messages=user),
            ChatCompletionRequest(model="x", messages=user, tools=[]),
        )
        for request in undeclared:
            self.assertFalse(self.chat._tool_call_parsing_active(request))

        self.chat.tool_call_parser = "minimax-m3"
        for request in undeclared:
            self.assertTrue(self.chat._tool_call_parsing_active(request))
        declared_none = ChatCompletionRequest(
            model="x",
            messages=user,
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            tool_choice="none",
        )
        self.assertFalse(self.chat._tool_call_parsing_active(declared_none))

    def test_m3_request_contract_rejects_what_the_template_cannot_honor(self):
        """The M3 format check requires 400 for these; other models keep accepting them."""
        weather = {"name": "get_weather", "arguments": "{}"}
        calls = [
            {"id": "call_1", "type": "function", "function": weather},
            {"id": "call_2", "type": "function", "function": weather},
        ]
        user = {"role": "user", "content": "Hi?"}
        cases = {
            "thinking unknown": {"messages": [user], "thinking": {"type": "off"}},
            "root not first": {
                "messages": [user, {"role": "root", "content": "You are A."}]
            },
            "unanswered call": {
                "messages": [
                    user,
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                ]
            },
            "mismatched call id": {
                "messages": [
                    user,
                    {"role": "assistant", "content": None, "tool_calls": calls[:1]},
                    {"role": "tool", "tool_call_id": "call_9", "content": "sunny"},
                ]
            },
        }
        for is_m3 in (False, True):
            self.chat.is_minimax_m3 = is_m3
            for name, fields in cases.items():
                with self.subTest(is_m3=is_m3, case=name):
                    error = self.chat._validate_request(
                        ChatCompletionRequest(model="x", **fields)
                    )
                    self.assertEqual(error is not None, is_m3)

    def test_m3_request_contract_accepts_valid_requests(self):
        """Disabled thinking, a trailing tool call, a leading root and answered calls
        in any order are valid M3 requests."""
        self.chat.is_minimax_m3 = True
        weather = {"name": "get_weather", "arguments": "{}"}
        calls = [
            {"id": "call_1", "type": "function", "function": weather},
            {"id": "call_2", "type": "function", "function": weather},
        ]
        user = {"role": "user", "content": "Hi?"}
        for fields in (
            {"messages": [user], "thinking": {"type": "disabled"}},
            {
                "messages": [
                    user,
                    {"role": "assistant", "content": None, "tool_calls": calls[:1]},
                ]
            },
            {
                "messages": [
                    {"role": "root", "content": "You are A."},
                    user,
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "tool", "tool_call_id": "call_2", "content": "rain"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
                    {"role": "user", "content": "Thanks"},
                ]
            },
        ):
            with self.subTest(fields=fields):
                request = ChatCompletionRequest(model="x", **fields)
                self.assertIsNone(self.chat._validate_request(request))

    def test_m3_thinking_mode_follows_request_intent(self):
        """thinking outranks reasoning_effort, an effort only ever turns thinking off,
        and values the template cannot use are dropped."""
        from sglang.srt.entrypoints.openai.serving_chat import (
            _normalize_minimax_m3_request,
        )

        user = [{"role": "user", "content": "Hi?"}]
        for fields, mode, effort in (
            ({"reasoning_effort": "high"}, None, "high"),
            ({"reasoning_effort": "minimal"}, "disabled", None),
            ({"reasoning_effort": "none"}, "disabled", None),
            ({"reasoning_effort": 0.5}, None, None),
            (
                {"reasoning_effort": "high", "thinking": {"type": "adaptive"}},
                "adaptive",
                "high",
            ),
            (
                {"reasoning_effort": "high", "thinking": {"type": "disabled"}},
                "disabled",
                None,
            ),
            (
                {
                    "reasoning_effort": "high",
                    "chat_template_kwargs": {"thinking_mode": "adaptive"},
                },
                "adaptive",
                "high",
            ),
        ):
            request = ChatCompletionRequest(model="x", messages=user, **fields)
            _normalize_minimax_m3_request(request)
            with self.subTest(fields=fields):
                self.assertEqual(
                    request.chat_template_kwargs.get("thinking_mode"), mode
                )
                self.assertEqual(request.reasoning_effort, effort)

    def test_m3_effort_from_template_kwargs_is_normalized(self):
        """An effort sent in chat_template_kwargs or set by the server is normalized
        like the top-level one, and a server default never overrides thinking the
        request turned off."""
        self.chat.is_minimax_m3 = True
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        user = [{"role": "user", "content": "Hi?"}]
        enabled = {"thinking_mode": "enabled"}
        # (request fields, server defaults, rendered effort, rendered thinking_mode):
        # the request's own settings outrank the server defaults, which fill gaps.
        cases = (
            (
                {"chat_template_kwargs": {"reasoning_effort": "none"}},
                {},
                None,
                "disabled",
            ),
            ({}, {"reasoning_effort": "none"}, None, "disabled"),
            ({"chat_template_kwargs": {"reasoning_effort": "high"}}, {}, "high", None),
            ({"thinking": {"type": "disabled"}}, enabled, None, "disabled"),
            ({"reasoning_effort": "none"}, enabled, None, "disabled"),
            ({}, {**enabled, "reasoning_effort": "high"}, "high", "enabled"),
        )
        for fields, server_defaults, effort, mode in cases:
            self.chat.default_chat_template_kwargs = server_defaults
            request = ChatCompletionRequest(model="x", messages=user, **fields)
            self.chat._convert_to_internal_request(request, self.fastapi_request)
            kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
            with self.subTest(fields=fields, server_defaults=server_defaults):
                self.assertEqual(kwargs.get("reasoning_effort"), effort)
                self.assertEqual(kwargs.get("thinking_mode"), mode)
        self.chat.default_chat_template_kwargs = {}

    def test_m3_tool_results_render_in_call_order(self):
        """The M3 template pairs tool results by position, so results sent out of
        order reach it in call order."""
        self.chat.is_minimax_m3 = True
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        weather = {"name": "get_weather", "arguments": "{}"}
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": weather},
                        {"id": "call_2", "type": "function", "function": weather},
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": "rain"},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
        )
        self.chat._process_messages(request, is_multimodal=False)
        rendered = self.tm.tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(
            [m["tool_call_id"] for m in rendered if m["role"] == "tool"],
            ["call_1", "call_2"],
        )

    def test_root_message_tools_are_effective_tools(self):
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "root",
                    "content": "You are A.",
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                },
                {"role": "user", "content": "Hi?"},
            ],
        )
        self.assertEqual(
            [tool.function.name for tool in self.chat._effective_tools(request)],
            ["lookup"],
        )
        self.assertEqual(request.tool_choice, "auto")

    @staticmethod
    def _close_event_loops():
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and not loop.is_closed():
            loop.close()
        asyncio.set_event_loop(None)
        gc.collect()

    @staticmethod
    def _render_tool_results_in_call_order(messages, **kwargs):
        """Block-level tool_call_id association, like the GLM chat templates."""
        del kwargs
        rendered = []
        index = 0
        while index < len(messages):
            message = messages[index]
            index += 1
            tool_calls = message.get("tool_calls") or []
            if message.get("role") != "assistant" or not tool_calls:
                continue
            run = []
            while index < len(messages) and messages[index].get("role") == "tool":
                run.append(messages[index])
                index += 1
            by_id = {result.get("tool_call_id"): result for result in run}
            for tool_call in tool_calls:
                result = by_id.get(tool_call.get("id"))
                if result is None:
                    continue
                for part in result.get("content") or []:
                    if part.get("type") == "text":
                        rendered.append(part.get("text", ""))
                    else:
                        rendered.append(f"<{part.get('type')}>")
        return "".join(rendered)

    @staticmethod
    def _tool_round(call_ids, result_ids, part_type="image_url"):
        assistant = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": call_id, "function": {"name": call_id, "arguments": {}}}
                for call_id in call_ids
            ],
        }
        part_key = {"image_url": "image_url", "video_url": "video_url"}[part_type]
        results = [
            {
                "role": "tool",
                "tool_call_id": result_id,
                "content": [{"type": part_type, part_key: {"url": result_id}}],
            }
            for result_id in result_ids
        ]
        return [assistant] + results

    def test_canonicalize_tool_message_order_sorts_media_runs(self):
        messages = self._tool_round(
            ["call-a", "call-b"], ["call-b", "call-a"]
        ) + self._tool_round(["call-c", "call-d"], ["call-d", "call-c"], "video_url")

        canonical = self.chat._canonicalize_tool_message_order(messages)

        self.assertEqual(
            [
                message["tool_call_id"]
                for message in canonical
                if message.get("role") == "tool"
            ],
            ["call-a", "call-b", "call-c", "call-d"],
        )

    def test_canonicalize_tool_message_order_keeps_unassociable_runs(self):
        cases = {
            "unknown_id": (["call-a", "call-b"], ["call-b", "call-z"]),
            "duplicate_result_id": (["call-a", "call-b"], ["call-b", "call-b"]),
            "missing_call_id": ([None, "call-b"], ["call-b", None]),
        }
        for name, (call_ids, result_ids) in cases.items():
            with self.subTest(name=name):
                messages = self._tool_round(call_ids, result_ids)
                canonical = self.chat._canonicalize_tool_message_order(messages)
                self.assertEqual(canonical, messages)

    def test_canonicalize_tool_message_order_keeps_text_only_runs(self):
        messages = self._tool_round(["call-a", "call-b"], ["call-b", "call-a"])
        for message in messages[1:]:
            message["content"] = [{"type": "text", "text": "done"}]

        canonical = self.chat._canonicalize_tool_message_order(messages)

        self.assertEqual(canonical, messages)

    def test_jinja_path_recovers_tool_result_images_only_when_template_needs_it(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "openai"
        self.template_manager.jinja_template_may_reorder_tool_results = (
            jinja_template_may_reorder_tool_results(_TOOL_RESULT_REORDER_TEMPLATE)
        )
        self.assertTrue(self.template_manager.jinja_template_may_reorder_tool_results)
        self.tm.tokenizer.apply_chat_template.side_effect = (
            self._render_tool_results_in_call_order
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "inspect"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-a",
                            "type": "function",
                            "function": {"name": "a", "arguments": {}},
                        },
                        {
                            "id": "call-b",
                            "type": "function",
                            "function": {"name": "b", "arguments": {}},
                        },
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-b",
                    "content": [{"type": "image_url", "image_url": {"url": "image-b"}}],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call-a",
                    "content": [{"type": "image_url", "image_url": {"url": "image-a"}}],
                },
            ],
        )

        result = self.chat._apply_jinja_template(request, None, is_multimodal=True)

        self.assertEqual(
            [item.url for item in result.image_data], ["image-a", "image-b"]
        )
        self.assertEqual(self.tm.tokenizer.apply_chat_template.call_count, 1)
        rendered_messages = self.tm.tokenizer.apply_chat_template.call_args[0][0]
        self.assertEqual(
            [m.get("tool_call_id") for m in rendered_messages if "tool_call_id" in m],
            ["call-a", "call-b"],
        )

        # An already-ordered request must produce the exact same render input.
        ordered_request = ChatCompletionRequest(
            model="x",
            messages=request.messages[:2] + request.messages[2:][::-1],
        )
        self.tm.tokenizer.apply_chat_template.reset_mock()
        self.chat._apply_jinja_template(ordered_request, None, is_multimodal=True)
        self.tm.tokenizer.apply_chat_template.assert_not_called()

        self.template_manager.jinja_template_may_reorder_tool_results = False
        self.tm.tokenizer.apply_chat_template.reset_mock()
        result = self.chat._apply_jinja_template(request, None, is_multimodal=True)
        self.assertEqual(
            [item.url for item in result.image_data], ["image-b", "image-a"]
        )
        self.assertEqual(self.tm.tokenizer.apply_chat_template.call_count, 1)

    def test_parsers_follow_the_control_plane_overlay(self):
        """Template detection records the parsers through `override`, so they
        answer from the bags; `ServerArgs` keeps the launcher's seed."""
        self.tm.server_args.tool_call_parser = "auto"
        self.tm.server_args.reasoning_parser = "auto"
        self.tm._config_overrides.update(
            {"tool_call_parser": "qwen25", "reasoning_parser": None}
        )

        chat = OpenAIServingChat(self.tm, self.template_manager)

        self.assertEqual(chat.tool_call_parser, "qwen25")
        self.assertIsNone(chat.reasoning_parser)
        self.assertEqual(self.tm.server_args.tool_call_parser, "auto")

    def test_the_xgrammar_gate_follows_the_overlay(self):
        """A detected `reasoning_parser` must gate xgrammar, not the seed's "auto"."""
        self.tm.server_args.reasoning_parser = "auto"
        self.tm._config_overrides["reasoning_parser"] = "qwen3"
        chat = OpenAIServingChat(self.tm, self.template_manager)
        self.assertEqual(chat.reasoning_parser, "qwen3")
        # the gate reads the same value the parser was built from
        self.assertIsNotNone(chat.reasoning_parser)

    def test_text_only_model_rejects_media_before_generation(self):
        media_parts = {
            "image_url": {
                "type": "image_url",
                "image_url": {"url": "https://example.com/image.png"},
            },
            "video_url": {
                "type": "video_url",
                "video_url": {"url": "https://example.com/video.mp4"},
            },
            "audio_url": {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/audio.wav"},
            },
        }

        for media_type, media_part in media_parts.items():
            with self.subTest(media_type=media_type):
                request = ChatCompletionRequest(
                    model="x",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "describe"},
                                media_part,
                            ],
                        }
                    ],
                )
                response = get_or_create_event_loop().run_until_complete(
                    self.chat.handle_request(request, self.fastapi_request)
                )
                error = json.loads(response.body)
                self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
                self.assertEqual(error["type"], "BadRequestError")
                self.assertIn(media_type, error["message"])
        self.tm.generate_request.assert_not_called()

    def test_media_validation_does_not_reject_supported_content(self):
        text_request = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "tool_reference", "name": "get_weather"}],
                }
            ],
        )
        self.assertIsNone(self.chat._validate_request(text_request))

        self.tm.model_config.is_multimodal = True
        multimodal_request = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png"},
                        }
                    ],
                }
            ],
        )
        self.assertIsNone(self.chat._validate_request(multimodal_request))

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with (
            patch(
                "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
            ) as conv_mock,
            patch.object(self.chat, "_process_messages") as proc_mock,
        ):
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            self.basic_req.return_sampling_mask = True
            self.basic_req.sampling_logprobs_mode = "support"
            self.basic_req.return_meta_info = True
            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertTrue(adapted.return_sampling_mask)
            self.assertEqual(adapted.sampling_logprobs_mode, "support")
            self.assertEqual(adapted.session_id, "session-1")
            self.assertEqual(processed, self.basic_req)

    def test_response_template_prefix_uses_prompt_token_ids(self):
        class ResponseTemplateAlias(ResponseTemplateToolDetector):
            pass

        processed_messages = MessageProcessingResult(
            "prompt decoded with default spacing",
            [1, 2, 3],
            None,
            None,
            [],
            [],
            None,
            uses_response_template=True,
        )
        self.chat.tool_call_parser = "response-template-alias"
        self.basic_req.input_ids = [1, 2, 3]
        self.basic_req.tools = [
            Tool(function=Function(name="get_weather", parameters={}))
        ]
        self.basic_req.tool_choice = "auto"
        self.tm.tokenizer.decode.return_value = "<first><second>"

        with (
            patch.dict(
                FunctionCallParser.ToolCallParserEnum,
                {"response-template-alias": ResponseTemplateAlias},
            ),
            patch.object(
                self.chat,
                "_process_messages",
                return_value=processed_messages,
            ),
        ):
            self.assertTrue(
                self.chat._requires_response_template_detokenization(self.basic_req)
            )
            _, request = self.chat._convert_to_internal_request(self.basic_req)

        self.assertEqual(request._response_parser_prefix, "<first><second>")
        self.tm.tokenizer.decode.assert_called_once_with(
            [1, 2, 3],
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def test_response_template_prefix_uses_engine_prompt_text(self):
        processed_messages = MessageProcessingResult(
            "exact engine prompt",
            [],
            None,
            None,
            [],
            [],
            None,
        )
        self.chat.reasoning_parser = "response_template"
        self.chat._reasoning_detector = ResponseTemplateReasoningDetector(
            response_template=_RESPONSE_TEMPLATE,
            prefix="",
        )
        self.tm.model_config.is_multimodal = True

        def render(request, _):
            self.assertIsNone(request.chat_template_kwargs)
            return processed_messages

        with patch.object(
            self.chat,
            "_apply_conversation_template",
            side_effect=render,
        ):
            adapted, request = self.chat._convert_to_internal_request(self.basic_req)

        self.assertEqual(request._response_parser_prefix, "exact engine prompt")
        self.assertFalse(request.skip_special_tokens)
        self.assertFalse(adapted.sampling_params["spaces_between_special_tokens"])
        self.tm.tokenizer.decode.assert_not_called()

    def _use_delimited_response_template(
        self, reasoning_parser, template=_DELIMITED_RESPONSE_TEMPLATE
    ):
        self.tm.tokenizer.response_template = template
        self.tm.tokenizer.encode.side_effect = lambda text, **_: (
            [7] if text == "</call>" else [1, 2, 3] if text else []
        )
        self.tm.model_config.is_multimodal = True
        self.chat.tool_call_parser = "response_template"
        self.chat.reasoning_parser = reasoning_parser
        self.chat._reasoning_detector = (
            ResponseTemplateReasoningDetector(response_template=template)
            if reasoning_parser
            else None
        )
        # Always-on reasoning: the output grammar starts after the reasoning.
        self.template_manager.reasoning_config = (
            ReasoningToggleConfig(special_case="always") if reasoning_parser else None
        )

    @staticmethod
    def _stopped_on_token(text, token_id):
        return {
            "text": text,
            "meta_info": {
                "id": "chatcmpl-stopped",
                "prompt_tokens": 5,
                "completion_tokens": 9,
                "cached_tokens": 0,
                "weight_version": "default",
                "finish_reason": {"type": "stop", "matched": token_id},
            },
            "index": 0,
        }

    def _convert_rendered_as(self, prompt, request):
        rendered = MessageProcessingResult(
            prompt=prompt,
            prompt_ids=[],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        with patch.object(
            self.chat, "_apply_conversation_template", return_value=rendered
        ):
            return self.chat._convert_to_internal_request(request)

    def test_response_template_parser_context_records_grammar_start(self):
        """JSON-schema output starts after the reasoning only when the grammar
        backend waits for it, which a pattern-only reasoning closer cannot."""
        named = {"type": "function", "function": {"name": "get_weather"}}
        required = {"tools": [_WEATHER_TOOL], "tool_choice": "required"}
        pattern_closed = {
            **_DELIMITED_RESPONSE_TEMPLATE,
            "fields": {
                **_DELIMITED_RESPONSE_TEMPLATE["fields"],
                "thinking": {"open": "<think>", "close_pattern": "</think>"},
            },
        }
        cases = {
            "auto tool choice": ("response_template", {"tools": [_WEATHER_TOOL]}, None),
            "named tool after reasoning": (
                "response_template",
                {"tools": [_WEATHER_TOOL], "tool_choice": named},
                GRAMMAR_AFTER_REASONING,
            ),
            "response_format after reasoning": (
                "response_template",
                {"response_format": {"type": "json_object"}},
                GRAMMAR_AFTER_REASONING,
            ),
            "required tool without reasoning": (None, required, GRAMMAR_FROM_START),
            "required tool, reasoning without an end token": (
                "response_template",
                required,
                GRAMMAR_FROM_START,
                pattern_closed,
            ),
        }
        for name, (reasoning_parser, fields, grammar_start, *template) in cases.items():
            with self.subTest(name):
                self._use_delimited_response_template(reasoning_parser, *template)
                _, request = self._convert_rendered_as(
                    "<assistant>",
                    ChatCompletionRequest(
                        model="x",
                        messages=[{"role": "user", "content": "Weather?"}],
                        **fields,
                    ),
                )
                self.assertEqual(request._response_parser_grammar_start, grammar_start)

    def _chat_outcome(self, adapted, request, output):
        """(calls, reasoning, content, finish reasons) as a client reconstructs them."""
        if not request.stream:
            choice = self.chat._build_chat_response(request, [output], 0).choices[0]
            calls = [
                (call.function.name, call.function.arguments)
                for call in choice.message.tool_calls or []
            ]
            return (
                calls,
                choice.message.reasoning_content or "",
                choice.message.content or "",
                [choice.finish_reason],
            )

        async def generate():
            yield output

        self.tm.generate_request = Mock(return_value=generate())
        choices = [
            choice
            for chunk in self._parse_chunks(self._run_chat_stream(adapted, request))
            for choice in chunk.get("choices", [])
        ]
        deltas = [choice.get("delta") or {} for choice in choices]
        calls: dict[int, list[str]] = {}
        for delta in deltas:
            for call in delta.get("tool_calls") or []:
                name, arguments = calls.setdefault(call["index"], ["", ""])
                calls[call["index"]] = [
                    name + (call["function"].get("name") or ""),
                    arguments + (call["function"].get("arguments") or ""),
                ]
        return (
            [tuple(calls[index]) for index in sorted(calls)],
            "".join(delta.get("reasoning_content") or "" for delta in deltas),
            "".join(delta.get("content") or "" for delta in deltas),
            [
                choice["finish_reason"]
                for choice in choices
                if choice.get("finish_reason")
            ],
        )

    def test_required_tool_json_is_parsed_under_delimited_template(self):
        """Required-tool JSON after the reasoning becomes the call, also when it
        ends on the tool closer that detokenization keeps."""
        payload = '[{"name": "get_weather", "parameters": {"city": "Paris"}}]'
        for reasoning_parser in (None, "response_template"):
            self._use_delimited_response_template(reasoning_parser)
            reasoning = "Check the forecast." if reasoning_parser else ""
            text = (f"<think>{reasoning}</think>" if reasoning else "") + payload
            text += "</call>"
            for stream in (False, True):
                with self.subTest(reasoning_parser=reasoning_parser, stream=stream):
                    adapted, request = self._convert_rendered_as(
                        "<assistant>",
                        ChatCompletionRequest(
                            model="x",
                            messages=[{"role": "user", "content": "Weather?"}],
                            tools=[_WEATHER_TOOL],
                            tool_choice="required",
                            stream=stream,
                        ),
                    )
                    calls, got_reasoning, content, finish_reasons = self._chat_outcome(
                        adapted, request, self._stopped_on_token(text, 7)
                    )

                    ((name, arguments),) = calls
                    self.assertEqual(name, "get_weather")
                    self.assertEqual(json.loads(arguments), {"city": "Paris"})
                    self.assertEqual(got_reasoning, reasoning)
                    self.assertEqual(content, "")
                    self.assertEqual(finish_reasons, ["tool_calls"])

    def test_json_response_format_with_auto_tools_stays_content(self):
        """A JSON response format with automatic tools must return the JSON
        intact; template tool syntax inside its strings is never a call."""
        payload = '{"example": "<call:get_weather>{}</call>"}'
        for reasoning_parser in (None, "response_template"):
            self._use_delimited_response_template(reasoning_parser)
            reasoning = "Plan" if reasoning_parser else ""
            text = (f"<think>{reasoning}</think>" if reasoning else "") + payload
            for stream in (False, True):
                with self.subTest(reasoning_parser=reasoning_parser, stream=stream):
                    adapted, request = self._convert_rendered_as(
                        "<assistant>",
                        ChatCompletionRequest(
                            model="x",
                            messages=[{"role": "user", "content": "Weather?"}],
                            tools=[_WEATHER_TOOL],
                            tool_choice="auto",
                            response_format={"type": "json_object"},
                            stream=stream,
                        ),
                    )
                    outcome = self._chat_outcome(
                        adapted, request, self._stopped_on_token(text, 2)
                    )

                    self.assertEqual(outcome, ([], reasoning, payload, ["stop"]))

    def test_native_output_under_other_constraints_is_parsed_by_the_template(self):
        """A structural tag, regex or EBNF can spell the template's own framing,
        so its content and a call ending on the kept tool closer still parse."""
        native = '<answer>Sunny</answer><call:get_weather>{"city": "Paris"}</call>'
        structural_tag = {
            "type": "structural_tag",
            "structures": [
                {
                    "begin": "<call:get_weather>",
                    "schema": _WEATHER_TOOL["function"]["parameters"],
                    "end": "</call>",
                }
            ],
            "triggers": ["<call:"],
        }
        regex = "<answer>[^<]+</answer><call:get_weather>[^<]+</call>"
        ebnf = 'root ::= "<answer>" [^<]+ "</answer><call:get_weather>" [^<]+ "</call>"'
        constraints = {
            "structural_tag": {"response_format": structural_tag},
            "regex": {"regex": regex},
            "ebnf": {"ebnf": ebnf},
        }
        self._use_delimited_response_template("response_template")
        for name, fields in constraints.items():
            with self.subTest(name):
                _, request = self._convert_rendered_as(
                    "<assistant>",
                    ChatCompletionRequest(
                        model="x",
                        messages=[{"role": "user", "content": "Weather?"}],
                        tools=[_WEATHER_TOOL],
                        **fields,
                    ),
                )
                output = self._stopped_on_token("<think>Plan</think>" + native, 7)
                choice = self.chat._build_chat_response(request, [output], 0).choices[0]
                (call,) = choice.message.tool_calls
                self.assertEqual(
                    (choice.message.reasoning_content, choice.message.content),
                    ("Plan", "Sunny"),
                )
                self.assertEqual(
                    (call.function.name, json.loads(call.function.arguments)),
                    ("get_weather", {"city": "Paris"}),
                )
                self.assertEqual(choice.finish_reason, "tool_calls")

    def test_text_only_prompt_reuses_rendered_token_ids(self):
        self.tm.model_config.is_multimodal = True
        self.chat.chat_encoding_spec = None
        self.template_manager.chat_template_name = None
        self.chat._tokenizer_auto_adds_specials = False
        self.chat._prompt_text_round_trip_is_lossy = False
        processed = MessageProcessingResult(
            prompt="rendered prompt",
            prompt_ids=[11, 22, 33],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

        with patch.object(self.chat, "_process_messages", return_value=processed):
            adapted, _ = self.chat._convert_to_internal_request(self.basic_req)

        self.assertEqual(adapted.input_ids, [11, 22, 33])
        self.assertIsNone(adapted.text)

    def test_prompt_reuse_keeps_media_path(self):
        self.tm.model_config.is_multimodal = True
        self.chat.chat_encoding_spec = None
        self.template_manager.chat_template_name = None
        self.chat._tokenizer_auto_adds_specials = False
        self.chat._prompt_text_round_trip_is_lossy = False
        processed = MessageProcessingResult(
            prompt="rendered prompt",
            prompt_ids=[11, 22, 33],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )
        processed.image_data = ["image"]

        with patch.object(self.chat, "_process_messages", return_value=processed):
            adapted, _ = self.chat._convert_to_internal_request(self.basic_req)

        self.assertEqual(adapted.text, "rendered prompt")
        self.assertIsNone(adapted.input_ids)

    def test_prompt_reuse_keeps_always_on_processor_path(self):
        self.tm.model_config.is_multimodal = True
        self.tm.model_config.hf_config.architectures = [
            "MossVLForConditionalGeneration"
        ]
        self.chat.chat_encoding_spec = None
        self.template_manager.chat_template_name = None
        self.chat._tokenizer_auto_adds_specials = False
        self.chat._prompt_text_round_trip_is_lossy = False
        processed = MessageProcessingResult(
            prompt="rendered prompt",
            prompt_ids=[11, 22, 33],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

        with patch.object(self.chat, "_process_messages", return_value=processed):
            adapted, _ = self.chat._convert_to_internal_request(self.basic_req)

        self.assertEqual(adapted.text, "rendered prompt")
        self.assertIsNone(adapted.input_ids)

    def test_chat_applies_pd_header_overrides(self):
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            rid="body-rid",
            routed_dp_rank=3,
            disagg_prefill_dp_rank=4,
            priority=5,
        )
        self.fastapi_request.headers = {
            "x-override-rid": "header-rid",
            "x-override-bootstrap-host": "header-host",
            "x-override-bootstrap-port": "8998",
            "x-override-bootstrap-room": "456",
            "x-override-conversation-id": "conversation-1",
            "x-override-routed-dp-rank": "6",
            "x-override-disagg-prefill-dp-rank": "7",
            "x-override-priority": "8",
        }
        body = request.model_dump()

        processed_messages = MessageProcessingResult(
            "Test prompt", [1, 2, 3], None, None, [], [], None
        )
        with (
            envs.SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES.override(True),
            patch.object(
                self.chat, "_process_messages", return_value=processed_messages
            ),
        ):
            response = get_or_create_event_loop().run_until_complete(
                self.chat.handle_request(request, self.fastapi_request)
            )

        self.assertEqual(response.choices[0].message.content, "Test response")
        adapted_request = self.tm.generate_request.call_args.args[0]
        self.assertEqual(adapted_request.bootstrap_room, 456)
        self.assertEqual(adapted_request.bootstrap_host, "header-host")
        self.assertEqual(adapted_request.bootstrap_port, 8998)
        self.assertEqual(adapted_request.rid, "header-rid")
        self.assertEqual(adapted_request.conversation_id, "conversation-1")
        self.assertEqual(adapted_request.routed_dp_rank, 6)
        self.assertEqual(adapted_request.disagg_prefill_dp_rank, 7)
        self.assertEqual(adapted_request.priority, 8)
        self.assertEqual(request.model_dump(), body)
        self.assertFalse(hasattr(request, "conversation_id"))

    def test_convert_to_internal_request_rejects_stream_token_ids(self):
        for field in ("return_prompt_token_ids", "return_token_ids"):
            req = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Hi?"}],
                stream=True,
                **{field: True},
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                self.chat._convert_to_internal_request(req, self.fastapi_request)

    def test_validate_request_rejects_sampling_mask_without_meta_info(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            return_sampling_mask=True,
        )

        self.assertEqual(
            self.chat._validate_request(req),
            "return_sampling_mask requires return_meta_info=true.",
        )

    def test_validate_request_rejects_flat_output_top_logprobs_without_meta_info(
        self,
    ):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            logprobs=True,
            top_logprobs=2,
            return_flat_raw_output_top_logprobs=True,
        )

        self.assertEqual(
            self.chat._validate_request(req),
            "return_flat_raw_output_top_logprobs requires return_meta_info=true.",
        )

    def test_convert_to_internal_request_flat_output_top_logprobs_default_to_b64(self):
        self.tm.tokenizer = None
        # (request flags, expected flat output, expected b64)
        cases = (
            ({"return_flat_raw_output_top_logprobs": True}, True, True),
            (
                {
                    "return_flat_raw_output_top_logprobs": True,
                    "return_flat_raw_top_logprobs_b64": False,
                },
                True,
                False,
            ),
            ({}, False, False),
        )
        for flags, flat, b64 in cases:
            req = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Hi?"}],
                input_ids=[101, 102, 103],
                logprobs=True,
                top_logprobs=2,
                return_meta_info=True,
                **flags,
            )
            with self.subTest(flags=flags):
                adapted, _ = self.chat._convert_to_internal_request(
                    req, self.fastapi_request
                )
                # The b64 default must not trip GenerateReqInput validation.
                adapted.normalize_batch_and_arguments()

                self.assertEqual(adapted.top_logprobs_num, 2)
                self.assertEqual(adapted.return_flat_raw_output_top_logprobs, flat)
                self.assertEqual(adapted.return_flat_raw_top_logprobs_b64, b64)

    def test_convert_to_internal_request_rejects_stream_return_meta_info(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            return_meta_info=True,
        )

        with self.assertRaisesRegex(
            ValueError, "return_meta_info is not supported with streaming"
        ):
            self.chat._convert_to_internal_request(req, self.fastapi_request)

    def test_convert_to_internal_request_input_ids_bypasses_template(self):
        self.tm.tokenizer = None
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            input_ids=[101, 102, 103],
            stop=["STOP"],
            return_prompt_token_ids=True,
            cache_salt="tenant-a",
            extra_key="classification",
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            adapted, processed = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

        self.assertEqual(processed, req)
        self.assertEqual(adapted.input_ids, [101, 102, 103])
        self.assertTrue(adapted.return_prompt_token_ids)
        self.assertEqual(adapted.sampling_params["stop"], ["STOP"])
        self.assertEqual(adapted.cache_salt, "tenant-a")
        self.assertEqual(adapted.extra_key, "classification")
        conv_mock.assert_not_called()

    def test_kimi_k3_usage_excludes_assistant_generation_stub(self):
        self.chat.chat_encoding_spec = "kimi_k3"
        ret = [
            {
                "text": "Answer",
                "meta_info": {
                    "id": "chatcmpl-kimi-k3-usage",
                    "prompt_tokens": 2075,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "image_tokens": 2035,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.chat._build_chat_response(self.basic_req, ret, created=123)

        self.assertEqual(response.usage.prompt_tokens, 2072)
        self.assertEqual(response.usage.total_tokens, 2073)
        self.assertEqual(response.usage.prompt_tokens_details.image_tokens, 2035)

    def test_kimi_tool_call_keeps_default_reasoning(self):
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                        },
                    },
                }
            ],
            tool_choice="required",
        )

        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = MessageProcessingResult(
                "",
                [1, 2, 3],
                None,
                None,
                [],
                [],
                None,
                require_reasoning=True,
            )

            adapted, _ = self.chat._convert_to_internal_request(req)

        self.assertTrue(adapted.require_reasoning)

    def test_process_messages_records_template_reasoning_state(self):
        self.chat.default_chat_template_kwargs = {"thinking": True}
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=False
        )
        self.chat.reasoning_parser = "deepseek-v3"
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )
        rendered = MessageProcessingResult(
            prompt="prompt",
            prompt_ids=[1, 2, 3],
            image_data=None,
            audio_data=None,
            video_data=None,
            modalities=[],
            stop=[],
        )

        with patch.object(
            self.chat, "_apply_conversation_template", return_value=rendered
        ):
            processed = self.chat._process_messages(request, is_multimodal=False)

        self.assertTrue(processed.require_reasoning)

    def test_kimi_tool_call_respects_explicit_reasoning_disable(self):
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                        },
                    },
                }
            ],
            tool_choice="required",
            chat_template_kwargs={"thinking": False},
        )

        with patch.object(self.chat, "_process_messages") as proc_mock:
            proc_mock.return_value = MessageProcessingResult(
                "",
                [1, 2, 3],
                None,
                None,
                [],
                [],
                None,
            )

            adapted, _ = self.chat._convert_to_internal_request(req)

        self.assertFalse(adapted.require_reasoning)

    def test_default_chat_template_kwargs_applied_when_request_unset(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.chat.default_chat_template_kwargs = {"enable_thinking": False}

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertIs(kwargs["enable_thinking"], False)

    def test_default_chat_template_kwargs_overridden_per_request(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.chat.default_chat_template_kwargs = {"enable_thinking": False}

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            chat_template_kwargs={"enable_thinking": True},
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertIs(kwargs["enable_thinking"], True)

    def test_default_chat_template_kwargs_mirrors_reasoning_effort(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.chat.default_chat_template_kwargs = {"reasoning_effort": "high"}

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        self.chat._process_messages(req, is_multimodal=False)

        self.assertEqual(req.reasoning_effort, "high")

    def test_hunyuan_default_reasoning_effort_is_normalized_for_template(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="hunyuan_effort"
        )
        self.chat.reasoning_parser = "hunyuan"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        cases = [
            ("none", None, "no_think"),
            ("none", "xhigh", "high"),
        ]
        for default_effort, request_effort, normalized_effort in cases:
            with self.subTest(
                default_effort=default_effort, request_effort=request_effort
            ):
                self.chat.default_chat_template_kwargs = {
                    "reasoning_effort": default_effort
                }
                req = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    reasoning_effort=request_effort,
                )

                self.chat._process_messages(req, is_multimodal=False)

                kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
                self.assertEqual(req.reasoning_effort, normalized_effort)
                self.assertEqual(kwargs["reasoning_effort"], normalized_effort)
                self.assertNotIn("reasoning_effort", req.chat_template_kwargs)

    def test_hunyuan_reasoning_effort_precedence_survives_conversion(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="hunyuan_effort"
        )
        self.chat.reasoning_parser = "hunyuan"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        cases = [
            ("xhigh", "none", "high"),
            (None, "no_think", "no_think"),
        ]
        for request_effort, template_effort, normalized_effort in cases:
            with self.subTest(
                request_effort=request_effort, template_effort=template_effort
            ):
                req = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "What is 2+2?"}],
                    reasoning_effort=request_effort,
                    chat_template_kwargs={"reasoning_effort": template_effort},
                )

                self.chat._convert_to_internal_request(req)

                kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
                self.assertEqual(req.reasoning_effort, normalized_effort)
                self.assertEqual(kwargs["reasoning_effort"], normalized_effort)
                self.assertNotIn("reasoning_effort", req.chat_template_kwargs)

    def test_non_hunyuan_default_reasoning_effort_is_unchanged(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.chat.default_chat_template_kwargs = {"reasoning_effort": "medium"}
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(req.reasoning_effort, "medium")
        self.assertEqual(kwargs["reasoning_effort"], "medium")
        self.assertEqual(req.chat_template_kwargs["reasoning_effort"], "medium")

    def test_k2_selected_terminator_reaches_sampling_params(self):
        self.tm._config_overrides["reasoning_parser"] = "k2_horizon"
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.tm.tokenizer.encode.side_effect = lambda text, **_: (
            [8, 9] if text == "</ifm|think_fast>" else [1, 2, 3]
        )
        req = ChatCompletionRequest(
            model="IFM/K2-Horizon-7B",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"reasoning_effort": "medium"},
        )

        processed = self.chat._process_messages(req, is_multimodal=False)
        self.assertEqual(processed.reasoning_end_token_ids, [8, 9])

        with patch.object(self.chat, "_process_messages", return_value=processed):
            adapted, _ = self.chat._convert_to_internal_request(req)
        self.assertEqual(
            adapted.sampling_params["custom_params"][
                REQUEST_REASONING_END_TOKEN_IDS_KEY
            ],
            [8, 9],
        )

    def test_kimi_tool_call_keeps_template_default_thinking(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                        },
                    },
                }
            ],
            tool_choice="required",
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertNotIn("thinking", kwargs)

    def test_kimi_tool_call_keeps_explicit_template_thinking(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                        },
                    },
                }
            ],
            tool_choice="required",
            chat_template_kwargs={"thinking": True},
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertTrue(kwargs["thinking"])

    def test_kimi_tool_call_keeps_explicit_template_thinking_false(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"type": "integer"}},
                        },
                    },
                }
            ],
            tool_choice="required",
            chat_template_kwargs={"thinking": False},
        )

        self.chat._process_messages(req, is_multimodal=False)

        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertFalse(kwargs["thinking"])

    def test_jinja_uses_openai_tool_schema_first(self):
        """Ensure Jinja chat templates receive OpenAI-shaped tools by default."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        expected_tools = [tool.model_dump() for tool in req.tools]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], expected_tools)

    def test_glm47_without_tools_has_no_tool_call_constraint(self):
        """A plain GLM47 chat must not get the full-assistant EBNF: its
        terminal state finishes the request even under ignore_eos."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        self.chat.tool_call_parser = "glm47"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
        )

        processed = self.chat._process_messages(req, is_multimodal=False)

        self.assertIsNone(processed.tool_call_constraint)

    def test_jinja_tool_schema_fallback_to_flat_function(self):
        """Fallback to function-only schema when template rejects OpenAI wrapper."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.tm.tokenizer.apply_chat_template.side_effect = [
            RuntimeError("template expects flat tools format"),
            [1, 2, 3],
        ]

        self.chat._process_messages(req, is_multimodal=False)

        first_tools = self.tm.tokenizer.apply_chat_template.call_args_list[0].kwargs[
            "tools"
        ]
        second_tools = self.tm.tokenizer.apply_chat_template.call_args_list[1].kwargs[
            "tools"
        ]
        self.assertEqual(first_tools, [tool.model_dump() for tool in req.tools])
        self.assertEqual(
            second_tools, [tool.function.model_dump() for tool in req.tools]
        )

    def test_glm47_constraint_includes_message_tools(self):
        """Constrain every active message tool, but none when tool use is disabled.

        A terminal GLM47 grammar on ``tool_choice="none"`` ends generation early,
        while omitting message-scoped tools rejects otherwise valid tool calls.
        """
        import xgrammar as xgr

        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.chat.tool_call_parser = "glm47"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        compiler = xgr.GrammarCompiler(
            xgr.TokenizerInfo(
                [bytes([i]) for i in range(256)], vocab_type=xgr.VocabType.RAW
            ),
            max_threads=1,
        )
        message_tool = {
            "type": "function",
            "function": {"name": "message_tool", "parameters": {"type": "object"}},
        }
        top_level_tool = {
            "type": "function",
            "function": {"name": "top_level_tool", "parameters": {"type": "object"}},
        }
        named = ToolChoice(function=ToolChoiceFuncName(name="message_tool"))
        for role in ("system", "developer"):
            for tools in (None, [], [top_level_tool]):
                for choice in ("auto", "required", named, "none"):
                    with self.subTest(role=role, tools=tools, choice=choice):
                        request = ChatCompletionRequest(
                            model="x",
                            messages=[
                                {"role": role, "content": "", "tools": [message_tool]},
                                {"role": "user", "content": "Use message_tool."},
                            ],
                            tools=tools,
                            tool_choice=choice,
                            chat_template_kwargs={"enable_thinking": False},
                        )
                        result = self.chat._process_messages(
                            request, is_multimodal=False
                        )
                        if choice == "none":
                            self.assertIsNone(result.tool_call_constraint)
                            continue
                        self.assertIsNotNone(result.tool_call_constraint)
                        kind, ebnf = result.tool_call_constraint
                        self.assertEqual(kind, "full_assistant_ebnf")
                        grammar = compiler.compile_grammar(xgr.Grammar.from_ebnf(ebnf))
                        for name, allowed in (
                            ("message_tool", True),
                            ("top_level_tool", bool(tools) and choice != named),
                            ("unknown_tool", False),
                        ):
                            matcher = xgr.GrammarMatcher(grammar)
                            accepted = (
                                matcher.accept_string(f"<tool_call>{name}</tool_call>")
                                and matcher.is_completed()
                            )
                            self.assertEqual(accepted, allowed, name)

    def test_xgrammar_tag_omits_reasoning_when_parser_owns_it(self):
        """ReasonerGrammarBackend owns the thinking prefix when a parser is set."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=True
        )
        self.tm.server_args.reasoning_parser = "kimi_k2"
        self.tm.server_args.tool_call_parser = "kimi_k2"
        self.chat.reasoning_parser = "kimi_k2"
        self.chat.tool_call_parser = "kimi_k2"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                        "strict": True,
                    },
                }
            ],
            tool_choice="required",
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as parser_cls:
            parser_cls.ToolCallParserEnum = FunctionCallParser.ToolCallParserEnum
            parser = parser_cls.return_value
            parser.get_structure_constraint.return_value = ("structural_tag", "tag")

            self.chat._process_messages(req, is_multimodal=False)

            parser.get_structure_constraint.assert_called_once()
            self.assertFalse(
                parser.get_structure_constraint.call_args.kwargs["thinking_mode"]
            )

    def test_kimi_k3_constraint_failure_keeps_native_stop_format(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.chat.chat_encoding_spec = "kimi_k3"
        self.chat.tool_call_parser = "kimi_k3"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "strict": True,
            },
        }

        cases = (
            (None, [TOOLS_CLOSE]),
            ("USER_STOP", ["USER_STOP", TOOLS_CLOSE]),
            (["USER_STOP"], ["USER_STOP", TOOLS_CLOSE]),
            ([TOOLS_CLOSE], [TOOLS_CLOSE]),
        )
        for request_stop, expected in cases:
            with (
                self.subTest(request_stop=request_stop),
                patch(
                    "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
                ) as parser_cls,
            ):
                parser_cls.ToolCallParserEnum = FunctionCallParser.ToolCallParserEnum
                parser = parser_cls.return_value
                parser.detector.eot_token = TOOLS_CLOSE
                parser.detector.parses_required_natively.return_value = False
                parser.get_structure_constraint.return_value = None
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Weather in Paris?"}],
                    tools=[tool],
                    tool_choice="required",
                    stop=request_stop,
                )
                original_stop = (
                    list(request.stop)
                    if isinstance(request.stop, list)
                    else request.stop
                )

                result = self.chat._process_messages(request, is_multimodal=False)

                self.assertEqual(result.stop, expected)
                self.assertEqual(request.stop, original_stop)
                self.assertIsNone(result.tool_call_constraint)

    def test_kimi_k3_tool_call_stop_is_scoped_to_active_tools(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        self.chat.chat_encoding_spec = "kimi_k3"
        self.chat.tool_call_parser = "kimi_k3"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="none",
        )

        result = self.chat._process_messages(request, is_multimodal=False)

        self.assertIsNone(result.stop)

    def test_dsv_tool_choice_none_omits_tools_from_prompt(self):
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"
        tool = {
            "type": "function",
            "function": {
                "name": "calculator",
                "parameters": {"type": "object"},
            },
        }

        self.chat.chat_encoding_spec = "dsv32"
        for tool_choice, expects_tools in (("none", False), ("auto", True)):
            with self.subTest(tool_choice=tool_choice):
                self.tm.tokenizer.encode.reset_mock()
                request = ChatCompletionRequest(
                    model="x",
                    messages=[{"role": "user", "content": "Solve 100 / 7."}],
                    tools=[tool],
                    tool_choice=tool_choice,
                )

                self.chat._process_messages(request, is_multimodal=False)

                prompt = self.tm.tokenizer.encode.call_args.args[0]
                self.assertEqual("calculator" in prompt, expects_tools)

    def test_kimi_k3_encoder_receives_wire_request_fields(self):
        self.template_manager.chat_template_name = None
        self.chat.chat_encoding_spec = "kimi_k3"
        self.tm.model_config.is_multimodal = True
        self.tm.tokenizer.apply_chat_template.return_value = [7, 8, 9]
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
            },
        }
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "developer",
                    "content": "<|kimi_image_placeholder|>",
                    "tools": [tool],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Explain <|kimi_image_placeholder|>",
                        },
                        {"type": "image_url", "image_url": {"url": "image-1"}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Inspect <|kimi_image_placeholder|>",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "inspect",
                                "arguments": {
                                    "source": "<|kimi_image_placeholder|>",
                                    "nested": ["<|kimi_image_placeholder|>"],
                                },
                            },
                        }
                    ],
                },
            ],
            tools=[tool],
            tool_choice="required",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": False,
                },
            },
        )

        result = self.chat._process_messages(request, is_multimodal=True)

        call = self.tm.tokenizer.apply_chat_template.call_args
        rendered_messages = call.args[0]
        self.assertEqual(rendered_messages[0]["role"], "system")
        self.assertEqual(
            rendered_messages[0]["content"], "<| kimi_image_placeholder |>"
        )
        self.assertNotIn("strict", rendered_messages[0]["tools"][0]["function"])
        self.assertEqual(
            rendered_messages[1]["content"][0]["text"],
            "Explain <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            rendered_messages[2]["reasoning_content"],
            "Inspect <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            rendered_messages[2]["tool_calls"][0]["function"]["arguments"],
            {
                "source": "<| kimi_image_placeholder |>",
                "nested": ["<| kimi_image_placeholder |>"],
            },
        )
        self.assertEqual(call.kwargs["image_prompts"], ["<|media_pad|>"])
        self.assertEqual(call.kwargs["tool_choice"], "required")
        self.assertNotIn("strict", call.kwargs["tools"][0]["function"])
        self.assertEqual(
            call.kwargs["response_format"]["json_schema"]["schema"],
            {"type": "object"},
        )
        self.assertNotIn("schema_", call.kwargs["response_format"]["json_schema"])
        self.assertEqual(result.prompt_ids, [7, 8, 9])
        self.assertEqual(result.image_data[0].url, "image-1")

    def test_kimi_k3_neutralizes_text_only_assistant_history(self):
        self.template_manager.chat_template_name = None
        self.chat.chat_encoding_spec = "kimi_k3"
        self.tm.tokenizer.apply_chat_template.return_value = [1, 2, 3]
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Run it"},
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "Read <|kimi_image_placeholder|>",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": "not-json <|kimi_image_placeholder|>",
                            },
                        }
                    ],
                },
            ],
        )

        self.chat._process_messages(request, is_multimodal=False)

        messages = self.tm.tokenizer.apply_chat_template.call_args.args[0]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(messages[-1]["role"], "assistant")
        self.assertEqual(
            messages[-1]["reasoning_content"],
            "Read <| kimi_image_placeholder |>",
        )
        self.assertEqual(
            messages[-1]["tool_calls"][0]["function"]["arguments"],
            "not-json <| kimi_image_placeholder |>",
        )
        self.assertNotIn("image_prompts", kwargs)

    def test_message_tools_participate_in_validation_across_encodings(self):
        tool = {
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {"type": "object"},
            },
        }
        messages = [
            {"role": "system", "content": "", "tools": [tool]},
            {"role": "user", "content": "Weather?"},
        ]
        for chat_encoding_spec in (None, "dsv4", "dsv32", "kimi_k3"):
            with self.subTest(chat_encoding_spec=chat_encoding_spec):
                self.chat.chat_encoding_spec = chat_encoding_spec
                automatic = ChatCompletionRequest(
                    model="x", messages=messages, tool_choice=None
                )
                self.assertEqual(automatic.tool_choice, "auto")
                self.assertIsNone(self.chat._validate_request(automatic))

                required = ChatCompletionRequest(
                    model="x", messages=messages, tool_choice="required"
                )
                self.assertIsNone(self.chat._validate_request(required))

                duplicate = ChatCompletionRequest(
                    model="x",
                    messages=messages,
                    tools=[tool],
                    tool_choice="required",
                )
                self.assertEqual(
                    self.chat._validate_request(duplicate),
                    "Tool names must be unique across request and message tools.",
                )

    def test_jinja_rejects_non_object_tool_call_arguments(self):
        """History tool call arguments must parse to a JSON object."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        for arguments in ['"Beijing"', '["Beijing"]']:
            with self.subTest(arguments=arguments):
                self.tm.tokenizer.apply_chat_template.reset_mock()
                req = ChatCompletionRequest(
                    model="x",
                    messages=[
                        {"role": "user", "content": "Where is it raining?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": arguments,
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": "call_1",
                            "content": "sunny",
                        },
                    ],
                )

                with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                    self.chat._process_messages(req, is_multimodal=False)

                self.tm.tokenizer.apply_chat_template.assert_not_called()

    def test_jinja_accepts_object_tool_call_arguments_string(self):
        """OpenAI JSON string arguments are converted to dicts for templates."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Where is it raining?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                },
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        messages = self.tm.tokenizer.apply_chat_template.call_args.args[0]
        self.assertEqual(
            messages[1]["tool_calls"][0]["function"]["arguments"],
            {"city": "Beijing"},
        )

    def test_dsv_encoders_reject_non_object_tool_call_arguments(self):
        """DeepSeek encoders should reject history tool call scalars as BadRequest."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        self.chat.chat_encoding_spec = "dsv32"
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Where is it raining?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '"Beijing"',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "must be a JSON object"):
            self.chat._process_messages(req, is_multimodal=False)

    def test_dsv_encoders_accept_object_tool_call_arguments_string(self):
        """DeepSeek encoders accept object-shaped OpenAI JSON string arguments."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        self.chat.chat_encoding_spec = "dsv32"
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Where is it raining?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                },
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
            created=123,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")

        # Parse the result to verify content
        self.assertTrue(result.startswith("data: "))
        chunk = json.loads(result[6:])
        self.assertEqual(chunk["created"], 123)
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        arguments = tool_calls[0]["function"]["arguments"]
        self.assertIn(', "unit": "celsius"}', arguments)

        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
            created=123,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    def test_unstreamed_tool_args_raw_string_no_completion_needed(self):
        """Test raw JSON string arguments are not JSON-encoded again at finish."""

        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = [
            {"name": "report_template_recognition_commit", "arguments": "{}"}
        ]
        mock_detector.streamed_args_for_tool = ["{}"]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "commit"}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "report_template_recognition_commit"},
                }
            ],
        )

        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
            created=123,
        )

        self.assertIsNone(result, "Should not append encoded quotes")

    def test_unstreamed_tool_args_raw_string_completion(self):
        """Test remaining raw JSON string arguments are sent at finish."""

        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
            }
        ]
        mock_detector.streamed_args_for_tool = ['{"location": "San Francisco"']
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
            created=123,
        )

        self.assertIsNotNone(result, "Should return chunk with remaining arguments")
        chunk = json.loads(result[6:])
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(tool_calls[0]["function"]["arguments"], ', "unit": "celsius"}')

    def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
            created=123,
        )

        # Should return None since there's no parser data
        self.assertIsNone(
            result, "Should return None when parser has no tool call data"
        )

    # ------------- kimi_k2 tool_call_id formatting -------------
    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA and Tokyo?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=False,
        )

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Loa Angeles"}'
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            call_info.tool_index = 0

            call_info2 = Mock()
            call_info2.name = "get_weather"
            call_info2.parameters = '{"city":"Tokyo"}'
            call_info2.tool_index = 1

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = (
                "",
                [call_info, call_info2],
            )

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            history_tool_calls_cnt = self.chat._get_history_tool_calls_cnt(req)
            tool_calls, remaining_text, _ = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
                history_tool_calls_cnt=history_tool_calls_cnt,
            )

            self.assertEqual(history_tool_calls_cnt, 1)
            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:1")
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(tool_calls[1].id, "functions.get_weather:2")
            self.assertEqual(tool_calls[1].function.name, "get_weather")

    def test_non_streaming_tool_call_index_is_the_call_ordinal(self):
        """Two calls to one tool are numbered 0 and 1, as in the streaming deltas,
        not by the detector's tool_index (0 for both)."""
        self.chat.tool_call_parser = "deepseekv4"
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value
            calls = []
            for city in ("San Francisco", "London"):
                call_info = Mock()
                call_info.name = "get_weather"
                call_info.parameters = json.dumps({"location": city})
                call_info.tool_index = 0
                calls.append(call_info)
            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", calls)

            tool_calls, _, finish_reason = self.chat._process_tool_calls(
                text="<｜DSML｜tool_calls>...",
                tools=tools,
                finish_reason={"type": "stop", "matched": None},
                history_tool_calls_cnt=0,
            )

        self.assertEqual([tc.index for tc in tool_calls], [0, 1])
        self.assertEqual(
            [tc.function.arguments for tc in tool_calls],
            [
                json.dumps({"location": "San Francisco"}),
                json.dumps({"location": "London"}),
            ],
        )
        self.assertEqual(finish_reason["type"], "tool_calls")

    def test_iquest_release_tool_modes_through_streaming_serving(self):
        self.chat.tool_call_parser = "iquest_q1"
        self.chat.reasoning_parser = "iquest_q1"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"],
                    },
                },
            }
        ]
        named = ToolChoice(function=ToolChoiceFuncName(name="run"))
        payloads = (
            (
                "auto",
                "before<iquest_tool_call>run<arg_key>code</arg_key>"
                "<arg_value>你好</arg_value></iquest_tool_call>after",
                "beforeafter",
            ),
            ("required", '[{"name":"run","parameters":{"code":"你好"}}]', ""),
            (named, '{ "code": "你好" }', ""),
        )
        for choice, payload, expected_content in payloads:
            for thinking in (True, False):
                wire = ("work</think>" if thinking else "") + payload
                for finish in ("stop", "length"):
                    for size in (1,):
                        with self.subTest(
                            choice=choice, thinking=thinking, finish=finish, size=size
                        ):
                            req = ChatCompletionRequest(
                                model="x",
                                messages=[{"role": "user", "content": "run"}],
                                input_ids=[1],
                                tools=tools,
                                tool_choice=choice,
                                chat_template_kwargs={"thinking": thinking},
                                stream=True,
                            )
                            processed = self.chat._process_messages(req, False)
                            self.assertFalse(req.skip_special_tokens)
                            constraint = processed.tool_call_constraint
                            if choice == "auto":
                                self.assertIsNone(constraint)
                            else:
                                self.assertEqual(constraint[0], "json_schema")
                                self.assertEqual(
                                    constraint[1]["type"],
                                    "object" if choice == named else "array",
                                )

                            async def generate():
                                for end in range(size, len(wire) + size, size):
                                    yield {
                                        "text": wire[:end],
                                        "meta_info": {
                                            "id": "chatcmpl-iquest-release",
                                            "prompt_tokens": 1,
                                            "completion_tokens": min(end, len(wire)),
                                            "finish_reason": (
                                                {"type": finish, "matched": None}
                                                if end >= len(wire)
                                                else None
                                            ),
                                        },
                                        "index": 0,
                                    }

                            self.tm.generate_request.return_value = generate()
                            chunks = self._parse_chunks(
                                self._run_chat_stream(None, req)
                            )
                            self.assertFalse(any("error" in chunk for chunk in chunks))
                            choices = [
                                c for chunk in chunks for c in chunk.get("choices", [])
                            ]
                            deltas = [c.get("delta", {}) for c in choices]
                            self.assertEqual(
                                "".join(d.get("content") or "" for d in deltas),
                                expected_content,
                            )
                            self.assertEqual(
                                "".join(
                                    d.get("reasoning_content") or "" for d in deltas
                                ),
                                "work" if thinking else "",
                            )
                            calls = [
                                c for d in deltas for c in (d.get("tool_calls") or [])
                            ]
                            self.assertEqual(
                                [
                                    c["function"]["name"]
                                    for c in calls
                                    if c["function"].get("name")
                                ],
                                ["run"],
                            )
                            self.assertTrue(all(c["index"] == 0 for c in calls))
                            self.assertEqual(
                                json.loads(
                                    "".join(
                                        c["function"].get("arguments") or ""
                                        for c in calls
                                    )
                                ),
                                {"code": "你好"},
                            )
                            self.assertEqual(
                                [
                                    c["finish_reason"]
                                    for c in choices
                                    if c.get("finish_reason")
                                ],
                                ["tool_calls" if finish == "stop" else finish],
                            )

    def test_iquest_named_tool_rejects_invalid_arguments(self):
        self.chat.tool_call_parser = "iquest_q1"
        choice = ToolChoice(function=ToolChoiceFuncName(name="run"))
        for payload, finish in (
            ('{"code":', {"type": "length", "length": 16}),
            ("42", {"type": "stop", "matched": "stop-marker"}),
        ):
            with self.subTest(payload=payload):
                calls, text, finish_reason = self.chat._process_tool_calls(
                    text=payload,
                    tools=ChatCompletionRequest(
                        model="test",
                        messages=[],
                        tools=[{"type": "function", "function": {"name": "run"}}],
                    ).tools,
                    finish_reason=dict(finish),
                    tool_choice=choice,
                )
                self.assertIsNone(calls)
                self.assertEqual(text, payload)
                self.assertEqual(finish_reason, finish)

    def test_required_tool_choice_skips_json_fallback_for_native_parser(self):
        """A structural-tag parser owns the output format, so a missing tool
        call must not be pushed through the json_schema array fallback."""
        self.chat.tool_call_parser = "kimi_k3"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        texts = {
            "prose": "<|open|>response<|sep|>I'll check that.<|close|>response<|sep|>",
            "empty": "",
            "json_object": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
        }
        for choice in (
            "required",
            ToolChoice(function=ToolChoiceFuncName(name="get_weather")),
        ):
            for label, text in texts.items():
                with self.subTest(tool_choice=choice, payload=label):
                    finish_reason = {"type": "stop", "matched": None}
                    with self.assertLogs(
                        "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
                    ) as logs:
                        tool_calls, remaining, finish_reason = (
                            self.chat._process_tool_calls(
                                text=text,
                                tools=tools,
                                finish_reason=finish_reason,
                                tool_choice=choice,
                            )
                        )
                    self.assertIsNone(tool_calls)
                    self.assertEqual(remaining, text)
                    self.assertEqual(finish_reason["type"], "stop")
                    self.assertNotIn("Tool call parsing error", "\n".join(logs.output))

    def test_truncated_native_tool_call_logs_and_drops(self):
        """A tools section cut off before its closing tag parses to zero calls
        without raising; the sync path used to drop it with no log at all."""
        self.chat.tool_call_parser = "kimi_k3"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        truncated = TOOLS_OPEN + '<|open|>call tool="get_weather" index="1"<|sep|>'
        for choice in ("auto", "required"):
            with self.subTest(tool_choice=choice):
                finish_reason = {"type": "stop", "matched": None}
                with self.assertLogs(
                    "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
                ) as logs:
                    tool_calls, remaining, finish_reason = (
                        self.chat._process_tool_calls(
                            text=truncated,
                            tools=tools,
                            finish_reason=finish_reason,
                            tool_choice=choice,
                        )
                    )
                self.assertIsNone(tool_calls)
                self.assertEqual(remaining, "")
                self.assertEqual(finish_reason["type"], "stop")
                self.assertIn("no complete call", "\n".join(logs.output))

    def test_required_tool_choice_json_fallback_tolerates_odd_shapes(self):
        """Parsers without a structural tag keep the JSON array fallback, but a
        non-array payload degrades instead of raising an opaque TypeError."""
        self.chat.tool_call_parser = "glm45"
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]
        cases = [
            (
                '[{"name": "get_weather", "parameters": {"city": "Paris"}}]',
                '{"city": "Paris"}',
            ),
            (
                '{"name": "get_weather", "parameters": {"city": "Paris"}}',
                '{"city": "Paris"}',
            ),
            ('{"name": "get_weather"}', "{}"),
        ]
        for text, expected_args in cases:
            with self.subTest(text=text):
                tool_calls, _, finish_reason = self.chat._process_tool_calls(
                    text=text,
                    tools=tools,
                    finish_reason={"type": "stop", "matched": None},
                    tool_choice="required",
                )
                self.assertEqual(len(tool_calls), 1)
                self.assertEqual(tool_calls[0].function.name, "get_weather")
                self.assertEqual(tool_calls[0].function.arguments, expected_args)
                self.assertEqual(finish_reason["type"], "tool_calls")

        finish_reason = {"type": "stop", "matched": None}
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_chat", level="ERROR"
        ):
            tool_calls, remaining, finish_reason = self.chat._process_tool_calls(
                text='["get_weather"]',
                tools=tools,
                finish_reason=finish_reason,
                tool_choice="required",
            )
        self.assertIsNone(tool_calls)
        self.assertEqual(remaining, '["get_weather"]')
        self.assertEqual(finish_reason["type"], "stop")

    def test_required_tool_choice_rejects_conflicting_output_constraint(self):
        """response_format and a forced tool call cannot both be honored: the
        tool-call constraint was dropped with only a warning, so the model was
        constrained to a shape that can never contain a tool call."""
        tools = [{"type": "function", "function": {"name": "get_weather"}}]
        constraint = ("structural_tag", None)
        conflicting = [
            {"type": "json_object"},
            {
                "type": "json_schema",
                "json_schema": {"name": "a", "schema": {"type": "object"}},
            },
        ]
        for tool_choice in (
            "required",
            ToolChoice(function=ToolChoiceFuncName(name="get_weather")),
        ):
            for response_format in conflicting:
                with self.subTest(tool_choice=tool_choice, rf=response_format["type"]):
                    request = ChatCompletionRequest(
                        model="x",
                        messages=[{"role": "user", "content": "hi"}],
                        tools=tools,
                        tool_choice=tool_choice,
                        response_format=response_format,
                    )
                    with self.assertRaises(ValueError) as ctx:
                        request.to_sampling_params(
                            stop=[],
                            model_generation_config={},
                            tool_call_constraint=constraint,
                        )
                    self.assertIn("cannot be combined", str(ctx.exception))

    def test_auto_tool_choice_keeps_response_format_without_raising(self):
        """ "auto" means the model need not call a tool, so dropping the
        tool-call constraint still leaves a satisfiable request."""
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            tool_choice="auto",
            response_format={"type": "json_object"},
        )
        sampling_params = request.to_sampling_params(
            stop=[],
            model_generation_config={},
            tool_call_constraint=("structural_tag", None),
        )
        self.assertEqual(sampling_params["json_schema"], '{"type": "object"}')

    def test_kimi_k2_streaming_tool_call_id_with_history(self):
        """Ensure streaming first chunk tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                    created=123,
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            self.assertEqual(payload["created"], 123)
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:1")

    def test_dpsk_v32_encoding_path(self):
        """Test DeepSeek V3.2 encoding path detection and application."""
        from sglang.srt.parser.template_manager import TemplateManager

        # Only mock the fields that _use_dpsk_v32_encoding() actually reads:
        # tokenizer.chat_template and hf_config.architectures
        tm = _MockTokenizerManager()

        mock_hf_config = Mock()
        mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]
        mock_hf_config.to_dict.return_value = {}
        tm.model_config.hf_config = mock_hf_config

        # Case 1: No chat template + DeepSeek V3.2 arch -> should use dsv32 encoding
        tm.tokenizer.chat_template = None
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "dsv32")

        # Case 2: Chat template exists -> should NOT use dsv32 encoding
        tm.tokenizer.chat_template = "some template"
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertIsNone(serving_chat.chat_encoding_spec)

        # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dsv32 encoding
        tm.tokenizer.chat_template = None
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertIsNone(serving_chat.chat_encoding_spec)

        # Case 4: DeepseekV4 arch -> always dsv4, even with chat_template
        # (release ships a stale V3 jinja we deliberately override).
        mock_hf_config.architectures = ["DeepseekV4ForCausalLM"]
        for chat_template in ("stale v3 jinja", None):
            tm.tokenizer.chat_template = chat_template
            self.assertEqual(
                resolve_chat_encoding_spec(
                    hf_config=mock_hf_config, tokenizer=tm.tokenizer
                ),
                "dsv4",
            )

    def test_kimi_k3_encoding_detection(self):
        from sglang.srt.parser.template_manager import TemplateManager

        tm = _MockTokenizerManager()
        tm.model_config.hf_config.architectures = ["KimiK3ForConditionalGeneration"]
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "kimi_k3")

        tm.model_config.hf_config.architectures = ["LlamaForCausalLM"]
        tm.server_args.tool_call_parser = "kimi_k3"
        serving_chat = OpenAIServingChat(tm, TemplateManager())
        self.assertEqual(serving_chat.chat_encoding_spec, "kimi_k3")

    def test_custom_encoders_own_reasoning_history(self):
        """Every custom encoding spec takes reasoning history natively.

        The alternative splices a detector's markers into content, which an
        encoder that frames its own channels turns into visible raw markers.
        A new spec must not silently default to that path.
        """
        for spec in _ALL_CHAT_ENCODING_SPECS:
            with self.subTest(chat_encoding_spec=spec):
                self.chat.chat_encoding_spec = spec
                self.assertTrue(self.chat.supports_native_reasoning_history())

        # The HF chat-template path keeps the wrap-into-content behaviour.
        self.chat.chat_encoding_spec = None
        self.assertFalse(self.chat.supports_native_reasoning_history())

    # ------------- dsv4 task -------------
    def test_dsv4_task_field_schema(self):
        """Top-level `task` accepts the 6 DS task tokens and rejects others."""
        for valid in ("action", "query", "authority", "domain", "title", "read_url"):
            req = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
                task=valid,
            )
            self.assertEqual(req.task, valid)

        # None / unset is fine
        self.assertIsNone(self.basic_req.task)

        # Bogus value rejected at validation time
        from pydantic import ValidationError

        with self.assertRaises(ValidationError):
            ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "hi"}],
                task="bogus",
            )

    def test_streaming_abort_yields_error(self):
        """Test that an abort finish reason during streaming correctly yields an error and stops."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort():
            yield {
                "text": "Partial ",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": [],
                    "output_token_logprobs_length": 0,
                    "output_top_logprobs": [],
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate_abort()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
            logprobs=True,
            top_logprobs=5,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                try:
                    async for chunk in self.chat._generate_chat_stream(
                        adapted_request, req, self.fastapi_request
                    ):
                        chunks.append(chunk)
                except Exception as e:
                    print(f"Error during stream iteration: {e}")
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        error_chunk_data = None
        for c in chunks:
            if "error" in c:
                error_chunk_data = json.loads(c[len("data: ") :])
                break
        self.assertIsNotNone(error_chunk_data, "Error chunk not found in stream")
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

        # Ensure the stream stops after the abort error
        # The last chunk should be "data: [DONE]\n\n"
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")

        # Check that there is an error chunk and a DONE chunk
        self.assertEqual(len(chunks), 2)
        self.assertIn("error", chunks[0])

    def test_streaming_abort_with_ids_enabled(self):
        """Test that a terminal abort with input/output ids enabled yields only an error and [DONE]."""
        err_msg = "Aborted by scheduler"
        err_code = HTTPStatus.INTERNAL_SERVER_ERROR

        async def _mock_generate_abort():
            yield {
                "text": "Partial ",
                "prompt_token_ids": [4, 5, 6],
                "output_ids": [1, 2, 3],
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": err_msg,
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate_abort()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
            return_input_ids_in_sglext=True,
            return_output_ids_in_sglext=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                try:
                    async for chunk in self.chat._generate_chat_stream(
                        adapted_request, req, self.fastapi_request
                    ):
                        chunks.append(chunk)
                except Exception as e:
                    print(f"Error during stream iteration: {e}")
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        # Exactly one error chunk followed by [DONE]; no sglext ids leak.
        self.assertIn("error", chunks[0])
        self.assertEqual(chunks[1], "data: [DONE]\n\n")
        self.assertFalse(
            any("input_ids" in c or "output_ids" in c for c in chunks),
            "sglext ids event leaked after abort error",
        )

        error_chunk_data = json.loads(chunks[0][len("data: ") :])
        self.assertEqual(error_chunk_data["error"]["message"], err_msg)
        self.assertEqual(error_chunk_data["error"]["code"], err_code.value)

    def test_streaming_error_abort_still_finalizes_other_choices(self):
        """An error abort ends generation but still runs finalization, so a
        sibling choice's finish chunk and the usage chunk follow the error."""
        err_code = HTTPStatus.SERVICE_UNAVAILABLE

        async def _mock_generate():
            yield {
                "text": "hello",
                "prompt_token_ids": [4, 5, 6],
                "output_ids": [1, 2],
                "meta_info": {
                    "id": "chatcmpl-multi-abort",
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }
            yield {
                "text": "partial",
                "prompt_token_ids": [4, 5, 6],
                "output_ids": [7],
                "meta_info": {
                    "id": "chatcmpl-multi-abort",
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {
                        "type": "abort",
                        "status_code": err_code,
                        "message": "Aborted by scheduler",
                    },
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 1,
            }

        self.tm.generate_request.return_value = _mock_generate()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            n=2,
            stream=True,
            stream_options={"include_usage": True},
            return_input_ids_in_sglext=True,
            return_output_ids_in_sglext=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )
            chunks = self._run_chat_stream(adapted_request, req)

        error_idx = next(i for i, c in enumerate(chunks) if "error" in c)
        after_error = self._parse_chunks(chunks[error_idx + 1 :])

        finish_reasons = [
            choice["finish_reason"]
            for c in after_error
            for choice in c.get("choices", [])
            if choice.get("finish_reason") is not None
        ]
        self.assertEqual(finish_reasons, ["stop"])
        self.assertTrue(
            any(c.get("usage") is not None for c in after_error),
            "usage chunk dropped after error abort",
        )

    def test_truncated_response_template_call_keeps_stop_without_arguments(self):
        self.chat.tool_call_parser = "response_template"
        self.tm.tokenizer.response_template = _RESPONSE_TEMPLATE
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        request._response_parser_prefix = "<assistant>"

        async def generate():
            yield {
                "text": '<call:get_weather>{"city":',
                "meta_info": {
                    "id": "chatcmpl-truncated-call",
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": "CUSTOM_STOP"},
                },
                "index": 0,
            }

        self.tm.generate_request = Mock(return_value=generate())
        adapted_request = GenerateReqInput(
            text="<assistant>",
            sampling_params={},
            stream=True,
        )

        chunks = self._run_chat_stream(adapted_request, request)
        parsed = self._parse_chunks(chunks)
        arguments = "".join(
            tool_call["function"]["arguments"]
            for chunk in parsed
            for choice in chunk.get("choices", [])
            for tool_call in choice.get("delta", {}).get("tool_calls", [])
        )
        finish = next(
            choice
            for chunk in parsed
            for choice in chunk.get("choices", [])
            if choice.get("finish_reason") is not None
        )

        self.assertEqual(arguments, "")
        self.assertEqual(finish["finish_reason"], "stop")
        self.assertEqual(finish["matched_stop"], "CUSTOM_STOP")

    def _run_chat_stream(self, adapted_request, req):
        async def run_stream():
            chunks = []
            async for chunk in self.chat._generate_chat_stream(
                adapted_request, req, self.fastapi_request
            ):
                chunks.append(chunk)
            return chunks

        return get_or_create_event_loop().run_until_complete(run_stream())

    def _parse_chunks(self, chunks):
        parsed = []
        for c in chunks:
            if c.startswith("data: ") and c != "data: [DONE]\n\n":
                parsed.append(json.loads(c[len("data: ") :]))
        return parsed

    def test_streaming_created_is_stable_across_chunks(self):
        for spec in (None, "inkling"):
            with self.subTest(spec=spec):
                if spec == "inkling":
                    self._stream_inkling_output_in_two_chunks()
                self._assert_streaming_created_is_stable()

    def _stream_inkling_output_in_two_chunks(self):
        from sglang.srt.parser.inkling_renderer import load_tml_renderers

        tokenizer = load_tml_renderers().tokenizer
        output_ids = [
            tokenizer.encode_special("message_model"),
            tokenizer.encode_special("content_text"),
            *tokenizer.encode_ordinary("Hello there"),
            tokenizer.encode_special("end_message"),
        ]

        async def generate():
            for count, finish in (
                (len(output_ids) - 1, None),
                (len(output_ids), {"type": "stop", "matched": None}),
            ):
                yield {
                    "text": "",
                    "output_ids": output_ids[:count],
                    "index": 0,
                    "meta_info": {
                        "id": "chatcmpl-inkling",
                        "prompt_tokens": 2,
                        "completion_tokens": count,
                        "cached_tokens": 0,
                        "finish_reason": finish,
                    },
                }

        self.chat.chat_encoding_spec = "inkling"
        self.chat.reasoning_parser = "inkling"
        self.chat.tool_call_parser = "inkling"
        self.tm.tokenizer = None
        self.chat._encoding_adapter = self.chat._build_chat_encoding_adapter()
        self.tm.generate_request.return_value = generate()

    def _assert_streaming_created_is_stable(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            stream_options={"include_usage": True},
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.time.time",
            side_effect=[100, 101, 102, 103],
        ) as current_time:
            chunks = self._run_chat_stream(Mock(), req)

        parsed = self._parse_chunks(chunks)
        self.assertGreater(len(parsed), 1)
        identities = {
            (chunk["id"], chunk["created"], chunk["model"]) for chunk in parsed
        }
        self.assertEqual(len(identities), 1)
        response_id, created, model = identities.pop()
        self.assertTrue(response_id)
        self.assertEqual((created, model), (100, "x"))
        self.assertTrue(
            any(not chunk["choices"] and chunk.get("usage") for chunk in parsed)
        )
        current_time.assert_called_once_with()

    async def _collect_stream_content(self, content, choice_logprobs, req):
        chunks = []
        async for chunk in self.chat._generate_stream_content(
            content=content,
            index=0,
            request=req,
            created=123,
            stream_offsets={},
            reasoning_parser_dict={},
            parser_dict={},
            has_tool_calls={},
            choice_logprobs=choice_logprobs,
            finish_reason_type="stop",
            continuous_usage_stats=False,
            prompt_tokens={0: 5},
            reasoning_tokens={0: 0},
            completion_tokens={0: 1},
        ):
            chunks.append(chunk)
        return chunks

    def test_streaming_top_logprobs_follow_each_token_in_chunk(self):
        """Each token of a multi-token streaming chunk keeps its own alternatives."""
        content = {
            "meta_info": {
                "output_token_logprobs": [
                    (-0.1, 1, "a"),
                    (-0.2, 2, "b"),
                    (-0.3, 3, "c"),
                ],
                "output_top_logprobs": [
                    [(-0.1, 1, "a"), (-2.0, 9, "x")],
                    [(-0.2, 2, "b"), (-3.0, 8, "y")],
                    [(-0.3, 3, "c"), (-4.0, 7, "z")],
                ],
            },
        }
        choice_logprobs = self.chat._process_streaming_logprobs(content, 1, 3)
        tokens = [entry.token for entry in choice_logprobs.content]
        alternatives = [
            [top.token for top in entry.top_logprobs]
            for entry in choice_logprobs.content
        ]
        self.assertEqual(tokens, ["b", "c"])
        self.assertEqual(alternatives, [["b", "y"], ["c", "z"]])

    def test_streaming_logprobs_attached_with_reasoning_parser(self):
        """Logprobs must ride on the reasoning chunk when a reasoning parser is active."""
        self.chat.reasoning_parser = "qwen3"

        content = {
            "text": "thinking...",
            "meta_info": {
                "id": "chatcmpl-reasoning",
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [(0.1, 1, "think"), (0.2, 2, "ing")],
                "output_top_logprobs": [],
                "output_token_logprobs_length": 2,
            },
            "index": 0,
        }
        choice_logprobs = self.chat._process_streaming_logprobs(
            content, 0, 2
        ).model_dump()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            logprobs=True,
            separate_reasoning=True,
        )

        with patch.object(self.chat, "_process_reasoning_stream") as proc_mock:
            proc_mock.return_value = ("Let me think", "")
            chunks = get_or_create_event_loop().run_until_complete(
                self._collect_stream_content(content, choice_logprobs, req)
            )

        parsed = self._parse_chunks(chunks)
        reasoning_chunks = [
            c
            for c in parsed
            if c["choices"][0]["delta"].get("reasoning_content") is not None
        ]
        self.assertTrue(
            reasoning_chunks, "reasoning_parser did not emit a reasoning_content chunk"
        )
        logprob_chunks = [
            c for c in parsed if c["choices"][0].get("logprobs") is not None
        ]
        self.assertTrue(
            logprob_chunks,
            "logprobs dropped: no chunk carried logprobs with reasoning_parser active",
        )

    def test_streaming_logprobs_flushed_when_tool_parser_buffers_delta(self):
        """Logprobs must be flushed on a standalone chunk when the tool parser emits no content delta."""
        self.chat.tool_call_parser = "hermes"

        content = {
            "text": "(<",
            "meta_info": {
                "id": "chatcmpl-tool",
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [(0.3, 9, "(<")],
                "output_top_logprobs": [],
                "output_token_logprobs_length": 1,
            },
            "index": 0,
        }
        choice_logprobs = self.chat._process_streaming_logprobs(
            content, 0, 1
        ).model_dump()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
            logprobs=True,
        )

        async def _empty_tool_stream(*args, **kwargs):
            return
            yield  # make it an async generator

        with patch.object(self.chat, "_process_tool_call_stream", _empty_tool_stream):
            chunks = get_or_create_event_loop().run_until_complete(
                self._collect_stream_content(content, choice_logprobs, req)
            )

        parsed = self._parse_chunks(chunks)
        logprob_chunks = [
            c for c in parsed if c["choices"][0].get("logprobs") is not None
        ]
        self.assertTrue(
            logprob_chunks,
            "logprobs dropped: no flush chunk carried logprobs when tool parser buffered the delta",
        )

    def test_streaming_logprobs_not_flushed_on_empty_delta_step_without_parser(self):
        """With no parser active, an empty-delta step must not emit a standalone
        empty-delta logprobs chunk — clients expect each chunk to carry real
        content/reasoning/tool_calls or a finish_reason."""
        self.chat.reasoning_parser = None
        self.chat.tool_call_parser = None

        async def _mock_generate():
            yield {
                "text": "",
                "meta_info": {
                    "id": "chatcmpl-empty",
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.5, 7, "")],
                    "output_top_logprobs": [],
                    "output_token_logprobs_length": 1,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
            logprobs=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins
            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )
            chunks = self._run_chat_stream(adapted_request, req)

        parsed = self._parse_chunks(chunks)
        empty_logprob_chunks = [
            c
            for c in parsed
            if c["choices"][0].get("logprobs") is not None
            and not c["choices"][0]["delta"].get("content")
            and not c["choices"][0]["delta"].get("reasoning_content")
            and not c["choices"][0]["delta"].get("tool_calls")
            and not c["choices"][0].get("finish_reason")
        ]
        self.assertFalse(
            empty_logprob_chunks,
            "empty-delta logprobs chunk emitted without a parser; would break client chunk-shape assumptions",
        )

    def test_non_streaming_extension_fields_emit_sglext_without_meta_info(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            return_meta_info=False,
            return_routed_experts=True,
            return_cached_tokens_details=True,
        )
        ret = [
            {
                "text": "Cached response",
                "meta_info": {
                    "id": "chatcmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "routed_experts": "cm91dGUtYQ==",
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.chat._build_chat_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertEqual(response.sglext.routed_experts, "cm91dGUtYQ==")
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )
        self.assertIsNone(response.choices[0].meta_info)
        dumped_response = response.model_dump()
        self.assertIn("sglext", dumped_response)
        self.assertNotIn("meta_info", dumped_response["choices"][0])

    def test_non_streaming_meta_info_omits_response_level_routed_experts(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            n=2,
            return_meta_info=True,
            return_routed_experts=True,
        )
        ret = [
            {
                "text": f"Response {index}",
                "meta_info": {
                    "id": "chatcmpl-meta-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": index,
                    "routed_experts": routed_experts,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
            for index, routed_experts in enumerate(["cm91dGUtYQ==", "cm91dGUtYg=="])
        ]

        response = self.chat._build_chat_response(req, ret, 1234567890)

        self.assertIsNone(
            response.sglext,
            "sglext is absent only when routed_experts is the sole extension",
        )
        self.assertEqual(
            [choice.meta_info for choice in response.choices],
            [ret_item["meta_info"] for ret_item in ret],
        )
        dumped_response = response.model_dump()
        self.assertNotIn("sglext", dumped_response)
        self.assertEqual(
            [choice["meta_info"] for choice in dumped_response["choices"]],
            [ret_item["meta_info"] for ret_item in ret],
        )
        serialized_response = json.dumps(dumped_response)
        self.assertEqual(serialized_response.count('"routed_experts"'), 2)

    def test_non_streaming_meta_info_preserves_cache_and_spec_in_sglext(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            n=2,
            return_meta_info=True,
            return_routed_experts=True,
            return_cached_tokens_details=True,
            return_spec_tokens_details=True,
        )
        routed_experts = ["cm91dGUtYQ==", "cm91dGUtYg=="]
        ret = [_spec_result(index) for index in range(2)]
        for index, ret_item in enumerate(ret):
            ret_item["meta_info"].update(
                {
                    "cached_tokens_details": {
                        "device": 4 - index,
                        "host": index,
                    },
                    "routed_experts": routed_experts[index],
                }
            )

        response = self.chat._build_chat_response(req, ret, 1234567890)

        self.assertIsNotNone(response.sglext)
        self.assertIsNone(response.sglext.routed_experts)
        self.assertEqual(
            response.sglext.cached_tokens_details.model_dump(exclude_none=True),
            {"device": 4, "host": 0},
        )
        self.assertEqual(
            [item.spec_cap_length for item in response.sglext.spec_tokens_details],
            [1.0, 2.0],
        )
        self.assertEqual(
            [choice.meta_info for choice in response.choices],
            [ret_item["meta_info"] for ret_item in ret],
        )
        dumped_response = response.model_dump()
        self.assertIn("sglext", dumped_response)
        self.assertIn("spec_tokens_details", dumped_response["sglext"])
        self.assertNotIn("routed_experts", dumped_response["sglext"])
        self.assertEqual(
            dumped_response["sglext"]["cached_tokens_details"],
            {"device": 4, "host": 0},
        )
        serialized_response = json.dumps(dumped_response)
        self.assertEqual(serialized_response.count('"routed_experts"'), 2)

    def test_parallel_sampling_returns_spec_details_per_choice(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            n=2,
            return_spec_tokens_details=True,
        )
        ret = [_spec_result(index) for index in range(2)]

        response = self.chat._build_chat_response(req, ret, 1234567890)

        details = response.sglext.spec_tokens_details
        self.assertEqual([item.spec_cap_length for item in details], [1.0, 2.0])
        self.assertEqual(
            [item.spec_cap_lens_histogram for item in details],
            [[0, 1], [1, 1]],
        )

        single_req = req.model_copy(update={"n": 1})
        single_response = self.chat._build_chat_response(
            single_req, ret[:1], 1234567890
        )
        self.assertEqual(
            single_response.sglext.spec_tokens_details.spec_cap_length,
            1.0,
        )

    def test_non_streaming_chat_response_returns_requested_token_ids_and_meta_info(
        self,
    ):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            return_prompt_token_ids=True,
            return_token_ids=True,
            return_meta_info=True,
        )
        ret = [
            {
                "text": "Answer",
                "output_ids": [21, 22],
                "prompt_token_ids": [11, 12, 13],
                "meta_info": {
                    "id": "chatcmpl-token-ids",
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
        ]

        response = self.chat._build_chat_response(req, ret, created=123)
        choice = response.choices[0]

        self.assertEqual(choice.prompt_token_ids, [11, 12, 13])
        self.assertEqual(choice.response_token_ids, [21, 22])
        self.assertEqual(choice.meta_info, ret[0]["meta_info"])
        dumped_choice = response.model_dump()["choices"][0]
        self.assertEqual(dumped_choice["prompt_token_ids"], [11, 12, 13])
        self.assertEqual(dumped_choice["response_token_ids"], [21, 22])
        self.assertEqual(dumped_choice["meta_info"], ret[0]["meta_info"])

    def test_streaming_cached_tokens_details_emits_sglext(self):
        """Test that streaming chat responses emit cached token details in sglext."""

        async def _mock_generate_with_cached_tokens_details():
            yield {
                "text": "Cached response",
                "meta_info": {
                    "id": "chatcmpl-cache-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 6,
                    "cached_tokens_details": {
                        "device": 4,
                        "host": 1,
                        "storage": 1,
                        "storage_backend": "file",
                    },
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = (
            _mock_generate_with_cached_tokens_details()
        )

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            stream=True,
            return_cached_tokens_details=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                async for chunk in self.chat._generate_chat_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        sglext_chunks = []
        for chunk in chunks:
            if not chunk.startswith("data: ") or chunk.strip() == "data: [DONE]":
                continue
            data = json.loads(chunk[len("data: ") :])
            if "sglext" in data:
                sglext_chunks.append(data)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        self.assertEqual(
            sglext_chunks[0]["sglext"]["cached_tokens_details"],
            {
                "device": 4,
                "host": 1,
                "storage": 1,
                "storage_backend": "file",
            },
        )

    def _output_ids_ret(self, *output_ids_by_choice):
        return [
            {
                "text": "Answer",
                "prompt_token_ids": [1, 2, 3],
                "output_ids": list(output_ids),
                "meta_info": {
                    "id": "chatcmpl-output-ids",
                    "prompt_tokens": 3,
                    "completion_tokens": len(output_ids),
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "default",
                },
            }
            for output_ids in output_ids_by_choice
        ]

    def test_non_streaming_ids_emit_sglext(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            n=2,
            return_input_ids_in_sglext=True,
            return_output_ids_in_sglext=True,
        )

        response = self.chat._build_chat_response(
            req, self._output_ids_ret([5, 6, 7], [8, 9]), 123
        )

        self.assertIsNotNone(response.sglext)
        self.assertEqual(response.sglext.input_ids, [1, 2, 3])
        self.assertEqual(response.sglext.output_ids, [[5, 6, 7], [8, 9]])
        dumped = json.loads(response.model_dump_json())
        self.assertEqual(dumped["sglext"]["input_ids"], [1, 2, 3])
        self.assertEqual(dumped["sglext"]["output_ids"], [[5, 6, 7], [8, 9]])
        # None sglext fields stay stripped from the serialized response.
        self.assertNotIn("routed_experts", dumped["sglext"])

    def test_non_streaming_ids_not_returned_by_default(self):
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )

        response = self.chat._build_chat_response(
            req, self._output_ids_ret([5, 6, 7]), 123
        )

        self.assertIsNone(response.sglext)

    def test_non_streaming_ids_server_default_enables_flag(self):
        enter_override(
            self,
            get_context().override_server_args(
                return_input_ids=True, return_output_ids=True
            ),
        )
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )

        response = self.chat._build_chat_response(
            req, self._output_ids_ret([5, 6, 7]), 123
        )

        self.assertIsNotNone(response.sglext)
        self.assertEqual(response.sglext.input_ids, [1, 2, 3])
        self.assertEqual(response.sglext.output_ids, [[5, 6, 7]])

    def test_ids_headers_enable_flags(self):
        self.fastapi_request.headers = {
            "x-sglext-return-input-ids": "1",
            "x-sglext-return-output-ids": "1",
        }
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
        )
        processed_messages = MessageProcessingResult(
            "Test prompt",
            [1, 2, 3],
            None,
            None,
            [],
            ["</s>"],
            None,
        )

        with patch.object(
            self.chat, "_process_messages", return_value=processed_messages
        ):
            _, processed_request = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

        self.assertTrue(processed_request.return_input_ids_in_sglext)
        self.assertTrue(processed_request.return_output_ids_in_sglext)

    def _run_output_ids_stream(
        self,
        chunk_output_ids,
        incremental,
        framed=False,
        return_raw=False,
        cached_tokens_details=None,
    ):
        """Stream chunks with incremental or cumulative output_ids;
        return parsed sglext chunks, or raw SSE strings when return_raw."""
        enter_override(
            self,
            get_context().override_server_args(
                incremental_streaming_output=incremental
            ),
        )
        if framed:
            self.fastapi_request.headers["x-sglext-ids-framed"] = "1"

        async def _mock_generate():
            for i, ids in enumerate(chunk_output_ids):
                finished = i == len(chunk_output_ids) - 1
                yield {
                    "text": "chunk",
                    "prompt_token_ids": [1, 2, 3],
                    "output_ids": list(ids),
                    "meta_info": {
                        "id": "chatcmpl-output-ids-stream",
                        "prompt_tokens": 3,
                        "completion_tokens": 1 + i,
                        "cached_tokens": 0,
                        "cached_tokens_details": cached_tokens_details,
                        "finish_reason": (
                            {"type": "stop", "matched": None} if finished else None
                        ),
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = _mock_generate()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            stream=True,
            return_input_ids_in_sglext=True,
            return_output_ids_in_sglext=True,
            return_cached_tokens_details=cached_tokens_details is not None,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )
            chunks = self._run_chat_stream(adapted_request, req)

        if return_raw:
            return chunks
        return [c for c in self._parse_chunks(chunks) if "sglext" in c]

    def test_streaming_output_ids_incremental_accumulates_deltas(self):
        sglext_chunks = self._run_output_ids_stream([[5, 6], [7]], incremental=True)

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["choices"], [])
        # input_ids is captured once as a flat shared prompt, not accumulated.
        self.assertEqual(sglext_chunks[0]["sglext"]["input_ids"], [1, 2, 3])
        self.assertEqual(sglext_chunks[0]["sglext"]["output_ids"], [[5, 6, 7]])

    def test_streaming_output_ids_non_incremental_keeps_latest_full_list(self):
        sglext_chunks = self._run_output_ids_stream(
            [[5, 6], [5, 6, 7]], incremental=False
        )

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["sglext"]["input_ids"], [1, 2, 3])
        self.assertEqual(sglext_chunks[0]["sglext"]["output_ids"], [[5, 6, 7]])

    _SGLEXT_IDS_EVENT_PREFIX = "event: sglext_ids\ndata: "

    def test_streaming_framed_ids_emits_named_event(self):
        raw = self._run_output_ids_stream(
            [[5, 6], [5, 6, 7]], incremental=False, framed=True, return_raw=True
        )

        named = [c for c in raw if c.startswith(self._SGLEXT_IDS_EVENT_PREFIX)]
        self.assertEqual(len(named), 1)
        payload = json.loads(named[0][len(self._SGLEXT_IDS_EVENT_PREFIX) :])
        self.assertEqual(payload["choices"], [])
        # The named event carries ONLY the id fields.
        self.assertEqual(set(payload["sglext"]), {"input_ids", "output_ids"})
        self.assertEqual(payload["sglext"]["input_ids"], [1, 2, 3])
        self.assertEqual(payload["sglext"]["output_ids"], [[5, 6, 7]])
        # All requested sglext fields were ids, so no plain sglext chunk.
        self.assertEqual([c for c in self._parse_chunks(raw) if "sglext" in c], [])

    def test_streaming_framed_splits_non_id_fields_into_plain_chunk(self):
        raw = self._run_output_ids_stream(
            [[5, 6], [5, 6, 7]],
            incremental=False,
            framed=True,
            return_raw=True,
            cached_tokens_details={"device": 2, "host": 1},
        )

        named = [c for c in raw if c.startswith(self._SGLEXT_IDS_EVENT_PREFIX)]
        self.assertEqual(len(named), 1)
        named_payload = json.loads(named[0][len(self._SGLEXT_IDS_EVENT_PREFIX) :])
        self.assertEqual(set(named_payload["sglext"]), {"input_ids", "output_ids"})

        plain_sglext = [c for c in self._parse_chunks(raw) if "sglext" in c]
        self.assertEqual(len(plain_sglext), 1)
        self.assertEqual(
            plain_sglext[0]["sglext"]["cached_tokens_details"],
            {"device": 2, "host": 1},
        )
        self.assertNotIn("input_ids", plain_sglext[0]["sglext"])
        self.assertNotIn("output_ids", plain_sglext[0]["sglext"])
        # The plain non-id chunk precedes the named ids event.
        plain_raw_idx = next(
            i
            for i, c in enumerate(raw)
            if c.startswith("data: ") and "cached_tokens_details" in c
        )
        self.assertLess(plain_raw_idx, raw.index(named[0]))

    def _run_output_ids_stream_with_graceful_abort(
        self, normal_chunks, abort_output_ids, incremental, abort_completion_tokens
    ):
        """Stream normal chunks then a graceful-abort chunk; return parsed sglext chunks."""
        enter_override(
            self,
            get_context().override_server_args(
                incremental_streaming_output=incremental
            ),
        )

        async def _mock_generate():
            generated = 0
            for ids in normal_chunks:
                generated = generated + len(ids) if incremental else len(ids)
                yield {
                    "text": "chunk",
                    "output_ids": list(ids),
                    "meta_info": {
                        "id": "chatcmpl-abort-ids-stream",
                        "prompt_tokens": 3,
                        "completion_tokens": generated,
                        "cached_tokens": 0,
                        "finish_reason": None,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }
            # Graceful abort terminal chunk (no status_code): falls through to
            # the normal finalization path.
            yield {
                "text": "chunk",
                "output_ids": list(abort_output_ids),
                "meta_info": {
                    "id": "chatcmpl-abort-ids-stream",
                    "prompt_tokens": 3,
                    "completion_tokens": abort_completion_tokens,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "abort", "message": "Aborted."},
                    "output_token_logprobs": None,
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.tm.generate_request.return_value = _mock_generate()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            stream=True,
            return_output_ids_in_sglext=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )
            chunks = self._run_chat_stream(adapted_request, req)

        return [c for c in self._parse_chunks(chunks) if "sglext" in c]

    def test_streaming_output_ids_incremental_ignores_abort_chunk(self):
        # Abort re-sends the last token; it must not be appended again.
        sglext_chunks = self._run_output_ids_stream_with_graceful_abort(
            normal_chunks=[[5, 6], [7]],
            abort_output_ids=[7],
            incremental=True,
            abort_completion_tokens=3,
        )

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["sglext"]["output_ids"], [[5, 6, 7]])

    def test_streaming_output_ids_incremental_keeps_coalesced_abort_deltas(self):
        # Coalesced abort chunk: keep real deltas, drop the repeated last token.
        sglext_chunks = self._run_output_ids_stream_with_graceful_abort(
            normal_chunks=[[5, 6]],
            abort_output_ids=[7, 8, 8],
            incremental=True,
            abort_completion_tokens=4,
        )

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["sglext"]["output_ids"], [[5, 6, 7, 8]])

    def test_streaming_output_ids_non_incremental_abort_supersedes_earlier(self):
        sglext_chunks = self._run_output_ids_stream_with_graceful_abort(
            normal_chunks=[[5, 6]],
            abort_output_ids=[5, 6, 7],
            incremental=False,
            abort_completion_tokens=3,
        )

        self.assertEqual(len(sglext_chunks), 1)
        self.assertEqual(sglext_chunks[0]["sglext"]["output_ids"], [[5, 6, 7]])

    def test_streaming_parallel_sampling_orders_spec_details_by_choice(self):
        async def mock_generate():
            for index in (1, 0):
                yield _spec_result(index)

        self.tm.generate_request.return_value = mock_generate()
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            max_tokens=100,
            n=2,
            stream=True,
            return_spec_tokens_details=True,
        )

        parsed = self._parse_chunks(self._run_chat_stream(Mock(), req))
        details = next(chunk["sglext"] for chunk in parsed if "sglext" in chunk)[
            "spec_tokens_details"
        ]
        self.assertEqual([item["spec_cap_length"] for item in details], [1.0, 2.0])
        self.assertEqual(
            [item["spec_cap_lens_histogram"] for item in details],
            [[0, 1], [1, 1]],
        )

    def _collect_continuous_usage(self, cached_tokens):
        content = {
            "text": "Hello",
            "meta_info": {
                "id": "chatcmpl-cont-usage",
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "cached_tokens": cached_tokens,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": None,
                "output_top_logprobs": None,
            },
            "index": 0,
        }
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=True,
        )

        async def _collect():
            chunks = []
            async for chunk in self.chat._generate_stream_content(
                content=content,
                index=0,
                request=req,
                created=123,
                stream_offsets={},
                reasoning_parser_dict={},
                parser_dict={},
                has_tool_calls={},
                choice_logprobs=None,
                finish_reason_type="stop",
                continuous_usage_stats=True,
                prompt_tokens={0: 10},
                reasoning_tokens={0: 0},
                completion_tokens={0: 2},
            ):
                chunks.append(chunk)
            return chunks

        chunks = get_or_create_event_loop().run_until_complete(_collect())
        return [c["usage"] for c in self._parse_chunks(chunks) if c.get("usage")]

    def test_continuous_usage_reports_cached_tokens(self):
        """continuous_usage_stats chunks include cached tokens when cache reporting is on."""
        enter_override(
            self, get_context().override_server_args(enable_cache_report=True)
        )
        for cached_tokens in (0, 6):
            with self.subTest(cached_tokens=cached_tokens):
                usages = self._collect_continuous_usage(cached_tokens=cached_tokens)
                self.assertTrue(usages, "continuous_usage_stats attached no usage")
                self.assertEqual(
                    usages[0]["prompt_tokens_details"]["cached_tokens"],
                    cached_tokens,
                )

    def test_continuous_usage_omits_cached_tokens_when_report_disabled(self):
        """With cache reporting off, continuous_usage_stats must not leak cached tokens."""
        enter_override(
            self, get_context().override_server_args(enable_cache_report=False)
        )
        usages = self._collect_continuous_usage(cached_tokens=6)
        self.assertTrue(usages, "continuous_usage_stats attached no usage")
        self.assertIsNone(usages[0].get("prompt_tokens_details"))

    # ------------- incremental streaming output tests -------------
    def test_incremental_streaming_output_delta(self):
        """Test that streaming with incremental_streaming_output produces correct deltas.

        When incremental_streaming_output is enabled, content["text"] is already the
        incremental delta (not the full accumulated text). The delta computation must
        use content["text"] directly instead of slicing by the accumulated buffer length.

        Regression test for https://github.com/sgl-project/sglang/issues/22510.
        """
        # Enable incremental_streaming_output on the mock
        enter_override(
            self, get_context().override_server_args(incremental_streaming_output=True)
        )

        # Simulate incremental streaming: each yield has ONLY the new text (delta),
        # NOT the full accumulated text.
        incremental_chunks = [
            ("I am", None),
            (" a large", None),
            (" language model", None),
            (".", {"type": "stop", "matched": None}),
        ]

        async def _mock_generate_incremental():
            for text, finish_reason in incremental_chunks:
                yield {
                    "text": text,
                    "meta_info": {
                        "id": "chatcmpl-incr-test",
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "cached_tokens": 0,
                        "finish_reason": finish_reason,
                        "output_token_logprobs": None,
                        "output_top_logprobs": None,
                    },
                    "index": 0,
                }

        self.tm.generate_request.return_value = _mock_generate_incremental()

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_mock.return_value = conv_ins

            adapted_request, _ = self.chat._convert_to_internal_request(
                req, self.fastapi_request
            )

            async def run_stream():
                chunks = []
                async for chunk in self.chat._generate_chat_stream(
                    adapted_request, req, self.fastapi_request
                ):
                    chunks.append(chunk)
                return chunks

        loop = get_or_create_event_loop()
        chunks = loop.run_until_complete(run_stream())

        # Extract content deltas from SSE chunks
        deltas = []
        for c in chunks:
            if not c.startswith("data: ") or c.strip() == "data: [DONE]":
                continue
            data = json.loads(c[len("data: ") :])
            if "choices" in data and data["choices"]:
                content = data["choices"][0]["delta"].get("content")
                if content:
                    deltas.append(content)

        joined = "".join(deltas)
        self.assertEqual(
            joined,
            "I am a large language model.",
            f"Streaming deltas produced broken text: {deltas!r}",
        )

    # ------------- X-Data-Parallel-Rank header tests -------------
    def test_extract_routed_dp_rank_from_header_no_header(self):
        """Test that None is returned when no header is present."""
        self.fastapi_request.headers = {}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=None
        )
        self.assertIsNone(result)

    def test_extract_routed_dp_rank_header_overrides_body(self):
        """Test that header value has higher priority than body."""
        self.fastapi_request.headers = {"x-data-parallel-rank": "3"}
        result = self.chat.extract_routed_dp_rank_from_header(
            self.fastapi_request, body_routed_dp_rank=1
        )
        self.assertEqual(result, 3)  # header wins

    def test_extract_routed_dp_rank_from_header_invalid(self):
        """Test that invalid header value raises HTTPException."""
        from fastapi import HTTPException

        self.fastapi_request.headers = {"x-data-parallel-rank": "abc"}
        with self.assertRaises(HTTPException) as context:
            self.chat.extract_routed_dp_rank_from_header(
                self.fastapi_request, body_routed_dp_rank=None
            )
        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("must be an integer", context.exception.detail)

    def test_hunyuan_reasoning_effort_dispatch(self):
        tm = _MockTokenizerManager()
        tm.server_args.reasoning_parser = "hunyuan"
        chat = OpenAIServingChat(tm, _MockTemplateManager())
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        cases = [
            ("no_think", False),
            ("none", False),
            (None, False),
            ("high", True),
            ("low", True),
        ]
        for effort, expected in cases:
            with self.subTest(effort=effort):
                req.reasoning_effort = effort
                self.assertEqual(chat._get_reasoning_from_request(req), expected)

    def _setup_nemotron_super(self):
        """Drive _apply_jinja_template (chat_template_name=None) with a
        Nemotron-3 Super reasoning_config carrying effort_kwarg."""
        self.tm.server_args.reasoning_parser = "nemotron_3"
        self.chat.reasoning_parser = "nemotron_3"
        self.tm.server_args.served_model_name = (
            "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
        )
        self.template_manager.chat_template_name = None
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking",
            default_enabled=True,
            effort_kwarg="low_effort",
        )
        self.chat.chat_encoding_spec = None

    def _run_jinja_with_effort(self, effort):
        req = ChatCompletionRequest(
            model="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=effort,
        )
        with (
            patch.object(self.chat, "_encode_messages", return_value=None),
            patch.object(
                self.chat,
                "_handle_last_assistant_message",
                return_value=([{"role": "user", "content": "hi"}], None),
            ),
        ):
            self.chat._process_messages(req, False)
        _, kwargs = self.tm.tokenizer.apply_chat_template.call_args
        return kwargs

    def test_nemotron_super_low_effort_mapped_to_kwarg(self):
        self._setup_nemotron_super()
        kwargs = self._run_jinja_with_effort("low")
        self.assertTrue(kwargs.get("low_effort"))

    def test_nemotron_super_high_effort_warns_without_kwarg(self):
        self._setup_nemotron_super()
        with self.assertLogs(
            "sglang.srt.entrypoints.openai.serving_chat", level="WARNING"
        ) as logs:
            kwargs = self._run_jinja_with_effort("high")
        self.assertNotIn("low_effort", kwargs)
        self.assertTrue(any("only 'low' reasoning effort" in m for m in logs.output))

    def test_nemotron_nano_no_effort_kwarg(self):
        # Nano template has no low_effort, so effort_kwarg stays None and no
        # warning is emitted even for non-low effort.
        self.tm.server_args.reasoning_parser = "nemotron_3"
        self.chat.reasoning_parser = "nemotron_3"
        self.template_manager.chat_template_name = None
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        )
        self.chat.chat_encoding_spec = None
        kwargs = self._run_jinja_with_effort("high")
        self.assertNotIn("low_effort", kwargs)

    def _use_preserved_reasoning_template(self):
        template = _compile_jinja_template(
            "{% for m in messages if m.role == 'assistant' %}"
            "{% if m.reasoning_content is not string %}"
            "{{ raise_exception('Assistant thinking fields must be strings') }}"
            "{% endif %}<think>{{ m.reasoning_content }}</think>{% endfor %}"
        )
        self.tm.tokenizer.apply_chat_template.side_effect = lambda messages, **kwargs: (
            template.render(messages=messages)
        )
        self.template_manager.jinja_template_content_format = "string"
        self.template_manager.force_reasoning = True
        self.chat.reasoning_parser = self.chat.tool_call_parser = "k2_horizon"

    def _assert_empty_reasoning_replays(self, message):
        self.assertEqual(message["reasoning_content"], "")
        request = ChatCompletionRequest(
            model="x",
            messages=[*self.basic_req.messages, message, *self.basic_req.messages],
        )
        self.chat._apply_jinja_template(request, None, False)
        self.assertEqual(self.tm.tokenizer.encode.call_args.args[0], "<think></think>")

    _K2_TOOL = {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object"}},
    }
    _K2_TOOL_TEXT = '<ifm|tool_call>{"name":"lookup","arguments":{}}</ifm|tool_call>'

    def test_empty_reasoning_response_replays(self):
        self._use_preserved_reasoning_template()
        for tool_only in (False, True):
            with self.subTest(tool_only=tool_only):
                request = ChatCompletionRequest(
                    model="x",
                    messages=self.basic_req.messages,
                    tools=[self._K2_TOOL] if tool_only else None,
                )
                ret = _spec_result(0)
                ret["text"] = "</ifm|think>" + (
                    self._K2_TOOL_TEXT if tool_only else "Hello"
                )
                response = self.chat._build_chat_response(request, [ret], created=123)
                message = response.model_dump(mode="json")["choices"][0]["message"]
                self.assertEqual(message["content"], "" if tool_only else "Hello")
                self.assertEqual(bool(message.get("tool_calls")), tool_only)
                self._assert_empty_reasoning_replays(message)

    def test_empty_stream_reasoning_each_choice(self):
        self.template_manager.force_reasoning = True
        self.chat.reasoning_parser = self.chat.tool_call_parser = "k2_horizon"
        for tool_only in (False, True):
            with self.subTest(tool_only=tool_only):
                request = ChatCompletionRequest(
                    model="x",
                    messages=self.stream_req.messages,
                    stream=True,
                    n=2,
                    tools=[self._K2_TOOL] if tool_only else None,
                )

                async def generate():
                    for text in (
                        "</ifm|thi",
                        "</ifm|think>" + (self._K2_TOOL_TEXT if tool_only else "Hello"),
                    ):
                        for index in range(request.n):
                            ret = _spec_result(index)
                            ret["text"] = text
                            if text == "</ifm|thi":
                                ret["meta_info"]["finish_reason"] = None
                            yield ret

                self.tm.generate_request.return_value = generate()
                chunks = self._parse_chunks(self._run_chat_stream(None, request))
                self.assertFalse(any("error" in chunk for chunk in chunks), chunks)
                choices = [c for chunk in chunks for c in chunk.get("choices", [])]
                for index in range(request.n):
                    with self.subTest(index=index):
                        selected = [c for c in choices if c["index"] == index]
                        self.assertTrue(selected)
                        self.assertEqual(selected[0]["delta"]["reasoning_content"], "")
                        for choice in selected:
                            self.assertIn(
                                choice["delta"].get("reasoning_content"), (None, "")
                            )
                        self.assertEqual(
                            selected[-1]["finish_reason"],
                            "tool_calls" if tool_only else "stop",
                        )
                        self.assertEqual(
                            "".join(c["delta"].get("content") or "" for c in selected),
                            "" if tool_only else "Hello",
                        )

    def test_unparsed_reasoning_remains_null(self):
        for parser, separate in ((None, True), ("k2_horizon", False)):
            with self.subTest(parser=parser, separate=separate):
                self.chat.reasoning_parser = parser
                request = self.basic_req.model_copy(
                    update={"separate_reasoning": separate}
                )
                response = self.chat._build_chat_response(
                    request, [_spec_result(0)], created=123
                )
                self.assertIsNone(response.choices[0].message.reasoning_content)

                async def generate():
                    yield _spec_result(0)

                self.tm.generate_request.return_value = generate()
                chunks = self._parse_chunks(self._run_chat_stream(None, request))
                self.assertFalse(any("error" in chunk for chunk in chunks))
                for chunk in chunks:
                    for choice in chunk.get("choices", []):
                        self.assertIsNone(choice["delta"].get("reasoning_content"))

    def test_empty_reasoning_is_null_for_non_k2_parser(self):
        """A non-K2 parser that finds no reasoning reports reasoning_content as None, not ''."""
        self.chat.reasoning_parser = "qwen3"
        self.template_manager.force_reasoning = False
        request = self.basic_req.model_copy(update={"separate_reasoning": True})
        ret = _spec_result(0)
        ret["text"] = "Hello"
        response = self.chat._build_chat_response(request, [ret], created=123)
        message = response.choices[0].message
        self.assertEqual(message.content, "Hello")
        self.assertIsNone(message.reasoning_content)

    def test_missing_reasoning_history_still_rejected(self):
        self._use_preserved_reasoning_template()
        for reasoning in ({}, {"reasoning_content": None}):
            with self.subTest(reasoning=reasoning):
                request = ChatCompletionRequest(
                    model="x",
                    messages=[
                        *self.basic_req.messages,
                        {"role": "assistant", "content": "Hello", **reasoning},
                        *self.basic_req.messages,
                    ],
                )
                with self.assertRaisesRegex(
                    ValueError, "thinking fields must be strings"
                ):
                    self.chat._apply_jinja_template(request, None, False)

    def test_non_stream_reasoning_response_preserves_payload_whitespace(self):
        self.chat.reasoning_parser = "qwen3"
        self.template_manager.force_reasoning = False

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            stream=False,
            separate_reasoning=True,
        )
        ret = [
            {
                "text": "<think>\nLet me think\n</think>\n\nThe answer is 42.\n",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "weight_version": "test",
                },
                "index": 0,
            }
        ]

        response = self.chat._build_chat_response(req, ret, created=123)

        message = response.choices[0].message
        self.assertEqual(message.reasoning_content, "\nLet me think\n")
        self.assertEqual(message.content, "\n\nThe answer is 42.\n")

    # ------------- reasoning config tests -------------
    def test_get_reasoning_from_request_default_true_toggle(self):
        self.tm.server_args.reasoning_parser = "qwen3"
        self.chat.reasoning_parser = "qwen3"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        )

        enabled_by_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        disabled_explicitly = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertTrue(self.chat._get_reasoning_from_request(enabled_by_default))
        self.assertFalse(self.chat._get_reasoning_from_request(disabled_explicitly))

    def test_get_reasoning_from_request_default_false_toggle(self):
        self.tm.server_args.reasoning_parser = "deepseek-v3"
        self.chat.reasoning_parser = "deepseek-v3"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="thinking", default_enabled=False
        )

        disabled_by_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        enabled_explicitly = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"thinking": True},
        )

        self.assertFalse(self.chat._get_reasoning_from_request(disabled_by_default))
        self.assertTrue(self.chat._get_reasoning_from_request(enabled_explicitly))

    def test_get_reasoning_from_request_special_cases(self):
        self.tm.server_args.reasoning_parser = "mistral"
        self.chat.reasoning_parser = "mistral"
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )

        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="always"
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req))

        self.template_manager.reasoning_config = ReasoningToggleConfig(
            special_case="mistral"
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req))
        req.reasoning_effort = "medium"
        self.assertTrue(self.chat._get_reasoning_from_request(req))

    # --- fallback path tests (config=None, uses reasoning_default) ---

    def _setup_fallback(self, parser_name):
        """Set up reasoning with config=None to exercise the fallback path."""
        self.tm.server_args.reasoning_parser = parser_name
        self.chat = OpenAIServingChat(self.tm, self.template_manager)
        self.chat.reasoning_parser = parser_name
        self.template_manager.reasoning_config = None

    def test_fallback_always_mode(self):
        self._setup_fallback("deepseek-r1")
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req))

    def test_fallback_mistral_mode(self):
        self._setup_fallback("mistral")
        req_no_effort = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_no_effort))

        req_with_effort = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            reasoning_effort="high",
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_with_effort))

    def test_fallback_enable_thinking_mode_default_on(self):
        self._setup_fallback("qwen3")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_default))

        req_disabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": False},
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_disabled))

    def test_fallback_explicit_thinking_mode_default_off(self):
        self._setup_fallback("deepseek-v3")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_default))

        req_enabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"thinking": True},
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_enabled))

    def test_fallback_explicit_enable_thinking_mode_default_off(self):
        self._setup_fallback("mimo")
        req_default = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req_default))

        req_enabled = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        self.assertTrue(self.chat._get_reasoning_from_request(req_enabled))

    def test_fallback_ling3_default_on(self):
        """Ling3 public checkpoints default `thinking_option='on'` in the chat
        template when `enable_thinking` is omitted, and the template detector
        cannot infer that indirect assignment. The parser fallback must mirror
        the template default: omitted kwargs enable reasoning, only an explicit
        `enable_thinking=False` disables it. Regression: the detector shipped
        with `explicit_enable_thinking`, which left `reasoning_content` null on
        default requests while the model was in fact thinking."""
        self._setup_fallback("ling3")
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        cases = [
            (None, True),  # no chat_template_kwargs → thinking (template default)
            ({}, True),  # empty kwargs → thinking
            ({"enable_thinking": True}, True),  # explicit on
            ({"enable_thinking": False}, False),  # explicit off
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                req.chat_template_kwargs = kwargs
                self.assertEqual(self.chat._get_reasoning_from_request(req), expected)

    def test_fallback_no_detector_returns_false(self):
        self.chat.reasoning_parser = "qwen3"
        self.chat._reasoning_detector = None
        self.template_manager.reasoning_config = None
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hi?"}]
        )
        self.assertFalse(self.chat._get_reasoning_from_request(req))

    def test_build_chat_response_qwen3_thinking_forces_reasoning(self):
        self.tm.server_args.reasoning_parser = "qwen3-thinking"
        self.chat.reasoning_parser = "qwen3-thinking"
        self.template_manager.reasoning_config = ReasoningToggleConfig(
            toggle_param="enable_thinking", default_enabled=True
        )

        req = ChatCompletionRequest(
            model="Qwen/Qwen3-0.6B",
            messages=[{"role": "user", "content": "Hi?"}],
            separate_reasoning=True,
            chat_template_kwargs={"enable_thinking": False},
        )
        ret_item = {
            "text": "42",
            "meta_info": {
                "id": f"chatcmpl-{uuid.uuid4()}",
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "weight_version": "default",
                "finish_reason": {"type": "stop", "matched": None},
            },
            "index": 0,
        }

        response = self.chat._build_chat_response(req, [ret_item], created=0)
        msg = response.choices[0].message
        self.assertEqual(msg.content, "")
        self.assertEqual(msg.reasoning_content, "42")

    # --- poolside_v1 (Laguna-XS.2) regression tests ---

    def test_poolside_v1_enable_thinking_dispatch(self):
        """Laguna chat template defaults `enable_thinking=false`. Parser must
        follow that default — must NOT return True via the generic fallback.
        After the reasoning-config refactor, this is driven by
        `_PoolsideV1Detector.reasoning_default = "explicit_enable_thinking"`."""
        self._setup_fallback("poolside_v1")
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "hi"}]
        )
        cases = [
            (None, False),  # no chat_template_kwargs → non-thinking (default)
            ({}, False),  # empty kwargs → non-thinking
            ({"enable_thinking": False}, False),  # explicit off
            ({"enable_thinking": True}, True),  # explicit on
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                req.chat_template_kwargs = kwargs
                self.assertEqual(self.chat._get_reasoning_from_request(req), expected)

    def test_poolside_v1_does_not_double_prepend_think(self):
        """When `enable_thinking=True` for poolside_v1, the HF chat template
        already emits `<think>` via add_generation_prompt — server must NOT
        append a second `<think>`. After the refactor this is guarded by
        `_PoolsideV1Detector.thinks_internally = True` (inherited from Qwen3Detector).
        """
        self._setup_fallback("poolside_v1")
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
            chat_template_kwargs={"enable_thinking": True},
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "BASE_PROMPT"
            conv_ins.image_data = conv_ins.audio_data = conv_ins.video_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = []
            conv_mock.return_value = conv_ins
            result = self.chat._apply_conversation_template(req, is_multimodal=False)
        self.assertEqual(result.prompt, "BASE_PROMPT")

    # ------------- hook method tests -------------
    def test_encode_messages_returns_none_by_default(self):
        """Default _encode_messages returns None (use standard encoding)."""
        result = self.chat._encode_messages([], Mock(), False, image_data=[])
        self.assertIsNone(result)


class TestProcessToolCallsWithRequiredToolChoice(unittest.TestCase):
    """Test _process_tool_calls with tool_choice='required' uses model-specific parser."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        tm = _MockTokenizerManager()
        tm.server_args.tool_call_parser = "kimi_k2"
        self.chat = OpenAIServingChat(tm, _MockTemplateManager())

    def test_required_with_parser_uses_function_call_parser(self):
        """tool_choice='required' should use FunctionCallParser when tool_call_parser is set."""
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"location":"Tokyo"}'
            call_info.tool_index = 0

            parser_instance = ParserMock.return_value
            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            tool_calls, text, fr = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...<|tool_calls_section_end|>",
                tools=tools,
                finish_reason=finish_reason,
                tool_choice="required",
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(fr["type"], "tool_calls")

    def test_empty_parser_result_is_not_reported_as_tool_call(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value
            parser_instance.has_tool_call.return_value = True
            parser_instance.detector.supports_structural_tag.return_value = True
            parser_instance.parse_non_stream.return_value = ("Visible prefix.", [])

            finish_reason = {"type": "stop", "matched": None}
            tools = [{"type": "function", "function": {"name": "get_weather"}}]

            tool_calls, text, fr = self.chat._process_tool_calls(
                text="<|malformed_tool_call|>",
                tools=tools,
                finish_reason=finish_reason,
                tool_choice="required",
            )

            self.assertIsNone(tool_calls)
            self.assertEqual(text, "Visible prefix.")
            self.assertEqual(fr, {"type": "stop", "matched": None})

    def test_required_without_parser_falls_back_to_json(self):
        """tool_choice='required' without parser should parse as JSON array."""
        self.chat.tool_call_parser = None

        finish_reason = {"type": "stop", "matched": None}
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        tool_calls, text, fr = self.chat._process_tool_calls(
            text='[{"name":"get_weather","parameters":{"location":"Tokyo"}}]',
            tools=tools,
            finish_reason=finish_reason,
            tool_choice="required",
        )

        self.assertIsNotNone(tool_calls)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_required_without_parser_invalid_json_returns_none(self):
        """tool_choice='required' without parser and invalid JSON returns tool_calls=None."""
        self.chat.tool_call_parser = None

        finish_reason = {"type": "stop", "matched": None}
        tools = [{"type": "function", "function": {"name": "get_weather"}}]

        tool_calls, text, fr = self.chat._process_tool_calls(
            text="<|tool_calls_section_begin|>not json",
            tools=tools,
            finish_reason=finish_reason,
            tool_choice="required",
        )

        self.assertIsNone(tool_calls)


class TestNormalizeToolContent(unittest.TestCase):
    """Unit tests for normalize_tool_content()."""

    def test_multiple_text_parts_joined(self):
        result = normalize_tool_content(
            "tool",
            [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}],
        )
        self.assertEqual(result, "hello world")

    def test_non_text_part_list_preserved(self):
        content = [{"name": "func", "output": "result"}]
        result = normalize_tool_content("tool", content)
        self.assertIs(result, content)

    def test_string_content_unchanged(self):
        self.assertEqual(normalize_tool_content("tool", "hello"), "hello")

    def test_empty_list_returns_empty_string(self):
        self.assertEqual(normalize_tool_content("tool", []), "")

    def test_non_tool_role_unchanged(self):
        content = [{"type": "text", "text": "hi"}]
        result = normalize_tool_content("user", content)
        self.assertIs(result, content)

    def test_mixed_str_and_dict_parts(self):
        result = normalize_tool_content(
            "tool", ["plain", {"type": "text", "text": "rich"}]
        )
        self.assertEqual(result, "plain rich")


class TestUsageReasoningTokens(CustomTestCase):
    def test_reasoning_tokens_use_the_openai_field(self):
        """OpenAI clients read reasoning tokens from usage.completion_tokens_details."""
        from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor

        usage = UsageProcessor.calculate_token_usage(
            prompt_tokens=5, completion_tokens=7, reasoning_tokens=3
        )
        self.assertEqual(usage.completion_tokens_details.reasoning_tokens, 3)
        no_reasoning = UsageProcessor.calculate_token_usage(
            prompt_tokens=5, completion_tokens=7
        )
        self.assertIsNone(no_reasoning.completion_tokens_details)


class TestRequestChatTemplateTrustGate(CustomTestCase):
    def setUp(self):
        super().setUp()
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.chat = OpenAIServingChat(_MockTokenizerManager(), _MockTemplateManager())

    def _request(self, **kwargs):
        return ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "hi"}], **kwargs
        )

    def test_rejected_by_default(self):
        for template in ("{{ messages }}", "", None):
            with self.subTest(template=template):
                error = self.chat._validate_request(
                    self._request(chat_template_kwargs={"chat_template": template})
                )
                self.assertIn("--trust-request-chat-template", error)

    def test_allowed_when_trusted(self):
        with get_context().override_server_args(trust_request_chat_template=True):
            self.assertIsNone(
                self.chat._validate_request(
                    self._request(
                        chat_template_kwargs={"chat_template": "{{ messages }}"}
                    )
                )
            )

    def test_other_kwargs_unchanged(self):
        for kwargs in (None, {}, {"enable_thinking": False}, {"chat_templates": "x"}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(
                    self.chat._validate_request(
                        self._request(chat_template_kwargs=kwargs)
                    )
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
