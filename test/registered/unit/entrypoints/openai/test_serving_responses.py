"""
Unit-tests for OpenAIServingResponses.

Pure mock-based; no model weights, no server, CPU only.
Run with:
    python test/registered/unit/entrypoints/openai/test_serving_responses.py -v
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import unittest
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock

import orjson
from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)

from sglang.srt.entrypoints.context import SimpleContext
from sglang.srt.entrypoints.openai.protocol import (
    RequestResponseMetadata,
    ResponsesRequest,
    ResponsesResponse,
)
from sglang.srt.entrypoints.openai.serving_responses import OpenAIServingResponses
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ----------------------------- mocks ------------------------------------


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingResponses.__init__."""

    def __init__(self, model_type: str = "llama"):
        self.model_config = Mock(is_multimodal=False)
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        mock_hf_config.model_type = model_type  # not "gpt_oss" -> use_harmony=False
        self.model_config.hf_config = mock_hf_config

        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
            stream_response_default_include_usage=False,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        self.abort_request = Mock()
        self.create_abort_task = Mock()


class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None
        self.reasoning_config = None
        self.force_reasoning = False


# --------------------------- helpers -----------------------------------


def _make_assistant_output(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_test",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(
                type="output_text", text=text, annotations=[], logprobs=None
            )
        ],
    )


def _make_reasoning_output(text: str) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id="rs_test",
        type="reasoning",
        summary=[],
        content=[ResponseReasoningTextContent(type="reasoning_text", text=text)],
        status=None,
    )


def _run(coro):
    loop = get_or_create_event_loop()
    return loop.run_until_complete(coro)


def _decode_error(orjson_response) -> dict:
    """Extract dict from ORJSONResponse.body."""
    return orjson.loads(orjson_response.body)


class _ServingResponsesTestBase(unittest.TestCase):
    """Shared setUp: a fresh OpenAIServingResponses instance per test."""

    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.serving = OpenAIServingResponses(self.tm, self.template_manager)


# ============================================================
#   _construct_input_messages (non-Harmony path)
# ============================================================


class TestConstructInputMessages(_ServingResponsesTestBase):
    # ---------- single-turn ----------

    def test_single_turn_with_instructions(self):
        req = ResponsesRequest(
            model="x", input="What's 2+2?", instructions="Be concise."
        )
        msgs = self.serving._construct_input_messages(req, prev_response=None)
        self.assertEqual(
            msgs,
            [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What's 2+2?"},
            ],
        )

    def test_single_turn_no_instructions(self):
        req = ResponsesRequest(model="x", input="hi")
        msgs = self.serving._construct_input_messages(req, prev_response=None)
        self.assertEqual(msgs, [{"role": "user", "content": "hi"}])

    def test_input_is_list_extends_directly(self):
        # When input is a list of role-typed items, it should be appended as-is.
        custom_items = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        req = ResponsesRequest(model="x", input=custom_items)
        msgs = self.serving._construct_input_messages(req, prev_response=None)
        self.assertEqual(msgs, custom_items)

    # ---------- multi-turn (PR fix) ----------

    def test_multi_turn_assistant_role_and_text(self):
        """Regression: previous output must be inserted as an `assistant`
        message containing the original output text -- NOT a `system` message
        echoing the new request's instructions.
        """
        prev_id = "resp_prev1"
        # The store keeps the *user side* of the previous turn (and possibly a
        # system msg from that turn -- which must be stripped per OpenAI spec).
        self.serving.msg_store[prev_id] = [
            {"role": "system", "content": "OLD instructions"},
            {"role": "user", "content": "The secret word is BANANA."},
        ]
        prev_resp = Mock(spec=ResponsesResponse)
        prev_resp.id = prev_id
        prev_resp.output = [_make_assistant_output("Got it, BANANA.")]

        req = ResponsesRequest(
            model="x",
            input="What was the secret?",
            instructions="Reply in one word.",
            previous_response_id=prev_id,
        )

        msgs = self.serving._construct_input_messages(req, prev_response=prev_resp)

        roles = [m["role"] for m in msgs]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

        # Current-turn instructions used (only once).
        self.assertEqual(msgs[0], {"role": "system", "content": "Reply in one word."})
        # Previous user kept.
        self.assertEqual(
            msgs[1], {"role": "user", "content": "The secret word is BANANA."}
        )
        # Previous output is `assistant` with original text -- guards Bug 2.
        self.assertEqual(msgs[2], {"role": "assistant", "content": "Got it, BANANA."})
        # Old "system / request.instructions" bug must not re-appear.
        self.assertNotEqual(msgs[2]["content"], "Reply in one word.")
        self.assertEqual(sum(1 for m in msgs if m["role"] == "system"), 1)
        # New input appended last.
        self.assertEqual(msgs[3], {"role": "user", "content": "What was the secret?"})

    def test_multi_turn_strips_prev_system(self):
        """Previous turn's system messages must NOT be carried over."""
        prev_id = "resp_prev2"
        self.serving.msg_store[prev_id] = [
            {"role": "system", "content": "secret-key=A"},
            {"role": "system", "content": "secret-key=B"},
            {"role": "user", "content": "ping"},
        ]
        prev_resp = Mock(spec=ResponsesResponse)
        prev_resp.id = prev_id
        prev_resp.output = [_make_assistant_output("pong")]

        req = ResponsesRequest(model="x", input="next", previous_response_id=prev_id)
        msgs = self.serving._construct_input_messages(req, prev_response=prev_resp)

        # No system-role messages should leak from prev turn (req has no
        # instructions, so total system count must be 0).
        self.assertEqual([m for m in msgs if m["role"] == "system"], [])
        # And the leaked system contents must not appear anywhere.
        flat = " ".join(str(m.get("content", "")) for m in msgs)
        self.assertNotIn("secret-key=A", flat)
        self.assertNotIn("secret-key=B", flat)

    def test_multi_turn_skips_reasoning_item(self):
        """ResponseReasoningItem from the previous response must be skipped."""
        prev_id = "resp_prev3"
        self.serving.msg_store[prev_id] = [{"role": "user", "content": "Q"}]
        prev_resp = Mock(spec=ResponsesResponse)
        prev_resp.id = prev_id
        prev_resp.output = [
            _make_reasoning_output("internal CoT must NOT leak"),
            _make_assistant_output("final answer"),
        ]

        req = ResponsesRequest(model="x", input="Q2", previous_response_id=prev_id)
        msgs = self.serving._construct_input_messages(req, prev_response=prev_resp)

        flat = " ".join(str(m.get("content", "")) for m in msgs)
        self.assertNotIn("internal CoT", flat)
        # The assistant message from the non-reasoning output must still be there.
        self.assertIn(
            {"role": "assistant", "content": "final answer"},
            msgs,
        )

    def test_multi_turn_concatenates_multi_part_text(self):
        """When an output_item has several text parts they should be joined
        into one assistant message.
        """
        prev_id = "resp_prev4"
        self.serving.msg_store[prev_id] = []
        out = ResponseOutputMessage(
            id="msg_multi",
            type="message",
            role="assistant",
            status="completed",
            content=[
                ResponseOutputText(
                    type="output_text", text="Hello ", annotations=[], logprobs=None
                ),
                ResponseOutputText(
                    type="output_text", text="world!", annotations=[], logprobs=None
                ),
            ],
        )
        prev_resp = Mock(spec=ResponsesResponse)
        prev_resp.id = prev_id
        prev_resp.output = [out]

        req = ResponsesRequest(model="x", input="next", previous_response_id=prev_id)
        msgs = self.serving._construct_input_messages(req, prev_response=prev_resp)

        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        self.assertEqual(len(assistant_msgs), 1)
        self.assertEqual(assistant_msgs[0]["content"], "Hello world!")

    def test_multi_turn_skips_output_without_text(self):
        """Output items with no text-bearing content must NOT produce an
        empty assistant message.
        """
        prev_id = "resp_prev5"
        self.serving.msg_store[prev_id] = [{"role": "user", "content": "Q"}]
        out = Mock()
        out.content = [Mock(spec=[])]  # no `text` attribute
        prev_resp = Mock(spec=ResponsesResponse)
        prev_resp.id = prev_id
        prev_resp.output = [out]

        req = ResponsesRequest(model="x", input="Q2", previous_response_id=prev_id)
        msgs = self.serving._construct_input_messages(req, prev_response=prev_resp)

        # No empty assistant should be inserted.
        for m in msgs:
            if m["role"] == "assistant":
                self.assertTrue(m["content"])


# ============================================================
#   responses_full_generator: usage extraction (PR fix)
# ============================================================


class TestUsageExtraction(_ServingResponsesTestBase):
    async def _drive_full_generator(self, final_res_dict):
        """Run responses_full_generator with a SimpleContext whose last_output
        is the given dict. Returns the ResponsesResponse (or ORJSONResponse).
        """
        ctx = SimpleContext()
        ctx.last_output = final_res_dict

        async def _empty_gen():
            return
            yield  # pragma: no cover - make this an async-generator

        req = ResponsesRequest(model="x", input="hi")
        sampling_params = req.to_sampling_params(
            default_max_tokens=128, default_params={}
        )
        metadata = RequestResponseMetadata(request_id=req.request_id)
        return await self.serving.responses_full_generator(
            request=req,
            sampling_params=sampling_params,
            result_generator=_empty_gen(),
            context=ctx,
            model_name="x",
            tokenizer=self.tm.tokenizer,
            request_metadata=metadata,
        )

    def test_usage_populated_from_dict_meta_info(self):
        """Bug 1 regression: final_res is a dict, usage must be read via
        dict-key access ('meta_info' in final_res), not hasattr(...).
        """
        final_res = {
            "text": "hello",
            "meta_info": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 2,
                "reasoning_tokens": 3,
            },
        }
        resp = _run(self._drive_full_generator(final_res))
        self.assertIsInstance(resp, ResponsesResponse)
        self.assertIsNotNone(resp.usage)
        self.assertEqual(resp.usage.prompt_tokens, 10)
        self.assertEqual(resp.usage.completion_tokens, 5)
        # total_tokens must reflect prompt + completion.
        self.assertEqual(resp.usage.total_tokens, 15)
        # And the regression guard: NOT zero.
        self.assertNotEqual(resp.usage.total_tokens, 0)

    def test_usage_dict_with_missing_fields_defaults_to_zero(self):
        final_res = {"text": "hello", "meta_info": {"prompt_tokens": 7}}
        resp = _run(self._drive_full_generator(final_res))
        self.assertIsInstance(resp, ResponsesResponse)
        self.assertEqual(resp.usage.prompt_tokens, 7)
        self.assertEqual(resp.usage.completion_tokens, 0)
        self.assertEqual(resp.usage.total_tokens, 7)

    def test_usage_no_meta_info_falls_back_to_zero(self):
        # Plain dict, no meta_info, no fallback fields -> all zeros.
        final_res = {"text": "hello"}
        resp = _run(self._drive_full_generator(final_res))
        self.assertIsInstance(resp, ResponsesResponse)
        self.assertEqual(resp.usage.prompt_tokens, 0)
        self.assertEqual(resp.usage.completion_tokens, 0)
        self.assertEqual(resp.usage.total_tokens, 0)


# ============================================================
#   retrieve_responses / cancel_responses
# ============================================================


class TestRetrieveAndCancel(_ServingResponsesTestBase):
    # ---------- retrieve_responses ----------

    def test_retrieve_invalid_id_returns_400(self):
        out = _run(self.serving.retrieve_responses("bogus_xyz"))
        self.assertEqual(out.status_code, 400)
        body = _decode_error(out)
        self.assertIn("Invalid", body["error"]["message"])
        self.assertEqual(body["error"]["param"], "response_id")

    def test_retrieve_not_found_returns_404(self):
        out = _run(self.serving.retrieve_responses("resp_doesnotexist"))
        self.assertEqual(out.status_code, HTTPStatus.NOT_FOUND)
        body = _decode_error(out)
        self.assertIn("not found", body["error"]["message"])

    def test_retrieve_hit_returns_stored_object(self):
        stored = Mock(spec=ResponsesResponse)
        self.serving.response_store["resp_abc"] = stored
        out = _run(self.serving.retrieve_responses("resp_abc"))
        self.assertIs(out, stored)

    # ---------- cancel_responses ----------

    def test_cancel_invalid_id_returns_400(self):
        out = _run(self.serving.cancel_responses("nope"))
        self.assertEqual(out.status_code, 400)

    def test_cancel_not_found_returns_404(self):
        out = _run(self.serving.cancel_responses("resp_unknown"))
        self.assertEqual(out.status_code, HTTPStatus.NOT_FOUND)

    def test_cancel_completed_rejected(self):
        stored = Mock(spec=ResponsesResponse)
        stored.status = "completed"
        self.serving.response_store["resp_done"] = stored
        out = _run(self.serving.cancel_responses("resp_done"))
        self.assertEqual(out.status_code, 400)
        body = _decode_error(out)
        self.assertIn("Cannot cancel a synchronous response", body["error"]["message"])
        # Status must NOT have been mutated.
        self.assertEqual(stored.status, "completed")
        self.tm.abort_request.assert_not_called()

    def test_cancel_in_progress_marks_cancelled_and_aborts(self):
        stored = Mock(spec=ResponsesResponse)
        stored.status = "in_progress"
        self.serving.response_store["resp_run"] = stored
        out = _run(self.serving.cancel_responses("resp_run"))
        self.assertIs(out, stored)
        self.assertEqual(stored.status, "cancelled")
        self.tm.abort_request.assert_called_once_with(rid="resp_run")


# ============================================================
#   Misc helpers
# ============================================================


class TestMisc(_ServingResponsesTestBase):
    def test_request_id_prefix_is_resp(self):
        self.assertEqual(self.serving._request_id_prefix(), "resp_")

    def test_create_error_response_payload_shape(self):
        out = self.serving.create_error_response(
            "boom", err_type="invalid_request_error", status_code=422, param="input"
        )
        self.assertEqual(out.status_code, 422)
        body = _decode_error(out)
        self.assertEqual(body["error"]["message"], "boom")
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["param"], "input")
        self.assertEqual(body["error"]["code"], 422)


if __name__ == "__main__":
    unittest.main()
