"""Unit tests for the non-streaming Fallback B (post-inference unclosed
reasoning recovery) helpers and async orchestration.

We mock the tokenizer and tokenizer_manager.generate_request so no real
server / model is involved.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.recover_unclosed_reasoning import (
    _strip_redirect_owned_keys,
    build_continuation_request,
    is_recovery_eligible_request,
    is_recovery_enabled,
    maybe_recover_non_streaming,
    merge_continuation_into_ret,
    should_recover_ret_item,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.test_utils import CustomTestCase


def _basic_request(**overrides):
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


def _qwen3_parser(force=False):
    return ReasoningParser(
        model_type="qwen3", stream_reasoning=False, force_reasoning=force
    )


# ----- pure decision helpers --------------------------------------------


class TestIsRecoveryEnabled(CustomTestCase):
    def test_default_off(self):
        self.assertFalse(is_recovery_enabled(_basic_request(), False))

    def test_default_on(self):
        self.assertTrue(is_recovery_enabled(_basic_request(), True))

    def test_per_request_true_overrides_off(self):
        req = _basic_request(recover_unclosed_reasoning=True)
        self.assertTrue(is_recovery_enabled(req, False))

    def test_per_request_false_overrides_on(self):
        req = _basic_request(recover_unclosed_reasoning=False)
        self.assertFalse(is_recovery_enabled(req, True))


class TestIsRecoveryEligibleRequest(CustomTestCase):
    def test_basic_qwen3_eligible(self):
        self.assertTrue(is_recovery_eligible_request(_basic_request(), "qwen3"))

    def test_no_parser_not_eligible(self):
        self.assertFalse(is_recovery_eligible_request(_basic_request(), None))
        self.assertFalse(is_recovery_eligible_request(_basic_request(), ""))

    def test_separate_reasoning_off(self):
        req = _basic_request(separate_reasoning=False)
        self.assertFalse(is_recovery_eligible_request(req, "qwen3"))

    def test_n_gt_1_not_eligible(self):
        req = _basic_request(n=3)
        self.assertFalse(is_recovery_eligible_request(req, "qwen3"))

    def test_tools_not_eligible(self):
        req = _basic_request(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        self.assertFalse(is_recovery_eligible_request(req, "qwen3"))

    def test_continue_final_message_not_eligible(self):
        req = _basic_request(continue_final_message=True)
        self.assertFalse(is_recovery_eligible_request(req, "qwen3"))


class TestShouldRecoverRetItem(CustomTestCase):
    def _ret(self, *, text, finish_type="stop"):
        return {
            "text": text,
            "output_ids": [1, 2, 3],
            "meta_info": {"finish_reason": {"type": finish_type}},
        }

    def test_unclosed_reasoning_triggers(self):
        # `<think>x</think>` would be properly closed; here we have no
        # closing tag so reasoning is non-empty and normal_text is empty.
        item = self._ret(text="<think>partial reasoning")
        self.assertTrue(should_recover_ret_item(item, _qwen3_parser()))

    def test_closed_reasoning_with_content_does_not_trigger(self):
        item = self._ret(text="<think>r</think>real answer")
        self.assertFalse(should_recover_ret_item(item, _qwen3_parser()))

    def test_closed_reasoning_with_only_whitespace_triggers(self):
        # Aggressive Scene B: redirect succeeded but content is empty/blank.
        item = self._ret(text="<think>r</think>   \n  ")
        self.assertTrue(should_recover_ret_item(item, _qwen3_parser()))

    def test_finish_length_does_not_trigger(self):
        item = self._ret(text="<think>partial reasoning", finish_type="length")
        self.assertFalse(should_recover_ret_item(item, _qwen3_parser()))

    def test_no_reasoning_does_not_trigger(self):
        # Pure normal output, no <think> at all → reasoning_text empty.
        item = self._ret(text="hello world")
        self.assertFalse(should_recover_ret_item(item, _qwen3_parser()))


# ----- request building --------------------------------------------------


class TestStripRedirectOwnedKeys(CustomTestCase):
    def test_drops_only_redirect_keys(self):
        cp = {
            "think_end_token_id": 7,
            "think_start_token_id": 8,
            "redirect_eos_token_ids": [5],
            "force_reasoning": True,
            "prob_threshold": 0.5,
            "user_field": "keep_me",
        }
        out = _strip_redirect_owned_keys(cp)
        self.assertEqual(out, {"user_field": "keep_me"})

    def test_returns_none_when_only_redirect_keys(self):
        cp = {
            "think_end_token_id": 7,
            "redirect_eos_token_ids": [5],
        }
        self.assertIsNone(_strip_redirect_owned_keys(cp))

    def test_returns_none_for_none_or_non_dict(self):
        self.assertIsNone(_strip_redirect_owned_keys(None))
        self.assertIsNone(_strip_redirect_owned_keys("not-a-dict"))


class TestBuildContinuationRequest(CustomTestCase):
    def _adapted(self, **overrides):
        kwargs = dict(
            input_ids=[10, 11, 12],
            sampling_params={"temperature": 0.7, "custom_params": None},
            stream=False,
        )
        kwargs.update(overrides)
        return GenerateReqInput(**kwargs)

    def _ret(self, output_ids):
        return {
            "output_ids": list(output_ids),
            "text": "doesn't matter",
            "meta_info": {"finish_reason": {"type": "stop"}},
        }

    def test_input_ids_concatenation(self):
        adapted = self._adapted()
        ret = self._ret([20, 21])
        follow = build_continuation_request(
            adapted_request=adapted,
            ret_item=ret,
            think_end_token_ids=[99],
        )
        self.assertIsNotNone(follow)
        self.assertEqual(follow.input_ids, [10, 11, 12, 20, 21, 99])
        self.assertFalse(follow.stream)
        self.assertFalse(follow.return_logprob)
        # Crucially: no custom logit processor on the continuation.
        self.assertIsNone(follow.custom_logit_processor)

    def test_strips_our_redirect_custom_params(self):
        adapted = self._adapted(
            sampling_params={
                "temperature": 0.7,
                "custom_params": {
                    "think_end_token_id": 99,
                    "redirect_eos_token_ids": [5],
                    "user_field": "keep_me",
                },
            }
        )
        ret = self._ret([20])
        follow = build_continuation_request(
            adapted_request=adapted, ret_item=ret, think_end_token_ids=[99]
        )
        self.assertIsNotNone(follow)
        # follow.sampling_params is normalized into a list of len-1 by
        # GenerateReqInput.normalize_batch_and_arguments(), so peek into it.
        sp = follow.sampling_params
        if isinstance(sp, list):
            sp = sp[0]
        self.assertEqual(sp["custom_params"], {"user_field": "keep_me"})

    def test_text_mode_returns_none(self):
        adapted = GenerateReqInput(
            text="hello",
            sampling_params={"temperature": 0.7},
        )
        ret = self._ret([20])
        follow = build_continuation_request(
            adapted_request=adapted, ret_item=ret, think_end_token_ids=[99]
        )
        self.assertIsNone(follow)

    def test_batched_input_returns_none(self):
        adapted = GenerateReqInput(
            input_ids=[[10, 11], [12, 13]],
            sampling_params=[
                {"temperature": 0.7},
                {"temperature": 0.7},
            ],
        )
        ret = self._ret([20])
        follow = build_continuation_request(
            adapted_request=adapted, ret_item=ret, think_end_token_ids=[99]
        )
        self.assertIsNone(follow)


# ----- merge -------------------------------------------------------------


class TestMergeContinuationIntoRet(CustomTestCase):
    def test_text_and_output_ids_concatenation(self):
        original = {
            "text": "<think>partial",
            "output_ids": [1, 2],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "completion_tokens": 2,
                "cached_tokens": 0,
            },
        }
        follow = {
            "text": " continuation",
            "output_ids": [50, 51, 52],
            "meta_info": {
                "finish_reason": {"type": "length"},
                "completion_tokens": 3,
                "cached_tokens": 1,
            },
        }
        merged = merge_continuation_into_ret(
            original_ret_item=original,
            follow_ret_item=follow,
            think_end_str="</think>",
            think_end_token_ids=[99],
        )
        self.assertEqual(merged["text"], "<think>partial</think> continuation")
        self.assertEqual(merged["output_ids"], [1, 2, 99, 50, 51, 52])
        self.assertEqual(merged["meta_info"]["finish_reason"], {"type": "length"})
        # completion_tokens accumulate
        self.assertEqual(merged["meta_info"]["completion_tokens"], 5)
        self.assertEqual(merged["meta_info"]["cached_tokens"], 1)

    def test_does_not_mutate_inputs(self):
        original = {
            "text": "x",
            "output_ids": [1],
            "meta_info": {"finish_reason": {"type": "stop"}},
        }
        follow = {
            "text": "y",
            "output_ids": [2],
            "meta_info": {"finish_reason": {"type": "length"}},
        }
        merge_continuation_into_ret(
            original_ret_item=original,
            follow_ret_item=follow,
            think_end_str="|",
            think_end_token_ids=[9],
        )
        self.assertEqual(original["text"], "x")
        self.assertEqual(original["output_ids"], [1])
        self.assertEqual(
            original["meta_info"]["finish_reason"], {"type": "stop"}
        )


# ----- async orchestration ----------------------------------------------


class _FakeTokenizer:
    def __init__(self, mapping):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        if text in self.mapping:
            return list(self.mapping[text])
        return [9999, 9998]  # default multi-token


class _FakeTemplateManager:
    force_reasoning = False


class _FakeTokenizerManager:
    def __init__(self, tokenizer, server_args, follow_up_ret):
        self.tokenizer = tokenizer
        self.server_args = server_args
        self._follow_up_ret = follow_up_ret
        self.captured_follow_req = None

    def generate_request(self, follow_req, raw_request):
        self.captured_follow_req = follow_req
        ret_value = self._follow_up_ret

        async def _gen():
            yield ret_value

        return _gen()


def _fake_handler(*, server_default, follow_up_ret, parser_name="qwen3"):
    handler = SimpleNamespace()
    handler.tokenizer_manager = _FakeTokenizerManager(
        tokenizer=_FakeTokenizer({"</think>": [99]}),
        server_args=SimpleNamespace(
            auto_recover_unclosed_reasoning=server_default,
        ),
        follow_up_ret=follow_up_ret,
    )
    handler.template_manager = _FakeTemplateManager()
    handler.reasoning_parser = parser_name
    handler._get_reasoning_from_request = lambda req: None
    return handler


def _ret_unclosed():
    return {
        "text": "<think>thinking only",
        "output_ids": [1, 2, 3],
        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 3},
    }


def _ret_normal():
    return {
        "text": "<think>r</think>real answer",
        "output_ids": [1, 2, 3, 99, 4, 5],
        "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 6},
    }


class TestMaybeRecoverNonStreaming(CustomTestCase):
    def _run(self, coro):
        return asyncio.new_event_loop().run_until_complete(coro)

    def test_no_op_when_disabled(self):
        handler = _fake_handler(server_default=False, follow_up_ret={"text": "X"})
        adapted = GenerateReqInput(input_ids=[10, 11], sampling_params={})
        original = [_ret_unclosed()]
        out = self._run(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=original,
            )
        )
        # Disabled → returned as-is, no follow-up issued.
        self.assertIs(out, original)
        self.assertIsNone(handler.tokenizer_manager.captured_follow_req)

    def test_per_request_override_enables_recovery(self):
        follow = {
            "text": " here is the answer",
            "output_ids": [200, 201],
            "meta_info": {"finish_reason": {"type": "stop"}, "completion_tokens": 2},
        }
        handler = _fake_handler(server_default=False, follow_up_ret=follow)
        adapted = GenerateReqInput(input_ids=[10, 11], sampling_params={})
        out = self._run(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(recover_unclosed_reasoning=True),
                raw_request=None,
                ret_list=[_ret_unclosed()],
            )
        )
        self.assertEqual(len(out), 1)
        merged = out[0]
        # original text is "<think>thinking only", bridge "</think>", then follow text
        self.assertIn("</think>", merged["text"])
        self.assertIn("here is the answer", merged["text"])
        # completion_tokens accumulate (3 + 2 = 5)
        self.assertEqual(merged["meta_info"]["completion_tokens"], 5)
        # follow-up was issued, with input_ids carrying think_end token
        follow_req = handler.tokenizer_manager.captured_follow_req
        self.assertIsNotNone(follow_req)
        self.assertEqual(follow_req.input_ids, [10, 11, 1, 2, 3, 99])
        # And the continuation must NOT carry our redirect processor.
        self.assertIsNone(follow_req.custom_logit_processor)

    def test_skips_when_finish_is_length(self):
        handler = _fake_handler(server_default=True, follow_up_ret={"text": "X"})
        adapted = GenerateReqInput(input_ids=[10, 11], sampling_params={})
        original = [
            {
                "text": "<think>partial",
                "output_ids": [1, 2, 3],
                "meta_info": {
                    "finish_reason": {"type": "length"},
                    "completion_tokens": 3,
                },
            }
        ]
        out = self._run(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=original,
            )
        )
        self.assertEqual(out[0]["text"], "<think>partial")
        self.assertIsNone(handler.tokenizer_manager.captured_follow_req)

    def test_skips_when_already_has_content(self):
        handler = _fake_handler(server_default=True, follow_up_ret={"text": "X"})
        adapted = GenerateReqInput(input_ids=[10, 11], sampling_params={})
        out = self._run(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=[_ret_normal()],
            )
        )
        self.assertEqual(out[0]["text"], "<think>r</think>real answer")
        self.assertIsNone(handler.tokenizer_manager.captured_follow_req)

    def test_text_mode_request_is_not_recovered(self):
        handler = _fake_handler(server_default=True, follow_up_ret={"text": "X"})
        adapted = GenerateReqInput(text="prompt text", sampling_params={})
        out = self._run(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=[_ret_unclosed()],
            )
        )
        # No input_ids → recovery cannot construct a follow-up.
        self.assertEqual(out[0]["text"], "<think>thinking only")
        self.assertIsNone(handler.tokenizer_manager.captured_follow_req)


if __name__ == "__main__":
    unittest.main()
