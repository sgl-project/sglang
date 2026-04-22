"""Integration-level tests for the unclosed-reasoning recovery feature.

These tests stitch together the registry, the redirect logit processor, the
parser dedupe, and the Fallback B orchestration without bringing up a real
SGLang server.

Focus areas:
- Recursion protection (continuation request is never recovered again).
- Multi-token think_end models: Path-1 is correctly disabled at startup but
  Fallback B still works because it only needs `tokenizer.encode(...)`.
- Path-1 + Fallback B coexist: redirect succeeded but model still produced
  empty content → Fallback B kicks in.
- Tool-call requests are skipped end-to-end.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="stage-a-test-cpu")

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.recover_unclosed_reasoning import (
    _RECOVERY_SENTINEL_KEY,
    build_continuation_request,
    is_recovery_eligible_request,
    maybe_recover_non_streaming,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.parser.reasoning_redirect_registry import build_redirect_config
from sglang.test.test_utils import CustomTestCase


def _basic_request(**overrides):
    base = {
        "model": "test",
        "messages": [{"role": "user", "content": "hi"}],
    }
    base.update(overrides)
    return ChatCompletionRequest(**base)


# ----- Recursion protection ---------------------------------------------


class TestRecursionProtection(CustomTestCase):
    def test_continuation_request_carries_sentinel(self):
        adapted = GenerateReqInput(input_ids=[1, 2], sampling_params={})
        ret = {
            "output_ids": [10, 11],
            "text": "<think>x",
            "meta_info": {"finish_reason": {"type": "stop"}},
        }
        follow = build_continuation_request(
            adapted_request=adapted, ret_item=ret, think_end_token_ids=[99]
        )
        self.assertIsNotNone(follow)
        sp = follow.sampling_params
        if isinstance(sp, list):
            sp = sp[0]
        self.assertTrue(sp["custom_params"][_RECOVERY_SENTINEL_KEY])

    def test_recover_skipped_when_sentinel_present(self):
        # Simulate a "continuation" request being passed back into recovery.
        adapted = GenerateReqInput(
            input_ids=[1, 2, 99],
            sampling_params={
                "custom_params": {_RECOVERY_SENTINEL_KEY: True},
            },
        )

        class _FakeTok:
            eos_token_id = 0

            def encode(self, t, add_special_tokens=False):
                return [99]

        handler = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                tokenizer=_FakeTok(),
                server_args=SimpleNamespace(
                    auto_recover_unclosed_reasoning=True,
                ),
                generate_request=lambda *a, **kw: (_ for _ in ()),
            ),
            template_manager=SimpleNamespace(force_reasoning=False),
            reasoning_parser="qwen3",
            _get_reasoning_from_request=lambda r: None,
        )

        called = {"count": 0}

        def gen(req, raw):
            called["count"] += 1

            async def _g():
                yield {
                    "text": "x",
                    "output_ids": [1],
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }

            return _g()

        ret = [
            {
                "text": "<think>still empty",
                "output_ids": [1, 2],
                "meta_info": {"finish_reason": {"type": "stop"}},
            }
        ]
        out = asyncio.new_event_loop().run_until_complete(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(recover_unclosed_reasoning=True),
                raw_request=None,
                ret_list=ret,
                generate_request_fn=gen,
            )
        )
        # Sentinel present → recovery is a no-op, no follow-up issued.
        self.assertEqual(called["count"], 0)
        self.assertEqual(out[0]["text"], "<think>still empty")


# ----- Multi-token think_end downgrade ----------------------------------


class _FakeMultiTokenTokenizer:
    """`</think>` encodes as MULTIPLE tokens (e.g. 2 ids). Single `<|im_end|>`."""

    eos_token_id = 200

    def encode(self, text, add_special_tokens=False):
        if text == "</think>":
            return [123, 124]  # multi-token!
        if text == "<|im_end|>":
            return [200]
        if text == "<think>":
            return [125, 126]
        return [9999, 9998]


class TestMultiTokenDowngrade(CustomTestCase):
    def test_path1_registry_returns_none_for_multi_token(self):
        cfg = build_redirect_config("qwen3", _FakeMultiTokenTokenizer())
        self.assertIsNone(cfg)

    def test_fallback_b_still_works_for_multi_token(self):
        # Even when Path-1 is unavailable, Fallback B can still close the
        # reasoning section because it only requires `encode(</think>)` to
        # return a non-empty list.
        adapted = GenerateReqInput(input_ids=[1, 2, 3], sampling_params={})

        handler = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                tokenizer=_FakeMultiTokenTokenizer(),
                server_args=SimpleNamespace(
                    auto_recover_unclosed_reasoning=True,
                ),
            ),
            template_manager=SimpleNamespace(force_reasoning=False),
            reasoning_parser="qwen3",
            _get_reasoning_from_request=lambda r: None,
        )

        captured_follow = {}

        def gen(req, raw):
            captured_follow["req"] = req

            async def _g():
                yield {
                    "text": "and the answer is 42",
                    "output_ids": [50, 51, 52, 53, 54],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 5,
                    },
                }

            return _g()

        ret = [
            {
                "text": "<think>thinking only no closer",
                "output_ids": [10, 11, 12],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "completion_tokens": 3,
                },
            }
        ]
        out = asyncio.new_event_loop().run_until_complete(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=ret,
                generate_request_fn=gen,
            )
        )
        merged = out[0]
        self.assertIn("</think>", merged["text"])
        self.assertIn("and the answer is 42", merged["text"])
        # Continuation input_ids carry the BOTH think_end token ids.
        follow_req = captured_follow["req"]
        self.assertEqual(follow_req.input_ids, [1, 2, 3, 10, 11, 12, 123, 124])
        # output_ids merged: original + think_end_ids + follow
        self.assertEqual(
            merged["output_ids"], [10, 11, 12, 123, 124, 50, 51, 52, 53, 54]
        )


# ----- Path-1 + Fallback B coexistence ----------------------------------


class TestPath1AndFallbackBCoexist(CustomTestCase):
    """Scene B in the design doc: redirect successfully wrote `</think>` into
    output_ids but the model immediately produced an EOS token. The parsed
    reasoning is non-empty and content is empty/whitespace, so Fallback B must
    still trigger to actually generate the answer."""

    def test_redirect_produced_think_end_then_eos_still_recovered(self):
        adapted = GenerateReqInput(input_ids=[1, 2], sampling_params={})

        class _FakeTok:
            eos_token_id = 200

            def encode(self, text, add_special_tokens=False):
                if text == "</think>":
                    return [99]
                return [9999]

        handler = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                tokenizer=_FakeTok(),
                server_args=SimpleNamespace(
                    auto_recover_unclosed_reasoning=True,
                ),
            ),
            template_manager=SimpleNamespace(force_reasoning=False),
            reasoning_parser="qwen3",
            _get_reasoning_from_request=lambda r: None,
        )

        def gen(req, raw):
            async def _g():
                yield {
                    "text": "the actual content",
                    "output_ids": [50, 51],
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "completion_tokens": 2,
                    },
                }

            return _g()

        # Path-1 succeeded: text already contains `</think>` but content after
        # it is empty (model emitted EOS right after the redirected think_end).
        ret = [
            {
                "text": "<think>thinking</think>",
                "output_ids": [10, 11, 99],
                "meta_info": {
                    "finish_reason": {"type": "stop"},
                    "completion_tokens": 3,
                },
            }
        ]
        out = asyncio.new_event_loop().run_until_complete(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(),
                raw_request=None,
                ret_list=ret,
                generate_request_fn=gen,
            )
        )
        merged = out[0]
        # Note we still bridge a `</think>` between the two — the parser
        # dedupe (Stage 2) will swallow the leading duplicate when the client
        # parses the final text.
        self.assertIn("the actual content", merged["text"])


# ----- Tool-call eligibility / runtime judgement end-to-end -------------


class TestToolCallSkipsRecovery(CustomTestCase):
    def _tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def test_tool_choice_required_is_hard_excluded(self):
        # tool_choice="required" is the only hard-exclude — the model is
        # contractually obliged to emit a tool call.
        req = _basic_request(tools=self._tools(), tool_choice="required")
        self.assertFalse(is_recovery_eligible_request(req, "qwen3"))

    def test_tools_with_auto_choice_is_eligible(self):
        # Tools offered but tool_choice="auto" → runtime judgement,
        # gate passes at the request-eligibility stage.
        req = _basic_request(tools=self._tools(), tool_choice="auto")
        self.assertTrue(is_recovery_eligible_request(req, "qwen3"))

    def _fake_handler(self, tool_call_parser=None):
        class _FakeTok:
            eos_token_id = 200

            def encode(self, *a, **kw):
                return [99]

        return SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                tokenizer=_FakeTok(),
                server_args=SimpleNamespace(
                    auto_recover_unclosed_reasoning=True,
                ),
            ),
            template_manager=SimpleNamespace(force_reasoning=False),
            reasoning_parser="qwen3",
            tool_call_parser=tool_call_parser,
            _get_reasoning_from_request=lambda r: None,
        )

    def test_recovery_fires_when_tools_offered_but_no_tool_call_emitted(self):
        # Tools offered, model stops with unclosed <think> and no tool-call
        # sigil in the text → runtime check passes → recovery continues.
        adapted = GenerateReqInput(input_ids=[1, 2], sampling_params={})
        handler = self._fake_handler(tool_call_parser="qwen25")

        called = {"count": 0}

        def gen(req, raw):
            called["count"] += 1

            async def _g():
                yield {"text": " answer", "output_ids": [50], "meta_info": {}}

            return _g()

        ret = [
            {
                "text": "<think>thinking",
                "output_ids": [1, 2],
                "meta_info": {"finish_reason": {"type": "stop"}},
            }
        ]
        out = asyncio.new_event_loop().run_until_complete(
            maybe_recover_non_streaming(
                handler=handler,
                adapted_request=adapted,
                request=_basic_request(
                    tools=self._tools(), tool_choice="auto"
                ),
                raw_request=None,
                ret_list=ret,
                generate_request_fn=gen,
            )
        )
        # Recovery was attempted (follow-up generation was called at least once).
        self.assertGreaterEqual(called["count"], 1)
        self.assertIn("answer", out[0]["text"])

    def test_recovery_skipped_when_tool_call_actually_emitted(self):
        # Tools offered AND the generated text already contains a tool call
        # (mocked via FunctionCallParser.has_tool_call returning True).
        # Recovery must NOT trigger.
        import sglang.srt.function_call.function_call_parser as _fcp_mod

        class _FakeFCP:
            def __init__(self, *a, **kw):
                pass

            def has_tool_call(self, text):
                return True

        orig = _fcp_mod.FunctionCallParser
        _fcp_mod.FunctionCallParser = _FakeFCP
        try:
            adapted = GenerateReqInput(input_ids=[1, 2], sampling_params={})
            handler = self._fake_handler(tool_call_parser="qwen25")

            called = {"count": 0}

            def gen(req, raw):
                called["count"] += 1

                async def _g():
                    yield {"text": "x", "output_ids": [1], "meta_info": {}}

                return _g()

            ret = [
                {
                    "text": "<think>thinking",
                    "output_ids": [1, 2],
                    "meta_info": {"finish_reason": {"type": "stop"}},
                }
            ]
            out = asyncio.new_event_loop().run_until_complete(
                maybe_recover_non_streaming(
                    handler=handler,
                    adapted_request=adapted,
                    request=_basic_request(
                        tools=self._tools(), tool_choice="auto"
                    ),
                    raw_request=None,
                    ret_list=ret,
                    generate_request_fn=gen,
                )
            )
            self.assertEqual(called["count"], 0)
            self.assertEqual(out[0]["text"], "<think>thinking")
        finally:
            _fcp_mod.FunctionCallParser = orig


if __name__ == "__main__":
    unittest.main()
