"""Unit test for DetokenizerManager.trim_matched_stop.

Under speculative decoding the output handed to trim_matched_stop may carry
content after the matched stop. With no_stop_trim the stop string must be kept
but the trailing over-generation still dropped (`output[:end]`); without it the
stop is removed (`output[:pos]`). Pure CPU: calls the method with a stub self,
so no DetokenizerManager.__init__ / IPC / tokenizer."""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.detokenizer_manager import DetokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

GPT_OSS_CALL_TOKEN = 200012
HARMONY_CALL_TOKENS = {"llm-jp-4": 13, "gpt-oss": 200012}


def _trim(output, matched, no_stop_trim, *, harmony_call_token_id=None, tool_close=()):
    stub = SimpleNamespace(
        harmony_call_token_id=harmony_call_token_id,
        tool_close_token_ids=frozenset(tool_close),
    )
    finished_reason = None if matched is None else {"matched": matched}
    return DetokenizerManager.trim_matched_stop(
        stub, output, finished_reason, no_stop_trim
    )


class TestTrimMatchedStop(CustomTestCase):
    def test_no_matched_returns_output(self):
        stub = SimpleNamespace(harmony_call_token_id=None)
        self.assertEqual(
            DetokenizerManager.trim_matched_stop(stub, "abc", {}, False), "abc"
        )

    # --- stop string ---
    def test_str_trim_removes_stop(self):
        # no_stop_trim=False: drop the stop string and anything after it.
        self.assertEqual(_trim("ans\n\nQuestion: A", "Question", False), "ans\n\n")

    def test_str_no_trim_keeps_stop_but_drops_trailing(self):
        # no_stop_trim=True: keep through the stop, drop the over-generated tail.
        self.assertEqual(
            _trim("ans\n\nQuestion: A", "Question", True), "ans\n\nQuestion"
        )

    # --- stop token ---
    def test_token_trim_drops_last(self):
        self.assertEqual(_trim([1, 2, 3], 3, False), [1, 2])

    def test_token_no_trim_keeps_all(self):
        self.assertEqual(_trim([1, 2, 3], 3, True), [1, 2, 3])

    def test_token_zero_is_a_stop_token(self):
        # Token id 0 is falsy; the guard checks `matched is None`, not truthiness.
        self.assertEqual(_trim([123, 0], 0, False), [123])
        self.assertEqual(_trim([123, 0], 0, True), [123, 0])

    def test_matched_none_inside_a_finished_reason_returns_output(self):
        stub = SimpleNamespace(harmony_call_token_id=None)
        trimmed = DetokenizerManager.trim_matched_stop(
            stub, [123, 0], {"type": "stop", "matched": None}, False
        )
        self.assertEqual(trimmed, [123, 0])

    def test_token_gpt_oss_call_kept(self):
        # gpt-oss tool-call token is also an eos; keep it even when trimming.
        self.assertEqual(
            _trim(
                [1, 2, GPT_OSS_CALL_TOKEN],
                GPT_OSS_CALL_TOKEN,
                False,
                harmony_call_token_id=GPT_OSS_CALL_TOKEN,
            ),
            [1, 2, GPT_OSS_CALL_TOKEN],
        )

    def test_token_tool_close_kept(self):
        self.assertEqual(_trim([1, 2, 3], 3, False, tool_close=[3]), [1, 2, 3])
        self.assertEqual(_trim([1, 2, 4], 4, False, tool_close=[3]), [1, 2])


class TestResolveHarmonyCallTokenId(CustomTestCase):
    def test_resolves_call_token_with_selected_parser(self):
        for model, call_token_id in HARMONY_CALL_TOKENS.items():
            with self.subTest(model=model):
                tokenizer = SimpleNamespace(
                    unk_token_id=0,
                    convert_tokens_to_ids=lambda token, call_token_id=call_token_id: (
                        call_token_id if token == "<|call|>" else 0
                    ),
                )
                manager = SimpleNamespace(tokenizer=tokenizer)

                token_id = DetokenizerManager._resolve_harmony_call_token_id(
                    manager, "gpt-oss"
                )

                self.assertEqual(token_id, call_token_id)

    def test_only_selected_parser_preserves_resolved_call_token(self):
        """A non-GPT-OSS token layout must preserve its tool stop, while the same
        token remains an ordinary stop when that parser is not selected."""
        for parser, expected in (("gpt-oss", [1, 13]), ("another-parser", [1])):
            manager = SimpleNamespace(
                tokenizer=SimpleNamespace(
                    unk_token_id=0, convert_tokens_to_ids=lambda token: 13
                )
            )
            resolved = DetokenizerManager._resolve_harmony_call_token_id(
                manager, parser
            )
            self.assertEqual(
                _trim([1, 13], 13, False, harmony_call_token_id=resolved), expected
            )

    def test_unknown_or_invalid_token_is_not_preserved(self):
        for value in (None, -1, 0, "13"):
            with self.subTest(value=value):
                manager = SimpleNamespace(
                    tokenizer=SimpleNamespace(
                        unk_token_id=0,
                        convert_tokens_to_ids=lambda token, value=value: value,
                    )
                )
                self.assertIsNone(
                    DetokenizerManager._resolve_harmony_call_token_id(
                        manager, "gpt-oss"
                    )
                )

    def test_tokenizer_without_token_conversion_returns_none(self):
        manager = SimpleNamespace(tokenizer=SimpleNamespace())

        token_id = DetokenizerManager._resolve_harmony_call_token_id(manager, "gpt-oss")

        self.assertIsNone(token_id)


if __name__ == "__main__":
    unittest.main()
