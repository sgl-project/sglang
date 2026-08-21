"""Unit tests for dropping empty assistant turns before mistral_common — no model loading.

mistral_common rejects an assistant message that carries neither content nor tool
calls, so an OpenAI-compatible request that works on every other model would fail.
patch_mistral_common_tokenizer removes those turns; these tests drive that wiring
through a stub tokenizer and assert on what mistral_common would have received.
"""

import unittest

from sglang.srt.utils.hf_transformers.mistral_utils import (
    patch_mistral_common_tokenizer,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _user(text):
    return {"role": "user", "content": text}


def _assistant(**fields):
    return {"role": "assistant", **fields}


class _MistralCommonStub:
    """Stands in for MistralCommonBackend; the class name gates the patch."""

    def __init__(self):
        self.seen = None
        self.chat_template = "x"

    def apply_chat_template(self, messages, **kwargs):
        self.seen = messages
        return []

    def add_special_tokens(self, *args, **kwargs):
        return 0

    def convert_tokens_to_ids(self, value):
        return 0

    def decode(self, *args, **kwargs):
        return ""

    def batch_decode(self, *args, **kwargs):
        return []


class TestDropEmptyAssistantMessages(unittest.TestCase):
    def _roles_passed_through(self, messages):
        tokenizer = patch_mistral_common_tokenizer(_MistralCommonStub())
        tokenizer.apply_chat_template(messages)
        return [msg["role"] for msg in tokenizer.seen]

    def test_empty_assistant_turn_is_dropped(self):
        """The reported failing case: user / assistant("") / user."""
        roles = self._roles_passed_through(
            [_user("Say hello."), _assistant(content=""), _user("Continue.")]
        )

        self.assertEqual(roles, ["user", "user"])

    def test_whitespace_and_none_content_are_dropped(self):
        for content in ("   ", None):
            with self.subTest(content=content):
                roles = self._roles_passed_through(
                    [_user("a"), _assistant(content=content), _user("b")]
                )

                self.assertEqual(roles, ["user", "user"])

    def test_assistant_with_tool_calls_is_kept(self):
        """An empty-content turn still matters when it carries tool calls."""
        roles = self._roles_passed_through(
            [
                _user("a"),
                _assistant(content="", tool_calls=[{"id": "call_1"}]),
                _user("b"),
            ]
        )

        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_normal_and_multimodal_assistant_are_kept(self):
        for content in ("hi", [{"type": "text", "text": "hi"}]):
            with self.subTest(content=content):
                roles = self._roles_passed_through(
                    [_user("a"), _assistant(content=content), _user("b")]
                )

                self.assertEqual(roles, ["user", "assistant", "user"])


if __name__ == "__main__":
    unittest.main()
