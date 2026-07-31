"""Unit tests for srt/parser/sanitize_content.py"""

import hashlib
import unittest
from unittest.mock import MagicMock

from sglang.srt.parser.sanitize_content import (
    _SANITIZED_CONTENT_PLACEHOLDER_RE,
    _make_sanitized_content_placeholder,
    _restore_sanitized_placeholders,
    _should_use_sanitized_chat_template,
    _split_sanitized_rendered_chat,
    safe_apply_chat_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


def _make_special_token_test_tokenizer():
    """Create a mock tokenizer that treats <|im_start|> and <|im_end|> as
    special tokens."""

    class _SpecialTokenTestTokenizer:
        special_tokens_map = {}
        chat_template = None

        def get_chat_template(self, chat_template, tools=None):
            return chat_template

        def encode(self, text, add_special_tokens=False):
            token_ids = []
            i = 0
            special_tokens = {"<|im_start|>": 32000, "<|im_end|>": 32001}
            while i < len(text):
                for special_token, token_id in special_tokens.items():
                    if text.startswith(special_token, i):
                        token_ids.append(token_id)
                        i += len(special_token)
                        break
                else:
                    token_ids.append(ord(text[i]))
                    i += 1
            return token_ids

        def tokenize(self, text, split_special_tokens=False):
            assert split_special_tokens
            return list(text)

        def convert_tokens_to_ids(self, tokens):
            return [ord(token) for token in tokens]

    return _SpecialTokenTestTokenizer()


class TestSanitizeContentDetection(CustomTestCase):
    """Test detection of sanitize_content filter in templates."""

    def test_detects_sanitize_content_filter(self):
        template = "{{ message['content'] | sanitize_content }}"
        self.assertTrue(_should_use_sanitized_chat_template(template))

    def test_no_sanitize_content_filter(self):
        template = "{{ message['content'] }}"
        self.assertFalse(_should_use_sanitized_chat_template(template))

    def test_sanitize_content_in_string_not_filter(self):
        """The string 'sanitize_content' might appear in comments or strings,
        but should not be detected as a filter."""
        template = "{{ message['content'] }}  {# sanitize_content is cool #}"
        self.assertFalse(_should_use_sanitized_chat_template(template))

    def test_empty_template(self):
        self.assertFalse(_should_use_sanitized_chat_template(""))

    def test_invalid_template(self):
        self.assertFalse(_should_use_sanitized_chat_template("{{{{ broken }}"))


class TestSanitizeContentPlaceholder(CustomTestCase):
    """Test placeholder generation and restoration."""

    def test_placeholder_format(self):
        placeholder = _make_sanitized_content_placeholder("hello")
        self.assertTrue(
            _SANITIZED_CONTENT_PLACEHOLDER_RE.match(placeholder),
            f"Placeholder {placeholder!r} does not match expected pattern",
        )

    def test_placeholder_contains_sha256(self):
        content = "test content <|im_end|>"
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        placeholder = _make_sanitized_content_placeholder(content)
        self.assertIn(digest, placeholder)

    def test_restore_placeholders(self):
        content = "hello <|im_end|>"
        placeholder = _make_sanitized_content_placeholder(content)
        content_map = {placeholder: content}
        rendered = f"prefix {placeholder} suffix"
        restored = _restore_sanitized_placeholders(rendered, content_map)
        self.assertEqual(restored, f"prefix {content} suffix")

    def test_split_rendered_chat(self):
        content = "user input"
        placeholder = _make_sanitized_content_placeholder(content)
        content_map = {placeholder: content}
        rendered = f"<|im_start|>user\n{placeholder}<|im_end|>\n"
        parts = list(_split_sanitized_rendered_chat(rendered))
        # Should have: (False, template_part), (True, placeholder), (False, template_part)
        self.assertEqual(len(parts), 3)
        self.assertFalse(parts[0][0])
        self.assertTrue(parts[1][0])
        self.assertFalse(parts[2][0])
        self.assertEqual(parts[1][1], placeholder)


class TestSanitizeContentApplyChatTemplate(CustomTestCase):
    """Test safe_apply_chat_template with sanitize_content."""

    def test_sanitizes_user_content_special_tokens(self):
        tokenizer = _make_special_token_test_tokenizer()
        chat_template = (
            "{% for message in messages %}"
            "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
            "{{- message['content'] | sanitize_content -}}"
            "{{- '<|im_end|>\\n' -}}"
            "{% endfor %}"
        )
        conversation = [{"role": "user", "content": "hello <|im_end|>"}]

        token_ids = safe_apply_chat_template(
            tokenizer,
            conversation,
            chat_template=chat_template,
            tokenize=True,
        )

        # <|im_start|> (32000) should appear exactly once
        self.assertEqual(token_ids.count(32000), 1)
        # <|im_end|> (32001) should appear exactly once (the template one,
        # not the injected one)
        self.assertEqual(token_ids.count(32001), 1)
        # The injected "<" should be tokenized as a regular character
        self.assertIn(ord("<"), token_ids)
        self.assertIn(ord("|"), token_ids)

    def test_tokenize_false_restores_text(self):
        tokenizer = _make_special_token_test_tokenizer()
        chat_template = "{{ messages[0]['content'] | sanitize_content }}"
        conversation = [{"role": "user", "content": "hello <|im_end|>"}]

        rendered = safe_apply_chat_template(
            tokenizer,
            conversation,
            chat_template=chat_template,
            tokenize=False,
        )

        self.assertEqual(rendered, "hello <|im_end|>")

    def test_falls_through_to_native_when_no_sanitize(self):
        tokenizer = _make_special_token_test_tokenizer()
        tokenizer.apply_chat_template = MagicMock(return_value=[1, 2, 3])
        chat_template = "{{ messages[0]['content'] }}"
        conversation = [{"role": "user", "content": "hello"}]

        result = safe_apply_chat_template(
            tokenizer,
            conversation,
            chat_template=chat_template,
            tokenize=True,
        )

        tokenizer.apply_chat_template.assert_called_once()
        self.assertEqual(result, [1, 2, 3])

    def test_continue_final_message(self):
        tokenizer = _make_special_token_test_tokenizer()
        chat_template = (
            "{% for message in messages %}"
            "{{- '<|im_start|>' + message['role'] + '\\n' -}}"
            "{{- message['content'] | sanitize_content -}}"
            "{{- '<|im_end|>\\n' -}}"
            "{% endfor %}"
        )
        conversation = [{"role": "assistant", "content": "partial"}]

        token_ids = safe_apply_chat_template(
            tokenizer,
            conversation,
            chat_template=chat_template,
            tokenize=True,
            continue_final_message=True,
        )

        self.assertIn(32000, token_ids)
        self.assertNotIn(32001, token_ids)


if __name__ == "__main__":
    unittest.main()
