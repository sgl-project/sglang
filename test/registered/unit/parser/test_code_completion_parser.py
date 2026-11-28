"""Unit tests for srt/parser/code_completion_parser.py"""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.parser.code_completion_parser import (
    CompletionTemplate,
    FimPosition,
    completion_template_exists,
    completion_templates,
    generate_completion_prompt,
    generate_completion_prompt_from_request,
    register_completion_template,
    set_completion_template,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="extra-test-cpu-intel")


class TestRegisterCompletionTemplate(CustomTestCase):
    def test_builtin_templates_registered(self):
        """Test that deepseek_coder, star_coder, qwen_coder are pre-registered."""
        self.assertTrue(completion_template_exists("deepseek_coder"))
        self.assertTrue(completion_template_exists("star_coder"))
        self.assertTrue(completion_template_exists("qwen_coder"))

    def test_unregistered_template_not_found(self):
        """Test that a non-existent template returns False."""
        self.assertFalse(completion_template_exists("nonexistent_template"))

    def test_register_duplicate_raises(self):
        """Test that registering a duplicate name without override raises."""
        with self.assertRaises(AssertionError):
            register_completion_template(
                CompletionTemplate(
                    name="deepseek_coder",
                    fim_begin_token="x",
                    fim_middle_token="y",
                    fim_end_token="z",
                    fim_position=FimPosition.MIDDLE,
                )
            )

    def test_register_duplicate_with_override(self):
        """Test that override=True allows re-registration."""
        original = completion_templates["deepseek_coder"]
        try:
            register_completion_template(
                CompletionTemplate(
                    name="deepseek_coder",
                    fim_begin_token="<new>",
                    fim_middle_token="<new_m>",
                    fim_end_token="<new_e>",
                    fim_position=FimPosition.END,
                ),
                override=True,
            )
            self.assertEqual(
                completion_templates["deepseek_coder"].fim_begin_token, "<new>"
            )
        finally:
            # Restore original
            completion_templates["deepseek_coder"] = original


class TestGenerateCompletionPrompt(CustomTestCase):
    def test_deepseek_coder_middle_position(self):
        """Test FIM prompt with MIDDLE position (deepseek_coder style)."""
        result = generate_completion_prompt(
            "prefix_code", "suffix_code", "deepseek_coder"
        )
        t = completion_templates["deepseek_coder"]
        expected = f"{t.fim_begin_token}prefix_code{t.fim_middle_token}suffix_code{t.fim_end_token}"
        self.assertEqual(result, expected)

    def test_star_coder_end_position(self):
        """Test FIM prompt with END position (star_coder style)."""
        result = generate_completion_prompt("prefix_code", "suffix_code", "star_coder")
        t = completion_templates["star_coder"]
        expected = f"{t.fim_begin_token}prefix_code{t.fim_end_token}suffix_code{t.fim_middle_token}"
        self.assertEqual(result, expected)

    def test_qwen_coder_end_position(self):
        """Test FIM prompt with END position (qwen_coder style)."""
        result = generate_completion_prompt("prefix", "suffix", "qwen_coder")
        t = completion_templates["qwen_coder"]
        expected = (
            f"{t.fim_begin_token}prefix{t.fim_end_token}suffix{t.fim_middle_token}"
        )
        self.assertEqual(result, expected)


class TestGenerateCompletionPromptFromRequest(CustomTestCase):
    def test_empty_suffix_returns_prompt_directly(self):
        """Test that empty suffix bypasses FIM formatting."""
        request = CompletionRequest(prompt="just code", suffix="")
        result = generate_completion_prompt_from_request(request)
        self.assertEqual(result, "just code")

    def test_nonempty_suffix_uses_fim_template(self):
        """Test that non-empty suffix triggers FIM formatting."""
        with patch(
            "sglang.srt.parser.code_completion_parser.completion_template_name",
            "deepseek_coder",
        ):
            request = CompletionRequest(prompt="prefix", suffix="suffix")
            result = generate_completion_prompt_from_request(request)
            t = completion_templates["deepseek_coder"]
            expected = (
                f"{t.fim_begin_token}prefix{t.fim_middle_token}suffix{t.fim_end_token}"
            )
            self.assertEqual(result, expected)

    def test_unset_suffix_returns_prompt_directly(self):
        """Test that a request without a suffix (None) bypasses FIM formatting."""
        with patch(
            "sglang.srt.parser.code_completion_parser.completion_template_name",
            "deepseek_coder",
        ):
            request = CompletionRequest(prompt="just code")
            self.assertIsNone(request.suffix)
            result = generate_completion_prompt_from_request(request)
            self.assertEqual(result, "just code")

    def test_unset_suffix_keeps_list_prompt(self):
        """Test that a batched prompt without a suffix is returned unchanged."""
        with patch(
            "sglang.srt.parser.code_completion_parser.completion_template_name",
            "deepseek_coder",
        ):
            request = CompletionRequest(prompt=["a", "b"])
            result = generate_completion_prompt_from_request(request)
            self.assertEqual(result, ["a", "b"])

    def test_list_prompt_formats_each_prompt(self):
        """Test that each prompt in a batch gets its own FIM formatting."""
        with patch(
            "sglang.srt.parser.code_completion_parser.completion_template_name",
            "deepseek_coder",
        ):
            request = CompletionRequest(prompt=["a", "b"], suffix="s")
            result = generate_completion_prompt_from_request(request)
            t = completion_templates["deepseek_coder"]
            expected = [
                f"{t.fim_begin_token}{p}{t.fim_middle_token}s{t.fim_end_token}"
                for p in ("a", "b")
            ]
            self.assertEqual(result, expected)

    def test_token_ids_prompt_with_suffix_raises(self):
        """Test that token-id prompts with a suffix are rejected, not stringified."""
        with patch(
            "sglang.srt.parser.code_completion_parser.completion_template_name",
            "deepseek_coder",
        ):
            for prompt in ([1, 2, 3], [[1, 2], [3, 4]]):
                with self.subTest(prompt=prompt):
                    request = CompletionRequest(prompt=prompt, suffix="s")
                    with self.assertRaises(ValueError):
                        generate_completion_prompt_from_request(request)


class TestSetCompletionTemplate(CustomTestCase):
    def test_set_only_once(self):
        """Test that set_completion_template only sets the name once."""
        import sglang.srt.parser.code_completion_parser as module

        with patch.object(module, "completion_template_name", None):
            set_completion_template("star_coder")
            self.assertEqual(module.completion_template_name, "star_coder")
            # Second call should be ignored
            set_completion_template("qwen_coder")
            self.assertEqual(module.completion_template_name, "star_coder")


if __name__ == "__main__":
    unittest.main()
