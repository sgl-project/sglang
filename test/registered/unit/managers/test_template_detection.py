"""Unit tests for sglang.srt.managers.template_detection."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import re
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.template_detection import (
    DetectionRule,
    ReasoningToggleConfig,
    TemplateDetectionContext,
    build_detection_context,
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
    match_rules,
)
from sglang.test.test_utils import CustomTestCase


class TestReasoningToggleConfig(CustomTestCase):
    def test_always_on_true_when_special_case_is_always(self):
        """always_on is True when special_case is 'always'."""
        cfg = ReasoningToggleConfig(special_case="always")
        self.assertTrue(cfg.always_on)

    def test_always_on_false_when_special_case_is_other_value(self):
        """always_on is False when special_case is any value other than 'always'."""
        cfg = ReasoningToggleConfig(special_case="mistral")
        self.assertFalse(cfg.always_on)

    def test_always_on_false_when_special_case_is_none(self):
        """always_on is False when special_case is None."""
        cfg = ReasoningToggleConfig()
        self.assertFalse(cfg.always_on)


class TestTemplateDetectionContext(CustomTestCase):
    def setUp(self):
        self.ctx = TemplateDetectionContext(
            template="hello <|channel|> world",
            reasoning_config=None,
            force_reasoning=False,
            vocab={"<think>", "<tool_call>"},
        )

    def test_has_text_present(self):
        """has_text returns True when the needle is a substring of the template."""
        self.assertTrue(self.ctx.has_text("<|channel|>"))

    def test_has_text_absent(self):
        """has_text returns False when the needle is not in the template."""
        self.assertFalse(self.ctx.has_text("<missing>"))

    def test_has_vocab_present(self):
        """has_vocab returns True when the token exists in the vocabulary set."""
        self.assertTrue(self.ctx.has_vocab("<think>"))

    def test_has_vocab_absent(self):
        """has_vocab returns False when the token is not in the vocabulary set."""
        self.assertFalse(self.ctx.has_vocab("<unknown>"))

    def test_has_pattern_matches(self):
        """has_pattern returns True when the regex matches the template."""
        self.assertTrue(self.ctx.has_pattern(r"<\|channel\|>"))

    def test_has_pattern_no_match(self):
        """has_pattern returns False when the regex does not match the template."""
        self.assertFalse(self.ctx.has_pattern(r"<missing>"))

    def test_has_pattern_with_dotall_flag(self):
        """has_pattern with re.DOTALL matches a pattern that spans a newline."""
        ctx = TemplateDetectionContext(
            template="start\nend",
            reasoning_config=None,
            force_reasoning=False,
            vocab=set(),
        )
        self.assertTrue(ctx.has_pattern(r"start.end", re.DOTALL))
        self.assertFalse(ctx.has_pattern(r"start.end"))


class TestBuildDetectionContext(CustomTestCase):
    def test_none_template_returns_none(self):
        """build_detection_context returns None when template is None."""
        self.assertIsNone(build_detection_context(None, tokenizer=None))

    def test_none_tokenizer_yields_empty_vocab(self):
        """build_detection_context with tokenizer=None produces an empty vocab."""
        ctx = build_detection_context("some template", tokenizer=None)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.vocab, set())

    def test_tokenizer_vocab_is_loaded(self):
        """build_detection_context populates vocab from tokenizer.get_vocab()."""
        mock_tok = MagicMock()
        mock_tok.get_vocab.return_value = {"<think>": 0, "<tool_call>": 1}
        ctx = build_detection_context("template", mock_tok)
        self.assertIn("<think>", ctx.vocab)
        self.assertIn("<tool_call>", ctx.vocab)

    def test_tokenizer_get_vocab_exception_yields_empty_vocab(self):
        """build_detection_context returns empty vocab when tokenizer.get_vocab() raises."""
        mock_tok = MagicMock()
        mock_tok.get_vocab.side_effect = RuntimeError("broken vocab")
        ctx = build_detection_context("template", mock_tok)
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.vocab, set())


class TestMatchRules(CustomTestCase):
    def setUp(self):
        self.ctx = TemplateDetectionContext(
            template="",
            reasoning_config=None,
            force_reasoning=False,
            vocab=set(),
        )

    def test_first_matching_rule_is_returned(self):
        """match_rules returns the value of the first rule whose predicate is True."""
        rules = (
            DetectionRule(name="r1", value="v1", predicate=lambda c: True),
            DetectionRule(name="r2", value="v2", predicate=lambda c: True),
        )
        self.assertEqual(match_rules(self.ctx, rules, "test"), "v1")

    def test_no_matching_rule_returns_none(self):
        """match_rules returns None when no predicate matches."""
        rules = (DetectionRule(name="r1", value="v1", predicate=lambda c: False),)
        self.assertIsNone(match_rules(self.ctx, rules, "test"))

    def test_exception_in_predicate_skips_that_rule(self):
        """match_rules skips a rule whose predicate raises and continues to the next rule."""

        def _bad_predicate(ctx):
            raise ValueError("oops")

        rules = (
            DetectionRule(name="bad", value="bad_val", predicate=_bad_predicate),
            DetectionRule(name="good", value="good_val", predicate=lambda c: True),
        )
        self.assertEqual(match_rules(self.ctx, rules, "test"), "good_val")


class TestDetectReasoningPattern(CustomTestCase):
    def test_none_template_returns_false_and_none(self):
        """detect_reasoning_pattern returns (False, None) for None template."""
        is_always_on, config = detect_reasoning_pattern(None)
        self.assertFalse(is_always_on)
        self.assertIsNone(config)

    def test_unrecognized_template_returns_false_and_none(self):
        """detect_reasoning_pattern returns (False, None) for a template with no known markers."""
        is_always_on, config = detect_reasoning_pattern(
            "plain template without special markers"
        )
        self.assertFalse(is_always_on)
        self.assertIsNone(config)

    def test_gpt_oss_channel_marker_detected_as_always_on(self):
        """Template containing <|channel|> is detected as always-on reasoning."""
        is_always_on, config = detect_reasoning_pattern("some text <|channel|> here")
        self.assertTrue(is_always_on)
        self.assertEqual(config, ReasoningToggleConfig(special_case="always"))

    def test_force_reasoning_pattern_detected_as_always_on(self):
        """Template with im_start/think pattern is detected as always-on when thinking is absent."""
        # literal \\n (backslash-n) in the template string matches the \\\\n in the regex
        template = "<|im_start|>assistant\\n<think>\\n"
        is_always_on, config = detect_reasoning_pattern(template)
        self.assertTrue(is_always_on)
        self.assertEqual(config, ReasoningToggleConfig(special_case="always"))

    def test_mistral_reasoning_effort_detected(self):
        """Template with reasoning_effort and [THINK] is detected as mistral reasoning config."""
        template = "reasoning_effort [THINK] is supported"
        is_always_on, config = detect_reasoning_pattern(template)
        self.assertFalse(is_always_on)
        self.assertEqual(config, ReasoningToggleConfig(special_case="mistral"))

    def test_explicit_enable_thinking_default_false_detected(self):
        """Template with enable_thinking=false default block is detected correctly."""
        template = (
            "{% if not enable_thinking is defined %}"
            "{% set enable_thinking = false %}"
        )
        is_always_on, config = detect_reasoning_pattern(template)
        self.assertFalse(is_always_on)
        self.assertEqual(
            config,
            ReasoningToggleConfig(
                toggle_param="enable_thinking", default_enabled=False
            ),
        )


class TestDetectReasoningParser(CustomTestCase):
    def test_none_template_returns_none(self):
        """detect_reasoning_parser returns None when template is None."""
        self.assertIsNone(detect_reasoning_parser(None, None))

    def test_kimi_template_detected(self):
        """Template with kimi thinking marker is detected as kimi reasoning parser."""
        result = detect_reasoning_parser("text with \u25c1think\u25b7 marker", None)
        self.assertEqual(result, "kimi")


class TestDetectToolCallParser(CustomTestCase):
    def test_none_template_returns_none(self):
        """detect_tool_call_parser returns None when template is None."""
        self.assertIsNone(detect_tool_call_parser(None, None))

    def test_minimax_template_detected(self):
        """Template with minimax marker is detected as minimax-m2 tool-call parser."""
        result = detect_tool_call_parser("uses <minimax:tool_call>", None)
        self.assertEqual(result, "minimax-m2")


if __name__ == "__main__":
    unittest.main()
