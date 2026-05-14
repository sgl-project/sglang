import unittest
from types import SimpleNamespace

from sglang.srt.managers.template_detection import (
    ReasoningToggleConfig,
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
    resolve_auto_parsers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "stage-a-test-cpu")


class _DummyTokenizer:
    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return {token: i for i, token in enumerate(self._vocab)}


class TestTemplateManagerReasoningDetection(unittest.TestCase):

    def _detect(self, template, vocab):
        force, config = detect_reasoning_pattern(template)
        parser = detect_reasoning_parser(
            template, _DummyTokenizer(vocab), config, force
        )
        return force, config, parser

    def test_qwen3_template_not_misclassified_as_glm45(self):
        template = """
        {% set enable_thinking = enable_thinking if enable_thinking is defined else true %}
        {% if '</think>' in content %}
        <tool_call>
        """
        _, config, parser = self._detect(
            template, ["<tool_call>", "<|endoftext|>", "</think>"]
        )

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "qwen3")

    def test_glm45_requires_glm_specific_template_markers(self):
        template = """
        [gMASK]<sop>
        {% set enable_thinking = enable_thinking if enable_thinking is defined else true %}
        /nothink
        <tool_call>
        """
        _, config, parser = self._detect(
            template, ["<tool_call>", "<|endoftext|>", "<|user|>"]
        )

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "glm45")

    def test_interns1_detects_enable_thinking_default_true(self):
        template = """
        {% set default_thinking_sys %}...<think>...</think>{% endset %}
        {% if enable_thinking is not defined or enable_thinking %}
        """
        _, config, parser = self._detect(template, ["<|endoftext|>"])

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "interns1")

    def test_nemotron_detects_uppercase_true_assignment(self):
        template = """
        {% set enable_thinking = enable_thinking if enable_thinking is defined else True %}
        {% set truncate_history_thinking = truncate_history_thinking if truncate_history_thinking is defined else True %}
        """
        _, config, parser = self._detect(template, ["<|endoftext|>"])

        self.assertEqual(
            config,
            ReasoningToggleConfig(toggle_param="enable_thinking", default_enabled=True),
        )
        self.assertEqual(parser, "nemotron_3")

    def test_minimax_uses_template_signature_without_toggle_config(self):
        template = """
        {%- set toolcall_begin_token = '<minimax:tool_call>' -%}
        """
        _, config, parser = self._detect(template, ["<minimax:tool_call>"])

        self.assertIsNone(config)
        self.assertEqual(parser, "minimax")


class TestTemplateDetectionRuleMatrix(unittest.TestCase):
    """Table-driven tests for REASONING_PARSER_RULES and REASONING_MODE_RULES."""

    def _detect(self, template, vocab=None):
        if vocab is None:
            vocab = []
        force, config = detect_reasoning_pattern(template)
        parser = detect_reasoning_parser(
            template, _DummyTokenizer(vocab), config, force
        )
        return force, config, parser

    PARSER_RULES_MATRIX = [
        # (name, template_snippet, vocab, expected_parser, expected_toggle_param)
        (
            "deepseek_r1_think_tags",
            "<think>\nLet me reason about this\n</think>\nAnswer here",
            [],
            "deepseek-r1",
            None,  # matched by deepseek_r1_think_tags rule (has <think> text)
        ),
        (
            "deepseek_v3",
            "{% if not thinking is defined %}{% set thinking = false %}{% endif %}\n"
            "<think>",
            [],
            "deepseek-v3",
            "thinking",
        ),
        (
            "qwen3_enable_thinking_true",
            "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n",
            [],
            "qwen3",
            "enable_thinking",
        ),
        (
            "kimi_unicode_markers",
            "\u25c1think\u25b7some text\u25c1/think\u25b7",
            [],
            "kimi",
            None,
        ),
        (
            "mistral_reasoning_effort",
            "{% if reasoning_effort %}[THINK]{% endif %}",
            [],
            "mistral",
            None,  # special_case="mistral"
        ),
        (
            "gpt_oss_channel",
            "<|channel|>analysis<|message|>",
            [],
            "gpt-oss",
            None,  # special_case="always"
        ),
        (
            "kimi_k2_with_tool_vocab",
            "{% set thinking = thinking if thinking is defined else true %}\n<think>",
            ["<|tool_calls_section_begin|>", "<|tool_calls_section_end|>"],
            "kimi_k2",
            "thinking",
        ),
        (
            "mimo_enable_thinking_false",
            "{% if not enable_thinking is defined %}{% set enable_thinking = false %}{% endif %}\n"
            "enable_thinking",
            [],
            "mimo",
            "enable_thinking",
        ),
    ]

    def test_parser_rules_matrix(self):
        for (
            name,
            template,
            vocab,
            expected_parser,
            expected_toggle,
        ) in self.PARSER_RULES_MATRIX:
            with self.subTest(name=name):
                _, config, parser = self._detect(template, vocab)
                self.assertEqual(
                    parser,
                    expected_parser,
                    f"Rule '{name}': expected parser '{expected_parser}', got '{parser}'",
                )
                if expected_toggle is not None:
                    self.assertIsNotNone(
                        config, f"Rule '{name}': expected config, got None"
                    )
                    self.assertEqual(
                        config.toggle_param,
                        expected_toggle,
                        f"Rule '{name}': expected toggle '{expected_toggle}', "
                        f"got '{config.toggle_param}'",
                    )

    def test_unrecognized_template_returns_none(self):
        template = "Hello {{ user_message }}, how can I help you?"
        _, config, parser = self._detect(template)

        self.assertIsNone(config)
        self.assertIsNone(parser)

    def test_empty_template_returns_none(self):
        _, config, parser = self._detect("")

        self.assertIsNone(config)
        self.assertIsNone(parser)

    def test_qwen3_precedence_over_deepseek_r1(self):
        """Template with enable_thinking=true but no <think> tag should be qwen3, not deepseek_r1."""
        template = "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
        _, config, parser = self._detect(template)

        self.assertEqual(parser, "qwen3")
        self.assertEqual(config.toggle_param, "enable_thinking")
        self.assertTrue(config.default_enabled)


class TestToolCallParserDetection(unittest.TestCase):
    """Tests for detect_tool_call_parser() using real model tokenizers."""

    def _detect_all(self, model_name):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        template = tok.chat_template
        force, config = detect_reasoning_pattern(template)
        rp = detect_reasoning_parser(template, tok, config, force)
        tcp = detect_tool_call_parser(template, tok, config, force)
        return rp, tcp

    def test_qwen3_detects_qwen_tool_call_parser(self):
        rp, tcp = self._detect_all("Qwen/Qwen3-0.6B")
        self.assertEqual(rp, "qwen3")
        self.assertEqual(tcp, "qwen")

    def test_tool_call_parser_rule_values_via_snippets(self):
        """Table-driven: verify tool-call rule values differ from reasoning where expected."""
        cases = [
            # (name, template, vocab, expected_tool_call)
            (
                "qwen_maps_from_qwen3_config",
                "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}",
                [],
                "qwen",
            ),
            ("gpt_oss", "<|channel|>analysis<|message|>", [], "gpt-oss"),
            ("gemma4", "<|channel>content", [], "gemma4"),
            ("minimax_maps_to_m2", "<minimax:tool_call>", [], "minimax-m2"),
            (
                "deepseekv3",
                "{% if not thinking is defined %}{% set thinking = false %}{% endif %}",
                [],
                "deepseekv3",
            ),
            (
                "kimi_k2",
                "{% set thinking = thinking if thinking is defined else true %}\n<think>",
                ["<|tool_calls_section_begin|>"],
                "kimi_k2",
            ),
        ]
        for name, template, vocab, expected in cases:
            with self.subTest(name=name):
                force, config = detect_reasoning_pattern(template)
                result = detect_tool_call_parser(
                    template, _DummyTokenizer(vocab), config, force
                )
                self.assertEqual(result, expected)

    def test_none_template_returns_none(self):
        self.assertIsNone(detect_tool_call_parser(None, None))

    def test_unrecognized_template_returns_none(self):
        force, config = detect_reasoning_pattern("Hello {{ user }}")
        result = detect_tool_call_parser("Hello {{ user }}", None, config, force)
        self.assertIsNone(result)


class TestResolveAutoParsers(unittest.TestCase):
    """Tests for resolve_auto_parsers() using real model tokenizers."""

    def _make_server_args(self, reasoning_parser=None, tool_call_parser=None):
        return SimpleNamespace(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            model_path="Qwen/Qwen3-0.6B",
            trust_remote_code=False,
        )

    def test_resolves_both_parsers_with_real_model(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        resolve_auto_parsers(args)
        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertEqual(args.tool_call_parser, "qwen")

    def test_resolves_reasoning_parser_only(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser=None)
        resolve_auto_parsers(args)
        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertIsNone(args.tool_call_parser)

    def test_resolves_tool_call_parser_only(self):
        args = self._make_server_args(reasoning_parser="qwen3", tool_call_parser="auto")
        resolve_auto_parsers(args)
        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertEqual(args.tool_call_parser, "qwen")

    def test_neither_auto_is_noop(self):
        args = self._make_server_args(reasoning_parser="qwen3", tool_call_parser="qwen")
        resolve_auto_parsers(args)
        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertEqual(args.tool_call_parser, "qwen")

    def test_nonexistent_model_disables_both_parsers(self):
        args = SimpleNamespace(
            reasoning_parser="auto",
            tool_call_parser="auto",
            model_path="nonexistent/model-does-not-exist-xyz",
            trust_remote_code=False,
        )
        resolve_auto_parsers(args)
        self.assertIsNone(args.reasoning_parser)
        self.assertIsNone(args.tool_call_parser)


if __name__ == "__main__":
    unittest.main()
