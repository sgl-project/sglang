import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.managers.template_detection import (
    REASONING_PARSER_RULES,
    TOOL_CALL_PARSER_RULES,
    ReasoningToggleConfig,
    detect_reasoning_parser,
    detect_reasoning_pattern,
    detect_tool_call_parser,
    resolve_auto_parsers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(2.0, "base-a-test-cpu")


class _DummyTokenizer:
    def __init__(self, vocab, chat_template=None):
        self._vocab = vocab
        self.chat_template = chat_template

    def get_vocab(self):
        return {token: i for i, token in enumerate(self._vocab)}


def _patch_hf_transformers_utils(get_tokenizer, get_config=None):
    module = ModuleType("sglang.srt.utils.hf_transformers_utils")
    module.get_tokenizer = get_tokenizer
    if get_config is not None:
        module.get_config = get_config
    return patch.dict(sys.modules, {module.__name__: module})


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
            "qwen3_5_no_enable_thinking",
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>assistant\n<think>\nThinking...\n</think>\n",
            [],
            "qwen3",
            None,
        ),
        (
            "deepseek_v4_reasoning_effort",
            "{% if reasoning_effort is defined %}<think>...</think><｜Assistant｜>",
            [],
            "deepseek-v3",
            None,
        ),
        (
            "apertus2509_via_unique_vocab_token",
            "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n",
            ["<|inner_prefix|>"],
            "apertus2509",
            "enable_thinking",
        ),
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
            "deepseek_v4_dsml_tool_calls",
            "{% if not thinking is defined %}{% set thinking = false %}{% endif %}\n"
            '<｜DSML｜tool_calls><｜DSML｜invoke name="tool"></｜DSML｜invoke>',
            [],
            "deepseek-v4",
            "thinking",
        ),
        (
            "hunyuan_interleaved_thinking",
            "{% set reasoning_effort = reasoning_effort | default('no_think', true) %}\n"
            "{% set interleaved_thinking = interleaved_thinking | default(false, true) %}\n"
            "<think>reasoning</think><tool_calls><tool_call>name<tool_sep>",
            ["<tool_calls>", "<tool_sep>", "<arg_key>", "<arg_value>"],
            "hunyuan",
            None,
        ),
        (
            "poolside_v1_enable_thinking_false",
            "{% if not enable_thinking is defined %}{% set enable_thinking = false %}{% endif %}\n"
            "<tool_call>{{ name }}\n<arg_key>{{ key }}</arg_key><arg_value>{{ value }}</arg_value>",
            ["<tool_call>", "<arg_key>", "<arg_value>"],
            "poolside_v1",
            "enable_thinking",
        ),
        (
            "poolside_v1_actual_template_shape",
            "{% set enable_thinking = enable_thinking | default(false) %}\n"
            "Wrap your thinking in '<think>', '</think>' tags, followed by a function call.\n"
            "For each function call, return an unescaped XML-like object with function name "
            "and arguments within '<tool_call>' and '</tool_call>' tags, like here:\n"
            "<tool_call>function-name\n"
            "<arg_key>argument-key</arg_key>\n"
            "<arg_value>value-of-argument-key</arg_value>\n"
            "</tool_call>",
            ["<tool_call>"],
            "poolside_v1",
            None,
        ),
        (
            "lfm2_not_deepseek_r1_from_history_cleanup",
            "{% set keep_past_thinking = keep_past_thinking | default(false) %}\n"
            "{% if not keep_past_thinking and '</think>' in content %}"
            "{{ content.split('</think>')[-1] }}{% endif %}\n"
            '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>',
            ["<|tool_call_start|>", "<|tool_call_end|>"],
            None,
            None,
        ),
        (
            "qwen3_coder_actual_template_shape_has_no_reasoning",
            "<tools>\n"
            "<tool_call><function=get_weather>"
            "<parameter=city>Paris</parameter></function></tool_call>",
            ["<tool_call>"],
            None,
            None,
        ),
        (
            "step3p5_actual_template_shape",
            "{% if reasoning_effort is defined %}Reasoning: {{ reasoning_effort }}{% endif %}\n"
            "<tool_call><function=get_weather>"
            "<parameter=city>Paris</parameter></function></tool_call>\n"
            "{% if '<think>' in content %}{{ content.split('</think>')[-1] }}{% endif %}",
            ["<tool_call>", "<tool_calls>"],
            "step3p5",
            None,
        ),
        (
            "step3p5_think_tags",
            "Step3.5-Flash\n<function=tool><parameter=arg>value</parameter></function>\n<think>",
            [],
            "step3p5",
            None,
        ),
        (
            "step3_steptml",
            "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
            '<steptml:invoke name="tool"><steptml:parameter name="arg">value</steptml:parameter>'
            "</steptml:invoke><｜tool_call_end｜><think>",
            [],
            "step3",
            None,
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
                "deepseekv31",
                (
                    "{% if not thinking is defined %}{% set thinking = false %}{% endif %}\n"
                    "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool<｜tool▁sep｜>{}<｜tool▁call▁end｜>"
                ),
                [],
                "deepseekv31",
            ),
            (
                "deepseekv32",
                '<｜DSML｜function_calls><｜DSML｜invoke name="tool"></｜DSML｜invoke></｜DSML｜function_calls>',
                [],
                "deepseekv32",
            ),
            (
                "deepseekv4",
                '<｜DSML｜tool_calls><｜DSML｜invoke name="tool"></｜DSML｜invoke></｜DSML｜tool_calls>',
                [],
                "deepseekv4",
            ),
            (
                "kimi_k2",
                "{% set thinking = thinking if thinking is defined else true %}\n<think>",
                ["<|tool_calls_section_begin|>"],
                "kimi_k2",
            ),
            (
                "minicpm5",
                (
                    "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
                    '\n<function name="{{ tool.name }}">'
                    '\n<param name="{{ param.name }}">{{ param.value }}</param>'
                    "\n</function>"
                ),
                ["<function", "<param"],
                "minicpm5",
            ),
            (
                "lfm2",
                '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>',
                ["<|tool_call_start|>", "<|tool_call_end|>"],
                "lfm2",
            ),
            (
                "hunyuan",
                "<tool_calls><tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call></tool_calls>",
                ["<tool_calls>", "<tool_sep>", "<arg_key>", "<arg_value>"],
                "hunyuan",
            ),
            (
                "poolside_v1",
                "{% if not enable_thinking is defined %}{% set enable_thinking = false %}{% endif %}\n"
                "<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
                ["<tool_call>", "<arg_key>", "<arg_value>"],
                "poolside_v1",
            ),
            (
                "poolside_v1_actual_template_shape",
                "{% set enable_thinking = enable_thinking | default(false) %}\n"
                "return an unescaped XML-like object with function name and arguments "
                "within '<tool_call>' and '</tool_call>' tags\n"
                "<tool_call>function-name\n"
                "<arg_key>argument-key</arg_key>\n"
                "<arg_value>value-of-argument-key</arg_value>\n"
                "</tool_call>",
                ["<tool_call>"],
                "poolside_v1",
            ),
            (
                "qwen3_coder",
                "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n"
                "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
                ["<tool_call>"],
                "qwen3_coder",
            ),
            (
                "step3p5",
                "Step3.5-Flash\n<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>",
                ["<tool_call>"],
                "step3p5",
            ),
            (
                "step3p5_actual_template_shape",
                "{% if reasoning_effort is defined %}Reasoning: {{ reasoning_effort }}{% endif %}\n"
                "<tool_call><function=get_weather>"
                "<parameter=city>Paris</parameter></function></tool_call>",
                ["<tool_call>", "<tool_calls>"],
                "step3p5",
            ),
            (
                "step3",
                "<｜tool_calls_begin｜><｜tool_call_begin｜>function<｜tool_sep｜>"
                '<steptml:invoke name="get_weather"><steptml:parameter name="city">Paris</steptml:parameter>'
                "</steptml:invoke><｜tool_call_end｜><｜tool_calls_end｜>",
                [],
                "step3",
            ),
            (
                "glm47_compact_tool_call",
                (
                    "[gMASK]<sop>\n"
                    "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n"
                    "{% for tc in m.tool_calls %}\n"
                    "{{- '<tool_call>' + tc.name -}}\n"
                    "{% set _args = tc.arguments %}"
                    "{% for k, v in _args.items() %}"
                    "<arg_key>{{ k }}</arg_key><arg_value>{{ v }}</arg_value>"
                    "{% endfor %}</tool_call>\n"
                    "{% endfor %}"
                ),
                ["<tool_call>", "<arg_key>", "<arg_value>", "<|endoftext|>"],
                "glm47",
            ),
            (
                "glm45_newline_tool_call",
                (
                    "[gMASK]<sop>\n"
                    "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}\n"
                    "{% for tc in m.tool_calls %}\n"
                    "{{ '\\n<tool_call>' + tc.name }}\n"
                    "{% set _args = tc.arguments %}\n"
                    "{% for k, v in _args.items() %}\n"
                    "<arg_key>{{ k }}</arg_key>\n<arg_value>{{ v }}</arg_value>\n"
                    "{% endfor %}\n</tool_call>\n"
                    "{% endfor %}"
                ),
                ["<tool_call>", "<arg_key>", "<arg_value>", "<|endoftext|>"],
                "glm45",
            ),
            (
                "xml_kv_tool_call_via_vocab",
                "{% set reasoning_effort = reasoning_effort | default('high', true) %}\n<think>",
                ["<tool_call>", "<arg_key>", "<arg_value>", "<|endoftext|>"],
                "glm45",
            ),
        ]
        for name, template, vocab, expected in cases:
            with self.subTest(name=name):
                force, config = detect_reasoning_pattern(template)
                result = detect_tool_call_parser(
                    template, _DummyTokenizer(vocab), config, force
                )
                self.assertEqual(result, expected)

    def test_glm45_rule_precedes_xml_kv_fallback(self):
        # The specific GLM-4.5 family check must run before the generic
        # xml_kv_tool_call fallback. Both currently map to "glm45", so the
        # value-based test above can't catch a swap — assert positions directly.
        rule_index = {rule.name: i for i, rule in enumerate(TOOL_CALL_PARSER_RULES)}
        self.assertLess(rule_index["glm47"], rule_index["glm45"])
        self.assertLess(rule_index["glm45"], rule_index["xml_kv_tool_call"])

    def test_specific_rules_precede_broad_fallbacks(self):
        reasoning_index = {
            rule.name: i for i, rule in enumerate(REASONING_PARSER_RULES)
        }
        self.assertLess(reasoning_index["deepseek_v4"], reasoning_index["deepseek_v3"])
        self.assertLess(
            reasoning_index["hunyuan"], reasoning_index["deepseek_r1_think_tags"]
        )
        self.assertLess(reasoning_index["poolside_v1"], reasoning_index["mimo"])
        self.assertLess(
            reasoning_index["step3p5"], reasoning_index["deepseek_r1_think_tags"]
        )
        self.assertLess(
            reasoning_index["step3"], reasoning_index["deepseek_r1_think_tags"]
        )

        tool_index = {rule.name: i for i, rule in enumerate(TOOL_CALL_PARSER_RULES)}
        self.assertLess(tool_index["deepseek_v31"], tool_index["deepseek_v3"])
        self.assertLess(tool_index["hunyuan"], tool_index["xml_kv_tool_call"])
        self.assertLess(tool_index["poolside_v1"], tool_index["xml_kv_tool_call"])
        self.assertLess(tool_index["step3p5"], tool_index["qwen3_coder"])
        self.assertLess(tool_index["step3"], tool_index["deepseek_v3"])
        self.assertLess(tool_index["qwen3_coder"], tool_index["qwen"])

    def test_xml_kv_requires_both_arg_tokens(self):
        template = "Hello {{ user }}"
        force, config = detect_reasoning_pattern(template)
        for vocab in (["<arg_key>"], ["<arg_value>"], []):
            with self.subTest(vocab=vocab):
                result = detect_tool_call_parser(
                    template, _DummyTokenizer(vocab), config, force
                )
                self.assertIsNone(result)

    def test_none_template_returns_none(self):
        self.assertIsNone(detect_tool_call_parser(None, None))

    def test_unrecognized_template_returns_none(self):
        force, config = detect_reasoning_pattern("Hello {{ user }}")
        result = detect_tool_call_parser("Hello {{ user }}", None, config, force)
        self.assertIsNone(result)

    def test_minicpm5_rule_precedes_broad_fallback_rules(self):
        rule_names = [rule.name for rule in TOOL_CALL_PARSER_RULES]
        minicpm5_idx = rule_names.index("minicpm5")
        self.assertLess(minicpm5_idx, rule_names.index("mimo"))
        self.assertLess(minicpm5_idx, rule_names.index("qwen"))

    def test_minicpm5_not_misclassified_as_qwen(self):
        template = (
            "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
            '\n<function name="{{ tool.name }}">'
            '\n<param name="{{ param.name }}">{{ param.value }}</param>'
            "\n</function>"
        )
        force, config = detect_reasoning_pattern(template)
        result = detect_tool_call_parser(
            template, _DummyTokenizer(["<function", "<param"]), config, force
        )
        self.assertEqual(result, "minicpm5")


class TestResolveAutoParsers(unittest.TestCase):
    """Tests for resolve_auto_parsers()."""

    qwen3_template = "{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}"

    def _make_server_args(
        self, reasoning_parser=None, tool_call_parser=None, chat_template=None
    ):
        return SimpleNamespace(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            model_path="Qwen/Qwen3-0.6B",
            trust_remote_code=False,
            chat_template=chat_template,
        )

    def test_resolves_both_parsers_with_tokenizer_template(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        tokenizer = _DummyTokenizer([], chat_template=self.qwen3_template)

        with _patch_hf_transformers_utils(Mock(return_value=tokenizer)):
            resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertEqual(args.tool_call_parser, "qwen")

    def test_resolves_reasoning_parser_only(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser=None)
        tokenizer = _DummyTokenizer([], chat_template=self.qwen3_template)

        with _patch_hf_transformers_utils(Mock(return_value=tokenizer)):
            resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "qwen3")
        self.assertIsNone(args.tool_call_parser)

    def test_resolves_tool_call_parser_only(self):
        args = self._make_server_args(reasoning_parser="qwen3", tool_call_parser="auto")
        tokenizer = _DummyTokenizer([], chat_template=self.qwen3_template)

        with _patch_hf_transformers_utils(Mock(return_value=tokenizer)):
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
        with _patch_hf_transformers_utils(
            Mock(side_effect=RuntimeError("tokenizer unavailable")),
            Mock(side_effect=RuntimeError("config unavailable")),
        ):
            resolve_auto_parsers(args)

        self.assertIsNone(args.reasoning_parser)
        self.assertIsNone(args.tool_call_parser)

    def test_none_chat_template_disables_both_parsers(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        tokenizer = _DummyTokenizer([])

        with _patch_hf_transformers_utils(Mock(return_value=tokenizer)):
            resolve_auto_parsers(args)

        self.assertIsNone(args.reasoning_parser)
        self.assertIsNone(args.tool_call_parser)

    def test_deepseek_v32_arch_without_chat_template_uses_custom_encoder(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        tokenizer = _DummyTokenizer([])
        config = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"])

        with _patch_hf_transformers_utils(
            Mock(return_value=tokenizer), Mock(return_value=config)
        ):
            resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "deepseek-v3")
        self.assertEqual(args.tool_call_parser, "deepseekv32")

    def test_deepseek_v4_arch_without_chat_template_uses_custom_encoder(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        tokenizer = _DummyTokenizer([])
        config = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])

        with _patch_hf_transformers_utils(
            Mock(return_value=tokenizer), Mock(return_value=config)
        ):
            resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "deepseek-v4")
        self.assertEqual(args.tool_call_parser, "deepseekv4")

    def test_deepseek_arch_fallback_runs_when_tokenizer_load_fails(self):
        args = self._make_server_args(reasoning_parser="auto", tool_call_parser="auto")
        config = SimpleNamespace(architectures=["DeepseekV32ForCausalLM"])

        with _patch_hf_transformers_utils(
            Mock(side_effect=RuntimeError("tokenizer unavailable")),
            Mock(return_value=config),
        ):
            resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "deepseek-v3")
        self.assertEqual(args.tool_call_parser, "deepseekv32")

    def test_explicit_non_jinja_template_skips_architecture_fallback(self):
        args = self._make_server_args(
            reasoning_parser="auto",
            tool_call_parser="auto",
            chat_template="chatml",
        )
        args.model_path = "deepseek-ai/DeepSeek-V3.2"
        tokenizer = _DummyTokenizer([])
        get_config = Mock()

        with _patch_hf_transformers_utils(Mock(return_value=tokenizer), get_config):
            resolve_auto_parsers(args)

        get_config.assert_not_called()
        self.assertIsNone(args.reasoning_parser)
        self.assertIsNone(args.tool_call_parser)

    def test_explicit_jinja_template_takes_precedence(self):
        tokenizer = _DummyTokenizer([], chat_template=None)

        with tempfile.NamedTemporaryFile("w", suffix=".jinja") as f:
            f.write(
                "{% if not thinking is defined %}{% set thinking = false %}{% endif %}\n"
                '<｜DSML｜function_calls><｜DSML｜invoke name="tool"></｜DSML｜invoke>'
            )
            f.flush()
            args = self._make_server_args(
                reasoning_parser="auto",
                tool_call_parser="auto",
                chat_template=f.name,
            )

            with _patch_hf_transformers_utils(Mock(return_value=tokenizer)):
                resolve_auto_parsers(args)

        self.assertEqual(args.reasoning_parser, "deepseek-v3")
        self.assertEqual(args.tool_call_parser, "deepseekv32")


if __name__ == "__main__":
    unittest.main()
