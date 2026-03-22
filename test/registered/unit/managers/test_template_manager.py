"""Unit tests for template_manager.py — no server, no model loading."""

from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-cpu-only")

import unittest

from sglang.srt.managers.template_manager import TemplateManager


class TestTemplateManagerInit(unittest.TestCase):

    def test_defaults(self):
        tm = TemplateManager()
        self.assertIsNone(tm.chat_template_name)
        self.assertIsNone(tm.completion_template_name)
        self.assertEqual(tm.jinja_template_content_format, "openai")
        self.assertFalse(tm.force_reasoning)


class TestDetectReasoningPattern(unittest.TestCase):

    def test_none_template(self):
        self.assertFalse(TemplateManager()._detect_reasoning_pattern(None))

    def test_no_reasoning_pattern(self):
        self.assertFalse(
            TemplateManager()._detect_reasoning_pattern("Hello {{ message }}")
        )

    def test_with_reasoning_pattern(self):
        template = r"<|im_start|>assistant\n<think>\n"
        self.assertTrue(TemplateManager()._detect_reasoning_pattern(template))

    def test_think_tag_alone_does_not_match(self):
        """The regex requires the full im_start + assistant + think sequence."""
        self.assertFalse(TemplateManager()._detect_reasoning_pattern("<think>"))


class TestSelectNamedTemplate(unittest.TestCase):

    def _make_tokenizer_manager(self, preferred_name=None):
        tm = Mock()
        tm.server_args = Mock()
        tm.server_args.hf_chat_template_name = preferred_name
        return tm

    def test_preferred_name_found(self):
        tm = TemplateManager()
        templates = {"default": "template_a", "tool_use": "template_b"}
        result = tm._select_named_template(
            templates, self._make_tokenizer_manager("tool_use")
        )
        self.assertEqual(result, "template_b")

    def test_preferred_name_not_found_raises(self):
        tm = TemplateManager()
        with self.assertRaises(ValueError) as ctx:
            tm._select_named_template(
                {"default": "t"}, self._make_tokenizer_manager("nonexistent")
            )
        self.assertIn("nonexistent", str(ctx.exception))

    def test_empty_dict_raises(self):
        tm = TemplateManager()
        with self.assertRaises(ValueError):
            tm._select_named_template({}, self._make_tokenizer_manager(None))

    def test_no_preference_uses_first(self):
        tm = TemplateManager()
        templates = {"alpha": "tmpl_a", "beta": "tmpl_b"}
        result = tm._select_named_template(
            templates, self._make_tokenizer_manager(None)
        )
        self.assertEqual(result, "tmpl_a")


class TestResolveHfChatTemplate(unittest.TestCase):

    def test_string_template_returned(self):
        tm = TemplateManager()
        tok_mgr = Mock()
        tok_mgr.processor = None
        tok_mgr.tokenizer = Mock()
        tok_mgr.tokenizer.chat_template = "simple template"
        self.assertEqual(tm._resolve_hf_chat_template(tok_mgr), "simple template")

    def test_none_template(self):
        tm = TemplateManager()
        tok_mgr = Mock()
        tok_mgr.processor = None
        tok_mgr.tokenizer = Mock()
        tok_mgr.tokenizer.chat_template = None
        self.assertIsNone(tm._resolve_hf_chat_template(tok_mgr))

    def test_processor_prioritized_over_tokenizer(self):
        """mm-processor's chat_template takes precedence over tokenizer's."""
        tm = TemplateManager()
        tok_mgr = Mock()
        tok_mgr.processor = Mock()
        tok_mgr.processor.chat_template = "processor_template"
        tok_mgr.tokenizer = Mock()
        tok_mgr.tokenizer.chat_template = "tokenizer_template"
        self.assertEqual(tm._resolve_hf_chat_template(tok_mgr), "processor_template")

    def test_no_tokenizer_no_processor(self):
        tm = TemplateManager()
        tok_mgr = Mock()
        tok_mgr.processor = None
        tok_mgr.tokenizer = None
        self.assertIsNone(tm._resolve_hf_chat_template(tok_mgr))

    def test_dict_template_selects_first(self):
        """Dict templates dispatch to _select_named_template; without a
        preference the first key is used."""
        tm = TemplateManager()
        tok_mgr = Mock()
        tok_mgr.processor = None
        tok_mgr.tokenizer = Mock()
        tok_mgr.tokenizer.chat_template = {"default": "tmpl_a", "tool": "tmpl_b"}
        tok_mgr.server_args = Mock()
        tok_mgr.server_args.hf_chat_template_name = None
        self.assertEqual(tm._resolve_hf_chat_template(tok_mgr), "tmpl_a")


class TestGuessFromModelPath(unittest.TestCase):

    @patch("sglang.srt.managers.template_manager.get_conv_template_by_model_path")
    def test_known_model_sets_name(self, mock_get):
        mock_get.return_value = "chatml"
        tm = TemplateManager()
        tm.guess_chat_template_from_model_path("/models/qwen-7b")
        self.assertEqual(tm.chat_template_name, "chatml")

    @patch("sglang.srt.managers.template_manager.get_conv_template_by_model_path")
    def test_unknown_model_stays_none(self, mock_get):
        mock_get.return_value = None
        tm = TemplateManager()
        tm.guess_chat_template_from_model_path("/models/unknown")
        self.assertIsNone(tm.chat_template_name)


if __name__ == "__main__":
    unittest.main()
