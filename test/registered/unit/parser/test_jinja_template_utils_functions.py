"""Unit tests for srt/parser/jinja_template_utils.py — internal helper functions.

These tests cover the internal AST analysis functions (_is_var_access, _is_attr_access,
_is_var_or_elems_access, _try_extract_ast) which are not directly tested by
test_jinja_template_utils.py.
"""

import unittest

import jinja2

from sglang.srt.parser.jinja_template_utils import (
    _is_attr_access,
    _is_var_access,
    _is_var_or_elems_access,
    _try_extract_ast,
    detect_jinja_template_content_format,
    process_content_for_template_format,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-b-test-cpu")


class TestIsVarAccess(CustomTestCase):
    """Test _is_var_access helper."""

    def test_name_node_matching(self):
        """Test that a Name node with matching varname returns True."""
        template = "{{ message }}"
        ast = _try_extract_ast(template)
        # Get the output node from the template
        output_node = ast.nodes[0]
        self.assertTrue(_is_var_access(output_node, "message"))

    def test_name_node_non_matching(self):
        """Test that a Name node with non-matching varname returns False."""
        template = "{{ message }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_var_access(output_node, "other"))

    def test_non_name_node_returns_false(self):
        """Test that non-Name nodes return False."""
        template = "{{ message.text }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_var_access(output_node, "message"))


class TestIsAttrAccess(CustomTestCase):
    """Test _is_attr_access helper."""

    def test_getitem_access(self):
        """Test Getitem access like message['content']."""
        template = "{{ message['content'] }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertTrue(_is_attr_access(output_node, "message", "content"))

    def test_getattr_access(self):
        """Test Getattr access like message.content."""
        template = "{{ message.content }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertTrue(_is_attr_access(output_node, "message", "content"))

    def test_getitem_wrong_key(self):
        """Test Getitem access with wrong key returns False."""
        template = "{{ message['other'] }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_attr_access(output_node, "message", "content"))

    def test_getattr_wrong_key(self):
        """Test Getattr access with wrong attribute returns False."""
        template = "{{ message.other }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_attr_access(output_node, "message", "content"))

    def test_non_getitem_getattr_returns_false(self):
        """Test that non-Getitem/Getattr nodes return False."""
        template = "{{ message }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_attr_access(output_node, "message", "content"))


class TestIsVarOrElemsAccess(CustomTestCase):
    """Test _is_var_or_elems_access helper."""

    def test_filter_on_var(self):
        """Test that Filter node wrapping a variable access is detected."""
        template = "{{ message['content'] | trim }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertTrue(_is_var_or_elems_access(output_node, "message", "content"))

    def test_test_on_var(self):
        """Test that Test node wrapping a variable access is detected."""
        template = "{% if message['content'] is string %}yes{% endif %}"
        ast = _try_extract_ast(template)
        test_node = ast.nodes[0]
        self.assertTrue(_is_var_or_elems_access(test_node, "message", "content"))

    def test_slice_on_var(self):
        """Test that Slice node wrapping a variable access is detected."""
        template = "{{ message['content'][:5] }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertTrue(_is_var_or_elems_access(output_node, "message", "content"))

    def test_nested_filter(self):
        """Test nested filter chain."""
        template = "{{ message['content'] | trim | safe }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        # The outer filter node should detect the inner var access through chain
        self.assertTrue(_is_var_or_elems_access(output_node, "message", "content"))

    def test_plain_var_access(self):
        """Test plain variable access (no filter/test/slice)."""
        template = "{{ message['content'] }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertTrue(_is_var_or_elems_access(output_node, "message", "content"))

    def test_wrong_varname(self):
        """Test that wrong varname returns False."""
        template = "{{ message['content'] }}"
        ast = _try_extract_ast(template)
        output_node = ast.nodes[0]
        self.assertFalse(_is_var_or_elems_access(output_node, "other", "content"))


class TestTryExtractAst(CustomTestCase):
    """Test _try_extract_ast helper."""

    def test_valid_template_returns_ast(self):
        """Test that a valid Jinja template returns an AST."""
        template = "{{ message['content'] }}"
        ast = _try_extract_ast(template)
        self.assertIsNotNone(ast)

    def test_invalid_template_returns_none(self):
        """Test that an invalid template returns None."""
        template = "{{{{ invalid }}}}"
        ast = _try_extract_ast(template)
        self.assertIsNone(ast)

    def test_empty_template_returns_ast(self):
        """Test that an empty template returns an AST (or None gracefully)."""
        template = ""
        ast = _try_extract_ast(template)
        # Should return None for empty template
        self.assertIsNone(ast)

    def test_complex_template_returns_ast(self):
        """Test that a complex Jinja template with loops and conditions returns an AST."""
        template = """
{%- for message in messages %}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for item in message['content'] %}
            {{- item }}
        {%- endfor %}
    {%- endif %}
{%- endfor %}
        """
        ast = _try_extract_ast(template)
        self.assertIsNotNone(ast)


class TestDetectJinjaTemplateContentFormatAstPath(CustomTestCase):
    """Test AST-based detection paths that bypass the keyword shortcut."""

    def test_ast_detects_msg_content_loop(self):
        """Test AST detection with msg.content iteration."""
        template = """
{%- for msg in messages %}
    {%- if msg.content is string %}
        {{- msg.content }}
    {%- else %}
        {%- for item in msg.content %}
            {{- item.text }}
        {%- endfor %}
    {%- endif %}
{%- endfor %}
        """
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "openai")

    def test_ast_detects_m_content_loop(self):
        """Test AST detection with m.content iteration."""
        template = """
{%- for m in messages %}
    {%- if m.content is string %}
        {{- m.content }}
    {%- else %}
        {%- for item in m.content %}
            {{- item.text }}
        {%- endfor %}
    {%- endif %}
{%- endfor %}
        """
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "openai")

    def test_ast_falls_back_to_string_when_no_content_loop(self):
        """Test that AST analysis returns string when no content loop is found."""
        template = """
{%- for message in messages %}
    {{- message['role'] }}: {{ message['content'] }}
{%- endfor %}
        """
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "string")

    def test_keyword_shortcut_image(self):
        """Test that 'image' keyword triggers openai shortcut."""
        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        # Add image keyword to trigger shortcut
        template_with_image = template + "<!-- image -->"
        result = detect_jinja_template_content_format(template_with_image)
        self.assertEqual(result, "openai")

    def test_keyword_shortcut_video(self):
        """Test that 'video' keyword triggers openai shortcut."""
        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        template_with_video = template + "<!-- video content -->"
        result = detect_jinja_template_content_format(template_with_video)
        self.assertEqual(result, "openai")

    def test_keyword_shortcut_audio(self):
        """Test that 'audio' keyword triggers openai shortcut."""
        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        template_with_audio = template + "<!-- audio -->"
        result = detect_jinja_template_content_format(template_with_audio)
        self.assertEqual(result, "openai")

    def test_keyword_shortcut_vision(self):
        """Test that 'vision' keyword triggers openai shortcut."""
        template = "{% for msg in messages %}{{ msg.content }}{% endfor %}"
        template_with_vision = template + "<!-- vision -->"
        result = detect_jinja_template_content_format(template_with_vision)
        self.assertEqual(result, "openai")


class TestProcessContentToolReference(CustomTestCase):
    """Test process_content_for_template_format with tool_reference content type."""

    def test_tool_reference_passed_through(self):
        """Test that tool_reference content type is passed through unchanged."""
        msg_dict = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_reference",
                    "name": "get_weather",
                    "id": "call_123",
                },
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(len(result["content"]), 1)
        self.assertEqual(result["content"][0]["type"], "tool_reference")
        self.assertEqual(result["content"][0]["name"], "get_weather")

    def test_mixed_content_with_tool_reference(self):
        """Test content with text, image_url, and tool_reference parts."""
        msg_dict = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "The weather is "},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/chart.png"},
                },
                {
                    "type": "tool_reference",
                    "name": "get_weather",
                    "id": "call_456",
                },
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        # Image should be extracted
        self.assertEqual(len(image_data), 1)
        self.assertEqual(image_data[0].url, "http://example.com/chart.png")

        # Content should preserve order with tool_reference
        content_types = [c["type"] for c in result["content"]]
        self.assertEqual(content_types, ["text", "image", "tool_reference"])


class TestProcessContentEdgeCases(CustomTestCase):
    """Additional edge cases for process_content_for_template_format."""

    def test_none_content_returns_empty(self):
        """Test that None content returns empty dict (after filtering None values)."""
        msg_dict = {"role": "user", "content": None}
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(result["content"], "")

    def test_string_content_unchanged_for_openai(self):
        """Test that string content passes through unchanged for openai format."""
        msg_dict = {"role": "user", "content": "Just a plain string"}
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(result["content"], "Just a plain string")

    def test_string_content_unchanged_for_string_format(self):
        """Test that string content passes through unchanged for string format."""
        msg_dict = {"role": "user", "content": "Just a plain string"}
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(result["content"], "Just a plain string")

    def test_invalid_format_raises(self):
        """Test that invalid content_format raises ValueError."""
        msg_dict = {
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}],
        }
        with self.assertRaises(ValueError) as ctx:
            process_content_for_template_format(
                msg_dict, "invalid_format", [], [], [], []
            )
        self.assertIn("Invalid content format", str(ctx.exception))

    def test_image_url_with_max_dynamic_patch(self):
        """Test image_url with max_dynamic_patch is extracted correctly."""
        msg_dict = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "http://example.com/img.jpg",
                        "detail": "high",
                        "max_dynamic_patch": 4,
                    },
                },
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(len(image_data), 1)
        self.assertEqual(image_data[0].url, "http://example.com/img.jpg")
        self.assertEqual(image_data[0].detail, "high")
        self.assertEqual(image_data[0].max_dynamic_patch, 4)
        # Content should be normalized to {"type": "image"}
        self.assertEqual(result["content"], [{"type": "image"}])

    def test_video_url_with_modalities(self):
        """Test video_url with modalities field is extracted."""
        msg_dict = {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": "http://example.com/v.mp4"},
                    "modalities": ["video"],
                },
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(len(video_data), 1)
        self.assertEqual(video_data[0], "http://example.com/v.mp4")
        self.assertEqual(len(modalities), 1)
        self.assertEqual(modalities[0], ["video"])

    def test_audio_url_normalized(self):
        """Test audio_url content is normalized to simple audio type."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen: "},
                {
                    "type": "audio_url",
                    "audio_url": {"url": "http://example.com/audio.mp3"},
                },
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        self.assertEqual(len(audio_data), 1)
        self.assertEqual(audio_data[0], "http://example.com/audio.mp3")
        # Content should be normalized
        expected = [
            {"type": "text", "text": "Listen: "},
            {"type": "audio"},
        ]
        self.assertEqual(result["content"], expected)

    def test_string_format_ignores_multimodal(self):
        """Test that string format ignores image/audio/video parts."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello "},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/img.jpg"},
                },
                {"type": "text", "text": "World"},
            ],
        }
        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, video_data, audio_data, modalities
        )

        # String format flattens to text only
        self.assertEqual(result["content"], "Hello World")
        # No image data extracted for string format
        self.assertEqual(len(image_data), 0)


if __name__ == "__main__":
    unittest.main()