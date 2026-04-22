"""Unit tests for srt/parser/jinja_template_utils.py"""

import unittest

from sglang.srt.parser.jinja_template_utils import (
    detect_jinja_template_content_format,
    process_content_for_template_format,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")


class TestTemplateContentFormatDetection(CustomTestCase):
    """Test template content format detection functionality."""

    def test_detect_llama4_openai_format(self):
        """Test detection of llama4-style template (should be 'openai' format)."""
        llama4_pattern = """
{%- for message in messages %}
    {%- if message['content'] is string %}
        {{- message['content'] }}
    {%- else %}
        {%- for content in message['content'] %}
            {%- if content['type'] == 'image' %}
                {{- '<|image|>' }}
            {%- elif content['type'] == 'text' %}
                {{- content['text'] | trim }}
            {%- endif %}
        {%- endfor %}
    {%- endif %}
{%- endfor %}
        """

        result = detect_jinja_template_content_format(llama4_pattern)
        self.assertEqual(result, "openai")

    def test_detect_deepseek_string_format(self):
        """Test detection of deepseek-style template (should be 'string' format)."""
        deepseek_pattern = """
{%- for message in messages %}
    {%- if message['role'] == 'user' %}
        {{- '<|User|>' + message['content'] + '<|Assistant|>' }}
    {%- endif %}
{%- endfor %}
        """

        result = detect_jinja_template_content_format(deepseek_pattern)
        self.assertEqual(result, "string")

    def test_detect_invalid_template(self):
        """Test handling of invalid template (should default to 'string')."""
        invalid_pattern = "{{{{ invalid jinja syntax }}}}"

        result = detect_jinja_template_content_format(invalid_pattern)
        self.assertEqual(result, "string")

    def test_detect_empty_template(self):
        """Test handling of empty template (should default to 'string')."""
        result = detect_jinja_template_content_format("")
        self.assertEqual(result, "string")

    def test_detect_msg_content_pattern(self):
        """Test detection of template with msg.content pattern (should be 'openai' format)."""
        msg_content_pattern = """
[gMASK]<sop>
{%- for msg in messages %}
    {%- if msg.role == 'system' %}
<|system|>
{{ msg.content }}
    {%- elif msg.role == 'user' %}
<|user|>{{ '\n' }}
        {%- if msg.content is string %}
{{ msg.content }}
        {%- else %}
            {%- for item in msg.content %}
                {%- if item.type == 'video' or 'video' in item %}
<|begin_of_video|><|video|><|end_of_video|>
                {%- elif item.type == 'image' or 'image' in item %}
<|begin_of_image|><|image|><|end_of_image|>
                {%- elif item.type == 'text' %}
{{ item.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- elif msg.role == 'assistant' %}
        {%- if msg.metadata %}
<|assistant|>{{ msg.metadata }}
{{ msg.content }}
        {%- else %}
<|assistant|>
{{ msg.content }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}<|assistant|>
{% endif %}
        """

        result = detect_jinja_template_content_format(msg_content_pattern)
        self.assertEqual(result, "openai")

    def test_detect_m_content_pattern(self):
        """Test detection of template with m.content pattern (should be 'openai' format)."""
        msg_content_pattern = """
[gMASK]<sop>
{%- for m in messages %}
    {%- if m.role == 'system' %}
<|system|>
{{ m.content }}
    {%- elif m.role == 'user' %}
<|user|>{{ '\n' }}
        {%- if m.content is string %}
{{ m.content }}
        {%- else %}
            {%- for item in m.content %}
                {%- if item.type == 'video' or 'video' in item %}
<|begin_of_video|><|video|><|end_of_video|>
                {%- elif item.type == 'image' or 'image' in item %}
<|begin_of_image|><|image|><|end_of_image|>
                {%- elif item.type == 'text' %}
{{ item.text }}
                {%- endif %}
            {%- endfor %}
        {%- endif %}
    {%- elif m.role == 'assistant' %}
        {%- if m.metadata %}
<|assistant|>{{ m.metadata }}
{{ m.content }}
        {%- else %}
<|assistant|>
{{ m.content }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{% if add_generation_prompt %}<|assistant|>
{% endif %}
        """

        result = detect_jinja_template_content_format(msg_content_pattern)
        self.assertEqual(result, "openai")

    def test_process_content_openai_format(self):
        """Test content processing for openai format."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image:"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
                {"type": "text", "text": "What do you see?"},
            ],
        }

        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        # Check that image_data was extracted
        self.assertEqual(len(image_data), 1)
        self.assertEqual(image_data[0].url, "http://example.com/image.jpg")

        # Check that content was normalized
        expected_content = [
            {"type": "text", "text": "Look at this image:"},
            {"type": "image"},  # normalized from image_url
            {"type": "text", "text": "What do you see?"},
        ]
        self.assertEqual(result["content"], expected_content)
        self.assertEqual(result["role"], "user")

    def test_process_content_string_format(self):
        """Test content processing for string format."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                },
                {"type": "text", "text": "world"},
            ],
        }

        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, video_data, audio_data, modalities
        )

        # For string format, should flatten to text only
        self.assertEqual(result["content"], "Hello world")
        self.assertEqual(result["role"], "user")

        # Image data should not be extracted for string format
        self.assertEqual(len(image_data), 0)

    def test_process_content_with_audio(self):
        """Test content processing with audio content."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Listen to this:"},
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

        # Check that audio_data was extracted
        self.assertEqual(len(audio_data), 1)
        self.assertEqual(audio_data[0], "http://example.com/audio.mp3")

        # Check that content was normalized
        expected_content = [
            {"type": "text", "text": "Listen to this:"},
            {"type": "audio"},  # normalized from audio_url
        ]
        self.assertEqual(result["content"], expected_content)

    def test_process_content_already_string(self):
        """Test processing content that's already a string."""
        msg_dict = {"role": "user", "content": "Hello world"}

        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        # Should pass through unchanged
        self.assertEqual(result["content"], "Hello world")
        self.assertEqual(result["role"], "user")
        self.assertEqual(len(image_data), 0)

    def test_process_content_with_modalities(self):
        """Test content processing with modalities field."""
        msg_dict = {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "http://example.com/image.jpg"},
                    "modalities": ["vision"],
                }
            ],
        }

        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, video_data, audio_data, modalities
        )

        # Check that modalities was extracted
        self.assertEqual(len(modalities), 1)
        self.assertEqual(modalities[0], ["vision"])

    def test_process_content_filter_none_values(self):
        """Test that None values are filtered out of processed messages."""
        msg_dict = {
            "role": "user",
            "content": "Hello",
            "name": None,
            "tool_call_id": None,
        }

        image_data = []
        video_data = []
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, video_data, audio_data, modalities
        )

        # None values should be filtered out
        expected_keys = {"role", "content"}
        self.assertEqual(set(result.keys()), expected_keys)

    def test_process_content_with_video(self):
        """Test content processing with video_url content."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Watch this:"},
                {"type": "video_url", "video_url": {"url": "http://example.com/v.mp4"}},
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
        self.assertEqual(result["content"][1], {"type": "video"})

    def test_process_content_video_with_max_dynamic_patch(self):
        """Test video_url with max_dynamic_patch stores structured dict."""
        msg_dict = {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": "http://example.com/v.mp4",
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
        self.assertEqual(len(video_data), 1)
        self.assertIsInstance(video_data[0], dict)
        self.assertEqual(video_data[0]["max_dynamic_patch"], 4)

    def test_process_content_v32_encoding(self):
        """Test v32 encoding mode flattens text and ignores structured content parts."""
        msg_dict = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
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
            msg_dict,
            "openai",
            image_data,
            video_data,
            audio_data,
            modalities,
            use_dpsk_v32_encoding=True,
        )
        # v32 encoding: content is joined text, not list
        self.assertEqual(result["content"], "Hello World")
        # Image data is still extracted
        self.assertEqual(len(image_data), 1)

    def test_process_content_invalid_format_raises(self):
        """Test that invalid content_format raises ValueError."""
        msg_dict = {
            "role": "user",
            "content": [{"type": "text", "text": "Hi"}],
        }
        with self.assertRaises(ValueError):
            process_content_for_template_format(
                msg_dict, "invalid_format", [], [], [], []
            )

    def test_process_content_video_with_modalities(self):
        """Test that video content with modalities field is extracted."""
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
        self.assertEqual(len(modalities), 1)
        self.assertEqual(modalities[0], ["video"])

    def test_detect_template_with_filter(self):
        """Test that content access through a Jinja filter is detected as openai."""
        # Template with | trim filter on content iteration
        template = """
{%- for message in messages %}
    {%- for content in message['content'] | trim %}
        {{- content }}
    {%- endfor %}
{%- endfor %}
        """
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "openai")

    def test_detect_template_with_is_test(self):
        """Test that 'is string' test on content triggers openai detection."""
        # Template with 'is string' test that also iterates content
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
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "openai")

    def test_detect_template_with_slice(self):
        """Test that content access through slice is detected as openai."""
        template = """
{%- for message in messages %}
    {%- for item in message['content'][:5] %}
        {{- item }}
    {%- endfor %}
{%- endfor %}
        """
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "openai")

    def test_detect_template_no_content_loop_is_string(self):
        """Test that template without content iteration returns string format."""
        template = """
{%- for message in messages %}
    {{- message['role'] }}: {{ message['content'] }}
{%- endfor %}
        """
        # No "image"/"audio"/"video" keyword, no content loop → string
        result = detect_jinja_template_content_format(template)
        self.assertEqual(result, "string")

    def test_detect_msg_content_without_multimodal_keywords(self):
        """Test AST detection of 'for item in msg.content' without keyword shortcut.
        Templates that contain 'image'/'video'/'audio'/'vision' take a shortcut.
        This template deliberately avoids those keywords to test the AST path."""
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


if __name__ == "__main__":
    unittest.main()
