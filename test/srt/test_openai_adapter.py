"""
Unit tests for OpenAI adapter utils.
"""

import unittest
from unittest.mock import patch

from sglang.srt.openai_api.utils import (
    detect_template_content_format,
    process_content_for_template_format,
)
from sglang.test.test_utils import CustomTestCase


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

        result = detect_template_content_format(llama4_pattern)
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

        result = detect_template_content_format(deepseek_pattern)
        self.assertEqual(result, "string")

    def test_detect_invalid_template(self):
        """Test handling of invalid template (should default to 'string')."""
        invalid_pattern = "{{{{ invalid jinja syntax }}}}"

        result = detect_template_content_format(invalid_pattern)
        self.assertEqual(result, "string")

    def test_detect_empty_template(self):
        """Test handling of empty template (should default to 'string')."""
        result = detect_template_content_format("")
        self.assertEqual(result, "string")

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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, audio_data, modalities
        )

        # Check that image_data was extracted
        self.assertEqual(len(image_data), 1)
        self.assertEqual(image_data[0], "http://example.com/image.jpg")

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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, audio_data, modalities
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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, audio_data, modalities
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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, audio_data, modalities
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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "openai", image_data, audio_data, modalities
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
        audio_data = []
        modalities = []

        result = process_content_for_template_format(
            msg_dict, "string", image_data, audio_data, modalities
        )

        # None values should be filtered out
        expected_keys = {"role", "content"}
        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
