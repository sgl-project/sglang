"""
Tests for lightonai/LightOnOCR-2-1B model support.

Usage:
    python3 -m unittest test_lightonocr_models.TestLightOnOCRServer
    python3 -m unittest test_lightonocr_models.TestLightOnOCRServer.test_single_image_ocr
"""

import unittest

import openai

from sglang.test.vlm_utils import IMAGE_MAN_IRONING_URL, TestOpenAIMLLMServerBase

MODEL = "lightonai/LightOnOCR-2-1B"
# OCR test image with visible text content
OCR_IMAGE_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png"


class TestLightOnOCRServer(TestOpenAIMLLMServerBase):
    """Functional tests for LightOnOCR OpenAI-compatible API."""

    model = MODEL
    trust_remote_code = False
    extra_args = [
        "--mem-fraction-static=0.85",
        "--cuda-graph-max-bs=4",
    ]

    def test_single_image_ocr(self):
        """Verify OCR output contains recognized text from a document image."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": OCR_IMAGE_URL},
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=4096,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert len(text) > 0, "OCR output should not be empty"

        # Verify basic response metadata
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_single_image_chat_completion(self):
        """Verify the model can describe a general image."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in a sentence.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=256,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert len(text) > 0, "Response should not be empty"
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_multi_image_ocr(self):
        """Verify the model handles multiple images in a single request."""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": OCR_IMAGE_URL},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_MAN_IRONING_URL},
                            "modalities": "multi-images",
                        },
                        {
                            "type": "text",
                            "text": "Describe both images.",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=1024,
            **(self.get_vision_request_kwargs()),
        )

        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)
        assert len(text) > 0
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0


# Delete mixin base classes to prevent pytest auto-collection
del TestOpenAIMLLMServerBase


if __name__ == "__main__":
    unittest.main()
