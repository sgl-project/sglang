"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import unittest

from test_vision_openai_server_common import *


class TestLlavaServer(ImageOpenAITestMixin):
    model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"


class TestQwen25VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "Qwen/Qwen2.5-VL-7B-Instruct"
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]


class TestQwen3VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    extra_args = ["--cuda-graph-max-bs=4"]


class TestQwen3OmniServer(OmniOpenAITestMixin):
    model = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    extra_args = [  # workaround to fit into H100
        "--mem-fraction-static=0.90",
        "--disable-cuda-graph",
        "--disable-fast-image-processor",
        "--grammar-backend=none",
    ]


class TestQwen2VLContextLengthServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-VL-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--context-length",
                "300",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
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
                                "text": "Give a lengthy description of this picture",
                            },
                        ],
                    },
                ],
                temperature=0,
            )

        # context length is checked first, then max_req_input_len, which is calculated from the former
        assert (
            "Multimodal prompt is too long after expanding multimodal tokens."
            in str(cm.exception)
            or "is longer than the model's context length" in str(cm.exception)
        )


# flaky
# class TestMllamaServer(ImageOpenAITestMixin):
#     model = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class TestInternVL25Server(ImageOpenAITestMixin):
    model = "OpenGVLab/InternVL2_5-2B"
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]


class TestMiniCPMV4Server(ImageOpenAITestMixin):
    model = "openbmb/MiniCPM-V-4"
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]


class TestMiniCPMo26Server(ImageOpenAITestMixin, AudioOpenAITestMixin):
    model = "openbmb/MiniCPM-o-2_6"
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]


class TestGemma3itServer(ImageOpenAITestMixin):
    model = "google/gemma-3-4b-it"
    extra_args = [
        "--cuda-graph-max-bs=4",
    ]


class TestKimiVLServer(ImageOpenAITestMixin):
    model = "moonshotai/Kimi-VL-A3B-Instruct"
    extra_args = [
        "--context-length=8192",
        "--dtype=bfloat16",
    ]

    def test_video_images_chat_completion(self):
        # model context length exceeded
        pass


@unittest.skip(
    "Temporarily disabling this test to fix CI. It should be re-enabled when #11800 is done."
)
class TestGLM41VServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "zai-org/GLM-4.1V-9B-Thinking"
    extra_args = [
        "--reasoning-parser=glm45",
    ]


class TestQwen2AudioServer(AudioOpenAITestMixin):
    model = "Qwen/Qwen2-Audio-7B-Instruct"


class TestDeepseekOCRServer(TestOpenAIMLLMServerBase):
    model = "deepseek-ai/DeepSeek-OCR"
    trust_remote_code = False

    def verify_single_image_response_for_ocr(self, response):
        """Verify DeepSeek-OCR grounding output with coordinates"""
        assert response.choices[0].message.role == "assistant"
        text = response.choices[0].message.content
        assert isinstance(text, str)

        # DeepSeek-OCR uses grounding format, outputs coordinates
        assert "text" in text.lower(), f"OCR text: {text}, should contain 'text'"

        # Verify coordinate format [[x1, y1, x2, y2]]
        import re

        coord_pattern = r"\[\[[\d\s,]+\]\]"
        assert re.search(
            coord_pattern, text
        ), f"OCR text: {text}, should contain coordinate format [[x1, y1, x2, y2]]"

        # Verify basic response fields
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_single_image_chat_completion(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        image_url = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/ocr-text.png"

        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url},
                        },
                        {
                            "type": "text",
                            "text": "<|grounding|>Convert the document to markdown.",
                        },
                    ],
                },
            ],
            temperature=0,
            **(self.get_vision_request_kwargs()),
        )

        self.verify_single_image_response_for_ocr(response)


if __name__ == "__main__":
    del (
        TestOpenAIMLLMServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
        OmniOpenAITestMixin,
    )
    unittest.main()
