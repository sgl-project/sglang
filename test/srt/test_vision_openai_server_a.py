"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import unittest

from test_vision_openai_server_common import *


class TestLlavaServer(ImageOpenAITestMixin):
    model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"


class TestQwen2VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
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
                "--mem-fraction-static",
                "0.35",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestQwen3VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-VL-30B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--mem-fraction-static",
                "0.80",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestQwen25VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "Qwen/Qwen2.5-VL-7B-Instruct"
    other_args = [
        "--mem-fraction-static",
        "0.35",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestQwen2VLContextLengthServer(CustomTestCase):
    model = "Qwen/Qwen2-VL-7B-Instruct"
    other_args = [
        "--context-length",
        "300",
        "--mem-fraction-static=0.75",
        "--cuda-graph-max-bs",
        "4",
    ]

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


class TestMllamaServer(ImageOpenAITestMixin):
    model = "meta-llama/Llama-3.2-11B-Vision-Instruct"


class TestInternVL25Server(ImageOpenAITestMixin):
    model = "OpenGVLab/InternVL2_5-2B"
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestMiniCPMV4Server(ImageOpenAITestMixin):
    model = "openbmb/MiniCPM-V-4"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.35",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestMiniCPMo26Server(ImageOpenAITestMixin, AudioOpenAITestMixin):
    model = "openbmb/MiniCPM-o-2_6"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.65",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestMiMoVLServer(ImageOpenAITestMixin):
    model = "XiaomiMiMo/MiMo-VL-7B-RL"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.6",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestVILAServer(ImageOpenAITestMixin):
    model = "Efficient-Large-Model/NVILA-Lite-2B-hf-0626"
    revision = "6bde1de5964b40e61c802b375fff419edc867506"
    other_args = [
        "--trust-remote-code",
        "--context-length=65536",
        f"--revision={revision}",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestGemma3itServer(ImageOpenAITestMixin):
    model = "google/gemma-3-4b-it"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.70",
        "--enable-multimodal",
        "--cuda-graph-max-bs",
        "4",
    ]


class TestKimiVLServer(ImageOpenAITestMixin):
    model = "moonshotai/Kimi-VL-A3B-Instruct"
    other_args = [
        "--trust-remote-code",
        "--context-length",
        "8192",
        "--dtype=bfloat16",
        "--mem-fraction-static=0.7",
    ]

    def test_video_images_chat_completion(self):
        # model context length exceeded
        pass


class TestGLM41VServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = "zai-org/GLM-4.1V-9B-Thinking"
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.68",
        "--cuda-graph-max-bs",
        "4",
        "--reasoning-parser",
        "glm45",
    ]


if __name__ == "__main__":
    del (
        TestOpenAIOmniServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
    )
    unittest.main()
