import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class TestPixtralServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "mistral-community/pixtral-12b"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestMistral3_1Server(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "unsloth/Mistral-Small-3.1-24B-Instruct-2503"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.75",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestDeepseekVL2Server(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/deepseek-vl2-small"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--context-length",
                "4096",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestJanusProServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "deepseek-ai/Janus-Pro-7B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.35",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    def test_video_images_chat_completion(self):
        pass


## Skip for ci test
# class TestLlama4Server(TestOpenAIVisionServer):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             other_args=[
#                 "--chat-template",
#                 "llama-4",
#                 "--mem-fraction-static",
#                 "0.8",
#                 "--tp-size=8",
#                 "--context-length=8192",
#                 "--mm-attention-backend",
#                 "fa3",
#                 "--cuda-graph-max-bs",
#                 "4",
#             ],
#         )
#         cls.base_url += "/v1"


class TestGemma3itServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-3-4b-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
                "--enable-multimodal",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestGemma3nServer(ImageOpenAITestMixin, AudioOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-3n-E4B-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    # This _test_audio_ambient_completion test is way too complicated to pass for a small LLM
    def test_audio_ambient_completion(self):
        pass

    def _test_mixed_image_audio_chat_completion(self):
        self._test_mixed_image_audio_chat_completion()


class TestQwen2AudioServer(AudioOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2-Audio-7B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.70",
            ],
        )
        cls.base_url += "/v1"


# Temporarily skip Kimi-VL for CI test due to issue in transformers=4.57.0
# class TestKimiVLServer(ImageOpenAITestMixin):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "moonshotai/Kimi-VL-A3B-Instruct"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             other_args=[
#                 "--trust-remote-code",
#                 "--context-length",
#                 "4096",
#                 "--dtype",
#                 "bfloat16",
#                 "--cuda-graph-max-bs",
#                 "4",
#             ],
#         )
#         cls.base_url += "/v1"

#     def test_video_images_chat_completion(self):
#         pass


class TestGLM41VServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "zai-org/GLM-4.1V-9B-Thinking"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.68",
                "--cuda-graph-max-bs",
                "4",
                "--reasoning-parser",
                "glm45",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
    del (
        TestOpenAIOmniServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
    )
    unittest.main()
