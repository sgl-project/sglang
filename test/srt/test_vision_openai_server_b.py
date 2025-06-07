import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestPixtralServer(TestOpenAIVisionServer):
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
                "0.73",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestMistral3_1Server(TestOpenAIVisionServer):
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
                "0.8",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestDeepseekVL2Server(TestOpenAIVisionServer):
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
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestJanusProServer(TestOpenAIVisionServer):
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
                "0.4",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass

    def test_single_image_chat_completion(self):
        # Skip this test because it is flaky
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
#             ],
#         )
#         cls.base_url += "/v1"

#     def test_video_chat_completion(self):
#         pass


class TestGemma3itServer(TestOpenAIVisionServer):
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
                "0.75",
                "--enable-multimodal",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestKimiVLServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-VL-A3B-Instruct"
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
                "--dtype",
                "bfloat16",
            ],
        )
        cls.base_url += "/v1"

    def test_video_chat_completion(self):
        pass


class TestPhi4MMServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        # Manually download LoRA adapter_config.json as it's not downloaded by the model loader by default.
        from huggingface_hub import constants, snapshot_download

        snapshot_download(
            "microsoft/Phi-4-multimodal-instruct",
            allow_patterns=["**/adapter_config.json"],
        )

        cls.model = "microsoft/Phi-4-multimodal-instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        revision = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.75",
                "--disable-radix-cache",
                "--max-loras-per-batch",
                "1",
                "--revision",
                revision,
                "--lora-paths",
                f"vision={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/vision-lora",
            ],
        )
        cls.base_url += "/v1"

    def get_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "vision",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    def test_video_chat_completion(self):
        pass


class TestVILAServer(TestOpenAIVisionServer):
    @classmethod
    def setUpClass(cls):
        cls.model = "AndyZijianZhang/NVILA-Lite-2B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--context-length=65536",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
    unittest.main()
