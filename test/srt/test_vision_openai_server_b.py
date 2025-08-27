import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
<<<<<<< HEAD
    DEFAULT_URL_FOR_TEST,
=======
>>>>>>> origin/main
    popen_launch_server,
)


<<<<<<< HEAD
class TestPixtralServer(TestOpenAIVisionServer):
=======
class TestPixtralServer(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
                "0.73",
=======
                "0.70",
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
    def test_video_chat_completion(self):
        pass


class TestMistral3_1Server(TestOpenAIVisionServer):
=======

class TestMistral3_1Server(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
                "0.8",
=======
                "0.75",
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
    def test_video_chat_completion(self):
        pass


class TestDeepseekVL2Server(TestOpenAIVisionServer):
=======

class TestDeepseekVL2Server(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
    def test_video_chat_completion(self):
        pass


class TestJanusProServer(TestOpenAIVisionServer):
=======

class TestJanusProServer(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
                "0.4",
=======
                "0.35",
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
    def test_video_chat_completion(self):
        pass

    def test_single_image_chat_completion(self):
        # Skip this test because it is flaky
=======
    def test_video_images_chat_completion(self):
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
#                 "--mm-attention-backend",
#                 "fa3",
#                 "--cuda-graph-max-bs",
#                 "4",
>>>>>>> origin/main
#             ],
#         )
#         cls.base_url += "/v1"

<<<<<<< HEAD
#     def test_video_chat_completion(self):
#         pass


class TestGemma3itServer(TestOpenAIVisionServer):
=======

class TestGemma3itServer(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
    def test_video_chat_completion(self):
        pass


class TestKimiVLServer(TestOpenAIVisionServer):
=======

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


class TestKimiVLServer(ImageOpenAITestMixin):
>>>>>>> origin/main
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
<<<<<<< HEAD
=======
                "--cuda-graph-max-bs",
                "4",
>>>>>>> origin/main
            ],
        )
        cls.base_url += "/v1"

<<<<<<< HEAD
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
        cls.model = "Efficient-Large-Model/NVILA-Lite-2B-hf-0626"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.revision = "6bde1de5964b40e61c802b375fff419edc867506"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--context-length=65536",
                f"--revision={cls.revision}",
            ],
        )
        cls.base_url += "/v1"


if __name__ == "__main__":
=======
    def test_video_images_chat_completion(self):
        pass


# Skip for ci test
# class TestGLM41VServer(TestOpenAIVisionServer):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "zai-org/GLM-4.1V-9B-Thinking"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             other_args=[
#                 "--trust-remote-code",
#                 "--mem-fraction-static",
#                 "0.68",
#                 "--cuda-graph-max-bs",
#                 "4",
#                 "--reasoning-parser",
#                 "glm45",
#             ],
#         )
#         cls.base_url += "/v1"

#     def test_video_chat_completion(self):
#         self._test_video_chat_completion()


if __name__ == "__main__":
    del (
        TestOpenAIOmniServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
    )
>>>>>>> origin/main
    unittest.main()
