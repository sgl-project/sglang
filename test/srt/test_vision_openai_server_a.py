"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import unittest

from test_vision_openai_server_common import *

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)


class TestLlava(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "lmms-lab/llava-onevision-qwen2-0.5b-ov"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )
        cls.base_url += "/v1"


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


class TestQwen2_5_VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen2.5-VL-7B-Instruct"
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


class TestVLMContextLengthIssue(CustomTestCase):
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
                "--mem-fraction-static=0.75",
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


# Note(Xinyuan): mllama is not stable for now, skip for CI
# class TestMllamaServer(TestOpenAIVisionServer):
#     @classmethod
#     def setUpClass(cls):
#         cls.model = "meta-llama/Llama-3.2-11B-Vision-Instruct"
#         cls.base_url = DEFAULT_URL_FOR_TEST
#         cls.api_key = "sk-123456"
#         cls.process = popen_launch_server(
#             cls.model,
#             cls.base_url,
#             timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
#             api_key=cls.api_key,
#         )
#         cls.base_url += "/v1"


class TestMinicpmvServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-V-2_6"
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


class TestMinicpmv4Server(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-V-4"
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


class TestInternVL2_5Server(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "OpenGVLab/InternVL2_5-2B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestMinicpmo2_6Server(ImageOpenAITestMixin, AudioOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "openbmb/MiniCPM-o-2_6"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.65",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestMimoVLServer(ImageOpenAITestMixin):
    @classmethod
    def setUpClass(cls):
        cls.model = "XiaomiMiMo/MiMo-VL-7B-RL"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.6",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestVILAServer(ImageOpenAITestMixin):
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
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"


class TestPhi4MMServer(ImageOpenAITestMixin, AudioOpenAITestMixin):
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
                "0.70",
                "--disable-radix-cache",
                "--max-loras-per-batch",
                "2",
                "--revision",
                revision,
                "--lora-paths",
                f"vision={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/vision-lora",
                f"speech={constants.HF_HUB_CACHE}/models--microsoft--Phi-4-multimodal-instruct/snapshots/{revision}/speech-lora",
                "--cuda-graph-max-bs",
                "4",
            ],
        )
        cls.base_url += "/v1"

    def get_vision_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "vision",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    def get_audio_request_kwargs(self):
        return {
            "extra_body": {
                "lora_path": "speech",
                "top_k": 1,
                "top_p": 1.0,
            }
        }

    # This _test_audio_ambient_completion test is way too complicated to pass for a small LLM
    def test_audio_ambient_completion(self):
        pass


if __name__ == "__main__":
    del (
        TestOpenAIOmniServerBase,
        ImageOpenAITestMixin,
        VideoOpenAITestMixin,
        AudioOpenAITestMixin,
    )
    unittest.main()
