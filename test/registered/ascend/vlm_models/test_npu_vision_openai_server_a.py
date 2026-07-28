"""
Usage:
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_mixed_batch
python3 -m unittest test_vision_openai_server.TestOpenAIVisionServer.test_multi_images_chat_completion
"""

import unittest

import openai

from sglang.test.ascend.test_ascend_utils import (
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
    LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH,
    MINICPM_O_2_6_WEIGHTS_PATH,
    MINICPM_V_2_6_WEIGHTS_PATH,
    QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import *
from sglang.test.ascend.vlm_utils import (
    AudioOpenAITestMixin,
    ImageOpenAITestMixin,
    OmniOpenAITestMixin,
    TestOpenAIMLLMServerBase,
    VideoOpenAITestMixin,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=3200, suite="full-2-npu-a3", nightly=True)


class TestLlavaServer(ImageOpenAITestMixin):
    model = LLAVA_ONEVISION_QWEN2_7B_OV_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]


class TestQwen3VL8BServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.5",
        "--mm-attention-backend",
        "ascend",
    ]


class TestQwen3VLServer(ImageOpenAITestMixin, VideoOpenAITestMixin):
    model = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mem-fraction-static",
        "0.9",
        "--tp-size",
        "2",
        "--mm-attention-backend",
        "ascend",
    ]


class TestQwen2VLContextLengthServer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN2_VL_2B_INSTRUCT_WEIGHTS_PATH
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
                "--cuda-graph-max-bs-decode",
                "4",
                "--attention-backend",
                "ascend",
                "--mem-fraction-static",
                "0.90",
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
                                "image_url": {"url": IMAGE_MAN_IRONING_PATH},
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


class TestMiniCPMV4Server(ImageOpenAITestMixin):
    model = MINICPM_V_2_6_WEIGHTS_PATH
    extra_args = [
        "--cuda-graph-max-bs-decode",
        "4",
        "--attention-backend",
        "ascend",
    ]


class TestMiniCPMo26Server(ImageOpenAITestMixin, AudioOpenAITestMixin):
    model = MINICPM_O_2_6_WEIGHTS_PATH
    extra_args = [
        "--cuda-graph-max-bs-decode",
        "4",
        "--attention-backend",
        "ascend",
    ]


class TestKimiVLServer(ImageOpenAITestMixin):
    model = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
    extra_args = [
        "--context-length",
        "8192",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.8",
    ]

    def test_video_images_chat_completion(self):
        # model context length exceeded
        pass


# Delete the mixin classes so that they are not collected by pytest
del (
    TestOpenAIMLLMServerBase,
    ImageOpenAITestMixin,
    VideoOpenAITestMixin,
    AudioOpenAITestMixin,
    OmniOpenAITestMixin,
)


if __name__ == "__main__":
    unittest.main()
