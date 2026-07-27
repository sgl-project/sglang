import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_1_1_PATH,
    QWEN3_VL_8B_THINKING_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)
IMAGE_SGL_LOGO_URL = IMAGES_1_1_PATH


class TestQwenVLPPAccuracy(CustomTestCase):
    """
    Using a multimodal language model,
    configure the pp-size and evaluate GSM8K accuracy as well as the correctness of image reasoning.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_8B_THINKING_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                1,
                "--pp-size",
                4,
                "--chunked-prefill-size",
                8192,
                "--enable-multimodal",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        self.assertGreaterEqual(metrics["score"], 0.9)
        # Wait a little bit so that the memory check happens.

    def test_images_request(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_SGL_LOGO_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in a sentence.",
                    },
                ],
            },
        ]
        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        assert response.status_code == 200
        self.assertIn("image", response.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()
