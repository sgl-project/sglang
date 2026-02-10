import unittest

import requests
from sglang.test.ascend.test_ascend_utils import PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestDisableFastImageProcessor(CustomTestCase):
    """Testcase：Verify set --disable-fast-image-processor, can normally handle multimodal (picture+text) requests.

       [Test Category] Parameter
       [Test Target] --disable-fast-image-processor
       """
    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    IMAGE_SGL_LOGO_URL = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"

    @classmethod
    def setUpClass(cls):
        other_args = (
            [
                "--disable-fast-image-processor",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
            ]
        )
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_disable_fast_image_processor(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                "model": self.model,
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a helpful assistant."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": self.IMAGE_SGL_LOGO_URL},
                            },
                            {
                                "type": "text",
                                "text": "What is the content of the caption ?",
                            },
                        ],
                    },
                ],
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("caption", response.text)


if __name__ == "__main__":
    unittest.main()
