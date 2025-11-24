import unittest

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDisableFastImageProcessor(CustomTestCase):
    def test_disable_fast_image_processor(self):
        IMAGE_SGL_LOGO_URL = "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
        other_args = (
            [
                "--disable-fast-image-processor",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--trust-remote-code",
            ]
            if is_npu()
            else [
                "--disable-fast-image-processor",
                "--trust-remote-code",
            ]
        )
        process = popen_launch_server(
            (
                "/root/.cache/modelscope/hub/models/LLM-Research/Phi-4-multimodal-instruct"
                if is_npu()
                else DEFAULT_SMALL_MODEL_NAME_FOR_TEST
            ),
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        self.assertEqual(response.status_code, 200)

        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json={
                    "model": "/root/.cache/modelscope/hub/models/LLM-Research/Phi-4-multimodal-instruct",
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
                                    "image_url": {"url": IMAGE_SGL_LOGO_URL},
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
            self.assertIn("dog", response.text)
            response = requests.get(DEFAULT_URL_FOR_TEST + "/get_server_info")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["disable_fast_image_processor"], True)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
