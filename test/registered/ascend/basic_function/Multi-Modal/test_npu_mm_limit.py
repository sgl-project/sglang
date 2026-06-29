import multiprocessing as mp
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_LOGO_PATH,
    IMAGES_MAN_PATH,
    QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

MODEL = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH

# image
IMAGE_MAN_IRONING_URL = IMAGES_MAN_PATH
IMAGE_SGL_LOGO_URL = IMAGES_LOGO_PATH


def _send_parallel_request_task1(base_url, image_url):
    requests.packages.urllib3.disable_warnings()
    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={"messages": messages, "temperature": 0, "max_completion_tokens": 512},
    )

    assert resp.status_code == 200


class TestLimitMMDatePerRequest(TestVLMModels, CustomTestCase):
    """Testcase: Configuring Multi-Modal to send different multimodal inference requests,
       each containing multiple multimodal input data.

    [Test Category] Parameter
    [Test Target] --enable-broadcast-mm-inputs-process; --limit-mm-data-per-request
    """

    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        limit_mm = '{"image":1, "video":1}'
        other_args = [
            "--mem-fraction-static",
            "0.5",
            "--enable-multimodal",
            "--enable-broadcast-mm-inputs-process",
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--tp-size",
            "4",
            "--disable-cuda-graph",
            "--limit-mm-data-per-request",
            limit_mm,
        ]
        cls.process = popen_launch_server(
            MODEL,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_multi_turn_request(self):
        # Input video and image respectively
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
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
        self.assertEqual(response.status_code, 200)

    def _run_multi_turn_request1(self):
        # Enter two images
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_SGL_LOGO_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this video in a sentence.",
                    },
                ],
            },
        ]
        response2 = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "messages": messages2,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )

        self.assertEqual(response2.status_code, 400)

    def _run_parallel_two_requests(self):
        url = self.base_url + "/v1"
        p1 = mp.Process(
            target=_send_parallel_request_task1,
            args=(url, IMAGE_MAN_IRONING_URL),
        )
        p2 = mp.Process(
            target=_send_parallel_request_task1,
            args=(url, IMAGE_MAN_IRONING_URL),
        )

        p1.start()
        p2.start()
        p1.join()
        p2.join()

        self.assertEqual(p1.exitcode, 0)
        self.assertEqual(p2.exitcode, 0)

    def test_vlm(self):
        self._run_multi_turn_request()
        self._run_multi_turn_request1()
        self._run_parallel_two_requests()

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
