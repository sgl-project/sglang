import random
import unittest
import requests
import multiprocessing as mp

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

MODEL = QWEN3_VL_30B_A3B_INSTRUCT_WEIGHTS_PATH

# image
IMAGE_MAN_IRONING_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png"
IMAGE_SGL_LOGO_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png"

# video
VIDEO_JOBS_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/videos/jobs_presenting_ipod.mp4"


def popen_launch_server_wrapper(base_url, model, other_args):
    process = popen_launch_server(
        model,
        base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=other_args,
    )
    return process


class TestLimitMMDatePerRequest(CustomTestCase):
    """Testcase: Configuring '--limit-mm-data-per-request {"image":1, "video":1}' to send different multimodal inference requests,
       each containing multiple multimodal input data, with verfication ensuring that only one data point is processed at a time

    [Test Category] Parameter
    [Test Target] --limit-mm-data-per-request
    """

    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.base_url += "/v1"
        cls.api_key = "sk-123456"

        limit_mm = '{"image":1, "video":1}'
        other_args = [
            "--mem-fraction-static",
            "0.5",
            "--enable-multimodal",
            "--limit-mm-data-per-request",
            limit_mm,
            "--attention-backend",
            "ascend",
            "--device",
            "npu",
            "--tp-size",
            "16",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server_wrapper(
            DEFAULT_URL_FOR_TEST, MODEL, other_args
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
                        "type": "video_url",
                        "video_url": {"url": VIDEO_JOBS_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this video in a sentence.",
                    },
                ],
            },
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
        response = requests.post(self.base_url + '/chat/completions',
                                 json={"messages": messages, "temperature": 0, "max_completion_tokens": 1024})
        assert response.status_code == 200

    def _run_multi_turn_request1(self):
        # Input video and image
        messages1 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "vedio_url",
                        "vedio_url": {"url": VIDEO_JOBS_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this video in a sentence.",
                    },
                ],
            },
        ]
        response1 = requests.post(self.base_url + '/chat/completions',
                                  json={"messages": messages1, "temperature": 0, "max_completion_tokens": 1024})
        assert response1.status_code == 400

    def _run_multi_turn_request2(self):
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
        response2 = requests.post(self.base_url + '/chat/completions',
                                  json={"messages": messages2, "temperature": 0, "max_completion_tokens": 1024})
        assert response2.status_code == 400

    def test_vlm(self):
        self._run_multi_turn_request()
        self._run_multi_turn_request1()
        self._run_multi_turn_request2()


if __name__ == "__main__":
    unittest.main()
