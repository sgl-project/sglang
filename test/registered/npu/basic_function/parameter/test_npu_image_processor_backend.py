import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    IMAGES_EXAMPLE_PATH,
    KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-4-npu-a3", nightly=True)


class TestNpuImageProcessorBackend(CustomTestCase):
    """Testcase：验证 --image-processor-backend 参数在 NPU 上生效，auto/torchvision/pil 三种后端均可正常完成 VLM 图片推理。

    [Test Category] Parameter
    [Test Target] --image-processor-backend
    """

    model = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
    image_url = IMAGES_EXAMPLE_PATH

    def _launch_and_infer(self, backend):
        process = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--mem-fraction-static",
                "0.35",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--image-processor-backend",
                backend,
            ],
        )
        try:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": self.image_url},
                                },
                                {"type": "text", "text": "What is in this image?"},
                            ],
                        }
                    ],
                    "temperature": 0,
                    "max_tokens": 16,
                },
            )
            response.raise_for_status()
            body = response.json()
            self.assertTrue(body["choices"][0]["message"]["content"])

            # image_tokens 来自 prefill 的多模态 item offsets，非零说明图片到达了 vision tower。
            usage_details = body["usage"].get("prompt_tokens_details")
            self.assertIsNotNone(usage_details, "prompt carried no multimodal tokens")
            self.assertGreater(usage_details.get("image_tokens", 0), 0)
        finally:
            kill_process_tree(process.pid)

    def test_image_processor_backend_auto(self):
        self._launch_and_infer("auto")

    def test_image_processor_backend_torchvision(self):
        self._launch_and_infer("torchvision")

    def test_image_processor_backend_pil(self):
        self._launch_and_infer("pil")


if __name__ == "__main__":
    unittest.main()