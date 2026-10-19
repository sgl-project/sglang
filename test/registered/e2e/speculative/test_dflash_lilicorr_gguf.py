import base64
import io
import unittest

import requests
from PIL import Image, ImageDraw

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=600, stage="extra-b", runner_config="1-gpu-large")


class TestLiLiCorrGGUFServer(CustomTestCase, GSM8KMixin):
    model = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
    draft_model = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF"
    gsm8k_accuracy_thres = 0.80
    gsm8k_num_threads = 4
    gsm8k_temperature = 0.0

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "1",
                "--context-length",
                "4096",
                "--max-running-requests",
                "4",
                "--max-total-tokens",
                "8192",
                "--max-mamba-cache-size",
                "32",
                "--mem-fraction-static",
                "0.6",
                "--speculative-algorithm",
                "DFLASH",
                "--speculative-draft-model-path",
                cls.draft_model,
                "--speculative-dflash-block-size",
                "8",
                "--speculative-draft-attention-backend",
                "triton",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        GSM8KMixin.test_gsm8k(self)
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        state = response.json()["internal_states"][0]
        self.assertGreater(state["avg_spec_accept_length"], 2.0)

    def test_image_inputs(self):
        for count in (3, 5):
            with self.subTest(count=count):
                image = Image.new("RGB", (640, 256), "white")
                draw = ImageDraw.Draw(image)
                for index in range(count):
                    x = 64 + 128 * index
                    draw.ellipse((x - 35, 93, x + 35, 163), fill="green")
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                encoded = base64.b64encode(buffer.getvalue()).decode()
                response = requests.post(
                    self.base_url + "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": "data:image/png;base64," + encoded
                                        },
                                    },
                                    {
                                        "type": "text",
                                        "text": "How many green circles are visible? Reply with only the number.",
                                    },
                                ],
                            }
                        ],
                        "temperature": 0,
                        "max_tokens": 16,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=300,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                self.assertEqual(content.strip(), str(count))


if __name__ == "__main__":
    unittest.main()
