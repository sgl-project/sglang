import base64
import io
import unittest
from types import SimpleNamespace

import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=250, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=200, suite="stage-b-test-1-gpu-large-amd")


class TestAWQ(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_AWQ_MOE_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--trust-remote-code"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=32,
        )

        metrics = run_sgl_eval(args)
        self.assertGreater(metrics["score"], 0.64)


@unittest.skipIf(is_in_amd_ci(), "AWQ Marlin is not supported on AMD GPUs")
class TestAWQMarlinBfloat16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--dtype", "bfloat16", "--quantization", "awq_marlin"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=32,
        )

        metrics = run_sgl_eval(args)
        self.assertGreater(metrics["score"], 0.80)


@unittest.skipIf(is_in_amd_ci(), "AWQ Marlin is not supported on AMD GPUs")
class TestSenseNovaLoopAWQ(DefaultServerBase):
    model = "sensenova/SenseNova-Loop-9B-AWQ"
    other_args = ["--context-length", "4096", "--cuda-graph-max-bs-decode", "8"]

    def test_text_and_image(self):
        image = io.BytesIO()
        Image.new("RGB", (128, 128), "red").save(image, format="PNG")
        image_url = (
            "data:image/png;base64," + base64.b64encode(image.getvalue()).decode()
        )
        for content, expected in (
            ("What is 6 times 7? Answer with just the number.", "42"),
            (
                [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {
                        "type": "text",
                        "text": "What color is this image? Answer with one word.",
                    },
                ],
                "red",
            ),
        ):
            with self.subTest(expected=expected):
                response = requests.post(
                    self.base_url + "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": content}],
                        "temperature": 0,
                        "max_tokens": 32,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=60,
                )
                response.raise_for_status()
                answer = response.json()["choices"][0]["message"]["content"]
                self.assertIn(expected, answer.lower())


if __name__ == "__main__":
    unittest.main()
