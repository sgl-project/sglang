"""DP attention and decoder SWA replay for V4.1, with and without DSpark.

Set DSV41_MODEL_PATH to the checkpoint available on the test runner.
"""

import base64
import io
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests
from PIL import Image

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.basic_scheduler_stress_kit import BasicSchedulerStressMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="nightly", runner_config="4-gpu-gb300")


class _DSV41Server(CustomTestCase):
    dspark = False
    extra_args = ()
    process = None
    base_url = DEFAULT_URL_FOR_TEST
    sanity_max_new_tokens_short = 64
    _decode_generate = BasicDecodeCorrectnessMixin._decode_generate

    @classmethod
    def setUpClass(cls):
        cls.model = os.environ.get("DSV41_MODEL_PATH")
        if not cls.model:
            raise unittest.SkipTest("Set DSV41_MODEL_PATH to the V4.1 checkpoint")
        args = [
            "--trust-remote-code",
            "--tp",
            "4",
            "--ep-size",
            "4",
            "--attention-backend",
            "dsv4",
            "--moe-runner-backend",
            "flashinfer_mxfp4",
            "--mem-fraction-static",
            "0.80",
            "--chunked-prefill-size",
            "4096",
            "--context-length",
            "16384",
            "--max-running-requests",
            "8",
            "--cuda-graph-backend-prefill",
            "disabled",
            "--cuda-graph-max-bs",
            "8",
            "--random-seed",
            "0",
        ]
        if cls.dspark:
            args += [
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-dspark-block-size",
                "5",
                "--enable-decoder-swa-bounded-replay",
            ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[*args, *cls.extra_args],
        )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)
            cls.process = None


class TestDSV41BoundedReplayDP(_DSV41Server):
    test_generation = BasicDecodeCorrectnessMixin.test_capital_france
    test_streaming = BasicSchedulerStressMixin.test_streaming_response

    def test_image_input(self):
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), "red").save(buffer, format="PNG")
        image = base64.b64encode(buffer.getvalue()).decode("ascii")
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
                                "image_url": {"url": f"data:image/png;base64,{image}"},
                            },
                            {
                                "type": "text",
                                "text": "What color is the image? Reply with one color word.",
                            },
                        ],
                    }
                ],
                "chat_template_kwargs": {"thinking": False},
                "temperature": 0,
                "max_tokens": 32,
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        text = response.json()["choices"][0]["message"]["content"]
        self.assertIn("red", text.lower())

    # TP4/DP2 also exercises tail lengths that need attention-TP padding.
    extra_args = (
        "--enable-dp-attention",
        "--dp-size",
        "2",
        "--enable-decoder-swa-bounded-replay",
    )

    def test_uneven_prefills_and_idle_rank(self):
        def generate(rank, repetitions):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "text": (
                        "The capital of France is Paris.\n" * repetitions
                        + "What is the capital of France? Answer with the city name only:"
                    ),
                    "routed_dp_rank": rank,
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                },
                timeout=180,
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertIn("Paris", response.json()["text"])

        # Force an empty peer first, then uneven concurrent prefills (including
        # chunked prefill) whose decode steps may overlap a peer's prefill.
        generate(0, 200)
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(generate, rank, repetitions)
                for rank, repetitions in ((0, 600), (1, 30), (0, 70), (1, 200))
            ]
            for future in futures:
                future.result()


class TestDSV41BoundedReplayDPDSpark(TestDSV41BoundedReplayDP):
    dspark = True
    extra_args = (*TestDSV41BoundedReplayDP.extra_args, "--enable-dp-lm-head")

    def test_generation_with_speculation(self):
        BasicDecodeCorrectnessMixin.test_capital_france(self)
        output = self._decode_generate("List the numbers from 1 to 20:", 64)
        self.assertTrue(output.strip())
        response = requests.get(self.base_url + "/server_info", timeout=30)
        self.assertEqual(response.status_code, 200, response.text)
        accept_length = response.json()["internal_states"][0]["avg_spec_accept_length"]
        self.assertGreater(accept_length, 1.0)


if __name__ == "__main__":
    unittest.main()
