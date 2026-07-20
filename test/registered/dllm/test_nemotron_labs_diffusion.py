from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

_MODEL = "nvidia/Nemotron-Labs-Diffusion-8B"
_CONFIG_DIR = "test/registered/dllm/configs"
_COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.9",
    "--max-running-requests",
    "4",
    "--attention-backend",
    "flashinfer",
    "--cuda-graph-bs",
    "1",
    "2",
    "3",
    "4",
    "--context-length",
    "4096",
]

_PROMPT = "In one sentence, explain why batching improves GPU inference throughput."
_SPEED_PROMPT = (
    "Human: Explain linear self-speculation for language model serving.\n\n"
    "Assistant:"
)


def _completion(base_url, model, max_new_tokens=96):
    response = requests.post(
        base_url + "/v1/completions",
        json={
            "model": model,
            "prompt": _PROMPT,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        },
        timeout=180,
    )
    if response.status_code != 200:
        raise AssertionError(response.text)
    body = response.json()
    completion_tokens = body.get("usage", {}).get("completion_tokens", 0)
    text = body["choices"][0]["text"]
    return completion_tokens, text


class NemotronLabsDiffusionTestBase:
    algorithm = None
    config_name = None
    min_speed = None
    summary_name = None

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_SERVER_ARGS
            + [
                "--dllm-algorithm",
                cls.algorithm,
                "--dllm-algorithm-config",
                f"{_CONFIG_DIR}/{cls.config_name}",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generates_completion(self):
        completion_tokens, text = _completion(self.base_url, self.model)
        self.assertGreater(completion_tokens, 0, "model produced zero tokens")
        self.assertIsInstance(text, str)
        self.assertGreater(len(text.strip()), 0)

    def test_bs_1_speed(self):
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            max_new_tokens=512,
            prompt=_SPEED_PROMPT,
        )
        _acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed ({self.summary_name}) tp=1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, self.min_speed)


class TestNemotronLabsDiffusionFastDiffuser(
    NemotronLabsDiffusionTestBase, CustomTestCase
):
    algorithm = "FastDiffuser"
    config_name = "nemotron_labs_fastdiffuser.yaml"
    min_speed = 30
    summary_name = "nemotron-labs-diffusion-8b-fastdiffuser"


if __name__ == "__main__":
    unittest.main()
