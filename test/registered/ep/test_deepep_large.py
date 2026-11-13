import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

# The rerun workflow executes this file directly from ``test/``.  Python then
# puts only ``test/registered/ep`` on sys.path, so add the test root before
# importing the manual modules that provide the CI entry-point wrappers.
TEST_ROOT = Path(__file__).resolve().parents[2]
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from manual.ep.test_elastic_scale import (
    TestElasticScaleUp4To6 as _TestElasticScaleUp4To6,
)
from manual.ep.test_nixl_ep import (
    TestNixlEPTP as _TestNixlEPTP,
)

register_cuda_ci(est_time=1050, stage="base-c", runner_config="8-gpu-h200")

DEEPSEEK_V32_MODEL_PATH = "deepseek-ai/DeepSeek-V3.2"


class TestDeepseekMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_in_ci():
            raise unittest.SkipTest("DeepEP large tests are disabled in CI")
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--disable-overlap-schedule",
                "--trust-remote-code",
                "--tp",
                "8",
                "--attn-dp-size",
                "8",
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--moe-a2a-backend",
                "deepep",
                "--moe-runner-backend",
                "deep_gemm",
                "--enable-two-batch-overlap",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs-decode",
                "64",  # TODO: increase it to 128 when TBO is supported in draft_extend
                "--max-running-requests",
                "512",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--disable-radix-cache",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}',
            ],
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_sgl_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.92)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


class TestDeepseekV32TBO(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if is_in_ci():
            raise unittest.SkipTest("DeepEP large tests are disabled in CI")
        cls.model = DEEPSEEK_V32_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--attn-dp-size",
            "8",
            "--enable-two-batch-overlap",
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-max-bs-decode",
            "256",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        terminate_and_kill_process_tree(cls.process)

    def test_a_gsm8k(
        self,
    ):  # Append an "a" to make this test run first (alphabetically) to warm up the server
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_sgl_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.92)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")


class TestNixlEPTP(_TestNixlEPTP):
    """CI entry point for the registered NIXL EP test."""


@unittest.skip("Elastic EP scale test is temporarily disabled in registered CI")
class TestElasticScaleUp4To6(_TestElasticScaleUp4To6):
    """CI entry point for the registered Elastic EP scale test."""

    # Scaling mutates the live service from EP4 to EP6 and cannot be replayed
    # against the same process after a post-scale assertion failure.
    expect_server_exit = True


if __name__ == "__main__":
    unittest.main()
