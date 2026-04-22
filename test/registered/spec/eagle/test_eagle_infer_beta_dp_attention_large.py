import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

# 16 GPU test (4 TP x 4 DP), runs on 2x 8-GPU B200 nodes
register_cuda_ci(est_time=600, suite="nightly-8-gpu-b200", nightly=True)


def test_gsm8k(base_url: str, model: str):
    requests.get(base_url + "/flush_cache")

    args = SimpleNamespace(
        base_url=base_url,
        model=model,
        eval_name="gsm8k",
        api="completion",
        max_tokens=512,
        num_examples=200,
        num_threads=128,
    )
    metrics = run_eval(args)
    server_info = requests.get(base_url + "/server_info").json()
    avg_spec_accept_length = server_info["internal_states"][0]["avg_spec_accept_length"]

    print(f"{metrics=}")
    print(f"{avg_spec_accept_length=}")
    return metrics, avg_spec_accept_length


class TestEagleDPAttnServerLarge(CustomTestCase):
    # FIXME: move this large mode test into nightly tests
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp-size",
            "4",
            "--dp-size",
            "4",
            "--enable-dp-attention",
            "--attention-backend",
            "trtllm_mla",
            "--moe-runner-backend",
            "flashinfer_trtllm",
            "--quantization",
            "modelopt_fp4",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true,"num_threads": 64}',
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(
            True
        ), envs.SGLANG_SPEC_NAN_DETECTION.override(
            True
        ), envs.SGLANG_SPEC_OOB_DETECTION.override(
            True
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        metrics, avg_spec_accept_length = test_gsm8k(self.base_url, self.model)

        self.assertGreater(metrics["score"], 0.94)
        self.assertGreater(avg_spec_accept_length, 2.7)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4 mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )


if __name__ == "__main__":
    unittest.main()
