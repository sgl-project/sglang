import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPSEEK_NVFP4_MODEL_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)


def test_gsm8k(base_url: str):
    requests.get(base_url + "/flush_cache")

    args = SimpleNamespace(
        num_shots=5,
        data_path=None,
        num_questions=200,
        max_new_tokens=512,
        parallel=128,
        host="http://127.0.0.1",
        port=int(base_url.split(":")[-1]),
    )
    metrics = run_eval_few_shot_gsm8k(args)
    server_info = requests.get(base_url + "/get_server_info")
    avg_spec_accept_length = server_info.json()["internal_states"][0][
        "avg_spec_accept_length"
    ]

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
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
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
        metrics, avg_spec_accept_length = test_gsm8k(self.base_url)

        self.assertGreater(metrics["accuracy"], 0.94)
        # TODO: Update accept len to 2.04 once the bug is fixed
        self.assertGreater(avg_spec_accept_length, 1.4)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-fp4 mtp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )


if __name__ == "__main__":
    unittest.main()
