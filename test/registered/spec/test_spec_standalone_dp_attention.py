import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_STANDALONE_DP_ATTN,
    DEFAULT_TARGET_MODEL_STANDALONE_DP_ATTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    kill_process_tree,
    popen_launch_server,
    write_github_step_summary,
)

# STANDALONE with DP attention (tp=4, dp=4, requires 4 GPUs).
register_cuda_ci(est_time=120, stage="extra-b", runner_config="4-gpu-h100")
register_amd_ci(est_time=200, suite="stage-c-test-4-gpu-amd")

# Measured 0.925 (dp) / 0.93 (non-dp) on Qwen3-30B-A3B.
GSM8K_ACCURACY_FLOOR = 0.90
# Measured 3.95 on GSM8K; margin for fa3/triton drift.
SPEC_ACCEPT_LENGTH_FLOOR = 3.5


class TestStandaloneSpecDPAttention(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_TARGET_MODEL_STANDALONE_DP_ATTN
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--speculative-algorithm",
            "STANDALONE",
            "--speculative-draft-model-path",
            DEFAULT_DRAFT_MODEL_STANDALONE_DP_ATTN,
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--tp-size",
            "4",
            "--dp-size",
            "4",
            "--enable-dp-attention",
            "--attention-backend",
            "triton" if is_in_amd_ci() else "fa3",
            "--mem-fraction-static",
            "0.75",
            "--cuda-graph-max-bs-decode",
            "64",
        ]
        with envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1):
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
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (STANDALONE DP Attention)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], GSM8K_ACCURACY_FLOOR)
            self.assertGreater(avg_spec_accept_length, SPEC_ACCEPT_LENGTH_FLOOR)

    def test_b_bs1_no_hang(self):
        """Single-request decode is the idle-rank scenario the capture-mode fix
        repairs: with dp attention the other DP ranks go idle, and before the
        fix they fell to eager while the active rank replayed the cuda graph,
        mismatching the MoE all-gather and hanging NCCL. If that regresses this
        test hangs and times out.
        """
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=256)
        acc_length, speed = send_one_prompt(args)
        print(f"{acc_length=:.2f} {speed=:.2f} token/s")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1 (STANDALONE DP Attention)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            # Speed is printed for visibility, not asserted (hardware-dependent).
            self.assertGreater(acc_length, 1.0)


if __name__ == "__main__":
    unittest.main()
