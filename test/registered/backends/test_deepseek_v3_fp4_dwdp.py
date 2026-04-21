"""Backend test for DWDP (Distributed Weight Data Parallelism) on DeepSeek V3 NVFP4.

Exercises the DWDP path: ping-pong CUDA IPC prefetch of peer MoE expert
weights, multi-B CuteDSL kernel dispatch, DP-attention with per-rank
independent MoE compute. Enabled via ``--dwdp-size`` which forces:
  - dp_size = tp_size (DP attention on)
  - ep_size = dwdp_size
  - moe_runner_backend = flashinfer_cutedsl
  - moe_a2a_backend = none (tokens never cross ranks in MoE)

Requires 4 GPUs (SM100 / Blackwell). Run from repo root with:
  python -m pytest test/registered/backends/test_deepseek_v3_fp4_dwdp.py -v -s
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=900, suite="nightly-4-gpu-b200", nightly=True)

FULL_DEEPSEEK_V3_NVFP4_MODEL_PATH = "nvidia/DeepSeek-V3-0324-NVFP4"
SERVER_LAUNCH_TIMEOUT = 1000
GSM8K_ACCURACY_THRESHOLD = 0.935


class TestDeepseekV3NVFP4DWDP(CustomTestCase):
    """DWDP path: ping-pong IPC prefetch + multi-B CuteDSL, TP=DP=EP=4."""

    @classmethod
    def setUpClass(cls):
        cls.model = FULL_DEEPSEEK_V3_NVFP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "4",
            "--dwdp-size",
            "4",
            "--mem-fraction-static",
            "0.85",
            "--attention-backend",
            "trtllm_mla",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--moe-runner-backend",
            "flashinfer_cutedsl",
            "--quantization",
            "modelopt_fp4",
            "--disable-piecewise-cuda-graph",
            "--disable-cuda-graph",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (deepseek-v3-nvfp4-dwdp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
            )
        self.assertGreater(metrics["accuracy"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
