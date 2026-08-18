import os
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_xpu, kill_process_tree
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cuda_ci,
    register_xpu_ci,
)
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=4800,
    suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35-triton-dcp",
    nightly=True,
)
register_cuda_ci(est_time=4800, stage="nightly", runner_config="4-gpu-b200")
register_xpu_ci(est_time=1200, suite="nightly-xpu-4-gpu", nightly=True)

_IS_XPU = is_xpu()

# The Triton DCP path is shared by CUDA, ROCm and Intel XPU, so the same accuracy
# gate runs on all three. Only the model, shape and extra server args differ:
#   - Qwen3.5-397B needs 4 B200 / 8 MI35x; XPU substitutes Qwen2.5-1.5B-Instruct.
#   - get_num_kv_heads shards KV over tp // dcp_size groups, so every rank of a
#     DCP group holds the same KV heads. tp=4/dcp=4 is the upstream shape;
#     Qwen2.5-1.5B (2 KV heads) is validated at tp=4/dcp=2 on XPU.
#   - A 1.5B model cannot reach 0.90; the XPU gate is sized to catch a broken
#     DCP merge (which collapses accuracy toward zero), not to certify quality.
if _IS_XPU:
    QWEN35_MODEL_PATH = os.environ.get(
        "QWEN3_5_MODEL_PATH", DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN
    )
    SERVER_LAUNCH_TIMEOUT = 1200
    TP_SIZE = 4
    DCP_SIZE = 2
    GSM8K_ACCURACY_THRESHOLD = 0.30
    GSM8K_NUM_EXAMPLES = 200
    _PLATFORM_ARGS = ["--device", "xpu", "--mem-fraction-static", "0.6"]
else:
    QWEN35_MODEL_PATH = os.environ.get(
        "QWEN3_5_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B-FP8"
    )
    SERVER_LAUNCH_TIMEOUT = 4800
    TP_SIZE = 4
    DCP_SIZE = 4
    GSM8K_ACCURACY_THRESHOLD = 0.90
    GSM8K_NUM_EXAMPLES = 1319
    _PLATFORM_ARGS = [
        "--context-length",
        "1048576",
        "--json-model-override-args",
        (
            '{"rope_scaling":{"rope_type":"yarn","factor":4.0,'
            '"original_max_position_embeddings":262144}}'
        ),
    ]


class TestQwen35TritonDCPGsm8k(CustomTestCase):
    """Qwen3.5 Triton DCP (TP4/DCP4) full GSM8K accuracy."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp",
            str(TP_SIZE),
            "--dcp-size",
            str(DCP_SIZE),
            "--attention-backend",
            "triton",
            "--disable-radix-cache",
            *_PLATFORM_ARGS,
        ]
        env = os.environ.copy()
        if torch.version.hip:
            env["SGLANG_USE_AITER"] = "1"
            env["HSA_NO_SCRATCH_RECLAIM"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=32,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (triton-dcp{DCP_SIZE})\n"
                f'{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
