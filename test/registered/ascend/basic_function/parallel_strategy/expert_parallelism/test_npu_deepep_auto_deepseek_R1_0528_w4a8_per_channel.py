import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDeepEpAutoDeepseekR1(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify the accuracy of DeepSeek-R1 model on MMLU and GSM8K tasks with --deepep-mode auto on Ascend backend.

    [Test Category] Parameter
    [Test Target] --moe-a2a-backend; --deepep-mode
    """

    model = DEEPSEEK_R1_0528_W4A8_PER_CHANNEL_WEIGHTS_PATH
    accuracy = 0.96
    other_args = [
        "--tp",
        "16",
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--device",
        "npu",
        "--quantization",
        "modelslim",
        "--watchdog-timeout",
        "9000",
        "--cuda-graph-bs",
        "4",
        "8",
        "20",
        "21",
        "22",
        "--mem-fraction-static",
        "0.85",
        "--max-total-tokens",
        "20480",
        "--max-running-requests",
        "352",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        "-1",
        "--max-prefill-tokens",
        "1500",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--enable-dp-attention",
        "--dp-size",
        "16",
        "--enable-dp-lm-head",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "2",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "3",
        "--dtype",
        "bfloat16",
    ]
    env = {
        **os.environ,
        "SGLANG_SET_CPU_AFFINITY": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "STREAMS_PER_DEVICE": "32",
        "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
        "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "88",
        "HCCL_BUFFSIZE": "1600",
        "DEEPEP_HCCL_BUFFSIZE": "1600",
        "DEEPEP_NORMAL_LONG_SEQ_ROUND": "10",
        "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "512",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
        "SGLANG_NPU_USE_MLAPO": "1",
        "SGLANG_USE_FIA_NZ": "1",
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    }

    def test_mmlu(self):
        # Test Scenario: Verify the model's performance on MMLU dataset (general knowledge evaluation)
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        # Execute MMLU evaluation and get metrics
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
