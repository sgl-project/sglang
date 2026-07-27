import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.run_eval import run_eval as run_ascend_eval
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestDeepEpDeepseek(GSM8KAscendMixin, CustomTestCase):
    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    accuracy = 0.34
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-cuda-graph",
        "--dp-size",
        "8",
        "--enable-dp-attention",
        "--chunked-prefill-size",
        "1024",
        "--mem-fraction-static",
        "0.7",
    ]
    env = {
        **os.environ,
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
        "HCCL_BUFFSIZE": "2048",
        "MOE_ENABLE_TOPK_NEG_ONE": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    }

    def test_mmlu(self):
        expect_score = 0.58
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
            api="completion",
            num_shots=5,
        )
        metrics = run_ascend_eval(args)
        self.assertGreater(metrics["score"], expect_score)


if __name__ == "__main__":
    unittest.main()
