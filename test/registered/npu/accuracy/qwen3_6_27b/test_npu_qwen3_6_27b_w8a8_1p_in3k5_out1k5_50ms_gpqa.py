import logging
import unittest

import requests

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_6_27B_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.send_one import BenchArgs, send_one_prompt

register_npu_ci(
    est_time=8400,
    suite="tmp-acc-2-npu-a3",
    nightly=True,
)

logger = logging.getLogger(__name__)

# Fail fast when the NEXTN draft model stops being accepted. With an acceptance
# length of 1.0 (no draft token accepted) decoding is several times slower and
# this 198-question GPQA run cannot finish within the nightly budget, it only
# ends in a harness timeout after 3.5 hours.
MIN_SPEC_ACCEPT_LENGTH = 1.5
SPEC_PRE_CHECK_ATTEMPTS = 3
SPEC_PRE_CHECK_MAX_TOKENS = 256

QWEN3_6_27B_3K5_1K5_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "130",
    "ASCEND_USE_FIA": "1",
}

QWEN3_6_27B_3K5_1K5_OTHER_ARGS = [
    "--tp-size",
    2,
    "--nnodes",
    1,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    60000,
    "--disable-radix-cache",
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--max-mamba-cache-size",
    74,
    "--mem-fraction-static",
    0.7,
    "--cuda-graph-bs-decode",
    2,
    8,
    16,
    32,
    40,
    45,
    50,
    54,
    "--enable-multimodal",
    "--quantization",
    "modelslim",
    "--mm-attention-backend",
    "ascend_attn",
    "--dtype",
    "bfloat16",
    "--mamba-ssm-dtype",
    "bfloat16",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--reasoning-parser",
    "qwen3",
    "--tool-call-parser",
    "qwen3_coder",
]


class TestNPUQwen3_6_27B_1P_In3k5_Out1k5_gpqa(TestNpuAccuracyTestCaseBase):
    model = QWEN3_6_27B_W8A8_MODEL_PATH
    envs = QWEN3_6_27B_3K5_1K5_ENVS
    other_args = QWEN3_6_27B_3K5_1K5_OTHER_ARGS
    max_retries = 2
    accuracy = 0.878
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    eval_batch_size = 16
    generation_config = {
        "max_tokens": 81920,
        "temperature": 0,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }

    def _check_spec_accept_length(self):
        """Abort early when speculative decoding acceptance has collapsed.

        A draft model whose tokens are almost always rejected slows decoding
        down several times, so the GPQA evaluation would spend the whole job
        budget without ever reaching the accuracy assertion. ``send_one_prompt``
        warms the server up as a side effect (``avg_spec_accept_length`` only
        exists once a decode step has run) and reports
        ``completion_tokens / spec_verify_ct`` for the same request.
        """
        args = BenchArgs(
            base_url=self.base_url, max_new_tokens=SPEC_PRE_CHECK_MAX_TOKENS
        )
        acc_length = 0.0
        for attempt in range(1, SPEC_PRE_CHECK_ATTEMPTS + 1):
            requests.post(self.base_url + "/flush_cache", timeout=60).raise_for_status()
            acc_length, speed = send_one_prompt(
                args, label=f"spec pre-check attempt {attempt}", print_output=False
            )
            logger.info(
                "spec pre-check attempt %d: acc_length=%.3f, speed=%.2f token/s",
                attempt,
                acc_length,
                speed,
            )
            if acc_length >= MIN_SPEC_ACCEPT_LENGTH:
                break

        self.assertGreaterEqual(
            acc_length,
            MIN_SPEC_ACCEPT_LENGTH,
            f"Speculative decoding acceptance too low: acc_length="
            f"{acc_length:.3f} < {MIN_SPEC_ACCEPT_LENGTH} after "
            f"{SPEC_PRE_CHECK_ATTEMPTS} attempts. Draft tokens are mostly "
            "rejected, decoding is several times slower and this GPQA "
            "evaluation cannot finish within the time budget.",
        )

    def test_accuracy(self):
        self._check_spec_accept_length()
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
