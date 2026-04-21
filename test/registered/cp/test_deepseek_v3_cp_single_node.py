"""MLA prefill-CP accuracy gate on real ``DeepSeek-V3-0324`` weights.

MLA counterpart of ``test_deepseek_v32_cp_single_node.py`` (NSA CP)
and ``test_qwen3_30b.py`` (MHA CP) — one e2e file per CP flavor.
Cheaper helper-level coverage lives in ``test_cp_utils.py`` and
``test_mla_cp_fa3_parity.py``.

Topology: ``tp=8, dp=2, attn-cp=4``, FA3 backend — the same
``attn_cp x dp`` layout NSA CP uses for DSv3.2. No MTP in this
baseline (MTP x MLA CP is a follow-up).

GSM8k threshold 0.935 matches the non-CP DSv3 baseline in
``test_deepseek_v3_basic.py`` / ``test_deepseek_v3_mtp.py``.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=500, suite="stage-c-test-deepep-8-gpu-h200")

DEEPSEEK_V3_MODEL_PATH = "deepseek-ai/DeepSeek-V3-0324"

# Matches the non-CP DSv3 production baseline in
# ``test_deepseek_v3_basic.py`` / ``test_deepseek_v3_mtp.py``. Pinning
# MLA CP to the same threshold makes this test double as a regression
# gate against the known production accuracy.
GSM8K_ACCURACY_THRESHOLD = 0.935


class TestDeepseekV3CPInSeqSplit(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "8",
            "--enable-dp-attention",
            "--dp",
            "2",
            "--attn-cp-size",
            "4",
            "--enable-prefill-context-parallel",
            "--attention-backend",
            "fa3",
            "--mem-frac",
            "0.7",
            "--cuda-graph-max-bs",
            "32",
            "--max-running-requests",
            "32",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    # "test_a_" prefix pins alphabetical first-run ordering so this
    # warms up the server before any follow-up sibling test methods.
    def test_a_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=500,
            num_threads=32,
            num_shots=20,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_a_gsm8k (deepseek-v3-mla-cp-in-seq-split)\n"
                f'{metrics["score"]=:.3f}\n'
            )
            self.assertGreater(metrics["score"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
