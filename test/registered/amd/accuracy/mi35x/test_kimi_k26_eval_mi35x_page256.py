"""MI35x Kimi-K2.6 GSM8K Eval (8-GPU) — DIAGNOSTIC: single-node with page_size=256.

Control for the Kimi-K2.6 non-MTP disagg GSM8K drop (~0.88 vs single-node
0.944). The only attention-affecting arg the disagg recipe adds over the passing
single-node eval and that has not been isolated at single-node scale is
`--page-size 256` (disagg page_size 1 vs 256 were both ~0.88, but single-node has
only ever run at the default page size). This test is byte-identical to
test_kimi_k26_eval_mi35x.py plus `--page-size 256`:

  * if this ALSO drops to ~0.88 -> the bug is page_size=256 in the (absorbed-MLA)
    decode path, NOT disaggregation.
  * if this stays ~0.94 -> confirms the drop is disagg-specific.

Registered into the same suite (nightly-amd-accuracy-8-gpu-mi35x-kimi-k26) so the
existing MI35x Kimi-K2.6 nightly job runs default + page256 back-to-back on the
same image/base for a clean A/B.
"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(
    est_time=5400, suite="nightly-amd-accuracy-8-gpu-mi35x-kimi-k26", nightly=True
)

KIMI_K26_MODEL_PATH = "moonshotai/Kimi-K2.6"
SERVER_LAUNCH_TIMEOUT = 5400
ACCURACY_THRESHOLD = 0.92
TP_SIZE = 8
PAGE_SIZE = 256


class TestKimiK26EvalMI35xPage256(CustomTestCase):
    """Kimi-K2.6 GSM8K eval on MI35x with page_size=256 (single-node control)."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_kimi_k26_gsm8k_accuracy_page256(self):
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--decode-attention-backend",
            "triton",
            "--prefill-attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
            # The one variable under test.
            "--page-size",
            str(PAGE_SIZE),
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_ROCM_FUSED_DECODE_MLA"] = "0"

        process = popen_launch_server(
            KIMI_K26_MODEL_PATH,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
            requests.get(self.base_url + "/flush_cache")

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
            acc = metrics["accuracy"]

            passed = acc >= ACCURACY_THRESHOLD
            status = "✅ PASS" if passed else "❌ FAIL"
            print(
                f"  [page_size={PAGE_SIZE}] accuracy={acc:.3f} "
                f"threshold={ACCURACY_THRESHOLD} {status}"
            )

            if is_in_ci():
                summary = "### Kimi-K2.6 Model (MI35x, page_size=256 control)\n\n"
                summary += "| Model | TP | page_size | Accuracy | Threshold | Status |\n"
                summary += "| ----- | -- | --------- | -------- | --------- | ------ |\n"
                summary += (
                    f"| {KIMI_K26_MODEL_PATH} | {TP_SIZE} | {PAGE_SIZE} | "
                    f"{acc:.3f} | {ACCURACY_THRESHOLD} | {status} |\n"
                )
                write_github_step_summary(summary)

            self.assertGreaterEqual(
                acc,
                ACCURACY_THRESHOLD,
                f"Kimi-K2.6 (page_size={PAGE_SIZE}) accuracy {acc:.3f} "
                f"below threshold {ACCURACY_THRESHOLD}",
            )
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
