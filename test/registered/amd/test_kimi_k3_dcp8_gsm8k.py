"""Kimi-K3 aiter DCP8 end-to-end accuracy on AMD MI35x (gfx950).

K3 runs DCP on aiter's MLA kernel, which tiles the query heads and so
serves K3's 96 gathered heads at tp8 dcp8. This test pins the end state:
tp8 + dcp8 on aiter must match the non-DCP baseline.

Two separate failure modes are covered, and the second one is why this file
does not consist of an accuracy check alone:

* ``test_a_dcp_config_applied`` -- the K3 override in ``arg_groups/overrides.py``
  silently picks page_size 32, aiter prefill/decode and the ag_rs merge. If any
  of that stops being applied, the server still starts and still answers
  correctly, just without DCP. A pure accuracy assertion cannot see that; it
  would go green while the feature under test is switched off.
* ``test_gsm8k`` -- accuracy itself.

Note on the threshold: this configuration measures 0.958 here (and 0.947-0.955
across earlier sessions with a different gsm8k harness, so the two agree). The
binomial standard error is +/-0.6 pt at n=1319 and the run-to-run sd is 0.68 pt
from base-model nondeterminism, which is present at W=1 too. 0.90 therefore
sits ~5.8 pt below the observed score and far outside the noise band -- it is a
"DCP is broken" tripwire, not a fine-grained regression detector.

"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=900, suite="stage-c-test-large-8-gpu-amd-mi35x")

KIMI_K3_MODEL_PATH = os.environ.get("KIMI_K3_MODEL_PATH", "moonshotai/Kimi-K3")
SERVER_LAUNCH_TIMEOUT = 1800
TP_SIZE = 8
DCP_SIZE = 8
# The K3 DCP override picks these; they are asserted rather than passed so the
# test fails if the override stops firing.
EXPECTED_PAGE_SIZE = 32
EXPECTED_DECODE_BACKEND = "aiter"
GSM8K_ACCURACY_THRESHOLD = 0.90
# 8192 would trip aiter's `context_len > 8192` derate of mem-fraction (x0.85),
# which for K3 lands under its ~0.735 floor and yields a negative pool budget.
CONTEXT_LENGTH = 5200


class TestKimiK3Dcp8Gsm8k(CustomTestCase):
    """Kimi-K3 aiter DCP (tp=8, dcp=8) GSM8K accuracy on AMD MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K3_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--trust-remote-code",
            "--tp",
            str(TP_SIZE),
            "--dcp-size",
            str(DCP_SIZE),
            "--attention-backend",
            "aiter",
            "--dtype",
            "bfloat16",
            "--context-length",
            str(CONTEXT_LENGTH),
            "--mem-fraction-static",
            "0.85",
            "--mamba-full-memory-ratio",
            "1.95",
            "--disable-radix-cache",
            "--reasoning-parser",
            "kimi_k3",
            "--tool-call-parser",
            "kimi_k3",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_AITER_K3_OPT"] = "1"
        env["AITER_FLYDSL_FORCE"] = "1"
        # K3's MoE is MXFP4; a8w4 (fp8 activations) is the validated combination
        # on this stack.
        env["AITER_SITUV2_A8W4"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_a_dcp_config_applied(self):
        """The DCP recipe must actually be in effect, not silently fallen back.

        /server_info reports the RESOLVED config, i.e. after the Kimi-K3
        override has run, so this sees what the engine is really using.
        """
        response = requests.get(self.base_url + "/server_info", timeout=30)
        response.raise_for_status()
        info = response.json()

        self.assertEqual(info["dcp_size"], DCP_SIZE, "DCP is not enabled")
        self.assertEqual(
            info["page_size"],
            EXPECTED_PAGE_SIZE,
            "the K3 DCP override no longer sets page_size; the kernel derives its KV "
            "tile from the paged block size, so a smaller page silently costs "
            "an order of magnitude on the decode step",
        )
        decode_backend = info.get("decode_attention_backend") or info.get(
            "attention_backend"
        )
        self.assertEqual(
            decode_backend,
            EXPECTED_DECODE_BACKEND,
            "decode is not on aiter, so the DCP path under test is not "
            "the one being exercised",
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=1024,
            num_examples=1319,
            num_threads=32,
            num_shots=5,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (kimi-k3-aiter-dcp8)\n" f'{metrics["score"]=:.3f}\n'
            )
        self.assertGreater(metrics["score"], GSM8K_ACCURACY_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
