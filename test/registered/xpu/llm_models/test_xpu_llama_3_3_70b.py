"""Llama-3.3-70B-Instruct GSM8K lm-eval Evaluation Test on Intel XPU (TP=8).

Mirrors the AMD lm-eval flow (see test/registered/amd/accuracy/mi30x/
test_qwen35_eval_amd.py): docker pull + install lm-eval, launch the SGLang
server, run lm_eval.simple_evaluate against /v1/completions, and validate
results against the YAML in test/lm_eval_configs/.
"""

import os
import unittest
from pathlib import Path

import numpy as np
import requests
import torch
import yaml

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_xpu_ci(est_time=2700, suite="nightly-xpu-8-gpu", nightly=True)

LLAMA_MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct"
# 70B + TP=8 + Intel XPU graph capture / warmup takes longer than the 600s
# default; mirror the AMD Qwen3.5 lm-eval test which uses a 1h budget.
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8


@unittest.skip("Not yet validated end-to-end on intel-bmg-nightly; re-enable once a passing run is recorded.")
@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestLlama3370BInstructXPU(LMEvalMixin, CustomTestCase):
    """Llama-3.3-70B-Instruct GSM8K lm-eval Test for Intel XPU."""

    model_config_name = "lm_eval_configs/Llama-3.3-70B-Instruct.yaml"

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--device",
            "xpu",
            "--attention-backend",
            "intel_xpu",
            "--dtype",
            "bfloat16",
            "--trust-remote-code",
            "--disable-overlap-schedule",
            "--disable-radix-cache",
            "--tp",
            str(TP_SIZE),
            "--max-total-tokens",
            "63356",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lm_eval(self):
        """Run lm-eval and write a markdown summary like the AMD flow does."""
        requests.get(self.base_url + "/flush_cache")

        eval_config = yaml.safe_load(
            Path(self.model_config_name).read_text(encoding="utf-8")
        )
        results = self.launch_lm_eval(eval_config)
        rtol = eval_config.get("rtol", self.default_rtol)
        model_name = eval_config.get("model_name", self.model)

        success = True
        summary = f"### lm-eval accuracy ({model_name})\n"
        summary += "| task | metric | expected | measured | status |\n"
        summary += "| ---- | ------ | -------- | -------- | ------ |\n"
        for task in eval_config["tasks"]:
            for metric in task["metrics"]:
                expected = metric["value"]
                measured = results["results"][task["name"]][metric["name"]]
                passed = bool(np.isclose(expected, measured, rtol=rtol))
                status = "PASS" if passed else "FAIL"
                summary += (
                    f"| {task['name']} | {metric['name']} | "
                    f"{expected:.4f} | {measured:.4f} | {status} |\n"
                )
                print(
                    f"{task['name']} | {metric['name']}: "
                    f"expected={expected:.3f} | measured={measured:.3f} | rtol={rtol}"
                )
                success = success and passed

        if is_in_ci():
            write_github_step_summary(summary)

        self.assertTrue(success, "lm-eval validation failed")


if __name__ == "__main__":
    unittest.main()
