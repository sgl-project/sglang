"""AMD Qwen 3.5 GSM8K lm-eval Evaluation Test (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B (MoE, Hybrid Attention with Gated Delta Networks)
with lm-eval GSM8K benchmark on MI325/MI300X, matching the AMD Day 0 article.

Registry: nightly-amd-accuracy-8-gpu-qwen35 suite
"""

import os
import unittest
from pathlib import Path

import numpy as np
import yaml

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.kits.lm_eval_kit import LMEvalMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=3600, suite="nightly-amd-accuracy-8-gpu-qwen35", nightly=True)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8


class TestQwen35EvalAMD(LMEvalMixin, CustomTestCase):
    """Qwen 3.5 GSM8K lm-eval Test for AMD MI325/MI300X."""

    model_config_name = "lm_eval_configs/Qwen3.5-397B-A17B.yaml"

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            str(TP_SIZE),
            "--attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
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

    def test_lm_eval(self):
        """Override to write accuracy results to GitHub step summary."""
        import requests

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
                status = "✅" if passed else "❌"
                summary += f"| {task['name']} | {metric['name']} | {expected:.4f} | {measured:.4f} | {status} |\n"
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
