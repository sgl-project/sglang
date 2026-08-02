"""MI35x Qwen 3.5 GSM8K lm-eval Evaluation Tests (8-GPU)

Tests Qwen/Qwen3.5-397B-A17B (MoE, Hybrid Attention with Gated Delta Networks)
with lm-eval GSM8K benchmark on MI35x, matching the AMD Day 0 article, plus the
MXFP4-AttnFP8 checkpoint run with the fused AR+RMSNorm+quant path enabled.

Registry: nightly-amd-accuracy-8-gpu-mi35x-qwen35 suite
"""

import os
import unittest
from pathlib import Path

import numpy as np
import requests
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

register_amd_ci(
    est_time=7200, suite="nightly-amd-accuracy-8-gpu-mi35x-qwen35", nightly=True
)

QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"
QWEN35_MXFP4_ATTNFP8_MODEL_PATH = "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8"
SERVER_LAUNCH_TIMEOUT = 3600
TP_SIZE = 8


class TestQwen35EvalMI35x(LMEvalMixin, CustomTestCase):
    """Qwen 3.5 GSM8K lm-eval Test for AMD MI35x."""

    model_config_name = "lm_eval_configs/Qwen3.5-397B-A17B.yaml"
    model_path = QWEN35_MODEL_PATH
    tp_size = TP_SIZE
    extra_args: list[str] = []
    extra_env: dict[str, str] = {}

    @classmethod
    def setUpClass(cls):
        cls.model = cls.model_path
        cls.base_url = DEFAULT_URL_FOR_TEST

    def test_lm_eval(self):
        """Override to handle server lifecycle and write results to summary."""
        other_args = [
            "--tp",
            str(self.tp_size),
            "--attention-backend",
            "aiter",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
            "--watchdog-timeout",
            "1200",
            *self.extra_args,
        ]
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "1"
        env.update(self.extra_env)

        process = popen_launch_server(
            self.model_path,
            self.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

        try:
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
        finally:
            kill_process_tree(process.pid)


class TestQwen35MXFP4AttnFP8ARFusionMI35x(TestQwen35EvalMI35x):
    """Same eval with the fused AR+RMSNorm+quant path (PR #29723) enabled.

    `--enable-aiter-allreduce-fusion` folds the TP all-reduce, the following
    RMSNorm and the per-token quant of the next GEMM into one aiter kernel. This
    checkpoint drives both fused-quant formats: mxfp4 (MoE/GDN weights) and
    fp8_per_token (attention/GDN input projections, needs the env var below).
    The feature is default-off, so a plain green CI run never executes it.
    """

    model_config_name = "lm_eval_configs/Qwen3.5-397B-A17B-MXFP4-AttnFP8.yaml"
    model_path = QWEN35_MXFP4_ATTNFP8_MODEL_PATH
    # TP=8 shards shared_expert.down_proj (K=1024) to K=128, which aiter's fp8
    # GEMM has no tuned config for and crashes cuda-graph capture.
    tp_size = 4
    extra_args = [
        "--enable-aiter-allreduce-fusion",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--disable-radix-cache",
        "--mem-fraction-static",
        "0.85",
    ]
    extra_env = {"SGLANG_USE_AITER_FP8_PER_TOKEN": "1"}


if __name__ == "__main__":
    unittest.main()
