"""Kimi-K2.5-MXFP4 aiter breakable CUDA-graph (BCG) capture accuracy test
(MI35x, PR-CI)

Exercises the AMD breakable (BCG) CUDA-graph prefill capture path on a
deepseek-family (Kimi-K2.5) aiter model so the code added in this PR actually
runs in PR CI:

  * runner_backend/breakable_cuda_graph_backend.py
        the ROCm/HIP hard block is removed -> BCG must capture on AMD.
  * models/deepseek_common/attention_backend_handler.py (handle_attention_aiter)
        during BCG capture the aiter prefill is routed through the MHA path
        (is_in_breakable_cuda_graph()).

These branches are gated and default-off, so the default full-cuda-graph CI
tests never touch them. This test launches the server with
``--cuda-graph-backend-prefill breakable`` (the config validated in the PR
description) and asserts GSM8K accuracy, turning "did the capture succeed and
stay correct?" into a CI signal.

Validated config (PR): Kimi-K2.5-MXFP4 TP4 MI35x, aiter prefill/decode, fp8 KV,
--enable-aiter-allreduce-fusion -> GSM8K ~0.93 for BCG.
"""

import os
import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List

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

# PR-CI AMD MI35x suite (same suite as the existing Kimi-K2.5-MXFP4 test), so
# the PR's changed BCG code path is exercised on every run-ci.
register_amd_ci(est_time=2100, suite="stage-c-test-large-8-gpu-amd-mi35x")

KIMI_K25_MXFP4_MODEL_PATH = "amd/Kimi-K2.5-MXFP4"
# Same pinned revision as test_kimi_k25_mxfp4.py: revisions from 94d8c1bd
# onward quantize shared_experts to MXFP4 so shared-experts fusion works.
KIMI_K25_MXFP4_REVISION = "419004c8716cf22c929aa15d39b85e09a8a2091a"
SERVER_LAUNCH_TIMEOUT = 3600
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
ACCURACY_THRESHOLD = 0.92


@dataclass
class CaptureConfig:
    """A prefill cuda-graph capture backend variant to validate."""

    variant: str
    # Extra server args that select the prefill capture backend.
    capture_args: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)


# Common args mirror the PR's validated launch command (TP4, aiter
# prefill/decode, fp8 KV, allreduce fusion, 8192 chunked prefill).
COMMON_ARGS: List[str] = [
    "--revision",
    KIMI_K25_MXFP4_REVISION,
    "--tensor-parallel-size",
    "4",
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.765",
    "--disable-radix-cache",
    "--prefill-attention-backend",
    "aiter",
    "--decode-attention-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--max-running-requests",
    "1024",
    "--enable-aiter-allreduce-fusion",
    "--chunked-prefill-size",
    "8192",
    "--max-prefill-tokens",
    "8192",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
]


def get_capture_configs() -> List[CaptureConfig]:
    return [
        # BCG: breakable prefill capture (the path this PR unblocks on ROCm).
        CaptureConfig(
            variant="bcg",
            capture_args=[
                "--cuda-graph-backend-prefill",
                "breakable",
                "--cuda-graph-backend-decode",
                "full",
            ],
        ),
    ]


class TestKimiK25MXFP4BcgMI35x(CustomTestCase):
    """Validate aiter BCG prefill capture accuracy on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K25_MXFP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.configs = get_capture_configs()

    def _run_variant(self, config: CaptureConfig) -> float:
        env = os.environ.copy()
        env["SGLANG_AITER_MLA_PERSIST"] = "1"
        for key, value in config.env_vars.items():
            env[key] = value

        other_args = list(COMMON_ARGS) + list(config.capture_args)
        process = popen_launch_server(
            self.model,
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
                num_questions=GSM8K_NUM_QUESTIONS,
                parallel=GSM8K_NUM_QUESTIONS,
                max_new_tokens=512,
                host="http://127.0.0.1",
                port=int(self.base_url.split(":")[-1]),
            )
            metrics = run_eval_few_shot_gsm8k(args)
            print(f"[{config.variant}] {metrics=}")
            return metrics["accuracy"]
        finally:
            kill_process_tree(process.pid)

    def test_bcg_gsm8k(self):
        summary = "### Kimi-K2.5-MXFP4 aiter BCG capture (MI35x, TP4)\n\n"
        summary += "| Capture backend | Accuracy | Threshold | Status |\n"
        summary += "| --------------- | -------- | --------- | ------ |\n"

        failures = []
        for config in self.configs:
            with self.subTest(variant=config.variant):
                acc = self._run_variant(config)
                passed = acc >= ACCURACY_THRESHOLD
                status = "PASS" if passed else "FAIL"
                summary += (
                    f"| {config.variant} | {acc:.3f} | "
                    f"{ACCURACY_THRESHOLD} | {status} |\n"
                )
                if not passed:
                    failures.append((config.variant, acc))

        if is_in_ci():
            write_github_step_summary(summary)

        self.assertEqual(
            failures,
            [],
            f"BCG accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
