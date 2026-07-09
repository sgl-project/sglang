"""MI35x PR-CI coverage for Qwen3.5-FP8 aiter AR-fusion.

This consolidates the previous Qwen3.5-FP8 MI35x perf coverage with the
PR-specific fused AR+RMSNorm+per-group-quant accuracy check. The file runs in
the 8-GPU MI35x stage-c suite and launches two TP4 servers in parallel:

* GPUs 0-3: fused AR+RMSNorm+per-group FP8 quant enabled.
* GPUs 4-7: same launch with SGLANG_DISABLE_FUSED_AR_QUANT=1 fallback.

Each server runs GSM8K and reports accuracy, invalid rate, latency, and output
throughput. This keeps the PR signal focused while avoiding a second standalone
Qwen3.5-FP8 CI script.
"""

import os
import re
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=4800, suite="stage-c-test-large-8-gpu-amd-mi35x")

QWEN35_FP8_MODEL_PATH = os.environ.get(
    "QWEN35_FP8_MODEL_PATH",
    "Qwen/Qwen3.5-397B-A17B-FP8",
)
SERVER_LAUNCH_TIMEOUT = 4800
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
ACCURACY_THRESHOLD = 0.94


@dataclass
class FusionVariant:
    """A Qwen3.5-FP8 AR-fusion configuration to validate."""

    variant: str
    hip_visible_devices: str
    port_offset: int
    env_vars: Dict[str, str] = field(default_factory=dict)


COMMON_ARGS: List[str] = [
    "--tensor-parallel-size",
    "4",
    "--trust-remote-code",
    "--attention-backend",
    "aiter",
    "--kv-cache-dtype",
    "fp8_e4m3",
    "--page-size",
    "16",
    "--chunked-prefill-size",
    "8192",
    "--mem-fraction-static",
    "0.8",
    "--disable-radix-cache",
    "--enable-aiter-allreduce-fusion",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]


def _base_url_with_port_offset(offset: int) -> str:
    host, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
    return f"{host}:{int(port) + offset}"


def get_fusion_variants() -> List[FusionVariant]:
    return [
        FusionVariant(
            variant="fused-ar-rms-per-group-quant",
            hip_visible_devices="0,1,2,3",
            port_offset=0,
            env_vars={
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
            },
        ),
        FusionVariant(
            variant="disable-fused-ar-quant-opt-out",
            hip_visible_devices="4,5,6,7",
            port_offset=1,
            env_vars={
                "SGLANG_USE_AITER": "1",
                "SGLANG_USE_AITER_UNIFIED_ATTN": "1",
                "SGLANG_DISABLE_FUSED_AR_QUANT": "1",
            },
        ),
    ]


def _parse_gsm8k_metrics(stdout: str) -> Dict[str, float]:
    metrics = {}
    for key, pattern in {
        "accuracy": r"Accuracy:\s*([0-9.]+)",
        "invalid": r"Invalid:\s*([0-9.]+)",
        "latency": r"Latency:\s*([0-9.]+)\s*s",
        "output_throughput": r"Output throughput:\s*([0-9.]+)\s*token/s",
    }.items():
        match = re.search(pattern, stdout)
        if match is None:
            raise AssertionError(f"Could not parse {key} from GSM8K output:\n{stdout}")
        metrics[key] = float(match.group(1))
    return metrics


class TestQwen35Fp8ArFusionMI35x(CustomTestCase):
    """Validate Qwen3.5-FP8 AR-fusion accuracy and throughput on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_FP8_MODEL_PATH
        cls.variants = get_fusion_variants()

    def _run_gsm8k(self, base_url: str) -> Dict[str, float]:
        port = int(base_url.rsplit(":", 1)[-1])
        command = [
            "python3",
            "benchmark/gsm8k/bench_sglang.py",
            "--num-questions",
            str(GSM8K_NUM_QUESTIONS),
            "--parallel",
            str(GSM8K_NUM_QUESTIONS),
            "--num-shots",
            "5",
            "--port",
            str(port),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise AssertionError(
                "GSM8K benchmark failed:\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        print(result.stdout)
        return _parse_gsm8k_metrics(result.stdout)

    def _run_variant(self, variant: FusionVariant) -> Dict[str, float]:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = variant.hip_visible_devices
        env.update(variant.env_vars)
        base_url = _base_url_with_port_offset(variant.port_offset)

        process = popen_launch_server(
            self.model,
            base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=list(COMMON_ARGS),
            env=env,
        )
        try:
            requests.get(base_url + "/flush_cache", timeout=10)
            metrics = self._run_gsm8k(base_url)
            print(f"[{variant.variant}] {metrics=}")
            return metrics
        finally:
            kill_process_tree(process.pid)

    def test_qwen35_fp8_ar_fusion_accuracy_and_perf(self):
        summary = "### Qwen3.5-FP8 aiter AR-fusion (MI35x, parallel TP4)\n\n"
        summary += (
            "| Variant | GPUs | Accuracy | Invalid | Latency (s) | Output tok/s | "
            "Threshold | Status |\n"
        )
        summary += "| ------- | ---- | -------- | ------- | ----------- | ------------ | --------- | ------ |\n"

        failures = []
        with ThreadPoolExecutor(max_workers=len(self.variants)) as executor:
            future_to_variant = {
                executor.submit(self._run_variant, variant): variant
                for variant in self.variants
            }
            for future in as_completed(future_to_variant):
                variant = future_to_variant[future]
                with self.subTest(variant=variant.variant):
                    metrics = future.result()
                    accuracy = metrics["accuracy"]
                    passed = accuracy >= ACCURACY_THRESHOLD
                    status = "PASS" if passed else "FAIL"
                    summary += (
                        f"| {variant.variant} | {variant.hip_visible_devices} | "
                        f"{accuracy:.3f} | {metrics['invalid']:.3f} | "
                        f"{metrics['latency']:.2f} | "
                        f"{metrics['output_throughput']:.2f} | "
                        f"{ACCURACY_THRESHOLD} | {status} |\n"
                    )
                    if not passed:
                        failures.append((variant.variant, accuracy))

        if is_in_ci():
            write_github_step_summary(summary)
        print(summary)

        self.assertEqual(
            failures,
            [],
            f"Qwen3.5-FP8 AR-fusion accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
