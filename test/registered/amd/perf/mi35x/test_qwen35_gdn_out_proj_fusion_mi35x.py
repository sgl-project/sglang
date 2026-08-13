"""MI35x PR-CI accuracy gate for the Qwen3.5 GDN out_proj fusion.

Two parallel TP4 servers on the MXFP4-AttnFP8 checkpoint (8-GPU stage-c): fused
gated-RMSNorm + FP8 per-token quant into the out_proj a8w8 GEMM (GPUs 0-3) vs
the unfused path (GPUs 4-7, SGLANG_DISABLE_GDN_OUT_PROJ_FUSION=1). The fused
path must hold GSM8K accuracy against the baseline.
"""

import os
import re
import subprocess
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
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
from sglang.utils import download_and_cache_file

register_amd_ci(est_time=4800, suite="stage-c-test-large-8-gpu-amd-mi35x")

# Only the AttnFP8 checkpoints quantize linear_attn to per-token a8w8 fp8.
QWEN35_ATTNFP8_MODEL_PATH = os.environ.get(
    "QWEN35_ATTNFP8_MODEL_PATH",
    "amd/Qwen3.5-397B-A17B-MXFP4-AttnFP8-V2",
)
SERVER_LAUNCH_TIMEOUT = 4800
GSM8K_NUM_QUESTIONS = int(os.environ.get("GSM8K_NUM_QUESTIONS", "1319"))
# The fusion is accuracy-neutral (~0.94-0.97 here), so 0.92 gates regressions.
ACCURACY_THRESHOLD = 0.92

# bench_sglang.py is at the repo root (this file is 5 levels below).
REPO_ROOT = Path(__file__).resolve().parents[5]
GSM8K_BENCH_SCRIPT = REPO_ROOT / "benchmark" / "gsm8k" / "bench_sglang.py"
GSM8K_DATA_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

# TP=4 so both variants fit on the 8-GPU runner at once.
COMMON_ARGS: List[str] = [
    "--attention-backend",
    "aiter",
    "--tp",
    "4",
    "--trust-remote-code",
    "--disable-radix-cache",
    "--mem-fraction-static",
    "0.85",
    "--model-loader-extra-config",
    '{"enable_multithread_load": true}',
    "--watchdog-timeout",
    "1200",
]

# AITER-only; per-token a8w8 supplies the pre-shuffled weights the GEMM needs.
COMMON_ENV = {
    "SGLANG_USE_AITER": "1",
    "SGLANG_USE_AITER_FP8_PER_TOKEN": "1",
}


@dataclass
class OutProjFusionVariant:
    """A Qwen3.5 GDN out_proj fusion configuration to validate."""

    variant: str
    hip_visible_devices: str
    port_offset: int
    env_vars: Dict[str, str] = field(default_factory=dict)


def _base_url_with_port_offset(offset: int) -> str:
    host, port = DEFAULT_URL_FOR_TEST.rsplit(":", 1)
    return f"{host}:{int(port) + offset}"


def get_out_proj_fusion_variants() -> List[OutProjFusionVariant]:
    return [
        OutProjFusionVariant(
            variant="fused-gdn-out-proj",
            hip_visible_devices="0,1,2,3",
            port_offset=0,
        ),
        OutProjFusionVariant(
            variant="unfused-baseline",
            hip_visible_devices="4,5,6,7",
            port_offset=1,
            env_vars={"SGLANG_DISABLE_GDN_OUT_PROJ_FUSION": "1"},
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


class TestQwen35GdnOutProjFusionMI35x(CustomTestCase):
    """Validate Qwen3.5 GDN fused out_proj accuracy on MI35x."""

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_ATTNFP8_MODEL_PATH
        cls.variants = get_out_proj_fusion_variants()
        # Pre-fetch once so the two parallel subprocesses don't race the cache write.
        cls.gsm8k_data_path = download_and_cache_file(GSM8K_DATA_URL)

    def _run_gsm8k(self, base_url: str) -> Dict[str, float]:
        port = int(base_url.rsplit(":", 1)[-1])
        command = [
            "python3",
            str(GSM8K_BENCH_SCRIPT),
            "--num-questions",
            str(GSM8K_NUM_QUESTIONS),
            "--parallel",
            str(GSM8K_NUM_QUESTIONS),
            "--num-shots",
            "5",
            "--data-path",
            str(self.gsm8k_data_path),
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

    def _run_variant(self, variant: OutProjFusionVariant) -> Dict[str, float]:
        env = os.environ.copy()
        env["HIP_VISIBLE_DEVICES"] = variant.hip_visible_devices
        env.update(COMMON_ENV)
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

    def test_qwen35_gdn_out_proj_fusion_accuracy(self):
        summary = (
            "### Qwen3.5 MXFP4-AttnFP8 GDN out_proj fusion GSM8K "
            "(MI35x, parallel TP4)\n\n"
        )
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
            f"Qwen3.5 GDN out_proj fusion accuracy below {ACCURACY_THRESHOLD}: {failures}",
        )


if __name__ == "__main__":
    unittest.main()
