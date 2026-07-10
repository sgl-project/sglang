"""DeepSeek-OCR-2 olmOCR-bench accuracy on Intel XPU (1-GPU nightly).

Unlike ``test_deepseek_ocr_2.py`` (a boot + short-decode smoke test), this
launches the server with the tuned OCR serving config and runs the full
olmOCR-bench via ``benchmark/ocr/bench_sglang.py``, then asserts the
aggregate score and records the per-split breakdown to the GitHub step
summary.

Threshold: the full-bench aggregate must be >= 0.80. A full XPU run scored
86.6% overall (arxiv_math/old_scans_math ~99.6%, headers_footers 93.6%,
multi_column 84.0%, table_tests 78.4%, long_tiny_text 53.4%, old_scans
26.0% — the last dragged down by transient broken-pipe drops at concurrency
26, not model quality). 0.80 leaves headroom above that baseline. For
reference, olmOCR v0.4.0 ~= 82.4 on the same bench.

Dataset: the olmOCR-bench bench_data/ directory (~1,403 PDFs) is downloaded
by the nightly workflow step before this runs:

    hf download --repo-type dataset allenai/olmOCR-bench --local-dir ./olmOCR-bench

and its location is passed via the OLMOCR_BENCH_DIR env var.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    write_github_step_summary,
)

register_xpu_ci(est_time=7200, suite="nightly-xpu-1-gpu", nightly=True)

# Repo root: test/registered/xpu/<this file> -> parents[3].
_REPO_ROOT = Path(__file__).resolve().parents[3]
# Default matches the workflow's --local-dir; override with OLMOCR_BENCH_DIR.
_DEFAULT_BENCH_DIR = _REPO_ROOT / "olmOCR-bench" / "bench_data"


@unittest.skipUnless(
    torch.xpu.is_available(),
    "Intel XPU not available (torch.xpu.is_available() returned False)",
)
class TestDeepSeekOCR2OlmBenchXPU(CustomTestCase):
    model = "deepseek-ai/DeepSeek-OCR-2"
    # Aggregate olmOCR-bench score (total_passed / total_tests) must clear
    # this. Validated at 86.6% on a full XPU run; 0.80 leaves headroom.
    accuracy = 0.80
    # Full olmOCR-bench run at concurrency 26 is long; give the whole flow a
    # generous budget (matches register est_time / the workflow per-file cap).
    timeout_for_server_launch = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH

    # Defaults match the runbook (full bench). Overridable via env for quick
    # sanity runs without editing the test (e.g. OLMOCR_BENCH_SPLIT=arxiv_math
    # OLMOCR_BENCH_MAX_SAMPLES=5).
    concurrency = int(os.environ.get("OLMOCR_BENCH_CONCURRENCY", "26"))
    split = os.environ.get("OLMOCR_BENCH_SPLIT", "all")
    max_samples = int(os.environ.get("OLMOCR_BENCH_MAX_SAMPLES", "-1"))

    # Server args mirror the DeepSeek-OCR-2 XPU launch command in the runbook.
    other_args = [
        "--dtype",
        "bfloat16",
        "--trust-remote-code",
        "--disable-radix-cache",
        "--attention-backend",
        "intel_xpu",
        "--mem-fraction-static",
        "0.65",
        "--max-running-requests",
        "26",
        "--enable-mixed-chunk",
        "--chunked-prefill-size",
        "8192",
        "--disable-cuda-graph",
    ]
    env = {"SGLANG_USE_SGL_XPU": "1"}

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.bench_dir = Path(os.environ.get("OLMOCR_BENCH_DIR", _DEFAULT_BENCH_DIR))
        cls.output_dir = _REPO_ROOT / "ocr_bench_results"
        env = {**os.environ, **cls.env}
        try:
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout_for_server_launch,
                other_args=list(cls.other_args),
                env=env,
            )
        except Exception as e:
            write_github_step_summary(f"Failed to launch server for {cls.model}: {e}")
            raise AssertionError(f"Test failed for {cls.model}: {e}")

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None):
            kill_process_tree(cls.process.pid)

    def test_olmocr_bench(self):
        if not self.bench_dir.exists():
            self.fail(
                f"olmOCR-bench data not found at {self.bench_dir}. Download it first:\n"
                "  hf download --repo-type dataset allenai/olmOCR-bench "
                "--local-dir ./olmOCR-bench"
            )

        port = urlparse(self.base_url).port
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "benchmark" / "ocr" / "bench_sglang.py"),
            "--port",
            str(port),
            "--split",
            self.split,
            "--concurrency",
            str(self.concurrency),
            "--model",
            self.model,
            *(["--max-samples", str(self.max_samples)] if self.max_samples > 0 else []),
            "--bench-dir",
            str(self.bench_dir),
            "--output-dir",
            str(self.output_dir),
        ]

        try:
            subprocess.run(cmd, check=True, cwd=str(_REPO_ROOT))
        except subprocess.CalledProcessError as e:
            self.fail(f"olmOCR-bench run failed for {self.model}: {e}")

        summary_path = self.output_dir / "summary.json"
        if not summary_path.exists():
            self.fail(f"Benchmark produced no summary at {summary_path}")

        with open(summary_path, encoding="utf-8") as f:
            results = json.load(f)

        total_tests = sum(r.get("total_tests", 0) for r in results.values())
        total_passed = sum(r.get("total_passed", 0) for r in results.values())
        total_errored = sum(r.get("error_samples", 0) for r in results.values())
        score = total_passed / total_tests if total_tests else 0.0

        lines = [
            f"## DeepSeek-OCR-2 olmOCR-bench (XPU, concurrency {self.concurrency})",
            "",
            "| Split | Tests | Passed | Score | Errored |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for split, r in results.items():
            lines.append(
                f"| {split} | {r.get('total_tests', 0)} | "
                f"{r.get('total_passed', 0)} | {r.get('overall_score', 0.0):.1f}% | "
                f"{r.get('error_samples', 0)} |"
            )
        lines.append(
            f"| **TOTAL** | {total_tests} | {total_passed} | "
            f"**{100.0 * score:.1f}%** | {total_errored} |"
        )
        write_github_step_summary("\n".join(lines) + "\n")

        # Guard against a silent empty run (server up but zero tests scored)
        # before comparing the score, so a 0/0 reads clearly.
        self.assertGreater(
            total_tests, 0, f"olmOCR-bench scored 0 tests for {self.model}"
        )
        self.assertGreaterEqual(
            score,
            self.accuracy,
            f"olmOCR-bench aggregate for {self.model} is {100.0 * score:.1f}%, "
            f"below the {100.0 * self.accuracy:.0f}% threshold "
            f"({total_errored} samples errored)",
        )


if __name__ == "__main__":
    unittest.main()
