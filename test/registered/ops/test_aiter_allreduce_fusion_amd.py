import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_amd_ci

# Dedicated AMD 8-GPU suite for AITER fused allreduce+rmsnorm validation.
register_amd_ci(est_time=240, suite="stage-c-test-aiter-fusion-8-gpu-amd")


class TestAiterAllreduceFusionAmd(unittest.TestCase):
    def test_fused_ar_rms_benchmark(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA/ROCm device is not available.")
        if torch.cuda.device_count() < 8:
            self.skipTest("This test requires at least 8 GPUs.")

        repo_root = Path(__file__).resolve().parents[3]
        benchmark_script = (
            repo_root
            / "benchmark"
            / "kernels"
            / "all_reduce"
            / "benchmark_fused_ar_rms_amd.py"
        )
        self.assertTrue(
            benchmark_script.exists(),
            f"Missing benchmark script: {benchmark_script}",
        )

        with tempfile.TemporaryDirectory(prefix="aiter_fused_ar_rms_") as tmpdir:
            csv_path = Path(tmpdir) / "fused_ar_rms_check.csv"
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(benchmark_script),
                "--dtype",
                "bf16",
                "--prefill-shapes",
                # Include both <=64MiB and >64MiB shapes to verify default gate behavior.
                "128x7168,512x7168,2048x7168,4096x7168,5120x7168",
                "--decode-shapes",
                "1x7168,8x7168,64x7168,512x7168",
                "--warmup",
                "3",
                "--iters",
                "15",
                "--repeats",
                "2",
                "--csv-out",
                str(csv_path),
            ]

            env = os.environ.copy()
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=1200,
            )

            if result.returncode != 0:
                self.fail(
                    "Benchmark command failed.\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{result.stdout}"
                )

            self.assertTrue(csv_path.exists(), f"CSV output not found: {csv_path}")

            with open(csv_path, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            self.assertGreater(len(rows), 0, "CSV contains no rows.")

            eager_rows = [r for r in rows if r["mode"] == "eager"]
            graph_rows = [r for r in rows if r["mode"] == "graph"]
            self.assertGreater(len(eager_rows), 0, "Missing eager rows in CSV.")
            self.assertGreater(len(graph_rows), 0, "Missing graph rows in CSV.")

            # Correctness should always pass regardless of fused availability.
            bad_rows = [r for r in rows if r["correctness_ok"] != "True"]
            self.assertEqual(
                [],
                bad_rows,
                f"Found correctness failures: {bad_rows}",
            )

            # We should see fused path active for small shapes in both modes.
            self.assertTrue(
                any(r["fused_available"] == "True" for r in eager_rows),
                "Expected at least one eager row with fused_available=True.",
            )
            self.assertTrue(
                any(r["fused_available"] == "True" for r in graph_rows),
                "Expected at least one graph row with fused_available=True.",
            )

            # Default gate should reject at least one oversized eager shape.
            large_eager_rows = [
                r for r in eager_rows if int(r["bytes_per_rank"]) > 64 * 1024 * 1024
            ]
            self.assertTrue(
                any(r["fused_available"] == "False" for r in large_eager_rows),
                "Expected fused fallback for oversized eager shape(s) under default gate.",
            )


if __name__ == "__main__":
    unittest.main()
