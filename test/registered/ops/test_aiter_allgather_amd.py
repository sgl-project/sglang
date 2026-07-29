import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch

from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=180, suite="stage-c-test-large-8-gpu-amd")


class TestAiterAllGatherAmd(unittest.TestCase):

    @staticmethod
    def _gpu_count():
        return torch.cuda.device_count() if torch.cuda.is_available() else 0

    def test_aiter_allgather_matches_rccl(self):
        gpu_count = self._gpu_count()
        if gpu_count < 2:
            self.skipTest("This test requires at least 2 GPUs.")

        repo_root = Path(__file__).resolve().parents[3]
        benchmark_script = (
            repo_root / "benchmark" / "kernels" / "all_gather" / "benchmark_aiter.py"
        )
        self.assertTrue(
            benchmark_script.exists(),
            f"Missing benchmark script: {benchmark_script}",
        )

        dtype_names = [
            "float32",
            "float16",
            "bfloat16",
            "uint64_t",
            "int64_t",
            "uint32_t",
            "int32_t",
            "int16_t",
            "uint8_t",
            "int8_t",
        ]
        # Keep the CI matrix compact: one small metadata shape and one
        # medium aligned 2-D shape exercise both naive and vectorized kernels.
        shapes = "16,;8,1024"
        dims = "0,-1"
        tp_sizes = [tp for tp in (2, 4, 8) if gpu_count >= tp]

        outputs = []
        for tp_size in tp_sizes:
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={tp_size}",
                str(benchmark_script),
                "--dtype",
                ",".join(dtype_names),
                "--shapes",
                shapes,
                "--dims",
                dims,
                "--warmup",
                "0",
                "--iters",
                "1",
                "--correctness-only",
            ]

            env = os.environ.copy()
            env.setdefault("AITER_AOT_IMPORT", "1")
            result = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=240,
            )
            outputs.append(f"### TP={tp_size}\n{result.stdout}")

            if result.returncode != 0:
                self.fail(
                    "Aiter all-gather correctness sweep failed.\n"
                    f"Return code: {result.returncode}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"Output:\n{result.stdout}"
                )

            expected_rows = (
                len(dtype_names) * len(shapes.split(";")) * len(dims.split(","))
            )
            rows = [
                line.strip()
                for line in result.stdout.splitlines()
                if line.strip().startswith(tuple(dtype_names))
            ]
            self.assertEqual(
                expected_rows,
                len(rows),
                f"Expected {expected_rows} rows for TP={tp_size}, got:\n{result.stdout}",
            )

            bad_rows = [row for row in rows if " True " not in row]
            self.assertEqual(
                [],
                bad_rows,
                f"Correctness failed for one or more all-gather rows:\n{result.stdout}",
            )

        if gpu_count >= 8:
            self.assertEqual([2, 4, 8], tp_sizes)
        else:
            self.assertEqual([2, 4][: len(tp_sizes)], tp_sizes)


if __name__ == "__main__":
    unittest.main()
