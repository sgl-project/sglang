import os
import re
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
        nproc = min(self._gpu_count(), 4)
        if nproc < 2:
            self.skipTest("This test requires at least 2 GPUs.")

        repo_root = Path(__file__).resolve().parents[3]
        benchmark_script = (
            repo_root / "benchmark" / "kernels" / "all_gather" / "benchmark_aiter.py"
        )
        self.assertTrue(
            benchmark_script.exists(),
            f"Missing benchmark script: {benchmark_script}",
        )

        shapes = "1,32320;2,32320;4,32320"
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={nproc}",
            str(benchmark_script),
            "--dtype",
            "bfloat16",
            "--shapes",
            shapes,
            "--warmup",
            "3",
            "--iters",
            "10",
        ]

        env = os.environ.copy()
        result = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            self.fail(
                "Aiter all-gather benchmark failed.\n"
                f"Return code: {result.returncode}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Output:\n{result.stdout}"
            )

        rows = [
            line.strip()
            for line in result.stdout.splitlines()
            if re.match(r"^\(\d+, 32320\)", line.strip())
        ]
        self.assertEqual(
            3,
            len(rows),
            f"Expected 3 benchmark rows for shapes {shapes}, got:\n{result.stdout}",
        )

        bad_rows = [row for row in rows if " True " not in row]
        self.assertEqual(
            [],
            bad_rows,
            f"Correctness failed for one or more all-gather rows:\n{result.stdout}",
        )


if __name__ == "__main__":
    unittest.main()
