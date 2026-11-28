import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


class TestDiffusionKernelRouting(unittest.TestCase):
    def test_kernel_suite_arguments(self):
        workflow = ROOT / ".github/workflows/pr-test-jit-kernel.yml"
        job = yaml.safe_load(workflow.read_text())["jobs"]["jit-kernel-test"]
        command = next(
            step["run"] for step in job["steps"] if step.get("name") == "Run test"
        )
        command = command.replace(
            "${{ matrix.suite_args }}", "--fork-worker-batch-size 20"
        )

        # exercise the runner's actual file selection and capture the invocation before any GPU work
        script = 'python3() { printf "%s\\n" "$@"; }\n' + command
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(
                os.environ,
                SUITE="unit-test-1-gpu-small",
                KERNEL_BENCH="false",
                RUNNER_TEMP=tmp,
            )

            def arguments(scoped, shard):
                return subprocess.check_output(
                    ["bash", "-e", "-c", script],
                    cwd=ROOT,
                    env=dict(env, DIFFUSION_KERNEL_ONLY=scoped, SHARD=str(shard)),
                    text=True,
                ).splitlines()

            common = [
                "run_suite.py",
                "--hw",
                "cuda",
                "--suite",
                "base-b-kernel-unit-test-1-gpu-small",
                "--fork-worker-batch-size",
                "20",
            ]
            for shard in range(3):
                self.assertEqual(
                    arguments("false", shard),
                    common
                    + ["--auto-partition-id", str(shard), "--auto-partition-size", "3"],
                )
            args = arguments("true", 0)
            self.assertEqual(args[:-1], common + ["--only-file-list"])
            paths = set(Path(args[-1]).read_text().splitlines())
            self.assertIn("test/registered/kernels/ops/diffusion/test_rope.py", paths)
            self.assertIn(
                "test/registered/kernels/benchmark/diffusion/bench_ltx25_decoder_rope.py",
                paths,
            )
            self.assertIn(
                "test/registered/kernels/ops/attention/test_flashinfer_fa2_attention.py",
                paths,
            )
            self.assertIn(
                "test/registered/unit/multimodal_gen/test_flux3_swiglu_fp8.py", paths
            )
            self.assertNotIn(
                "test/registered/unit/mem_cache/test_prefix_valid_fp8_pool.py", paths
            )
