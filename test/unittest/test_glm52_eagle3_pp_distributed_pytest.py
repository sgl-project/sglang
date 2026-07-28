"""Pytest wrapper that launches the real distributed test via torchrun.

This does NOT reimplement the distributed behavior — it merely launches
the script as a subprocess and checks the return code and JSON output.

Set SGLANG_RUN_GPU_INTEGRATION=1 to enable NCCL/CUDA tests.
Otherwise only the Gloo/CPU test runs.
"""
import json
import os
import subprocess
import sys

import pytest


SCRIPT = "test/unittest/test_glm52_eagle3_pp_distributed.py"
PY = sys.executable


def _run_torchrun(backend, device, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nproc_per_node=2",
        SCRIPT,
        "--backend", backend,
        "--device", device,
        "--warmup", "5",
        "--iterations", "20",
        "--output-json", "/tmp/glm52_pp_test_output.json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, env=env,
    )
    return result


class TestDistributedGloo:
    """Real two-process Gloo distributed test."""

    def test_gloo_aux_propagation(self):
        """Run the distributed test with Gloo backend on CPU."""
        result = _run_torchrun("gloo", "cpu")
        assert result.returncode == 0, (
            f"torchrun failed (RC={result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
        assert "All distributed tests PASSED" in result.stdout

        # Check JSON output
        json_path = "/tmp/glm52_pp_test_output.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            assert data["backend"] == "gloo"
            assert "send_latency_us" in data or data.get("rank") == 1


@pytest.mark.skipif(
    os.environ.get("SGLANG_RUN_GPU_INTEGRATION") != "1",
    reason="Set SGLANG_RUN_GPU_INTEGRATION=1 to run GPU tests",
)
class TestDistributedNCCL:
    """Real two-GPU NCCL distributed test (opt-in)."""

    def test_nccl_aux_propagation(self):
        """Run the distributed test with NCCL backend on CUDA."""
        result = _run_torchrun(
            "nccl", "cuda",
            extra_env={"NCCL_DEBUG": "INFO", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"},
        )
        assert result.returncode == 0, (
            f"torchrun failed (RC={result.returncode}):\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
        assert "All distributed tests PASSED" in result.stdout
