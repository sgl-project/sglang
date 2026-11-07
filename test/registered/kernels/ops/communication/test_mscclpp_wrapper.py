"""CI correctness check for the MSCCL++ all-reduce backend.

Runs ``benchmark/kernels/all_reduce/benchmark_mscclpp.py`` under torchrun at each
world size. The benchmark brings up SGLang's distributed runtime with MSCCL++
enabled (the ``--enable-mscclpp`` path) and fails if the MSCCL++ communicator is
unavailable. For every message size, dtype, and MSCCL++ all-reduce algorithm
(not only the one tuning picked), it checks MSCCL++ eager and CUDA-graph
all-reduce, including graph replay on new input, and
``GroupCoordinator.all_reduce`` under graph capture bit-exactly against NCCL,
and checks that ``GroupCoordinator.all_reduce`` dispatches to MSCCL++ there.

Requires the ``mscclpp`` package (built into the SGLang Docker image).

Usage::

    python test/registered/kernels/ops/communication/test_mscclpp_wrapper.py
"""

import os
import signal
import subprocess
import unittest
from pathlib import Path

import torch

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=240, stage="extra-b", runner_config="8-gpu-h200")
# Nightly is not redundant here: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1 to expand get_ci_test_range sweeps.
register_cuda_ci(est_time=360, stage="nightly", runner_config="8-gpu-h200")

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BENCHMARK = (
    _REPO_ROOT / "benchmark" / "kernels" / "all_reduce" / "benchmark_mscclpp.py"
)

# 2 KiB - 8 MiB spans all three single-node algorithms MSCCL++ tunes for
# all-reduce: NVLS packet, packet, and RSAG zero-copy (from 512 KiB).
MIN_BYTES = 2 << 10
MAX_BYTES = 8 << 20
DTYPES = get_ci_test_range(["float16", "bfloat16", "float32"], ["bfloat16"])
# Covers MSCCL++ start-up tuning (~1-2 min) plus the full dtype sweep.
TIMEOUT_SECONDS = 1200


class TestMscclppAllReduce(CustomTestCase):
    def test_world_size_4(self):
        self._run_benchmark(world_size=4)

    def test_world_size_8(self):
        self._run_benchmark(world_size=8)

    def _run_benchmark(self, world_size: int) -> None:
        available = torch.cuda.device_count()
        if available < world_size:
            self.skipTest(f"needs {world_size} GPUs, {available} available")
        command = [
            "torchrun",
            "--nproc_per_node",
            str(world_size),
            "--local-addr",
            "127.0.0.1",
            str(_BENCHMARK),
            "--min-bytes",
            str(MIN_BYTES),
            "--max-bytes",
            str(MAX_BYTES),
            "--dtypes",
            *DTYPES,
            "--mscclpp-algorithms",
            "all",
            "--expect-graph-backend",
            "mscclpp",
        ]
        returncode = _run_process_group(command, timeout=TIMEOUT_SECONDS)
        self.assertEqual(
            returncode, 0, f"benchmark_mscclpp.py failed at world size {world_size}"
        )


def _run_process_group(command: list[str], timeout: int) -> int:
    env = dict(os.environ)
    # Single-machine launch, as in sglang.test.kernels.multigpu.
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    proc = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # Kill torchrun and its workers so no rank keeps holding GPUs.
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise


if __name__ == "__main__":
    unittest.main()
