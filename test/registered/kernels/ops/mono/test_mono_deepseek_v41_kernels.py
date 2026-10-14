"""The DeepSeek-V4.1 gfx950 monokernels against SGLang's ops (RFC #43949), each
under torchrun at TP2 and TP4 (``sglang.test.kernels.mono.deepseek_v41_hip``): K1's seam
mixes, SWA KV records and query; K2's attention core, wo_a and wo_b stages; the
FFN launch's seam, norm and MoE; each output against the spec's bound, then a
bitwise graph replay across widths with every rank agreeing."""

import subprocess
import sys
import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=1800, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")


def _gfx950_gpus() -> int:
    if not (torch.cuda.is_available() and torch.version.hip):
        return 0
    if "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return 0
    return torch.cuda.device_count()


@unittest.skipIf(_gfx950_gpus() < 4, "needs 4 gfx950 GPUs")
class TestMonoDeepseekV41Kernels(CustomTestCase):
    def _check(self, module: str, *args: str) -> None:
        for tp in (2, 4):
            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={tp}",
                f"--master-port={29600 + tp}",
                "-m",
                f"sglang.test.kernels.mono.deepseek_v41_hip.{module}",
                *args,
            ]
            result = subprocess.run(cmd, timeout=1800)
            self.assertEqual(result.returncode, 0, f"{module} at TP{tp}")

    def test_ffn_launch(self):
        self._check("check_ffn", "--replays", "200")

    def test_k1(self):
        self._check("check_k1", "--trials", "2")

    def test_k2(self):
        self._check("check_k2", "--replays", "100")


if __name__ == "__main__":
    unittest.main()
