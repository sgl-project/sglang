"""decode_gemm runs a table row only on the device, ROCm and hipBLASLt build it names."""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import unquant_hip
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

# this device is gfx950 with 256 CUs on ROCm 7.15, where solution 18487 is sol_a
TABLE = """gfx,cu_num,M,N,K,outdtype,libtype,solidx,splitK,kernelName,rocm
gfx950,256,400,160,64,torch.bfloat16,hipblaslt,18487.0,,sol_a,7.15
gfx950,256,440,160,64,torch.bfloat16,hipblaslt,18488.0,,sol_a,7.15
gfx950,256,480,160,64,torch.bfloat16,hipblaslt,18487.0,,sol_a,7.2
gfx950,240,520,160,64,torch.bfloat16,hipblaslt,18487.0,,sol_a,7.15
gfx942,256,560,160,64,torch.bfloat16,hipblaslt,18487.0,,sol_a,7.15
"""


class TestDecodeGemmTable(CustomTestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = os.path.join(tmp.name, "table.csv")
        with open(path, "w") as f:
            f.write(TABLE)
        self.launched = []
        aiter = types.ModuleType("aiter")
        aiter.hipb_create_extension = lambda: None
        aiter.getHipblasltKernelName = lambda solidx: (
            "sol_a" if solidx == 18487 else "sol_b"
        )
        aiter.hipb_mm = self._hipb_mm
        chip_info = types.ModuleType("aiter.jit.utils.chip_info")
        chip_info.get_gfx_runtime = lambda: "gfx950"
        chip_info.get_cu_num = lambda: 256
        modules = {"aiter": aiter, chip_info.__name__: chip_info}
        self.enterContext(envs.SGLANG_ROCM_DECODE_GEMM_TABLE.override(path))
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(unquant_hip, "_use_aiter", True))
        self.enterContext(patch.object(unquant_hip, "_is_gfx95", True))
        self.enterContext(
            patch("sglang.srt.utils.device.get_hip_version", lambda: (7, 15, 26333))
        )
        for cached in (unquant_hip._decode_rows, unquant_hip._hipblaslt_solution_ok):
            cached.cache_clear()
            self.addCleanup(cached.cache_clear)

    def _hipb_mm(self, x, weight_t, solidx, out_dtype=None, out=None):
        self.launched.append((x.shape[0], solidx, out_dtype))
        return torch.mm(x, weight_t)

    def _run(self, m):
        x = torch.ones(m, 64, dtype=torch.bfloat16)
        weight = torch.ones(160, 64, dtype=torch.bfloat16)
        return unquant_hip.decode_gemm(x, weight)

    def test_runs_the_row_of_this_exact_shape(self):
        self.assertEqual(self._run(400).shape, (400, 160))
        self.assertEqual(self.launched, [(400, 18487, torch.bfloat16)])
        self.assertIsNone(self._run(401))

    def test_skips_rows_this_runtime_cannot_trust(self):
        # a renumbered solution, another ROCm, another CU count, another gfx
        for m in (440, 480, 520, 560):
            self.assertIsNone(self._run(m), m)
        self.assertEqual(self.launched, [])


if __name__ == "__main__":
    unittest.main()
