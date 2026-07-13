import os
import subprocess
import sys
import textwrap
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRocmNoDeviceImports(CustomTestCase):
    @unittest.skipIf(
        torch.version.hip is None or torch.cuda.is_available(),
        "requires a ROCm build without a visible HIP device",
    )
    def test_rocm_quant_modules_import_without_visible_device(self):
        code = textwrap.dedent("""
            import sglang.kernels.ops.attention.rocm_mla_decode_rope  # noqa: F401
            import sglang.srt.layers.quantization.fp8_kernel  # noqa: F401
            import sglang.srt.layers.quantization.fp8_utils  # noqa: F401
            import sglang.srt.layers.quantization.mxfp4  # noqa: F401
            import sglang.srt.layers.quantization.quark.utils  # noqa: F401
            import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4  # noqa: F401
            import sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4_moe  # noqa: F401
            import sglang.srt.layers.quantization.quark_int4fp8_moe  # noqa: F401
            """)
        env = os.environ.copy()
        env["SGLANG_USE_AITER"] = "0"

        subprocess.run([sys.executable, "-c", code], check=True, env=env)


if __name__ == "__main__":
    unittest.main()
