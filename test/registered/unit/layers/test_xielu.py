import unittest

import torch

from sglang.srt.layers.activation import XIELU
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestXIELU(CustomTestCase):
    def test_builds_on_the_meta_device_with_the_same_scalars(self):
        # Built on the meta device, nothing can be read back from a buffer.
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                on_cpu = XIELU(beta=0.3, eps=-1e-6, dtype=dtype)
                with torch.device("meta"):
                    on_meta = XIELU(beta=0.3, eps=-1e-6, dtype=dtype)
                self.assertEqual(on_meta.beta.device.type, "meta")
                self.assertEqual(on_meta._beta_scalar, on_cpu._beta_scalar)
                self.assertEqual(on_meta._eps_scalar, on_cpu._eps_scalar)
                # Rounded through dtype, like the buffers they mirror.
                self.assertEqual(on_cpu._beta_scalar, on_cpu.beta.float().item())
                self.assertEqual(on_cpu._eps_scalar, on_cpu.eps.float().item())


if __name__ == "__main__":
    unittest.main()
