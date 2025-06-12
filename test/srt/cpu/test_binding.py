import re
import unittest

import sgl_kernel
import torch

kernel = torch.ops.sgl_kernel

from sglang.test.test_utils import CustomTestCase


class TestGemm(CustomTestCase):
    def test_binding(self):
        start_id = 1
        n_cpu = 6

        expected_cores = list(map(str, range(start_id, start_id + n_cpu)))
        cpu_ids = ",".join(expected_cores)
        output = kernel.init_cpu_threads_env(cpu_ids)

        bindings = re.findall(r"OMP tid: \d+, core (\d+)", output)
        self.assertEqual(len(bindings), n_cpu)

        self.assertEqual(bindings, expected_cores)


if __name__ == "__main__":
    unittest.main()
