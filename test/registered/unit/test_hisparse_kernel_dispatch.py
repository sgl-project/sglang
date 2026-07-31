import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.kvcache import hisparse
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHiSparseHostCacheLocsDispatch(unittest.TestCase):
    def tearDown(self):
        hisparse._jit_sparse_module.cache_clear()

    def test_dtype_selects_a_distinct_cpp_specialization(self):
        int32_module = object()
        int64_module = object()
        hisparse._jit_sparse_module.cache_clear()

        with patch.object(
            hisparse,
            "load_jit",
            side_effect=[int32_module, int64_module],
        ) as load_jit:
            got_int32 = hisparse._jit_sparse_module(
                32, 256, 4, 8, torch.int32, is_mla=True
            )
            got_int64 = hisparse._jit_sparse_module(
                32, 256, 4, 8, torch.int64, is_mla=True
            )

        self.assertIs(got_int32, int32_module)
        self.assertIs(got_int64, int64_module)
        self.assertEqual(load_jit.call_count, 2)
        self.assertIn(
            "int32_t",
            load_jit.call_args_list[0].kwargs["cuda_wrappers"][0][1],
        )
        self.assertIn(
            "int64_t",
            load_jit.call_args_list[1].kwargs["cuda_wrappers"][0][1],
        )


if __name__ == "__main__":
    unittest.main()
