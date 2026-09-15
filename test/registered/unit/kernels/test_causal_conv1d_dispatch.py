import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.mamba import causal_conv1d_triton
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCausalConv1dDispatch(CustomTestCase):
    def test_gfx950_glm_prefill_config(self):
        with patch.object(
            causal_conv1d_triton,
            "is_gfx95_supported",
            return_value=True,
        ):
            for dim in (3072, 6144):
                with self.subTest(dim=dim):
                    config = causal_conv1d_triton._causal_conv1d_prefill_config(
                        dim=dim,
                        width=4,
                        total_tokens=8192,
                        dtype=torch.bfloat16,
                    )
                    self.assertEqual(config["BLOCK_M"], 16)
                    self.assertEqual(config["BLOCK_N"], 256)
                    self.assertEqual(config["num_warps"], 4)
                    self.assertEqual(config["num_stages"], 3)

    def test_other_geometries_keep_default(self):
        base = dict(
            dim=6144,
            width=4,
            total_tokens=8192,
            dtype=torch.bfloat16,
        )
        with patch.object(
            causal_conv1d_triton,
            "is_gfx95_supported",
            return_value=True,
        ):
            for field, value in (
                ("dim", 4096),
                ("width", 3),
                ("total_tokens", 8191),
                ("dtype", torch.float16),
            ):
                with self.subTest(field=field, value=value):
                    config = causal_conv1d_triton._causal_conv1d_prefill_config(
                        **(base | {field: value})
                    )
                    self.assertEqual(config["BLOCK_M"], 8)
                    self.assertEqual(config["num_stages"], 2)
        with patch.object(
            causal_conv1d_triton,
            "is_gfx95_supported",
            return_value=False,
        ):
            config = causal_conv1d_triton._causal_conv1d_prefill_config(**base)
            self.assertEqual(config["BLOCK_M"], 8)
            self.assertEqual(config["num_stages"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=3)
