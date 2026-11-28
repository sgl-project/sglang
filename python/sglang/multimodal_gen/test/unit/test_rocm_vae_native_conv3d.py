# SPDX-License-Identifier: Apache-2.0
import os
import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanCausalConv3d
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform


class TestRocmVaeNativeConv3d(unittest.TestCase):
    """SGLANG_ROCM_VAE_NATIVE_CONV3D=1 keeps the original Conv3d weight and runs one
    F.conv3d per causal conv. It must match the default temporal-unfold path."""

    def _build(self, env_value):
        torch.manual_seed(0)
        conv = WanCausalConv3d(8, 16, 3, padding=1)
        env = dict(os.environ)
        env.pop("SGLANG_ROCM_VAE_NATIVE_CONV3D", None)
        if env_value is not None:
            env["SGLANG_ROCM_VAE_NATIVE_CONV3D"] = env_value
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                RocmPlatform._replace_conv3d_with_conv2d(conv, use_bf16=False), 1
            )
        return conv

    def test_native_conv3d_matches_unfold_path_in_fp32(self):
        unfold = self._build(None)
        native = self._build("1")
        self.assertTrue(torch.equal(native.weight, unfold.weight))
        self.assertFalse(hasattr(unfold, "_weight_3d"))
        self.assertTrue(hasattr(native, "_weight_3d"))

        torch.manual_seed(1)
        x = torch.randn(1, 8, 3, 12, 12)
        cache_x = torch.randn(1, 8, 2, 12, 12)
        residual = torch.randn(1, 16, 3, 12, 12)
        with torch.no_grad():
            for kwargs in ({}, {"residual": residual}, {"skip_bias": True}):
                with self.subTest(kwargs=list(kwargs)):
                    torch.testing.assert_close(
                        native(x, cache_x, **kwargs),
                        unfold(x, cache_x, **kwargs),
                        atol=1e-4,
                        rtol=1e-4,
                    )


if __name__ == "__main__":
    unittest.main()
