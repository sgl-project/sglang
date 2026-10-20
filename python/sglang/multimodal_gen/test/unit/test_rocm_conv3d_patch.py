# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.models.vaes.wanvae import WanCausalConv3d
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform


class TestRocmConv3dPatchEpilogueKwargs(unittest.TestCase):
    """The Wan VAE decoder calls its causal convs with residual= and skip_bias=;
    the ROCm Conv3d->Conv2d replacement must accept them and match the original."""

    def test_patched_conv_matches_original_with_residual_and_skip_bias(self):
        torch.manual_seed(0)
        ref = WanCausalConv3d(8, 16, 3, padding=1)
        patched = WanCausalConv3d(8, 16, 3, padding=1)
        patched.load_state_dict(ref.state_dict())
        self.assertEqual(RocmPlatform._replace_conv3d_with_conv2d(patched), 1)

        x = torch.randn(1, 8, 3, 12, 12)
        cache_x = torch.randn(1, 8, 2, 12, 12)
        residual = torch.randn(1, 16, 3, 12, 12)
        for kwargs in ({"residual": residual}, {"skip_bias": True}):
            with self.subTest(kwargs=list(kwargs)), torch.no_grad():
                torch.testing.assert_close(
                    patched(x, cache_x, **kwargs), ref(x, cache_x, **kwargs)
                )


if __name__ == "__main__":
    unittest.main()
