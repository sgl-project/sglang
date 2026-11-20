# SPDX-License-Identifier: Apache-2.0
import unittest

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
    WanAttentionBlock,
    WanCausalConv3d,
)
from sglang.multimodal_gen.runtime.platforms.rocm import RocmPlatform
from sglang.srt.utils import is_gfx1250_supported


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

    @unittest.skipUnless(is_gfx1250_supported(), "NHWC conv path is gfx1250-only")
    def test_channels_last_weight_runs_nhwc_and_keeps_channels_last_3d(self):
        torch.manual_seed(0)
        ref = WanCausalConv3d(8, 16, 3, padding=1)
        patched = WanCausalConv3d(8, 16, 3, padding=1)
        patched.load_state_dict(ref.state_dict())
        patched.weight.data = patched.weight.data.contiguous(
            memory_format=torch.channels_last_3d
        )
        self.assertEqual(RocmPlatform._replace_conv3d_with_conv2d(patched), 1)
        self.assertTrue(
            patched._weight_2d.is_contiguous(memory_format=torch.channels_last)
        )

        x = torch.randn(1, 8, 3, 12, 12).contiguous(
            memory_format=torch.channels_last_3d
        )
        cache_x = torch.randn(1, 8, 2, 12, 12)
        residual = torch.randn(1, 16, 3, 12, 12)
        for kwargs in ({}, {"residual": residual}, {"skip_bias": True}):
            with self.subTest(kwargs=list(kwargs)), torch.no_grad():
                out = patched(x, cache_x, **kwargs)
                torch.testing.assert_close(out, ref(x, cache_x, **kwargs))
                if not kwargs:
                    self.assertTrue(
                        out.is_contiguous(memory_format=torch.channels_last_3d)
                    )


class TestWanVaeMathAttention(unittest.TestCase):
    """With ``_sgl_math_sdpa`` set (gfx1250), the VAE attention block computes
    the math SDPA ops explicitly; it must equal the math backend bit for bit."""

    def test_explicit_math_attention_is_bit_exact(self):
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        block = WanAttentionBlock(64).to(device)
        x = torch.randn(1, 64, 2, 12, 16, device=device)
        with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
            ref = block(x)
            block._sgl_math_sdpa = True
            out = block(x)
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
