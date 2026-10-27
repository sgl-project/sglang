# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.3 image VAE (:class:`Flux3ImageVAE`): fused GroupNorm+SiLU
wiring and spatial-parallel decode wiring (GPU-free, no checkpoint).
"""

import unittest
import weakref
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.vaes.flux3_image import Flux3ImageVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import (
    VaeFastPathGate,
    use_vae_fast_path,
)
from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
    AttnBlock,
    Encoder,
    Flux3ImageVAE,
    Upsample,
    enable_spatial_parallel_decoder,
)


def _meta_vae() -> Flux3ImageVAE:
    with torch.device("meta"):
        return Flux3ImageVAE(Flux3ImageVAEConfig())


class TestFlux3ImageVAE(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_memory_efficient_attention_keeps_fp32_full_context(self):
        torch.manual_seed(0)
        block = AttnBlock(512, strided=True).cuda().eval()
        block.fast_path_gate = VaeFastPathGate()
        x = torch.randn(1, 512, 129, 129, device="cuda")
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.inference_mode():
                expected = block(x)
                block.fast_path_gate.enabled = True
                actual = block(x)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32

    def test_encoder_releases_dead_activations(self):
        encoder = Encoder(32, 3, 32, [1, 1], 1, 4).eval()
        activations = []
        encoder.conv_in.register_forward_hook(
            lambda _module, _args, output: activations.append(weakref.ref(output))
        )
        encoder.mid.block_1.register_forward_pre_hook(
            lambda _module, _args: self.assertIsNone(activations[0]())
        )
        with torch.inference_mode():
            encoder(torch.randn(1, 3, 32, 32))

    def test_fused_norm_act_is_wired(self):
        """apply_group_norm_silu only fuses when handed an nn.SiLU; without it
        every GroupNorm -> swish chain silently runs eager."""
        vae = _meta_vae()
        for module in (vae.encoder, vae.decoder, vae.encoder.mid.block_1):
            self.assertIsInstance(module.act, torch.nn.SiLU)
        with use_vae_fast_path(vae, True):
            self.assertTrue(vae.decoder.mid.attn_1.fast_path_gate.enabled)
        self.assertFalse(vae.decoder.mid.attn_1.fast_path_gate.enabled)

    def test_enable_spatial_parallel_swaps_convs(self):
        vae = _meta_vae()
        # No decode-parallel group in a plain test process: serial decoder.
        self.assertFalse(vae._spatial_parallel)
        before = set(vae.decoder.state_dict().keys())
        n_upsamples = sum(1 for m in vae.decoder.modules() if isinstance(m, Upsample))

        # Fake a world>1 decode group so SpatialParallelConv2d constructs
        # without real distributed init.
        pc = "sglang.multimodal_gen.runtime.layers.parallel_conv"
        with (
            mock.patch(f"{pc}.get_decode_parallel_rank", return_value=0),
            mock.patch(f"{pc}.get_decode_parallel_world_size", return_value=2),
            torch.device("meta"),
        ):
            returned = enable_spatial_parallel_decoder(vae.decoder)

        self.assertEqual(returned, n_upsamples)
        self.assertEqual(returned, 3)  # ch_mult=[1,2,4,4] -> 3 upsamples
        # state_dict keys unchanged (SpatialParallelConv2d subclasses nn.Conv2d).
        self.assertEqual(before, set(vae.decoder.state_dict().keys()))
        # every plain Conv2d became SpatialParallelConv2d.
        self.assertEqual(
            sum(1 for m in vae.decoder.modules() if type(m) is torch.nn.Conv2d), 0
        )


if __name__ == "__main__":
    unittest.main()
