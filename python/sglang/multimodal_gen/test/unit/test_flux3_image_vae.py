# SPDX-License-Identifier: Apache-2.0
"""Tests for the FLUX.3 image VAE (:class:`Flux3ImageVAE`): fused GroupNorm+SiLU
wiring and spatial-parallel decode wiring (GPU-free, no checkpoint).
"""

import unittest
from unittest import mock

import torch

from sglang.multimodal_gen.configs.models.vaes.flux3_image import Flux3ImageVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import (
    Flux3ImageVAE,
    Upsample,
    enable_spatial_parallel_decoder,
)


def _meta_vae() -> Flux3ImageVAE:
    with torch.device("meta"):
        return Flux3ImageVAE(Flux3ImageVAEConfig())


class TestFlux3ImageVAE(unittest.TestCase):
    def test_fused_norm_act_is_wired(self):
        """apply_group_norm_silu only fuses when handed an nn.SiLU; without it
        every GroupNorm -> swish chain silently runs eager."""
        vae = _meta_vae()
        for module in (vae.encoder, vae.decoder, vae.encoder.mid.block_1):
            self.assertIsInstance(module.act, torch.nn.SiLU)

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
