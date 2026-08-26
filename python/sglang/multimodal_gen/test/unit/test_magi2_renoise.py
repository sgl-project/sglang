# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.magi2 import (
    renoise as magi2_renoise,
)


class TestZeroSnrSigmas(unittest.TestCase):
    def test_schedule_descends_to_zero_signal(self):
        sigmas = magi2_renoise.zero_snr_sigmas()

        self.assertEqual(sigmas.shape, (magi2_renoise.NUM_TRAIN_TIMESTEPS,))
        self.assertTrue((sigmas.diff() < 0).all())
        self.assertEqual(float(sigmas[-1]), 0.0)
        self.assertAlmostEqual(float(sigmas[0]), 0.999575, places=5)

    def test_production_index_selects_the_reference_signal_coefficient(self):
        # The order is only observable through the index the refiner actually
        # uses. An ascending table holds 0.152 here, which renoises to 98.8%
        # noise instead of 54.4% and quietly flattens the output.
        sigmas = magi2_renoise.zero_snr_sigmas()
        self.assertAlmostEqual(float(sigmas[220]), 0.838910, places=5)


class TestUpsampleLatent(unittest.TestCase):
    def test_time_axis_maps_t_frames_to_two_t_minus_one(self):
        for frames in (1, 3, 8):
            with self.subTest(frames=frames):
                out = magi2_renoise.upsample_latent(
                    torch.randn(1, 4, frames, 8, 8), height=16, width=16
                )
                self.assertEqual(out.shape, (1, 4, 2 * frames - 1, 16, 16))

    def test_aligned_corners_keep_original_frames_on_even_slots(self):
        latent = torch.randn(1, 2, 4, 4, 4)
        out = magi2_renoise.upsample_latent(latent, height=4, width=4)
        self.assertTrue(torch.allclose(out[:, :, ::2], latent, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
