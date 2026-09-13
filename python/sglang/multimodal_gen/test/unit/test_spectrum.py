# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.cache.spectrum import (
    ChebyshevForecaster,
    SpectrumForecaster,
)


class TestSpectrumForecaster(unittest.TestCase):
    def test_chebyshev_fit_and_predict(self) -> None:
        """Predict returns the expected feature shape after fitting on prior steps."""
        forecaster = ChebyshevForecaster(
            M=2, K=10, lam=0.1, num_steps=10, feature_shape=(4, 8)
        )
        for step in range(6):
            forecaster.update(float(step), torch.randn(4, 8))
        predicted = forecaster.predict(6.0)
        self.assertEqual(predicted.shape, (4, 8))

    def test_chebyshev_fit_and_predict_bfloat16(self) -> None:
        """bfloat16 inputs produce bfloat16 predictions."""
        forecaster = ChebyshevForecaster(
            M=2, K=10, lam=0.1, num_steps=10, feature_shape=(4, 8)
        )
        for step in range(6):
            forecaster.update(float(step), torch.randn(4, 8, dtype=torch.bfloat16))
        predicted = forecaster.predict(6.0)
        self.assertEqual(predicted.shape, (4, 8))
        self.assertEqual(predicted.dtype, torch.bfloat16)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA autocast")
    def test_chebyshev_fit_and_predict_bfloat16_under_autocast(self) -> None:
        """CUDA bf16 autocast keeps prediction working and preserves bf16 output."""
        forecaster = ChebyshevForecaster(
            M=2, K=10, lam=0.1, num_steps=10, feature_shape=(4, 8)
        ).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for step in range(6):
                forecaster.update(
                    float(step), torch.randn(4, 8, device="cuda", dtype=torch.bfloat16)
                )
            predicted = forecaster.predict(6.0)
        self.assertEqual(predicted.shape, (4, 8))
        self.assertEqual(predicted.dtype, torch.bfloat16)

    def test_spectrum_blend_predict(self) -> None:
        """The blended Spectrum forecaster returns the expected output shape."""
        cheb = ChebyshevForecaster(
            M=2, K=10, lam=0.1, num_steps=10, feature_shape=(2, 3)
        )
        blend = SpectrumForecaster(cheb, taylor_order=1, w=0.5)
        for step in range(4):
            blend.update(float(step), torch.ones(2, 3) * step)
        out = blend.predict(4.0)
        self.assertEqual(out.shape, (2, 3))

    def test_chebyshev_prediction_error_is_bounded_on_smooth_signal(self) -> None:
        """Prediction error stays very low on a deterministic smooth signal."""

        def smooth_feature(step: float) -> torch.Tensor:
            # Linear trend should be modeled accurately by M=1 Chebyshev basis.
            base = 0.5 + 0.125 * step
            return torch.tensor(
                [[base, base + 0.1], [0.75 * base, -0.5 * base]],
                dtype=torch.float32,
            )

        forecaster = ChebyshevForecaster(
            M=1, K=16, lam=1e-6, num_steps=50, feature_shape=(2, 2)
        )
        for step in range(8):
            forecaster.update(float(step), smooth_feature(float(step)))

        target = smooth_feature(8.0)
        predicted = forecaster.predict(8.0)
        rel_l2 = torch.norm(predicted - target) / torch.norm(target)
        self.assertLess(rel_l2.item(), 1e-3)

    def test_chebyshev_tau_horizon_matches_reference(self) -> None:
        """Tau normalization matches the fixed 50-step horizon used by the reference."""
        f50 = ChebyshevForecaster(M=0, num_steps=50, feature_shape=(1,))
        f20 = ChebyshevForecaster(M=0, num_steps=20, feature_shape=(1,))
        t = torch.tensor([10.0])
        self.assertAlmostEqual(f50._taus(t).item(), -0.6, places=5)
        self.assertAlmostEqual(f20._taus(t).item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
