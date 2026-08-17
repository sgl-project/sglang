# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.multimodal_gen.runtime.cache.spectrum import (
    ChebyshevForecaster,
    SpectrumContext,
    SpectrumForecaster,
    SpectrumMixin,
)


class _DummySpectrum(SpectrumMixin):
    prefix = "wan"

    def __init__(self) -> None:
        self._init_spectrum_state()


class _DummySharedSpectrum(SpectrumMixin):
    prefix = "flux"

    def __init__(self) -> None:
        self._init_spectrum_state()


class _SpectrumParams:
    window_size = 2.0
    flex_window = 0.75
    warmup_steps = 5
    history_size = 20
    m = 2
    lam = 0.1
    tau_num_steps = 50
    taylor_order = 1
    w = 1.0


def _spectrum_context(
    *, is_cfg_negative: bool, total_forward_steps: int = 50
) -> SpectrumContext:
    return SpectrumContext(
        current_step=0,
        num_inference_steps=total_forward_steps,
        total_forward_steps=total_forward_steps,
        do_cfg=True,
        is_cfg_negative=is_cfg_negative,
        spectrum_params=_SpectrumParams(),
        debug=False,
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


class TestSpectrumLifecycle(unittest.TestCase):
    def test_cfg_parallel_negative_branch_initializes_at_generation_start(
        self,
    ) -> None:
        model = _DummySpectrum()
        model.spectrum_cnt = 7
        model.spectrum_cnt_negative = 7
        model.spectrum_num_consecutive_cached_steps_negative = 3
        model.spectrum_curr_ws_negative = 9.0
        model.spectrum_real_steps_negative = 4
        model.spectrum_skipped_steps_negative = 2
        model.spectrum_shadow_rel_l2_sum_negative = 1.5
        model.spectrum_shadow_rel_l2_count_negative = 3
        model._get_spectrum_context = lambda: _spectrum_context(
            is_cfg_negative=True
        )

        self.assertTrue(model.begin_spectrum_step())

        self.assertEqual(model.spectrum_cnt_negative, 1)
        self.assertEqual(
            model.spectrum_curr_ws_negative, _SpectrumParams.window_size
        )
        self.assertEqual(model.spectrum_num_consecutive_cached_steps_negative, 0)
        self.assertEqual(model.spectrum_real_steps_negative, 1)
        self.assertEqual(model.spectrum_skipped_steps_negative, 0)
        self.assertEqual(model.spectrum_shadow_rel_l2_sum_negative, 0.0)
        self.assertEqual(model.spectrum_shadow_rel_l2_count_negative, 0)
        self.assertEqual(model.spectrum_cnt, 7)

    def test_cfg_parallel_negative_branch_drops_previous_forecaster(self) -> None:
        model = _DummySpectrum()
        previous_forecaster = object()
        model.spectrum_forecaster_negative = previous_forecaster
        model._get_spectrum_context = lambda: _spectrum_context(
            is_cfg_negative=True, total_forward_steps=1
        )

        self.assertTrue(model.begin_spectrum_step())

        self.assertIsNone(model.spectrum_forecaster_negative)

    def test_serial_cfg_branches_start_with_synchronized_counters(self) -> None:
        model = _DummySpectrum()
        context = _spectrum_context(is_cfg_negative=False)
        model._get_spectrum_context = lambda: context

        self.assertTrue(model.begin_spectrum_step())
        context = _spectrum_context(is_cfg_negative=True)
        self.assertTrue(model.begin_spectrum_step())

        self.assertEqual(model.spectrum_cnt, 1)
        self.assertEqual(model.spectrum_cnt_negative, 1)
        self.assertEqual(model.spectrum_curr_ws, _SpectrumParams.window_size)
        self.assertEqual(
            model.spectrum_curr_ws_negative, _SpectrumParams.window_size
        )

    def test_serial_cfg_shared_counter_is_not_reset_by_negative_branch(self) -> None:
        model = _DummySharedSpectrum()
        context = _spectrum_context(
            is_cfg_negative=False, total_forward_steps=100
        )
        model._get_spectrum_context = lambda: context

        self.assertTrue(model.begin_spectrum_step())
        context = _spectrum_context(
            is_cfg_negative=True, total_forward_steps=100
        )
        self.assertTrue(model.begin_spectrum_step())

        self.assertEqual(model.spectrum_cnt, 2)


if __name__ == "__main__":
    unittest.main()
