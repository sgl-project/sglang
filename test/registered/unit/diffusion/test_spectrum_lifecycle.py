# SPDX-License-Identifier: Apache-2.0
"""Spectrum request and CFG branch isolation regressions for #35053."""

import unittest
from types import SimpleNamespace

import torch

from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.runtime.cache.spectrum import SpectrumMixin
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# The cache package imports optional diffusion dependencies supplied by this runner.
register_cuda_ci(est_time=10, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class _SpectrumModel(SpectrumMixin):
    def __init__(self, prefix="wan"):
        self.prefix = prefix
        self._init_spectrum_state()


class TestSpectrumLifecycle(CustomTestCase):
    def setUp(self):
        self.params = SpectrumParams(
            warmup_steps=2, window_size=4.0, flex_window=0.5, m=1, history_size=8
        )

    def _forward(self, model, step, features, *, negative=False, do_cfg=True):
        batch = SimpleNamespace(
            enable_spectrum=True,
            spectrum_params=self.params,
            do_classifier_free_guidance=do_cfg,
            is_cfg_negative=negative,
            num_inference_steps=8,
            debug=True,
        )
        with set_forward_context(step, None, forward_batch=batch):
            actual_forward = model.begin_spectrum_step()
            if actual_forward:
                model.spectrum_record_features(features)
                return True, features
            return False, model.spectrum_predict_features(torch.zeros_like(features))

    def test_cfg_ranks_initialize_the_same_skip_schedule(self):
        """A negative-only rank must use the configured window from its first request."""
        positive, negative = _SpectrumModel(), _SpectrumModel()
        decisions = []
        for step in range(8):
            features = torch.full((2, 3), float(step + 1))
            positive_real, positive_output = self._forward(positive, step, features)
            negative_real, negative_output = self._forward(
                negative, step, features, negative=True
            )
            self.assertEqual(positive_real, negative_real)
            torch.testing.assert_close(positive_output, negative_output)
            self.assertEqual(
                positive.spectrum_curr_ws, negative.spectrum_curr_ws_negative
            )
            decisions.append(negative_real)
        self.assertIn(False, decisions)

    def test_reused_model_matches_fresh_model_after_request_restart(self):
        """Completed and interrupted requests must not leak forecasts or debug stats."""
        for prefix, negative, do_cfg in (
            ("wan", True, True),
            ("wan", False, True),
            ("wan", False, False),
            ("flux", False, False),
        ):
            for previous_steps in (7, 8):
                with self.subTest(
                    prefix=prefix,
                    negative=negative,
                    do_cfg=do_cfg,
                    previous_steps=previous_steps,
                ):
                    reused = _SpectrumModel(prefix)
                    for step in range(previous_steps):
                        self._forward(
                            reused,
                            step,
                            torch.full((2, 3), float(step + 20)),
                            negative=negative,
                            do_cfg=do_cfg,
                        )
                    previous_forecaster = reused._get_spectrum_forecaster()
                    fresh = _SpectrumModel(prefix)
                    for step in range(8):
                        # A new shape also makes stale history fail immediately.
                        features = torch.full((3, 4), float(step + 1))
                        actual, output = self._forward(
                            reused, step, features, negative=negative, do_cfg=do_cfg
                        )
                        expected_actual, expected_output = self._forward(
                            fresh, step, features, negative=negative, do_cfg=do_cfg
                        )
                        self.assertEqual(actual, expected_actual)
                        torch.testing.assert_close(output, expected_output)
                        self.assertEqual(
                            reused._get_spectrum_branch_state(),
                            fresh._get_spectrum_branch_state(),
                        )
                        suffix = "_negative" if negative else ""
                        for stat in (
                            "real_steps",
                            "skipped_steps",
                            "shadow_rel_l2_sum",
                            "shadow_rel_l2_count",
                        ):
                            attribute = f"spectrum_{stat}{suffix}"
                            self.assertEqual(
                                getattr(reused, attribute), getattr(fresh, attribute)
                            )
                        forecaster = reused._get_spectrum_forecaster()
                        self.assertIsNot(forecaster, previous_forecaster)
                        self.assertEqual(
                            forecaster.cheb._count,
                            fresh._get_spectrum_forecaster().cheb._count,
                        )

    def test_serial_cfg_keeps_branch_forecasts_independent(self):
        """Initializing either CFG branch must preserve the other branch's history."""
        for branches in ((False, True), (True, False)):
            with self.subTest(branches=branches):
                serial = _SpectrumModel()
                separate = {False: _SpectrumModel(), True: _SpectrumModel()}
                for step in range(8):
                    for negative in branches:
                        features = torch.full(
                            (3 if negative else 2, 2),
                            float(step + (10 if negative else 1)),
                        )
                        actual, output = self._forward(
                            serial, step, features, negative=negative
                        )
                        expected_actual, expected_output = self._forward(
                            separate[negative], step, features, negative=negative
                        )
                        self.assertEqual(actual, expected_actual)
                        torch.testing.assert_close(output, expected_output)
                self.assertIsNot(
                    serial.spectrum_forecaster, serial.spectrum_forecaster_negative
                )

                serial.reset_spectrum_state(self.params)
                for negative in branches:
                    serial.spectrum_is_cfg_negative = negative
                    self.assertIsNone(serial._get_spectrum_forecaster())
                    self.assertEqual(serial._get_spectrum_branch_state(), (0, 0, 4.0))

    def test_shared_cfg_state_is_not_reset_twice(self):
        """A shared counter and history must retain both serial step-zero forwards."""
        model = _SpectrumModel("flux")
        for negative in (False, True):
            features = torch.full((2, 3), 2.0 if negative else 1.0)
            actual, _ = self._forward(model, 0, features, negative=negative)
            self.assertTrue(actual)
        self.assertEqual(model.spectrum_cnt, 2)
        self.assertEqual(model.spectrum_forecaster.cheb._count, 2)
        _, history = model.spectrum_forecaster.cheb._recent(2)
        torch.testing.assert_close(history[0], torch.ones(6))
        torch.testing.assert_close(history[1], torch.full((6,), 2.0))


if __name__ == "__main__":
    unittest.main()
