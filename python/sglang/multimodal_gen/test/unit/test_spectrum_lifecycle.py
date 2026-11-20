# SPDX-License-Identifier: Apache-2.0
"""Spectrum request and CFG branch isolation regressions for #35053."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.runtime.cache.spectrum import SpectrumMixin
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.test.test_utils import CustomTestCase

# Any prefix outside SpectrumMixin._CFG_SUPPORTED_PREFIXES shares one CFG state.
SHARED = "shared"


class _SpectrumModel(SpectrumMixin):
    def __init__(self, prefix="wan"):
        self.prefix = prefix
        self._init_spectrum_state()


class TestSpectrumLifecycle(CustomTestCase):
    def setUp(self):
        self.params = SpectrumParams(
            warmup_steps=2, window_size=4.0, flex_window=0.5, m=1, history_size=8
        )
        self.server_args = SimpleNamespace(enable_cfg_parallel=False)
        server_args_patch = patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=self.server_args,
        )
        server_args_patch.start()
        self.addCleanup(server_args_patch.stop)

    def _forward(
        self, model, step, features, *, negative=False, do_cfg=True, cfg_scale=None
    ):
        batch = SimpleNamespace(
            enable_spectrum=True,
            spectrum_params=self.params,
            do_classifier_free_guidance=do_cfg,
            is_cfg_negative=negative,
            num_inference_steps=8,
            debug=True,
            guidance_scale=cfg_scale,
        )
        with set_forward_context(step, None, forward_batch=batch):
            actual_forward = model.begin_spectrum_step()
            if actual_forward:
                model.spectrum_record_features(features)
                return True, features
            return False, model.spectrum_predict_features(torch.zeros_like(features))

    def test_cfg_ranks_initialize_the_same_skip_schedule(self):
        """A negative-only rank must use the configured window from its first request."""
        self.server_args.enable_cfg_parallel = True
        for prefix in ("wan", "flux", SHARED):
            with self.subTest(prefix=prefix):
                positive, negative = _SpectrumModel(prefix), _SpectrumModel(prefix)
                decisions = []
                for step in range(8):
                    features = torch.full((2, 3), float(step + 1))
                    positive_real, positive_output = self._forward(
                        positive, step, features
                    )
                    negative_real, negative_output = self._forward(
                        negative, step, features, negative=True
                    )
                    self.assertEqual(positive_real, negative_real)
                    torch.testing.assert_close(positive_output, negative_output)
                    self.assertEqual(
                        positive._get_spectrum_branch_state(),
                        negative._get_spectrum_branch_state(),
                    )
                    decisions.append(negative_real)
                self.assertIn(False, decisions)
                # Each rank runs one forward per step, so both wrap after the run.
                self.assertEqual(negative._get_spectrum_branch_state(), (0, 0, 4.0))

    def test_flux_true_cfg_falls_back_to_real_forwards_through_final_step(self):
        from sglang.multimodal_gen.runtime.models.dits.flux import (
            FluxTransformer2DModel,
        )

        # The real model's cache policy does not need transformer weights.
        model = FluxTransformer2DModel.__new__(FluxTransformer2DModel)
        torch.nn.Module.__init__(model)
        model.prefix = "flux"
        model._init_spectrum_state()
        with self.assertLogs(
            "sglang.multimodal_gen.runtime.models.dits.flux", level="WARNING"
        ) as logs:
            for step in range(8):
                for negative in (False, True):
                    with self.subTest(step=step, negative=negative):
                        # Curvature makes even the final local linear forecast
                        # differ from the real output.
                        features = torch.full(
                            (2, 3), float((step + 1) ** 2 + 10 * negative)
                        )
                        real, output = self._forward(
                            model, step, features, negative=negative, cfg_scale=4.0
                        )
                        self.assertTrue(real)
                        torch.testing.assert_close(output, features, atol=0, rtol=0)
        self.assertEqual(len(logs.records), 1)
        # The fallback is the uncached path: no forecaster is ever built.
        self.assertIsNone(model.spectrum_forecaster)
        self.assertIsNone(model.spectrum_forecaster_negative)
        # Request-owned parameters must not be rewritten by the model policy.
        self.assertEqual(self.params.warmup_steps, 2)
        self.assertEqual(self.params.w, 1.0)
        self.assertEqual(self.params.history_size, 8)

        # Embedded guidance without true CFG retains the existing schedule.
        reference = _SpectrumModel("flux")
        for step in range(8):
            features = torch.full((2, 3), float(step + 1))
            actual = self._forward(model, step, features, do_cfg=False)
            expected = self._forward(reference, step, features, do_cfg=False)
            self.assertEqual(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)

    def test_flux_true_cfg_up_to_two_uses_bounded_schedule(self):
        from sglang.multimodal_gen.runtime.models.dits.flux import (
            FluxTransformer2DModel,
        )

        model = FluxTransformer2DModel.__new__(FluxTransformer2DModel)
        torch.nn.Module.__init__(model)
        model.prefix = "flux"
        model._init_spectrum_state()
        # 8 steps: real for round(0.4 * 8) = 3 warmup steps, then every other
        # step with the window fixed at 2, and the last 2 steps real.
        expected = [True, True, True, False, True, False, True, True]
        for scale in (1.5, 2.0):
            for negative in (False, True):
                with self.subTest(scale=scale, negative=negative):
                    actual = [
                        self._forward(
                            model,
                            step,
                            torch.full((2, 3), float(step + 1)),
                            negative=negative,
                            cfg_scale=scale,
                        )[0]
                        for step in range(8)
                    ]
                    self.assertEqual(actual, expected)
        # Request-owned parameters must not be rewritten by the model policy.
        self.assertEqual(self.params.warmup_steps, 2)
        self.assertEqual(self.params.window_size, 4.0)
        self.assertEqual(self.params.final_real_steps, 0)

    def test_final_real_steps_keeps_last_steps_real(self):
        self.params = SpectrumParams(
            warmup_steps=2, window_size=4.0, flex_window=0.0, final_real_steps=2
        )
        model = _SpectrumModel("wan")
        actual = [
            self._forward(model, step, torch.full((2, 3), float(step + 1)))[0]
            for step in range(8)
        ]
        self.assertEqual(actual, [True, True, False, False, False, True, True, True])
        with self.assertRaises(ValueError):
            SpectrumParams(final_real_steps=-1)

    def test_reused_model_matches_fresh_model_after_request_restart(self):
        """Completed and interrupted requests must not leak forecasts or debug stats."""
        for prefix, negative, do_cfg, parallel in (
            ("wan", True, True, True),
            ("wan", False, True, False),
            ("wan", False, False, False),
            ("flux", True, True, True),
            (SHARED, False, False, False),
            (SHARED, True, True, True),
        ):
            for previous_steps in (7, 8):
                with self.subTest(
                    prefix=prefix,
                    negative=negative,
                    do_cfg=do_cfg,
                    parallel=parallel,
                    previous_steps=previous_steps,
                ):
                    self.server_args.enable_cfg_parallel = parallel
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
                        separate = negative and reused._spectrum_supports_cfg_cache
                        suffix = "_negative" if separate else ""
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
        model = _SpectrumModel(SHARED)
        for negative in (False, True):
            features = torch.full((2, 3), 2.0 if negative else 1.0)
            actual, _ = self._forward(model, 0, features, negative=negative)
            self.assertTrue(actual)
        self.assertEqual(model.spectrum_cnt, 2)
        self.assertEqual(model.spectrum_forecaster.cheb._count, 2)
        _, history = model.spectrum_forecaster.cheb._recent(2)
        torch.testing.assert_close(history[0], torch.ones(6))
        torch.testing.assert_close(history[1], torch.full((6,), 2.0))

    def test_total_forward_steps_count_local_forwards(self):
        """A shared counter sees both serial CFG branches but one branch per rank."""
        for separate, cfg_parallel, expected in (
            (False, False, 16),
            (False, True, 8),
            (True, False, 8),
            (True, True, 8),
        ):
            with self.subTest(separate=separate, cfg_parallel=cfg_parallel):
                self.assertEqual(
                    self.params.get_total_forward_steps(
                        8, True, separate, cfg_parallel=cfg_parallel
                    ),
                    expected,
                )
                self.assertEqual(
                    self.params.get_total_forward_steps(
                        8, False, separate, cfg_parallel=cfg_parallel
                    ),
                    8,
                )

    def test_flux_cfg_gating_keeps_cond_schedule(self):
        """Dropping the uncond forward after the gate must not shift cond decisions."""
        gate_step = 3
        gated, ungated = _SpectrumModel("flux"), _SpectrumModel("flux")
        for step in range(8):
            for negative in (False, True):
                features = torch.full((2, 3), float(step + (10 if negative else 1)))
                expected = self._forward(ungated, step, features, negative=negative)
                if negative and step >= gate_step:
                    continue  # CFG gating reuses the cached cond-uncond delta
                actual, output = self._forward(gated, step, features, negative=negative)
                if not negative:
                    self.assertEqual(actual, expected[0], f"step={step}")
                    torch.testing.assert_close(output, expected[1])
        gated.spectrum_is_cfg_negative = False
        # The cond counter saw one forward per step and wrapped at the end.
        self.assertEqual(gated._get_spectrum_branch_state(), (0, 0, 4.0))
        self.assertEqual(gated.spectrum_skipped_steps, ungated.spectrum_skipped_steps)


if __name__ == "__main__":
    unittest.main()
