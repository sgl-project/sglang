# SPDX-License-Identifier: Apache-2.0
"""DenoisingStage resets every DiT's cache state at the start of a request.

A boundary expert (Wan2.2 ``transformer_2``) only runs the low-noise steps, so it
never sees denoising step 0, where the DiTs reset TeaCache and Spectrum themselves.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.configs.sample.spectrum import SpectrumParams
from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.cache.spectrum import SpectrumMixin
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.test.test_utils import CustomTestCase

STEPS = 8
FIRST_LOW_NOISE_STEP = 4  # where the second expert takes over


class _Expert(SpectrumMixin, TeaCacheMixin):
    def __init__(self):
        self.prefix = "wan"
        self._init_spectrum_state()
        self._init_teacache_state()


class TestDenoisingCacheReset(CustomTestCase):
    def setUp(self):
        server_args = patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=SimpleNamespace(enable_cfg_parallel=False),
        )
        server_args.start()
        self.addCleanup(server_args.stop)

    def _batch(self, spectrum=False, teacache=False):
        return SimpleNamespace(
            enable_spectrum=spectrum,
            spectrum_params=SpectrumParams(
                warmup_steps=1, window_size=2.0, flex_window=0.5, m=1
            ),
            enable_teacache=teacache,
            teacache_params=TeaCacheParams(
                teacache_thresh=0.5, coefficients=[1.0, 0.0], start_skipping=1
            ),
            do_classifier_free_guidance=False,
            is_cfg_negative=False,
            num_inference_steps=STEPS,
            debug=False,
        )

    def _stage(self, expert, batch):
        stage = DenoisingStage.__new__(DenoisingStage)
        stage.transformer, stage.transformer_2 = MagicMock(spec=[]), expert
        stage._reset_dit_cache_states(batch)

    def _spectrum_request(self, expert, batch, shape):
        """Run the second expert's steps; return (ran blocks, output) per step."""
        outputs = []
        for step in range(FIRST_LOW_NOISE_STEP, STEPS):
            features = torch.full(shape, float(step))
            with set_forward_context(step, None, forward_batch=batch):
                if expert.begin_spectrum_step():
                    expert.spectrum_record_features(features)
                    outputs.append((True, features))
                else:
                    outputs.append(
                        (
                            False,
                            expert.spectrum_predict_features(
                                torch.zeros_like(features)
                            ),
                        )
                    )
        return outputs

    def _teacache_request(self, expert, batch, offset):
        decisions = []
        for step in range(FIRST_LOW_NOISE_STEP, STEPS):
            inputs = torch.full((2,), offset + step * 0.01)
            with set_forward_context(step, None, batch):
                skip = WanTransformer3DModel.should_skip_forward_for_cached_states(
                    expert, timestep_proj=inputs, temb=inputs
                )
            expert.cnt += 1
            decisions.append(skip)
        return decisions

    def test_second_expert_spectrum_restarts_like_fresh(self):
        """A changed feature shape must not hit the previous request's forecaster."""
        batch = self._batch(spectrum=True)
        reused, fresh = _Expert(), _Expert()
        self._spectrum_request(reused, batch, (2, 3))
        self._stage(reused, batch)
        self._stage(fresh, batch)
        for (ran, out), (fresh_ran, fresh_out) in zip(
            self._spectrum_request(reused, batch, (3, 4)),
            self._spectrum_request(fresh, batch, (3, 4)),
            strict=True,
        ):
            self.assertEqual(ran, fresh_ran)
            torch.testing.assert_close(out, fresh_out)

    def test_second_expert_teacache_restarts_like_fresh(self):
        """Counters, residuals and L1 history must not carry over."""
        batch = self._batch(teacache=True)
        reused, fresh = _Expert(), _Expert()
        self._teacache_request(reused, batch, offset=1.0)
        reused.previous_residual = torch.ones(2)
        self._stage(reused, batch)
        self.assertEqual(reused.cnt, 0)
        self.assertIsNone(reused.previous_modulated_input)
        self.assertIsNone(reused.previous_residual)
        self.assertEqual(reused.accumulated_rel_l1_distance, 0.0)
        self.assertEqual(
            self._teacache_request(reused, batch, offset=5.0),
            self._teacache_request(fresh, batch, offset=5.0),
        )

    def test_disabled_caches_and_plain_modules_are_left_alone(self):
        """Only enabled caches reset, and modules without cache state are skipped."""
        expert = MagicMock()
        self._stage(expert, self._batch())
        expert.reset_teacache_state.assert_not_called()
        expert.reset_spectrum_state.assert_not_called()
        batch = self._batch(spectrum=True, teacache=True)
        self._stage(expert, batch)
        expert.reset_teacache_state.assert_called_once_with()
        expert.reset_spectrum_state.assert_called_once_with(batch.spectrum_params)


if __name__ == "__main__":
    unittest.main()
