# SPDX-License-Identifier: Apache-2.0
"""Regression tests for request and CFG branch isolation in TeaCache."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.sample.teacache import TeaCacheParams
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheMixin
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanTransformer3DModel
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="diffusion-unit-1-gpu-h100")


class TestTeaCache(CustomTestCase):
    def setUp(self):
        self.params = TeaCacheParams(
            teacache_thresh=0.5, coefficients=[1.0, 0.0], start_skipping=1
        )
        self.batch = SimpleNamespace(
            enable_teacache=True,
            teacache_params=self.params,
            num_inference_steps=5,
            do_classifier_free_guidance=True,
            is_cfg_negative=False,
        )
        self.server_args = SimpleNamespace(enable_cfg_parallel=False)
        server_args_patch = patch(
            "sglang.multimodal_gen.runtime.server_args.get_global_server_args",
            return_value=self.server_args,
        )
        server_args_patch.start()
        self.addCleanup(server_args_patch.stop)

    def _new_cache(self):
        cache = TeaCacheMixin()
        cache.prefix = "wan"
        cache._init_teacache_state()
        return cache

    def _step(self, cache, timestep, negative, modulated_input):
        self.batch.is_cfg_negative = negative
        with set_forward_context(timestep, None, self.batch):
            should_skip = WanTransformer3DModel.should_skip_forward_for_cached_states(
                cache, timestep_proj=modulated_input, temb=modulated_input
            )
        cache.cnt += 1
        return not should_skip

    def test_serial_cfg_preserves_both_branches_at_first_timestep(self):
        """The negative first step must not discard the positive branch cache."""
        cache = self._new_cache()
        positive = torch.tensor([1.0, 2.0])
        negative = torch.tensor([4.0, 8.0])
        residual = torch.tensor([0.25, 0.5])
        self.assertTrue(self._step(cache, 0, False, positive))
        WanTransformer3DModel.maybe_cache_states(
            cache, (positive + residual).unsqueeze(0), positive
        )
        self.assertTrue(self._step(cache, 0, True, negative))
        WanTransformer3DModel.maybe_cache_states(
            cache, (negative + residual * 2).unsqueeze(0), negative
        )
        self.assertEqual(cache.cnt, 2)
        torch.testing.assert_close(cache.previous_residual, residual)
        torch.testing.assert_close(cache.previous_modulated_input, positive)
        torch.testing.assert_close(cache.previous_modulated_input_negative, negative)
        self.assertFalse(self._step(cache, 1, False, positive))
        torch.testing.assert_close(
            WanTransformer3DModel.retrieve_cached_states(cache, positive),
            positive + residual,
        )
        self.assertFalse(self._step(cache, 1, True, negative))
        torch.testing.assert_close(
            WanTransformer3DModel.retrieve_cached_states(cache, negative),
            negative + residual * 2,
        )

    def test_new_request_clears_completed_or_interrupted_state(self):
        """New requests must not inherit counters, residuals or L1 history."""
        for do_cfg, parallel, branches in (
            (True, False, (False, True)),
            (True, True, (False,)),
            (True, True, (True,)),
            (False, False, (False,)),
        ):
            for num_steps in (2, self.batch.num_inference_steps):
                with self.subTest(
                    do_cfg=do_cfg, parallel=parallel, branches=branches, steps=num_steps
                ):
                    self.server_args.enable_cfg_parallel = parallel
                    self.batch.do_classifier_free_guidance = do_cfg
                    cache = self._new_cache()
                    for timestep in range(num_steps):
                        for negative in branches:
                            self._step(
                                cache,
                                timestep,
                                negative,
                                torch.tensor([1.0 + timestep * 0.01]),
                            )
                    cache.previous_residual = torch.tensor([3.0])
                    cache.previous_residual_negative = torch.tensor([7.0])
                    self.assertGreater(cache.cnt, 0)
                    active_distance = (
                        cache.accumulated_rel_l1_distance_negative
                        if branches == (True,)
                        else cache.accumulated_rel_l1_distance
                    )
                    if num_steps == 2:
                        self.assertGreater(active_distance, 0)

                    self.batch.is_cfg_negative = branches[0]
                    with set_forward_context(0, None, self.batch):
                        ctx = cache._get_teacache_context()
                    self.assertEqual(ctx.is_cfg_negative, branches[0])
                    self.assertEqual(cache.cnt, 0)
                    self.assertIsNone(cache.previous_modulated_input)
                    self.assertIsNone(cache.previous_modulated_input_negative)
                    self.assertIsNone(cache.previous_residual)
                    self.assertIsNone(cache.previous_residual_negative)
                    self.assertEqual(cache.accumulated_rel_l1_distance, 0.0)
                    self.assertEqual(cache.accumulated_rel_l1_distance_negative, 0.0)

                    fresh = self._new_cache()
                    for timestep in range(2):
                        for negative in branches:
                            inp = torch.tensor([10.0 + negative + timestep * 0.01])
                            self.assertEqual(
                                self._step(cache, timestep, negative, inp),
                                self._step(fresh, timestep, negative, inp),
                            )
                    self.assertEqual(cache.cnt, fresh.cnt)

    def test_wan_skip_window_matches_timesteps_on_each_cfg_rank(self):
        """Wan must pass CFG topology through to its local skip-window accounting."""
        self.params.start_skipping = 2
        self.params.end_skipping = -1
        for parallel, branches in (
            (False, (False, True)),
            (True, (False,)),
            (True, (True,)),
        ):
            with self.subTest(parallel=parallel, branches=branches):
                self.server_args.enable_cfg_parallel = parallel
                cache = self._new_cache()
                for timestep, expected_calc in enumerate(
                    (True, True, False, False, True)
                ):
                    for negative in branches:
                        self.assertEqual(
                            self._step(cache, timestep, negative, torch.ones(2)),
                            expected_calc,
                            f"timestep={timestep}, negative={negative}",
                        )

    def test_skip_boundaries_count_local_forwards(self):
        """CFG parallel has one local forward per timestep, serial CFG has two."""
        for start, end, expected in ((5, -1, (5, 49)), (0.1, 0.8, (5, 40))):
            params = TeaCacheParams(start_skipping=start, end_skipping=end)
            with self.subTest(start=start, end=end):
                self.assertEqual(params.get_skip_boundaries(50, False), expected)
                self.assertEqual(
                    params.get_skip_boundaries(50, True),
                    (expected[0] * 2, expected[1] * 2),
                )
                for do_cfg in (False, True):
                    self.assertEqual(
                        params.get_skip_boundaries(50, do_cfg, cfg_parallel=True),
                        expected,
                    )


if __name__ == "__main__":
    unittest.main()
