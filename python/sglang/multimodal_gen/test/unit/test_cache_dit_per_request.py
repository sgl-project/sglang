# SPDX-License-Identifier: Apache-2.0
"""Per-request Cache-DiT: request-param validation and the mount/refresh/
unmount transitions in DenoisingStage (CPU-only, mount boundary patched)."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    cache_dit_overrides_key,
    resolve_cache_dit_request_overrides,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    denoising as denoising_module,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class TestResolveCacheDitRequestOverrides(unittest.TestCase):
    def test_none_returns_empty_dict(self):
        self.assertEqual(resolve_cache_dit_request_overrides(None), {})

    def test_valid_overrides_are_copied(self):
        raw = {
            "residual_diff_threshold": 0.12,
            "scm_preset": "fast",
            "secondary": {"max_warmup_steps": 2},
        }
        resolved = resolve_cache_dit_request_overrides(raw)
        self.assertEqual(resolved, raw)
        self.assertIsNot(resolved, raw)
        self.assertIsNot(resolved["secondary"], raw["secondary"])

    def test_non_dict_raises(self):
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            resolve_cache_dit_request_overrides("fast")

    def test_unknown_key_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown cache_dit_params keys"):
            resolve_cache_dit_request_overrides({"residual_diff_thresh": 0.1})

    def test_secondary_unknown_key_raises(self):
        with self.assertRaisesRegex(ValueError, "secondary"):
            resolve_cache_dit_request_overrides({"secondary": {"scm_preset": "fast"}})

    def test_secondary_non_dict_raises(self):
        with self.assertRaisesRegex(ValueError, "secondary"):
            resolve_cache_dit_request_overrides({"secondary": 3})

    def test_overrides_key_detects_changes(self):
        key_a = cache_dit_overrides_key({"Fn_compute_blocks": 1, "scm_preset": "fast"})
        key_b = cache_dit_overrides_key({"scm_preset": "fast", "Fn_compute_blocks": 1})
        key_c = cache_dit_overrides_key({"Fn_compute_blocks": 2, "scm_preset": "fast"})
        self.assertEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_c)

    def test_overrides_key_freezes_nested_values(self):
        key = cache_dit_overrides_key(
            {"scm_compute_bins": [4, 2], "secondary": {"Bn_compute_blocks": 1}}
        )
        hash(key)  # must be hashable / comparable


def _batch(
    *,
    enable_cache_dit=None,
    cache_dit_params=None,
    is_warmup=False,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_warmup=is_warmup,
        do_classifier_free_guidance=False,
        sampling_params=SimpleNamespace(
            enable_cache_dit=enable_cache_dit,
            cache_dit_params=cache_dit_params,
        ),
    )


class TestPerRequestCacheDitTransitions(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.stage.server_args = SimpleNamespace(
            enable_breakable_cuda_graph=False,
            enable_torch_compile=False,
        )
        self.stage.transformer = object()
        self.stage.transformer_2 = None
        self.stage._cache_dit_enabled = False
        self.stage._cached_num_steps = None
        self.stage._cache_dit_request_overrides = {}
        self.stage._cache_dit_active_key = None

        self.enable_calls = []
        self.disable_calls = []
        self.refresh_calls = []

        def fake_enable(transformer, config, **kwargs):
            self.enable_calls.append(config)
            return transformer

        def fake_disable(transformer):
            self.disable_calls.append(transformer)
            return transformer

        def fake_refresh(transformer, num_inference_steps, scm_preset=None, **kwargs):
            self.refresh_calls.append(num_inference_steps)

        patchers = [
            patch.object(denoising_module, "enable_cache_on_transformer", fake_enable),
            patch.object(
                denoising_module, "disable_cache_on_transformer", fake_disable
            ),
            patch.object(
                denoising_module, "refresh_context_on_transformer", fake_refresh
            ),
            patch.object(denoising_module, "get_world_size", return_value=1),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_request_enable_mounts_without_env_default(self):
        self.stage._maybe_enable_cache_dit(8, _batch(enable_cache_dit=True))
        self.assertEqual(len(self.enable_calls), 1)
        self.assertTrue(self.stage._cache_dit_enabled)
        self.assertEqual(self.stage._cache_dit_active_key, cache_dit_overrides_key({}))

    def test_server_default_off_unmounts_after_request_enable(self):
        self.stage._maybe_enable_cache_dit(8, _batch(enable_cache_dit=True))
        self.stage._maybe_enable_cache_dit(8, _batch(enable_cache_dit=None))
        self.assertEqual(len(self.disable_calls), 1)
        self.assertFalse(self.stage._cache_dit_enabled)
        self.assertIsNone(self.stage._cache_dit_active_key)

    def test_explicit_disable_wins_over_env_default(self):
        with patch.object(self.stage, "_cache_dit_requested", return_value=True):
            self.stage._maybe_enable_cache_dit(8, _batch(enable_cache_dit=False))
            self.assertEqual(self.enable_calls, [])
            self.stage._maybe_enable_cache_dit(8, _batch(enable_cache_dit=None))
            self.assertEqual(len(self.enable_calls), 1)

    def test_same_overrides_refresh_without_remount(self):
        params = {"residual_diff_threshold": 0.12}
        self.stage._maybe_enable_cache_dit(
            8, _batch(enable_cache_dit=True, cache_dit_params=dict(params))
        )
        self.stage._maybe_enable_cache_dit(
            12, _batch(enable_cache_dit=True, cache_dit_params=dict(params))
        )
        self.assertEqual(len(self.enable_calls), 1)
        self.assertEqual(self.disable_calls, [])
        self.assertEqual(self.refresh_calls, [12])

    def test_changed_overrides_unmount_and_remount(self):
        self.stage._maybe_enable_cache_dit(
            8,
            _batch(
                enable_cache_dit=True,
                cache_dit_params={"residual_diff_threshold": 0.3},
            ),
        )
        self.stage._maybe_enable_cache_dit(
            8,
            _batch(
                enable_cache_dit=True,
                cache_dit_params={"residual_diff_threshold": 0.1},
            ),
        )
        self.assertEqual(len(self.disable_calls), 1)
        self.assertEqual(len(self.enable_calls), 2)
        self.assertEqual(self.refresh_calls, [])
        self.assertEqual(self.enable_calls[0].residual_diff_threshold, 0.3)
        self.assertEqual(self.enable_calls[1].residual_diff_threshold, 0.1)

    def test_request_knobs_reach_cache_dit_config(self):
        self.stage._maybe_enable_cache_dit(
            8,
            _batch(
                enable_cache_dit=True,
                cache_dit_params={
                    "Fn_compute_blocks": 4,
                    "max_warmup_steps": 2,
                    "scm_policy": "static",
                },
            ),
        )
        (config,) = self.enable_calls
        self.assertEqual(config.Fn_compute_blocks, 4)
        self.assertEqual(config.max_warmup_steps, 2)
        self.assertEqual(config.steps_computation_policy, "static")
        self.assertEqual(config.num_inference_steps, 8)

    def test_invalid_request_params_raise(self):
        with self.assertRaisesRegex(ValueError, "Unknown cache_dit_params keys"):
            self.stage._maybe_enable_cache_dit(
                8, _batch(enable_cache_dit=True, cache_dit_params={"bogus": 1})
            )
        self.assertEqual(self.enable_calls, [])

    def test_ordinary_warmup_does_not_mount(self):
        self.stage._maybe_enable_cache_dit(
            8, _batch(enable_cache_dit=True, is_warmup=True)
        )
        self.assertEqual(self.enable_calls, [])
        self.assertFalse(self.stage._cache_dit_enabled)

    def test_secondary_inherits_request_primary_overrides(self):
        self.stage._cache_dit_request_overrides = resolve_cache_dit_request_overrides(
            {
                "Fn_compute_blocks": 5,
                "secondary": {"Bn_compute_blocks": 7},
            }
        )
        primary = self.stage._build_cache_dit_config(
            10, steps_computation_mask=None, scm_policy="dynamic"
        )
        secondary = self.stage._build_cache_dit_config(
            10, steps_computation_mask=None, scm_policy="dynamic", secondary=True
        )
        self.assertEqual(primary.Fn_compute_blocks, 5)
        self.assertEqual(secondary.Fn_compute_blocks, 5)  # inherited from primary
        self.assertEqual(secondary.Bn_compute_blocks, 7)

    def test_request_scm_bins_require_both(self):
        self.stage._cache_dit_request_overrides = resolve_cache_dit_request_overrides(
            {"scm_compute_bins": [4, 2]}
        )
        with self.assertRaisesRegex(ValueError, "scm_compute_bins and scm_cache_bins"):
            self.stage._parse_cache_dit_scm_bins()


if __name__ == "__main__":
    unittest.main()
