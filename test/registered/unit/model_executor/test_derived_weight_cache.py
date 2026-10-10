"""Online updates must reject model-owned derived buffers before writing."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _invalidate_derived_weights,
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestDerivedWeightCache(unittest.TestCase):
    def test_nested_model_cache_rejects_updates(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._derived_weight_cache_error = "derived scales require restart"
        original = model[0].weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("0.weight", torch.zeros_like(original))], load_format="direct"
                ),
                (False, "derived scales require restart"),
            )
        self.assertTrue(torch.equal(model[0].weight, original))

    def test_direct_update_invalidates_fallback_caches_before_writing(self):
        seen = []

        class Cache(torch.nn.Module):
            def invalidate_derived_weights(self):
                seen.append(model[0].weight.detach().clone())

        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False, device="cpu"), Cache()
        )
        original = model[0].weight.detach().clone()
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=0),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("0.weight", torch.zeros_like(original))], load_format="direct"
                ),
                (True, "Success"),
            )
        self.assertEqual(len(seen), 1)
        self.assertTrue(torch.equal(seen[0], original))
        self.assertTrue(torch.equal(model[0].weight, torch.zeros_like(original)))

    def test_update_session_invalidates_then_refreshes(self):
        calls = []

        class Cache(torch.nn.Module):
            def invalidate_derived_weights(self):
                calls.append("invalidate")

            def refresh_derived_weights(self):
                calls.append("refresh")

        model = torch.nn.Sequential(Cache())
        updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )
        # post_load_weights rebuilds the caches; the hook must not run twice.
        updater.begin_weight_update()
        updater.end_weight_update(run_post_load=True)
        self.assertEqual(calls, ["invalidate"])
        # Writes after a load in the same session are picked up at the end.
        updater.begin_weight_update()
        updater.end_weight_update(run_post_load=False)
        self.assertEqual(calls, ["invalidate", "invalidate", "refresh"])

    def test_hook_lookup_does_not_keep_the_model_alive(self):
        class Model(torch.nn.Module):
            def invalidate_derived_weights(self):
                pass

        model = Model()
        _invalidate_derived_weights(model)
        alive = weakref.ref(model)
        del model
        gc.collect()
        self.assertIsNone(alive())

    def test_model_without_derived_cache_keeps_updates_enabled(self):
        model = torch.nn.Module()
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))


if __name__ == "__main__":
    unittest.main()
