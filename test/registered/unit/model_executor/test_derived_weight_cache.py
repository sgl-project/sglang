"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.models import sdar
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


class TestDerivedWeightCache(CustomTestCase):
    def _sdar_model(self, refresh_pair):
        model = object.__new__(sdar.SDARForCausalLM)
        # Exercise the startup guard without allocating model layers or GPU memory.
        with (
            patch.object(
                sdar.envs.SGLANG_SDAR_REFRESH_PAIR, "get", return_value=refresh_pair
            ),
            patch.object(
                sdar, "get_parallel", side_effect=RuntimeError("stop before layers")
            ),
            self.assertRaisesRegex(RuntimeError, "stop before layers"),
        ):
            sdar.SDARForCausalLM.__init__(model, None)
        return model

    def test_sdar_refresh_pair_rejects_all_weight_update_entries(self):
        model = self._sdar_model(refresh_pair=True)
        model.load_weights = Mock()
        updater = SimpleNamespace(
            get_model=lambda: model, _assert_weight_cache_inactive=Mock()
        )
        for method, args in (
            ("update_weights_from_tensor", ([], "direct")),
            ("load_weights_from_distributed", ([],)),
            ("update_weights_from_disk", ("unused", "auto")),
            ("update_weights_from_ipc", (SimpleNamespace(),)),
        ):
            with self.subTest(method=method):
                success, message = getattr(WeightUpdater, method)(updater, *args)
                self.assertFalse(success)
                self.assertIn("SDAR refresh sharing", message)
        model.load_weights.assert_not_called()

    def test_sdar_without_refresh_pair_keeps_weight_updates_enabled(self):
        model = self._sdar_model(refresh_pair=False)
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))

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

    def test_model_without_derived_cache_keeps_updates_enabled(self):
        model = torch.nn.Module()
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))


if __name__ == "__main__":
    unittest.main()
