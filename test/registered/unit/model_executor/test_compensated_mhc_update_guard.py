import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompensatedMhcUpdateGuard(CustomTestCase):
    def test_all_update_entries_reject_before_writes(self):
        for field in ("_hc_attn_tf32_parts", "_hc_ffn_tf32_parts"):
            for method, args in (
                ("update_weights_from_tensor", ([], "direct")),
                ("update_weights_from_tensor", ([], None)),
                ("update_weights_from_tensor", ({}, "flattened_bucket")),
                ("update_weights_from_distributed", ([], [], [], "unused")),
                ("update_weights_from_disk", ("unused", "auto")),
                ("update_weights_from_ipc", (SimpleNamespace(),)),
            ):
                with self.subTest(field=field, method=method, args=args):
                    model = torch.nn.Sequential(torch.nn.Linear(1, 1))
                    original = model[0].weight.detach().clone()
                    setattr(model[0], field, (torch.ones(1), torch.zeros(1)))
                    model.load_weights = Mock()
                    updater = SimpleNamespace(
                        get_model=lambda: model, _assert_weight_cache_inactive=Mock()
                    )
                    with patch(
                        "sglang.srt.model_executor.model_runner_components.weight_updater.default_weight_loader"
                    ) as loader:
                        ok, message = getattr(WeightUpdater, method)(updater, *args)
                    self.assertFalse(ok)
                    self.assertIn("compensated mHC", message)
                    loader.assert_not_called()
                    model.load_weights.assert_not_called()
                    torch.testing.assert_close(
                        model[0].weight, original, rtol=0, atol=0
                    )

    def test_models_without_derived_splits_keep_update_support(self):
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        model[0]._hc_attn_tf32_parts = model[0]._hc_ffn_tf32_parts = None
        with patch(
            "sglang.kernels.ops.attention.dsv4.gemm.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))
            self.assertIsNone(_unsupported_derived_weight_cache_error())


if __name__ == "__main__":
    unittest.main()
