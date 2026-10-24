"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)


class TestDerivedWeightCache(CustomTestCase):
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

    def test_public_declaration_and_legacy_alias(self):
        model = torch.nn.Sequential(torch.nn.Module())
        for public, legacy, expected in (
            (None, None, None),
            ("plugin requires restart", None, "plugin requires restart"),
            (None, "legacy reason", "legacy reason"),
            ("public reason", "legacy reason", "public reason"),
            ("", "legacy reason", ""),
        ):
            with (
                self.subTest(public=public, legacy=legacy),
                patch(
                    "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
                    return_value=False,
                ),
            ):
                model[0].weight_update_unsupported_reason = public
                model[0]._derived_weight_cache_error = legacy
                self.assertEqual(
                    _unsupported_derived_weight_cache_error(model), expected
                )

    def test_public_declaration_rejects_each_load_path_before_writing(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
        reason = "plugin owns relocated weights"
        model[0].weight_update_unsupported_reason = reason
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
        updates = (
            lambda: updater.update_weights_from_disk("unused", "auto"),
            lambda: updater.load_weights_from_distributed([]),
            lambda: updater.update_weights_from_tensor(
                [("0.weight", torch.zeros_like(original))], load_format="direct"
            ),
            lambda: updater.update_weights_from_ipc(SimpleNamespace()),
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_available_gpu_memory",
                side_effect=AssertionError("disk setup ran despite the refusal"),
            ),
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.monkey_patch_torch_reductions",
                side_effect=AssertionError("tensor setup ran despite the refusal"),
            ),
        ):
            for path, update in enumerate(updates):
                with self.subTest(path=path):
                    self.assertEqual(update(), (False, reason))
                    self.assertTrue(torch.equal(model[0].weight, original))


if __name__ == "__main__":
    unittest.main()
