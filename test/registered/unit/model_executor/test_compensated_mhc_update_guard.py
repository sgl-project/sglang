"""Weight-update entry points refuse a model carrying compensated-mHC derived
weight caches before any weight is written."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _unsupported_derived_weight_cache_error,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestCompensatedMhcUpdateGuard(CustomTestCase):
    def test_all_update_entries_reject_before_writes(self):
        for field in (
            "_hc_attn_tf32_parts",
            "_hc_ffn_tf32_parts",
            "_hc_attn_bf16_parts",
            "_hc_ffn_bf16_parts",
        ):
            for method, args in (
                ("update_weights_from_tensor", ([], "direct")),
                ("load_weights_from_distributed", ([],)),
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
        model[0]._hc_attn_bf16_parts = model[0]._hc_ffn_bf16_parts = None
        with patch(
            "sglang.kernels.ops.gemm.bf16_fp32.hpc_bf16xfp32_gemm_enabled",
            return_value=False,
        ):
            self.assertIsNone(_unsupported_derived_weight_cache_error(model))
            self.assertIsNone(_unsupported_derived_weight_cache_error())


class TestOnlineQuantizedUpdateGuard(CustomTestCase):
    def setUp(self):
        override = get_context().override_server_args(weight_cache_mode="off")
        override.install()
        self.addCleanup(override.restore)

    def _updater(self, model):
        return WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders={},
            get_model=lambda: model,
            update_model_fields=Mock(),
            recapture_cuda_graph=Mock(),
            get_model_runner=Mock(),
        )

    def test_session_entries_reject_before_mutating_weights(self):
        """Begin/end must not restore or postprocess unsupported weights, even without a load call."""
        model = torch.nn.Sequential(torch.nn.Linear(1, 1))
        model[0]._derived_weight_cache_error = "load-time FP8 weights cannot be updated"
        original = model[0].weight.detach().clone()
        updater = self._updater(model)

        def mutate(*args):
            with torch.no_grad():
                model[0].weight.fill_(123)

        module = "sglang.srt.model_executor.model_runner_components.weight_updater"
        with (
            patch(
                f"{module}.DefaultModelLoader.restore_weights_before_loading",
                side_effect=mutate,
            ),
            patch(
                f"{module}.DefaultModelLoader.postprocess_weights", side_effect=mutate
            ),
            patch(f"{module}.post_load_weights", side_effect=mutate),
        ):
            for method, kwargs in (
                (updater.begin_weight_update, {}),
                (updater.end_weight_update, {"run_post_load": True}),
            ):
                with self.subTest(method=method.__name__):
                    with self.assertRaisesRegex(ValueError, "load-time FP8"):
                        method(**kwargs)
                    torch.testing.assert_close(
                        model[0].weight, original, rtol=0, atol=0
                    )

    def test_selection_disables_updates_on_every_pipeline_stage(self):
        """A stage with no selected layers must reject just like a stage with converted layers."""
        with get_context().override_server_args(
            nvfp4_ckpt_fp8_dense_modules=["model.layers.0.self_attn.o_proj"]
        ):
            for converted in (False, True):
                model = torch.nn.Sequential(torch.nn.Linear(1, 1))
                if converted:
                    model[0]._derived_weight_cache_error = "stage has FP8 weights"
                updater = self._updater(model)
                original = model[0].weight.detach().clone()
                with self.subTest(converted=converted):
                    with self.assertRaisesRegex(ValueError, "stages without matched"):
                        updater.begin_weight_update()
                    for method, args in (
                        (updater.update_weights_from_disk, ("unused", "auto")),
                        (updater.update_weights_from_tensor, ([],)),
                        (updater.load_weights_from_distributed, ([],)),
                        (updater.update_weights_from_ipc, (SimpleNamespace(),)),
                    ):
                        ok, message = method(*args)
                        self.assertFalse(ok)
                        self.assertIn("stages without matched", message)
                    torch.testing.assert_close(
                        model[0].weight, original, rtol=0, atol=0
                    )


if __name__ == "__main__":
    unittest.main()
