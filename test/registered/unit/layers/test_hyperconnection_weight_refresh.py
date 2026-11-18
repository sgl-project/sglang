"""GatedResidual mix-weight caches must track online weight updates.

The JIT mix path keeps a permuted/padded copy of ``input_mix_weight_up``
and the FlyDSL path keeps packed weights keyed on (data_ptr, version),
which in-place loader writes do not change. The weight loader attached
in ``GatedResidual.__init__`` refreshes both; ``load_format='direct'``
bypasses the loader and is refused while a cache can exist.

CPU-only.

    python -m pytest test/registered/unit/layers/test_hyperconnection_weight_refresh.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.gemm.hc_mix_splitk import permute_pad_up_weight
from sglang.srt.layers.hyperconnection import GatedResidual, HyperConnectionConfig
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
    _direct_weight_update_error,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import get_parallel
from sglang.test.test_utils import CustomTestCase


class _FakeModel(torch.nn.Module):
    """Minimal model whose load_weights dispatches per-param loaders."""

    def __init__(self, mixer):
        super().__init__()
        self.mixer = mixer

    def load_weights(self, weights):
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, loaded_weight)


class TestMixWeightRefresh(CustomTestCase):
    def _make_module(self, lowrank=16, hidden_size=512):
        config = HyperConnectionConfig(
            hc_count=4,
            hidden_size=hidden_size,
            params_dtype=torch.bfloat16,
            hc_lowrank=lowrank,
        )
        return GatedResidual(config)

    def _make_updater(self, model):
        return WeightUpdater(
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

    def test_loader_refreshes_padded_copy_in_place(self):
        torch.manual_seed(0)
        module = self._make_module()
        up = module.input_mix_weight_up.weight
        module._mix_up_weight_padded = permute_pad_up_weight(up, module.hc_count)
        padded = module._mix_up_weight_padded
        new_weight = torch.randn_like(up)
        up.weight_loader(up, new_weight)
        # The live weight takes the new values ...
        self.assertTrue(torch.equal(up, new_weight))
        # ... and the cached copy is rewritten in place, so captured CUDA
        # graphs keep reading the right weights through the same pointer.
        self.assertIs(module._mix_up_weight_padded, padded)
        self.assertTrue(
            torch.equal(padded, permute_pad_up_weight(new_weight, module.hc_count))
        )

    def test_loader_is_plain_copy_without_caches(self):
        module = self._make_module()
        down = module.input_mix_weight_down.weight
        new_weight = torch.randn_like(down)
        down.weight_loader(down, new_weight)
        self.assertTrue(torch.equal(down, new_weight))
        self.assertIsNone(module._mix_up_weight_padded)
        self.assertIsNone(module._mix_flydsl_weight_key)

    def test_loader_invalidates_flydsl_key(self):
        module = self._make_module()
        module._mix_flydsl_weight_key = ("sentinel",)
        down = module.input_mix_weight_down.weight
        down.weight_loader(down, torch.randn_like(down))
        self.assertIsNone(module._mix_flydsl_weight_key)

    def test_direct_error_unset_without_derived_cache(self):
        module = self._make_module()
        self.assertIsNone(module._direct_weight_update_error)
        self.assertIsNone(
            _direct_weight_update_error(SimpleNamespace(modules=lambda: [module]))
        )

    def test_direct_error_set_when_jit_cache_can_exist(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            module = self._make_module()
        self.assertIsNotNone(module._direct_weight_update_error)
        self.assertIn("direct", module._direct_weight_update_error)

    def test_direct_error_set_for_flydsl_shapes_on_rocm(self):
        # CPU torch has no HIP build, so the FlyDSL cache cannot exist.
        self.assertIsNone(self._make_module(lowrank=32)._direct_weight_update_error)
        with patch.object(torch.version, "hip", "7.0.0"):
            module = self._make_module(lowrank=32)
        self.assertIsNotNone(module._direct_weight_update_error)
        self.assertIn("direct", module._direct_weight_update_error)

    def test_updater_refuses_direct_format(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(10, 0)),
        ):
            module = self._make_module()
        model = torch.nn.Sequential(module)
        original = module.input_mix_weight_up.weight.detach().clone()
        updater = self._make_updater(model)
        with patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [
                        (
                            "0.input_mix_weight_up.weight",
                            torch.zeros_like(original),
                        )
                    ],
                    load_format="direct",
                ),
                (False, module._direct_weight_update_error),
            )
        self.assertTrue(torch.equal(module.input_mix_weight_up.weight, original))

    def test_updater_default_format_refreshes_cache(self):
        torch.manual_seed(0)
        module = self._make_module()
        up = module.input_mix_weight_up.weight
        module._mix_up_weight_padded = permute_pad_up_weight(up, module.hc_count)
        padded = module._mix_up_weight_padded
        new_weight = torch.randn_like(up)
        model = _FakeModel(module)
        updater = self._make_updater(model)
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=0),
        ):
            self.assertEqual(
                updater.update_weights_from_tensor(
                    [("mixer.input_mix_weight_up.weight", new_weight)],
                    load_format=None,
                ),
                (True, "Success"),
            )
        self.assertTrue(torch.equal(up, new_weight))
        self.assertIs(module._mix_up_weight_padded, padded)
        self.assertTrue(
            torch.equal(padded, permute_pad_up_weight(new_weight, module.hc_count))
        )


if __name__ == "__main__":
    unittest.main()
