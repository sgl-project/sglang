"""Online updates must reject model-owned derived buffers before writing."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


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

    def test_direct_only_cache_rejects_direct_updates(self):
        class Model(torch.nn.Sequential):
            def load_weights(self, weights):
                for name, tensor in weights:
                    self.get_parameter(name).data.copy_(tensor)

        model = Model(
            torch.nn.Linear(2, 2, bias=False, device="cpu"),
            torch.nn.Sequential(torch.nn.Module()),
        )
        model[1][0]._direct_weight_update_error = "direct skips the loader"
        original = model[0].weight.detach().clone()
        updater = _cpu_updater(model)
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
                (False, "direct skips the loader"),
            )
            self.assertTrue(torch.equal(model[0].weight, original))
            success, _ = updater.update_weights_from_tensor(
                [("0.weight", torch.zeros_like(original))], load_format=None
            )
        self.assertTrue(success)
        self.assertTrue(torch.equal(model[0].weight, torch.zeros_like(original)))


def _cpu_updater(model):
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


if __name__ == "__main__":
    unittest.main()
