"""A custom weight loader refuses a tensor update by raising WeightUpdateRejected."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    WeightUpdater,
)
from sglang.srt.model_loader.utils import WeightUpdateRejected
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.model_runner_components.weight_updater"
_LOADER = "my_pkg.loaders.load"


def _reject(model, named_tensors):
    raise WeightUpdateRejected("layer 3 is frozen")


def _crash(model, named_tensors):
    raise RuntimeError("loader bug")


class TestCustomLoaderReject(CustomTestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.original = self.model.weight.detach().clone()
        self.updater = WeightUpdater(
            tp_rank=0,
            device="cpu",
            gpu_id=0,
            model_config=None,
            custom_weight_loaders=[_LOADER],
            get_model=lambda: self.model,
            update_model_fields=lambda *args, **kwargs: None,
            recapture_cuda_graph=lambda: None,
            get_model_runner=lambda: None,
        )

    def _update(self, loader):
        # monkey_patch_torch_reductions() would patch torch's tensor pickling
        # for the rest of the process; the plain tensors here do not need it.
        with (
            patch(f"{_MODULE}.monkey_patch_torch_reductions"),
            patch(f"{_MODULE}.dynamic_import", return_value=loader),
            patch(
                f"{_MODULE}.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=0),
        ):
            return self.updater.update_weights_from_tensor(
                [("weight", torch.zeros(2, 2))], load_format=_LOADER
            )

    def test_rejected_update_returns_failure(self):
        success, message = self._update(_reject)
        self.assertFalse(success)
        self.assertIn(_LOADER, message)
        self.assertIn("layer 3 is frozen", message)
        self.assertTrue(torch.equal(self.model.weight.detach(), self.original))

    def test_other_loader_errors_propagate(self):
        with self.assertRaises(RuntimeError):
            self._update(_crash)


if __name__ == "__main__":
    unittest.main()
