"""Tensor weight updates: the deployment TP rank's entry is loaded, and a custom
loader can refuse an update by raising WeightUpdateRejected."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
from sglang.srt.model_loader.utils import WeightUpdateRejected
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

_MODULE = "sglang.srt.model_executor.model_runner_components.weight_updater"
_LOADER = "my_pkg.loaders.load"


def _updater(model, custom_weight_loaders):
    return WeightUpdater(
        tp_rank=0,
        device="cpu",
        gpu_id=0,
        model_config=None,
        custom_weight_loaders=custom_weight_loaders,
        get_model=lambda: model,
        update_model_fields=lambda *args, **kwargs: None,
        recapture_cuda_graph=lambda: None,
        get_model_runner=lambda: None,
    )


class TestTensorUpdateRank(unittest.TestCase):
    def test_a_narrowed_updater_reads_the_deployment_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        # An attention-owning draft records its attention-TP rank at init.
        updater = _updater(model, {})
        per_rank = [torch.full((2, 2), float(rank)) for rank in range(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        with (
            patch(
                f"{_MODULE}.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=1),
        ):
            success, message = updater.update_weights_from_tensor(
                [("weight", payload)], load_format="direct"
            )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[1]))


def _reject(model, named_tensors):
    raise WeightUpdateRejected("layer 3 is frozen")


def _crash(model, named_tensors):
    raise RuntimeError("loader bug")


class TestCustomLoaderReject(CustomTestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2, bias=False)
        self.original = self.model.weight.detach().clone()
        self.updater = _updater(self.model, [_LOADER])

    def _update(self, loader, pp_size=1):
        # monkey_patch_torch_reductions() would patch torch's tensor pickling
        # for the rest of the process; the plain tensors here do not need it.
        with (
            patch(f"{_MODULE}.monkey_patch_torch_reductions"),
            patch(f"{_MODULE}.dynamic_import", return_value=loader),
            patch(
                f"{_MODULE}.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=0, pp_rank=0, pp_size=pp_size),
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

    def test_rejection_propagates_with_pipeline_parallelism(self):
        # Only the first stage reports, so a later stage's rejection must not
        # turn into a success that the caller sees.
        with self.assertRaises(WeightUpdateRejected):
            self._update(_reject, pp_size=2)


if __name__ == "__main__":
    unittest.main()
