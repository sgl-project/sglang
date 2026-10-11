"""A tensor weight update loads the entry of the process's deployment TP rank."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.hyperconnection import GatedResidual
from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
from sglang.srt.model_loader.loader import DefaultModelLoader, post_load_weights
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTensorUpdateRank(unittest.TestCase):
    def test_a_narrowed_updater_reads_the_deployment_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        # An attention-owning draft records its attention-TP rank at init.
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
        per_rank = [torch.full((2, 2), float(rank)) for rank in range(2)]
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        with (
            patch(
                "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
                return_value=SimpleNamespace(weight_cache_mode="off"),
            ),
            get_parallel().override(tp_rank=1),
        ):
            success, message = updater.update_weights_from_tensor(
                [("weight", payload)], load_format="direct"
            )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[1]))


class TestHCWeightUpdateCompletion(CustomTestCase):
    def test_pending_hc_rejects_completion_even_when_post_load_is_skipped(self):
        """A successful bucket is not a complete HC update or a valid initial load."""
        hc = GatedResidual.__new__(GatedResidual)
        torch.nn.Module.__init__(hc)
        hc._packed_sum_weights = True
        hc._hc_loaded_weights = {"down"}
        hc._sum_weights_dirty = True
        hc._sum_operators = (object(), object())
        model = torch.nn.Module()
        model.add_module("hc", hc)
        model.load_weights = lambda weights: None
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
                "sglang.srt.model_executor.model_runner_components.weight_updater."
                "DefaultModelLoader.postprocess_weights",
            ) as postprocess,
            patch("sglang.srt.model_loader.loader.is_cuda_alike", return_value=False),
        ):
            with self.assertRaisesRegex(ValueError, "hc:.*missing Inject"):
                updater.end_weight_update(run_post_load=False)
            with self.assertRaisesRegex(ValueError, "hc:.*missing Inject"):
                post_load_weights(model)
            # Initial/default disk loading has no begin/end update session.
            with self.assertRaisesRegex(ValueError, "hc:.*missing Inject"):
                DefaultModelLoader.load_weights_and_postprocess(
                    model, [], torch.device("cpu")
                )
            postprocess.assert_not_called()

            hc._hc_loaded_weights.clear()
            hc._sum_weights_dirty = False
            updater.end_weight_update(run_post_load=False)
            postprocess.assert_called_once()


if __name__ == "__main__":
    unittest.main()
