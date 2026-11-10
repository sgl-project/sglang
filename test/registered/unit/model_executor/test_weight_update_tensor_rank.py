"""A tensor weight update loads the entry of the process's engine rank."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.model_runner_components.weight_updater import (
    LocalSerializedTensor,
    WeightUpdater,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import MultiprocessingSerializer
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


def _updater(model):
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


def _copies_tagged_by_rank(count):
    return [torch.full((2, 2), float(rank)) for rank in range(count)]


def _update(updater, payload, **ranks):
    with (
        patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.get_model",
            return_value=SimpleNamespace(weight_cache_mode="off"),
        ),
        # The CUDA-IPC reduction patch is process-wide and breaks serializing
        # CPU tensors afterwards.
        patch(
            "sglang.srt.model_executor.model_runner_components.weight_updater.monkey_patch_torch_reductions"
        ),
        get_parallel().override(**ranks),
    ):
        return updater.update_weights_from_tensor(
            [("weight", payload)], load_format="direct"
        )


class TestTensorUpdateRank(unittest.TestCase):
    def test_a_narrowed_updater_reads_the_deployment_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        per_rank = _copies_tagged_by_rank(2)
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        success, message = _update(
            _updater(model), payload, tp_size=2, pp_size=1, pp_rank=0, tp_rank=1
        )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[1]))

    def test_a_later_pipeline_stage_reads_its_engine_rank_entry(self):
        model = torch.nn.Linear(2, 2, bias=False)
        per_rank = _copies_tagged_by_rank(4)
        payload = LocalSerializedTensor(
            values=[MultiprocessingSerializer.serialize(t) for t in per_rank]
        )
        success, message = _update(
            _updater(model), payload, tp_size=2, pp_size=2, pp_rank=1, tp_rank=0
        )
        self.assertTrue(success, message)
        self.assertTrue(torch.equal(model.weight.detach(), per_rank[2]))


if __name__ == "__main__":
    unittest.main()
