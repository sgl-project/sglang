import unittest

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Backbone(torch.nn.Module):
    def load_weights(self, weights):
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            params[name].data.copy_(loaded_weight)


class _DraftLoadHarness(DSparkDraftMixin, _Backbone):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.backbone = torch.nn.Linear(1, 1, bias=False)
        self.markov_head = torch.nn.Linear(1, 1, bias=False)
        self.confidence_head = torch.nn.Linear(1, 1, bias=False)


class TestDSparkWeightLoading(CustomTestCase):
    def test_streams_weights_before_buffer_reuse(self):
        shared_buffer = torch.tensor([[1.0]])

        def weights():
            for name, value in (
                ("markov_head.weight", 1.0),
                ("backbone.weight", 2.0),
                ("confidence_head.weight", 3.0),
            ):
                shared_buffer.fill_(value)
                view = shared_buffer[:]
                setattr(view, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, view

        model = _DraftLoadHarness()
        model.load_weights(weights())

        self.assertEqual(model.markov_head.weight.item(), 1.0)
        self.assertEqual(model.backbone.weight.item(), 2.0)
        self.assertEqual(model.confidence_head.weight.item(), 3.0)

    def test_missing_confidence_weight_still_errors(self):
        model = _DraftLoadHarness()
        weights = [
            ("markov_head.weight", torch.tensor([[1.0]])),
            ("backbone.weight", torch.tensor([[2.0]])),
        ]

        with self.assertRaisesRegex(ValueError, "checkpoint is missing"):
            model.load_weights(weights)


if __name__ == "__main__":
    unittest.main()
