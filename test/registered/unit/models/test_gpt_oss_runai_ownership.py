"""Hermetic unit tests for gpt-oss RunAI-streamed weight ownership.

The RunAI streamer hands out zero-copy views into a staging buffer it reuses
between tensors, so a view read after later tensors arrive can come back as
garbage. `_load_weights_mxfp4` therefore has to consume the expert weights as
they are yielded, and take its own copy of anything it keeps for later.

Pure Python (no GPU, no model weights): the model object is built without
`__init__` and both loader halves are replaced with recorders.
"""

import unittest

import torch

from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
from sglang.srt.models.gpt_oss import GptOssForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _streamed(value: float) -> torch.Tensor:
    """A tensor marked the way the RunAI streamer marks its buffer views."""
    tensor = torch.full((4,), value)
    setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
    return tensor


class _Mxfp4QuantConfig:
    def get_name(self) -> str:
        return "mxfp4"


class TestGptOssRunaiOwnership(CustomTestCase):
    def _model(self, on_experts, on_normal):
        model = object.__new__(GptOssForCausalLM)
        model.quant_config = _Mxfp4QuantConfig()
        model._load_mxfp4_experts_weights = on_experts
        model._load_normal_weights = on_normal
        return model

    def test_expert_weights_are_consumed_as_they_arrive(self):
        produced = []

        def stream():
            for i in range(3):
                produced.append(i)
                yield f"model.layers.{i}.mlp.experts.gate_up_proj_blocks", _streamed(i)

        produced_when_seen = []

        def on_experts(weights):
            for _name, _weight in weights:
                produced_when_seen.append(len(produced))
            return set()

        model = self._model(on_experts, lambda *a, **k: None)
        model._load_weights_mxfp4(stream(), is_nextn=False, weight_name_mapping=None)

        # One produced per one consumed: the loader never runs ahead of itself
        # and leaves earlier views waiting on the buffer.
        self.assertEqual(produced_when_seen, [1, 2, 3])

    def test_retained_weights_are_copied_out_of_the_buffer(self):
        streamed = _streamed(1.0)
        plain = torch.full((4,), 2.0)
        kept = {}

        def on_normal(weights, **kwargs):
            kept.update({name: tensor for name, tensor in weights})

        model = self._model(lambda weights: {n for n, _ in weights}, on_normal)
        model._load_weights_mxfp4(
            iter([("model.embed_tokens.weight", streamed), ("lm_head.weight", plain)]),
            is_nextn=False,
            weight_name_mapping=None,
        )

        held = kept["model.embed_tokens.weight"]
        self.assertIsNot(held, streamed)
        self.assertNotEqual(held.data_ptr(), streamed.data_ptr())
        # Held until the stream ends, so it belongs on the host rather than
        # in device memory the streamer's limit does not account for.
        self.assertEqual(held.device.type, "cpu")
        torch.testing.assert_close(held, streamed)

        # Anything not streamed is left alone rather than copied for nothing.
        self.assertIs(kept["lm_head.weight"], plain)


if __name__ == "__main__":
    unittest.main()
