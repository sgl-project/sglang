"""Qwen2 legacy weight loading keeps the checkpoint name for unstacked params."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models.qwen2 import Qwen2ForCausalLM  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _load(params: dict[str, torch.nn.Parameter], name: str) -> None:
    model = Qwen2ForCausalLM.__new__(Qwen2ForCausalLM)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace()
    model.config = SimpleNamespace(tie_word_embeddings=False)
    with patch.object(
        Qwen2ForCausalLM, "named_parameters", return_value=params.items()
    ):
        model._legacy_load_weights([(name, torch.ones(4, 2))])


class TestQwen2LegacyWeightLoader(CustomTestCase):
    def test_prefused_gate_up_checkpoint_is_not_renamed(self):
        # "up_proj" is a substring of "gate_up_proj"; the stacked mapping must
        # not rewrite the name used by the unstacked fallback.
        name = "model.layers.0.mlp.gate_up_proj.weight"
        param = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
        param.weight_loader = Mock()
        _load({name: param}, name)
        param.weight_loader.assert_called_once()
        self.assertEqual(len(param.weight_loader.call_args.args), 2)

    def test_split_checkpoint_still_loads_into_stacked_param(self):
        param = torch.nn.Parameter(torch.zeros(4, 2), requires_grad=False)
        param.weight_loader = Mock()
        _load(
            {"model.layers.0.mlp.gate_up_proj.weight": param},
            "model.layers.0.mlp.up_proj.weight",
        )
        param.weight_loader.assert_called_once()
        self.assertEqual(param.weight_loader.call_args.args[2], 1)


if __name__ == "__main__":
    unittest.main()
