"""Unit tests for GLM-5.3 NextN block-FP8 scale loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.models.glm5_next_nextn import (
    Glm5NextForConditionalGenerationNextN,
)
from sglang.test.test_utils import CustomTestCase


class TestGlm5NextNextNScaleLoading(CustomTestCase):
    def _load(self, weights, params):
        model = SimpleNamespace(
            config=SimpleNamespace(
                num_nextn_predict_layers=1, num_hidden_layers=45, n_routed_experts=2
            ),
            num_fused_shared_experts=0,
            fuse_qkv_a_proj=False,
            quant_config=None,
            named_parameters=lambda: params.items(),
            model=SimpleNamespace(decoder=SimpleNamespace(self_attn=None)),
        )
        weights = [(f"model.language_model.layers.45.{k}", v) for k, v in weights]
        Glm5NextForConditionalGenerationNextN.load_weights(model, weights)

    def _param(self):
        param = torch.nn.Parameter(torch.full((1, 2), -1.0), requires_grad=False)
        param.weight_loader = Mock(
            side_effect=lambda p, w, *args, **kwargs: p.data.copy_(w)
        )
        return param

    def test_existing_scale_parameter_takes_priority(self):
        for source, target in (
            ("self_attn.q_b_proj", "self_attn.q_b_proj."),
            ("mlp.experts.0.gate_proj", "mlp.experts.w13_"),
        ):
            with self.subTest(source=source):
                name = f"model.decoder.{target}weight_scale"
                param, unused = self._param(), self._param()
                scale = torch.tensor([[0.25, 0.5]])
                self._load(
                    [(f"{source}.weight_scale", scale)],
                    {name: param, f"{name}_inv": unused},
                )
                torch.testing.assert_close(param, scale)
                param.weight_loader.assert_called_once()
                unused.weight_loader.assert_not_called()


if __name__ == "__main__":
    sys.exit(unittest.main())
