"""Production loader regressions using real parameter storage."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _empty(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


class TestFFNModelLoaders(unittest.TestCase):
    def test_longcat_nextn_requantizes_both_mlp_projections(self):
        from sglang.srt.models import longcat_flash_nextn as longcat

        model = _empty(longcat.LongcatFlashForCausalLMNextN)
        model.config = SimpleNamespace(q_lora_rank=None)
        model.quant_config = SimpleNamespace(weight_block_size=[128, 128])
        model.model = nn.Module()
        layer = _empty(longcat.LongcatFlashDenseDecoderLayer)
        layer.self_attn = SimpleNamespace(
            kv_b_proj=object(),
            o_proj=object(),
            kv_a_proj_with_mqa=object(),
            q_proj=object(),
        )
        layer.mlp = _empty(longcat.LongcatFlashMLP)
        layer.mlp.gate_up_proj = nn.Linear(4, 12, bias=False)
        layer.mlp.down_proj = nn.Linear(6, 4, bias=False)
        for proj in (layer.mlp.gate_up_proj, layer.mlp.down_proj):
            proj.weight_scale_inv = torch.ones(1)
        model.model.decoder = layer
        with patch.object(longcat, "requant_weight_ue8m0_inplace") as requant:
            model._weight_requant_ue8m0()
        self.assertEqual(requant.call_count, 2)
        for call, proj in zip(
            requant.call_args_list, (layer.mlp.gate_up_proj, layer.mlp.down_proj)
        ):
            self.assertIs(call.args[0], proj.weight)
            self.assertIs(call.args[1], proj.weight_scale_inv)
            self.assertEqual(call.args[2], [128, 128])


if __name__ == "__main__":
    unittest.main()
