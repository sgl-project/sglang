"""FP8 cached weights must produce the same result under decode TP slicing."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
CTX = "sglang.srt.layers.cp.cp_decode_attn_tp"


class TestCPFP8Views(CustomTestCase):
    def test_decode_shards_and_restored_prefill_match_full_linear(self):
        weight = torch.arange(64.0).reshape(8, 8)
        x = torch.arange(8.0).reshape(1, 8)
        expected = x @ weight.T
        for cls, dim in ((ColumnParallelLinear, 0), (RowParallelLinear, 1)):
            outputs = []
            for rank in range(2):
                layer = cls.__new__(cls)
                nn.Module.__init__(layer)
                layer.weight = nn.Parameter(weight.clone(), requires_grad=False)
                layer._block_fp8_bf16_weight = weight.clone()
                size_attr = (
                    "input_size_per_partition" if dim else "output_size_per_partition"
                )
                setattr(layer, size_attr, 8)
                layer.use_decode_attn_tp = False
                ctx = object.__new__(CpDecodeAttnTpContext)
                ctx.decode_tp_rank, ctx.decode_tp_size = rank, 2
                ctx._slice_cache = {}
                with (
                    patch(f"{CTX}.is_cp_active", return_value=False),
                    patch(f"{CTX}.dsa_use_prefill_cp", return_value=False),
                ):
                    with ctx.maybe_use_decode_attn_tp(None, [layer]):
                        local_x = x.chunk(2, dim=1)[rank] if dim else x
                        outputs.append(local_x @ layer._block_fp8_bf16_weight.T)
                torch.testing.assert_close(x @ layer._block_fp8_bf16_weight.T, expected)
            actual = sum(outputs) if dim else torch.cat(outputs, dim=1)
            torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
