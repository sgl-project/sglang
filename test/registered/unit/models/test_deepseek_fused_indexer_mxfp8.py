"""An mxfp8 indexer wk / weights_proj must reach the fused bf16 wk_weights_proj dequantized with its UE8M0 scales."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.mxfp8_block_convert import (
    dequant_mxfp8_2d_to_bf16,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_indexer_wk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

HEAD_DIM, N_HEADS, HIDDEN = 128, 32, 256
PREFIX = "model.layers.0.self_attn.indexer"


class TestFusedIndexerMxfp8(CustomTestCase):
    def test_mxfp8_shards_are_dequantized_with_their_ue8m0_scales(self):
        """The scales are biased exponents: used as multipliers, the indexer keys
        are off by orders of magnitude until a bf16 update overwrites them."""
        torch.manual_seed(0)
        fused = torch.zeros(HEAD_DIM + N_HEADS, HIDDEN, dtype=torch.bfloat16)
        params_dict = {f"{PREFIX}.wk_weights_proj.weight": fused}
        quant_config = SimpleNamespace(weight_block_size=[1, 32])
        pending = {}
        expected = {}
        for shard, rows in (("wk", HEAD_DIM), ("weights_proj", N_HEADS)):
            weight = torch.randn(rows, HIDDEN).to(torch.float8_e4m3fn)
            scale = torch.randint(120, 130, (rows, HIDDEN // 32), dtype=torch.uint8)
            expected[shard] = dequant_mxfp8_2d_to_bf16(weight, scale)
            for name, tensor in (("weight", weight), ("weight_scale_inv", scale)):
                self.assertTrue(
                    _load_fused_indexer_wk(
                        f"{PREFIX}.{shard}.{name}",
                        tensor,
                        params_dict,
                        pending,
                        quant_config,
                    )
                )

        self.assertTrue(torch.equal(fused[:HEAD_DIM], expected["wk"]))
        self.assertTrue(torch.equal(fused[-N_HEADS:], expected["weights_proj"]))


if __name__ == "__main__":
    unittest.main()
