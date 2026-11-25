"""GGUF expert loading shards each projection along its TP-parallel axis."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gguf import GGMLQuantizationType, dequantize, quantize

from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.quantization.gguf import GGUFUninitializedParameter
from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_Q8_0 = GGMLQuantizationType.Q8_0
_Q8_0_BLOCK = 32


def _load_local_shard(*, weight, shard_id, tp_size, tp_rank, quant_type=_Q8_0):
    layer = SimpleNamespace(
        moe_tp_size=tp_size,
        moe_tp_rank=tp_rank,
        w2_qweight_type=SimpleNamespace(weight_type=int(quant_type)),
    )
    param = GGUFUninitializedParameter(requires_grad=False)
    set_weight_attrs(
        param,
        {"is_gguf_weight": True, "input_dim": 1, "output_dim": 0, "data_container": []},
    )
    handled = FusedMoE._load_gguf_weight(
        layer, param, torch.from_numpy(weight), shard_id, expert_id=0
    )
    assert handled
    return param.expert_data_map[(0, shard_id)].numpy()


class TestGgufExpertTpSharding(CustomTestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        # [hidden, intermediate]: two Q8_0 blocks along the packed input axis.
        self.down = rng.standard_normal((8, 2 * _Q8_0_BLOCK), dtype=np.float32)
        self.packed_down = quantize(self.down, _Q8_0)

    def test_down_projection_shards_its_input_axis(self):
        """Row-parallel w2 keeps every output row and takes this rank's input columns."""
        full = dequantize(self.packed_down, _Q8_0)
        for rank in range(2):
            local = _load_local_shard(
                weight=self.packed_down, shard_id="w2", tp_size=2, tp_rank=rank
            )
            cols = slice(rank * _Q8_0_BLOCK, (rank + 1) * _Q8_0_BLOCK)
            np.testing.assert_array_equal(dequantize(local, _Q8_0), full[:, cols])

    def test_gate_up_projection_shards_its_output_rows(self):
        gate = self.packed_down
        local = _load_local_shard(weight=gate, shard_id="w1", tp_size=2, tp_rank=1)
        np.testing.assert_array_equal(local, gate[gate.shape[0] // 2 :])

    def test_down_shard_that_splits_a_quant_block_is_rejected(self):
        """Three blocks cannot split across two ranks without cutting a block in half."""
        three_blocks = quantize(np.ones((4, 3 * _Q8_0_BLOCK), dtype=np.float32), _Q8_0)
        with self.assertRaises(ValueError):
            _load_local_shard(weight=three_blocks, shard_id="w2", tp_size=2, tp_rank=0)

    def test_f16_down_projection_shards_elements(self):
        down = self.down.astype(np.float16)
        local = _load_local_shard(
            weight=down,
            shard_id="w2",
            tp_size=2,
            tp_rank=1,
            quant_type=GGMLQuantizationType.F16,
        )
        np.testing.assert_array_equal(local, down[:, _Q8_0_BLOCK:])


if __name__ == "__main__":
    unittest.main()
