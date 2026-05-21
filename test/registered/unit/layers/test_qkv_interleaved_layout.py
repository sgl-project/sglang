"""Unit tests for the TP-rank-interleaved qkv_proj layout detection used by
QKVParallelLinear's FP8 weight and weight-scale loaders.

Background: when a model has asymmetric K/V (head_size != v_head_size) AND the
per-rank merged-QKV output is not divisible by the FP8 quantization block_n,
some checkpoints (e.g. MiMo-V2.5-Pro at TP=8 with head_size=192,
v_head_size=128, block_n=128) store qkv_proj.weight and
qkv_proj.weight_scale_inv in TP-rank-INTERLEAVED layout rather than flat
[Q | K | V]. The loader's heuristic must (a) detect this case for asymmetric
checkpoints, (b) NOT fire for symmetric/aligned checkpoints, and (c) split the
checkpoint correctly when it does fire.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")

import types
import unittest

import torch

from sglang.srt.layers.linear import QKVParallelLinear
from sglang.test.test_utils import CustomTestCase


def _make_layer(
    *,
    total_num_heads,
    total_num_kv_heads,
    head_size,
    v_head_size,
    tp_size,
    tp_rank,
    block_n,
):
    """Construct a minimal QKVParallelLinear-like object that bypasses the
    full __init__ but exposes the attributes the loader heuristic reads."""
    layer = QKVParallelLinear.__new__(QKVParallelLinear)
    layer.total_num_heads = total_num_heads
    layer.total_num_kv_heads = total_num_kv_heads
    layer.head_size = head_size
    layer.v_head_size = v_head_size
    layer.tp_size = tp_size
    layer.tp_rank = tp_rank

    quant_config = types.SimpleNamespace(weight_block_size=(block_n, block_n))
    layer.quant_method = types.SimpleNamespace(quant_config=quant_config)
    return layer


class _Param:
    """Minimal stand-in for a vLLMParameter — exposes data and output_dim."""

    def __init__(self, shape, output_dim=0, dtype=torch.float32):
        self.data = torch.zeros(*shape, dtype=dtype)
        self.output_dim = output_dim


class TestInterleavedQKVLayoutHeuristic(CustomTestCase):
    """The _is_tp_interleaved_qkv_layout helper must fire only for cases
    where per-rank merged-QKV output is not divisible by block_n."""

    def test_mimo_v25_pro_tp8_triggers(self):
        # MiMo-V2.5-Pro at TP=8: per-rank = 16*192 + 1*192 + 1*128 = 3392,
        # 3392 % 128 = 64. Asymmetric K/V — must trigger.
        layer = _make_layer(
            total_num_heads=128,
            total_num_kv_heads=8,
            head_size=192,
            v_head_size=128,
            tp_size=8,
            tp_rank=0,
            block_n=128,
        )
        self.assertTrue(layer._is_tp_interleaved_qkv_layout())

    def test_symmetric_kv_aligned_does_not_trigger(self):
        # DeepSeek-V3-style: head_size == v_head_size, all dims block-aligned.
        layer = _make_layer(
            total_num_heads=128,
            total_num_kv_heads=8,
            head_size=128,
            v_head_size=128,
            tp_size=8,
            tp_rank=0,
            block_n=128,
        )
        self.assertFalse(layer._is_tp_interleaved_qkv_layout())

    def test_llama_style_does_not_trigger(self):
        # Llama-3-style: 32 heads, 8 kv heads, head_size 128, TP=4.
        # per-rank = 8*128 + 2*128 + 2*128 = 1536, divisible by 128.
        layer = _make_layer(
            total_num_heads=32,
            total_num_kv_heads=8,
            head_size=128,
            v_head_size=128,
            tp_size=4,
            tp_rank=0,
            block_n=128,
        )
        self.assertFalse(layer._is_tp_interleaved_qkv_layout())

    def test_no_quant_config_returns_false(self):
        # Models without an FP8 block-scale quant config must not trigger.
        layer = QKVParallelLinear.__new__(QKVParallelLinear)
        layer.total_num_heads = 128
        layer.total_num_kv_heads = 8
        layer.head_size = 192
        layer.v_head_size = 128
        layer.tp_size = 8
        layer.tp_rank = 0
        layer.quant_method = types.SimpleNamespace(quant_config=None)
        self.assertFalse(layer._is_tp_interleaved_qkv_layout())


class TestInterleavedQKVScaleLoading(CustomTestCase):
    """When the scale checkpoint is in TP-rank-interleaved layout (more rows
    than the flat layout would predict), the loader must copy this rank's
    contiguous chunk into the parameter."""

    def _setup(self, tp_rank):
        # MiMo-V2.5-Pro shape: per-rank scale is (27, 48), total scale rows
        # are 8 * 27 = 216 (vs the 212 flat layout would predict).
        layer = _make_layer(
            total_num_heads=128,
            total_num_kv_heads=8,
            head_size=192,
            v_head_size=128,
            tp_size=8,
            tp_rank=tp_rank,
            block_n=128,
        )
        per_rank_blocks = 27
        in_blocks = 48
        # Synthetic checkpoint whose value at each row equals the row index, so
        # we can assert exactly which rows ended up in the rank's parameter.
        loaded = (
            torch.arange(8 * per_rank_blocks, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, in_blocks)
            .contiguous()
        )
        param = _Param((per_rank_blocks, in_blocks), output_dim=0)
        return layer, param, loaded, per_rank_blocks

    def test_rank0_loads_first_27_blocks(self):
        layer, param, loaded, per_rank_blocks = self._setup(tp_rank=0)
        layer._load_qkv_block_scale(param, loaded)
        # Rank 0 must end up with rows [0, 1, ..., 26].
        expected = torch.arange(per_rank_blocks, dtype=torch.float32)
        self.assertTrue(torch.equal(param.data[:, 0], expected))

    def test_rank4_loads_correct_chunk(self):
        layer, param, loaded, per_rank_blocks = self._setup(tp_rank=4)
        layer._load_qkv_block_scale(param, loaded)
        # Rank 4's chunk is rows [4*27, ..., 5*27 - 1] = [108, ..., 134].
        expected = torch.arange(
            4 * per_rank_blocks, 5 * per_rank_blocks, dtype=torch.float32
        )
        self.assertTrue(torch.equal(param.data[:, 0], expected))

    def test_rank7_loads_last_chunk(self):
        layer, param, loaded, per_rank_blocks = self._setup(tp_rank=7)
        layer._load_qkv_block_scale(param, loaded)
        expected = torch.arange(
            7 * per_rank_blocks, 8 * per_rank_blocks, dtype=torch.float32
        )
        self.assertTrue(torch.equal(param.data[:, 0], expected))


class TestInterleavedQKVWeightLoading(CustomTestCase):
    """When per-rank output isn't block-aligned, the weight checkpoint is
    also TP-rank-interleaved. _load_fused_module_from_checkpoint must copy
    this rank's slab directly rather than splitting flat [Q | K | V]."""

    def _setup(self, tp_rank):
        layer = _make_layer(
            total_num_heads=128,
            total_num_kv_heads=8,
            head_size=192,
            v_head_size=128,
            tp_size=8,
            tp_rank=tp_rank,
            block_n=128,
        )
        per_rank_size = 16 * 192 + 1 * 192 + 1 * 128  # 3392
        hidden_size = 6144
        loaded = (
            torch.arange(8 * per_rank_size, dtype=torch.float32)
            .view(-1, 1)
            .expand(-1, hidden_size)
            .contiguous()
        )
        param = _Param((per_rank_size, hidden_size), output_dim=0)
        return layer, param, loaded, per_rank_size

    def test_rank0_loads_first_slab(self):
        layer, param, loaded, per_rank_size = self._setup(tp_rank=0)
        layer._load_fused_module_from_checkpoint(param, loaded)
        expected = torch.arange(per_rank_size, dtype=torch.float32)
        self.assertTrue(torch.equal(param.data[:, 0], expected))

    def test_rank3_loads_correct_slab(self):
        layer, param, loaded, per_rank_size = self._setup(tp_rank=3)
        layer._load_fused_module_from_checkpoint(param, loaded)
        expected = torch.arange(
            3 * per_rank_size, 4 * per_rank_size, dtype=torch.float32
        )
        self.assertTrue(torch.equal(param.data[:, 0], expected))


if __name__ == "__main__":
    unittest.main()
