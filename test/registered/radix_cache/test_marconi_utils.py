"""
Unit tests for Marconi FLOP / memory utility functions.

Tests validate:
  - Per-layer FLOP counters against hand-computed formulas
  - compute_flops_saved with dense and MoE configs, and edge cases
  - compute_memory_bytes TP-sharding consistency
  - compute_flop_efficiency basic sanity
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.mem_cache.marconi_utils import (
    compute_flop_efficiency,
    compute_flops_saved,
    compute_memory_bytes,
    get_dense_mlp_flops,
    get_full_attn_flops,
    get_linear_attn_flops,
    get_moe_flops,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dense_config(
    num_layers=8,
    full_attention_interval=4,
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=1024,
    linear_num_value_heads=4,
    linear_value_head_dim=64,
    linear_key_head_dim=32,
):
    """Mock config for a dense hybrid model (e.g. Qwen3.5-27B)."""
    layers_block_type = [
        "attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
        for i in range(num_layers)
    ]
    full_attention_layer_ids = [
        i for i, t in enumerate(layers_block_type) if t == "attention"
    ]
    cfg = MagicMock()
    cfg.hidden_size = hidden_size
    cfg.num_attention_heads = num_attention_heads
    cfg.num_key_value_heads = num_key_value_heads
    cfg.head_dim = head_dim
    cfg.intermediate_size = intermediate_size
    cfg.linear_num_value_heads = linear_num_value_heads
    cfg.linear_value_head_dim = linear_value_head_dim
    cfg.linear_key_head_dim = linear_key_head_dim
    cfg.layers_block_type = layers_block_type
    cfg.full_attention_layer_ids = full_attention_layer_ids
    cfg.mlp_only_layers = []
    cfg.num_experts = 1  # dense — no routing
    return cfg


def _make_moe_config(
    num_layers=8,
    full_attention_interval=4,
    hidden_size=512,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    intermediate_size=1024,
    linear_num_value_heads=4,
    linear_value_head_dim=64,
    linear_key_head_dim=32,
    num_experts=16,
    num_experts_per_tok=2,
    moe_intermediate_size=256,
    shared_expert_intermediate_size=128,
):
    """Mock config for a MoE hybrid model (e.g. Qwen3-Next)."""
    cfg = _make_dense_config(
        num_layers=num_layers,
        full_attention_interval=full_attention_interval,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        intermediate_size=intermediate_size,
        linear_num_value_heads=linear_num_value_heads,
        linear_value_head_dim=linear_value_head_dim,
        linear_key_head_dim=linear_key_head_dim,
    )
    cfg.num_experts = num_experts
    cfg.num_experts_per_tok = num_experts_per_tok
    cfg.moe_intermediate_size = moe_intermediate_size
    cfg.shared_expert_intermediate_size = shared_expert_intermediate_size
    return cfg


def _make_cache_params(mamba_cache_per_req: int):
    p = MagicMock()
    p.mamba_cache_per_req = mamba_cache_per_req
    return p


def _make_model_config(total_kv_heads: int, tp_world_size: int = 1):
    mc = MagicMock()
    mc.get_num_kv_heads.side_effect = lambda tp: max(1, total_kv_heads // tp)
    return mc


# ---------------------------------------------------------------------------
# Per-layer FLOP counter tests
# ---------------------------------------------------------------------------


class TestGetFullAttnFlops(unittest.TestCase):
    def test_formula(self):
        seqlen, hidden_size, num_heads, num_kv_heads, head_dim = 10, 512, 8, 2, 64
        proj = 4 * seqlen * hidden_size * (hidden_size + num_kv_heads * head_dim)
        attn = 4 * seqlen**2 * num_heads * head_dim
        self.assertEqual(
            get_full_attn_flops(seqlen, hidden_size, num_heads, num_kv_heads, head_dim),
            proj + attn,
        )

    def test_gqa_uses_num_kv_heads(self):
        # With GQA (fewer KV heads), proj_flops should be smaller than MHA
        mha = get_full_attn_flops(10, 512, 8, 8, 64)   # num_kv_heads == num_heads
        gqa = get_full_attn_flops(10, 512, 8, 2, 64)   # num_kv_heads < num_heads
        self.assertGreater(mha, gqa)

    def test_scales_quadratically_with_seqlen(self):
        # attn_flops term is O(seqlen^2); for large seqlen it dominates
        f1 = get_full_attn_flops(100, 64, 4, 4, 16)
        f2 = get_full_attn_flops(200, 64, 4, 4, 16)
        # ratio should be between 2 (linear) and 4 (pure quadratic)
        self.assertGreater(f2 / f1, 2.0)
        self.assertLessEqual(f2 / f1, 4.0)

    def test_zero_seqlen(self):
        self.assertEqual(get_full_attn_flops(0, 512, 8, 2, 64), 0)


class TestGetLinearAttnFlops(unittest.TestCase):
    def test_formula(self):
        seqlen, hidden_size, num_heads, head_dim, state_size = 10, 512, 4, 64, 32
        intermediate_size = num_heads * head_dim
        expected = (
            2 * seqlen * intermediate_size * state_size
            + 8 * seqlen * hidden_size * intermediate_size
        )
        self.assertEqual(
            get_linear_attn_flops(seqlen, hidden_size, num_heads, head_dim, state_size),
            expected,
        )

    def test_linear_in_seqlen(self):
        f1 = get_linear_attn_flops(100, 512, 4, 64, 32)
        f2 = get_linear_attn_flops(200, 512, 4, 64, 32)
        self.assertAlmostEqual(f2 / f1, 2.0, places=9)

    def test_zero_seqlen(self):
        self.assertEqual(get_linear_attn_flops(0, 512, 4, 64, 32), 0)


class TestGetMoeFlops(unittest.TestCase):
    def test_formula(self):
        seqlen, hidden, k, expert_int, shared_int = 10, 512, 2, 256, 128
        expected = (
            4 * seqlen * hidden * expert_int * k
            + 4 * seqlen * hidden * shared_int
        )
        self.assertEqual(
            get_moe_flops(seqlen, hidden, k, expert_int, shared_int), expected
        )

    def test_includes_shared_expert(self):
        # shared-only FLOPs: set num_experts_per_tok=0 and check remaining == shared
        shared = get_moe_flops(10, 512, 0, 256, 128)
        expected_shared = 4 * 10 * 512 * 128
        self.assertEqual(shared, expected_shared)

    def test_zero_seqlen(self):
        self.assertEqual(get_moe_flops(0, 512, 2, 256, 128), 0)


class TestGetDenseMlpFlops(unittest.TestCase):
    def test_formula(self):
        seqlen, hidden, intermediate = 10, 512, 1024
        # SwiGLU: up + gate + down = 3 matmuls × 2 = 6
        self.assertEqual(
            get_dense_mlp_flops(seqlen, hidden, intermediate),
            6 * seqlen * hidden * intermediate,
        )

    def test_linear_in_seqlen(self):
        f1 = get_dense_mlp_flops(50, 512, 1024)
        f2 = get_dense_mlp_flops(100, 512, 1024)
        self.assertAlmostEqual(f2 / f1, 2.0, places=9)


# ---------------------------------------------------------------------------
# compute_flops_saved tests
# ---------------------------------------------------------------------------


class TestComputeFlopsSaved(unittest.TestCase):
    def setUp(self):
        # 8-layer model: attention at layers 3,7; linear at 0,1,2,4,5,6
        self.dense_cfg = _make_dense_config(num_layers=8, full_attention_interval=4)
        self.moe_cfg = _make_moe_config(num_layers=8, full_attention_interval=4)

    def test_zero_prefix_len(self):
        # No tokens cached → no FLOPs saved
        self.assertEqual(compute_flops_saved(0, 0, self.dense_cfg), 0.0)

    def test_positive_for_nonzero_prefix(self):
        self.assertGreater(compute_flops_saved(10, 20, self.dense_cfg), 0.0)

    def test_dense_vs_moe_ffn(self):
        # MoE model should save more FLOPs per prefix token (larger FFN)
        dense = compute_flops_saved(10, 20, self.dense_cfg)
        moe = compute_flops_saved(10, 20, self.moe_cfg)
        self.assertGreater(moe, dense)

    def test_deeper_prefix_saves_more_attn_flops(self):
        # For the same prefix_len, a deeper context (larger total_len) saves more
        # attention FLOPs due to the quadratic term
        shallow = compute_flops_saved(10, 20, self.dense_cfg)   # parent_len=10
        deep = compute_flops_saved(10, 100, self.dense_cfg)     # parent_len=90
        self.assertGreater(deep, shallow)

    def test_prefix_equals_total_len(self):
        # parent_len=0 → all attn FLOPs saved from scratch
        full = compute_flops_saved(20, 20, self.dense_cfg)
        partial = compute_flops_saved(10, 20, self.dense_cfg)
        self.assertGreater(full, partial)

    def test_dense_config_does_not_use_moe_flops(self):
        # For a purely dense model the FFN contribution should be
        # 6 * prefix_len * hidden_size * intermediate_size × num_layers
        prefix_len = 5
        cfg = self.dense_cfg
        num_layers = len(cfg.layers_block_type)
        expected_ffn = (
            6 * prefix_len * cfg.hidden_size * cfg.intermediate_size * num_layers
        )
        # Compute only FFN contribution by setting all attn/SSM params to 0
        cfg0 = _make_dense_config(
            num_layers=8,
            full_attention_interval=4,
            hidden_size=0,   # zero attn/SSM FLOPs
            num_attention_heads=0,
            num_key_value_heads=0,
            head_dim=0,
            intermediate_size=cfg.intermediate_size,
            linear_num_value_heads=0,
            linear_value_head_dim=0,
            linear_key_head_dim=0,
        )
        result = compute_flops_saved(prefix_len, 2 * prefix_len, cfg0)
        self.assertEqual(result, expected_ffn)


# ---------------------------------------------------------------------------
# compute_memory_bytes tests
# ---------------------------------------------------------------------------


class TestComputeMemoryBytes(unittest.TestCase):
    def _bytes(self, prefix_len, mamba_bytes, total_kv_heads, tp, head_dim, num_attn_layers, kv_dtype_bytes=2):
        cache_params = _make_cache_params(mamba_bytes)
        model_config = _make_model_config(total_kv_heads, tp)
        config = MagicMock()
        config.head_dim = head_dim
        config.full_attention_layer_ids = list(range(num_attn_layers))
        return compute_memory_bytes(
            prefix_len, cache_params, config, model_config, tp, kv_dtype_bytes
        )

    def test_zero_prefix_len_returns_ssm_only(self):
        # KV bytes = 0 when prefix_len = 0
        result = self._bytes(0, mamba_bytes=4096, total_kv_heads=8, tp=1, head_dim=64, num_attn_layers=2)
        self.assertEqual(result, 4096)

    def test_kv_scales_linearly_with_prefix_len(self):
        r1 = self._bytes(10, 0, 8, 1, 64, 2)
        r2 = self._bytes(20, 0, 8, 1, 64, 2)
        self.assertAlmostEqual(r2 / r1, 2.0, places=9)

    def test_tp_sharding_halves_kv_bytes(self):
        tp1 = self._bytes(10, 0, 8, tp=1, head_dim=64, num_attn_layers=2)
        tp2 = self._bytes(10, 0, 8, tp=2, head_dim=64, num_attn_layers=2)
        self.assertAlmostEqual(tp1 / tp2, 2.0, places=9)

    def test_tp_sharding_does_not_affect_ssm_bytes(self):
        # mamba_cache_per_req already carries TP-sharded values; it must not be divided again
        tp1 = self._bytes(0, mamba_bytes=8192, total_kv_heads=8, tp=1, head_dim=64, num_attn_layers=2)
        tp4 = self._bytes(0, mamba_bytes=8192, total_kv_heads=8, tp=4, head_dim=64, num_attn_layers=2)
        self.assertEqual(tp1, tp4)

    def test_kv_dtype_bytes_scales_result(self):
        fp16 = self._bytes(10, 0, 8, 1, 64, 2, kv_dtype_bytes=2)
        fp8 = self._bytes(10, 0, 8, 1, 64, 2, kv_dtype_bytes=1)
        self.assertAlmostEqual(fp16 / fp8, 2.0, places=9)

    def test_kv_formula(self):
        prefix_len, total_kv_heads, tp, head_dim, num_attn_layers = 10, 8, 2, 64, 3
        kv_dtype_bytes = 2
        kv_heads_per_gpu = max(1, total_kv_heads // tp)
        expected_kv = prefix_len * kv_heads_per_gpu * head_dim * 2 * kv_dtype_bytes * num_attn_layers
        result = self._bytes(prefix_len, 0, total_kv_heads, tp, head_dim, num_attn_layers, kv_dtype_bytes)
        self.assertEqual(result, expected_kv)

    def test_tp_head_replication_min_one(self):
        # When num_kv_heads < tp_world_size, heads are replicated: min 1 head per GPU
        result = self._bytes(10, 0, total_kv_heads=2, tp=8, head_dim=64, num_attn_layers=1)
        expected_kv = 10 * 1 * 64 * 2 * 2 * 1  # max(1, 2//8)=1
        self.assertEqual(result, expected_kv)


# ---------------------------------------------------------------------------
# compute_flop_efficiency tests
# ---------------------------------------------------------------------------


class TestComputeFlopEfficiency(unittest.TestCase):
    def setUp(self):
        self.cfg = _make_dense_config()
        self.cache_params = _make_cache_params(mamba_cache_per_req=1024)
        self.model_config = _make_model_config(total_kv_heads=4)

    def test_positive_for_nonzero_prefix(self):
        score = compute_flop_efficiency(
            10, 20, self.cache_params, self.cfg, self.model_config, tp_world_size=1
        )
        self.assertGreater(score, 0.0)

    def test_zero_prefix_returns_near_zero(self):
        score = compute_flop_efficiency(
            0, 0, self.cache_params, self.cfg, self.model_config, tp_world_size=1
        )
        # numerator=0, denominator=ssm_bytes+1e-8
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_longer_prefix_higher_efficiency(self):
        # Deeper prefix saves proportionally more attention FLOPs (quadratic benefit)
        short = compute_flop_efficiency(
            5, 10, self.cache_params, self.cfg, self.model_config, tp_world_size=1
        )
        long = compute_flop_efficiency(
            50, 100, self.cache_params, self.cfg, self.model_config, tp_world_size=1
        )
        self.assertGreater(long, short)


if __name__ == "__main__":
    unittest.main()
