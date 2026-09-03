"""Unit tests for GLM-5.3-Flash (glm5_next) per-layer LoRA geometry.

GLM-5.3-Flash mixes two attention layouts: KDA linear attention (q/k/v/o plus the b/f_a/f_b/
g_a/g_b gate projections, geometry from `linear_attn_config`) and DSA sparse MLA
(`fused_qkv_a_proj_with_mqa` / `q_b_proj` / `kv_b_proj`, MLA geometry). `o_proj` exists in both
but with different input widths, and the MLP is dense for the first `first_k_dense_replace`
layers and MoE (shared expert + routed experts) afterwards. The generic
`get_default_hidden_dim` assumes one attention shape for every layer, so
`Glm5NextForConditionalGeneration.get_hidden_dim` resolves the shapes per layer.

These tests exercise that hook and the KDA gate name registration directly against a small
config -- no CUDA, no server, no weights.

Usage:
    python -m pytest test/registered/unit/lora/test_glm5_next_lora_hidden_dim_unit.py -v
"""

from sglang.test.ci.ci_register import register_cpu_ci

# CPU-only unit test; no CUDA/distributed dependencies.
register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.lora.utils import (
    _KNOWN_LORA_TARGET_MODULES,
    ATTN_TP_LORA_MODULE_NAMES,
    KDA_GATE_LORA_NAMES,
    REPLICATED_LINEAR_LORA_NAMES,
    get_default_hidden_dim,
    get_normalized_target_modules,
)
from sglang.srt.models.glm5_next import Glm5NextForConditionalGeneration
from sglang.srt.utils.common import SUPPORTED_LORA_TARGET_MODULES

HIDDEN = 512
KDA_HEADS, KDA_HEAD_DIM = 8, 32
N_HEADS, QK_NOPE, V_HEAD = 4, 64, 48
Q_LORA, KV_LORA = 96, 48
INTER, MOE_INTER, N_SHARED = 1024, 128, 1
# 4 layers: KDA, DSA, KDA, DSA; layer 0 dense, layers 1-3 MoE (first_k_dense_replace=1)
KDA_LAYERS = [0, 2]


def _make_fake_model():
    config = Glm5NextTextConfig(
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        moe_intermediate_size=MOE_INTER,
        num_hidden_layers=4,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        n_routed_experts=16,
        n_shared_experts=N_SHARED,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=0,
        v_head_dim=V_HEAD,
        index_n_heads=4,
        index_head_dim=32,
        index_topk=64,
        linear_attn_config={
            "num_heads": KDA_HEADS,
            "head_dim": KDA_HEAD_DIM,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "kda_layers": KDA_LAYERS,
            "full_attn_layers": [1, 3],
        },
    )
    model = Glm5NextForConditionalGeneration.__new__(Glm5NextForConditionalGeneration)
    model.config = config
    return model


class TestGlm5NextPerLayerDims(unittest.TestCase):
    def setUp(self):
        self.model = _make_fake_model()
        self.kda_proj = KDA_HEADS * KDA_HEAD_DIM

    def dims(self, name, layer):
        return self.model.get_hidden_dim(name, layer)

    def test_kda_projections(self):
        for layer in KDA_LAYERS:
            self.assertEqual(self.dims("qkv_proj", layer), (HIDDEN, 3 * self.kda_proj))
            self.assertEqual(self.dims("o_proj", layer), (self.kda_proj, HIDDEN))
            self.assertEqual(self.dims("b_proj", layer), (HIDDEN, KDA_HEADS))
            self.assertEqual(self.dims("f_a_proj", layer), (HIDDEN, KDA_HEAD_DIM))
            self.assertEqual(self.dims("g_a_proj", layer), (HIDDEN, KDA_HEAD_DIM))
            self.assertEqual(self.dims("f_b_proj", layer), (KDA_HEAD_DIM, self.kda_proj))
            self.assertEqual(self.dims("g_b_proj", layer), (KDA_HEAD_DIM, self.kda_proj))

    def test_dsa_projections(self):
        for layer in (1, 3):
            # o_proj follows the MLA value geometry on DSA layers, the KDA geometry elsewhere
            self.assertEqual(self.dims("o_proj", layer), (N_HEADS * V_HEAD, HIDDEN))
            self.assertEqual(self.dims("fused_qkv_a_proj_with_mqa", layer), (HIDDEN, Q_LORA + KV_LORA))
            self.assertEqual(self.dims("q_b_proj", layer), (Q_LORA, N_HEADS * QK_NOPE))
            self.assertEqual(self.dims("kv_b_proj", layer), (KV_LORA, N_HEADS * (QK_NOPE + V_HEAD)))
        self.assertNotEqual(self.dims("o_proj", 0), self.dims("o_proj", 1))

    def test_mlp_dense_vs_moe(self):
        self.assertEqual(self.dims("gate_up_proj", 0), (HIDDEN, 2 * INTER))
        self.assertEqual(self.dims("down_proj", 0), (INTER, HIDDEN))
        shared = MOE_INTER * N_SHARED
        for layer in (1, 2, 3):
            self.assertEqual(self.dims("gate_up_proj", layer), (HIDDEN, 2 * shared))
            self.assertEqual(self.dims("down_proj", layer), (shared, HIDDEN))
            self.assertEqual(self.dims("gate_up_proj_moe", layer), (HIDDEN, 2 * MOE_INTER))
            self.assertEqual(self.dims("down_proj_moe", layer), (MOE_INTER, HIDDEN))

    def test_generic_fallback_for_kda_gates(self):
        # models without the per-layer hook still get the KDA gate geometry from linear_attn_config
        cfg = self.model.config
        self.assertEqual(get_default_hidden_dim("b_proj", cfg, 0), (HIDDEN, KDA_HEADS))
        self.assertEqual(get_default_hidden_dim("f_b_proj", cfg, 0), (KDA_HEAD_DIM, self.kda_proj))


class TestKdaGateRegistration(unittest.TestCase):
    def test_names_known_and_selectable(self):
        self.assertTrue(KDA_GATE_LORA_NAMES <= _KNOWN_LORA_TARGET_MODULES)
        for name in KDA_GATE_LORA_NAMES:
            self.assertIn(name, SUPPORTED_LORA_TARGET_MODULES)
        self.assertEqual(
            get_normalized_target_modules(["b_proj", "f_a_proj", "self_attn.g_b_proj"]),
            {"b_proj", "f_a_proj", "g_b_proj"},
        )

    def test_parallelism_tables(self):
        # low-rank gate inputs are replicated, head-sharded gates ride the attention-TP group
        for name in ("f_a_proj", "g_a_proj"):
            self.assertIn(name, REPLICATED_LINEAR_LORA_NAMES)
        for name in ("b_proj", "f_b_proj", "g_b_proj"):
            self.assertIn(name, ATTN_TP_LORA_MODULE_NAMES)

    def test_supported_modules_declared(self):
        for name in ("qkv_proj", "o_proj", "fused_qkv_a_proj_with_mqa", "q_b_proj", "kv_b_proj", "gate_up_proj", "down_proj"):
            self.assertIn(name, Glm5NextForConditionalGeneration.supported_lora_modules)
        for name in KDA_GATE_LORA_NAMES:
            self.assertIn(name, Glm5NextForConditionalGeneration.supported_lora_modules)


if __name__ == "__main__":
    unittest.main()
