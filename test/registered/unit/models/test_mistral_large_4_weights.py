import unittest

from sglang.srt.models.mistral_large_4 import mistral_large_4_weight_name
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_NUM_LAYERS = 72
_KV_SOURCES = list(range(_NUM_LAYERS))
for _consumer in range(39, _NUM_LAYERS, 4):
    _KV_SOURCES[_consumer] = 35

# fmt: off
_CHECKPOINT_TO_PARAM = {
    "tok_embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "lerope.alpha": "model.lerope_alpha",
    "layers.0.attention_norm.weight": "model.layers.0.input_layernorm.weight",
    "layers.0.ffn_norm.weight": "model.layers.0.post_attention_layernorm.weight",
    "layers.0.attention.wq.weight": "model.layers.0.self_attn.q_proj.weight",
    "layers.0.attention.wq.qscale_weight": "model.layers.0.self_attn.q_proj.weight_scale_inv",
    "layers.0.attention.wo.weight": "model.layers.0.self_attn.o_proj.weight",
    "layers.0.attention.wo.qscale_weight": "model.layers.0.self_attn.o_proj.weight_scale_inv",
    "layers.0.attention.wkv_a_with_mqa.weight": "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "layers.0.attention.wkv_b.weight": "model.layers.0.self_attn.kv_b_proj.weight",
    "layers.0.attention.ssmax_scaling.s": "model.layers.0.self_attn.ssmax_scaling_s",
    "layers.0.attention.ssmax_scaling.b": "model.layers.0.self_attn.ssmax_scaling_b",
    "layers.0.attention.indexer.wq_b.weight": "model.layers.0.self_attn.indexer.wq_b.weight",
    "layers.0.attention.indexer.wq_b.qscale_weight": "model.layers.0.self_attn.indexer.wq_b.weight_scale_inv",
    "layers.0.attention.indexer.wk.weight": "model.layers.0.self_attn.indexer.wk.weight",
    "layers.0.attention.indexer.weights_proj.weight": "model.layers.0.self_attn.indexer.weights_proj.weight",
    "layers.0.attention.indexer.k_norm.bias": "model.layers.0.self_attn.indexer.k_norm.bias",
    "layers.0.attention.indexer.q_input_norm.weight": "model.layers.0.self_attn.indexer.q_input_norm.weight",
    "layers.0.attention.indexer.ssmax_scaling.s": "model.layers.0.self_attn.indexer.ssmax_scaling_s",
    "layers.0.feed_forward.w1.qscale_weight": "model.layers.0.mlp.gate_proj.weight_scale_inv",
    "layers.0.feed_forward.w3.weight": "model.layers.0.mlp.up_proj.weight",
    "layers.1.gate.weight": "model.layers.1.mlp.gate.weight",
    "layers.1.experts.127.w2.qscale_weight": "model.layers.1.mlp.experts.127.down_proj.weight_scale_inv",
    "layers.1.shared_experts.w1.weight": "model.layers.1.mlp.shared_experts.gate_proj.weight",
    "layers.3.attention.wkv_a_with_mqa.weight": "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
    "layers.3.attention.kv_a_norm.weight": "model.layers.3.self_attn.kv_a_layernorm.weight",
    "layers.35.attention.wkv_a_with_mqa.weight": "model.layers.35.self_attn.kv_a_proj_with_mqa.weight",
    "layers.39.attention.wkv_a_with_mqa.weight": None,
    "layers.39.attention.kv_a_norm.weight": None,
    "layers.39.attention.wkv_b.weight": "model.layers.39.self_attn.kv_b_proj.weight",
    "layers.0.attention.k_fake_quantizer.qscale_act": None,
    "mtp_dspark.backbone.layers.0.attention.wq.weight": None,
    "mtp_dspark.autoregressive_head.embedding.weight": None,
    "vision_encoder.ln_pre.weight": None,
    "vision_language_adapter.w_in.weight": None,
}
# fmt: on


def _map(name: str):
    return mistral_large_4_weight_name(name, kv_sources=_KV_SOURCES, block_quant=True)


class TestMistralLarge4Weights(CustomTestCase):
    def test_checkpoint_tensors_map_to_params_or_explicit_skips(self):
        self.assertEqual(
            {name: _map(name) for name in _CHECKPOINT_TO_PARAM}, _CHECKPOINT_TO_PARAM
        )
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            _map("layers.0.attention.sinks")


if __name__ == "__main__":
    unittest.main()
