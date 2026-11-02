import contextlib
import json
import os
import tempfile
import unittest

from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    dsa_layer_skips_topk,
    dsa_supports_dense_attention,
    get_dsa_index_topk,
    is_deepseek_dsa,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KV_SOURCES = list(range(72))
for _consumer in range(39, 72, 4):
    _KV_SOURCES[_consumer] = 35

_INDEX_SHARING = (
    ["full"] * 17
    + ["shared", "shared", "full", "full"] * 13
    + ["shared", "shared", "full"]
)

_PARAMS = {
    "dim": 6144,
    "n_layers": 72,
    "head_dim": 128,
    "hidden_dim": 24576,
    "n_heads": 64,
    "n_kv_heads": 64,
    "rope_theta": 10000.0,
    "norm_eps": 1e-05,
    "vocab_size": 204800,
    "tied_embeddings": False,
    "max_position_embeddings": 262144,
    "ragged_attention": None,
    "nope": "rope,rope,rope,nope",
    "softmax_tanh_gating": 30.0,
    "llama_4_scaling": None,
    "q_lora_rank": None,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 64,
    "kv_lora_rank": 512,
    "v_head_dim": 128,
    "ssmax_scaling": True,
    "lerope": True,
    "quantization": {
        "qscheme_act": "BLOCK",
        "qformat_weight": "fp8_e4m3",
        "qdynamic_act": True,
        "qscheme_weight": "BLOCK",
        "use_ue8m0_for_qscales": True,
        "block_size_out": 128,
        "block_size_in": 128,
        "quantize_kv_a": False,
    },
    "yarn": {
        "original_max_position_embeddings": 512,
        "factor": 512,
        "apply_scale": False,
        "beta": 32,
        "alpha": 1,
    },
    "kv_sharing": {"source_layers_str": ",".join(map(str, _KV_SOURCES))},
    "moe": {
        "expert_parallel": 1,
        "expert_model_parallel": 1,
        "route_every_n": 1,
        "first_k_dense_replace": 1,
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "num_expert_groups": 1,
        "num_expert_groups_per_tok": 1,
        "routed_scale": 1.0,
        "expert_hidden_dim": 6144,
        "num_shared_experts": 1,
    },
    "vision_encoder": {
        "image_token_id": 10,
        "image_break_token_id": 12,
        "image_end_token_id": 13,
        "intermediate_size": 6144,
        "num_hidden_layers": 30,
        "num_attention_heads": 16,
        "mm_projector_id": "patch_merge",
        "spatial_merge_size": 2,
        "hidden_size": 2048,
        "num_channels": 3,
        "image_size": 1540,
        "max_image_size": 1540,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "add_pre_mm_projector_layer_norm": True,
        "adapter_bias": False,
    },
    "dsa": {
        "index_topk": 2048,
        "index_n_heads": 32,
        "index_head_dim": 128,
        "index_rope_head_dim": 64,
        "indexer_match_nope": True,
        "indexer_ssmax_scaling": True,
        "indexer_norm_eps": 1e-06,
        "index_sharing": ",".join(_INDEX_SHARING),
        "fp8_indexer": True,
        "fp8_kv_cache": True,
    },
}


@contextlib.contextmanager
def _params_dir(params=_PARAMS):
    with tempfile.TemporaryDirectory() as model_dir:
        with open(os.path.join(model_dir, "params.json"), "w") as f:
            json.dump(params, f)
        yield model_dir


class TestMistralLarge4Config(CustomTestCase):
    def test_params_only_checkpoint_resolves_full_config(self):
        for vision in (True, False):
            params = {
                k: v for k, v in _PARAMS.items() if vision or k != "vision_encoder"
            }
            with self.subTest(vision=vision), _params_dir(params) as model_dir:
                model_config = ModelConfig(model_path=model_dir)
                config = model_config.hf_config
                text = config.text_config if vision else config
                arch = (
                    "PixtralForConditionalGeneration"
                    if vision
                    else "MistralLarge4ForCausalLM"
                )
                self.assertEqual(config.architectures, [arch])
                self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
                self.assertEqual(
                    (
                        model_config.kv_lora_rank,
                        model_config.qk_nope_head_dim,
                        model_config.qk_rope_head_dim,
                        model_config.v_head_dim,
                        model_config.index_head_dim,
                    ),
                    (512, 64, 64, 128, 128),
                )
                expected = {
                    "model_type": "mistral_large_4",
                    "architectures": ["MistralLarge4ForCausalLM"],
                    "q_lora_rank": None,
                    "kv_lora_rank": 512,
                    "qk_nope_head_dim": 64,
                    "qk_rope_head_dim": 64,
                    "v_head_dim": 128,
                    "n_routed_experts": 128,
                    "num_experts_per_tok": 4,
                    "n_shared_experts": 1,
                    "first_k_dense_replace": 1,
                    "max_seq_len": 262144,
                    "softmax_tanh_gating": 30.0,
                    "norm_topk_prob": True,
                    "nope_layer_flags": [layer % 4 == 3 for layer in range(72)],
                    "indexer_nope_layer_flags": [layer % 4 == 3 for layer in range(72)],
                    "index_topk_pattern": [
                        "S" if value == "shared" else "F" for value in _INDEX_SHARING
                    ],
                    "index_topk": 2048,
                    "index_n_heads": 32,
                    "index_head_dim": 128,
                    "index_kv_dtype": "fp8",
                    "indexer_ssmax_scaling": True,
                    "kv_sharing_config": {
                        "source_layers_str": ",".join(map(str, _KV_SOURCES))
                    },
                }
                self.assertEqual(
                    {key: getattr(text, key) for key in expected}, expected
                )
                self.assertEqual(
                    config.quantization_config["weight_block_size"], [128, 128]
                )
                self.assertEqual(
                    config.quantization_config["activation_scheme"], "dynamic"
                )
                self.assertIn(
                    "model.layers.71.self_attn.kv_b_proj",
                    config.quantization_config["ignored_layers"],
                )
                self.assertTrue(is_deepseek_dsa(config))
                self.assertFalse(dsa_supports_dense_attention(config))
                self.assertEqual(get_dsa_index_topk(config), 2048)
                self.assertEqual(
                    [dsa_layer_skips_topk(config, layer) for layer in range(72)],
                    [value == "shared" for value in _INDEX_SHARING],
                )
                if vision:
                    self.assertEqual(
                        (
                            config.vision_config.hidden_size,
                            config.vision_config.num_hidden_layers,
                            config.vision_config.patch_size,
                        ),
                        (2048, 30, 14),
                    )


if __name__ == "__main__":
    unittest.main()
