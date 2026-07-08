import unittest
from types import SimpleNamespace

from sglang.srt.models.dspark import _normalize_dspark_transformer_config
from sglang.srt.speculative.dspark_components.dspark_utils import (
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _glm52_dspark_config():
    return {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
        "block_size": 8,
        "confidence_head_with_markov": True,
        "draft_vocab_size": 154880,
        "enable_confidence_head": True,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 154856,
        "speculators_config": {
            "algorithm": "dspark",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                }
            ],
            "verifier": {
                "architectures": ["GlmMoeDsaForCausalLM"],
                "name_or_path": "zai-org/GLM-5.2-FP8",
            },
        },
        "speculators_model_type": "dspark",
        "transformer_layer_config": {
            "attention_bias": False,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 6144,
            "intermediate_size": 12288,
            "layer_types": ["full_attention"] * 5,
            "max_position_embeddings": 1048576,
            "model_type": "qwen3",
            "num_attention_heads": 64,
            "num_hidden_layers": 5,
            "num_key_value_heads": 64,
            "rms_norm_eps": 1e-5,
            "rope_parameters": {
                "rope_theta": 8000000,
                "rope_type": "default",
            },
            "vocab_size": 154880,
        },
    }


class TestDSparkGLMSpeculatorConfig(CustomTestCase):
    def test_parse_speculators_config_shape(self):
        parsed = parse_dspark_draft_config(
            draft_hf_config=_glm52_dspark_config(),
        )

        self.assertEqual(parsed.gamma, 7)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.mask_token_id, 154856)
        self.assertEqual(parsed.markov_rank, 256)
        self.assertEqual(parsed.markov_head_type, "vanilla")

    def test_promote_transformer_layer_config(self):
        cfg = SimpleNamespace(**_glm52_dspark_config())
        _normalize_dspark_transformer_config(cfg)

        self.assertEqual(cfg.hidden_size, 6144)
        self.assertEqual(cfg.num_hidden_layers, 5)
        self.assertEqual(cfg.num_attention_heads, 64)
        self.assertEqual(cfg.num_key_value_heads, 64)
        self.assertEqual(cfg.vocab_size, 154880)
        self.assertEqual(cfg.rope_parameters["rope_theta"], 8000000)


if __name__ == "__main__":
    unittest.main()
