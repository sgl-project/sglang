import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import PretrainedConfig

from sglang.srt.configs.model_config import (
    _normalize_nested_transformer_config,
    _restore_glm_moe_dsa_head_dims_from_raw_config,
)
from sglang.srt.models.dspark import EntryClass, normalize_dspark_draft_config
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.draft_worker_common import (
    _resolve_draft_attention_backend,
    draft_is_deepseek_v4,
)
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dspark_components.dspark_utils import (
    parse_dspark_draft_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _glm52_redhat_dspark_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DSparkDraftModel"],
        aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
        block_size=8,
        confidence_head_with_markov=True,
        enable_confidence_head=True,
        markov_head_type="vanilla",
        markov_rank=256,
        mask_token_id=154856,
        speculators_config={
            "algorithm": "dspark",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                }
            ],
        },
        transformer_layer_config={
            "attention_bias": False,
            "head_dim": 64,
            "hidden_size": 6144,
            "intermediate_size": 12288,
            "layer_types": ["full_attention"] * 5,
            "model_type": "qwen3",
            "num_attention_heads": 64,
            "num_hidden_layers": 5,
            "num_key_value_heads": 64,
            "rms_norm_eps": 1e-5,
            "vocab_size": 154880,
        },
    )


def _glm52_redhat_dspark_raw_config() -> dict:
    return dict(vars(_glm52_redhat_dspark_config()))


def _make_draft_server_args() -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
    server_args.speculative_draft_model_revision = None
    server_args.speculative_draft_attention_backend = "triton"
    server_args.trust_remote_code = False
    server_args.json_model_override_args = "{}"
    server_args.model_config_parser = "auto"
    return server_args


class TestGLM52RedHatDSparkConfig(CustomTestCase):
    def test_model_config_normalizer_materializes_nested_transformer_config(self):
        config = PretrainedConfig(
            architectures=["DSparkDraftModel"],
            aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
            transformer_layer_config={
                "hidden_size": 6144,
                "num_hidden_layers": 5,
                "vocab_size": 154880,
            },
        )

        _normalize_nested_transformer_config(config)

        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.num_target_layers, 71)

    def test_dflash_parser_reads_nested_transformer_config_and_aux_layers(self):
        parsed = parse_dflash_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.num_hidden_layers, 5)
        self.assertEqual(parsed.block_size, 8)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.num_target_layers, 71)

    def test_dspark_parser_reads_redhat_fields(self):
        parsed = parse_dspark_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.gamma, 7)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.markov_rank, 256)
        self.assertEqual(parsed.markov_head_type, "vanilla")
        self.assertEqual(parsed.mask_token_id, 154856)

    def test_dspark_draft_model_is_registered(self):
        self.assertIn("DSparkDraftModel", {cls.__name__ for cls in EntryClass})

    def test_normalize_dspark_draft_config_materializes_backbone_fields(self):
        config = _glm52_redhat_dspark_config()
        normalize_dspark_draft_config(config)

        self.assertEqual(config.hidden_size, 6144)
        self.assertEqual(config.num_hidden_layers, 5)
        self.assertEqual(config.vocab_size, 154880)
        self.assertEqual(config.num_target_layers, 71)


class TestGLM52RedHatDSparkWorkerConfig(CustomTestCase):
    def test_draft_worker_common_accepts_missing_model_type_config(self):
        server_args = _make_draft_server_args()
        missing_model_type = ValueError(
            "Unrecognized model in RedHatAI/GLM-5.2-speculator.dspark. "
            "Should have a `model_type` key in its config.json."
        )

        with (
            patch(
                "sglang.srt.utils.hf_transformers.config.get_config",
                side_effect=missing_model_type,
            ),
            patch(
                "transformers.PretrainedConfig.get_config_dict",
                return_value=(_glm52_redhat_dspark_raw_config(), {}),
            ),
        ):
            self.assertFalse(draft_is_deepseek_v4(server_args=server_args))
            self.assertEqual(
                _resolve_draft_attention_backend(
                    draft_server_args=server_args, algo_label="DSpark"
                ),
                "triton",
            )


class TestGLM52RawHeadDims(CustomTestCase):
    def test_restore_raw_glm_moe_dsa_head_dims(self):
        config = PretrainedConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            qk_nope_head_dim=192,
            qk_rope_head_dim=192,
            qk_head_dim=384,
            v_head_dim=256,
        )
        raw_config = {
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "qk_head_dim": 256,
            "v_head_dim": 256,
        }

        _restore_glm_moe_dsa_head_dims_from_raw_config(
            config,
            raw_config,
            {},
            "unused",
            False,
            None,
            {},
        )

        self.assertEqual(config.qk_nope_head_dim, 192)
        self.assertEqual(config.qk_rope_head_dim, 64)
        self.assertEqual(config.qk_head_dim, 256)
        self.assertEqual(config.v_head_dim, 256)

    def test_json_override_wins_over_raw_head_dims(self):
        config = PretrainedConfig(
            architectures=["GlmMoeDsaForCausalLM"],
            qk_nope_head_dim=192,
            qk_rope_head_dim=192,
            qk_head_dim=384,
            v_head_dim=256,
        )

        _restore_glm_moe_dsa_head_dims_from_raw_config(
            config,
            {"qk_rope_head_dim": 64},
            {"qk_rope_head_dim": 128},
            "unused",
            False,
            None,
            {},
        )

        self.assertEqual(config.qk_rope_head_dim, 128)


if __name__ == "__main__":
    unittest.main()
