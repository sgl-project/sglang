import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.models.dspark import EntryClass
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config
from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.utils.hf_transformers import get_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _glm52_redhat_dspark_config_dict() -> dict:
    return {
        "architectures": ["DSparkDraftModel"],
        "aux_hidden_state_layer_ids": [8, 23, 39, 55, 70],
        "block_size": 8,
        "markov_head_type": "vanilla",
        "markov_rank": 256,
        "mask_token_id": 154856,
        "speculators_config": {
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
        "transformer_layer_config": {
            "hidden_size": 6144,
            "model_type": "qwen3",
            "num_hidden_layers": 5,
            "vocab_size": 154880,
        },
    }


def _to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _to_namespace(nested) for key, nested in value.items()}
        )
    if isinstance(value, list):
        return [_to_namespace(nested) for nested in value]
    return value


def _glm52_redhat_dspark_config() -> SimpleNamespace:
    return _to_namespace(_glm52_redhat_dspark_config_dict())


class TestGLM52RedHatDSparkConfig(CustomTestCase):
    def test_dflash_parser_reads_nested_transformer_config_and_aux_layers(self):
        parsed = parse_dflash_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.num_hidden_layers, 5)
        self.assertEqual(parsed.block_size, 8)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.num_target_layers, 71)

    def test_dspark_parser_prefers_speculators_tokens_for_gamma(self):
        parsed = parse_dspark_draft_config(
            draft_hf_config=_glm52_redhat_dspark_config()
        )

        self.assertEqual(parsed.gamma, 7)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])
        self.assertEqual(parsed.markov_rank, 256)
        self.assertEqual(parsed.markov_head_type, "vanilla")
        self.assertEqual(parsed.mask_token_id, 154856)

    def test_dspark_parser_selects_default_proposal_method(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["block_size"] = 8
        raw_config["speculators_config"]["default_proposal_method"] = "sample"
        raw_config["speculators_config"]["proposal_methods"] = [
            {
                "proposal_type": "greedy",
                "speculative_tokens": 7,
                "verifier_accept_k": 1,
            },
            {
                "proposal_type": "sample",
                "speculative_tokens": 5,
                "verifier_accept_k": 1,
            },
        ]

        parsed = parse_dspark_draft_config(draft_hf_config=_to_namespace(raw_config))

        self.assertEqual(parsed.gamma, 5)

    def test_dspark_parser_falls_back_to_first_proposal_method(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["speculators_config"]["default_proposal_method"] = "missing"
        raw_config["speculators_config"]["proposal_methods"] = [
            {
                "proposal_type": "greedy",
                "speculative_tokens": 4,
                "verifier_accept_k": 1,
            },
            {
                "proposal_type": "sample",
                "speculative_tokens": 6,
                "verifier_accept_k": 1,
            },
        ]

        parsed = parse_dspark_draft_config(draft_hf_config=raw_config)

        self.assertEqual(parsed.gamma, 4)

    def test_dspark_parser_rejects_invalid_speculative_tokens(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["speculators_config"]["proposal_methods"][0][
            "speculative_tokens"
        ] = 0

        with self.assertRaisesRegex(ValueError, "speculative_tokens must be positive"):
            parse_dspark_draft_config(draft_hf_config=raw_config)

    def test_dict_text_config_none_falls_back_to_transformer_layer_config(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["text_config"] = None

        parsed = parse_dflash_draft_config(draft_hf_config=raw_config)

        self.assertEqual(parsed.num_hidden_layers, 5)
        self.assertEqual(parsed.target_layer_ids, [8, 23, 39, 55, 70])

    def test_dflash_parser_rejects_empty_aux_layer_ids(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["aux_hidden_state_layer_ids"] = []

        with self.assertRaisesRegex(ValueError, "target_layer_ids must be non-empty"):
            parse_dflash_draft_config(draft_hf_config=raw_config)

    def test_prefixed_dspark_block_size_overrides_speculators_gamma(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["dspark_block_size"] = 3

        parsed = parse_dspark_draft_config(draft_hf_config=raw_config)

        self.assertEqual(parsed.gamma, 3)

    def test_prefixed_target_layer_ids_must_be_non_empty(self):
        raw_config = _glm52_redhat_dspark_config_dict()
        raw_config["dspark_target_layer_ids"] = []

        with self.assertRaisesRegex(ValueError, "must be a non-empty list"):
            parse_dspark_draft_config(draft_hf_config=raw_config)

    def test_dspark_draft_model_is_registered(self):
        self.assertIn("DSparkDraftModel", {cls.__name__ for cls in EntryClass})

    def test_hf_loader_falls_back_for_missing_top_level_model_type(self):
        raw_config = _glm52_redhat_dspark_config_dict()

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / "config.json"
            config_path.write_text(json.dumps(raw_config), encoding="utf-8")

            loaded = get_config(str(config_path.parent), trust_remote_code=True)

        self.assertEqual(loaded.architectures, ["DSparkDraftModel"])
        self.assertEqual(loaded.hidden_size, 6144)
        self.assertEqual(loaded.num_hidden_layers, 5)
        self.assertEqual(loaded.vocab_size, 154880)
        self.assertEqual(parse_dspark_draft_config(draft_hf_config=loaded).gamma, 7)


if __name__ == "__main__":
    unittest.main()
