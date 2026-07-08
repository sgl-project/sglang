import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.speculative_hook import (
    _handle_dflash,
    _handle_dspark,
    _target_checkpoint_bundles_dspark_draft,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_BUNDLED_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash-DSpark"
_PLAIN_MODEL_PATH = "deepseek-ai/DeepSeek-V4-Flash"


def _bundled_hf_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )


def _plain_hf_config() -> SimpleNamespace:
    return SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])


def _redhat_glm52_dspark_config() -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["DSparkDraftModel"],
        aux_hidden_state_layer_ids=[8, 23, 39, 55, 70],
        block_size=8,
        markov_head_type="vanilla",
        markov_rank=256,
        mask_token_id=154856,
        speculators_config={
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                }
            ],
        },
        transformer_layer_config={
            "hidden_size": 6144,
            "num_hidden_layers": 5,
            "vocab_size": 154880,
        },
    )


def _make_dspark_server_args(
    *, model_path: str, hf_config: SimpleNamespace
) -> ServerArgs:
    server_args = ServerArgs(model_path="dummy")
    server_args.model_path = model_path
    server_args.device = "cuda"
    server_args.speculative_algorithm = "DSPARK"
    server_args.speculative_draft_model_path = None
    server_args.speculative_dspark_block_size = 5
    server_args.model_config = SimpleNamespace(hf_config=hf_config)
    return server_args


class TestTargetCheckpointBundlesDsparkDraft(CustomTestCase):
    def test_bundled_dsv4_config_is_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        self.assertTrue(_target_checkpoint_bundles_dspark_draft(server_args))

    def test_plain_target_config_is_not_detected(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        self.assertFalse(_target_checkpoint_bundles_dspark_draft(server_args))


class TestDsparkDraftPathDefaulting(CustomTestCase):
    def test_bundled_checkpoint_defaults_draft_path_to_model_path(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        _handle_dspark(server_args)
        self.assertEqual(server_args.speculative_draft_model_path, _BUNDLED_MODEL_PATH)
        self.assertEqual(server_args.speculative_num_draft_tokens, 6)

    def test_plain_target_without_draft_path_raises(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        with self.assertRaises(ValueError):
            _handle_dspark(server_args)

    def test_explicit_draft_path_is_not_overwritten(self):
        server_args = _make_dspark_server_args(
            model_path=_BUNDLED_MODEL_PATH, hf_config=_bundled_hf_config()
        )
        server_args.speculative_draft_model_path = "deepseek-ai/some-other-dspark-draft"
        _handle_dspark(server_args)
        self.assertEqual(
            server_args.speculative_draft_model_path,
            "deepseek-ai/some-other-dspark-draft",
        )

    def test_external_config_infers_gamma_from_speculators_config(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
        server_args.speculative_dspark_block_size = None

        with patch(
            "sglang.srt.arg_groups.speculative_hook._load_speculative_draft_config",
            return_value=_redhat_glm52_dspark_config(),
        ):
            _handle_dspark(server_args)

        self.assertEqual(server_args.speculative_num_draft_tokens, 8)

    def test_explicit_num_draft_tokens_uses_external_config_loader(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
        server_args.speculative_dspark_block_size = None
        server_args.speculative_num_draft_tokens = 8

        with patch(
            "sglang.srt.arg_groups.speculative_hook._load_speculative_draft_config",
            return_value=_redhat_glm52_dspark_config(),
        ) as load_config:
            _handle_dspark(server_args)

        load_config.assert_called_once()
        self.assertEqual(server_args.speculative_num_draft_tokens, 8)

    def test_explicit_num_draft_tokens_validates_gamma_from_speculators_config(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
        server_args.speculative_dspark_block_size = None
        server_args.speculative_num_draft_tokens = 7

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook._load_speculative_draft_config",
                return_value=_redhat_glm52_dspark_config(),
            ),
            self.assertRaisesRegex(ValueError, "gamma \\+ 1"),
        ):
            _handle_dspark(server_args)

    def test_dflash_rejects_dspark_draft_checkpoint(self):
        server_args = _make_dspark_server_args(
            model_path=_PLAIN_MODEL_PATH, hf_config=_plain_hf_config()
        )
        server_args.speculative_algorithm = "DFLASH"
        server_args.speculative_draft_model_path = "RedHatAI/GLM-5.2-speculator.dspark"
        server_args.speculative_dspark_block_size = None

        with (
            patch(
                "sglang.srt.arg_groups.speculative_hook._load_speculative_draft_config",
                return_value=_redhat_glm52_dspark_config(),
            ),
            self.assertRaisesRegex(ValueError, "Use --speculative-algorithm DSPARK"),
        ):
            _handle_dflash(server_args)


if __name__ == "__main__":
    unittest.main()
