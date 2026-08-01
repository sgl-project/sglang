import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DsparkDraftSampler,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _base_dspark_hf_config(**overrides) -> SimpleNamespace:
    fields = dict(
        architectures=["DeepseekV4ForCausalLM"],
        dspark_block_size=5,
        dspark_markov_rank=256,
        dspark_markov_head_type="vanilla",
        dspark_target_layer_ids=[40, 41, 42],
        dspark_noise_token_id=128799,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


class TestDsparkSpeculatorsConventionDetection(CustomTestCase):
    def test_deepspec_checkpoint_not_flagged(self):
        # No speculators_model_type field at all -- the normal DeepSpec case.
        config = parse_dspark_draft_config(draft_hf_config=_base_dspark_hf_config())
        self.assertFalse(config.speculators_convention)

    def test_other_speculators_model_type_not_flagged(self):
        # speculators_model_type present but not "dspark" -- a different
        # speculators-trained architecture, not the DSpark slot-shift case.
        config = parse_dspark_draft_config(
            draft_hf_config=_base_dspark_hf_config(speculators_model_type="eagle3")
        )
        self.assertFalse(config.speculators_convention)

    def test_speculators_dspark_checkpoint_flagged(self):
        config = parse_dspark_draft_config(
            draft_hf_config=_base_dspark_hf_config(speculators_model_type="dspark")
        )
        self.assertTrue(config.speculators_convention)

    def test_speculators_dspark_checkpoint_flagged_case_insensitive(self):
        # Checkpoint config values are author-controlled strings, not a
        # validated enum -- a future speculators release or a different
        # checkpoint author could write "DSpark"/"Dspark" instead of the
        # lowercase "dspark" seen in every checkpoint verified so far.
        for variant in ("DSpark", "DSPARK", "Dspark"):
            with self.subTest(variant=variant):
                config = parse_dspark_draft_config(
                    draft_hf_config=_base_dspark_hf_config(
                        speculators_model_type=variant
                    )
                )
                self.assertTrue(config.speculators_convention)

    def test_non_string_speculators_model_type_not_flagged(self):
        # Malformed config where the field is present but not a string (e.g.
        # accidentally set to a number or a dict) -- must not crash on
        # .lower() and must not be treated as a match.
        config = parse_dspark_draft_config(
            draft_hf_config=_base_dspark_hf_config(speculators_model_type=123)
        )
        self.assertFalse(config.speculators_convention)


def _speculators_hf_config(
    *, block_size: int, speculative_tokens: int, default_method: str = "greedy"
) -> SimpleNamespace:
    # Matches the real structure of RedHatAI/GLM-5.2-speculator.dspark's
    # config.json (verified directly against the checkpoint on the Hub):
    # block_size is the full anchor+gamma block width, while
    # speculators_config.proposal_methods[i].speculative_tokens is the
    # authoritative gamma (real draft token count).
    return SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        block_size=block_size,
        markov_rank=256,
        markov_head_type="vanilla",
        mask_token_id=128799,
        speculators_model_type="dspark",
        speculators_config={
            "algorithm": "dspark",
            "default_proposal_method": default_method,
            "proposal_methods": [
                {
                    "proposal_type": default_method,
                    "speculative_tokens": speculative_tokens,
                    "verifier_accept_k": 1,
                }
            ],
        },
    )


class TestSpeculatorsProposalGamma(CustomTestCase):
    def test_gamma_from_speculators_config_not_block_size(self):
        # Ground truth: RedHatAI/GLM-5.2-speculator.dspark has block_size=8
        # but speculative_tokens=7 -- gamma must be 7, not 8. Using
        # block_size directly here is exactly the bug this whole fix exists
        # to prevent (one draft slot too many, anchor read as a draft slot).
        config = parse_dspark_draft_config(
            draft_hf_config=_speculators_hf_config(block_size=8, speculative_tokens=7)
        )
        self.assertEqual(config.gamma, 7)
        self.assertTrue(config.speculators_convention)

    def test_gamma_falls_back_to_block_size_without_speculators_config(self):
        # DeepSpec-native checkpoints have no speculators_config at all --
        # gamma must still resolve from block_size as before.
        config = parse_dspark_draft_config(draft_hf_config=_base_dspark_hf_config())
        self.assertEqual(config.gamma, 5)  # dspark_block_size=5 in the fixture
        self.assertFalse(config.speculators_convention)

    def test_gamma_respects_default_proposal_method_selection(self):
        # Multiple proposal methods present; must pick the one named by
        # default_proposal_method, not just proposal_methods[0].
        cfg = _speculators_hf_config(
            block_size=17, speculative_tokens=16, default_method="probabilistic"
        )
        cfg.speculators_config["proposal_methods"] = [
            {"proposal_type": "greedy", "speculative_tokens": 99},
            {"proposal_type": "probabilistic", "speculative_tokens": 16},
        ]
        config = parse_dspark_draft_config(draft_hf_config=cfg)
        self.assertEqual(config.gamma, 16)


def _dummy_draft_model():
    return SimpleNamespace(markov_head=SimpleNamespace())


class TestDraftBlockWidth(CustomTestCase):
    """draft_width is the one piece of state that must flip between gamma
    (DeepSpec, anchor is itself a trained draft slot) and gamma + 1
    (speculators, anchor is a separate untrained conditioning token) -- see
    DraftBlockProposer's docstring. Every other consumer (verify window
    sizing, KV commit, accept-length accounting) keeps reading plain gamma
    unchanged; only the draft-forward-pass's own block construction differs.
    """

    def test_deepspec_convention_draft_width_equals_gamma(self):
        proposer = DraftBlockProposer(
            draft_model=None,
            draft_model_runner=None,
            gamma=7,
            mask_token_id=0,
            draft_block_spec_info=None,
            bonus_anchor=False,
        )
        self.assertEqual(proposer.draft_width, 7)

    def test_speculators_convention_draft_width_is_gamma_plus_one(self):
        proposer = DraftBlockProposer(
            draft_model=None,
            draft_model_runner=None,
            gamma=7,
            mask_token_id=0,
            draft_block_spec_info=None,
            bonus_anchor=True,
        )
        self.assertEqual(proposer.draft_width, 8)

    def test_draft_sampler_draft_width_matches_proposer(self):
        for bonus_anchor, expected_width in ((False, 7), (True, 8)):
            with self.subTest(bonus_anchor=bonus_anchor):
                sampler = DsparkDraftSampler(
                    model=_dummy_draft_model(),
                    gamma=7,
                    max_bs=4,
                    device="cpu",
                    bonus_anchor=bonus_anchor,
                )
                self.assertEqual(sampler.draft_width, expected_width)
                # The sampled-token output buffer stays gamma-wide regardless
                # -- only the input hidden_states/input_ids width changes.
                self.assertEqual(sampler.out.shape, (4 * 7,))


if __name__ == "__main__":
    unittest.main()
