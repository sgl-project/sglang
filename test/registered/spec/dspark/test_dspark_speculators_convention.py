import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
)
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import resolve_num_tokens_per_req
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
    *,
    block_size: int,
    speculative_tokens: int,
    default_method: str = "greedy",
    **overrides,
) -> SimpleNamespace:
    # Matches the real structure of RedHatAI/GLM-5.2-speculator.dspark's
    # config.json (verified directly against the checkpoint on the Hub):
    # block_size is the full anchor+gamma block width, while
    # speculators_config.proposal_methods[i].speculative_tokens is the
    # authoritative gamma (real draft token count).
    fields = dict(
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
    fields.update(overrides)
    return SimpleNamespace(**fields)


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
                    folded_sampling=False,
                )
                self.assertEqual(sampler.draft_width, expected_width)
                # The sampled-token output buffer stays gamma-wide regardless
                # -- only the input hidden_states/input_ids width changes.
                self.assertEqual(sampler.out.shape, (4 * 7,))


class TestDraftCaptureWidth(CustomTestCase):
    def _spec_config(self, *, bonus_anchor: bool):
        return SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=8,
            speculative_dspark_bonus_anchor=bonus_anchor,
        )

    def test_dspark_draft_width_has_one_derivation_point(self):
        for bonus_anchor, expected_width in ((False, 7), (True, 8)):
            with self.subTest(bonus_anchor=bonus_anchor):
                with patch(
                    "sglang.srt.speculative.spec_utils.get_spec",
                    return_value=self._spec_config(bonus_anchor=bonus_anchor),
                ):
                    width = resolve_num_tokens_per_req(
                        phase="target_verify",
                        spec_algorithm=SpeculativeAlgorithm.DSPARK,
                        is_draft_worker=True,
                    )
                self.assertEqual(width, expected_width)

    def test_target_worker_keeps_full_verify_window(self):
        with patch(
            "sglang.srt.speculative.spec_utils.get_spec",
            return_value=self._spec_config(bonus_anchor=True),
        ):
            width = resolve_num_tokens_per_req(
                phase="target_verify",
                spec_algorithm=SpeculativeAlgorithm.DSPARK,
                is_draft_worker=False,
            )
        self.assertEqual(width, 8)

    def test_dspark_like_plugin_keeps_its_width_override(self):
        def custom_width(num_tokens, _):
            return num_tokens + 2

        plugin_algorithm = SimpleNamespace(
            is_dspark=lambda: True,
            get_num_tokens_per_req_for_target_verify=custom_width,
        )
        with patch(
            "sglang.srt.speculative.spec_utils.get_spec",
            return_value=self._spec_config(bonus_anchor=False),
        ):
            width = resolve_num_tokens_per_req(
                phase="target_verify",
                spec_algorithm=plugin_algorithm,
                is_draft_worker=True,
            )
        self.assertEqual(width, 10)

    def test_missing_declared_layout_field_fails_fast(self):
        spec_config = SimpleNamespace(
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=8,
        )
        with patch(
            "sglang.srt.speculative.spec_utils.get_spec", return_value=spec_config
        ):
            with self.assertRaises(AttributeError):
                resolve_num_tokens_per_req(
                    phase="target_verify",
                    spec_algorithm=SpeculativeAlgorithm.DSPARK,
                    is_draft_worker=True,
                )


class TestSampleFromAnchorIsAuthoritative(CustomTestCase):
    """`speculators_model_type` alone does not determine the block layout.

    Real speculators checkpoints ship both conventions: the trainer records the
    choice in `sample_from_anchor`, and the block geometry
    (block_size vs speculative_tokens) encodes the same fact independently.
    """

    def test_dense_speculators_checkpoint_not_flagged(self):
        # Ground truth: /data/suzhan/models/qwen3_6_35b_a3b_dspark_test --
        # speculators-trained but sample_from_anchor=True and
        # block_size == speculative_tokens == 16, i.e. the dense DeepSpec
        # layout. Flagging it would read a 17th slot that does not exist.
        config = parse_dspark_draft_config(
            draft_hf_config=_speculators_hf_config(
                block_size=16, speculative_tokens=16, sample_from_anchor=True
            )
        )
        self.assertFalse(config.speculators_convention)
        self.assertEqual(config.gamma, 16)

    def test_bonus_anchor_speculators_checkpoint_flagged(self):
        # The 1+N case: sample_from_anchor=False and block_size == tokens + 1.
        config = parse_dspark_draft_config(
            draft_hf_config=_speculators_hf_config(
                block_size=16, speculative_tokens=15, sample_from_anchor=False
            )
        )
        self.assertTrue(config.speculators_convention)
        self.assertEqual(config.gamma, 15)

    def test_sample_from_anchor_overrides_model_type_heuristic(self):
        # Both configs below are speculators_model_type="dspark"; only
        # sample_from_anchor separates them.
        dense = parse_dspark_draft_config(
            draft_hf_config=_speculators_hf_config(
                block_size=8, speculative_tokens=8, sample_from_anchor=True
            )
        )
        bonus = parse_dspark_draft_config(
            draft_hf_config=_speculators_hf_config(
                block_size=8, speculative_tokens=7, sample_from_anchor=False
            )
        )
        self.assertNotEqual(
            dense.speculators_convention, bonus.speculators_convention
        )

    def test_geometry_disagreeing_with_flag_fails_fast(self):
        # sample_from_anchor says dense, geometry says 1+N. Silently picking
        # either one misreads every draft slot, so this must raise.
        with self.assertRaises(ValueError):
            parse_dspark_draft_config(
                draft_hf_config=_speculators_hf_config(
                    block_size=16, speculative_tokens=15, sample_from_anchor=True
                )
            )


if __name__ == "__main__":
    unittest.main()
