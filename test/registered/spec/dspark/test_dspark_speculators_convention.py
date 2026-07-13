import unittest
from types import SimpleNamespace

from sglang.srt.speculative.dspark_components.dspark_config import (
    parse_dspark_draft_config,
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


if __name__ == "__main__":
    unittest.main()
