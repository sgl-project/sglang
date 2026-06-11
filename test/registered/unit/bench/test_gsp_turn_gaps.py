"""Unit tests for generated-shared-prefix multi-turn gap sampling."""

import random
import unittest
from argparse import Namespace

import numpy as np

from sglang.benchmark.datasets.generated_shared_prefix import (
    GeneratedSharedPrefixDataset,
    sample_generated_shared_prefix_requests,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _FakeTokenizer:
    """Whitespace tokenizer over a small word vocab; no downloads."""

    _vocab = {f"w{i}": i for i in range(512)}
    _id_to_word = {i: w for w, i in _vocab.items()}

    def get_vocab(self):
        return self._vocab

    def decode(self, token_ids):
        return " ".join(self._id_to_word[t] for t in token_ids)

    def encode(self, text):
        return [self._vocab.get(w, 0) for w in text.split()]


def _sample(seed=42, **overrides):
    random.seed(seed)
    np.random.seed(seed)
    kwargs = dict(
        num_groups=6,
        prompts_per_group=1,
        system_prompt_len=32,
        question_len=8,
        output_len=4,
        range_ratio=1.0,
        tokenizer=_FakeTokenizer(),
        seed=seed,
        num_turns=4,
        ordered=True,
    )
    kwargs.update(overrides)
    return sample_generated_shared_prefix_requests(**kwargs)


def _gap_args(**overrides):
    fields = dict(
        gsp_num_groups=2,
        gsp_prompts_per_group=1,
        gsp_system_prompt_len=32,
        gsp_question_len=8,
        gsp_output_len=4,
        gsp_range_ratio=1.0,
        gsp_fast_prepare=False,
        gsp_send_routing_key=False,
        gsp_num_turns=4,
        gsp_ordered=True,
        gsp_group_distribution="uniform",
        gsp_zipf_alpha=None,
        gsp_turn_gap_short_s=0.0,
        gsp_turn_gap_long_s=0.0,
        gsp_turn_gap_long_prob=0.0,
        seed=42,
        tokenize_prompt=False,
    )
    fields.update(overrides)
    return Namespace(**fields)


class TestTurnGapSampling(CustomTestCase):
    def test_gaps_disabled_by_default(self):
        rows = _sample()
        self.assertTrue(all(r.turn_gaps_s is None for r in rows))

    def test_gap_shape_and_values(self):
        rows = _sample(
            turn_gap_short_s=0.5, turn_gap_long_s=15.0, turn_gap_long_prob=0.5
        )
        all_gaps = []
        for r in rows:
            self.assertEqual(len(r.turn_gaps_s), 4)  # one slot per turn
            self.assertEqual(r.turn_gaps_s[0], 0.0)  # no pause before turn 0
            self.assertTrue(all(g in (0.5, 15.0) for g in r.turn_gaps_s[1:]))
            all_gaps.extend(r.turn_gaps_s[1:])
        # With prob=0.5 over 18 draws, both gap kinds should appear.
        self.assertIn(0.5, all_gaps)
        self.assertIn(15.0, all_gaps)

    def test_gap_sampling_is_deterministic(self):
        kwargs = dict(
            turn_gap_short_s=0.5, turn_gap_long_s=15.0, turn_gap_long_prob=0.5
        )
        first = [r.turn_gaps_s for r in _sample(**kwargs)]
        second = [r.turn_gaps_s for r in _sample(**kwargs)]
        self.assertEqual(first, second)

    def test_gap_extremes(self):
        all_short = _sample(
            turn_gap_short_s=0.5, turn_gap_long_s=15.0, turn_gap_long_prob=0.0
        )
        for r in all_short:
            self.assertEqual(r.turn_gaps_s[1:], [0.5] * 3)
        all_long = _sample(
            turn_gap_short_s=0.5, turn_gap_long_s=15.0, turn_gap_long_prob=1.0
        )
        for r in all_long:
            self.assertEqual(r.turn_gaps_s[1:], [15.0] * 3)

    def test_enabling_gaps_does_not_perturb_prompts(self):
        without = _sample()
        with_gaps = _sample(
            turn_gap_short_s=0.5, turn_gap_long_s=15.0, turn_gap_long_prob=0.5
        )
        self.assertEqual([r.prompt for r in without], [r.prompt for r in with_gaps])

    def test_single_turn_never_gets_gaps(self):
        rows = _sample(
            num_turns=1,
            turn_gap_short_s=0.5,
            turn_gap_long_s=15.0,
            turn_gap_long_prob=0.5,
            # num_turns=1 with default lengths would hit the on-disk dataset
            # cache; perturb range_ratio to bypass it.
            range_ratio=0.9,
        )
        self.assertTrue(all(r.turn_gaps_s is None for r in rows))


class TestTurnGapValidation(CustomTestCase):
    def test_negative_gap_rejected(self):
        with self.assertRaises(ValueError):
            GeneratedSharedPrefixDataset.from_args(_gap_args(gsp_turn_gap_short_s=-1.0))
        with self.assertRaises(ValueError):
            GeneratedSharedPrefixDataset.from_args(_gap_args(gsp_turn_gap_long_s=-1.0))

    def test_out_of_range_prob_rejected(self):
        with self.assertRaises(ValueError):
            GeneratedSharedPrefixDataset.from_args(
                _gap_args(gsp_turn_gap_long_prob=1.5)
            )
        with self.assertRaises(ValueError):
            GeneratedSharedPrefixDataset.from_args(
                _gap_args(gsp_turn_gap_long_prob=-0.1)
            )

    def test_gap_fields_flow_from_args(self):
        ds = GeneratedSharedPrefixDataset.from_args(
            _gap_args(
                gsp_turn_gap_short_s=0.5,
                gsp_turn_gap_long_s=15.0,
                gsp_turn_gap_long_prob=0.15,
            )
        )
        self.assertEqual(
            (ds.turn_gap_short_s, ds.turn_gap_long_s, ds.turn_gap_long_prob),
            (0.5, 15.0, 0.15),
        )


if __name__ == "__main__":
    unittest.main()
