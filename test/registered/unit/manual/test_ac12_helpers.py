"""Registered helper-level regressions for the AC-12 harness at
``test/manual/test_double_sparsity_v32.py``.

The manual file itself skips without env vars (hardware-required), so
its prompt-generation, needle-naming, and recall-scoring helpers need
CI coverage to stay correct.
"""

from __future__ import annotations

import importlib.util
import pathlib
import unittest


class TestAC12HarnessHelpers(unittest.TestCase):
    """CPU regressions for prompt + scoring helpers."""

    @classmethod
    def setUpClass(cls):
        import sys
        path = pathlib.Path(__file__).resolve()
        for parent in path.parents:
            cand = parent / "test" / "manual" / "test_double_sparsity_v32.py"
            if cand.exists():
                spec = importlib.util.spec_from_file_location("_ac12", cand)
                mod = importlib.util.module_from_spec(spec)
                # Register before exec so @dataclass can resolve __module__.
                sys.modules["_ac12"] = mod
                spec.loader.exec_module(mod)
                cls._h = mod
                return
        raise RuntimeError("could not locate test/manual/test_double_sparsity_v32.py")

    # ---- needle naming -----------------------------------------------

    def test_needle_is_stable_per_length_and_index(self):
        self.assertEqual(
            self._h._niah_needle(4096, 0),
            self._h._niah_needle(4096, 0),
        )
        self.assertNotEqual(
            self._h._niah_needle(4096, 0),
            self._h._niah_needle(4096, 1),
        )
        self.assertNotEqual(
            self._h._niah_needle(4096, 0),
            self._h._niah_needle(16384, 0),
        )

    def test_needle_format(self):
        n = self._h._niah_needle(4096, 7)
        self.assertTrue(n.startswith("NEEDLE-"))
        # Two zero-padded segments after the prefix.
        self.assertEqual(len(n.split("-")), 3)

    # ---- prompt generation ------------------------------------------

    def test_prompt_is_deterministic(self):
        p1 = self._h._make_niah_prompt(2048, seed=42, needle="NEEDLE-X")
        p2 = self._h._make_niah_prompt(2048, seed=42, needle="NEEDLE-X")
        self.assertEqual(p1, p2)

    def test_prompt_contains_needle_exactly_once(self):
        needle = "NEEDLE-UNIQUE-12345"
        p = self._h._make_niah_prompt(2048, seed=7, needle=needle)
        self.assertEqual(p.count(needle), 1)

    def test_prompt_length_approximately_matches_request(self):
        for L in (512, 4096, 8192):
            p = self._h._make_niah_prompt(L, seed=1, needle="X")
            n_words = len(p.split())
            # Within ±5% of the requested length.
            self.assertLessEqual(abs(n_words - L), max(32, L // 20),
                                 f"L={L}: got {n_words} words")

    def test_prompt_needle_placed_mid_prompt(self):
        needle = "NEEDLE-MID-99999"
        p = self._h._make_niah_prompt(4096, seed=11, needle=needle)
        words = p.split()
        idx = next(i for i, w in enumerate(words) if needle in w)
        depth_frac = idx / len(words)
        # Plan: 35-65% depth. Allow a tiny tolerance for word-boundary rounding.
        self.assertGreaterEqual(depth_frac, 0.30)
        self.assertLessEqual(depth_frac, 0.70)

    def test_prompt_ends_with_question_suffix(self):
        p = self._h._make_niah_prompt(2048, seed=3, needle="X")
        self.assertTrue(p.endswith("Output only the value."))
        self.assertIn("Question:", p)

    # ---- recall scoring ---------------------------------------------

    def test_recall_all_present(self):
        needles = ["N1", "N2", "N3"]
        responses = ["...N1...", "the N2 is here", "yes N3 yes"]
        self.assertEqual(self._h._niah_recall_hits(needles, responses), 3)

    def test_recall_none_present(self):
        needles = ["N1", "N2", "N3"]
        responses = ["x", "y", "z"]
        self.assertEqual(self._h._niah_recall_hits(needles, responses), 0)

    def test_recall_partial(self):
        needles = ["N1", "N2", "N3", "N4"]
        responses = ["N1 here", "no", "N3 in middle", "N4 N4 N4"]
        self.assertEqual(self._h._niah_recall_hits(needles, responses), 3)

    def test_recall_substring_not_word_boundary(self):
        """The plan uses substring containment, not word-boundary match —
        so a needle embedded in a longer token still counts. This is
        intentional: temperature=0 models may concatenate or stylize."""
        needles = ["KEY-42"]
        responses = ["the answer is theKEY-42was hidden"]
        self.assertEqual(self._h._niah_recall_hits(needles, responses), 1)

    # ---- OpenAI-compatible URL normalization (Round 26) ----------------

    def test_openai_base_url_appends_v1(self):
        self.assertEqual(
            self._h._openai_base_url("http://h:30000"),
            "http://h:30000/v1",
        )

    def test_openai_base_url_strips_trailing_slash(self):
        self.assertEqual(
            self._h._openai_base_url("http://h:30000/"),
            "http://h:30000/v1",
        )

    def test_openai_base_url_idempotent_on_v1(self):
        self.assertEqual(
            self._h._openai_base_url("http://h:30000/v1"),
            "http://h:30000/v1",
        )

    def test_openai_base_url_idempotent_on_v1_with_trailing_slash(self):
        self.assertEqual(
            self._h._openai_base_url("http://h:30000/v1/"),
            "http://h:30000/v1",
        )

    def test_openai_base_url_case_insensitive_suffix(self):
        # The suffix check is case-insensitive on /V1 because operators
        # paste URLs in mixed case; the canonical form is /v1.
        self.assertEqual(
            self._h._openai_base_url("HTTPS://example.com/V1"),
            "HTTPS://example.com/V1",
        )

    # ---- MMLU 5-shot prompt construction (Round 26) -------------------

    def _mmlu_row(self, q: str, gold: str) -> list:
        return [q, "alpha", "beta", "gamma", "delta", gold]

    def test_mmlu_format_example_with_answer(self):
        row = self._mmlu_row("What is 2+2?", "A")
        out = self._h._format_mmlu_example(row, include_answer=True)
        self.assertIn("What is 2+2?", out)
        for letter in ("A", "B", "C", "D"):
            self.assertIn(f"\n{letter}. ", out)
        self.assertTrue(out.endswith("Answer: A\n\n"))

    def test_mmlu_format_example_test_question(self):
        row = self._mmlu_row("What is 3+3?", "B")
        out = self._h._format_mmlu_example(row, include_answer=False)
        self.assertTrue(out.endswith("Answer:"))
        self.assertNotIn("Answer: B", out)

    def test_mmlu_5shot_prompt_contains_five_answered_examples(self):
        dev = [self._mmlu_row(f"Q{i}?", "ABCD"[i % 4]) for i in range(7)]
        test = self._mmlu_row("Final?", "A")
        prompt = self._h._make_mmlu_5shot_prompt(dev, "abstract_algebra", test)
        # Exactly 5 in-context "Answer: X" lines from the dev set.
        in_context_answers = sum(
            1 for letter in "ABCD" for _ in range(prompt.count(f"Answer: {letter}\n\n"))
        )
        self.assertEqual(in_context_answers, 5)
        # The TEST question must end with bare "Answer:" (no letter).
        self.assertTrue(prompt.endswith("Answer:"))
        # Header advertises the subject.
        self.assertIn("about abstract algebra", prompt)

    def test_mmlu_5shot_requires_at_least_5_dev_rows(self):
        dev = [self._mmlu_row("Q", "A") for _ in range(3)]
        test = self._mmlu_row("Final?", "A")
        with self.assertRaises(ValueError):
            self._h._make_mmlu_5shot_prompt(dev, "x", test)

    def test_mmlu_parse_letter(self):
        self.assertEqual(self._h._parse_mmlu_letter("A"), "A")
        self.assertEqual(self._h._parse_mmlu_letter(" The answer is B."), "B")
        self.assertEqual(self._h._parse_mmlu_letter("D — definitely"), "D")
        self.assertIsNone(self._h._parse_mmlu_letter("hmm"))
        # First A-D character wins (so "AB" → A).
        self.assertEqual(self._h._parse_mmlu_letter("AB"), "A")


if __name__ == "__main__":
    unittest.main()
