"""Registered helper-level regressions for the AC-12 harness at
``test/manual/test_double_sparsity_v32.py``.

The manual file itself skips without env vars (hardware-required), so
its prompt-generation, needle-naming, and recall-scoring helpers need
CI coverage to stay correct.
"""

from __future__ import annotations

import importlib.util
import os
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

    # ---- Round 27: robust answer-token parser -------------------------

    def test_parse_mmlu_answer_prefix_returns_real_letter(self):
        """Codex Round 26 review bug: ``Answer: B`` returned ``A``
        (the A in "Answer"). The Round 27 parser must return ``B``."""
        self.assertEqual(self._h._parse_mmlu_letter("Answer: B"), "B")

    def test_parse_mmlu_lowercase_letter(self):
        self.assertEqual(self._h._parse_mmlu_letter("b"), "B")

    def test_parse_mmlu_paren_wrapped(self):
        self.assertEqual(self._h._parse_mmlu_letter("(C)"), "C")

    def test_parse_mmlu_letter_with_period(self):
        self.assertEqual(self._h._parse_mmlu_letter("D."), "D")

    def test_parse_mmlu_bracket_wrapped(self):
        self.assertEqual(self._h._parse_mmlu_letter("[A]"), "A")

    def test_parse_mmlu_answer_is_marker(self):
        self.assertEqual(self._h._parse_mmlu_letter("answer is C"), "C")

    def test_parse_mmlu_the_answer_is_marker_with_punct(self):
        self.assertEqual(self._h._parse_mmlu_letter("The answer is D."), "D")

    def test_parse_mmlu_option_marker(self):
        self.assertEqual(self._h._parse_mmlu_letter("option B"), "B")

    def test_parse_mmlu_choice_marker_paren(self):
        self.assertEqual(self._h._parse_mmlu_letter("Choice (A)"), "A")

    def test_parse_mmlu_narrative_decoy_no_marker_returns_none(self):
        """Narrative text with no answer marker: do NOT guess. The
        Round 26 first-A-D-char parser would have returned ``A`` from
        the word ``Awful`` (or any other A-D-prefixed word)."""
        self.assertIsNone(
            self._h._parse_mmlu_letter("Awful question, no marker."),
        )

    def test_parse_mmlu_empty_string(self):
        self.assertIsNone(self._h._parse_mmlu_letter(""))

    def test_parse_mmlu_whitespace_only(self):
        self.assertIsNone(self._h._parse_mmlu_letter("   \n  "))

    def test_parse_mmlu_leading_punctuation_then_letter(self):
        # Whitespace + open-paren + letter + comma + ...
        self.assertEqual(self._h._parse_mmlu_letter(" (B), final"), "B")

    # ---- Round 27: MMLU data self-prep helper -------------------------

    def _make_fake_mmlu_tar(self, dest_tar: str) -> None:
        """Build a tiny .tar with the expected ``data/{dev,test}`` layout."""
        import tarfile
        import tempfile

        with tempfile.TemporaryDirectory() as src_root:
            for sub in ("dev", "test"):
                os.makedirs(os.path.join(src_root, "data", sub), exist_ok=True)
                with open(
                    os.path.join(src_root, "data", sub, f"foo_{sub}.csv"),
                    "w",
                ) as fh:
                    fh.write("q,a,b,c,d,gold\n")
            with tarfile.open(dest_tar, "w") as tar:
                tar.add(os.path.join(src_root, "data"), arcname="data")

    def test_ensure_mmlu_data_dir_downloads_and_extracts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "mmlu")
            fake_tar = os.path.join(tmp, "fake.tar")
            self._make_fake_mmlu_tar(fake_tar)

            def fake_urlretrieve(url, dst):
                import shutil as _sh
                _sh.copyfile(fake_tar, dst)
                return dst, None

            from unittest.mock import patch
            with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                dev, test = self._h._ensure_mmlu_data_dir(data_dir)
            self.assertEqual(dev, os.path.join(data_dir, "dev"))
            self.assertEqual(test, os.path.join(data_dir, "test"))
            self.assertTrue(os.path.isfile(os.path.join(dev, "foo_dev.csv")))
            self.assertTrue(os.path.isfile(os.path.join(test, "foo_test.csv")))

    def test_ensure_mmlu_data_dir_idempotent(self):
        """Second call must NOT re-download when dev/ + test/ exist."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "mmlu")
            fake_tar = os.path.join(tmp, "fake.tar")
            self._make_fake_mmlu_tar(fake_tar)

            calls = []

            def fake_urlretrieve(url, dst):
                calls.append(url)
                import shutil as _sh
                _sh.copyfile(fake_tar, dst)
                return dst, None

            from unittest.mock import patch
            with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                self._h._ensure_mmlu_data_dir(data_dir)
                self._h._ensure_mmlu_data_dir(data_dir)  # second call

            # urlretrieve called exactly once (first call); second was a no-op.
            self.assertEqual(len(calls), 1)

    def test_ensure_mmlu_data_dir_raises_on_download_failure(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "mmlu")

            def boom(url, dst):
                raise IOError("network unreachable")

            from unittest.mock import patch
            with patch("urllib.request.urlretrieve", side_effect=boom):
                with self.assertRaises(RuntimeError) as ctx:
                    self._h._ensure_mmlu_data_dir(data_dir)
            self.assertIn("download failed", str(ctx.exception))

    def test_ensure_mmlu_data_dir_raises_when_archive_missing_subdirs(self):
        """Tar with no data/dev or data/test must fail loudly."""
        import tarfile
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "mmlu")
            fake_tar = os.path.join(tmp, "bad.tar")
            # Build a tar that has data/ but no dev/test subdirs.
            with tempfile.TemporaryDirectory() as src:
                os.makedirs(os.path.join(src, "data"), exist_ok=True)
                with open(os.path.join(src, "data", "stray.txt"), "w") as fh:
                    fh.write("nope")
                with tarfile.open(fake_tar, "w") as tar:
                    tar.add(os.path.join(src, "data"), arcname="data")

            def fake_urlretrieve(url, dst):
                import shutil as _sh
                _sh.copyfile(fake_tar, dst)
                return dst, None

            from unittest.mock import patch
            with patch("urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                with self.assertRaises(RuntimeError) as ctx:
                    self._h._ensure_mmlu_data_dir(data_dir)
            self.assertIn("missing data/", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
