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

    # ---- Round 28: _load_mmlu_examples validates CSV usability --------

    def _write_csv(self, path: str, rows: List[List[str]]) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(",".join(row) + "\n")

    def _make_subject(
        self, dev_dir: str, test_dir: str, subject: str,
        *, dev_rows: int = 5, test_rows: int = 1, test_cols: int = 6,
    ) -> None:
        os.makedirs(dev_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        dev_data = [
            [f"q{i}", "alpha", "beta", "gamma", "delta", "ABCD"[i % 4]]
            for i in range(dev_rows)
        ]
        self._write_csv(
            os.path.join(dev_dir, f"{subject}_dev.csv"), dev_data,
        )
        # Test CSV: optionally truncate columns so each row has `test_cols`.
        test_data = []
        for i in range(test_rows):
            full = [f"tq{i}", "A1", "B1", "C1", "D1", "A"]
            test_data.append(full[:test_cols])
        self._write_csv(
            os.path.join(test_dir, f"{subject}_test.csv"), test_data,
        )

    def test_load_mmlu_examples_happy_path(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            self._make_subject(dev_dir, test_dir, "abstract_algebra",
                               dev_rows=5, test_rows=3)
            examples, totals = self._h._load_mmlu_examples(
                dev_dir, test_dir, max_examples=100,
            )
            self.assertEqual(len(examples), 3)
            self.assertEqual(totals["abstract_algebra"], 3)
            ex = examples[0]
            self.assertIn("subject", ex)
            self.assertIn("dev", ex)
            self.assertIn("row", ex)
            self.assertEqual(len(ex["dev"]), 5)

    def test_load_mmlu_examples_raises_on_empty_test_dir(self):
        """Codex Round 27 review bug: empty dev/test dirs used to skip."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            os.makedirs(dev_dir)
            os.makedirs(test_dir)
            with self.assertRaises(ValueError) as ctx:
                self._h._load_mmlu_examples(dev_dir, test_dir)
            self.assertIn("no subjects found", str(ctx.exception))

    def test_load_mmlu_examples_raises_on_too_few_dev_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            # 3 dev rows, expected ≥ 5 → rejected.
            self._make_subject(dev_dir, test_dir, "tiny",
                               dev_rows=3, test_rows=2)
            with self.assertRaises(ValueError) as ctx:
                self._h._load_mmlu_examples(dev_dir, test_dir)
            self.assertIn("dev rows, need 5", str(ctx.exception))

    def test_load_mmlu_examples_raises_on_malformed_test_rows(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            # 5 dev rows OK, but test rows only have 4 columns → no ≥6 row.
            self._make_subject(dev_dir, test_dir, "broken",
                               dev_rows=5, test_rows=2, test_cols=4)
            with self.assertRaises(ValueError) as ctx:
                self._h._load_mmlu_examples(dev_dir, test_dir)
            self.assertIn("≥6 columns", str(ctx.exception))

    def test_load_mmlu_examples_raises_on_missing_dev_csv(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            os.makedirs(dev_dir)
            os.makedirs(test_dir)
            # Only test CSV exists; dev CSV missing.
            self._write_csv(
                os.path.join(test_dir, "x_test.csv"),
                [["q", "A", "B", "C", "D", "A"]],
            )
            with self.assertRaises(ValueError) as ctx:
                self._h._load_mmlu_examples(dev_dir, test_dir)
            self.assertIn("missing dev or test CSV", str(ctx.exception))

    def test_load_mmlu_examples_deterministic_seed_and_cap(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            self._make_subject(dev_dir, test_dir, "z_subject",
                               dev_rows=5, test_rows=10)
            # Two calls with same seed → same example order.
            ex1, _ = self._h._load_mmlu_examples(
                dev_dir, test_dir, max_examples=10, seed=42,
            )
            ex2, _ = self._h._load_mmlu_examples(
                dev_dir, test_dir, max_examples=10, seed=42,
            )
            self.assertEqual(
                [e["row"][0] for e in ex1],
                [e["row"][0] for e in ex2],
            )
            # max_examples cap is honored.
            ex_capped, _ = self._h._load_mmlu_examples(
                dev_dir, test_dir, max_examples=3, seed=42,
            )
            self.assertEqual(len(ex_capped), 3)

    def test_load_mmlu_examples_works_without_pandas(self):
        """Round 28 review bug: the harness had `try: import pandas /
        self.skipTest()`. Round 29 dropped pandas — verify the loader
        runs under monkeypatched pandas import failure."""
        import builtins
        import sys
        import tempfile

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "pandas" or name.startswith("pandas."):
                raise ImportError(f"simulated absence of {name}")
            return real_import(name, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            self._make_subject(dev_dir, test_dir, "csv_only",
                               dev_rows=5, test_rows=2)

            # Block pandas — both as already-imported and as a fresh import.
            pandas_was = sys.modules.pop("pandas", None)
            try:
                builtins.__import__ = fake_import
                try:
                    examples, totals = self._h._load_mmlu_examples(
                        dev_dir, test_dir, max_examples=10,
                    )
                finally:
                    builtins.__import__ = real_import
            finally:
                if pandas_was is not None:
                    sys.modules["pandas"] = pandas_was

            self.assertEqual(len(examples), 2)
            self.assertIn("csv_only", totals)

    def test_load_mmlu_examples_explicit_subjects_filter(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            self._make_subject(dev_dir, test_dir, "alpha",
                               dev_rows=5, test_rows=2)
            self._make_subject(dev_dir, test_dir, "beta",
                               dev_rows=5, test_rows=4)
            examples, totals = self._h._load_mmlu_examples(
                dev_dir, test_dir, subjects=["beta"],
            )
            self.assertEqual(len(examples), 4)
            self.assertIn("beta", totals)
            self.assertNotIn("alpha", totals)

    # ---- Round 29: harness-level subjects-filter integration ----------

    def test_mmlu_5shot_subjects_filter_does_not_crash(self):
        """Round 28 review bug: `AC12_MMLU_SUBJECTS=beta` raised
        `NameError: name 'subjects' is not defined` because the
        artifact recorder referenced the pre-rename variable name.

        Drives the full `test_mmlu_5shot` harness path end-to-end:
        builds a tiny valid CSV tree with a `beta` subject, sets the
        env override, mocks ``_generate`` to return the gold answer
        (so the gate passes), captures the ``_record_artifact``
        payload, asserts no NameError and that the recorded
        ``subjects`` field is ``["beta"]``.
        """
        import sys
        import tempfile
        import unittest as _ut
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmp:
            dev_dir = os.path.join(tmp, "dev")
            test_dir = os.path.join(tmp, "test")
            # Build alpha + beta subjects; only beta is selected via env.
            self._make_subject(dev_dir, test_dir, "alpha",
                               dev_rows=5, test_rows=2)
            self._make_subject(dev_dir, test_dir, "beta",
                               dev_rows=5, test_rows=3)

            # Mock _generate to return the gold answer for every prompt
            # so the gate assertion passes. The gold is in row[5];
            # _make_subject populates it as "A".
            def fake_generate(url, prompt, *, max_new_tokens=4):
                return "A"

            captured: List[Dict[str, Any]] = []

            def fake_record(payload, *, suffix):
                captured.append({"payload": payload, "suffix": suffix})

            env_overrides = {
                "DS_BASE_URL": "http://127.0.0.1:1",
                "DSA_BASE_URL": "http://127.0.0.1:2",
                "AC12_MMLU_DATA_DIR": tmp,
                "AC12_MMLU_SUBJECTS": "beta",
                "AC12_MMLU_NUM_EXAMPLES": "10",
            }

            # The @unittest.skipUnless decorator on the class freezes
            # the env check at import time. Temporarily clear the
            # cached skip so the test body actually runs under our
            # patched env. We restore the originals in `finally` so
            # this regression doesn't pollute other registered tests.
            cls = self._h.TestDoubleSparsityV32Quality
            orig_skip = cls.__dict__.get("__unittest_skip__", False)
            orig_why = cls.__dict__.get("__unittest_skip_why__", "")
            cls.__unittest_skip__ = False
            cls.__unittest_skip_why__ = ""
            try:
                with patch.dict(os.environ, env_overrides), \
                     patch.object(self._h, "_generate", fake_generate), \
                     patch.object(self._h, "_record_artifact", fake_record):
                    suite = _ut.TestSuite([cls("test_mmlu_5shot")])
                    runner = _ut.TextTestRunner(
                        stream=open(os.devnull, "w"), verbosity=0,
                    )
                    result = runner.run(suite)
            finally:
                cls.__unittest_skip__ = orig_skip
                cls.__unittest_skip_why__ = orig_why

            # No errors and no failures — the AC-12 gate path completed
            # without NameError + the model's mock gold answers passed
            # the |DSA - DS| <= 1.0 pp check (both 100%, delta 0).
            self.assertEqual(
                len(result.errors), 0,
                f"unexpected errors: {result.errors}",
            )
            self.assertEqual(
                len(result.failures), 0,
                f"unexpected failures: {result.failures}",
            )

            # Recorder captured the MMLU artifact with subjects == ["beta"].
            mmlu_payloads = [
                c for c in captured if c["suffix"] == "mmlu_5shot"
            ]
            self.assertEqual(len(mmlu_payloads), 1)
            payload = mmlu_payloads[0]["payload"]
            self.assertEqual(payload["subjects"], ["beta"])
            # Only beta was evaluated; alpha must not appear.
            self.assertIn("beta", payload["dsa_per_subject"])
            self.assertNotIn("alpha", payload["dsa_per_subject"])

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
