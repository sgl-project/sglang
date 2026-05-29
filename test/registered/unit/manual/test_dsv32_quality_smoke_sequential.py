"""Registered CPU regressions for the AC-Q sequential quality smoke.

The shared library lives next to the manual hardware fixture
(``test/manual/_dsv32_quality_smoke_lib.py``) so the exact ``compute_gates``
math and the ``capture`` -> ``compare`` round-trip the single-node sequential
workflow uses are exercised here under CPU-only CI, with no live servers.

Guards:

* ``compute_gates`` returns all-pass when DS reproduces DSA, and fails the
  correct individual gate for each degradation (prefix, ROUGE-L, NIAH recall,
  first-8 divergence).
* ``evaluate_against_references`` round-trips a capture artifact: with
  ``generate`` monkeypatched to echo the DSA reference, the compare path
  reports all-pass; with garbage outputs it reports failure. This proves the
  capture-artifact -> compare wiring without needing two TP=8 servers.
* the reference-artifact schema validation rejects a malformed artifact.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
LIB_PATH = REPO_ROOT / "test" / "manual" / "_dsv32_quality_smoke_lib.py"


def _load_lib():
    spec = importlib.util.spec_from_file_location(
        "_dsv32_quality_smoke_lib", str(LIB_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dsv32_quality_smoke_lib"] = mod
    spec.loader.exec_module(mod)
    return mod


_lib = _load_lib()


def _identical_smoke(n=20):
    """n smoke triples where DS exactly reproduces DSA -> every gate passes."""
    pairs = []
    for i in range(n):
        txt = f"The deterministic answer number {i} is forty two and stable."
        pairs.append((f"prompt {i}", txt, txt))
    return pairs


def _hit_niah(n=5):
    needles = ["ZEBRA-7", "MARLIN-42", "ORCHID-99", "GLACIER-13", "PHARAOH-88"][:n]
    return [(needle, f"the answer is {needle}") for needle in needles]


class TestComputeGates(unittest.TestCase):
    def test_all_pass_when_ds_reproduces_dsa(self):
        res = _lib.compute_gates(_identical_smoke(), _hit_niah())
        self.assertTrue(res["all_pass"])
        for name, g in res["gates"].items():
            self.assertTrue(g["pass"], f"{name} should pass: {g}")

    def test_prefix_mismatch_fails_only_prefix_and_related(self):
        # Half the prompts get a totally different first 32 chars on DS, but we
        # keep enough token overlap that ROUGE/first-8 are not the focus here.
        pairs = _identical_smoke()
        for i in range(0, len(pairs), 2):  # 10/20 differ in the prefix
            p, dsa, _ = pairs[i]
            pairs[i] = (p, dsa, "Z" * 40 + dsa)
        res = _lib.compute_gates(pairs, _hit_niah())
        self.assertFalse(res["gates"]["prefix_match_rate"]["pass"])
        self.assertLess(res["gates"]["prefix_match_rate"]["value"], 0.80)
        self.assertFalse(res["all_pass"])

    def test_low_rouge_fails(self):
        pairs = [
            (f"prompt {i}", "alpha beta gamma delta epsilon", "totally unrelated wording here xyz")
            for i in range(20)
        ]
        res = _lib.compute_gates(pairs, _hit_niah())
        self.assertFalse(res["gates"]["mean_rouge_l"]["pass"])
        self.assertLess(res["gates"]["mean_rouge_l"]["value"], 0.85)

    def test_niah_miss_fails(self):
        # DS recalls only 3/5 needles -> below the 4/5 threshold.
        niah = _hit_niah()
        niah[0] = (niah[0][0], "i do not know")
        niah[1] = (niah[1][0], "no idea")
        res = _lib.compute_gates(_identical_smoke(), niah)
        self.assertFalse(res["gates"]["niah_mini_recall"]["pass"])
        self.assertEqual(res["gates"]["niah_mini_recall"]["hits"], 3)

    def test_first8_divergence_fails(self):
        # One prompt has zero first-8-token overlap between DS and DSA.
        pairs = _identical_smoke()
        pairs[0] = ("prompt 0", "one two three four five six seven eight nine",
                    "AA BB CC DD EE FF GG HH II")
        res = _lib.compute_gates(pairs, _hit_niah())
        self.assertFalse(res["gates"]["first_8_tokens_divergence"]["pass"])
        self.assertGreaterEqual(res["gates"]["first_8_tokens_divergence"]["value"], 1)


class TestCaptureCompareRoundTrip(unittest.TestCase):
    """Prove the capture-artifact -> compare wiring without live servers."""

    def setUp(self):
        self._orig_generate = _lib.generate
        self._orig_sha = _lib.server_commit_sha
        _lib.server_commit_sha = lambda url: "deadbeef"

    def tearDown(self):
        _lib.generate = self._orig_generate
        _lib.server_commit_sha = self._orig_sha

    def _make_refs(self):
        smoke = [{"prompt": f"p{i}", "dsa_text": f"answer {i} stable text here"}
                 for i in range(20)]
        niah = [{"prompt": f"n{i}", "needle": nd, "dsa_text": f"the answer is {nd}"}
                for i, nd in enumerate(["ZEBRA-7", "MARLIN-42", "ORCHID-99",
                                        "GLACIER-13", "PHARAOH-88"])]
        return {
            "schema": _lib.REFERENCE_SCHEMA, "dsa_commit_sha": "cafef00d",
            "captured_at": "20260529T000000Z",
            "smoke_max_new_tokens": 256, "niah_max_new_tokens": 16,
            "smoke": smoke, "niah": niah,
        }

    def test_compare_all_pass_when_ds_echoes_dsa(self):
        refs = self._make_refs()
        # DS reproduces the DSA reference for every prompt.
        lookup = {e["prompt"]: e["dsa_text"] for e in refs["smoke"]}
        lookup.update({e["prompt"]: e["dsa_text"] for e in refs["niah"]})
        _lib.generate = lambda url, prompt, **kw: lookup[prompt]

        res = _lib.evaluate_against_references("http://ds.invalid", refs)
        self.assertTrue(res["all_pass"], res["gates"])
        self.assertEqual(res["ds_commit_sha"], "deadbeef")
        self.assertEqual(res["dsa_commit_sha"], "cafef00d")
        self.assertEqual(len(res["smoke_records"]), 20)
        self.assertEqual(len(res["niah_records"]), 5)

    def test_compare_fails_when_ds_outputs_garbage(self):
        refs = self._make_refs()
        _lib.generate = lambda url, prompt, **kw: "qqq zzz www totally divergent"
        res = _lib.evaluate_against_references("http://ds.invalid", refs)
        self.assertFalse(res["all_pass"])

    def test_compare_rejects_bad_schema(self):
        bad = self._make_refs()
        bad["schema"] = "wrong_schema"
        _lib.generate = lambda url, prompt, **kw: "x"
        with self.assertRaises(ValueError):
            _lib.evaluate_against_references("http://ds.invalid", bad)


if __name__ == "__main__":
    unittest.main()
