"""Registered CPU regressions for the AC-10 M3-B verdict helper.

The helper lives next to the manual hardware fixture
(``test/manual/_m3b_label_capture_verdict.py``) so the same code path
the fixture uses is exercised here under CPU-only CI. These tests
guard the false-pass classes Codex's Round-36 review identified:

  * empty cold capture + non-zero ``cached_tokens`` must FAIL.
  * empty warm capture + populated cold + non-zero cached_tokens must
    FAIL.
  * populated cold + warm but ``cached_tokens == 0`` must FAIL.
  * ``slots_sha`` mismatch between cold and warm must FAIL.
  * per-layer label SHA divergence must FAIL.
  * ``per_layer_written_all_true=False`` on either side must FAIL.
  * everything in order must PASS.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
HELPER_PATH = REPO_ROOT / "test" / "manual" / "_m3b_label_capture_verdict.py"


def _load_verdict():
    """Load the manual-fixture helper as a registered-test dependency.

    Uses the importlib-spec pattern so the helper, which lives under
    ``test/manual/`` (not on the default test path), is reachable
    without polluting ``sys.path`` with the manual-tests directory.
    """
    spec = importlib.util.spec_from_file_location(
        "_m3b_label_capture_verdict", str(HELPER_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m3b_label_capture_verdict"] = mod
    spec.loader.exec_module(mod)
    return mod


_verdict_mod = _load_verdict()
_eval = _verdict_mod.evaluate_m3b_label_capture_verdict


def _good_record(*, slots_sha="aaa", layer_shas=("L0", "L1"),
                 written_all_true=True, prompt_len=8):
    return {
        "prompt_len": prompt_len,
        "slots_sha": slots_sha,
        "per_layer_label_sha": list(layer_shas),
        "per_layer_written_sha": list(layer_shas),
        "per_layer_written_all_true": [written_all_true] * len(layer_shas),
    }


class TestM3BLabelCaptureVerdict(unittest.TestCase):

    def test_all_conditions_met_is_pass(self):
        result = _eval(
            cold_capture=[_good_record()],
            warm_capture=[_good_record()],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["reasons"], [])

    def test_empty_cold_capture_fails_false_pass_guard(self):
        """Closes the Round-36 review false-pass path: a server that
        did not run with capture enabled returns an empty list; the
        helper must REJECT, not PASS."""
        result = _eval(
            cold_capture=[],
            warm_capture=[_good_record()],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cold capture missing or empty" in r for r in result["reasons"])
        )

    def test_empty_warm_capture_fails(self):
        result = _eval(
            cold_capture=[_good_record()],
            warm_capture=[],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("warm capture missing" in r for r in result["reasons"])
        )

    def test_none_capture_fails(self):
        """meta_info missing the key entirely → None → must FAIL,
        not PASS."""
        result = _eval(
            cold_capture=None, warm_capture=None, cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_zero_cached_tokens_fails(self):
        """Both captures populated but warm pass did not exercise the
        radix cache → FAIL. Otherwise the test only re-proves the CPU
        unit determinism property."""
        result = _eval(
            cold_capture=[_good_record()],
            warm_capture=[_good_record()],
            cached_tokens=0,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cached_tokens=0" in r for r in result["reasons"])
        )

    def test_slots_sha_mismatch_fails(self):
        result = _eval(
            cold_capture=[_good_record(slots_sha="aaa")],
            warm_capture=[_good_record(slots_sha="bbb")],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("slots_sha mismatch" in r for r in result["reasons"])
        )

    def test_layer_label_sha_mismatch_fails(self):
        result = _eval(
            cold_capture=[_good_record(layer_shas=("L0", "L1", "L2"))],
            warm_capture=[_good_record(layer_shas=("L0", "X", "L2"))],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        joined = "\n".join(result["reasons"])
        self.assertIn("per_layer_label_sha differs", joined)
        # The first-mismatch hint names layer 1.
        self.assertIn("'layer': 1", joined)

    def test_layer_label_length_mismatch_fails(self):
        result = _eval(
            cold_capture=[_good_record(layer_shas=("L0", "L1"))],
            warm_capture=[_good_record(layer_shas=("L0", "L1", "L2"))],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("length mismatch" in r for r in result["reasons"])
        )

    def test_unwritten_slot_on_cold_fails(self):
        result = _eval(
            cold_capture=[_good_record(written_all_true=False)],
            warm_capture=[_good_record(written_all_true=True)],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any(r.startswith("cold pass: per_layer_written_all_true=False")
                for r in result["reasons"])
        )

    def test_unwritten_slot_on_warm_fails(self):
        result = _eval(
            cold_capture=[_good_record(written_all_true=True)],
            warm_capture=[_good_record(written_all_true=False)],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any(r.startswith("warm pass: per_layer_written_all_true=False")
                for r in result["reasons"])
        )

    def test_non_list_capture_treated_as_missing(self):
        """meta_info value may arrive as a dict or other shape if the
        server doesn't follow the protocol; treat as missing → FAIL."""
        result = _eval(
            cold_capture={"oops": "wrong shape"},
            warm_capture=[_good_record()],
            cached_tokens=10,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cold capture missing" in r for r in result["reasons"])
        )


if __name__ == "__main__":
    unittest.main()
