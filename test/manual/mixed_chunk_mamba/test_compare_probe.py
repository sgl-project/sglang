"""Edge cases of compare_probe: a missing or invalid side must surface as an
incomplete result, never as an equality verdict. Not registered in CI; run
with ``pytest test/manual/mixed_chunk_mamba/test_compare_probe.py``."""

import copy
import importlib.util
import pathlib
import unittest

_SPEC = importlib.util.spec_from_file_location(
    "compare_probe", pathlib.Path(__file__).with_name("compare_probe.py")
)
compare_probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(compare_probe)

FP_A = [310.5, 1430.7, 4.9, -256.2, 72359.0]
FP_B = [332.3, 1486.9, 4.8, -398.2, 89859.9]
FP_R = [322.8, 1603.6, 4.8, -353.2, 86905.5]
ZERO = [0.0] * 5


def _phase(name, depths, restore_fp, *, tracked_in_mixed):
    def chain(rid):
        return {
            "claims": [],
            "mixed_batches": 3 if tracked_in_mixed else 0,
            "mixed_mask_for_rid": [True] * 3 if tracked_in_mixed else [],
            "donations": [],
            "restores": [],
        }

    p = chain(f"P-{name}")
    for depth, fp in depths.items():
        p["claims"].append(
            {
                "claim": {"depth": depth, "slot": 30 + depth % 7},
                "tracked_in_forwards": [
                    {"one_token_rows": 8 if tracked_in_mixed else 0}
                ],
            }
        )
        p["donations"].append(
            {
                "depth": depth,
                "slot": 30 + depth % 7,
                "kind": "unfinished",
                "mamba_exist": False,
                "fp": fp,
            }
        )
    chains = {f"P-{name}": p}
    for k in ("R", "P2"):
        c = chain(f"{k}-{name}")
        if restore_fp is not None:
            c["restores"].append(
                {"matched": max(depths) if depths else 0, "slot": 37, "fp": restore_fp}
            )
        chains[f"{k}-{name}"] = c
    return {
        "tag": f"probe-{name}",
        "chains": chains,
        "answers": {
            k: {"text": "14", "correct": True, "cached_tokens": None}
            for k in ("P", "R", "P2")
        },
    }


def _pair(mixed_depths, ref_depths, mixed_restore=FP_R, ref_restore=FP_R):
    return (
        _phase("mixed", mixed_depths, mixed_restore, tracked_in_mixed=True),
        _phase("ref", ref_depths, ref_restore, tracked_in_mixed=False),
    )


class TestCompareProbe(unittest.TestCase):
    def test_complete_and_equal(self):
        depths = {1984: FP_A, 4024: FP_B, 4784: FP_R}
        result = compare_probe.compare(*_pair(depths, depths))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["verdict"], "equal")
        self.assertTrue(all(r["mixed_equals_ref"] for r in result["depths"]))

    def test_complete_but_mismatching_is_reported_as_mismatch(self):
        mixed = {1984: ZERO, 4024: ZERO, 4784: ZERO}
        ref = {1984: FP_A, 4024: FP_B, 4784: FP_R}
        result = compare_probe.compare(*_pair(mixed, ref, mixed_restore=ZERO))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["verdict"], "mismatch")
        self.assertTrue(all(r["mixed_is_all_zero"] for r in result["depths"]))
        self.assertFalse(result["restores"]["R"]["content_equal"])

    def test_missing_reference_depth_is_incomplete(self):
        mixed = {1984: FP_A, 4024: FP_B, 4784: FP_R}
        ref = {1984: FP_A, 4784: FP_R}
        result = compare_probe.compare(*_pair(mixed, ref))
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["verdict"])
        self.assertNotIn("all_mixed_depths_equal_ref", result)
        self.assertTrue(
            any("4024" in r and "reference" in r for r in result["reasons"])
        )

    def test_reference_depth_outside_the_claims_is_incomplete(self):
        mixed = {1984: FP_A}
        ref = {1984: FP_A, 4024: FP_B}
        result = compare_probe.compare(*_pair(mixed, ref))
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("not claimed" in r for r in result["reasons"]))

    def test_no_claimed_depths_is_incomplete(self):
        result = compare_probe.compare(*_pair({}, {}))
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["verdict"])
        self.assertIn("mixed phase claimed no checkpoint depth", result["reasons"])

    def test_missing_restore_is_incomplete(self):
        depths = {1984: FP_A}
        result = compare_probe.compare(*_pair(depths, depths, ref_restore=None))
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("no reference restore" in r for r in result["reasons"]))

    def test_fingerprint_must_be_five_finite_numbers(self):
        depths = {1984: FP_A}
        for bad in (
            [1.0, 2.0, 3.0, 4.0],
            FP_A + [1.0],
            [float("nan")] + FP_A[1:],
            [float("inf")] + FP_A[1:],
            ["1.0"] + FP_A[1:],
            None,
            [True] + FP_A[1:],
        ):
            with self.subTest(bad=bad):
                mixed, ref = _pair(depths, depths)
                ref = copy.deepcopy(ref)
                ref["chains"]["P-ref"]["donations"][0]["fp"] = bad
                result = compare_probe.compare(mixed, ref)
                self.assertEqual(result["status"], "incomplete")
                self.assertIsNone(result["verdict"])
                self.assertTrue(
                    any("invalid reference fingerprint" in r for r in result["reasons"])
                )

    def test_freed_duplicate_donation_does_not_count(self):
        depths = {1984: FP_A}
        mixed, ref = _pair(depths, depths)
        mixed = copy.deepcopy(mixed)
        mixed["chains"]["P-mixed"]["donations"][0]["mamba_exist"] = True
        result = compare_probe.compare(mixed, ref)
        self.assertEqual(result["status"], "incomplete")
        self.assertTrue(any("no mixed-phase donation" in r for r in result["reasons"]))


if __name__ == "__main__":
    unittest.main()
