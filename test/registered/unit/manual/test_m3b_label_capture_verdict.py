"""Registered CPU regressions for the AC-10 M3-B verdict helper.

The helper lives next to the manual hardware fixture
(``test/manual/_m3b_label_capture_verdict.py``) so the same code path
the fixture uses is exercised here under CPU-only CI. These tests
guard the false-pass classes Codex's Round-36 review identified
PLUS the Round-37-review transport-shape / timing defects that
caused drift in Rounds 35→37.

Drift-recovery integration test (the load-bearing CI check):

* ``test_meta_info_transport_dict_shape_passes_verdict`` simulates
  the actual production transport — the scheduler's
  ``_maybe_collect_per_request_summary`` unwraps the producer's
  per-batch list ``v[i]`` per request, then the tokenizer surfaces a
  single dict per request in ``meta_info``. The verdict helper must
  accept that dict shape.

Plus false-pass regressions:

* empty/None/non-list-non-dict capture must FAIL.
* zero ``cached_tokens`` must FAIL.
* per-token slot SHA divergence within the cached prefix must FAIL
  AT THE FIRST DIVERGING POSITION.
* per-layer per-token label SHA divergence within the cached prefix
  must FAIL.
* extra decode-allocated slots beyond ``cached_tokens`` must NOT
  cause a false mismatch (the warm pass legitimately allocates more
  slots for the suffix or for generation; the cached prefix is what
  AC-10 is gating).
* ``per_layer_written_all_true=False`` on either side must FAIL.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
HELPER_PATH = REPO_ROOT / "test" / "manual" / "_m3b_label_capture_verdict.py"


def _load_verdict():
    spec = importlib.util.spec_from_file_location(
        "_m3b_label_capture_verdict", str(HELPER_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m3b_label_capture_verdict"] = mod
    spec.loader.exec_module(mod)
    return mod


_verdict_mod = _load_verdict()
_eval = _verdict_mod.evaluate_m3b_label_capture_verdict


def _good_record(
    *,
    prompt_len=5,
    slot_shas=None,
    layer_label_shas=None,
    written_all_true=True,
):
    """Build a capture record matching the producer
    (``build_request_capture``) shape with realistic per-token data."""
    if slot_shas is None:
        slot_shas = [f"slot_{i}" for i in range(prompt_len)]
    if layer_label_shas is None:
        # Two layers, per-token label sha is deterministic per position.
        layer_label_shas = [
            [f"L0_t{i}" for i in range(prompt_len)],
            [f"L1_t{i}" for i in range(prompt_len)],
        ]
    return {
        "prompt_len": prompt_len,
        "slots_sha": "agg_" + "_".join(slot_shas),
        "per_token_slot_sha": list(slot_shas),
        "per_layer_label_sha": ["L0_agg", "L1_agg"],
        "per_layer_written_sha": ["L0_w", "L1_w"],
        "per_layer_written_all_true": [written_all_true] * 2,
        "per_layer_per_token_label_sha": [list(row) for row in layer_label_shas],
    }


def _maybe_collect_per_request_summary_shim(producer_summary, i):
    """Mirror the scheduler's transport-side unwrap.

    See ``python/sglang/srt/managers/scheduler_components/batch_result_processor.py``
    ``_maybe_collect_per_request_summary``: each per-batch list value
    is indexed at the per-request position. The tokenizer then
    surfaces the per-request entry as ``meta_info[k] = entry`` (a
    dict for our shape).
    """
    out = {}
    for k, v in producer_summary.items():
        if v is None or i >= len(v):
            continue
        entry = v[i]
        if entry is None:
            continue
        out[k] = entry
    return out


class TestM3BLabelCaptureVerdictTransport(unittest.TestCase):
    """The integration test that caught Round 37's transport drift."""

    def test_meta_info_transport_dict_shape_passes_verdict(self):
        """Drift-recovery anchor: the actual production transport
        unwraps the producer's per-batch list into one dict per
        request in ``meta_info``. The verdict helper must accept that
        dict shape. Round 37 published the right producer shape but
        the verdict helper rejected dicts → false fail."""
        record = _good_record(prompt_len=5)
        # Producer-side per-request summary: per-batch list under
        # the namespace key. Single-request batch → list of length 1.
        producer_cold = {"double_sparsity_radix_capture": [record]}
        producer_warm = {"double_sparsity_radix_capture": [record]}

        # Transport: scheduler unwraps v[i] per request, then
        # tokenizer surfaces meta_info[k] = entry.
        cold_meta = _maybe_collect_per_request_summary_shim(producer_cold, 0)
        warm_meta = _maybe_collect_per_request_summary_shim(producer_warm, 0)
        # Per-request meta_info value is a DICT, not a list.
        self.assertIsInstance(
            cold_meta["double_sparsity_radix_capture"], dict,
        )

        result = _eval(
            cold_capture=cold_meta["double_sparsity_radix_capture"],
            warm_capture=warm_meta["double_sparsity_radix_capture"],
            cached_tokens=5,
        )
        self.assertEqual(
            result["verdict"], "PASS",
            f"transported dict-shape capture must PASS; got "
            f"{result['verdict']} reasons={result['reasons']!r}",
        )

    def test_meta_info_transport_dict_shape_with_extra_decode_slots(self):
        """Warm capture has more positions (suffix or generated
        tokens) than cold's cached prefix. The cached-prefix
        comparison ignores positions beyond cached_tokens, so the
        verdict still PASSes."""
        cold = _good_record(prompt_len=5)
        warm = _good_record(
            prompt_len=8,  # 5 prompt + 3 suffix/decode slots
            slot_shas=[
                "slot_0", "slot_1", "slot_2", "slot_3", "slot_4",
                "slot_NEW_5", "slot_NEW_6", "slot_NEW_7",
            ],
            layer_label_shas=[
                [
                    "L0_t0", "L0_t1", "L0_t2", "L0_t3", "L0_t4",
                    "L0_decode_5", "L0_decode_6", "L0_decode_7",
                ],
                [
                    "L1_t0", "L1_t1", "L1_t2", "L1_t3", "L1_t4",
                    "L1_decode_5", "L1_decode_6", "L1_decode_7",
                ],
            ],
        )
        producer_cold = {"double_sparsity_radix_capture": [cold]}
        producer_warm = {"double_sparsity_radix_capture": [warm]}
        cold_meta = _maybe_collect_per_request_summary_shim(producer_cold, 0)
        warm_meta = _maybe_collect_per_request_summary_shim(producer_warm, 0)

        result = _eval(
            cold_capture=cold_meta["double_sparsity_radix_capture"],
            warm_capture=warm_meta["double_sparsity_radix_capture"],
            cached_tokens=5,
        )
        self.assertEqual(
            result["verdict"], "PASS",
            f"extra decode slots beyond cached_tokens must NOT cause "
            f"a false mismatch; got reasons={result['reasons']!r}",
        )

    def test_meta_info_transport_dict_shape_with_cached_prefix_diverging(self):
        """Cold and warm differ at position 2 within the first
        ``cached_tokens=5`` positions → FAIL naming position 2."""
        cold = _good_record(prompt_len=5)
        warm = _good_record(
            prompt_len=5,
            slot_shas=[
                "slot_0", "slot_1",
                "slot_DIVERGED_2",  # position 2 differs from cold
                "slot_3", "slot_4",
            ],
        )
        producer_cold = {"double_sparsity_radix_capture": [cold]}
        producer_warm = {"double_sparsity_radix_capture": [warm]}
        cold_meta = _maybe_collect_per_request_summary_shim(producer_cold, 0)
        warm_meta = _maybe_collect_per_request_summary_shim(producer_warm, 0)

        result = _eval(
            cold_capture=cold_meta["double_sparsity_radix_capture"],
            warm_capture=warm_meta["double_sparsity_radix_capture"],
            cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        joined = "\n".join(result["reasons"])
        self.assertIn("first_diverging_position=2", joined)
        self.assertIn("kind='slot'", joined)

    def test_meta_info_transport_label_divergence_within_cached_prefix(self):
        """Slot SHAs agree but per-layer per-token label SHA differs
        at a position within the cached prefix → FAIL with
        kind='label'."""
        cold = _good_record(prompt_len=5)
        warm = _good_record(
            prompt_len=5,
            layer_label_shas=[
                ["L0_t0", "L0_t1", "L0_t2", "L0_DIVERGED_3", "L0_t4"],
                ["L1_t0", "L1_t1", "L1_t2", "L1_t3", "L1_t4"],
            ],
        )
        result = _eval(
            cold_capture=cold,  # dict shape direct (legacy path)
            warm_capture=warm,
            cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        joined = "\n".join(result["reasons"])
        self.assertIn("kind='label'", joined)
        self.assertIn("first_diverging_position=3", joined)
        self.assertIn("layer=0", joined)


class TestM3BLabelCaptureVerdictNegatives(unittest.TestCase):
    """False-pass guards on the verdict logic."""

    def test_all_conditions_met_dict_is_pass(self):
        rec = _good_record(prompt_len=5)
        result = _eval(
            cold_capture=rec, warm_capture=rec, cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_all_conditions_met_list_is_pass(self):
        """Legacy list shape still accepted by the helper."""
        rec = _good_record(prompt_len=5)
        result = _eval(
            cold_capture=[rec], warm_capture=[rec], cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_empty_cold_capture_fails(self):
        result = _eval(
            cold_capture={}, warm_capture=_good_record(),
            cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cold capture missing" in r for r in result["reasons"])
        )

    def test_empty_warm_capture_fails(self):
        result = _eval(
            cold_capture=_good_record(),
            warm_capture=[], cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("warm capture missing" in r for r in result["reasons"])
        )

    def test_none_capture_fails(self):
        result = _eval(
            cold_capture=None, warm_capture=None, cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_zero_cached_tokens_fails(self):
        result = _eval(
            cold_capture=_good_record(),
            warm_capture=_good_record(),
            cached_tokens=0,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cached_tokens=0" in r for r in result["reasons"])
        )

    def test_unwritten_slot_on_cold_fails(self):
        cold = _good_record(written_all_true=False)
        warm = _good_record(written_all_true=True)
        result = _eval(
            cold_capture=cold, warm_capture=warm, cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any(r.startswith("cold pass: per_layer_written_all_true=False")
                for r in result["reasons"])
        )

    def test_unwritten_slot_on_warm_fails(self):
        cold = _good_record(written_all_true=True)
        warm = _good_record(written_all_true=False)
        result = _eval(
            cold_capture=cold, warm_capture=warm, cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")

    def test_non_list_non_dict_capture_treated_as_missing(self):
        """Any other shape (e.g. a stray int from a protocol break)
        is treated as missing evidence."""
        result = _eval(
            cold_capture=42, warm_capture=_good_record(),
            cached_tokens=5,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertTrue(
            any("cold capture missing" in r for r in result["reasons"])
        )


if __name__ == "__main__":
    unittest.main()
