# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the BLASST skip-softmax calibration helpers.

Pure stdlib + ``warnings`` only -- no torch / no sglang server-args / no
modelopt required. Runnable in two ways::

    pytest test_modelopt_skip_softmax_calibration.py    # full discovery
    python  test_modelopt_skip_softmax_calibration.py   # plain script mode

The tested module lives next to the backend so it can be imported from a
unit-test process that doesn't have CUDA / triton available.
"""

from __future__ import annotations

import json
import math
import tempfile
import warnings
from pathlib import Path

from sglang.multimodal_gen.runtime.layers.attention.backends._modelopt_skip_softmax_calib import (
    DEFAULT_TARGET_SPARSE_RATIO,
    component_key_from_prefix,
    compute_threshold,
    entry_from_flat_ab,
    entry_from_modelopt_canonical,
    load_calibration_file,
    normalize_calibration,
    normalize_target_sparse_ratio,
    pick_calibration_entry,
    resolve_target_sparsity,
)


# ====================================================================== #
#  normalize_target_sparse_ratio                                         #
# ====================================================================== #
def test_normalize_target_sparse_ratio_scalar():
    assert normalize_target_sparse_ratio(0.4) == {"prefill": 0.4, "decode": 0.4}
    assert normalize_target_sparse_ratio(0) == {"prefill": 0.0, "decode": 0.0}


def test_normalize_target_sparse_ratio_dict():
    assert normalize_target_sparse_ratio({"prefill": 0.3, "decode": 0.7}) == {
        "prefill": 0.3,
        "decode": 0.7,
    }


def test_normalize_target_sparse_ratio_partial_dict():
    # Missing keys default to 0.5/0.5 (modelopt vllm convention).
    assert normalize_target_sparse_ratio({"prefill": 0.6}) == {
        "prefill": 0.6,
        "decode": 0.5,
    }


def test_normalize_target_sparse_ratio_none_and_invalid():
    for bad in (None, "0.4", [0.4], True, False):
        out = normalize_target_sparse_ratio(bad)
        assert out == DEFAULT_TARGET_SPARSE_RATIO, f"{bad!r} -> {out}"


# ====================================================================== #
#  entry_from_modelopt_canonical                                         #
# ====================================================================== #
def test_entry_modelopt_full():
    raw = {
        "threshold_scale_factor": {
            "prefill": {"a": 10.0, "b": 5.0},
            "decode": {"a": 11.0, "b": 6.0},
        },
        "target_sparse_ratio": {"prefill": 0.4, "decode": 0.6},
    }
    e = entry_from_modelopt_canonical(raw)
    assert e is not None
    assert e["phases"] == {"prefill": (10.0, 5.0), "decode": (11.0, 6.0)}
    assert e["target_sparse_ratio"] == {"prefill": 0.4, "decode": 0.6}


def test_entry_modelopt_prefill_only_fills_decode():
    raw = {"threshold_scale_factor": {"prefill": {"a": 10.0, "b": 5.0}}}
    e = entry_from_modelopt_canonical(raw)
    assert e is not None
    assert e["phases"]["prefill"] == e["phases"]["decode"] == (10.0, 5.0)
    assert e["target_sparse_ratio"] == DEFAULT_TARGET_SPARSE_RATIO


def test_entry_modelopt_missing_tsf_returns_none():
    assert entry_from_modelopt_canonical({"foo": "bar"}) is None
    assert entry_from_modelopt_canonical({}) is None


def test_entry_modelopt_bad_ab_skipped():
    raw = {
        "threshold_scale_factor": {
            "prefill": {"a": "not-a-number", "b": 5.0},
            "decode": {"a": 11.0, "b": 6.0},
        }
    }
    e = entry_from_modelopt_canonical(raw)
    # prefill skipped, decode kept, prefill back-filled from decode.
    assert e is not None
    assert e["phases"]["prefill"] == (11.0, 6.0)
    assert e["phases"]["decode"] == (11.0, 6.0)


# ====================================================================== #
#  normalize_calibration (schema auto-detect)                            #
# ====================================================================== #
def test_normalize_flat_one_model():
    cal = normalize_calibration({"a": 12.0, "b": 5.5})
    assert set(cal) == {"_default"}
    assert cal["_default"]["phases"]["prefill"] == (12.0, 5.5)
    assert cal["_default"]["target_sparse_ratio"] == DEFAULT_TARGET_SPARSE_RATIO


def test_normalize_flat_per_component():
    cal = normalize_calibration(
        {
            "transformer": {"a": 10.0, "b": 5.0},
            "transformer_2": {"a": 9.0, "b": 4.5},
        }
    )
    assert set(cal) == {"transformer", "transformer_2"}
    assert cal["transformer"]["phases"]["prefill"] == (10.0, 5.0)
    assert cal["transformer_2"]["phases"]["decode"] == (9.0, 4.5)


def test_normalize_modelopt_canonical_per_component():
    cal = normalize_calibration(
        {
            "transformer": {
                "threshold_scale_factor": {
                    "prefill": {"a": 8.0, "b": 7.0},
                    "decode": {"a": 8.0, "b": 7.0},
                },
                "target_sparse_ratio": 0.55,
            },
            "transformer_2": {
                "threshold_scale_factor": {
                    "prefill": {"a": 7.5, "b": 6.5},
                    "decode": {"a": 7.5, "b": 6.5},
                },
            },
        }
    )
    assert cal["transformer"]["target_sparse_ratio"] == {
        "prefill": 0.55,
        "decode": 0.55,
    }
    assert cal["transformer_2"]["target_sparse_ratio"] == DEFAULT_TARGET_SPARSE_RATIO
    assert cal["transformer"]["phases"]["prefill"] == (8.0, 7.0)


def test_normalize_mixed_schemas_in_one_file():
    cal = normalize_calibration(
        {
            "transformer": {"a": 10.0, "b": 5.0},
            "transformer_2": {
                "threshold_scale_factor": {"prefill": {"a": 9.0, "b": 4.5}},
            },
        }
    )
    assert cal["transformer"]["phases"]["prefill"] == (10.0, 5.0)
    assert cal["transformer_2"]["phases"]["prefill"] == (9.0, 4.5)


def test_normalize_bad_inputs_return_empty():
    assert normalize_calibration(None) == {}
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert normalize_calibration("not a dict") == {}
        assert normalize_calibration([1, 2, 3]) == {}
        assert any("must be a JSON object" in str(x.message) for x in w)


def test_normalize_entries_with_no_recognisable_schema_skipped():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cal = normalize_calibration(
            {
                "good": {"a": 10.0, "b": 5.0},
                "junk": {"random_field": 42},
                "bad_payload": "this should be a dict",
            }
        )
        assert set(cal) == {"good"}
        assert any("no recognised" in str(x.message) for x in w)
        assert any("is not a dict" in str(x.message) for x in w)


# ====================================================================== #
#  pick_calibration_entry                                                #
# ====================================================================== #
def test_pick_calibration_entry_direct_hit():
    cal = {"transformer": entry_from_flat_ab(10, 5)}
    e = pick_calibration_entry(cal, "transformer")
    assert e is cal["transformer"]


def test_pick_calibration_entry_fallback_to_default():
    cal = {"_default": entry_from_flat_ab(10, 5)}
    e = pick_calibration_entry(cal, "transformer_99")
    assert e is cal["_default"]


def test_pick_calibration_entry_no_hit():
    cal = {"transformer": entry_from_flat_ab(10, 5)}
    assert pick_calibration_entry(cal, "transformer_99") is None


# ====================================================================== #
#  compute_threshold                                                     #
# ====================================================================== #
def test_compute_threshold_happy_path():
    # threshold = 5 * exp(10 * 0.5) / 16384  = 5 * e^5 / 16384 ~= 0.04528
    out = compute_threshold(a=5.0, b=10.0, target_sparsity=0.5, seq_len_k=16384)
    assert out is not None
    expected = 5.0 * math.exp(10.0 * 0.5) / 16384
    assert math.isclose(out, expected, rel_tol=1e-9)


def test_compute_threshold_invalid_returns_none():
    assert compute_threshold(a=5, b=10, target_sparsity=0, seq_len_k=16384) is None
    assert compute_threshold(a=0, b=10, target_sparsity=0.5, seq_len_k=16384) is None
    assert compute_threshold(a=5, b=10, target_sparsity=0.5, seq_len_k=0) is None
    assert compute_threshold(a=5, b=10, target_sparsity=0.5, seq_len_k=-1) is None


def test_compute_threshold_above_one_warns_and_returns_none():
    # threshold = 100 * exp(10 * 0.5) / 1 == huge
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = compute_threshold(a=100.0, b=10.0, target_sparsity=0.5, seq_len_k=1)
        assert out is None
        assert any("out of valid lambda range" in str(x.message) for x in w)


def test_compute_threshold_overflow_returns_none():
    # 10 * exp(1000 * 1.0) overflows; must not crash, must return None.
    out = compute_threshold(a=10.0, b=1000.0, target_sparsity=1.0, seq_len_k=16384)
    assert out is None


# ====================================================================== #
#  component_key_from_prefix                                             #
# ====================================================================== #
def test_component_key_from_prefix():
    assert component_key_from_prefix("transformer.blocks.0.attn1.impl") == "transformer"
    assert component_key_from_prefix("transformer_2.blocks.5.attn") == "transformer_2"
    assert component_key_from_prefix("") == "_default"
    assert component_key_from_prefix("only_one_token") == "only_one_token"


# ====================================================================== #
#  resolve_target_sparsity                                               #
# ====================================================================== #
def test_resolve_target_sparsity_override_wins():
    entry = entry_from_flat_ab(10, 5)  # default ratio 0.5
    assert (
        resolve_target_sparsity(override=0.7, calib_entry=entry, phase="prefill") == 0.7
    )


def test_resolve_target_sparsity_falls_back_to_calib():
    entry = entry_from_modelopt_canonical(
        {
            "threshold_scale_factor": {"prefill": {"a": 10.0, "b": 5.0}},
            "target_sparse_ratio": {"prefill": 0.4},
        }
    )
    assert (
        resolve_target_sparsity(override=0.0, calib_entry=entry, phase="prefill") == 0.4
    )


def test_resolve_target_sparsity_falls_back_to_default_without_entry():
    assert (
        resolve_target_sparsity(override=0.0, calib_entry=None, phase="prefill") == 0.5
    )
    assert (
        resolve_target_sparsity(override=0.0, calib_entry=None, phase="decode") == 0.5
    )


def test_resolve_target_sparsity_negative_override_treated_as_unset():
    entry = entry_from_flat_ab(10, 5)
    assert (
        resolve_target_sparsity(override=-1.0, calib_entry=entry, phase="prefill")
        == 0.5
    )


# ====================================================================== #
#  load_calibration_file (disk round-trip)                               #
# ====================================================================== #
def test_load_calibration_file_round_trip():
    payload = {
        "transformer": {"a": 12.34, "b": 5.67},
        "transformer_2": {
            "threshold_scale_factor": {"prefill": {"a": 11.0, "b": 6.0}},
            "target_sparse_ratio": 0.6,
        },
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(payload, f)
        path = f.name
    try:
        cal = load_calibration_file(path)
        assert cal["transformer"]["phases"]["prefill"] == (12.34, 5.67)
        assert cal["transformer_2"]["phases"]["prefill"] == (11.0, 6.0)
        assert cal["transformer_2"]["target_sparse_ratio"] == {
            "prefill": 0.6,
            "decode": 0.6,
        }
    finally:
        Path(path).unlink(missing_ok=True)


# ====================================================================== #
#  End-to-end: resolve a threshold using only the helpers                #
# ====================================================================== #
def test_end_to_end_resolve_prefill_default_sparsity():
    """Mirror what ``ModelOptSkipSoftmaxImpl._resolve_threshold`` does, but
    with the impl class removed -- proves the pure-helper composition is
    sufficient and correct."""
    cal = normalize_calibration(
        {
            "transformer": {
                "threshold_scale_factor": {
                    "prefill": {"a": 6.0, "b": 8.0},
                    "decode": {"a": 6.0, "b": 8.0},
                },
                "target_sparse_ratio": {"prefill": 0.5, "decode": 0.5},
            }
        }
    )
    entry = pick_calibration_entry(
        cal, component_key_from_prefix("transformer.blocks.0")
    )
    assert entry is not None
    target = resolve_target_sparsity(override=0.0, calib_entry=entry, phase="prefill")
    a, b = entry["phases"]["prefill"]
    threshold = compute_threshold(a=a, b=b, target_sparsity=target, seq_len_k=16384)
    expected = 6.0 * math.exp(8.0 * 0.5) / 16384  # ~0.01825
    assert threshold is not None and math.isclose(threshold, expected, rel_tol=1e-9)


def test_end_to_end_override_then_out_of_range():
    """Override pushes target way past calibration range -> threshold > 1
    -> compute_threshold returns None (caller falls back to dense)."""
    cal = normalize_calibration({"a": 6.0, "b": 8.0})
    entry = pick_calibration_entry(cal, "_default")
    target = resolve_target_sparsity(override=5.0, calib_entry=entry, phase="prefill")
    a, b = entry["phases"]["prefill"]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = compute_threshold(a=a, b=b, target_sparsity=target, seq_len_k=16)
        assert out is None
        assert any("out of valid lambda range" in str(x.message) for x in w)


# ---------------------------------------------------------------------- #
#  Plain-python runner (no pytest required).                             #
# ---------------------------------------------------------------------- #
def _run_all_as_script() -> int:
    import inspect
    import sys

    tests = [
        (name, fn)
        for name, fn in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
        if name.startswith("test_")
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e!r}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e!r}")
        else:
            print(f"ok    {name}")
    print(f"\n{len(tests) - failed} passed, {failed} failed, {len(tests)} total")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all_as_script())
