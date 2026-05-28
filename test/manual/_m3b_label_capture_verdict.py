"""Pure verdict helper for the AC-10 M3-B label-capture fixture.

Split out from the manual fixture so the false-pass logic can be
exercised under CPU-only CI without spinning up an sglang server.

The verdict examines two ``meta_info["double_sparsity_radix_capture"]``
records (one per pass) plus the warm pass's ``cached_tokens`` value
and returns a structured `{verdict, reasons}` dict. The reasons list
names every check that failed so the operator's artifact records
exactly why the fixture refused.

A PASS requires ALL of:

  1. Cold capture is present and non-empty.
  2. Warm capture is present and non-empty.
  3. ``cached_tokens > 0`` on the warm pass — proves the radix
     cache actually reused slots (otherwise the test only re-proves
     the CPU unit determinism property, which the registered tests
     already establish).
  4. The cold and warm ``slots_sha`` agree — the radix cache reused
     the SAME physical slots.
  5. ``per_layer_label_sha`` is bit-equal between cold and warm.
  6. ``per_layer_written_all_true`` is True on both sides (every
     prompt slot is reachable).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _records(capture: Any) -> Optional[List[Dict[str, Any]]]:
    """``meta_info["double_sparsity_radix_capture"]`` may be None or a
    list. Treat any non-list as missing."""
    if isinstance(capture, list):
        return capture
    return None


def evaluate_m3b_label_capture_verdict(
    *,
    cold_capture: Any,
    warm_capture: Any,
    cached_tokens: int,
) -> Dict[str, Any]:
    """Compute the AC-10 M3-B verdict from two per-request capture
    payloads.

    Both payloads are the list shape produced by
    ``radix_fixture_capture.build_request_capture`` and ferried to
    the client via ``meta_info["double_sparsity_radix_capture"]``.
    Single-request fixtures pass a 1-element list.
    """
    reasons: List[str] = []

    cold = _records(cold_capture)
    warm = _records(warm_capture)

    if not cold:
        reasons.append(
            "cold capture missing or empty: meta_info["
            "'double_sparsity_radix_capture'] must be a non-empty list "
            "of per-request snapshot records. An empty list means the "
            "server-side capture path did not fire — check that the "
            "server was launched with SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 "
            "and that DS is enabled on the route under test."
        )
    if not warm:
        reasons.append("warm capture missing or empty (same hint as cold)")

    if cold is not None and warm is not None and cold and warm:
        # Both passes ran a single request; compare the first record.
        c = cold[0]
        w = warm[0]

        if not isinstance(cached_tokens, int) or cached_tokens <= 0:
            reasons.append(
                f"cached_tokens={cached_tokens!r} on the warm pass; "
                "the radix cache was not exercised. Without slot "
                "reuse this fixture would only re-prove 'two identical "
                "writes produce identical labels', which the CPU unit "
                "test already establishes."
            )

        c_slots_sha = c.get("slots_sha")
        w_slots_sha = w.get("slots_sha")
        if c_slots_sha != w_slots_sha:
            reasons.append(
                f"slots_sha mismatch: cold={c_slots_sha!r} "
                f"warm={w_slots_sha!r}. The radix cache did not reuse "
                "the same physical slots for the shared prefix."
            )

        c_label = c.get("per_layer_label_sha") or []
        w_label = w.get("per_layer_label_sha") or []
        if len(c_label) != len(w_label):
            reasons.append(
                f"per_layer_label_sha length mismatch: "
                f"cold={len(c_label)} warm={len(w_label)}"
            )
        else:
            mismatches = [
                {"layer": i, "cold_sha": c_label[i], "warm_sha": w_label[i]}
                for i in range(len(c_label))
                if c_label[i] != w_label[i]
            ]
            if mismatches:
                reasons.append(
                    "per_layer_label_sha differs between cold and warm "
                    f"on {len(mismatches)} layer(s); first mismatch: "
                    f"{mismatches[0]}"
                )

        c_written = c.get("per_layer_written_all_true") or []
        w_written = w.get("per_layer_written_all_true") or []
        if not all(c_written):
            bad = [i for i, v in enumerate(c_written) if not v]
            reasons.append(
                f"cold pass: per_layer_written_all_true=False on "
                f"layer(s) {bad}; some prompt slots are unreachable."
            )
        if not all(w_written):
            bad = [i for i, v in enumerate(w_written) if not v]
            reasons.append(
                f"warm pass: per_layer_written_all_true=False on "
                f"layer(s) {bad}; some prompt slots are unreachable."
            )

    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }
