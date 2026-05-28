"""Pure verdict helper for the AC-10 M3-B label-capture fixture.

Split out from the manual fixture so the false-pass logic + the
producer→transport→consumer chain can be exercised under CPU-only CI
without spinning up an sglang server.

Two shapes are accepted for the ``meta_info["double_sparsity_radix_capture"]``
value:

* **dict** — the actual production transport. The scheduler's
  ``_maybe_collect_per_request_summary`` unwraps the producer's
  per-batch list ``v[i]`` per request, then the tokenizer surfaces a
  single dict per request in ``meta_info``. This is what the real
  ``/generate`` response carries.
* **list** — the producer-side shape, useful for direct unit-tests
  of this helper. A single-request fixture's list contains exactly
  one record; multi-request lists are treated as one record per
  position but the helper only uses ``[0]``.

Anything else (None, missing key, dict with no records) is treated as
missing evidence and FAILs the verdict.

A PASS requires ALL of:

  1. Cold capture present and well-shaped.
  2. Warm capture present and well-shaped.
  3. ``cached_tokens > 0`` on the warm pass — proves the radix
     cache actually reused slots.
  4. For each of the first ``cached_tokens`` prompt positions, the
     ``per_token_slot_sha`` matches between cold and warm AND each
     layer's per-position label SHA matches. Extra decode-allocated
     slot positions beyond ``cached_tokens`` are ignored.
  5. Both passes report ``per_layer_written_all_true``.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any, Dict, List, Optional


# Reach into the production capture module to use its
# `compare_cached_prefix` helper. The helper module lives at
# ``python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py``
# but we cannot rely on PYTHONPATH being set when the manual fixture
# runs against a remote server, so load it via spec.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CAPTURE_PATH = (
    _REPO_ROOT
    / "python" / "sglang" / "srt" / "layers" / "attention"
    / "double_sparsity" / "radix_fixture_capture.py"
)


def _load_capture():
    if "sglang.srt.layers.attention.double_sparsity.radix_fixture_capture" in sys.modules:
        return sys.modules[
            "sglang.srt.layers.attention.double_sparsity.radix_fixture_capture"
        ]
    if _CAPTURE_PATH.is_file():
        spec = importlib.util.spec_from_file_location(
            "_radix_fixture_capture_helper", str(_CAPTURE_PATH),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_radix_fixture_capture_helper"] = mod
        spec.loader.exec_module(mod)
        return mod
    return None


def _records(capture: Any) -> Optional[List[Dict[str, Any]]]:
    """Normalize the meta_info value to a list of capture records.

    * ``dict`` → ``[dict]`` (production transport shape).
    * ``list`` → ``list`` (legacy / direct-helper shape).
    * anything else → ``None``.
    """
    if isinstance(capture, dict) and capture:
        return [capture]
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
    payloads + the warm pass's ``cached_tokens`` value."""
    reasons: List[str] = []

    cold = _records(cold_capture)
    warm = _records(warm_capture)

    if not cold:
        reasons.append(
            "cold capture missing or empty: meta_info["
            "'double_sparsity_radix_capture'] must be a dict (production "
            "transport) or a non-empty list (direct helper test). An "
            "empty/missing value means the server-side capture path "
            "did not fire — check that the server was launched with "
            "SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 and that DS is enabled "
            "on the route under test."
        )
    if not warm:
        reasons.append("warm capture missing or empty (same hint as cold)")

    if cold is not None and warm is not None and cold and warm:
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
        else:
            cap_mod = _load_capture()
            if cap_mod is None:
                reasons.append(
                    "radix_fixture_capture helper module is not "
                    "importable in this environment; cannot run the "
                    "cached-prefix comparison."
                )
            else:
                result = cap_mod.compare_cached_prefix(
                    cold=c, warm=w, cached_tokens=cached_tokens,
                )
                if not result["ok"]:
                    reasons.append(
                        f"cached-prefix divergence ("
                        f"kind={result['divergence_kind']!r}, "
                        f"first_diverging_position="
                        f"{result['first_diverging_position']}): "
                        f"{result['reason']}"
                    )

        c_written = c.get("per_layer_written_all_true") or []
        w_written = w.get("per_layer_written_all_true") or []
        if c_written and not all(c_written):
            bad = [i for i, v in enumerate(c_written) if not v]
            reasons.append(
                f"cold pass: per_layer_written_all_true=False on "
                f"layer(s) {bad}; some prompt slots are unreachable."
            )
        if w_written and not all(w_written):
            bad = [i for i, v in enumerate(w_written) if not v]
            reasons.append(
                f"warm pass: per_layer_written_all_true=False on "
                f"layer(s) {bad}; some prompt slots are unreachable."
            )

    return {
        "verdict": "PASS" if not reasons else "FAIL",
        "reasons": reasons,
    }
