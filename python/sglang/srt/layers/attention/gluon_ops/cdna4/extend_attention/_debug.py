"""Debug / observability knobs for the Gluon extend-attention path.

All dev-only flags live under a single ``SGLANG_GLUON_DEBUG`` env var, as a
comma-separated list of ``key`` or ``key=value`` tokens. Unknown keys are
ignored so rolling this forward is backwards-compatible.

Examples
--------
``SGLANG_GLUON_DEBUG=trace=100``
    Log dispatch-path counters every 100 extend calls.

``SGLANG_GLUON_DEBUG=compare=32,compare_log=/tmp/diff.jsonl``
    Run the first 32 calls against both Triton and Gluon, log the
    per-call max abs diff to the given file.

``SGLANG_GLUON_DEBUG=profile_shapes=/tmp/shapes.json,profile_dump_after=1000``
    Dump per-call shape counters to the given path every 1000 calls.

``SGLANG_GLUON_DEBUG=disable_cfg_cache,unify_causal_path``
    Pure flag tokens; same as ``=1``.

The user-facing FP8 safety rail (``SGLANG_GLUON_FP8_KV_FORCE_BF16``) lives
outside this module because it is documented in end-user error messages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


_KNOWN_KEYS = frozenset({
    "trace",
    "compare",
    "compare_log",
    "profile_shapes",
    "profile_dump_after",
    "disable_cfg_cache",
    "unify_causal_path",
})


@dataclass(frozen=True)
class GluonDebugConfig:
    # Dispatch-path trace. ``trace=N`` logs cumulative counters every N calls.
    # 0 disables.
    trace_interval: int = 0

    # Numerical parity checking. ``compare=N`` runs the first N live calls
    # through both Triton and Gluon and records the max abs diff.
    compare_remaining: int = 0
    compare_log: Optional[str] = None

    # Shape-profile dump. ``profile_shapes=<path>`` buckets per-call shapes;
    # ``profile_dump_after=N`` forces a flush every N calls (0 = on exit only).
    profile_shapes_path: Optional[str] = None
    profile_dump_after: int = 0

    # Correctness-debug escape hatches.
    disable_cfg_cache: bool = False
    unify_causal_path: bool = False


def _parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse() -> GluonDebugConfig:
    raw = os.getenv("SGLANG_GLUON_DEBUG", "") or ""
    if not raw.strip():
        return GluonDebugConfig()

    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    fields = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            k, v = k.strip(), v.strip()
        else:
            k, v = tok.strip(), "1"
        if k not in _KNOWN_KEYS:
            continue
        fields[k] = v

    return GluonDebugConfig(
        trace_interval=_parse_int(fields.get("trace", "0")),
        compare_remaining=_parse_int(fields.get("compare", "0")),
        compare_log=fields.get("compare_log") or None,
        profile_shapes_path=fields.get("profile_shapes") or None,
        profile_dump_after=_parse_int(fields.get("profile_dump_after", "0")),
        disable_cfg_cache=_parse_int(fields.get("disable_cfg_cache", "0")) != 0,
        unify_causal_path=_parse_int(fields.get("unify_causal_path", "0")) != 0,
    )


DEBUG: GluonDebugConfig = _parse()
