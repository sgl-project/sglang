"""Sidecar metadata writer for ``development/benchmark{,_baseline}.sh``.

Writes the AC-8 / AC-9 / AC-11 reproducibility JSON to stdout. All
inputs arrive via environment variables — most importantly
``SERVER_ARGS_JSON``, which carries the server's ``/get_server_info``
response as a raw JSON string. The bash scripts must NOT splice that
JSON into a Python heredoc as source code (booleans/nulls / nested
dicts would crash with ``NameError``); pass it as data, parse with
``json.loads`` here.

Schema (fields are always present; missing-source values are ``null``):

    {
      "commit_sha": str,
      "mode": "double_sparsity" | "native_nsa",
      "concurrency": int,
      "seed": int,
      "num_prompts": int,
      "isl_total_tokens": int,
      "osl_tokens": int,
      "timestamp_utc": str,
      "chunked_prefill_size": int | "unknown",
      "warmup_requests": int | null,
      "measurement_window_seconds": float | null,
      "trial_id": str,
      "server_args": dict,
      "server_args_error": str | null,
    }
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional, Tuple


_UNKNOWN = "unknown"


def _opt_int(name: str) -> Optional[int]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _opt_float(name: str) -> Optional[float]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _req_int(name: str, default: int = 0) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_server_args(raw: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """Parse the raw ``/get_server_info`` JSON safely.

    Returns ``(server_args, error)``. ``error`` is None on success;
    otherwise a short human-readable string and ``server_args`` is ``{}``.
    """
    raw = (raw or "").strip()
    if not raw:
        return {}, "server_args_json_empty"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {}, f"server_args_json_parse_error: {exc}"
    if not isinstance(parsed, dict):
        return {}, f"server_args_json_not_object: got {type(parsed).__name__}"
    return parsed, None


def build_meta() -> Dict[str, Any]:
    server_args, server_args_error = _parse_server_args(
        os.environ.get("SERVER_ARGS_JSON", "")
    )
    chunked_prefill = server_args.get("chunked_prefill_size", _UNKNOWN)
    return {
        "commit_sha": os.environ.get("COMMIT_SHA", _UNKNOWN),
        "mode": os.environ.get("MODE", _UNKNOWN),
        "concurrency": _req_int("CONCURRENCY"),
        "seed": _req_int("SEED"),
        "num_prompts": _req_int("NUM_PROMPTS"),
        "isl_total_tokens": _req_int("ISL_TOTAL_TOKENS"),
        "osl_tokens": _req_int("OSL_TOKENS"),
        "timestamp_utc": os.environ.get("TIMESTAMP_UTC", _UNKNOWN),
        "chunked_prefill_size": chunked_prefill,
        # AC-11 reproducibility fields.
        "warmup_requests": _opt_int("WARMUP_REQUESTS"),
        "warmup_seconds": _opt_float("WARMUP_SECONDS"),
        "measurement_window_seconds": _opt_float("MEASUREMENT_WINDOW_S"),
        "trial_id": os.environ.get("TRIAL_ID", "1"),
        "server_args": server_args,
        "server_args_error": server_args_error,
    }


def main() -> int:
    sys.stdout.write(json.dumps(build_meta(), indent=2))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
