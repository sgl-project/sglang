"""AC-10 M3-B label-capture fixture (DIRECT EVIDENCE via meta_info).

This is the fixture the AC-10 (DEC-2) guard flip requires. It verifies
that for two requests sharing the same prompt, the DS label rows
written for the cold pass are bit-equal to the DS label rows present
when the warm pass reuses those same physical KV slots via the radix
cache.

How the proof works
-------------------

The DS request path on the server side, when launched with
``SGLANG_DS_RADIX_FIXTURE_CAPTURE=1``, attaches a per-request snapshot
record to ``meta_info["double_sparsity_radix_capture"]``. Each record
contains the prompt-range physical slot SHA, per-layer SHA256 of
``signatures[L, slots]``, and per-layer ``written`` reachability
flags (see
``python/sglang/srt/layers/attention/double_sparsity/radix_fixture_capture.py``
::``build_request_capture``).

The fixture issues two identical-prompt requests, reads the snapshot
records straight off the responses, and asserts via the pure verdict
helper that:

  1. Both snapshots are present and non-empty.
  2. ``cached_tokens > 0`` on the warm pass — proves the radix cache
     actually reused slots, not just that two identical writes
     produce identical labels.
  3. ``slots_sha`` matches between cold and warm — same physical
     slots.
  4. ``per_layer_label_sha`` matches between cold and warm — label
     bytes for the shared prefix are bit-stable.
  5. ``per_layer_written_all_true`` on both sides.

The verdict helper is pure and CPU-unit-tested in
``test/registered/unit/manual/test_m3b_label_capture_verdict.py``, so
the false-pass classes (empty captures + ``cached_tokens > 0``,
``slots_sha`` mismatch, per-layer divergence, unwritten slots) are
all locked under registered CI.

Operator runbook
----------------

  # 1. Boot the DS server WITH capture enabled.
  SGLANG_DS_RADIX_OVERRIDE=1 \\
    SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \\
    bash development/serve_double_sparsity.sh
  # 2. (Optional) Run the continuation smoke as a pre-flight check.
  DS_BASE_URL=http://localhost:30000 \\
    pytest test/manual/test_dsv32_radix_cache_fixture.py -v
  # 3. Run THIS fixture against the same server. The capture data
  #    arrives in the response's meta_info, so the fixture works
  #    against remote servers without any extra wiring.
  DS_BASE_URL=http://localhost:30000 \\
    pytest test/manual/test_dsv32_radix_label_capture_fixture.py -v
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import unittest
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "development" / "results"
_VERDICT_HELPER_PATH = (
    pathlib.Path(__file__).resolve().parent
    / "_m3b_label_capture_verdict.py"
)


def _load_verdict_helper():
    spec = importlib.util.spec_from_file_location(
        "_m3b_label_capture_verdict", str(_VERDICT_HELPER_PATH),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_m3b_label_capture_verdict"] = mod
    spec.loader.exec_module(mod)
    return mod


_verdict_mod = _load_verdict_helper()


def _env(name: str) -> Optional[str]:
    return os.environ.get(name)


def _post_json(
    url: str, body: Dict[str, Any], *, timeout: float = 600.0,
) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _generate(
    base_url: str, prompt: str, *, max_new_tokens: int = 128,
) -> Dict[str, Any]:
    """Return the full /generate response (includes ``meta_info``)."""
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    return _post_json(
        f"{base_url.rstrip('/')}/generate", body, timeout=600.0,
    )


def _get_server_info(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/get_server_info"
    try:
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError):
        return {}


def _flush_cache(base_url: str) -> None:
    try:
        _post_json(f"{base_url.rstrip('/')}/flush_cache", {}, timeout=30.0)
    except urllib.error.URLError:
        pass


def _record_artifact(payload: Dict[str, Any], *, suffix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    path = RESULTS_DIR / f"dsv32_radix_label_capture_{suffix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[radix-label-capture] artifact written: {path}")


def _local_commit_sha() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT), capture_output=True, text=True,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _meta_info(resp: Dict[str, Any]) -> Dict[str, Any]:
    mi = resp.get("meta_info")
    return mi if isinstance(mi, dict) else {}


def _capture_records(resp: Dict[str, Any]) -> Any:
    """Read the per-request capture from meta_info. May be a list
    (server emitted it), missing key (capture env not set on the
    server), or other shape (protocol broken). The verdict helper
    treats anything non-list as missing."""
    return _meta_info(resp).get("double_sparsity_radix_capture")


def _cached_tokens(resp: Dict[str, Any]) -> int:
    val = _meta_info(resp).get("cached_tokens", 0)
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


# Identical prompt across both passes so the radix cache can reuse
# the shared-prefix slots. The continuation smoke fixture uses the
# same constant; both manual fixtures can run back-to-back against
# the same warm cache.
_SHARED_PREFIX_PROMPT = (
    "You are a careful technical reviewer. The following document "
    "describes a distributed key-value cache used by a large language "
    "model server. Read the document and continue the description in "
    "your own words for at least four sentences. Document:\n\n"
    "  The cache stores recent attention key/value tensors keyed by "
    "  request-pool slot. Each request owns a contiguous range of "
    "  physical slots; freed ranges are reclaimed by the allocator. "
    "  A radix tree indexes the shared prefix tokens across requests "
    "  so that two requests sharing an opening prompt can reuse the "
    "  exact same KV slots for the matching prefix range. The fp8 "
    "  KV blocks carry per-block scale factors; a sparse label table "
    "  projects each slot's K-noPE through a precomputed channel "
    "  selection. Operators verify that cold-prefix vs warm-prefix "
    "  label rows are bit-equal before enabling the radix cache "
    "  under Double Sparsity.\n\nContinue:"
)


_ENV_DS_URL = "DS_BASE_URL"


@unittest.skipUnless(
    _env(_ENV_DS_URL),
    f"{_ENV_DS_URL} must point at a running DS server. The server "
    "process itself must also have been launched with "
    "SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 so the per-request capture "
    "snapshots reach the response meta_info. Without that, the "
    "fixture's verdict helper FAILS the run rather than silently "
    "scoring it as a pass.",
)
class TestDSv32RadixLabelCaptureFixture(unittest.TestCase):
    """Cold-vs-warm DS label SHA equality via response meta_info."""

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env(_ENV_DS_URL)
        cls.server_info = _get_server_info(cls.ds_url)
        cls.local_commit_sha = _local_commit_sha()
        sa = cls.server_info.get("server_args") or {}
        if isinstance(sa, dict) and sa.get("disable_radix_cache") is True:
            raise unittest.SkipTest(
                "DS server reports disable_radix_cache=True; the M3-B "
                "label-capture fixture requires radix cache ENABLED "
                "(set SGLANG_DS_RADIX_OVERRIDE=1 + remove the flag for "
                "this one-shot run)."
            )

    def test_cold_warm_label_shas_bit_equal(self):
        run_id = uuid.uuid4().hex[:12]
        _flush_cache(self.ds_url)

        cold_resp = _generate(
            self.ds_url, _SHARED_PREFIX_PROMPT, max_new_tokens=128,
        )
        # No flush between passes — the radix cache MUST retain the
        # shared-prefix slots for the warm pass to exercise reuse.
        warm_resp = _generate(
            self.ds_url, _SHARED_PREFIX_PROMPT, max_new_tokens=128,
        )

        cold_capture = _capture_records(cold_resp)
        warm_capture = _capture_records(warm_resp)
        cached_tokens = _cached_tokens(warm_resp)

        result = _verdict_mod.evaluate_m3b_label_capture_verdict(
            cold_capture=cold_capture,
            warm_capture=warm_capture,
            cached_tokens=cached_tokens,
        )

        sa = (self.server_info.get("server_args")
              if isinstance(self.server_info, dict) else None)
        server_commit = sa.get("commit_sha") if isinstance(sa, dict) else None
        payload = {
            "fixture_kind": "label_capture_m3b",
            "run_id": run_id,
            "ds_base_url": self.ds_url,
            "local_commit_sha": self.local_commit_sha,
            "server_commit_sha": server_commit,
            "server_args": sa,
            "prompt": _SHARED_PREFIX_PROMPT,
            "cached_tokens_warm_pass": cached_tokens,
            "cold_capture": cold_capture,
            "warm_capture": warm_capture,
            "verdict": result["verdict"],
            "verdict_reasons": result["reasons"],
        }
        _record_artifact(payload, suffix="label_equality")

        self.assertEqual(
            result["verdict"], "PASS",
            "AC-10 M3-B label-capture fixture FAILED. Reasons:\n  "
            + "\n  ".join(result["reasons"]) +
            f"\nDiagnostic artifact written under {RESULTS_DIR}/. "
            "Do NOT call record_radix_fixture_passed and KEEP "
            "--disable-radix-cache in the DS launcher.",
        )


if __name__ == "__main__":
    unittest.main()
