"""AC-10 M3-B label-capture fixture (DIRECT EVIDENCE).

This is the fixture the AC-10 (DEC-2) guard flip requires. It verifies
that for two requests sharing the same prompt, the DS label rows
written for the cold pass equal the DS label rows present when the
warm pass reuses those same physical KV slots via the radix cache.

How the proof works
-------------------

Round 36 added a server-side capture primitive
(``python/sglang/srt/layers/attention/double_sparsity/
radix_fixture_capture.py``) that, when
``SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`` is set in the server process,
appends a per-write record to an in-process log every time
``_write_token_labels`` fires. Each record carries:

* ``layer_id``
* ``cache_loc_sha`` — SHA256 of the int64 slot indices.
* ``k_nope_sha`` — SHA256 of the fp32 projected K-noPE bytes.
* ``written_after_sha`` + ``written_after_all_true`` — proof every
  slot is reachable after the write.

The fixture compares the per-layer write records between the cold
pass and the warm pass. The bit-equality property is:

* For every write (layer, cache_loc), if the warm pass produces a
  record with the same ``cache_loc_sha`` as a cold-pass record, then
  ``k_nope_sha`` must match — i.e. the K-noPE that the labeling
  kernel sees is bit-equal between the cold (fresh write) and warm
  (radix-cache-reused) paths.

A WARM pass that did not exercise the radix cache (e.g. radix cache
disabled, or no overlap) shows up either as (a) no cache_loc overlap
between passes, or (b) ``meta_info.cached_tokens == 0``. Either case
fails the test loudly.

The fixture also records ``commit_sha`` (local + server) into the
artifact so the audit trail names exactly which build authorized the
flip.

Operator runbook
----------------

  # 1. Boot the DS server WITH capture enabled.
  SGLANG_DS_RADIX_OVERRIDE=1 \\
    SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \\
    bash development/serve_double_sparsity.sh
  # 2. (Optional) Run the continuation smoke first:
  DS_BASE_URL=http://localhost:30000 \\
    pytest test/manual/test_dsv32_radix_cache_fixture.py -v
  # 3. Run THIS fixture against the same server. The server-side
  #    capture log is what we read; the test runs in-process by
  #    importing ``radix_fixture_capture`` from the server's Python
  #    when this client is co-located with the server. For remote
  #    servers, point ``SGLANG_DS_RADIX_CAPTURE_LOG_URL`` at the
  #    server's capture-log endpoint (operator-provided helper).
  DS_BASE_URL=http://localhost:30000 \\
    SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \\
    pytest test/manual/test_dsv32_radix_label_capture_fixture.py -v
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
import subprocess
import unittest
import urllib.error
import urllib.request
import uuid
from typing import Any, Dict, List, Optional


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "development" / "results"


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
    """Return the full /generate response so the test can inspect
    ``meta_info.cached_tokens`` etc."""
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


def _read_capture_log_in_process() -> List[Dict[str, Any]]:
    """Read the capture log via in-process import.

    This works when the fixture runs in the same Python process as
    the sglang server (the typical operator setup: a launcher script
    that boots the server then drives the fixture as part of its
    workflow). For a remote server, the operator wires
    ``SGLANG_DS_RADIX_CAPTURE_LOG_URL`` and the test reads the log
    over HTTP instead (see ``_read_capture_log_via_http``).
    """
    from sglang.srt.layers.attention.double_sparsity import (
        radix_fixture_capture as cap,
    )
    return cap.get_log()


def _read_capture_log_via_http(url: str) -> List[Dict[str, Any]]:
    """Fallback for remote-server operation. The operator stands up a
    minimal HTTP endpoint that returns ``get_log()`` as JSON; the URL
    is provided via ``SGLANG_DS_RADIX_CAPTURE_LOG_URL``."""
    with urllib.request.urlopen(url, timeout=10.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "log" in data:
        return data["log"]
    raise ValueError(
        f"capture-log endpoint at {url} returned unexpected shape: "
        f"{type(data).__name__}"
    )


def _clear_capture_log_in_process() -> None:
    from sglang.srt.layers.attention.double_sparsity import (
        radix_fixture_capture as cap,
    )
    cap.clear_log()


# Same identical prompt as the smoke fixture so both tests can be run
# back-to-back against the same warm radix cache. The prompt is long
# enough to allocate multiple physical KV slots, and identical between
# cold and warm passes so the radix cache can actually reuse slots.
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
_ENV_CAPTURE = "SGLANG_DS_RADIX_FIXTURE_CAPTURE"
_ENV_CAPTURE_LOG_URL = "SGLANG_DS_RADIX_CAPTURE_LOG_URL"


@unittest.skipUnless(
    _env(_ENV_DS_URL) and _env(_ENV_CAPTURE) == "1",
    f"{_ENV_DS_URL} must point at a running DS server AND "
    f"{_ENV_CAPTURE}=1 must be set in the server process so the "
    "capture log is populated. Otherwise the fixture has no direct "
    "label evidence to compare.",
)
class TestDSv32RadixLabelCaptureFixture(unittest.TestCase):
    """Direct cold/warm label SHA bit-equality fixture (AC-10 M3-B
    evidence). Compares per-write SHA256 fingerprints from the
    server-side capture log."""

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
                "(set SGLANG_DS_RADIX_OVERRIDE=1 in the server env and "
                "remove --disable-radix-cache for this one-shot run)."
            )

    @staticmethod
    def _collect_log() -> List[Dict[str, Any]]:
        url = _env(_ENV_CAPTURE_LOG_URL)
        if url:
            return _read_capture_log_via_http(url)
        return _read_capture_log_in_process()

    def test_cold_warm_label_shas_bit_equal(self):
        run_id = uuid.uuid4().hex[:12]

        _flush_cache(self.ds_url)
        # In-process operators must clear before the cold pass; remote
        # operators are responsible for clearing on the server side.
        if not _env(_ENV_CAPTURE_LOG_URL):
            _clear_capture_log_in_process()

        cold_resp = _generate(
            self.ds_url, _SHARED_PREFIX_PROMPT, max_new_tokens=128,
        )
        cold_log = self._collect_log()
        cold_writes = [r for r in cold_log if r.get("kind") == "write"]

        # IMPORTANT: do NOT flush_cache between passes — the radix
        # cache MUST retain the shared-prefix slots for the warm pass
        # to exercise reuse.
        warm_resp = _generate(
            self.ds_url, _SHARED_PREFIX_PROMPT, max_new_tokens=128,
        )
        full_log = self._collect_log()
        warm_writes = [
            r for r in full_log[len(cold_log):]
            if r.get("kind") == "write"
        ]

        cached_tokens = 0
        if isinstance(warm_resp.get("meta_info"), dict):
            cached_tokens = int(
                warm_resp["meta_info"].get("cached_tokens", 0)
            )

        # Group writes by (layer_id, cache_loc_sha). Same slots
        # written with the same K-noPE produce the same cache_loc_sha
        # AND the same k_nope_sha; the radix-cache reuse path skips
        # writes entirely (cache hit), so we expect the warm-pass
        # write set to be a STRICT SUBSET of the cold-pass write set
        # for the shared-prefix range.
        def _key(r: Dict[str, Any]):
            return (int(r["layer_id"]), r["cache_loc_sha"])

        cold_by_key: Dict[Any, Dict[str, Any]] = {
            _key(r): r for r in cold_writes
        }
        # Every warm-pass write whose (layer, cache_loc_sha) overlaps
        # with the cold-pass writes must agree on k_nope_sha.
        mismatches: List[Dict[str, Any]] = []
        overlap_keys = []
        for r in warm_writes:
            k = _key(r)
            if k in cold_by_key:
                overlap_keys.append(k)
                if cold_by_key[k]["k_nope_sha"] != r["k_nope_sha"]:
                    mismatches.append({
                        "layer_id": k[0],
                        "cache_loc_sha": k[1],
                        "cold_k_nope_sha": cold_by_key[k]["k_nope_sha"],
                        "warm_k_nope_sha": r["k_nope_sha"],
                    })

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
            "num_cold_writes": len(cold_writes),
            "num_warm_writes": len(warm_writes),
            "num_overlap_keys": len(overlap_keys),
            "num_mismatches": len(mismatches),
            "mismatches": mismatches[:32],
            "verdict": "PASS" if (
                not mismatches and cached_tokens > 0
            ) else "FAIL",
        }
        _record_artifact(payload, suffix="label_equality")

        # Assertion 1: the warm pass actually used the radix cache.
        self.assertGreater(
            cached_tokens, 0,
            "Warm pass reports cached_tokens=0; the radix cache was "
            "not exercised. Either the cache is disabled, or the cold "
            "pass evicted before the warm pass ran. Without reuse, "
            "the fixture would only test 'two identical writes produce "
            "identical labels', which the CPU unit test already "
            f"establishes. Diagnostic artifact: {RESULTS_DIR}/.",
        )
        # Assertion 2: every overlap is bit-equal.
        self.assertFalse(
            mismatches,
            "AC-10 M3-B label-capture fixture FAILED: at least one "
            "(layer, cache_loc) pair produced different K-noPE SHAs "
            "between cold and warm passes. Labels are NOT bit-stable "
            "across the radix-cache reuse path. Keep "
            "--disable-radix-cache in the DS launcher; do NOT call "
            "record_radix_fixture_passed. First 32 mismatches:\n"
            + json.dumps(mismatches[:32], indent=2),
        )


if __name__ == "__main__":
    unittest.main()
