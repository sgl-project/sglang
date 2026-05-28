"""AC-10 M3-B radix-cache continuation-smoke fixture (PRE-FLIGHT ONLY).

**This file is the lightweight pre-flight smoke check, not the M3-B
evidence the AC-10 guard flip requires.**

The full M3-B-conformant evidence (per-layer label hash bit-equality
between cold and warm requests, plus FP8 block scale-factor stability)
lives in two companion fixtures:

* ``test/manual/test_dsv32_radix_label_capture_fixture.py`` — direct
  label SHA bit-equality between cold and warm; requires
  ``SGLANG_DS_RADIX_FIXTURE_CAPTURE=1`` to enable the in-process
  capture log inside the DS write hook.
* ``test/manual/test_dsv32_fp8_scale_stability.py`` — singleton vs
  packed-block FP8 quantization scale-factor equality.

What this smoke fixture does: issue paired cold + warm requests with
IDENTICAL prompts at temperature=0 and assert the continuations are
byte-identical. A pass here is necessary but NOT sufficient for the
guard flip:

* If continuations diverge → labels almost certainly diverge → DEC-2
  guard MUST remain in place.
* If continuations agree → labels MAY still diverge (e.g. divergence
  masked by argmax). The capture fixture is required to clear that
  ambiguity.

Operator runbook (pre-flight, then full M3-B):

  # 0. Boot DS server with radix cache ON for the one-shot fixture run.
  SGLANG_DS_RADIX_OVERRIDE=1 \\
    SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \\
    bash development/serve_double_sparsity.sh
  # 1. Smoke (this file).
  DS_BASE_URL=http://localhost:30000 \\
    pytest test/manual/test_dsv32_radix_cache_fixture.py -v
  # 2. Full M3-B label-capture fixture.
  DS_BASE_URL=http://localhost:30000 \\
    SGLANG_DS_RADIX_FIXTURE_CAPTURE=1 \\
    pytest test/manual/test_dsv32_radix_label_capture_fixture.py -v
  # 3. FP8 scale-stability proof.
  pytest test/manual/test_dsv32_fp8_scale_stability.py -v
  # 4. On all three passing, the launcher calls
  #    record_radix_fixture_passed(server_args, artifact_path=...)
  #    BEFORE validate_double_sparsity runs (boots the post-AC-10
  #    state without SGLANG_DS_RADIX_OVERRIDE).
  # 5. Remove --disable-radix-cache from serve_double_sparsity.sh
  #    (AC-10-FIXTURE-MARKER comment points at the line).
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
from typing import Any, Dict, Optional


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


def _generate(base_url: str, prompt: str, *, max_new_tokens: int = 64) -> str:
    body = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
        },
    }
    out = _post_json(f"{base_url.rstrip('/')}/generate", body, timeout=600.0)
    text = out.get("text")
    if text is None and "choices" in out:
        text = out["choices"][0].get("text", "")
    return text or ""


def _get_server_info(base_url: str) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/get_server_info"
    try:
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError):
        return {}


def _flush_cache(base_url: str) -> None:
    """Best-effort cache flush so the cold side really starts cold."""
    try:
        _post_json(f"{base_url.rstrip('/')}/flush_cache", {}, timeout=30.0)
    except urllib.error.URLError:
        pass


def _record_artifact(payload: Dict[str, Any], *, suffix: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ",
    )
    path = RESULTS_DIR / f"dsv32_radix_fixture_{suffix}_{ts}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[radix-fixture] artifact written: {path}")


# A shared-prefix payload long enough to allocate multiple physical KV
# slots and to make a continuation divergence detectable at
# temperature=0. The prompt is IDENTICAL across cold and warm passes —
# the run/pass distinction lives in the artifact metadata only.
# Otherwise the radix cache cannot reuse slots, and the comparison
# conflates prompt change with label change.
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


def _local_commit_sha() -> Optional[str]:
    """Local repo HEAD SHA. Best-effort; None on any failure."""
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


@unittest.skipUnless(
    _env("DS_BASE_URL"),
    "DS_BASE_URL env var must point at a running DS server with radix "
    "cache enabled (this fixture is operator-driven against H200).",
)
class TestDSv32RadixCacheContinuationSmoke(unittest.TestCase):
    """Cold-prefix vs warm-prefix continuation-equality SMOKE fixture.

    A passing run is NECESSARY but NOT SUFFICIENT for the AC-10 guard
    flip. The proper M3-B evidence (per-layer label SHA bit-equality
    via the in-process capture log + FP8 block scale-factor equality)
    lives in the two companion fixtures named in the module docstring.

    Failure here means cold and warm continuations diverge under
    temperature=0 — DS labels for the shared-prefix slots almost
    certainly diverge under radix-cache reuse. DEC-2 guard MUST
    remain in place; do not run the M3-B label-capture fixture until
    this smoke passes.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.server_info = _get_server_info(cls.ds_url)
        cls.local_commit_sha = _local_commit_sha()
        # The fixture's whole point is to verify the radix-on path; if
        # the server is still running with --disable-radix-cache, the
        # warm pass would also allocate fresh slots and the test would
        # not actually exercise the radix-cache reuse path.
        sa = cls.server_info.get("server_args") or {}
        if isinstance(sa, dict) and sa.get("disable_radix_cache") is True:
            raise unittest.SkipTest(
                "DS server reports disable_radix_cache=True; the AC-10 "
                "fixture requires radix cache ENABLED (this is the "
                "configuration the fixture is gating). Re-launch the "
                "server without --disable-radix-cache (use "
                "SGLANG_DS_RADIX_OVERRIDE=1 for this one-shot run)."
            )

    def test_cold_warm_continuation_smoke(self):
        run_id = uuid.uuid4().hex[:12]
        prompt = _SHARED_PREFIX_PROMPT  # identical for both passes

        # Best-effort flush so any prior request's KV state does not
        # warm-prime the shared-prefix slots before the cold pass.
        _flush_cache(self.ds_url)

        cold_text = _generate(self.ds_url, prompt, max_new_tokens=128)
        # No flush between requests — the radix cache must retain the
        # shared-prefix slots so the warm pass exercises the reuse path.
        warm_text = _generate(self.ds_url, prompt, max_new_tokens=128)

        equal = cold_text == warm_text
        sa = (self.server_info.get("server_args")
              if isinstance(self.server_info, dict) else None)
        # Record the SERVER's commit SHA when /get_server_info exposes
        # it (added in a future round); always record the local repo
        # SHA as well so the operator can confirm the deployed image
        # matches their checkout.
        server_commit = None
        if isinstance(sa, dict):
            server_commit = sa.get("commit_sha")
        payload = {
            "fixture_kind": "continuation_smoke",
            "fixture_caveat": (
                "NECESSARY-BUT-NOT-SUFFICIENT pre-flight; the AC-10 "
                "guard flip requires the M3-B label-capture fixture "
                "AND the FP8 scale-stability fixture to also pass."
            ),
            "run_id": run_id,
            "ds_base_url": self.ds_url,
            "local_commit_sha": self.local_commit_sha,
            "server_commit_sha": server_commit,
            "server_args": sa,
            "prompt": prompt,
            "cold_text": cold_text,
            "warm_text": warm_text,
            "continuation_bit_equal": equal,
        }
        _record_artifact(payload, suffix="continuation_smoke")

        self.assertTrue(
            equal,
            "AC-10 continuation-smoke FAILED: cold and warm "
            "continuations diverge under temperature=0 with identical "
            "prompts. DS labels for the shared-prefix slots almost "
            "certainly diverge across the radix-cache reuse path. Keep "
            "--disable-radix-cache in the DS launcher and do NOT run "
            "the M3-B label-capture fixture. Diagnostic artifact "
            f"written under {RESULTS_DIR}/."
        )


if __name__ == "__main__":
    unittest.main()
