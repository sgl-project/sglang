"""AC-10 M3-B radix-cache stability fixture — manual hardware harness.

Plan §AC-10 / DEC-2 gates Double Sparsity from running with the radix
cache enabled until a hardware fixture has proven that cold-prefix and
warm-prefix labels are bit-stable for the production model + channel
mask + FP8 KV configuration.

This harness is the operator-runnable side of that gate. It issues a
pair of requests against a DS-on server with radix cache ENABLED:

* **Cold request** — a unique shared prefix that the server has never
  seen, so the KV slots for the shared prefix are freshly allocated
  and the DS label-write path produces labels from a single-pass FP8
  dequant of the just-written K-noPE.
* **Warm request** — the same shared prefix re-sent immediately after
  the cold request. With radix cache ON, the server reuses the
  shared-prefix KV slots; the DS label table must therefore yield
  identical labels for the reused slots.

At temperature=0 with `max_new_tokens` long enough to exercise multiple
decode steps, divergent labels would manifest as divergent
continuations. Equal continuations are the operator-observable proxy
for label bit-stability (the unit-level proof of the labeling code's
determinism given identical K-noPE lives in
``test/registered/unit/layers/attention/test_double_sparsity_unit.py``
under ``TestAC10RadixCacheLabelBitStability``).

Operator runbook:

  # 1. Boot the DS server with radix cache ON (remove
  #    --disable-radix-cache from serve_double_sparsity.sh and set
  #    SGLANG_DS_RADIX_OVERRIDE=1 to bypass the boot guard for this
  #    one-shot fixture run).
  SGLANG_DS_RADIX_OVERRIDE=1 bash development/serve_double_sparsity.sh
  # 2. Run this fixture.
  DS_BASE_URL=http://localhost:30000 \\
    pytest test/manual/test_dsv32_radix_cache_fixture.py -v
  # 3. On pass, the artifact at
  #    development/results/dsv32_radix_fixture_<ts>.json records
  #    pass/fail + payloads. After verifying the artifact, the
  #    operator removes --disable-radix-cache from the launcher
  #    (marker comment names the exact edit point) and the launcher
  #    invokes record_radix_fixture_passed(server_args) so future
  #    boots no longer need SGLANG_DS_RADIX_OVERRIDE.
"""

from __future__ import annotations

import datetime
import json
import os
import pathlib
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
# temperature=0. The trailing unique suffix prevents the *outer* cache
# (radix or KV) from auto-matching this exact prompt across runs; the
# shared prefix portion is what the warm pass re-uses.
_SHARED_PREFIX_TEMPLATE = (
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
    "  under Double Sparsity.\n\n"
    "Continue (run-id {run_id}, pass-id {pass_id}):"
)


@unittest.skipUnless(
    _env("DS_BASE_URL"),
    "DS_BASE_URL env var must point at a running DS server with radix "
    "cache enabled (this fixture is operator-driven against H200).",
)
class TestDSv32RadixCacheStability(unittest.TestCase):
    """Cold-prefix vs warm-prefix continuation equality fixture.

    Pass = continuations are identical at temperature=0 ⇒ DS label rows
    for the shared-prefix slots are bit-equal between the freshly
    allocated (cold) and the radix-cache-reused (warm) paths.
    Fail = continuations diverge ⇒ DEC-2 guard MUST remain in place;
    do not remove `--disable-radix-cache` from the DS launcher.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds_url = _env("DS_BASE_URL")
        cls.server_info = _get_server_info(cls.ds_url)
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

    def test_cold_then_warm_continuation_is_bit_equal(self):
        run_id = uuid.uuid4().hex[:12]
        cold_prompt = _SHARED_PREFIX_TEMPLATE.format(
            run_id=run_id, pass_id="cold",
        )
        warm_prompt = _SHARED_PREFIX_TEMPLATE.format(
            run_id=run_id, pass_id="warm",
        )

        # Best-effort flush so any prior request's KV state does not
        # warm-prime the shared-prefix slots before the cold pass.
        _flush_cache(self.ds_url)

        cold_text = _generate(
            self.ds_url, cold_prompt, max_new_tokens=128,
        )
        # No flush between requests — the radix cache must retain the
        # shared-prefix slots so the warm pass exercises the reuse path.
        warm_text = _generate(
            self.ds_url, warm_prompt, max_new_tokens=128,
        )

        equal = cold_text == warm_text
        payload = {
            "run_id": run_id,
            "ds_base_url": self.ds_url,
            "server_args": (self.server_info.get("server_args")
                            if isinstance(self.server_info, dict) else None),
            "cold_prompt": cold_prompt,
            "warm_prompt": warm_prompt,
            "cold_text": cold_text,
            "warm_text": warm_text,
            "bit_equal": equal,
        }
        _record_artifact(payload, suffix="cold_warm")

        self.assertTrue(
            equal,
            "AC-10 radix-cache stability fixture FAILED: cold and warm "
            "continuations diverge under temperature=0. DS label rows "
            "for the shared-prefix slots are NOT bit-stable across the "
            "radix-cache reuse path. Keep --disable-radix-cache in the "
            "DS launcher and do NOT call record_radix_fixture_passed. "
            f"Diagnostic artifact written under {RESULTS_DIR}/.",
        )


if __name__ == "__main__":
    unittest.main()
