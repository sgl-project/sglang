"""
Generator + validator for KV-event block-hash parity fixtures.

Two modes:

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py
      Regenerate the committed JSON fixture from the locally-replicated
      algorithm. Run this when changing block-hash logic or adding new
      shape coverage. CI regenerates the file and checks its diff.

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py --validate-against-sglang
      Render through SGLang's real KV-event recorder, check its opaque LoRA
      IDs, and compare canonical lookup keys with every fixture case.
      Run this whenever SGLang is available on the Python path.

# Authority

Source-of-truth implementation:
  - `python/sglang/srt/mem_cache/utils.py::get_hash_str`
  - `python/sglang/srt/mem_cache/utils.py::hash_str_to_int64`

`hash_page_chain` below replicates that algorithm verbatim (no `import
sglang`) so the script runs without the heavy SGLang dependency tree and
can be audited at a glance. The algorithm is intentionally tiny:

    sha256(prior_digest_bytes ++ token_LE_u32 ++ token_LE_u32 ++ ...)
    truncate to i64 = signed(first 16 hex chars)

If SGLang ever changes the algorithm, update both the SGLang side AND
this script in the same commit; the Rust port in
`src/state/kv_events/hash.rs` will then need the corresponding
update. The nightly `--validate-against-sglang` job is the safety net
that catches an SGLang-side change the human forgot to mirror here.

# Output format

A JSON array of cases. Each case is:
    {
      "name": "<descriptive label>",
      "tokens": [<u32>, ...],
      "block_size": <usize>,
      "cache_salt": <str>,   # optional
      "lora_name": <str>,    # optional
      "expected_i64_hashes": [<i64>, ...]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def namespaced_hashes(
    tokens: list[int], block_size: int, cache_salt=None, lora_name=None
) -> list[int]:
    """Canonical lookup keys: salt-seeded token chain plus public adapter name.

    LoRA worker block IDs are opaque and are translated during event ingestion.
    """
    prior = None
    if cache_salt is not None:
        prior = hashlib.sha256(b"sglang-cache-salt-v1\0" + cache_salt.encode()).digest()
    hashes = hash_page_chain(tokens, block_size, prior)
    if lora_name is None:
        return hashes
    seed = hashlib.sha256(b"sglang-kv-event-lora-v1\0" + lora_name.encode()).digest()
    return [
        int.from_bytes(
            hashlib.sha256(seed + h.to_bytes(8, "big", signed=True)).digest()[:8],
            "big",
            signed=True,
        )
        for h in hashes
    ]


def hash_page_chain(
    tokens: list[int], block_size: int, prior_digest: bytes | None = None
) -> list[int]:
    """Compute the i64-truncated block hashes for `tokens` using SGLang's
    `RadixKey.hash_page` algorithm + `hash_str_to_int64`.

    Returns one i64 per full or partial block.  A partial last block (when
    `len(tokens) % block_size != 0`) chains against the previous block's
    full 32-byte SHA256 digest, matching SGLang's behaviour.
    """
    if block_size == 0:
        raise ValueError("block_size must be positive")

    out: list[int] = []
    n = len(tokens)
    if n == 0:
        return out
    # Walk every page boundary, including a trailing partial page.
    start = 0
    while start < n:
        end = min(start + block_size, n)
        hasher = hashlib.sha256()
        if prior_digest is not None:
            hasher.update(prior_digest)
        for t in tokens[start:end]:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))
        digest = hasher.digest()
        prior_digest = digest
        # hash_str_to_int64: first 16 hex chars (top 64 bits) -> signed i64.
        hex_digest = digest.hex()
        uint64_val = int(hex_digest[:16], 16)
        if uint64_val >= 2**63:
            i64 = uint64_val - 2**64
        else:
            i64 = uint64_val
        out.append(i64)
        start = end
    return out


# Cases mirror the three existing `cross_language_golden_*` tests plus
# additional shape coverage that exercises (a) zero-token edge, (b)
# block_size = 1, (c) very long sequences, (d) odd boundaries.
CASES: list[dict] = [
    {
        "name": "single_full_block",
        "tokens": [1, 2, 3, 4],
        "block_size": 4,
    },
    {
        "name": "partial_last_block",
        "tokens": [1, 2, 3, 4, 5],
        "block_size": 4,
    },
    {
        "name": "multi_block",
        "tokens": [10, 20, 30, 40, 50, 60, 70, 80],
        "block_size": 2,
    },
    {
        "name": "empty_tokens",
        "tokens": [],
        "block_size": 4,
    },
    {
        "name": "block_size_one",
        "tokens": [7, 8, 9],
        "block_size": 1,
    },
    {
        "name": "odd_boundary",
        "tokens": [100, 200, 300, 400, 500, 600, 700],
        "block_size": 3,
    },
    {
        "name": "long_sequence",
        # 128 tokens at block_size 16 → 8 blocks exactly.
        "tokens": list(range(1, 129)),
        "block_size": 16,
    },
    {
        "name": "salted",
        "tokens": [1, 2, 3, 4],
        "block_size": 2,
        "cache_salt": "tenant-a",
    },
    {"name": "lora", "tokens": [1, 2, 3, 4], "block_size": 2, "lora_name": "adapter-a"},
    {
        "name": "salted_lora",
        "tokens": [1, 2, 3, 4],
        "block_size": 2,
        "cache_salt": "tenant-a",
        "lora_name": "adapter-a",
    },
]
NAMESPACE_KEYS = ("cache_salt", "lora_name")


def _materialize_cases() -> list[dict]:
    return [
        {
            **c,
            "expected_i64_hashes": namespaced_hashes(
                c["tokens"], c["block_size"], *(c.get(k) for k in NAMESPACE_KEYS)
            ),
        }
        for c in CASES
    ]


def _lookup_keys_from_sglang(case: dict) -> list[int]:
    """Check opaque LoRA IDs, then derive lookup keys from the base event chain."""
    from array import array

    from sglang.srt.disaggregation.kv_events import (
        BlockStored,
        kv_event_namespace_seed,
        namespaced_block_hash,
    )
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey

    class LoraReq:
        pass

    def publish(name=None):
        cache = RadixCache.create_simulated(
            page_size=case["block_size"], enable_kv_cache_events=True
        )
        req = LoraReq()
        req.extra_key = req.lora_id = "lora-id" if name else None
        cache.kv_events.lora_names.register(req, name)
        cache.insert(
            InsertParams(
                key=RadixKey(
                    array("q", case["tokens"]),
                    extra_key=req.extra_key,
                    cache_salt=case.get("cache_salt"),
                )
            )
        )
        return [e for e in cache.take_events() if isinstance(e, BlockStored)]

    base = [h for e in publish() for h in e.block_hashes]
    name = case.get("lora_name")
    if name is None:
        return base
    events = publish(name)
    seed = kv_event_namespace_seed(extra_key="lora-id", cache_salt=None)
    assert [h for e in events for h in e.block_hashes] == [
        namespaced_block_hash(h, namespace_seed=seed) for h in base
    ]
    assert all(e.lora_name == name for e in events)
    seed = hashlib.sha256(b"sglang-kv-event-lora-v1\0" + name.encode()).digest()
    return [
        int.from_bytes(
            hashlib.sha256(seed + h.to_bytes(8, "big", signed=True)).digest()[:8],
            "big",
            signed=True,
        )
        for h in base
    ]


def _validate_against_sglang() -> int:
    """Check the engine's published IDs and the canonical prompt lookup keys."""
    try:
        import sglang  # noqa: F401
    except ImportError as e:
        print(
            f"--validate-against-sglang: cannot import sglang ({e}). "
            "Install sglang into the Python path before running this mode.",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for c in CASES:
        if not c["tokens"]:
            continue
        local = namespaced_hashes(
            c["tokens"], c["block_size"], *(c.get(k) for k in NAMESPACE_KEYS)
        )
        # The engine publishes full pages only, so it must match a prefix.
        published = _lookup_keys_from_sglang(c)
        if not published or published != local[: len(published)]:
            failures.append(f"case {c['name']}: local={local} sglang={published}")

    if failures:
        print(
            "--validate-against-sglang: replica/SGLang DRIFT detected:",
            file=sys.stderr,
        )
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"--validate-against-sglang: OK ({len(CASES)} cases agreed)")
    return 0


def _write_fixture(cases_out: list[dict]) -> pathlib.Path:
    out_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "kv_events_hash_parity.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cases_out, f, indent=2, sort_keys=False)
        f.write("\n")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-against-sglang",
        action="store_true",
        help="Compare the local replica to the imported SGLang implementation "
        "and exit non-zero on drift. Requires sglang on the Python path.",
    )
    args = parser.parse_args()

    if args.validate_against_sglang:
        return _validate_against_sglang()

    cases_out = _materialize_cases()
    out_path = _write_fixture(cases_out)
    print(f"wrote {len(cases_out)} cases to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
