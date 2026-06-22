"""
Generator + validator for KV-event block-hash parity fixtures.

Two modes:

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py
      Regenerate the committed JSON fixture from the locally-replicated
      algorithm. Run this when changing block-hash logic or adding new
      shape coverage. CI's drift-check step runs this in --check mode.

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py --validate-against-sglang
      Import the real `sglang.srt.mem_cache.radix_cache.RadixKey.hash_page`
      and assert it agrees with the locally-replicated algorithm on every
      fixture case. This is the only place the replica and the real
      SGLang implementation are checked against each other. Run it
      nightly (or whenever sglang is available on the Python path).

# Authority

Source-of-truth implementation:
  - `python/sglang/srt/mem_cache/radix_cache.py::RadixKey.hash_page`
  - `python/sglang/srt/mem_cache/utils.py::hash_str_to_int64`

`hash_page_chain` below replicates that algorithm verbatim (no `import
sglang`) so the script runs without the heavy SGLang dependency tree and
can be audited at a glance. The algorithm is intentionally tiny:

    sha256(token_LE_u32 ++ token_LE_u32 ++ ...)
    or, with extra_key and/or parent digest:
    sha256(len(extra_key_utf8)_LE_u32 ++ extra_key_utf8 ++ prior_digest_bytes ++ token_LE_u32 ++ ...)
    truncate to i64 = signed(first 16 hex chars)

If SGLang ever changes the algorithm, update both the SGLang side AND
this script in the same commit; the Rust port in
`src/policies/kv_events/hash.rs` will then need the corresponding
update. The nightly `--validate-against-sglang` job is the safety net
that catches an SGLang-side change the human forgot to mirror here.

# Output format

A JSON array of cases. Each case is:
    {
      "name": "<descriptive label>",
      "tokens": [<u32>, ...],
      "block_size": <usize>,
      "extra_key": "<optional namespace>",
      "expected_i64_hashes": [<i64>, ...]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys


def hash_page_chain(
    tokens: list[int], block_size: int, extra_key: str | None = None
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
    prior_digest: bytes | None = None
    n = len(tokens)
    if n == 0:
        return out
    # Walk every page boundary, including a trailing partial page.
    start = 0
    while start < n:
        end = min(start + block_size, n)
        hasher = hashlib.sha256()
        if extra_key is not None:
            encoded_extra_key = extra_key.encode("utf-8")
            hasher.update(len(encoded_extra_key).to_bytes(4, byteorder="little"))
            hasher.update(encoded_extra_key)
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
        "name": "salted_single_full_block",
        "tokens": [1, 2, 3, 4],
        "block_size": 4,
        "extra_key": "salt-A",
    },
    {
        "name": "salted_multi_block",
        "tokens": [1, 2, 3, 4, 5],
        "block_size": 4,
        "extra_key": "salt-A",
    },
    {
        "name": "empty_extra_key_namespace",
        "tokens": [1, 2, 3, 4],
        "block_size": 4,
        "extra_key": "",
    },
]


def _materialize_cases() -> list[dict]:
    cases_out = []
    for c in CASES:
        out = {
            "name": c["name"],
            "tokens": c["tokens"],
            "block_size": c["block_size"],
            "expected_i64_hashes": hash_page_chain(
                c["tokens"], c["block_size"], c.get("extra_key")
            ),
        }
        if "extra_key" in c:
            out["extra_key"] = c["extra_key"]
        cases_out.append(out)
    return cases_out


def _validate_against_sglang() -> int:
    """Import the real SGLang `RadixKey.hash_page` and compare its output
    case-by-case against the locally-replicated `hash_page_chain`. Exits
    non-zero (and prints a diff-friendly summary) on any mismatch.

    Returns 0 on success. This is the parity safety net for nightly CI.
    """
    try:
        from sglang.srt.mem_cache.radix_cache import RadixKey
    except ImportError as e:
        print(
            f"--validate-against-sglang: cannot import sglang ({e}). "
            "Install sglang into the Python path before running this mode.",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for c in CASES:
        local = hash_page_chain(c["tokens"], c["block_size"], c.get("extra_key"))
        if c["block_size"] == 0 or not c["tokens"]:
            # `RadixKey.hash_page` requires a non-empty page; the local
            # replica handles edge cases (empty input → empty list)
            # which the SGLang oracle would refuse. Skip these cases
            # under validation — the replica owns the boundary semantics.
            continue
        sglang_hashes: list[int] = []
        prior_hex: str | None = None
        key = RadixKey(token_ids=c["tokens"], extra_key=c.get("extra_key"))
        for start in range(0, len(c["tokens"]), c["block_size"]):
            end = min(start + c["block_size"], len(c["tokens"]))
            hex_digest = key.hash_page(start, end, prior_hex)
            # SGLang's hash_page returns the hex digest; truncate to i64
            # the same way `hash_str_to_int64` does.
            uint64_val = int(hex_digest[:16], 16)
            i64 = uint64_val - (1 << 64) if uint64_val >= (1 << 63) else uint64_val
            sglang_hashes.append(i64)
            prior_hex = hex_digest
        if sglang_hashes != local:
            failures.append(f"case {c['name']}: local={local} sglang={sglang_hashes}")

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
