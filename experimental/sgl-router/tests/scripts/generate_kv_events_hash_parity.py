"""
Generator + validator for KV-event block-hash parity fixtures.

Two modes:

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py
      Regenerate the committed JSON fixture from the locally-replicated
      algorithm. Run this when changing block-hash logic or adding new
      shape coverage. CI regenerates and diffs the fixture.

  python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py --validate-against-sglang
      Import the real `compute_node_event_hash_values`
      and assert it agrees with the locally-replicated algorithm on every
      fixture case. This is the only place the replica and the real
      SGLang implementation are checked against each other. Run it
      nightly (or whenever sglang is available on the Python path).

# Authority

Source-of-truth implementation:
  - `python/sglang/srt/mem_cache/utils.py::compute_node_event_hash_values`
  - `python/sglang/srt/mem_cache/cpp_utils/hash_binding.cpp`
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
    tokens: list[int],
    block_size: int,
    cache_salt: str | None = None,
    bigram: bool = False,
) -> list[int]:
    """Compute the i64-truncated block hashes for `tokens` using SGLang's
    event hash algorithm + `hash_str_to_int64`.

    Returns one i64 per full or partial block.  A partial last block (when
    `len(tokens) % block_size != 0`) chains against the previous block's
    full 32-byte SHA256 digest, matching SGLang's behaviour.
    """
    if block_size == 0:
        raise ValueError("block_size must be positive")

    out: list[int] = []
    prior_digest: bytes | None = None
    if cache_salt:
        prior_digest = hashlib.sha256(
            b"sglang-cache-salt-v1\0" + cache_salt.encode("utf-8")
        ).digest()
    n = max(0, len(tokens) - int(bigram))
    if n == 0:
        return out
    # Walk every page boundary, including a trailing partial page.
    start = 0
    while start < n:
        end = min(start + block_size, n)
        hasher = hashlib.sha256()
        if prior_digest is not None:
            hasher.update(prior_digest)
        for i in range(start, end):
            hasher.update(tokens[i].to_bytes(4, byteorder="little", signed=False))
            if bigram:
                hasher.update(
                    tokens[i + 1].to_bytes(4, byteorder="little", signed=False)
                )
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
]

# Keep the original unsalted cases unchanged; exercise salt normalization,
# UTF-8, namespace isolation, signed truncation and partial pages in both modes.
for salt in (None, "", "tenant-a", "tenant-b", "租户-A", "a\0b"):
    for bigram in (False, True):
        CASES.append(
            {
                "name": f"namespace_{salt!r}_bigram_{bigram}",
                "tokens": [0, 1, 2**32 - 1, 3, 4, 5],
                "block_size": 2,
                "cache_salt": salt,
                "bigram": bigram,
            }
        )
for tokens in ([], [1]):
    for bigram in (False, True):
        CASES.append(
            {
                "name": f"salted_short_{len(tokens)}_bigram_{bigram}",
                "tokens": tokens,
                "block_size": 1,
                "cache_salt": "tenant-a",
                "bigram": bigram,
            }
        )


def _materialize_cases() -> list[dict]:
    return [
        {
            **c,
            "expected_i64_hashes": hash_page_chain(
                c["tokens"],
                c["block_size"],
                c.get("cache_salt"),
                c.get("bigram", False),
            ),
        }
        for c in CASES
    ]


def _validate_against_sglang() -> int:
    """Import the real SGLang event hashing path and compare its output
    case-by-case against the locally-replicated `hash_page_chain`. Exits
    non-zero (and prints a diff-friendly summary) on any mismatch.

    Returns 0 on success. This is the parity safety net for nightly CI.
    """
    try:
        from array import array
        from types import SimpleNamespace

        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.utils import (
            compute_node_event_hash_values,
            hash_str_to_int64,
        )
    except ImportError as e:
        print(
            f"--validate-against-sglang: cannot import sglang ({e}). "
            "Install sglang into the Python path before running this mode.",
            file=sys.stderr,
        )
        return 2

    failures: list[str] = []
    for c in CASES:
        local = hash_page_chain(
            c["tokens"], c["block_size"], c.get("cache_salt"), c.get("bigram", False)
        )
        key = RadixKey(
            array("I", c["tokens"]),
            cache_salt=c.get("cache_salt"),
            is_bigram=c.get("bigram", False),
        )
        if len(key) == 0:
            # Empty radix nodes do not emit stored blocks.
            assert local == []
            continue
        node = SimpleNamespace(key=key, parent=None, event_hash_value=None)
        sglang_hashes = [
            hash_str_to_int64(h)
            for h in compute_node_event_hash_values(node, c["block_size"])
        ]
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
