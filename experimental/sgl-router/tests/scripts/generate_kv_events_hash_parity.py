"""
One-shot generator for KV-event block-hash parity fixtures.

Run manually when changing the block-hash algorithm or adding new
shape coverage:

    python3 experimental/sgl-router/tests/scripts/generate_kv_events_hash_parity.py

CI does NOT run this — it consumes the committed JSON.  A nightly job
SHOULD re-run it and diff against the committed file to catch drift on
either side; that wiring lives in M3 Task 7.

# Authority

Source-of-truth implementation:
  - `python/sglang/srt/mem_cache/radix_cache.py::RadixKey.hash_page`
  - `python/sglang/srt/mem_cache/utils.py::hash_str_to_int64`

The function below replicates that algorithm verbatim (no `import sglang`)
so the script runs without the heavy SGLang dependency tree and can be
audited at a glance.  The algorithm is intentionally tiny:

    sha256(prior_digest_bytes ++ token_LE_u32 ++ token_LE_u32 ++ ...)
    truncate to i64 = signed(first 16 hex chars)

If SGLang ever changes the algorithm, update both the SGLang side AND this
script in the same commit; the Rust port in `src/policies/kv_events/hash.rs`
will then need the corresponding update.

# Output format

A JSON array of cases.  Each case is:
    {
      "name": "<descriptive label>",
      "tokens": [<u32>, ...],
      "block_size": <usize>,
      "expected_i64_hashes": [<i64>, ...]
    }
"""

from __future__ import annotations

import hashlib
import json
import pathlib


def hash_page_chain(tokens: list[int], block_size: int) -> list[int]:
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
]


def main() -> int:
    cases_out = []
    for c in CASES:
        cases_out.append(
            {
                "name": c["name"],
                "tokens": c["tokens"],
                "block_size": c["block_size"],
                "expected_i64_hashes": hash_page_chain(c["tokens"], c["block_size"]),
            }
        )

    out_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "fixtures"
        / "kv_events_hash_parity.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(cases_out, f, indent=2)
        f.write("\n")
    print(f"wrote {len(cases_out)} cases to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
