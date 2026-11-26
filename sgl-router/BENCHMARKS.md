# sgl-router microbench harness

This file pairs with `sgl-router/benches/`.

## Scope

These are CPU-bound microbenches that don't need GPUs — they target
routing-decision latency only. The full E2E throughput measurement
(genai-bench at 4×H200 against a real SGLang fleet) is **not** part of
this file; it requires a real GPU cluster and is tracked separately.

## How to run

```bash
cd sgl-router
cargo bench --bench tree_lookup     -- --sample-size 30 --measurement-time 3
cargo bench --bench policy_select   -- --sample-size 30 --measurement-time 3
```

For the quick runs whose numbers are reproduced below: drop
`--sample-size` to 10 and `--measurement-time` to 2 (Criterion will
warn about reduced statistical confidence but the order-of-magnitude
reading stands).

## Quick-run Data Points (M1 MacBook, release profile)

These are NOT acceptance numbers — they're a sanity check on the order
of magnitude of the routing primitives. Real targets come from
cluster-scale runs and are tracked separately.

### Cache-aware lookup (`HashTree`)

| Bench | sgl-router | Notes |
|---|---|---|
| Insert for 1 worker (medium case) | `hashtree_insert/128` ≈ 21.5 µs | Inserts 128 pre-hashed `i64` block-hashes. The hashing step (`compute_block_hashes`) is upstream of `HashTree` and not measured here. |
| Match request prefix | `hashtree_match_prefix/w64_bpw128_q64` ≈ 47 ns | A short-circuit walk over `i64` hashes. A request's full cost includes `compute_block_hashes` (~ tens of µs depending on prompt length). |

**Read carefully.** The match step is not the end-to-end routing cost
an operator should expect — `compute_block_hashes` upstream dominates
in real traffic.

### Policy selection (non-cache-aware)

| Policy | n=4 workers | n=16 | n=64 | n=256 | Shape |
|---|---|---|---|---|---|
| `round_robin`     | 2.5 ns | 2.5 ns | 2.5 ns | 2.5 ns | O(1). |
| `random`          | — | — | — | — | `SliceRandom::choose` call — O(1). |
| `power_of_two`    | — | — | — | — | Two distinct indices sampled directly - O(1) (2× rand + 2× load read). |

Both `random` and `power_of_two` are now O(1), ensuring consistent
performance regardless of worker count.
TODO: Add a regression guard for the O(n) shape. Although `policy_select`
measures the metric, nothing currently runs or gates on these results.
