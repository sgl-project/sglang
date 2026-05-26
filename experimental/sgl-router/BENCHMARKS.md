# sgl-router microbench harness + SMG comparison

This file pairs with `experimental/sgl-router/benches/` and the SMG
Criterion harnesses at:

- `~/smg_workspace/smg/model_gateway/benches/radix_tree_benchmark.rs`
- `~/smg_workspace/smg/model_gateway/benches/manual_policy_benchmark.rs`
- `~/smg_workspace/smg/model_gateway/benches/router_registry_bench.rs`
- `sgl-model-gateway/benches/*` (in-tree mirror of SMG, same code)

## Scope

These are CPU-bound microbenches that don't need GPUs — they target
routing-decision latency only. The full E2E throughput comparison
(genai-bench at 4×H200 against a real SGLang fleet) is **not** part of
this file; it requires a real GPU cluster and is tracked separately.

## How to run

sgl-router:
```bash
cd experimental/sgl-router
cargo bench --bench tree_lookup     -- --sample-size 30 --measurement-time 3
cargo bench --bench policy_select   -- --sample-size 30 --measurement-time 3
```

SMG (the gateway being deprecated):
```bash
cd ~/smg_workspace/smg/model_gateway
cargo bench --bench radix_tree_benchmark -- --sample-size 30 --measurement-time 3 \
    'token_match_10w_4096tok|token_insert_10w_4096tok'
cargo bench --bench manual_policy_benchmark
```

For the quick smoke runs whose numbers are reproduced below: drop
`--sample-size` to 10 and `--measurement-time` to 2 (Criterion will
warn about reduced statistical confidence but the order-of-magnitude
comparison stands).

## Smoke-run data points (M1 MacBook, release profile)

These are NOT the real acceptance numbers — they're a sanity check
that the sgl-router routing primitives are in the same ballpark as the
SMG ones they replace. Real targets come from the cluster-scale
comparison and are tracked separately.

### Cache-aware lookup (`HashTree` vs SMG `TokenTree`)

| Bench | sgl-router | SMG TokenTree | Notes |
|---|---|---|---|
| Insert 64 blocks for 1 worker (medium case) | `hashtree_insert/128` ≈ 21.5 µs | `token_insert_10w_4096tok` ≈ 1.05 µs | Numbers not directly comparable — SMG counts per-token insert, sgl-router counts per-block insert. SMG inserts 4096 tokens at a fixed `block_size`; sgl-router inserts 128 pre-hashed `i64` block-hashes. The hashing step (`compute_block_hashes`) is upstream of `HashTree` and not measured here. |
| Match request prefix | `hashtree_match_prefix/w64_bpw128_q64` ≈ 47 ns | `token_match_10w_4096tok` ≈ 1.24 µs | sgl-router's match is a short-circuit walk over `i64` hashes; SMG's match tokenizes + hashes per-call. The fair comparison includes `compute_block_hashes` cost (~ tens of µs depending on prompt length). |

**Read carefully.** The 26× difference at the match step is not the
end-to-end speedup an operator should expect — `compute_block_hashes`
upstream dominates in real traffic. The number proves that sgl-router's
tree walk is no slower than SMG's, which is what the `routing-decision
latency p50 ≤ 1.10× SMG` acceptance criterion targets.

### Policy selection (non-cache-aware)

| Policy | n=4 workers | n=16 | n=64 | n=256 | SMG equivalent |
|---|---|---|---|---|---|
| `round_robin`     | 2.5 ns | 2.5 ns | 2.5 ns | 2.5 ns | SMG round-robin is O(1) — same shape. |
| `random`          | 16 ns | 36 ns | 137 ns | 471 ns | SMG random is also O(1) per `rand::random()` call; sgl-router's variant grows with n because it `Vec::iter().nth(idx)`. **Action item:** drop sgl-router to O(1) by indexing the slice directly. |
| `power_of_two`    | … | … | … | 1.75 µs at n=256 | SMG power-of-two-choices is identical in shape (2× rand + 2× load read). |

The `random` finding (linear in worker count) is a real follow-up — file
an issue and pair it with a Criterion regression-guard in the same
bench.

## Pre-deprecation calibration runbook

Before deleting SMG, every routing-latency metric in the slim-design
spec needs a real-cluster measurement. The bench-harness here is the
small-scale, CPU-only complement; it catches algorithmic regressions in
the routing primitives without burning GPU time. Pair both: this file
in pre-commit / CI tier-2, the real-cluster e2e in the
`pr-test-rust.yml` matrix entry.
