# RFC 0001: `sglang-radix` — a pure-Rust radix tree for SGLang's prefix cache

**Status:** draft (from branch
[`rust-scheduler`](https://github.com/sorenmat/sglang/tree/rust-scheduler),
where it ships behind `SGLANG_RUST_SCHEDULER=radix` and is verified by
differential parity + criterion gates)
**Contribution order:** first of the two Rust crates to upstream (pure
library, no engine coupling — the easiest review).

## Background

SGLang's prefix cache (`python/sglang/srt/mem_cache/radix_cache.py`,
`RadixCache`) is on every request's critical path: `match_prefix` at
admission, `inc/dec_lock_ref` parent walks around every KV run, `insert`
at finish/chunk-stash, `evict(n)` under memory pressure, plus the policy
walks (`calc_priority` for fcfs/lpm/dfs-weight). Upstream profiling puts
the already-optimized Python at ~3.9 µs for a 128-req filter pass and
~6.7 µs for finish-state processing; tree ops sit in the same budget.

The tree has also grown variants: sliding-window (SWA) eviction,
mamba/GDN hybrid states, the HiRadix host tier, and a unified multi-pool
tree (`CT_FULL` / `CT_SWA` / `CT_MAMBA` / `CT_C128`) — each a divergence
point for a second native implementation. A C++ variant
(`RadixCacheCpp`) exists upstream; on this branch it is frozen as a test
oracle while the Rust tree supersedes it.

## Proposal

Ship `sglang-radix` as a **pure Rust crate** (no pyo3, no torch, no I/O):
the tree + key model + eviction policies, with one optional `python`
feature that builds the PyO3 facades. SGLang consumes it through a thin
Python facade (`sglang.srt.mem_cache.rust_radix`) that **dual-writes** the
Python tree and the Rust tree and **resyncs on divergence**, so a bug in
the Rust tree degrades to the Python behavior instead of corrupting
serving.

### API surface (stabilized by use in this branch)

- `RadixTree`: `new`, `match_prefix` (+ `match_prefix_meta` fast path
  returning length/hit stats without the token copy), `insert`,
  `evict(n)` (LRU or LFU), `inc_lock_ref` / `dec_lock_ref` (parent walks),
  `total_size` / `evictable_size` / `protected_size`, node-children
  introspection for the trace/replay tooling.
- `RadixKey`: token-key with **page flooring** (the `page_size` contract
  of the paged allocator) and the EAGLE **bigram view** (the draft-model
  key shift).
- `EvictionPolicy::{Lru, Lfu}`.
- Variants: SWA windowed eviction, mamba hybrid-state bookkeeping,
  HiRadix host-tier phases (`PHASE_BACKUP_HOST` / `PHASE_BACKUP_STORAGE` /
  `PHASE_LOAD_BACK`), and the unified multi-pool tree with per-node
  content types.

### Compatibility & rollout

- **Zero default change.** The crate is loaded lazily and fail-soft: if
  the extension cannot build/load, the scheduler runs pure Python and
  logs one warning.
- **Staged flag.** `SGLANG_RUST_SCHEDULER=radix` enables the dual-write;
  `planner`/`core`/`stream` (see RFC 0002) build on it.
- **Determinism contract.** The tree is a pure function of its op
  sequence; replaying a recorded session through a fresh tree must
  reproduce every plan — the lossless-trace backbone (plan §4.2) makes
  this a CI gate, not a spot check.

## Evidence (from `rust-scheduler`)

- 72 crate tests: golden sequences + property tests (split/coalesce,
  lock-ref invariants, eviction order under LRU/LFU, page-floor edges).
- **Differential parity** (`test_rust_radix_parity.py`, CPU CI): the Rust
  tree vs the unmodified Python `RadixCache` on identical op sequences —
  hit lengths, hit-node identity, evict order, lock counts.
- Criterion benches `M1`–`M4` (match/insert/evict/lock-walk) at
  1k/10k/100k/1M-token trees, gated against the upstream profiled Python
  numbers (§4.1 of the plan).
- `cargo clippy --all-targets -- -D warnings`, rustfmt, and a dedicated
  `sglang-radix-unit` CI job in `pr-test-rust-exts.yml`.

## Alternatives considered

1. **Keep Python only** — the baseline; the tree stays on the
   per-request critical path and every new variant forks the Python
   code.
2. **Extend `RadixCacheCpp`** — a second native tree already exists; two
   native trees diverging is strictly more maintenance than one.
   (On this branch the C++ tree is frozen as an oracle and is a
   cleanup-removal item, not a development target.)

## Migration plan

1. Land the crate + facades behind the flag (done on this branch).
2. Run the full `test/srt` matrix at `SGLANG_RUST_SCHEDULER=radix`
   (one CI GPU job per stage).
3. Default-flip the `radix` stage after the matrix is green; the
   dual-write remains until the `core` cutover (RFC 0002) removes the
   Python tree from the hot path.
