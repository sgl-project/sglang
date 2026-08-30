# RFC 0002: `sglang-scheduler` — an opt-in Rust next-batch planner and scheduler core

**Status:** draft (from branch
[`rust-scheduler`](https://github.com/sorenmat/sglang/tree/rust-scheduler),
where it ships behind `SGLANG_RUST_SCHEDULER=planner|core|stream`)
**Depends on:** RFC 0001 (`sglang-radix`, #37126) for the admission tree.

## Background

Every scheduler iteration does CPU control-plane work in Python: the
next-batch decision (prefill vs decode, admission with LPM scoring and
token/page budgets, chunked-prefill parking, mixed-batch rules), the
new-token-ratio (NTR) tracker, retract/abort decisions, and per-iteration
result bookkeeping (accept runs, finish-state processing, stop-string
scans, spec-v2 accept-run resolution, ~40-list batch payload
construction, streaming output assembly). On a fast NVFP4 model doing
small speculative-decode steps this CPU time is a growing fraction of
iteration time and sometimes cannot hide behind GPU work.

## Proposal

Ship `sglang-scheduler` as a Rust crate with two front ends over one
pure decision engine, exposed to Python through the `_scheduler`
extension (PyO3, `python` feature):

1. **`plan_next_batch` (stateless shadow planner).** Python keeps the
   queues/tree/allocator and passes compact snapshots each iteration;
   the returned plan is **diffed against Python's own decision** (trace
   capture) before it is applied. This is the `planner` stage — a pure
   A/B with no behavior change.
2. **`SchedulerCore` (persistent core).** The engine owns the
   waiting/running queues, the radix tree (from `sglang-radix`), the NTR
   tracker and per-request token storage. Python feeds ingress + results
   and executes each plan + event list. This is the `core` stage.

   - In **bookkeeping mode** (default; `SGLANG_RUST_CORE_APPLY=0`) the
     core runs in lock-step and Python executes its own batches — the
     core is a mirror, never a double-free hazard.
   - In **cutover mode** (`SGLANG_RUST_CORE_APPLY=1`, experimental) the
     core's events are applied to the Python allocator/row pool.

3. **Egress + spec bookkeeping** (M5/M6): the per-iteration stream frame
   build (`stream` stage) and the spec-v2 accept-run resolution +
   per-req counters (`resolve_spec_runs` / `SpecCounters`, folded into
   `core.apply_result` via `ResultRow.spec`).

The core does **no CUDA work**; torch is only touched in `apply_plan` on
the scheduler thread, and planning uses CPU mirrors only.

## API surface

- `plan_next_batch(cfg, ntr, waiting, running, chunked, scores, deprio,
  env, iter)` → `(mode, batch_is_full, prefill, decode)` compact tuples.
- `SchedulerCore`: `ingest`, `apply_result(rows, kv_rows)` (rows carry
  optional spec-v2 metadata), `plan(env)`, `drop`, plus observability
  accessors (`waiting`/`running`/`tree_stats`/`spec_counters`/…).
- `resolve_spec_runs` + `SpecCounters` (plan §9).

## Compatibility & rollout

- **Zero default change**, staged flag
  `SGLANG_RUST_SCHEDULER=off|radix|planner|core|stream`; each stage
  implies the previous. Fail-soft load.
- **Determinism + lossless replay** are the backbone: every recorded
  plan/apply/drop is replayed through a fresh core and diffed
  field-for-field (`sglang.test.scripted_runtime.replay`). Plan-for-plan
  trace parity is a CI gate.

## Evidence (from `rust-scheduler`)

- 38 crate unit tests + the lossless-replay backbone
  (`test_rust_trace_replay.py`) + differential parity
  (`test_rust_radix_parity.py`, `test_rust_spec_parity.py`,
  `test_rust_stream_parity.py`).
- Criterion benches `M5`–`M7`/`M11` (prefill burst, decode steady, mixed,
  core loop) and the spec resolution benches (`spec_bench`), gated
  against the upstream profiled Python numbers.
- `cargo clippy --all-targets -- -D warnings`, rustfmt, CI job
  `sglang-scheduler-unit` (rlib + `--features parallel` + pyo3 cdylib +
  `smoke_test.py`).

## Migration plan / default flip

1. Land the crate + bindings behind the flag (done on this branch).
2. Run the full `test/srt` pytest matrix at `core` (and `stream`) —
   one CI GPU job per stage.
3. **Default flip** is a one-line change: the `SGLANG_RUST_SCHEDULER`
   default in `python/sglang/srt/environ.py` moves from `"off"` to the
   validated stage (`core`/`stream`). It is deliberately the last step
   and is only taken after step 2 is green on the target host — the
   bookkeeping/trace tooling makes the flip reversible in one env var.

## Alternatives considered

- **Micro-optimize the Python control plane** — the 25.8 µs→3.9 µs
  filter example shows the ceiling; the per-iteration work still fails
  to hide behind GPU work at c16, and it forks on every new batch mode
  (spec-v2, beam, dllm, disagg).
- **Full rewrite including the allocator** — out of scope by design
  (plan Phase 6 is data-driven and deferred; the paged allocator
  manipulates device tensors and stays in torch).
