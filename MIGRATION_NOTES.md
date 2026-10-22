# Async PP cache admission with shared queries

- Branch: `codex/pp-admission-v0522-20261016` (name retained).
- Rebased community main: `f7b8c6fe404ca4f29c3188cb50692f7735d45a99`, fetched 2026-10-21.
- Includes #43344 (`3eb4bd598f`) and the upstream first-blocked-poll timeout semantics.
- Previous release head remains recoverable at
  `refs/backup/pp-admission-before-query-rebase-20261021`.
- The independent PAI worktree/branch was not changed.

## Execution path

The existing `SGLANG_PP_L3_CACHE_ASYNC_ADMISSION` flag selects the same static
PD-prefill + PP + HiCache cache-mode + L3 profile. No additional flag is added.

Every stage registers its query descriptor when handling the native request;
only PP0 enqueues it. The existing ticket command worker broadcasts the descriptor,
queries each stage's storage namespace, and uses the native query MIN groups
(including PP). No cache allocation or radix mutation occurs in this worker.
A root-span query gives all stages the same logical coordinates even when their
resident L1/L2 prefixes differ. The result is clipped to each local operation's
missing suffix and handed to the native hit queue. The existing ticket state
handles both arrival orders; the worker never waits for the downstream request.

Chosen minimal contract (2026-10-22): `S_min = MIN(S_i)`, where `S_i` is the
root-relative L3 query result. No local L1/L2 length is included in that vote:
neither `MIN(A_i + B_i)` nor `MIN(max(A_i, S_i))` is introduced. PP0 still
submits the shared task when it has no local GET. After query agreement, each
stage clips GET to its missing suffix; resident cache is not truncated by
`S_min`. A storage miss on one stage can conservatively suppress useful GETs
on peers; this is an accepted hit-rate tradeoff, not a new retry mechanism.
The existing Gloo regression distinguishes this storage-only MIN from a combined
local/storage vote, and covers PP0 submitting a task without a local GET.

Cache allocation/eviction, GET and ACK publication remain native. Completion
groups include only stage-local TP/CP; **no cross-PP completion all-reduce is
restored**. The existing PPAdmissionGate still performs readiness, protected
full-prefix MIN and commit, then native selection/H2D. Query MIN is not a
replacement for that final agreement.

Cancellation uses the native retirement path. A cancelled query still participates
in its ordered MIN with zero; it does not publish a result to a retired operation.
The buffer-mode ticket allocation/trim/GET path is retained.

## Scope and validation

- Keep native main timeout deadlines, deadline accounting and retry suppression.
- Reuse the upstream ticket and storage-lifecycle fixtures for cache queries
  and timeout integration. Keep one focused gate test module; no AST extraction,
  private CPU loader, or duplicate controller fixture remains.
- Cover local-before-query / query-before-local arrival, PP0-only enqueue,
  unequal/missing hits with real three-process Gloo broadcast/MIN, cancellation,
  buffer allocation/trim, sync-group scopes, full-prefix admission/NACK, host
  reference ownership, native chunk-end replay and wait/deadline accounting.
- Root queries are conservative when L3 has prefix holes that resident caches
  cover. They can query already-resident pages; this update does not claim an
  optimal query budget or a measured performance improvement.
- Cleanup keeps the protocol unchanged: Prefill and Decode share the native
  release-ring sender, with native RID lists / async admission framing preserved.
  The lease owns the certified cap; native cache methods handle span retirement.
- Validation uses normal module imports in an isolated v0.5.21-cu130 container
  (Python 3.12.3, Torch 2.13.0, Transformers 5.12.1), with current source mounted
  read-only. Nine relevant suites passed: 142 tests and 194 subtests, including
  real three-process Gloo and native PrefillAdder/budget/PP-drain regressions.
  Gate, ticket and storage-lifecycle unittest entry points also passed (46 tests,
  reruns rather than additional coverage); compile, Ruff and diff checks passed.
  This is CPU regression coverage, not a full main dependency-matrix or GPU qualification.
- No GPU, storage-service deployment or benchmark was run for this update.
  Earlier v0.5.22 performance/accuracy numbers do not validate this rebased code.

## Hybrid query preparation (2026-10-22; not hybrid admission enablement)

Shared root queries now reuse the native component transfer builder, including
SWA trailing-window and Mamba state descriptors. Local prefetch uses that same
builder; query construction performs no staging allocation or I/O. Native
query MIN, sidecar key alignment, stop policies and all-or-nothing are unchanged.

The SWA/Mamba admission guard remains. The current Gate still promises only FULL
token capacity and may shorten a lease to a scalar MIN. Enabling hybrid models
also requires multi-pool capacity/lock ownership and a common restorable endpoint;
adding query descriptors alone does not establish either guarantee.

Validation: 47 tests and 50 subtests passed in the CPU container above (storage
lifecycle, upstream PP ticket and async admission suites), including a regression
using native SWA/Mamba descriptor methods and real three-rank Gloo. No GPU test.
