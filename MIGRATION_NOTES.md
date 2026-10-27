# PP storage query + loop completion consensus

- Branch: `codex/pp-admission-v0522-20261016`.
- Community comparison base: `f7b8c6fe404ca4f29c3188cb50692f7735d45a99`
  (includes #43344). This is the rebased `community/main`, not the older local
  `main` ref.
- This update replaces the optional admission Gate; there is no B/C environment
  switch. Previous commits and the pre-rebase backup ref remain recoverable.

## Execution path

1. Cache-mode requests reach native prefetch submission after grammar and skip
   checks (anchor, threshold, rate limit, ongoing work and deadline). Submission
   binds its operation directly to a shared query ticket; the early PP0-only
   request-ingress path remains exclusive to buffer mode. PP0 enqueues the ticket.
   Each stage queries its storage namespace;
   the existing query collective agrees the storage-only root endpoint
   `S_min = MIN(S_i)` and component query results. Local L1/L2 coverage is not
   part of this MIN.
2. Every submitted query produces one ordered native hit-queue item, including
   cancellation. The scheduler resolves it to the bound local suffix operation,
   then uses native host allocation and GET. Tickets use the native request handle
   and retry counter, so a new query cannot overwrite an unconsumed peer result.
3. The background completion worker keeps TP/CP reduction, but no PP collective.
   Its native ACKs enter the PP-loop completion queue.
4. The loop folds a batch of available ACKs along PP0 → … → PP-last → PP0.
   A stage with fewer ACKs shortens the batch; it never waits for local I/O.
   PP0 forwards the agreed results before the native cache-event drain.
5. Cache-mode PP best_effort may stop after the first positive common I/O batch
   is visible. Query/queued I/O remains pending; misses and failed reads still
   retire through native query/terminal-ACK handling. This is a PP progress
   floor, not a full-fetch requirement or fixed wait. Timeout retains its native
   deadline and wait_complete its natural completion. Cache-mode PP keeps
   the request pending until the ordered query drain (no I/O launched) or the
   final agreed ACK retires the operation. The native finalizer then inserts the
   usable prefix; scheduler selection and H2D consume it through the usual path.
   After a candidate requests stop, selection pauses its scan there until the
   result arrives, leaving later candidates' I/O running. Ordinary readiness
   waits still allow the scan to continue. No second queue or capacity credit
   is introduced.

The ring reuses the native P2P helpers and WORLD group with a distinct message
tag. Returns and send-buffer retirement are delayed by `pp_loop_size`, so the
current fold does not become a cross-stage rendezvous before the current forward.
The same transport hook is installed in unified PP, PD Prefill and PD Decode
loops. This does not qualify PD Decode + PP + L3: the native PP decode loop
does not drain HiCache events. Its direct prefetch entry now binds the query
and passes the local prefix offset, but full Decode support remains separate.
L2-only loops send no completion traffic. Native reset/detach retire the added ACK
ownership after stopping I/O and fence already-sent results with a storage
generation. Transport cadence is preserved across reset; late packets are
received but not applied to the new cache. Native administrative eligibility
checks are unchanged (local ACK availability must not make ranks disagree).

## Native guarantees retained

- Keep native PP0-led L2/cache-event progression and native request ownership.
  This is **not** the earlier independent-L2/full-prefix admission protocol.
  Query submission order and local suffix boundaries retain that native contract;
  query tests do not claim to support divergent local L2 trees.
- Keep native ACK order/kinds and operation identity. Only numeric results cross
  PP; each rank retains its operation and buffers. A terminal ACK is published
  only after the corresponding local terminal ACK exists on every stage.
- KV-derived sidecars use the native contiguous-prefix result. SWA/Mamba keep
  native all-or-nothing verification at the queried endpoint. No arbitrary
  post-publication full-prefix truncation or new model restriction is introduced.
- Remove Gate leases, credits, filtered candidates, chunk-layout replay and
  Gate-specific request/configuration restrictions. Keep policy evaluation in
  the cache with the PP progress floor above; reuse native storage retry,
  wait-metric, allocator, matcher and DSA backend paths.
- Existing native restrictions/assumptions still apply. This does not repair
  pre-existing native L2 divergence or add support for unsupported model layouts.

## Verification

CPU regression coverage includes native query/storage lifecycle, real Gloo
shared-query MIN, TP2PP3 completion transport alongside native request/cache-event
traffic, slow-stage ACK availability, native buffer publication, pool failures,
native reset/late-result fencing, and unified/PD event-loop regression suites.

Final CPU run: 145 tests and 179 subtests passed across ten suites, using the
existing v0.5.21-cu130 container with this checkout mounted read-only (Python
3.12.3 / Torch 2.13.0). Ruff and diff checks passed. The native stop-policy,
hybrid-result, selector and H2D implementations were also compared to the base.

2026-10-26 review follow-up: restricted early ingress prefetch to buffer mode,
bound shared queries at actual cache submission, and removed the transient
query pointer and ineffective detach ACK publication. Late-query identity checks
apply only to the added shared-query path. The affected CPU/Gloo suites passed
62 tests and 70 subtests, including the existing SWA/Mamba descriptor test. The
new grammar regression fails against the previous scheduler (PP0 submits the
grammar-pending request) and passes on all three simulated PP stages after the
fix. Query/ACK transport still has real Gloo coverage; grammar ingress uses CPU
doubles. These checks are not GPU end-to-end qualification.

2026-10-27 stop/finalize correction: the 4cd986a GPU diagnostic reproduced
best_effort consuming stale completed_tokens while valid ACKs were still in
the PP ring. All replay queries returned their full storage spans. Stop now
preserves native operation ownership until query/terminal-ACK retirement;
it adds no collective, fixed wait, request queue or policy-specific protocol.
The non-PP immediate-finalization path and buffer ticket path are unchanged.
The shared lifecycle fixture covers best_effort and expired timeout with
unfinished queries, partial/full ACKs, and abort-before-terminal cleanup.
All eight cases fail before the fix and pass after it; the four related
CPU/Gloo suites pass 52 tests and 75 subtests. Its Beijing N4000 run completed
4000/4000 with88.348% hit and9.858s TTFT: partial recovery. GPU traces then
showed some positive queries still stopped before their FIFO I/O turn.

2026-10-27 stop-boundary follow-up: best_effort now waits for positive progress
already agreed by the existing ACK path; no unsynchronized worker flag is used
for scheduling. A stopped candidate pauses native selection until it is settled,
instead of stopping later requests before its token/chunk budget is charged.
The shared lifecycle regression covers partial results, zero-progress failure,
query misses and abort. The existing scheduler fixture verifies that a stopped
candidate holds the scan while an ordinary pending request still permits a
ready follower. Eight related CPU/Gloo suites pass110 tests and141 subtests.
This follow-up's GPU/performance validation is pending.

The 4cd986a replacement completed one N4000 best_effort run in Singapore and
one N4000 timeout run in Beijing, both without HTTP failures. They are not
same-machine policy ablations, nor model-accuracy acceptance. Earlier opt-in
Gate benchmark results likewise do not qualify this different completion path.
