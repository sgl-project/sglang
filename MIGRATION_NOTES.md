# PP storage query + loop completion consensus

- Branch: `codex/pp-admission-v0522-20261016`.
- Community comparison base: `f7b8c6fe404ca4f29c3188cb50692f7735d45a99`
  (includes #43344). This is the rebased `community/main`, not the older local
  `main` ref.
- This update replaces the optional admission Gate; there is no B/C environment
  switch. Previous commits and the pre-rebase backup ref remain recoverable.

## Execution path

1. PP0 enqueues a shared query ticket. Each stage queries its storage namespace;
   the existing query collective agrees the storage-only root endpoint
   `S_min = MIN(S_i)` and component query results. Local L1/L2 coverage is not
   part of this MIN.
2. Every query produces one ordered native hit-queue item, including cancellation
   and no-local-GET cases. The scheduler binds it to the local suffix operation,
   then uses native host allocation and GET. Tickets use the native request handle
   and retry counter, so a new query cannot overwrite an unconsumed peer result.
3. The background completion worker keeps TP/CP reduction, but no PP collective.
   Its native ACKs enter the PP-loop completion queue.
4. The loop folds a batch of available ACKs along PP0 → … → PP-last → PP0.
   A stage with fewer ACKs shortens the batch; it never waits for local I/O.
   PP0 forwards the agreed results before the native cache-event drain.
5. The native ACK publisher, cache insertion / buffer ticket handoff, stop-policy
   evaluation, scheduler selection and H2D consume those results unchanged.

The ring reuses the native P2P helpers and WORLD group with a distinct message
tag. Returns and send-buffer retirement are delayed by `pp_loop_size`, so the
current fold does not become a cross-stage rendezvous before the current forward.
The same hook is installed in unified PP, PD Prefill and PD Decode loops. L2-only
loops send no completion traffic. Native reset/detach retire the added ACK
ownership after stopping I/O and fence already-sent results with a storage
generation. Transport cadence is preserved across reset; late packets are
received but not applied to the new cache. Native administrative eligibility
checks are unchanged (local ACK availability must not make ranks disagree).

## Native guarantees retained

- Keep native PP0-led L2/cache-event progression and native request ownership.
  This is **not** the earlier independent-L2/full-prefix admission protocol.
- Keep native ACK order/kinds and operation identity. Only numeric results cross
  PP; each rank retains its operation and buffers. A terminal ACK is published
  only after the corresponding local terminal ACK exists on every stage.
- KV-derived sidecars use the native contiguous-prefix result. SWA/Mamba keep
  native all-or-nothing verification at the queried endpoint. No arbitrary
  post-publication full-prefix truncation or new model restriction is introduced.
- Remove Gate leases, credits, filtered candidates, chunk-layout replay and
  Gate-specific request/configuration restrictions. Restore native policy,
  storage retry, wait-metric, allocator, matcher and DSA backend paths.
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

GPU model accuracy, deployment and performance have not been run for this
replacement. Earlier opt-in Gate benchmark results are not acceptance evidence
for this version; in particular, completion visibility now follows the loop
rather than the old background PP collective.
