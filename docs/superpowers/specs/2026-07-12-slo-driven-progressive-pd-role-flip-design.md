# SLO-Driven Progressive PD Role Flip Design

**Date:** 2026-07-12

## Summary

This design adds a controller-owned, SLO-driven state machine that progressively converts one decode worker into a prefill worker without restarting the model process. The controller monitors cluster-level prefill TTFT attainment and decode TPOT attainment. When prefill is below a configured threshold while decode remains above it, the controller migrates a configurable fraction of the source decode worker's running requests to another decode worker, observes whether SLO recovers, and either restores the source as a decode worker or migrates the remaining work and commits a decode-to-prefill role change.

Migrated requests use two-source KV reconstruction whenever possible. A continuous prefix is restored from Mooncake Store through HiCache, and the remaining committed KV is transferred directly from the source decode worker. Mooncake misses fall back to a larger direct transfer from the source decode worker. The target decode worker activates a request only after combined KV coverage and request execution state have been validated.

## Goals

- Monitor per-node request-level TTFT and TPOT results and aggregate them by PD role.
- Trigger D-to-P only when prefill TTFT attainment is below threshold and decode TPOT attainment remains at or above threshold.
- Migrate a configurable fraction of the source decode worker's running requests before deciding whether to change its role.
- Keep the source decode worker draining but actively decoding its remaining requests during a fixed observation window.
- Reconstruct migrated KV from Mooncake Store/HiCache plus direct source-decode transfer.
- Fall back to source-decode transfer for any prefix range unavailable in Mooncake.
- Preserve exactly-once request ownership and output across migration.
- Change the worker's scheduler event loop in process after final role commit.
- Make migration batches atomic: all selected requests commit or all remain on the source.

## Non-Goals

- P-to-D role conversion is not changed by this design.
- The worker-local `FlipStateMachine` does not make cluster-level role decisions.
- The first implementation does not optimize request selection by KV size, predicted remaining tokens, or per-request compute cost.
- The first implementation does not allow partial success within a selected migration batch.
- This design does not provide fault tolerance for simultaneous loss of both source and target workers.

## Ownership and Components

### Controller

`scripts/playground/disaggregation/pd_flip_controller.py` is the sole owner of the cluster state machine. It collects worker roles, router state, queue load, capacity, and request-level SLO results; selects source and target decode workers; calculates the effective first-migration ratio; coordinates migration and role changes; and reconciles interrupted sessions.

### Router

The router remains the traffic-plane owner. It marks workers draining or admitting and updates routing roles and bootstrap ports. It does not decide when a flip should happen or coordinate KV migration.

### Worker Scheduler

The scheduler exposes worker-local primitives to report role, active event loop, admission, queues, request slots, and KV capacity; manage migration sessions; freeze and restore requests; restore Mooncake prefixes; transfer source KV and deltas; validate stitched KV; change runtime role; and redispatch into the new role's event loop.

The worker-local flip FSM remains available for observability and safety checks, but it must not independently initiate or commit a role change.

## Configuration

The controller accepts:

```text
pd_flip_slo_threshold = 0.9
pd_flip_first_migration_ratio = 0.5
pd_flip_observation_seconds = 10
pd_flip_migration_timeout_seconds = 120
pd_flip_min_prefill_slo_samples = 20
pd_flip_min_decode_slo_samples = 20
```

`pd_flip_first_migration_ratio` must satisfy `0 < ratio < 1`. It is applied to the source worker's `running_batch` size at selection time.

## SLO Aggregation and Decisions

The monitor collects request-level results from every worker and computes role-level attainment from counts rather than averaging node percentages:

```text
prefill_ttft_attainment =
    total prefill requests meeting TTFT SLO /
    total prefill requests with a valid TTFT result

decode_tpot_attainment =
    total decode TPOT samples meeting TPOT SLO /
    total valid decode TPOT samples
```

A D-to-P attempt starts only when:

```text
prefill_ttft_attainment < pd_flip_slo_threshold
decode_tpot_attainment >= pd_flip_slo_threshold
prefill and decode both meet minimum sample requirements
a non-draining source decode worker exists
a different non-draining target decode worker exists
```

The observation window uses only samples produced after entry into `OBSERVING`. At the end of the window:

```text
prefill_ttft_attainment >= threshold
    => keep source as decode and return to SAFE

prefill_ttft_attainment < threshold
and decode_tpot_attainment < threshold
    => decode has no spare capacity; keep source as decode and return to SAFE

prefill_ttft_attainment < threshold
and decode_tpot_attainment >= threshold
    => migrate remaining source work and commit D-to-P
```

If either role lacks sufficient observation-window samples, the controller restores source decode admission and returns to `SAFE`.

## Controller State Machine

### `SAFE`

The cluster serves normally. When the D-to-P SLO condition holds, the controller creates a unique migration session and enters `SELECTING`.

### `SELECTING`

The controller selects a non-draining source decode worker, a different non-draining target decode worker, and the first N requests in `source.running_batch.reqs`.

```text
N = max(1, ceil(source_running_count * effective_ratio))
```

The configured ratio is tested against target request slots and KV capacity. If the selected prefix of N requests does not fit, the ratio is divided by two and N is recalculated. This repeats until the selection fits or one request still cannot fit. If one request cannot fit, the attempt returns to `SAFE` without migration.

### `FIRST_MIGRATING`

The controller drains the source in the router and pauses worker admission. The source stops accepting new requests but continues decoding existing requests.

The first N requests use the initial-copy and delta protocol below. The batch is atomic. The target holds every prepared request and does not schedule or emit output until the whole batch is ready and ownership transfers.

If any selected request fails, the target aborts the whole batch, the source retains or restores every selected request, admission resumes, and the state returns to `SAFE`.

If the batch commits, selected requests continue on the target. The source remains draining and enters `OBSERVING` with unselected requests still running locally.

### `OBSERVING`

The source remains a decode worker, receives no new requests, and continues decoding only unselected requests. Other decode workers serve normally. The monitor clears the triggering window and collects new samples for `pd_flip_observation_seconds`.

If SLO recovers or decode loses spare capacity, source admission resumes. Requests already moved in the first batch remain on the target. If prefill remains below threshold while decode remains healthy, the controller enters `SECOND_MIGRATING`.

### `SECOND_MIGRATING`

The controller migrates all remaining migratable source work, including running and eligible waiting requests. This batch is also atomic.

On failure, the second batch aborts, the source remains decode, admission resumes, and the controller returns to `SAFE`. Requests moved by the first batch remain on the target.

On success, the controller verifies that the source running batch, waiting queue, PD queues, and migration sessions are empty, then enters `FLIPPING_ROLE`.

### `FLIPPING_ROLE`

The controller changes the worker runtime role to prefill with `force=false`. The worker exits its decode event loop and redispatches into the prefill loop. Only after status reports both `role=prefill` and `active_event_loop=prefill` does the controller update the router role and bootstrap port, resume admission, undrain the worker, and return to `SAFE`.

## Two-Source KV Reconstruction

For each selected request:

```text
P  = prompt/prefill KV length
C0 = committed KV length at initial-copy snapshot
C1 = committed KV length at ownership-transfer boundary
H  = usable continuous Mooncake/HiCache prefix length
```

Mooncake/HiCache matches complete, continuous pages. To keep the page containing the prefill/decode boundary under one writer:

```text
H = min(
    mooncake_continuous_hit_length,
    floor(P / page_size) * page_size
)
```

The target reconstructs:

```text
Mooncake/HiCache restore: [0, H)
source decode initial transfer: [H, C0)
source decode delta transfer: [C0, C1)
```

This supports:

- `full_prefix_stitch`: Mooncake covers every complete prefill page;
- `partial_prefix_stitch`: Mooncake covers a shorter continuous prefix;
- `source_decode_full_fallback`: Mooncake hit length is zero and source decode supplies the full range.

The target scheduler owns semantic stitching. Mooncake Store provides content-addressed page lookup and transfer; the scheduler validates token ranges and installs the final `req_to_token` mapping.

### Target Prefix Restore

The manifest carries token history and cache identity needed to reproduce page hashes: input and output token IDs, model and KV layout identity, page size, cache salt and extra key, parallel layout metadata, and P/C0/C1 boundaries.

Prefill workers must use a HiCache write policy that publishes completed prompt KV pages to the shared Mooncake L3 namespace before those pages are considered eligible for stitching. A page that has not completed L3 write-back is treated as a Mooncake miss and is supplied by the source decode worker.

The target performs a local HiCache match, queries Mooncake L3 for continuous pages, starts L3-to-L2 prefetch, and loads restored pages into the GPU KV mapping. TP ranks use the minimum agreed hit length.

### Source Initial Transfer

The target returns the actual `decode_prefix_len=H`. The source consumes it before initializing transfer and sends only:

```text
source_req_to_token[H:C0]
```

The sender's expected page count is derived from this sliced range, not `[0:C0]`.

### Delta and Cutover

Initial copy runs while the source remains sole owner and continues decoding selected requests. Once initial transfer and prefix restore finish, the source freezes the selected batch at a scheduler batch boundary and records C1. It transfers `[C0:C1)` plus latest execution state.

Execution state includes output history and last emitted sequence, sampling parameters and RNG state, grammar state when present, priority and routing metadata, time and streaming metadata, and supported model-specific state.

The target verifies complete `[0:C1)` coverage on every TP rank before commit.

## Atomic Ownership Transfer

The target uses:

```text
PREPARED_HELD
READY_TO_ACTIVATE
ACTIVE
```

The controller performs:

```text
target prepare and validate
controller records commit intent
target commit => READY_TO_ACTIVATE, not scheduled
source finish => source relinquishes ownership and releases resources
target activate => requests enter target scheduling queues
```

Before source finish, the source is the only execution and output owner. After target activation, the target is the only owner. The target resumes after the last emitted output sequence, and stale or duplicate output sequences are discarded by the relay.

All selected requests move together. A failure before source ownership is relinquished aborts the target batch and unfreezes source requests. A role flip cannot proceed unless the entire second batch activates and the source becomes fully idle.

## Dynamic Scheduler Event-Loop Switching

Runtime mutation of `disaggregation_mode` alone is insufficient because the current scheduler dispatches its event loop once. The scheduler must use an outer redispatch loop:

```python
while not shutting_down:
    dispatch_event_loop(self)
```

Each role-specific loop checks its expected role at a safe iteration boundary and returns if the role changed. Mutation remains restricted to a fully idle scheduler, so overlap result queues and in-flight batches are drained first.

Runtime status reports both configured role and active event-loop role. Router role mutation is gated on their agreement.

## Capacity and Ratio Fallback

The target reports free request slots, available KV tokens/pages, held migration reservations, and effective maximum running requests per DP worker.

Preflight capacity uses the worst case: it reserves enough target capacity for the selected requests' complete committed KV plus a decode-growth allowance, without assuming a Mooncake hit. After the target discovers H, it may release reservation that is provably covered by reusable local KV. This guarantees that zero-hit fallback can still complete. The configured ratio is repeatedly halved until both request-slot and worst-case KV checks pass. Selected objects are always the first N requests in source running-batch order.

## Failure and Recovery

| Failure | Required behavior |
|---|---|
| No target decode worker | End attempt in `SAFE` |
| Target lacks capacity | Repeatedly halve ratio |
| One request still cannot fit | End attempt in `SAFE` |
| Mooncake full or partial miss | Source decode supplies `[H:C)` |
| First batch transfer or validation failure | Abort whole first batch; requests remain on source |
| SLO recovers after first commit | Keep moved requests on target; resume source decode admission |
| Decode SLO falls after first commit | Keep source as decode and resume admission |
| Second batch failure | Abort second batch; keep source as decode |
| Runtime role mutation fails | Keep source draining as decode and recover or retry |
| Worker role changes but router update fails | Keep worker draining and retry router update |
| Controller restarts | Reconcile source, target, and router by session ID |

Prepare, status, abort, commit, finish, activate, admission, and router-role operations are idempotent for a session ID.

## Observability

Each transition records state, direction, source/target, SLO sample counts and attainments, configured/effective ratio, fallback count, selected N, source/target capacity, residual queues, router/worker role, action latency, and retries.

Each migrated request records P/H/C0/C1, stitch mode, Mooncake/source/delta byte counts and durations, HiCache query/prefetch/load-back latency, held/freeze/commit/activation latency, source queue, final owner, output sequence boundary, and rollback reason.

## Testing

### Unit Tests

- Role-level SLO aggregation by sample counts.
- Trigger, recovery, decode-risk, and insufficient-sample decisions.
- First-N selection and `ceil(running * ratio)`.
- Repeated ratio halving against request-slot and KV limits.
- Mooncake full, partial, and zero hits.
- Page-aligned stitch boundaries.
- Source sender slicing by returned `decode_prefix_len`.
- Initial plus delta coverage without gaps or conflicting writers.
- Atomic batch abort when one request fails.
- Idempotent prepare, commit, finish, activate, and abort.
- Decode event-loop exit and prefill redispatch.

### Integration Tests

- Fake backend with exact ownership and output sequence.
- Mooncake validation of prefix, source-initial, and delta byte ranges.
- Full-hit, partial-hit, and source-full-fallback runs.
- Streaming and non-streaming requests.
- Mixed short and long requests.
- Controller restart and session reconciliation.
- Router update failure and retry.
- Four-node `1P3D -> 2P2D` transition.

### Acceptance Criteria

- No request is lost, duplicated, or executed by two decode workers simultaneously.
- Output tokens match a no-migration baseline without duplication or gaps.
- Target stitched KV matches source committed KV at cutover.
- A failed first or second batch leaves the source able to continue decoding.
- SLO recovery or decode SLO risk prevents role flip.
- Persistent prefill risk with healthy decode SLO completes D-to-P.
- Final router role, worker runtime role, and active scheduler event loop agree.
- Raw artifacts expose the state, SLO, capacity, transfer, stitch, and ownership timings needed to explain results.
