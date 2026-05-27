# Shared HiCache

This folder implements SGLang's Shared HiCache path: reuse KV pages that already
exist in another SGLang worker's HiCache host tier, then transfer those pages
directly into the target worker's GPU KV cache.

## Scope

- Shared HiCache is opt-in and scheduler-driven.
- The router provides a reuse plan on the request.
- The target scheduler decides whether the plan still applies.
- The source worker revalidates the requested host pages by block hash before
  transferring.
- The target inserts transferred device pages into its radix cache only after
  the transfer completes.
- Failures are fail-open: a request should continue with normal prefill if
  shared reuse cannot be completed.

Do not treat the router plan as an ownership lease. It is only a hint. The
source is authoritative at transfer time.

## Main Files

- `config.py`: config parsing and environment defaults.
- `plan.py`: `SharedHiCachePlan` parsing, validation, and serialization.
- `scheduler_mixin.py`: scheduler entry point for plan resolution and
  target-side reuse preparation.
- `manager.py`: main orchestration object used by the scheduler.
- `source.py`: source-side plan validation, host lookup, host-page protection,
  and transfer execution.
- `target.py`: target-side device allocation, transfer staging, and radix-cache
  insertion.
- `transfer.py`: direct-transfer backend abstraction and NIXL implementation.
- `service.py`: source-side async transfer worker service.
- `control.py`: target/source control-plane messages and transfer-handle glue.
- `pending.py`: pending target transfer state and timeout helpers.

## Router To Engine Walkthrough

Roles:

- Router: external request router that knows which workers have advertised
  host-tier KV blocks.
- Target worker: the SGLang worker selected to serve the request now.
- Source worker: the SGLang worker that may still have the wanted blocks in its
  HiCache host tier.

High-level flow:

```text
client request
  |
  v
router
  |  chooses target worker
  |  attaches SharedHiCachePlan hint:
  |    source_worker_id, target_worker_id, block hashes, TP shape, expiry
  v
target engine frontend / tokenizer path
  |
  v
target scheduler waiting queue
  |
  v
target SharedHiCache manager
  |  validates plan and current local prefix
  |  allocates target GPU KV pages
  |  sends async transfer request to source TP rank
  v
source SharedHiCache service
  |  revalidates plan, rank, expiry, and host block hashes
  |  decides whether it can serve now
  |  protects source host nodes only if it will transfer
  |  performs NIXL host-to-target-GPU transfer
  v
target SharedHiCache manager
  |  receives completion notification
  |  validates transferred hash sequence
  |  inserts transferred pages into target radix cache
  |  schedules request using the extended prefix
  v
model forward on target worker
```

Detailed lifecycle:

1. The router receives a client request and computes block hashes for the
   request prefix.
2. The router picks a target worker for the request.
3. If another worker has useful host-tier blocks, the router attaches a
   `shared_hicache_plan` to the engine request. This is a hint, not a lease.
4. The SGLang engine API carries `shared_hicache_plan` through request objects
   into `ScheduleBatch.Req`.
5. The target scheduler runs normal local prefix matching first.
6. The target scheduler asks Shared HiCache to prepare extra prefix blocks only
   for the remaining suffix beyond the local device/host hit.
7. The target manager validates that the plan names this worker as target, that
   the TP shape/rank is compatible, that the block size matches the local page
   size, and that the plan has not expired.
8. The target reserves GPU KV indices for the planned suffix. These indices are
   not visible to the radix cache yet.
9. The target sends a JSON transfer request to the source control endpoint for
   the matching TP rank. The request includes target session metadata, target KV
   pointers, target page indices, and the original plan.
10. The source service receives the request on its per-rank async worker.
11. The source revalidates source worker id, source/target TP rank, plan expiry,
    block size, and the requested block hashes against the current HiCache host
    block index.
12. The source may reject without pinning if blocks are missing, the request is
    malformed, topology/rank ownership is wrong, or source policy says it should
    not serve now.
13. If the source accepts, it protects the matched host nodes with
    `TreeNode.protect_host()`. Host eviction must not free those pages while the
    direct transfer is in flight.
14. The source transfer backend copies pages from source host memory into the
    target GPU KV pages and sends a completion notification to the target.
15. The source releases host protection in a `finally` path after transfer
    completion or failure.
16. The target observes transfer completion, validates that returned pages are
    contiguous and match the expected hashes, then inserts the target KV indices
    into the radix cache.
17. If insertion succeeds, `req.shared_hicache_hit_length` is updated and the
    scheduler uses the longer prefix for model forward.
18. If any step fails, the target frees or quarantines reserved GPU KV indices,
    records a miss/error metric, and the request continues with normal prefill.

Source admission rule:

- Source is authoritative at transfer time.
- Source can say no even if the router hint said it probably had the blocks.
- Once source says yes and starts transfer, it must keep the accepted host pages
  protected until transfer finishes or fails.

## Request Lifecycle

1. The scheduler receives a request with a Shared HiCache plan.
2. `scheduler_mixin.py` computes the current local prefix and calls the manager.
3. `manager.py` validates plan shape, TP ownership, block alignment, local cache
   state, and available transfer backend.
4. The target reserves target GPU KV indices through `SharedHiCacheTarget`.
5. The target submits a source transfer request through the control plane.
6. `service.py` runs source transfer work asynchronously.
7. `source.py` revalidates source worker id, TP rank, plan expiry, block size,
   and host block hashes.
8. Source host nodes are protected while host page indices are resolved and the
   transfer is in flight.
9. `transfer.py` copies source host pages into target GPU KV pages.
10. The target receives completion, validates contiguous hashes, inserts pages
    into the radix cache, and records metrics.
11. On any failure, reserved target indices are freed or quarantined depending
    on whether transfer outcome is known.

## Pinning And Lifetime

There are two separate lifetime windows:

- Router plan to source transfer:
  - No pin is held.
  - The plan can become stale.
  - The source must revalidate block hashes and fail open on misses.

- Source lookup to transfer completion:
  - Source host pages are protected with `TreeNode.protect_host()`.
  - HiCache host eviction skips nodes whose `host_ref_counter > 0`.
  - Protection is released with `TreeNode.release_host()` in a `finally` path.

Use `lookup_hicache_host_blocks(..., protect=True)` when resolving source host
blocks for direct transfer. Do not read host indices from the host block index
without protecting the owning nodes.

## Control Plane

Control endpoints are `tcp://` endpoints with `{tp_rank}` expansion. Each TP
rank owns its own source-transfer worker service and transfer backend.

Expected target-to-source request fields include:

- `transfer_id`
- `target_control_endpoint`
- `transfer_backend`
- `plan`
- `start_block`
- `max_blocks`
- `target_session_id`
- `target_kv_ptrs`
- `target_kv_item_lens`
- `target_page_indices`
- `target_metadata`

Payloads must stay JSON-serializable. Do not send Python objects over the
control plane.

## Plan Semantics

`SharedHiCachePlan` is block-based. Important fields:

- `source_worker_id`
- `target_worker_id`
- `block_hashes`
- `kv_block_hashes`
- `planned_prefix_blocks`
- `block_size_tokens`
- `source_tp_rank`
- `source_tp_size`
- `target_tp_rank`
- `target_tp_size`
- `start_block_index`
- `expires_at`

Rank fields may be omitted only when the plan is rank-generic and the local
rank can safely infer ownership. Explicit rank mismatches must reject.

Always compare transferred pages against the expected planned hashes before
inserting them into the target radix cache.

## Metrics

Metrics are enabled through the scheduler metrics gate. Do not create an
independent metrics path for this feature.

Important labels:

- `backend`: current transfer backend, usually `nixl`.
- `outcome`: `hit`, `miss`, `skip`, or `error`.
- `reason_code`: detailed reason such as `ok`, `insert_returned_zero`,
  `missing_first_block`, `target_staging_alloc_failed`, `fetch_worker_unavailable`,
  `source_endpoint_unavailable`, or timeout/error codes.
- `tp_rank`: local TP rank.

Successful Shared HiCache transfer hits use:

```text
outcome="hit", reason_code="ok"
```

## Development Rules

- Keep this path fail-open.
- Revalidate all router-provided plan data at the source.
- Keep control-plane payloads typed and JSON-compatible.
- Preserve TP-rank ownership checks.
- Do not block the HTTP request path on source transfer execution.
- Do not hold source host protection longer than the transfer needs.
- Free target device indices on known failures.
- Quarantine target device indices when direct-transfer outcome is
  indeterminate.
- Keep source and target cleanup idempotent.
- Prefer focused runtime validation over broad synthetic tests for this feature.

## Validation Checklist

For functional validation, check:

- Source phase completes without HTTP errors.
- Target phase completes without HTTP errors.
- Frontend logs have no `no endpoints available` during load.
- Source logs show `SharedHiCache NIXL transferred`.
- Target logs show `Shared HiCache staged ... direct=True`.
- `sglang:cached_tokens_total{cache_source="shared_hicache"}` is nonzero.
- `sglang:shared_hicache_tokens_total` is nonzero with
  `outcome="hit", reason_code="ok"`.
- No `direct_transfer_failed`, transfer timeout, source timeout, queue-full, or
  rank-rejection logs are present.
- GPU memory returns to idle after cleanup.

For launch scripts, readiness must prove a pinned request can route. A
successful `/v1/models` response alone is not sufficient.
