# M4 Test Survey — cache-aware-zmq + PD pool isolation

Snapshot of SMG tests touching cache-aware ZMQ + PD routing + active-load
tracking, categorised by what's portable to sgl-router.

The plan's Task 0 list is the input; this file is the output that drives Tasks 1–6.

## Source worktrees

| Short name | Absolute path | Role |
|---|---|---|
| **`MAIN`** | `/Users/kangyan.zhou/sglang_workspace/sglang/sgl-model-gateway/` | Canonical merged SMG on `main` of sgl-project/sglang. Cache-aware (non-zmq), registry (with PR #25184 PD pool isolation), tree, factory, PD tests. Prefer this for `policies/registry.rs` and PD test logic — has had the most cleanup. |
| **`ZMQ`** | `/Users/kangyan.zhou/sglang_workspace/add_message_queue/sgl-model-gateway/` | Unmerged `cache-aware-zmq/debug-routing` branch. Sole source of `cache_aware_zmq.rs`, `cache_aware_zmq_test.rs`, and the kv-events module. M3 already ported the wire/hash/tree/subscriber/index pieces. |
| sgl-router | `/Users/kangyan.zhou/sglang_workspace/feat/sgl-router-m1-http-tokenizer/experimental/sgl-router/` | Target (where M4 commits land). |

## Scope adjustment (from kickoff brief)

* Plan Task 2 (`workers/pd.rs` annotation parser) is **DROPPED**. M2 already
  populates `Worker.mode` via selector-based dispatch at discovery time. PD
  pool isolation in M4 reads `Worker.mode` via
  `WorkerRegistry::workers_for_mode`. No annotation parsing needed. The
  `bootstrap_port` field is added to `WorkerSpec` directly so PD dispatch
  can find each prefill worker's bootstrap server.
* PD-mode dual dispatch + bootstrap injection (`bootstrap_host`,
  `bootstrap_port`, `bootstrap_room`) is M4 scope — required for Task 6
  ("E2E in PD mode … bonus tokens decoded correctly") and the
  `test_pd_mode_response_has_decode_affinity_header` acceptance test. The
  router fans the same modified body to both prefill and decode via
  `tokio::join!`; the decode response is what the client sees.

## Category 1 — Inline tests inside SMG policy files

### `cache_aware_zmq.rs::tests` (`ZMQ`, lines 752–1493)

| SMG test | Action | Why |
|---|---|---|
| `cache_hit_routes_to_holding_worker` | **port-adapted** from `ZMQ` | Core happy path. Adapted to sgl-router's `SelectionContext` (no `SelectWorkerInfo`). |
| `no_tokenizer_falls_back_to_min_load` | **port-adapted** from `ZMQ` | Fallback path. In sgl-router, "no tokenizer" → fall back to round-robin / `Worker::active_load`-ordered pick. |
| `low_match_rate_falls_back_to_min_load` | **port-adapted** from `ZMQ` | Same shape as above; threshold gating. |
| `imbalanced_load_skips_cache_check` | **port-adapted** from `ZMQ` | Same logic; uses `Worker.active_load()`. |
| `remove_worker_clears_tree_entry` | **port-adapted** from `ZMQ` | Lifecycle. The policy delegates to `KvEventIndex::remove_worker` already ported in M3. |
| `block_removed_event_drops_worker_from_node` | **drop** | Already covered by `kv_events/index.rs::pump_handles_*` in M3. Re-doing it through the policy adds zero coverage. |
| `multi_model_isolation` | **port-adapted** from `ZMQ` | Per-model tree isolation — but note: sgl-router's `KvEventIndex::tree()` is **process-global**, not per model_id. The isolation assertion changes to: when worker A serves model M1 and worker B serves model M2, the policy only looks at workers in `healthy_workers_for(model_id)` so cross-model match is impossible at the selection layer (the tree itself may share state). Adapted accordingly. |
| `shutdown_is_idempotent` | **drop** | Covered by M3's `KvEventIndex::shutdown` tests. |
| `tokenizer_encode_error_falls_back_to_min_load` | **port-condensed** from `ZMQ` | Tokenizer encode errors → fall back. Will write a single condensed version using the `tiny_tokenizer` fixture pushed through a known-bad input (rare but worth pinning the fallback path). Actually — `dynamo_tokenizers::Tokenizer::encode` is very tolerant; if no good way to force an error, fold this into "missing tokenizer" coverage. |
| `add_worker_model_reassignment_clears_old_tree` | **drop (incompatible)** | sgl-router uses `WorkerSpec::model_ids` set at discovery time — there is no "reassign" event. Remove + add covers tree clearing. |
| `remove_worker_uses_historical_dp_size` | **drop (covered elsewhere)** | M3's `KvEventIndex::remove_worker` already remembers `dp_ranks` from `add_worker` time. |
| `unknown_worker_event_is_dropped` | **drop (covered elsewhere)** | M3's `index.rs::pump_drops_events_from_detached_workers` covers this. |

### `registry.rs::tests` (`MAIN`, 542–605)

| SMG test | Action | Why |
|---|---|---|
| `test_policy_registry_basic` | **drop (incompatible)** | SMG's per-model resolution uses worker-supplied policy hints; sgl-router resolves per-model policy purely from `Config::models[*].policy` at startup. Covered by `policies/factory.rs::registry_assigns_per_model`. |
| `test_policy_registry_cleanup` | **drop (incompatible)** | sgl-router policies are config-driven and don't lifecycle with worker counts. |
| `test_default_policy` | **drop (incompatible)** | Same. |

What we **do** need (no SMG analog): given `WorkerMode::Prefill` and
`WorkerMode::Decode` workers, the policy registry resolution correctly
filters per pool. New tests in M4 — Task 4 + gap-closer #1.

## Category 2 — Integration tests in `tests/routing/`

### `cache_aware_zmq_test.rs` (`ZMQ`, 825 lines)

| SMG test | Action | Why |
|---|---|---|
| `zmq_indexer_routes_to_publishing_worker_e2e` | **port** from `ZMQ` | Full ZMQ → SUB → mpsc → consumer → tree → routing pipeline. Adapted to sgl-router's `Policy::select` instead of SMG's `select_worker`. |
| `zmq_indexer_block_removed_clears_tree_e2e` | **drop** | M3's wire/subscriber/index tests already cover BlockRemoved end-to-end. |
| `zmq_indexer_all_blocks_cleared_e2e` | **drop** | M3 covers AllBlocksCleared. |
| `zmq_indexer_multi_batch_sequenced_delivery_e2e` | **drop** | M3 covers ordered delivery + non-dropped sequences. |
| `factory_dispatches_to_zmq_when_sync_mode_*` | **drop (incompatible)** | sgl-router only has the ZMQ variant — no `sync_mode` knob. Factory dispatch is one line in `factory.rs` and gets a one-liner test there. |
| `policy_config_*` | **drop (incompatible)** | Same. |

### `cache_aware_backward_compat_test.rs` (`ZMQ`)

| Test | Action |
|---|---|
| Backward-compat with mesh policy | **drop (incompatible)** — sgl-router never shipped mesh. |

### `pd_routing_test.rs` + `test_pd_routing.rs` (`MAIN` is canonical for both)

These exercise SMG's full HTTP request flow. The portable pieces are the
worker-mode separation patterns; everything else relies on SMG's PD bootstrap
mechanism that sgl-router doesn't have.

| SMG test | Action | Why |
|---|---|---|
| `test_pd_mode_basic_routing` | **port-adapted** from `MAIN` | Demonstrates PD prefill/decode pools route distinctly. In sgl-router: spin up MockWorkers tagged with `WorkerMode::Prefill` / `WorkerMode::Decode`, send `/v1/chat/completions`, assert only prefill workers are selected. (Decode-side handoff is M5.) |
| `test_pd_mode_round_robin` | **drop (covered)** | Round-robin already covered in unit tests. |
| `test_pd_mode_with_failing_decode_worker` | **drop (M5 scope)** | sgl-router doesn't have automatic retry/failover yet. |

The `test_pd_routing.rs` file is 920+ lines of SMG-specific format / serialization tests — none portable.

### `inflight_tracker_test.rs` + `load_guard_raii_test.rs` (`ZMQ`)

| SMG test | Action | Why |
|---|---|---|
| Inflight tracker tests | **drop (different design)** | SMG uses a global `InflightTracker` with bucket counts; sgl-router uses per-worker `AtomicUsize`. Already covered in `workers/worker.rs::tests`. |
| `test_guard_dropped_when_response_body_*` | **drop (already passing)** | Existing `LoadGuard` + `chat_routing_test::streaming_*` cover the body-attached drop semantics. |

## Category 3 — Plan-named gap closers (Task 0 bullet list)

| Gap | Status | Test target |
|---|---|---|
| **PD pool isolation under partial failure** | new test | `tests/pd_pool_isolation_test.rs`: prefill pool empty / all-breaker-open → 503 with `no_prefill_workers_available`. |
| **Active-load guard double-drop safety** | new test | Inline in `policies/active_load.rs`: Rust's affine type system makes literal double-drop unreachable. Test asserts the more interesting property — two guards on the same counter increment by exactly 2 and drop to 0 in either order. |
| **Decoder affinity vs decode-pool load** | new test | M4 dispatches both prefill and decode (dual-dispatch + bootstrap injection). The decoder-affinity logic (decode side prefers the decode worker on the same host as the chosen prefill) lives in `policies/registry.rs::decode_with_affinity`. Test: prefill on host A + decode workers on hosts {A, B} → request always lands on decode-A. |
| **Stale-request janitor expiry** | new test | Inline in `policies/active_load.rs`: register a request, advance an injected clock, run janitor → counter zeroed; idempotent on a second run. |

## Category 4 — New M4-specific tests (TDD targets)

### `policies/active_load.rs` (Task 1, new file)

1. `single_worker_increment_decrement_round_trip` — RED
2. `two_concurrent_guards_to_2_then_drop_to_0`
3. `guard_decrements_on_implicit_drop`
4. `guard_can_outlive_origin_via_arc_counter`
5. `janitor_expires_stale_requests` (gap closer)
6. `janitor_is_idempotent_on_double_run`
7. `concurrent_increment_decrement_stress` (multi-thread)

### `policies/cache_aware_zmq.rs` (Task 3, new file)

1. `empty_tree_falls_back_within_pool` — RED
2. `non_empty_tree_highest_overlap_wins`
3. `tie_break_by_lowest_active_load`
4. `imbalanced_pool_skips_cache_check`
5. `request_without_tokenizer_falls_back`
6. `lifecycle_add_then_remove_worker_clears_overlap`

### `policies/registry.rs` (Task 4, new file — sgl-router specific shape)

1. `pd_resolution_returns_distinct_pools` — RED (the PR #25184 carry-forward)
2. `prefill_pool_isolated_under_partial_failure_returns_503` (gap closer #1)
3. `plain_mode_resolution_returns_all_plain_workers`

### `tests/cache_aware_zmq_test.rs` (Task 6, e2e)

1. `zmq_indexer_routes_to_publishing_worker_e2e` — ported from
   `ZMQ`'s `cache_aware_zmq_test.rs::zmq_indexer_routes_to_publishing_worker_e2e`.
   Uses a `MockWorker` for the upstream HTTP side.

### `tests/pd_pool_isolation_test.rs` (Task 4 / gap closer #1)

End-to-end at the HTTP layer using MockWorker:
* `/v1/chat/completions` against a model with only `WorkerMode::Decode`
  workers healthy → 503 `no_prefill_workers_available`.
* `/v1/chat/completions` against a model with no workers at all → 503
  `no_healthy_workers` (existing error code).

## Carry-forward to M5

* `no_prefill_workers_available` and `stale_request_expired` codes added to
  `ApiError` in M4 (per kickoff brief).
* M5 is purely OpenAI completions + embeddings + tool-call parsing. The PD
  request path (dual dispatch + bootstrap injection) is owned end-to-end by
  M4 — M5 does not touch it.
