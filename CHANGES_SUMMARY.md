# JoyFuture SGLang Fork — Changes Summary

This document summarizes all changes made to the `ghshhf/sglang` fork,
organized by delivery month. All changes are **additive** (no behavior
modifications) and **gated** (off by default, opt-in via env vars).

---

## Month 1-2: Foundation Hardening

### Dead Code Cleanup
- Removed unused modules: `_decode_step_counter`, `SGLANG_DECODE_CLEAR_STEPS`,
  `SGLANG_KV_POOL_RETRACT_THRESHOLD_PCT`
- Fixed `bare except Exception: pass` → added `logger.debug(...)` messages
- Fixed type annotation: `RequestLatencyTracker.start_request` returns
  `Optional[RequestLatencyRecord]`

### Scheduler Environment Variables
- Added `SchedulerEnvs` class pattern in `scheduler_env_vars.py`
- Two active env vars: `SGLANG_ENABLE_PER_REQUEST_LATENCY`,
  `SGLANG_ENABLE_KV_TRANSFER_CHECKSUM`

### NVTX Profiling Guide
- Created `docs/NVTX_PROFILING_GUIDE.md` with usage instructions

---

## Month 3-4: Observability Deepening

### M3-4.1: RequestLatencyTracker → Prometheus Bridge
- Added `on_phase_complete` callback to `RequestLatencyTracker`
- Wired callback to `SchedulerMetricsCollector.observe_per_stage_req_latency`
- Feeds per-request latency phases (prefill, decode) into Prometheus histograms

### M3-4.2: KVTransferChecksumVerifier Integration
- Wired checksum recording into PD disaggregation transfer path
- `prefill.py`: `record_pre()` before `send_kv_chunk`
- `decode.py`: `record_post()` on `KVPoll.Success`

### M3-4.3: Batch Selection Latency Metric
- Added `scheduler.batch_selection` latency observation in `get_next_batch_to_run`

---

## Month 5-6: NVTX Coverage Expansion

### M5-6.1: Sub-Stage NVTX Markers
- Added `scheduler_nvtx_range` context manager helper
- 5 sub-stage markers inside `run_batch`: prebuilt, overlap, pdmux,
  non_overlap_spec, plain
- 6 sub-stage markers inside `process_batch_result`: decode, dllm,
  disagg_prefill, prefill, prebuilt, idle
- Extended `_NVTX_COLOR_MAP` with 14 new entries

### M5-6.2: Request Entry Point Markers
- `@scheduler_nvtx_method("scheduler.handle_generate_request")`
- `@scheduler_nvtx_method("scheduler.handle_batch_generate_request")`

### M5-6.3: Timeout/Abort/Idle Markers
- `@scheduler_nvtx_method("scheduler._abort_on_running_timeout")`
- `@scheduler_nvtx_method("scheduler._abort_on_queued_limit")`
- `@scheduler_nvtx_method("scheduler.abort_request")`
- `@scheduler_nvtx_method("scheduler.on_idle")`

---

## Month 7-8: Advanced Observability

### M7-8.1: FutureMap Overlap Relay Metrics
- 3 counters: `future_map_stash_total`, `future_map_publish_total`,
  `future_map_resolve_total`
- 1 histogram: `future_map_relay_latency_ms` (0.01-500ms buckets)
- Instrumented `FutureMap.publish`, `.stash`, `.resolve_seq_lens_cpu`
- Added `metrics_collector` param to `FutureMap.__init__` (backward-compatible)
- Wired through `SpeculativeAlgorithm.create_future_map`

### M7-8.2: PrefillDelayer Audit
- Audit confirmed existing `observe_prefill_delayer_outcome` covers all
  negotiation outcomes (delay, wait_success, wait_timeout, token_watermark)
  with forward_passes, wait_seconds, input_estimation labels
- No additional code needed

### M7-8.3: MinFreeSlotsDelayer Metrics
- 2 counters: `min_free_slots_delay_total`, `min_free_slots_checks_total`
- 2 histograms: `min_free_slots_running_bs`, `min_free_slots_allocatable`
- Instrumented `MinFreeSlotsDelayer.should_delay`

---

## Month 9-10: Operational Tooling

### M9-10.1: Scheduler Health Dashboard Metrics
- 3 counters: `scheduler_loop_iterations_total`, `scheduler_loop_batch_dispatches_total`,
  `scheduler_loop_idle_total`
- 1 histogram: `scheduler_loop_iteration_lag_ms` (0.01-1000ms buckets)
- 1 counter: `scheduler_aborts_total` with reason label
  (running_timeout, queue_full, waiting_timeout, user_abort)
- Instrumented all 5 event loops (normal, overlap, pp, pp_disagg_prefill,
  pp_disagg_decode)
- Instrumented 4 abort paths

### M9-10.2: NVTX Sampling Mode
- `SGLANG_SCHEDULER_NVTX_SAMPLE_RATE` env var (default 1 = always emit)
- `scheduler_nvtx_method_sampled` decorator
- `scheduler_nvtx_range_sampled` context manager
- Per-call-site counters ensure independent sampling for each marker

### M9-10.3: Performance Regression Detection Tool
- Status: Pending (not yet implemented)

---

## Environment Variables Summary

| Variable | Default | Purpose |
|----------|---------|---------|
| `SGLANG_ENABLE_PER_REQUEST_LATENCY` | false | Enable per-request latency tracking |
| `SGLANG_ENABLE_KV_TRANSFER_CHECKSUM` | false | Enable KV transfer checksum verification |
| `SGLANG_ENABLE_NVTX_SCHEDULER` | false | Enable scheduler NVTX markers |
| `SGLANG_ENABLE_NVTX_OPERATIONS` | false | Enable operations NVTX markers |
| `SGLANG_SCHEDULER_NVTX_SAMPLE_RATE` | 1 | Sample every N-th NVTX marker (1 = all) |

## Prometheus Metrics Summary

| Metric | Type | Purpose |
|--------|------|---------|
| `sglang:per_stage_req_latency_seconds` | Histogram | Per-phase request latency |
| `sglang:num_future_map_stash_total` | Counter | FutureMap stash operations |
| `sglang:num_future_map_publish_total` | Counter | FutureMap publish operations |
| `sglang:num_future_map_resolve_total` | Counter | FutureMap resolve operations |
| `sglang:future_map_relay_latency_ms` | Histogram | FutureMap relay latency |
| `sglang:min_free_slots_delay_total` | Counter | MinFreeSlotsDelayer delays |
| `sglang:min_free_slots_checks_total` | Counter | MinFreeSlotsDelayer checks |
| `sglang:scheduler_loop_iterations_total` | Counter | Event loop iterations |
| `sglang:scheduler_loop_batch_dispatches_total` | Counter | Batches dispatched |
| `sglang:scheduler_loop_idle_total` | Counter | Idle iterations |
| `sglang:scheduler_loop_iteration_lag_ms` | Histogram | Loop iteration duration |
| `sglang:scheduler_aborts_total` | Counter (reason label) | Aborted requests by reason |
