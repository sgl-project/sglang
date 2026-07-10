# JoyFuture SGLang Fork — 12-Month Roadmap

Fork: `ghshhf/sglang`
Base: `sgl-project/sglang`
Focus: **Scheduler observability and operational tooling**

---

## Guiding Principle

This fork does **not** compete with upstream on features. Our niche is
**production-grade observability, profiling, and operational tooling** for the
scheduler — the component that upstream is rapidly evolving but under-instrumenting.

Every change should be:
1. **Additive** — no modifications to existing behavior
2. **Gated** — off by default, opt-in via env vars
3. **Upstreamable** — designed so that if upstream wants it, they can take it

---

## Month 1-2: Foundation Hardening (已完成 ✅)

**Status**: Complete

| Task | Result |
|------|--------|
| 扩展 NVTX 颜色映射 | ✅ 9 个 markers |
| 4 个核心函数 NVTX 装饰器 | ✅ scheduler.py |
| 4 个 PP 路径 NVTX 装饰器 | ✅ scheduler_pp_mixin.py |
| 4 个 scheduler env vars | ✅ scheduler_env_vars.py |
| RequestLatencyTracker | ✅ 已接入 scheduler 生命周期 |
| KVTransferChecksumVerifier | ✅ 模块就绪（待后续接入） |
| NVTX  profiling 指南 | ✅ docs/NVTX_PROFILING_GUIDE.md |
| 死代码清理 | ✅ 移除 4 个未使用模块 |
| 类型标注修复 | ✅ start_request Optional 返回 |
| bare except 修复 | ✅ 2 处加 debug log |
| 同步上游 | ✅ 1069 commits merged |

---

## Month 3-4: Observability Deepening

**Goal**: Make the scheduler's internal state visible to operators.

### M3-4.1: Prometheus 指标桥接

**What**: Connect `RequestLatencyTracker` to `SchedulerMetricsCollector`.

**Why**: Currently latency data only goes to Python logger. Operators need
Prometheus histograms for dashboards.

**Tasks**:
- Add `per_stage_req_latency_seconds` histogram emission in `process_batch_result`
- Add `e2e_request_latency_seconds` histogram in `end_request`
- Add `ttft_seconds` histogram (time to first token)
- Add `aborted_requests_total` counter with reason label

**Upstreamability**: ★★☆ Medium — the histogram definitions exist upstream,
we just need to feed them.

### M3-4.2: KV Checksum 接入 PD 传输路径

**What**: Wire `KVTransferChecksumVerifier` into PD disaggregation transfer code.

**Why**: The verifier module exists but is never called. This is the #1
dead code issue from our audit.

**Tasks**:
- Find PD transfer call sites in `disaggregation/prefill.py` and `disaggregation/decode.py`
- Add `pre_checksum` before transfer, `post_checksum` after transfer
- Log mismatches as warnings with request_id + layer_idx
- Add `kv_transfer_checksum_mismatches_total` Prometheus counter

**Upstreamability**: ★★★ Low — PD transfer internals are JoyFuture-specific,
but the checksum concept is generally useful.

### M3-4.3: 调度决策指标

**What**: Add metrics for the core scheduling decision point.

**Why**: `get_next_batch_to_run` is a 300-line black box. Operators have no
visibility into why batches are or aren't selected.

**Tasks**:
- Add `scheduler_batch_selection_seconds` histogram
- Add `scheduler_idle_iterations_total` counter
- Add `scheduler_retraction_reason` gauge (memory_pressure / timeout / manual)
- Add `scheduler_running_batch_size` gauge per forward mode

---

## Month 5-6: NVTX Coverage Expansion

**Goal**: Cover the remaining uninstrumented hot paths.

### M5-6.1: 子阶段 NVTX 标记 ✅

**What**: Add NVTX markers inside already-decorated methods.

**Why**: Top-level markers show *that* run_batch took 5ms, but not *why*.
Sub-stage markers break it down.

**Status**: Complete

**Delivered**:
- Inside `run_batch`: prebuilt, overlap, pdmux, non_overlap_spec, plain (5 markers)
- Inside `process_batch_result`: decode, dllm, disagg_prefill, prefill, prebuilt, idle (6 markers)
- `scheduler_nvtx_range` context manager helper added
- `_NVTX_COLOR_MAP` extended with 14 new color entries

### M5-6.2: 请求入口 NVTX 标记 ✅

**What**: Add NVTX to `handle_generate_request` and `handle_batch_generate_request`.

**Why**: Currently zero visibility into request admission latency.

**Status**: Complete

### M5-6.3: 超时和中止路径 NVTX 标记 ✅

**What**: Add NVTX to timeout/abort handling.

**Why**: These paths are invisible in profiles but can cause latency spikes.

**Status**: Complete

**Delivered**:
- `@scheduler_nvtx_method` on `_abort_on_running_timeout`, `_abort_on_queued_limit`, `abort_request`, `on_idle`
- `scheduler.abort_request` and `scheduler.on_idle` color map entries

---

## Month 7-8: Advanced Observability

**Goal**: Go beyond timing — add causal tracing and anomaly detection.

### M7-8.1: FutureMap 可观测性 ✅

**What**: Add metrics for the overlap scheduler's FutureMap relay.

**Why**: FutureMap is "always-on" but has zero observability. Operators
cannot tell if relay lag is causing issues.

**Status**: Complete

**Delivered**:
- 3 counters: stash_total, publish_total, resolve_total
- 1 histogram: relay_latency_ms (0.01-500ms buckets)
- Instrumented `publish`, `stash`, `resolve_seq_lens_cpu` in FutureMap
- Wired through `SpeculativeAlgorithm.create_future_map` with `metrics_collector`

### M7-8.2: PrefillDelayer 决策审计 ✅

**What**: Verify PrefillDelayer metrics coverage.

**Why**: Audit confirmed `observe_prefill_delayer_outcome` already covers all
negotiation outcomes (delay, wait_success, wait_timeout, token_watermark)
with forward_passes, wait_seconds, input_estimation labels.

**Status**: Complete — no additional code needed, existing coverage is comprehensive

### M7-8.3: MinFreeSlotsDelayer 指标 ✅

**What**: Add metrics for the new MinFreeSlotsDelayer (upstream just added).

**Why**: It has zero metrics — not even the basic counters that PrefillDelayer has.

**Status**: Complete

**Delivered**:
- 2 counters: delay_total, checks_total
- 2 histograms: running_bs, allocatable (0-128 buckets)
- Instrumented `MinFreeSlotsDelayer.should_delay` with metrics_collector

---

## Month 9-10: Operational Tooling

**Goal**: Build tools that operators use daily.

### M9-10.1: 调度器健康仪表板指标 ✅

**What**: Scheduler loop iteration metrics for health dashboards.

**Why**: Currently operators piece together CPU-bound vs GPU-bound vs
network-bound from 5+ different metrics sources.

**Status**: Complete

**Delivered**:
- 3 counters: loop_iterations_total, batch_dispatches_total, idle_total
- 1 histogram: iteration_lag_ms (0.01-1000ms buckets)
- 1 counter: aborts_total with reason label (running_timeout, queue_full, waiting_timeout, user_abort)
- Instrumented all 5 event loops: normal, overlap, pp, pp_disagg_prefill, pp_disagg_decode
- Instrumented 4 abort paths: running_timeout, queued_limit, waiting_timeout, user_abort

### M9-10.2: NVTX 采样模式 ✅

**What**: Allow NVTX to sample a percentage of iterations instead of all/none.

**Why**: On/off is too coarse for production. Sampling 10% gives
statistical visibility with negligible overhead.

**Status**: Complete

**Delivered**:
- `SGLANG_SCHEDULER_NVTX_SAMPLE_RATE` env var (default 1 = always emit)
- `scheduler_nvtx_method_sampled` decorator
- `scheduler_nvtx_range_sampled` context manager
- Per-call-site counters ensure independent sampling for each marker

### M9-10.3: 性能回归检测工具

**What**: A script that compares NVTX profile data across runs.

**Why**: Operators need to detect performance regressions before they hit production.

**Status**: Pending

**Tasks**:
- Create `tools/nvtx_regression_check.py`
- Parse `nsys` exported CSV timelines
- Compare per-phase timing between baseline and current
- Exit with error if any phase regresses > 10%

---

## Month 11-12: Upstream Engagement & Polish

**Goal**: Submit upstream PRs and prepare for long-term maintenance.

### M11-12.1: Upstream PR #1 — NVTX 装饰器和子阶段标记

**What**: Submit the scheduler NVTX decorators and sub-stage markers as a PR.

**Why**: Upstream already added 5 NVTX markers. Our 8 additional markers
complement theirs without overlap. The `scheduler_nvtx_range` context manager
and sampling variants are also new contributions.

**Status**: Ready for submission

**Changes to include**:
- `scheduler_nvtx_range` context manager (new in JoyFuture)
- `scheduler_nvtx_method_sampled` / `scheduler_nvtx_range_sampled`
- 8 new decorator markers in scheduler.py
- 4 new decorator markers in scheduler_pp_mixin.py
- 14 new color map entries in nvtx_utils.py

**Upstreamability**: ★★★ Medium — depends on NVTX marker naming review

### M11-12.2: Upstream PR #2 — Scheduler Metrics 基础设施

**What**: Submit the FutureMap metrics and scheduler health dashboard metrics.

**Why**: These are additive observability improvements that benefit all
SGLang operators.

**Status**: Ready for submission

**Changes to include**:
- FutureMap metrics: stash/publish/resolve counters + relay latency histogram
- Scheduler health: loop iteration counters + lag histogram + abort reason counter
- MinFreeSlotsDelayer metrics

**Upstreamability**: ★★★ High — purely additive, no behavior changes

### M11-12.3: 文档完善

**What**: Fill the documentation gaps identified in our audit.

**Tasks**:
- Write `docs/scheduler_architecture.md` — event loop, overlap, PP
- Write `docs/production_tuning.md` — env var tuning guide
- Write `docs/observability_setup.md` — Prometheus + Grafana + Nsight
- Update `CHANGES_SUMMARY.md` with quarterly summaries

### M11-12.4: 长期维护模式

**What**: Establish a sustainable sync and release cadence.

**Tasks**:
- Set up monthly upstream sync (cherry-pick or merge)
- Set up CI that runs on every push to fork
- Tag quarterly releases (`joyfuture-v1.0`, `v1.1`, etc.)
- Document contribution workflow for future team members

---

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Upstream rewrites scheduler, invalidating our markers | Medium | High | Stay synced monthly; markers are additive, easy to reapply |
| Upstream adopts competing observability approach | Low | Medium | Our markers are complementary, not conflicting |
| Network instability prevents pushing to GitHub | Medium | Low | Already experienced; use SSH or alternative transport |
| PD disaggregation architecture changes | Medium | Medium | KV checksum is gated behind env var, safe to leave disconnected |
| Scheduler.py grows beyond maintainable size | High | Medium | Consider extracting mixins in M10+ |

---

## Success Metrics (12-Month)

| Metric | Target |
|--------|--------|
| NVTX markers in scheduler hot path | 20+ (from 8 today) |
| Prometheus histograms for request latency | 5+ (from 0 today) |
| Dead code ratio | 0% (from ~4 modules today) |
| Upstream PRs submitted | 2-3 |
| Upstream PRs merged | 1+ |
| Documentation pages | 6+ (from 1 today) |
| CI pass rate | 100% |

---

## Appendix: What We Will NOT Do

To maintain focus, the following are explicitly out of scope:

1. **Model support** — no new model architectures, quantization, or kernels
2. **Feature parity** — we will not chase every upstream feature
3. **Breaking changes** — all changes must be additive
4. **GUI / Web dashboard** — we use Prometheus + Grafana, not custom UIs
5. **Language bindings** — Python only
