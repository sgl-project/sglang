# SGLang Fork Changes Summary

Fork: `ghshhf/sglang` (based on `sgl-project/sglang`)
Author: JoyFuture
Purpose: Scheduler observability and tuning tooling for SGLang

---

## Changes Overview

This fork adds scheduler-level observability and operational tuning knobs to
upstream SGLang. All changes are additive — no existing behavior is modified.

### Upstreamability Assessment

| Change | Files | Lines Changed | Upstreamability |
|--------|-------|---------------|-----------------|
| Extended NVTX markers | 1 modified | ~10 additions | ★★★ Low — JoyFuture-specific profiling |
| Scheduler NVTX decorators | 1 modified | 4 additions | ★★☆ Medium — depends on NVTX marker review |
| Scheduler env vars | 1 new | ~8 | ★★☆ Medium — env var naming needs team discussion |
| Request latency tracker | 1 new | ~140 | ★☆☆ Low — JoyFuture-specific logging |
| KV transfer checksum | 1 new | ~90 | ★★☆ Medium — useful for PD disaggregation debugging |
| NVTX profiling guide | 1 new | ~120 | ★★☆ Medium — docs, but JoyFuture-specific content |

---

## Detailed Change Log

### 1. Extended NVTX Markers (`python/sglang/srt/utils/nvtx_utils.py`)

**What**: Added 4 new NVTX color markers for JoyFuture's extended scheduler events.

**Change**:
- Added `"scheduler.event_loop_normal"` → dark_blue
- Added `"scheduler.event_loop_overlap"` → teal
- Added `"scheduler.update_running_batch"` → orange
- Added `"scheduler.get_new_batch_prefill"` → magenta
- Updated module docstring to reference JoyFuture extensions

**Why**: These markers allow profiling the event loop mode and internal batch
selection logic with Nsight Systems.

**Upstream notes**: The color assignments are arbitrary; upstream would decide
standard colors or naming conventions.

### 2. Scheduler NVTX Decorators (`python/sglang/srt/managers/scheduler.py`)

**What**: Wrapped 4 scheduler methods with `@nvtx_annotated_method`.

**Decorators added**:
1. `event_loop_normal` (line ~1470) — "A normal scheduler loop"
2. `event_loop_overlap` (line ~1498) — "A scheduler loop that overlaps the CPU processing and GPU computation"
3. `get_new_batch_prefill` (line ~2614) — "Selecting new prefill batch"
4. `update_running_batch` (line ~2908) — "Update the current running decoding batch"

**Why**: These are the highest-level scheduler boundaries. Instrumenting them
provides a clear timeline of scheduler phases in Nsight Systems.

**Pattern**: Follows the existing decorator pattern used for other methods
(e.g., `dispatch_event_loop`, `dispatch_prefill_batch`).

**Upstream notes**: These specific function names are unlikely to change, but
the decorator addition should be coordinated with the upstream NVTX marker plan.

### 3. Scheduler Environment Variables (`python/sglang/srt/scheduler_env_vars.py`) — NEW

**What**: Added 4 environment variables for runtime scheduler tuning.

**Variables**:
| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SGLANG_ENABLE_PER_REQUEST_LATENCY` | bool | `False` | Enable per-request latency tracking |
| `SGLANG_ENABLE_KV_TRANSFER_CHECKSUM` | bool | `False` | Enable SHA-256 checksum verification for KV transfers |
| `SGLANG_DECODE_CLEAR_STEPS` | int | `0` | Number of decode steps between KV cache clear operations |
| `SGLANG_KV_POOL_RETRACT_THRESHOLD_PCT` | int | `0` | Percentage threshold (0-100) to trigger KV pool retraction |

**Why**: Provides operational tuning knobs without code changes. The latency
tracker and checksum features depend on these env vars.

**Pattern**: Follows the existing `sglang.srt.environ` pattern with `env_bool`
and `env_int` helpers.

**Upstream notes**: The env var names follow upstream conventions (`SGLANG_*`).
Naming and defaults would need team review before submission.

### 4. Request Latency Tracker (`python/sglang/srt/request_latency_tracker.py`) — NEW

**What**: Per-request latency tracking infrastructure.

**Classes**:
- `PhaseLatency` — Latency breakdown for a single phase (prefill, decode, transfer, sample)
- `RequestLatencyRecord` — Complete lifecycle record with TTFT, total elapsed, tokens/sec
- `RequestLatencyTracker` — Central registry managing all in-flight requests

**Features**:
- Phase-based timing (`start_phase` / `end_phase`)
- TTFT (Time To First Token) computation
- Retraction count tracking
- Abort tracking with reason
- Aggregated summary statistics (avg, p50, p95)

**Why**: Enables debugging of agentic workloads where individual request
latency matters more than aggregate throughput.

**Upstream notes**: The structured logging format (`to_log_dict`) is JoyFuture-specific.
The core timing logic could be useful upstream, but the logging integration would need adaptation.

### 5. KV Transfer Checksum (`python/sglang/srt/kv_transfer_checksum.py`) — NEW

**What**: SHA-256 checksum verification for PD disaggregation KV cache transfers.

**Classes**:
- `compute_kv_checksum()` — Computes a deterministic SHA-256 hash of a KV tensor
- `KVTransferChecksumRecord` — Records a single transfer with before/after checksums
- `KVTransferChecksumVerifier` — Batch verification tool for comparing source/destination checksums

**Why**: PD disaggregation transfers KV cache between prefill and decode nodes.
Without checksums, silent data corruption could go undetected.

**Pattern**: Uses `hashlib` (stdlib) + `torch` tensor operations. No external
dependencies beyond what SGLang already requires.

**Upstream notes**: The checksum computation is generally useful for PD
disaggregation debugging. The integration point would need to be determined.

### 6. NVTX Profiling Guide (`docs/NVTX_PROFILING_GUIDE.md`) — NEW

**What**: User-facing documentation for Nsight Systems profiling with SGLang.

**Contents**:
- Quick start (enable NVTX, profile with nsys, view results)
- Color-coded marker reference table
- Event loop mode comparison (normal vs overlap)
- Common bottleneck identification patterns
- Troubleshooting guide
- Advanced usage (selective profiling, export, CI integration)

**Why**: SGLang has existing profiling docs but no dedicated NVTX/Nsight guide.

**Upstream notes**: Most content is generic Nsight Systems usage. The marker
reference table is specific to this fork's additions.

---

## Dependencies

No new external dependencies. All new files use only:
- Python standard library (`hashlib`, `logging`, `dataclasses`, `time`, `typing`)
- `torch` (already a SGLang dependency)
- `sglang.srt.environ` (already exists in the codebase)
- `sglang.srt.utils.nvtx_utils` (already exists in the codebase)

---

## Testing Status

- All new files follow the project's Apache 2.0 license header
- All new files follow the project's import and style conventions
- The `scheduler_env_vars.py` module was tested for import correctness
- NVTX marker decorators use the same pattern as existing ones
- No existing tests were modified or broken

---

## Files Changed

```
Modified:
  python/sglang/srt/utils/nvtx_utils.py       (NVTX color map extended)
  python/sglang/srt/managers/scheduler.py     (4 NVTX decorators added)

Added:
  python/sglang/srt/scheduler_env_vars.py     (4 env vars)
  python/sglang/srt/request_latency_tracker.py (per-request latency)
  python/sglang/srt/kv_transfer_checksum.py   (KV transfer verification)
  docs/NVTX_PROFILING_GUIDE.md                (profiling documentation)
```

---

## Recommendations for Next Steps

1. **Short-term (this fork)**: Wire up the latency tracker in the scheduler
   pipeline — currently defined but not yet connected to request lifecycle hooks.

2. **Medium-term**: Integrate KV checksum verification into PD disaggregation
   transfer code paths when `SGLANG_ENABLE_KV_TRANSFER_CHECKSUM=1`.

3. **Long-term (upstream PR)**: Submit NVTX decorator additions and env var
   infrastructure as a PR to `sgl-project/sglang` after internal review.
