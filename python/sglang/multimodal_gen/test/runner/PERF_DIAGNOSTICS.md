# CI performance diagnostics

The NVIDIA two-GPU diffusion job sets `SGLANG_DIFFUSION_DIAGNOSTICS_DIR`.
Other jobs and local runs remain opt-in. This records evidence only: it does
not change baselines, tolerances, warmup inputs, retry policy, or exit codes.

Each invocation and retry gets a unique `attempt-N-*` directory:

| Artifact | Contents |
| --- | --- |
| `events.jsonl` | Actual checkout SHA, selected dependency versions, CI run/attempt/partition, observed case/module/worker/stage log boundaries, and subprocess exit status |
| `requests.jsonl` | Every formal request's existing E2E/stage/step/memory metrics, flushed before assertions, including cases with `run_perf_check=False` |
| `resources.jsonl` | Approximately 1 Hz process-tree CPU time, I/O, RSS, page faults and context switches; attributed GPU utilization, clocks, power/limit, temperature and throttle reasons |

JSONL is flushed per event so completed evidence survives interrupted pytest
sessions. A missing `attempt_end` means incomplete, not success. The workflow
uploads these files with `always()` separately from the existing final-result
report, so retries do not erase failed samples. Abrupt runner loss can still
prevent artifact upload.

## Interpretation

- `observed_boundary` timestamps are **stdout receipt times**, not synchronized
  CUDA timings. Use case-begin to workers-ready to locate startup, not the first
  tqdm `0/N` refresh, which may occur well into warmup. Do not treat this as a
  precise loading assertion. Module loading is a subset of startup.
- Request E2E is the existing worker forward metric, not total pytest wall time.
  Warmup, output saving, consistency checks and retries are separate costs.
- Cumulative CPU/I/O/fault counters must be differenced by `(pid, created)`.
  Processes shorter than a sample interval may be missed. Zero observed disk
  reads do not prove a warm cache or absence of I/O.
- GPU ownership is matched to live descendants, not CUDA ordinal assumptions.
  Before CUDA context creation there may be no attributed GPU. NVML/process PID
  namespace mismatches can also leave GPU samples empty; this is missing data,
  not zero utilization. Unsupported counters are explicit errors, not zero.
- Software power capping is not automatically faulty hardware. Compare its
  bitmask, power limit and clocks across equivalent runs. Resource samples alone
  cannot prove a kernel, synchronization or storage root cause; a focused trace
  may still be necessary. The sampler adds overhead; measure it before making
  performance claims from instrumented runs.

No arbitrary environment, command lines, prompts, file contents or full logs
are saved. Only allowlisted log markers and numeric process/resource data are
collected. Sampling runs in the pytest parent, never adds CUDA synchronization,
does not attach a debugger, and does not change device settings.
