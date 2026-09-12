# CI performance diagnostics

The NVIDIA H100 one- and two-GPU diffusion jobs set
`SGLANG_DIFFUSION_DIAGNOSTICS_DIR`.
Other jobs and local runs remain opt-in. This records evidence only: it does
not change baselines, tolerances, warmup inputs, retry policy, or exit codes.
The metric and failure contracts below apply independently of this sampler.

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
- Ordinary request E2E is the existing worker forward metric, not total pytest
  wall time. Realtime E2E measures the complete requested WebSocket generation
  session, from sending initialization to receiving its frames and chunk stats;
  it is not the last chunk's latency. Startup, warmup and later MP4 encoding are
  excluded. Keep these two E2E scopes distinct when comparing results.
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

## Metric and failure contracts

Every generated testcase must report finite, positive E2E, including cases with
`run_perf_check=False`. Missing request records, absent performance logs and
missing/invalid E2E fail CI. Disabling performance checks disables threshold
comparisons, not metric reporting. Explicit GT generation skips validation.

A performance failure stops the testcase's remaining repeated requests and
subsequent checks. It also prevents pytest retries, even if another testcase
has a retryable infrastructure failure. Standalone infrastructure failures
retain their existing retry policy. Valid failed measurements are recorded
before threshold validation; realtime chunk and memory guards remain enabled
according to their existing configuration.
