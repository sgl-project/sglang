---
name: ci-workflow-guide
description: Guide to SGLang CI workflow orchestration — stage ordering, fast-fail, gating, partitioning, execution modes, and debugging CI failures. Use when modifying CI workflows, adding stages, debugging CI pipeline issues, or understanding how tests are dispatched and gated across stages.
---

# SGLang CI Workflow Orchestration Guide

This skill covers the CI **infrastructure** layer — how tests are dispatched, gated, and fast-failed across stages. For test authoring (templates, fixtures, registration, model selection), see the [write-sglang-test skill](../write-sglang-test/SKILL.md).

---

## Naming Conventions

- **Suite**: `stage-{a,b,c}-test-{gpu_count}-gpu-{hardware}` (e.g., `stage-b-test-1-gpu-small`)
- **CI runner**: `{gpu_count}-gpu-{hardware}` (e.g., `1-gpu-5090`, `4-gpu-h100`, `8-gpu-h200`)

---

## Key Files

| File | Role |
|------|------|
| `.github/workflows/pr-test.yml` | Main workflow — all stages, jobs, conditions, matrix definitions |
| `.github/workflows/pr-gate.yml` | PR gating: draft check, `run-ci` label, per-user rate limiting |
| `.github/actions/check-stage-health/action.yml` | Cross-job fast-fail: queries API for any failed job |
| `.github/actions/wait-for-jobs/action.yml` | Stage gating: polls API until stage jobs complete |
| `.github/actions/check-maintenance/action.yml` | Maintenance mode check |
| `test/run_suite.py` | Suite runner: collects, filters, partitions, executes tests |
| `python/sglang/test/ci/ci_register.py` | Test registration (AST-parsed markers), LPT auto-partition |
| `python/sglang/test/ci/ci_utils.py` | `run_unittest_files()`: execution, retry, continue-on-error |
| `scripts/ci/utils/slash_command_handler.py` | Handles slash commands from PR comments |

---

## Architecture Overview

```
 ┌──────────────┐
 │ build kernel │
 └──────┬───────┘
        │
        ├─ check-changes ──── detects which packages changed
        │                      (main_package, sgl_kernel, jit_kernel, multimodal_gen)
        │
        ├─ call-gate ──────── pr-gate.yml (draft? label? rate limit?)
        │
        ├─────────────────────────────────────────────────────┐
        │                                                     │
        ▼                                                     │
 ┌─────────────────────────────────────┐                      │
 │          Stage A (~3 min)           │                      │
 │         pre-flight check            │                      │
 │                                     │                      │
 │  ┌─────────────────────────────┐    │                      │
 │  │ stage-a-test-1-gpu-small    │    │                      │
 │  │ (small GPUs)                │    │                      │
 │  └─────────────────────────────┘    │                      │
 │  ┌─────────────────────────────┐    │                      │
 │  │ stage-a-test-cpu            │    │                      │
 │  │ (CPU)                       │    │                      │
 │  └─────────────────────────────┘    │                      │
 └──────┬──────────────────────────────┘                      │
        │                                                     │
        ▼                                                     ▼
 ┌─────────────────────────────────────┐          ┌──────────────────────────┐
 │          Stage B (~30 min)          │          │      kernel test         │
 │           basic tests               │          └──────────────────────────┘
 │                                     │          ┌──────────────────────────┐
 │  ┌─────────────────────────────┐    │          │   multimodal gen test    │
 │  │ stage-b-test-1-gpu-small    │    │          └──────────────────────────┘
 │  │ (small GPUs, e.g. 5090)     │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-b-test-1-gpu-large    │    │
 │  │ (large GPUs, e.g. H100)     │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-b-test-2-gpu-large    │    │
 │  │ (large GPUs, e.g. H100)     │    │
 │  └─────────────────────────────┘    │
 └──────┬──────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────┐
 │          Stage C (~30 min)          │
 │          advanced tests             │
 │                                     │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-4-gpu-h100     │    │
 │  │ (H100 GPUs)                 │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-8-gpu-h200     │    │
 │  │ (8 x H200 GPUs)             │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ stage-c-test-4-gpu-b200     │    │
 │  │ (4 x B200 GPUs)             │    │
 │  └─────────────────────────────┘    │
 │  ┌─────────────────────────────┐    │
 │  │ Other advanced tests        │    │
 │  │ (DeepEP, PD Disagg, GB300)  │    │
 │  └─────────────────────────────┘    │
 └──────┬──────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────┐
 │         pr-test-finish              │
 │  aggregates all results, fails if   │
 │  any job failed/cancelled           │
 └─────────────────────────────────────┘
```

**Every stage test job** includes a `check-stage-health` step after checkout — if any job in the run has already failed, the job fast-fails (red X) with a root cause annotation.

**Scheduled runs** skip `wait-for-stage-*` jobs, running all stages in parallel. Fast-fail is also disabled.

---

## Fast-Fail Layers

4 layers of fast-fail, from fine to coarse:

| Layer | Mechanism | Granularity | Disabled on schedule? |
|-------|-----------|-------------|----------------------|
| **1. Test method → file** | `unittest -f` (failfast) | One test method fails → entire test file stops immediately | Yes |
| **2. File → suite** | `run_unittest_files()` default | One test file fails → entire suite stops (`--continue-on-error` off) | Yes |
| **3. Job → job (same stage)** | `check-stage-health` action | One job fails → other waiting jobs in same stage fast-fail (red X) | Yes |
| **4. Stage → stage (cross-stage)** | `wait-for-stage` + `needs` | Stage A fails → stage B/C jobs skip entirely (never get a runner) | Yes (wait jobs skipped) |

- **Layer 1**: `-f` flag appended to all `python3 -m pytest` / `unittest` invocations in `ci_utils.py`
- **Layer 2**: `--continue-on-error` flag in `run_suite.py` — off for PRs, on for scheduled runs
- **Layer 3**: `check-stage-health` auto-detects `schedule` event and skips; filters out cascade failures to show only root cause jobs
- **Layer 4**: `wait-for-stage-*` jobs are conditioned on `github.event_name == 'pull_request'` — skipped for scheduled runs

---

## Execution Modes

| Aspect | PR (`pull_request`) | Scheduled (`cron`, every 6h) | `/rerun-stage` (`workflow_dispatch`) |
|--------|---------------------|------------------------------|--------------------------------------|
| **Stage ordering** | Sequential: A → B → C via `wait-for-stage-*` | Parallel (all at once) | Single target stage only |
| **Cross-job fast-fail** | Yes (`check-stage-health`) | Yes | Yes |
| **continue-on-error** | No (stop at first failure within suite) | Yes (run all tests) | No |
| **Retry** | Enabled | Enabled | Enabled |
| **max_parallel** | 3 (default), 14 if `high priority` label | 14 | 3 (default), 14 if `high priority` |
| **PR gate** | Yes (draft, label, rate limit) | Skipped | Skipped |
| **Concurrency** | `cancel-in-progress: true` per branch | Queue (no cancel) | Isolated per stage+SHA |

---

## Stage Gating (`wait-for-jobs` action)

`wait-for-stage-a` and `wait-for-stage-b` are lightweight `ubuntu-latest` jobs that poll the GitHub Actions API.

**How it works:**
1. Calls `listJobsForWorkflowRun` to list all jobs in the current run
2. Matches jobs by exact name or prefix (for matrix jobs, e.g., `stage-b-test-1-gpu-small (3)`)
3. If any matched job has `conclusion === 'failure'` → fail immediately (fast-fail)
4. If all matched jobs are completed and count matches `expected_count` → success
5. Otherwise → sleep `poll-interval-seconds` (default: 60s) and retry
6. Timeout after `max-wait-minutes` (240 min for stage-a, 480 min for stage-b)

**Job specs example** (stage-b):
```json
[
  {"prefix": "stage-b-test-1-gpu-small", "expected_count": 8},
  {"prefix": "stage-b-test-1-gpu-large", "expected_count": 14},
  {"prefix": "stage-b-test-2-gpu-large", "expected_count": 4},
  {"prefix": "stage-b-test-4-gpu-b200", "expected_count": 1}
]
```

> **Critical**: `expected_count` must match the matrix size. If you add/remove matrix entries, update the wait job's spec accordingly.

**PR only**: Condition `github.event_name == 'pull_request' && !inputs.target_stage` — scheduled runs and `/rerun-stage` skip these entirely, allowing parallel execution.

---

## Cross-Job Fast-Fail (`check-stage-health` action)

Composite action called after checkout in every stage test job (21 jobs total across `pr-test.yml`, `pr-test-multimodal-gen.yml`, `pr-test-sgl-kernel.yml`, `pr-test-jit-kernel.yml`).

**How it works:**
1. Queries `listJobsForWorkflowRun` for the current workflow run
2. Filters for **root cause failures only** — jobs with `conclusion === 'failure'` whose failing step is NOT `check-stage-health` (excludes cascade failures)
3. If root cause failures found → calls `core.setFailed()` with the list of root cause job names
4. If none → does nothing (step succeeds)

**Cascade filtering**: When job A fast-fails due to health check, it also has `conclusion: failure`. Without filtering, job B would list both the original failure AND job A's fast-fail. The filter checks each failed job's `steps` array — if the failing step name contains `check-stage-health` or `Check stage health`, it's excluded from the root cause list.

**Usage pattern:**
```yaml
steps:
  - name: Checkout code
    uses: actions/checkout@v4
    ...

  - uses: ./.github/actions/check-stage-health
    id: stage-health

  - name: Install dependencies        # skipped automatically if health check failed
    ...                                # (default if: success() is false)

  - name: Run test                     # also skipped
    ...
```

**Visual effect**: Job shows **red X** (failure) with error annotation showing root cause job names. Subsequent steps are naturally skipped (default `if: success()` is false after a failed step). No per-step `if` guards needed.

**No stage filtering**: Checks ALL jobs in the run, not just the current stage. Any failure anywhere triggers fast-fail.

**Error message example:**
```
Fast-fail: skipping — root cause job(s): stage-b-test-1-gpu-small (0), stage-b-test-1-gpu-small (1)
```

---

## Within-Suite Failure Handling

Controlled by `run_unittest_files()` in `python/sglang/test/ci/ci_utils.py`.

### Flags

| Flag | PR default | Scheduled default | Effect |
|------|------------|-------------------|--------|
| `--continue-on-error` | Off | On | Off: stop at first failure. On: run all files, report all failures at end |
| `--enable-retry` | On | On | Retry retriable failures (accuracy/perf assertions) |
| `--max-attempts` | 2 | 2 | Max attempts per file including initial run |

### Retry Classification

When a test fails and retry is enabled, the output is classified:

**Non-retriable** (checked first — real code errors):
`SyntaxError`, `ImportError`, `ModuleNotFoundError`, `NameError`, `TypeError`, `AttributeError`, `RuntimeError`, `CUDA out of memory`, `OOM`, `Segmentation fault`, `core dumped`, `ConnectionRefusedError`, `FileNotFoundError`

**Retriable** (accuracy/performance):
`AssertionError` with comparison patterns (`not greater than`, `not less than`, `not equal to`), `accuracy`, `score`, `latency`, `throughput`, `timeout`

**Default**: Unknown `AssertionError` → retriable. Other unknown failures → not retriable.

### How `continue_on_error` is set

In `pr-test.yml`'s `check-changes` job:
- `schedule` runs or `run_all_tests` flag → `continue_on_error = 'true'`
- PR runs → `continue_on_error = 'false'`

Each test job propagates via:
```yaml
env:
  CONTINUE_ON_ERROR_FLAG: ${{ needs.check-changes.outputs.continue_on_error == 'true' && '--continue-on-error' || '' }}
run: |
  python3 run_suite.py --hw cuda --suite <name> $CONTINUE_ON_ERROR_FLAG
```

---

## Test Partitioning

Large suites are split across matrix jobs using the **LPT (Longest Processing Time) heuristic** in `ci_register.py:auto_partition()`:

1. Sort tests by `est_time` descending, filename as tie-breaker (deterministic)
2. Greedily assign each test to the partition with smallest cumulative time
3. Result: roughly equal total time per partition

**Partition table** (CUDA per-commit suites):

| Suite | Partitions | Runner | max_parallel |
|-------|-----------|--------|-------------|
| `stage-a-test-1-gpu-small` | 1 (no matrix) | `1-gpu-5090` | — |
| `stage-a-test-cpu` | 4 | `ubuntu-latest` | — |
| `stage-b-test-1-gpu-small` | 8 | `1-gpu-5090` | 8 |
| `stage-b-test-1-gpu-large` | 14 | `1-gpu-h100` | dynamic (3 or 14) |
| `stage-b-test-2-gpu-large` | 4 | `2-gpu-h100` | — |
| `stage-b-test-4-gpu-b200` | 1 (no matrix) | `4-gpu-b200` | — |
| `stage-b-kernel-unit-1-gpu-large` | 1 (no matrix) | `1-gpu-h100` | — |
| `stage-b-kernel-unit-8-gpu-h200` | 1 (no matrix) | `8-gpu-h200` | — |
| `stage-b-kernel-benchmark-1-gpu-large` | 1 (no matrix) | `1-gpu-h100` | — |
| `stage-c-test-4-gpu-h100` | 3 | `4-gpu-h100` | — |
| `stage-c-test-8-gpu-h200` | 4 | `8-gpu-h200` | — |
| `stage-c-test-8-gpu-h20` | 2 | `8-gpu-h20` | — |
| `stage-c-test-deepep-4-gpu-h100` | 1 (no matrix) | `4-gpu-h100` | — |
| `stage-c-test-deepep-8-gpu-h200` | 1 (no matrix) | `8-gpu-h200` | — |
| `stage-c-test-4-gpu-b200` | 4 | `4-gpu-b200` | — |
| `stage-c-test-4-gpu-gb200` | 1 (no matrix) | `4-gpu-gb200` | — |

> **Note**: Kernel suites (`stage-b-kernel-*`) run via `pr-test-jit-kernel.yml` and `pr-test-sgl-kernel.yml`, not the main `pr-test.yml`. Multimodal diffusion uses `python/sglang/multimodal_gen/test/run_suite.py`, not `test/run_suite.py`.

**Workflow usage:**
```yaml
strategy:
  matrix:
    partition: [0, 1, 2, 3, 4, 5, 6, 7]
steps:
  - run: python3 run_suite.py --hw cuda --suite stage-b-test-1-gpu-small \
           --auto-partition-id ${{ matrix.partition }} --auto-partition-size 8
```

---

## check-changes Job

Determines which test suites to run based on file changes.

### Detection Methods

| Trigger | Method | Details |
|---------|--------|---------|
| `pull_request` | `dorny/paths-filter` | Detects changes via GitHub diff |
| `workflow_dispatch` (with `pr_head_sha`) | GitHub API | `repos/{repo}/compare/main...{sha}` |
| `schedule` / `run_all_tests` | Force all true | Runs everything |

### Output Flags

| Output | Triggers |
|--------|----------|
| `main_package` | Stage A/B/C test suites |
| `sgl_kernel` | Kernel wheel builds + kernel test suites |
| `jit_kernel` | JIT kernel test workflow |
| `multimodal_gen` | Multimodal-gen test workflow |

> **Note**: `sgl_kernel` is forced to `false` when `target_stage` is set, because `sgl-kernel-build-wheels` won't run and wheel artifacts won't be available.

---

## Concurrency Control

```
group: pr-test-{event_name}-{branch}-{pr_sha}-{stage}
```

| Segment | Source | Purpose |
|---------|--------|---------|
| `event_name` | `github.event_name` | Prevents scheduled runs colliding with fork PRs named `main` |
| `branch` | `github.head_ref \|\| github.ref_name` | Per-branch isolation |
| `pr_sha` | `inputs.pr_head_sha \|\| 'current'` | Isolates `/rerun-stage` from main runs |
| `stage` | `inputs.target_stage \|\| 'all'` | Allows parallel stage dispatches |

`cancel-in-progress: true` for `pull_request` events (new push cancels old run), `false` for `workflow_call`.

---

## How To: Add a New Stage Job

1. Define the job in `pr-test.yml` with `needs: [check-changes, call-gate, wait-for-stage-X, ...]`
2. Copy the `if:` condition pattern from an existing same-stage job (handles `target_stage`, `schedule`, `main_package`)
3. Add `checkout` step
4. Add `check-stage-health` step (after checkout) — if any prior job failed, `core.setFailed()` fires and all subsequent steps auto-skip via default `if: success()`
5. Add `check-maintenance` step
6. Add `download-artifact` step if `sgl_kernel` changed
7. Add `install dependencies` step
8. Add `run test` step with `$CONTINUE_ON_ERROR_FLAG`
9. Add `upload-cuda-coredumps` step with `if: always()`
10. Register the suite name in `PER_COMMIT_SUITES` in `test/run_suite.py`
11. If using matrix, add `--auto-partition-id` and `--auto-partition-size` to the run command
12. **Update `wait-for-stage-X`** job spec with the new job name and `expected_count` (if matrix)
13. **Add the job to `pr-test-finish.needs`** list

---

## How To: Debug CI Failures

| Symptom | Likely cause | What to check |
|---------|-------------|---------------|
| All stage-B/C jobs green but steps skipped | Earlier job failed, `check-stage-health` triggered | Find the actual failed job (red X) |
| `wait-for-stage-b` timeout | `expected_count` doesn't match matrix size | Verify job spec counts match `matrix:` array length |
| `pr-test-finish` fails but all jobs green | A job was `cancelled` (counts as failure in finish) | Check concurrency cancellation |
| Tests pass locally but fail in CI | Partition assignment, runner GPU type, or `est_time` inaccuracy | Check which partition the test lands in; verify runner label |
| Flaky test retried and passed | Retriable failure (accuracy/perf) | Check `[CI Retry]` markers in job logs |
| Flaky test NOT retried | Matched non-retriable pattern | Check if error matches `NON_RETRIABLE_PATTERNS` in `ci_utils.py` |

---

## Slash Commands

| Command | Effect |
|---------|--------|
| `/tag-run-ci-label` | Adds `run-ci` label to PR |
| `/rerun-failed-ci` | Reruns failed jobs in the latest workflow run |
| `/tag-and-rerun-ci` | Adds label + reruns |
| `/rerun-stage <stage>` | Dispatches `pr-test.yml` with `target_stage=<stage>` |
| `/rerun-test <test-file>` | Reruns a specific test file via `rerun-test.yml` |

Handled by `scripts/ci/utils/slash_command_handler.py` → `.github/workflows/slash-command-handler.yml`.
