---
name: ci-workflow-guide
description: Guide to SGLang CI workflow orchestration — tiers and their workflows, stage ordering, fail-fast, gating, partitioning, execution modes, adding a runner to a tier, and debugging CI failures. Use when modifying CI workflows, adding stages or runners, debugging CI pipeline issues, or understanding how tests are dispatched and gated across stages.
---

# SGLang CI Workflow Orchestration Guide

This skill covers the CI **infrastructure** layer — how tests are dispatched, gated, and aborted on failure across stages. For test authoring (templates, fixtures, registration, which tier a test belongs to), see the [write-sglang-test skill](../write-sglang-test/SKILL.md). To run the tests a change touches on a PR, see [rerun-test](../rerun-test/SKILL.md).

The workflow files settle any disagreement with this guide. Counts that change with the test tree (shard counts, job counts, suite lists) are left out on purpose; read them from the files named below.

---

## Naming Conventions

- **Suite**: `{stage}-test-{runner_config}`, generated from each test's `register_cuda_ci(stage=..., runner_config=...)` (e.g. `base-b-test-1-gpu-small`, `nightly-test-8-gpu-b200`)
- **runner_config**: a runner-pool key (`1-gpu-small`, `4-gpu-h100`, `2-gpu-gb300`); `scripts/ci/runner_configs.yml` maps it to the physical runner label. Runner configs are independent of tiers: any tier can run any runner config.
- **Test group**: directory-level group under `test/registered/` (e.g. `hicache` maps to `test/registered/hicache/test_*.py`)

---

## Tiers

| Tier (stages) | Workflow | When it runs |
|---------------|----------|--------------|
| base (`base-a`, `base-b`, `base-c`) | `pr-test.yml` | Every PR with `run-ci`; on main twice daily (cron) |
| extra (`extra-a`, `extra-b`) | `pr-test-extra.yml` | PRs with `run-ci` and `run-ci-nv-extra`; on main from `pr-test.yml`'s scheduled run (`call-pr-test-extra`) |
| nightly (`nightly`) | `nightly-test-nvidia.yml` | Every two days, 14:00 UTC |
| weekly (`weekly`) | `weekly-test-nvidia.yml` | Sunday 00:00 UTC |

`pr-test-finish` (base) and `lint` are the required checks for merging. Which tier a test belongs to is in [write-sglang-test, Choosing a Tier](../write-sglang-test/SKILL.md#choosing-a-tier). `/rerun-test` runs any registered test on a PR, whatever its tier.

---

## Key Files

| File | Role |
|------|------|
| `.github/workflows/pr-test.yml` | Base tiers: stage jobs, wait jobs, `pr-test-finish` |
| `.github/workflows/pr-test-extra.yml` | Extra tier, gated by `run-ci` and `run-ci-nv-extra` |
| `.github/workflows/nightly-test-nvidia.yml`, `weekly-test-nvidia.yml` | Scheduled tiers: one matrix row per runner config |
| `.github/workflows/_pr-test-stage.yml`, `_pr-test-stage-cpu.yml` | Reusable stage: checkout, health check, install, `run_suite.py` per shard |
| `.github/workflows/_pr-test-check-changes.yml` | Path filters, partitions, retry and continue-on-error flags |
| `scripts/ci/utils/compute_partitions.py` | Shard count and `max_parallel` per suite |
| `scripts/ci/runner_configs.yml` | `runner_config` -> runner label |
| `test/run_suite.py` | Suite lists (`PER_COMMIT_SUITES`, `NIGHTLY_SUITES`, `OTHER_SUITES`); collects, filters, partitions, executes tests |
| `python/sglang/test/ci/ci_register.py` | Test registration (AST-parsed markers), LPT auto-partition |
| `python/sglang/test/ci/ci_utils.py` | `run_unittest_files()`: execution, retry, continue-on-error |
| `.github/workflows/pr-gate.yml` | PR gating: draft check, `run-ci` label, per-user rate limiting |
| `.github/actions/check-pr-test-health/action.yml` | Cross-job fail-fast: queries API for any failed job |
| `.github/actions/wait-for-jobs/action.yml` | Stage gating: polls API until stage jobs complete |
| `.github/scripts/ci-labels.cjs` | Resolves the CI control labels into dispatch axes |
| `scripts/ci/opt_in_ci_tiers.json` | Opt-in CI selections: name -> workflow files and an optional label |
| `scripts/ci/utils/slash_command_handler.py` | Handles slash commands from PR comments |

---

## Architecture Overview

`pr-test.yml` on a PR:

1. **`check-changes`** detects which packages changed (`main_package`, `sgl_kernel`, `jit_kernel`, `multimodal_gen`, ...) and computes the partitions; **`call-gate`** runs `pr-gate.yml` (draft, label, rate limit). Kernel and rust-ext builds start alongside.
2. **Base A** (pre-flight): `base-a-test` (`1-gpu-small`, one shard) and `base-a-test-cpu`.
3. **`wait-for-base-a`**, then **Base B**: the `base-b-test` matrix, one row per runner config.
4. **`wait-for-base-b`**, then **Base C**: the `base-c-test` matrix and `base-c-test-aarch64` (GB300 rows).
5. Kernel, JIT kernel, and multimodal-gen tests run as called workflows beside the stages.
6. **`pr-test-finish`** aggregates every job and fails on any failure or cancellation.

Each stage job is a matrix over `{ runner_config, run_timeout, timeout_per_file }` rows that calls `_pr-test-stage.yml`; that job runs `run_suite.py --suite {stage}-test-{runner_config}` once per shard. Extra, nightly, and weekly reuse the same reusable stage.

Every stage job runs `check-pr-test-health` after checkout: if any job in the run has already failed, it fails fast with the root cause. Scheduled runs skip the `wait-for-base-*` jobs and the health check, and run all stages in parallel.

---

## Fail-Fast Layers

4 layers of fail-fast, from fine to coarse:

| Layer | Mechanism | Granularity | Disabled on schedule? |
|-------|-----------|-------------|----------------------|
| **1. Test method → file** | `unittest -f` (failfast) | One test method fails → entire test file stops immediately | Yes |
| **2. File → suite** | `run_unittest_files()` default | One test file fails → entire suite stops (`--continue-on-error` off) | Yes |
| **3. Job → job** | `check-pr-test-health` action | One job fails → jobs that have not started fail fast (red X) | Yes |
| **4. Stage → stage** | `wait-for-base-*` + `needs` | Base A fails → base B/C jobs skip entirely (never get a runner) | Yes (wait jobs skipped) |

- **Layer 1**: `-f` flag appended to all `python3 -m pytest` / `unittest` invocations in `ci_utils.py`
- **Layer 2**: `--continue-on-error` flag in `run_suite.py` — off for PRs, on for scheduled runs
- **Layer 3**: skipped when `SKIP_PR_TEST_HEALTH_CHECK` is set (scheduled, `run_all_tests`, `test_parallel_dispatch`); filters out cascade failures to show only root cause jobs
- **Layer 4**: `wait-for-base-*` jobs run only on `pull_request` events without `test_parallel_dispatch`

The control labels `bypass-fail-fast`, `parallel-stages`, `max-concurrency`, and `highest-priority` (all three) relax these limits for one PR; `.github/scripts/ci-labels.cjs` resolves them, and the [contribution guide](https://docs.sglang.io/developer_guide/contribution_guide.html#ci-control-labels) describes each.

---

## Execution Modes

| Aspect | PR (`pull_request`) | Scheduled (cron, twice daily) | Manual dispatch (`workflow_dispatch`) |
|--------|---------------------|-------------------------------|--------------------------------------|
| **Stage ordering** | Sequential: A → B → C via `wait-for-base-*` | Parallel | Parallel (the wait jobs run only on PRs) |
| **continue-on-error** | No (stop at first failure within suite) | Yes (run all tests) | With `force_continue_on_error` or `run_all_tests` |
| **Retry** | Enabled | Enabled | Enabled |
| **Shards at once** | `max_parallel` (about a third of the shards); all with `max-concurrency` | All | All with `run_all_tests`; otherwise as on a PR |
| **PR gate** | Yes (draft, label, rate limit) | Skipped | Skipped with `test_parallel_dispatch` |
| **Extra tier** | Only with `run-ci-nv-extra` | Called from the same run | Called with `run_all_tests` or `test_parallel_dispatch` |

A single test, class, or method on a PR runs through `/rerun-test`, not a stage dispatch.

---

## Stage Gating (`wait-for-jobs` action)

`wait-for-base-a` and `wait-for-base-b` are lightweight `ubuntu-latest` jobs that poll the GitHub Actions API.

1. Calls `listJobsForWorkflowRun` to list all jobs in the current run
2. Matches jobs by name prefix (matrix shards are named like `base-b-test-1-gpu-small (3)`), so the stage jobs' `name:` is load-bearing
3. If any matched job has `conclusion === 'failure'` → fail immediately
4. If all matched jobs are completed and the count matches `expected_count` → success
5. Otherwise → sleep and retry, up to `max-wait-minutes` (240 for base-a, 350 for base-b). Keep budgets under the 6h hard cap on GitHub-hosted runners: a wait killed by the cap ends `cancelled`, not `timeout`. The base-b wait sets `fail-on-timeout: 'false'`, so a timeout lets base-c start; `pr-test-finish` still fails the run on any base-b failure

Each spec entry reads its `expected_count` from `check-changes.outputs.partitions['<suite>'].size`, so shard counts never need a hand edit. A new runner row in base-a or base-b still needs its own `{"prefix": ..., "expected_count": ...}` entry in the wait job.

---

## Cross-Job Fail-Fast (`check-pr-test-health` action)

Composite action called after checkout in the reusable stage workflows and in the kernel and multimodal-gen workflows.

1. Queries `listJobsForWorkflowRun` for the current workflow run
2. Keeps **root cause failures only** — failed jobs whose failing step is not the health check itself (a job that failed fast is excluded)
3. If any are found → `core.setFailed()` with their names; later steps skip through the default `if: success()`
4. Otherwise the step succeeds

It checks every job in the run, not only the current stage. Error message example:

```
Fail-fast: skipping — root cause job(s): base-b-test-1-gpu-small (0), base-b-test-1-gpu-small (1)
```

---

## Within-Suite Failure Handling

Controlled by `run_unittest_files()` in `python/sglang/test/ci/ci_utils.py`.

| Flag | PR default | Scheduled default | Effect |
|------|------------|-------------------|--------|
| `--continue-on-error` | Off | On | Off: stop at first failure. On: run all files, report all failures at end |
| `--enable-retry` | On | On | Retry retriable failures |
| `--max-attempts` | 2 | 2 | Max attempts per file including initial run |

Retry classification: a failure matching `NON_RETRIABLE_PATTERNS` (import, syntax, type errors, OOM, segfault, ...) is not retried; one matching `RETRIABLE_PATTERNS` (accuracy and performance assertions) or `INFRA_RETRIABLE_PATTERNS` is. An unknown `AssertionError` is retried; other unknown failures are not. The lists are in `ci_utils.py`.

`check-changes` sets `continue_on_error` (true for scheduled runs and `run_all_tests`), and `_pr-test-stage.yml` passes it to `run_suite.py` as `--continue-on-error`.

---

## Test Partitioning

`scripts/ci/utils/compute_partitions.py` runs in `check-changes` and emits, per dispatched suite, `{size, arr, max_parallel}`:

- **Per-file time**: the live estimate from the `sglang-ci-stats` model (`model.json`), falling back to the file's `est_time`.
- **Shard count**: enough shards that each fits `0.75 * run_timeout` of its stage row (0.75 is the inverse of LPT's 4/3 worst case). Lowering a row's `run_timeout` buys more shards, not shorter ones. `base-a-test-1-gpu-small` is pinned to one shard.
- **`max_parallel`**: `size // 3` (at least 1) for GPU suites; all shards for CPU suites, scheduled runs, and `max-concurrency`.

Within a suite, `ci_register.py:auto_partition()` assigns files to shards with the LPT heuristic: sort by time descending (filename as tie-breaker), then give each file to the shard with the least total time.

The `Compute partitions` step summary lists every suite's size and `max_parallel` for that run. Multimodal diffusion uses `python/sglang/multimodal_gen/test/run_suite.py`, not `test/run_suite.py`.

---

## check-changes Job

| Trigger | Method |
|---------|--------|
| `pull_request` | `dorny/paths-filter` on the PR diff |
| `workflow_dispatch` / `workflow_call` with `run_all_tests` | Force all true |
| `schedule` | Force all true |

`main_package` gates the base stages; `sgl_kernel` builds the PR-local kernel wheel, runs the kernel suites, and switches B200 jobs to the kernel-build runner labels; `jit_kernel` and `multimodal_gen` gate their workflows. The full output list is in `_pr-test-check-changes.yml`.

---

## Concurrency Control

`pr-test.yml` groups runs by `pr-test-{event_name}-{PR number or ref name}-{git_ref or 'all'}`:

- `event_name` keeps scheduled runs apart from fork PRs whose branch is named `main`
- a PR keys on its number, since `head_ref` has no owner and two forks with one branch name would cancel each other

`cancel-in-progress` is true except for `workflow_call`, so a new push cancels the old PR run.

---

## How To: Add a Runner to a Tier

Adding a test to a runner that its tier already runs needs no workflow edit: `stage=` and `runner_config=` on the registration are enough. When the tier lacks that runner, add one row; never move the test to another runner to fit a tier.

1. Add `{ runner_config: <key>, run_timeout: ..., ... }` to the tier's matrix: the stage job in `pr-test.yml` (`base-a-test`, `base-b-test`, `base-c-test`, or `base-c-test-aarch64` for aarch64 runners), the `extra-test` matrix in `pr-test-extra.yml` (with its `stage` and, for GB300, `arch: aarch64`), or the matrix in `nightly-test-nvidia.yml` / `weekly-test-nvidia.yml`.
2. Add the suite name `{stage}-test-{runner_config}` to `PER_COMMIT_SUITES` or `NIGHTLY_SUITES` in `test/run_suite.py`; `validate_all_suites` rejects a registration whose suite is not listed.
3. For a base-a or base-b row, add its prefix to the `wait-for-base-*` spec.
4. A new runner config (a new machine type) also needs its entry in `scripts/ci/runner_configs.yml`.

A suite left with no tests makes its stage row fail (the partition lookup returns null): remove the row when moving the last test off a runner.

---

## How To: Debug CI Failures

| Symptom | Likely cause | What to check |
|---------|-------------|---------------|
| Stage B/C jobs red with "Fail-fast: skipping" | An earlier job failed, `check-pr-test-health` triggered | The root cause job named in the annotation |
| `wait-for-base-b` waits until timeout | A stage job was renamed or a row lacks its spec entry | Job `name:` prefixes against the wait spec |
| `pr-test-finish` fails but all jobs green | A job was `cancelled` (counts as failure in finish) | Concurrency cancellation, or a `wait-for-base-*` job hit the 6h hosted-runner cap |
| Stage row fails before running tests | Its suite has no tests, so the partition is null | Registrations for `{stage}-test-{runner_config}` |
| Tests pass locally but fail in CI | Shard assignment, runner GPU type, or a timeout from the estimate | Which shard the file landed in, the runner label, the file's live estimate |
| Flaky test retried and passed | Retriable failure (accuracy/perf) | `[CI Retry]` markers in job logs |
| Flaky test NOT retried | Matched a non-retriable pattern | `NON_RETRIABLE_PATTERNS` in `ci_utils.py` |

---

## Slash Commands

Handled by `scripts/ci/utils/slash_command_handler.py` → `.github/workflows/slash-command-handler.yml`. The commands, their permissions, and their limits are in the [contribution guide](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests).

Base CI covers only the core default paths; tests in the extra, nightly, and weekly tiers do not run on a normal PR. To cover what a change touches, run those tests with `/rerun-test`; the [rerun-test](../rerun-test/SKILL.md) skill describes how to pick them.

### Opt-in CI tiers

An opt-in tier is a set of workflows a PR enters by carrying `run-ci-<tier>` on top of `run-ci`: `nv-extra` selects `pr-test-extra.yml` (with its Xeon CPU coverage), and `amd-extra` selects `pr-test-amd-extra.yml`. Each tier is one entry in `scripts/ci/opt_in_ci_tiers.json`; the slash handler reads it for `/tag-run-ci-label <tier>`, `/tag-and-rerun-ci <tier>` and `/run-<tier>-ci`, so adding a tier needs no handler change. A tier's workflows follow one convention, enforced by `scripts/lint/check_opt_in_ci_tiers.py` (pre-commit):

1. **Gate through `pr-gate.yml` with `require-label: run-ci-<tier>`.** The gate live-fetches labels, so a missing opt-in fails the gate job (red run) and a rerun after labeling passes. Keep this decision out of event-payload expressions, which do not refresh on rerun.
2. **No `labeled` trigger.** GitHub's rerun API reuses the original event payload, so a run started by an unrelated label event and skipped by an `if` on that payload stays skipped on rerun. Adding a label after a push takes a rerun instead (`/tag-and-rerun-ci <tier>` or `/run-<tier>-ci`).
3. **Every directly selectable workflow that accepts the label is listed in the tier**, so `/run-<tier>-ci` re-runs all of them. Reusable jobs chained from a listed workflow stay part of that run.

Tier names `base`, `cpu`, `extra`, `failed` and `full` are reserved: they collide with the base aliases, `/run-cpu-ci`, the hardware-specific Extra syntax, `/rerun-failed-ci` and `/run-full-ci`.
