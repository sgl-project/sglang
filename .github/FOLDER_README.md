# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/tag-and-rerun-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group`, `/run-cpu-ci` |

`/run-ci`, `/run-base-ci`, and `/rerun-base-ci` with no argument are aliases of
`/tag-and-rerun-ci`: labeling and retrying each use their existing permission
checks. With one registered tier argument, they are aliases of `/run-<tier>-ci`
and require both permissions. For example, `/run-ci nv-extra` equals `/run-extra-ci nv`.
Use `/run-ci nv` (or `/run-nv-ci`) for NVIDIA base and shared checks only.
Use `/run-ci amd`, `/run-base-ci amd`, or `/run-amd-ci` for AMD Base only.
The `nv` and `amd` entries have `label: null`: they add only `run-ci` and select
existing runs for the current commit, skipping workflows with no run. Existing
hardware labels and path filters are kept; future pushes follow the existing
workflow gates. AMD Base does not require a new `run-ci-amd` label.
Legacy `/tag-and-rerun-ci <tier>` still retries broadly; its arguments add labels.

`/run-extra-ci nv` and `/run-extra-ci amd` select NVIDIA/shared Extra and AMD
Extra, respectively. They add `run-ci` and the matching `run-ci-nv-extra` or
`run-ci-amd-extra` label, then rerun only that tier's workflows. Their aliases
include `/rerun-extra-ci <hardware>`, `/run-<hardware>-extra-ci`, and
`/run-ci <hardware>-extra`. NVIDIA/shared Extra retains its Xeon CPU coverage.

`/run-full-ci nv` and `/run-full-ci amd` combine the selected hardware's Base and
Extra tiers. Alias: `/rerun-full-ci <hardware>`. These commands select current
commit runs, including green runs, and require both labeling and rerun
permissions. Active runs are left alone; missing base runs are skipped and
missing extra runs are reported.

Extra and Full require exactly one hardware argument: `nv` or `amd`. Missing,
unknown, or multiple hardware arguments are rejected before adding labels or
rerunning workflows. Regular Base CI triggers and path filters are unchanged.

The shared `extra` tier and `run-ci-extra` label are retired. The old label no
longer enables either Extra workflow on updated branches. `/tag-run-ci-label extra` and `/tag-and-rerun-ci extra` are rejected with migration guidance. To
intentionally opt into both hardware tiers, use `/tag-run-ci-label nv-extra amd-extra`; rerun each tier with its dedicated command. Legacy
`/tag-and-rerun-ci <tier>` still retries broadly, so prefer the dedicated run
commands when selecting hardware.

Commands preserve existing labels and do not cancel queued or running workflows.
For existing PRs, update the branch to include the registry, handler, and gate
changes, then push to create fresh runs. Rerunning an older run keeps its original
workflow revision and label behavior.

The registry also contains `nv` and `amd` Base selections, `mlu`, selecting
`pr-test-mlu.yml` with `run-ci-mlu`, and `ppu`, selecting `pr-test-ppu.yml`
with `run-ci-ppu`. PPU keeps its existing path filters. Each opt-in workflow
requires its dedicated label through `require-label` in `pr-gate.yml`, in
addition to `run-ci`. Register each opt-in workflow under its matching tier.

`/rerun-test`, `/rerun-group` and `/run-cpu-ci` are gated on the commenter alone: either
`cooldown_interval_minutes: 0`, or `write`/`admin` permission on the repo. Where
the PR comes from makes no difference, and authoring it grants nothing.

Those are the same two signals `pr-gate.yml` already uses to waive its rate
limit, and that is the point -- a selective rerun dispatches `rerun-test.yml`
directly, which never passes through `pr-gate.yml`, so it bypasses the rate limit
by construction. Anyone allowed to run one is therefore unthrottled in practice,
which is exactly what a zero cooldown already declares.

Set a cooldown deliberately: `0` lets the holder run PR-head code on the
self-hosted GPU runners, and raising it above `0` takes that away again along
with their rate-limit waiver.

## Others
- `MAINTAINER.md` defines the code maintenance model.
