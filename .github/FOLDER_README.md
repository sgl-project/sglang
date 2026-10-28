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
and require both permissions. For example, `/run-ci extra` equals `/run-extra-ci`.
Use `/run-ci nv` (or `/run-nv-ci`) for NVIDIA base and shared checks only.
Use `/run-ci amd`, `/run-base-ci amd`, or `/run-amd-ci` for AMD Base only.
The `nv` and `amd` entries have `label: null`: they add only `run-ci` and select
existing runs for the current commit, skipping workflows with no run. Existing
hardware labels and path filters are kept; future pushes follow the existing
workflow gates. AMD Base does not require a new `run-ci-amd` label.
Legacy `/tag-and-rerun-ci <tier>` still retries broadly; its arguments add labels.

`/run-extra-ci <hardware>` (also `/rerun-extra-ci <hardware>`) selects the
`<hardware>-extra` tier only when it is registered. Otherwise, the existing
ignored-suffix behavior is preserved. The registry contains `extra`, covering
both NVIDIA and AMD extra workflows, `amd-extra`, selecting only the AMD extra
workflow with `run-ci-amd-extra`, `nv`, selecting NVIDIA base/shared checks,
`amd`, selecting `pr-test-amd.yml`, and `mlu`, selecting `pr-test-mlu.yml` with
`run-ci-mlu`. PPU remains outside the registry; its workflow uses `run-ci` and
`run-ci-ppu` labels plus PPU path filters.

`/run-full-ci amd` (alias `/rerun-full-ci amd`) combines `amd` and `amd-extra`:
it adds only `run-ci` and `run-ci-amd-extra`, then selects AMD Base and AMD Extra
runs for the current commit, including green runs. Active runs are left alone;
missing base runs are skipped and missing extra runs are reported. Both labeling
and rerun permissions are required.
This selection requires exactly one argument with both `<hardware>` and
`<hardware>-extra` registered. Bare `/run-full-ci` and all other full-CI forms
keep shared Extra and broad retries; `/run-full-ci nv` still uses that legacy path.

`/run-extra-ci amd` equals `/run-amd-extra-ci`: it adds `run-ci` and
`run-ci-amd-extra` and reruns only AMD Extra. Base CI behavior is unchanged.
`/run-extra-ci` keeps the shared `run-ci-extra` label and selects both hardware
extra workflows. Commands preserve existing labels: if a PR already has
`run-ci-extra`, remove that label to stop opting into NVIDIA Extra on future
pushes or broad reruns. The AMD command does not cancel existing NVIDIA runs.

Hardware extra workflows can accept their own tier label through `require-label`
and the shared label through `alternative-label` in `pr-gate.yml`. Either label
satisfies the live opt-in check; `run-ci` remains required. Register the workflow
under both accepted tiers so each command selects it. No handler change is needed.

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
