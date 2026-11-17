# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci`, `/run-ci <hardware...>`, `/run-extra-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/tag-and-rerun-ci`, `/run-ci <hardware...>`, `/run-extra-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group`, `/run-cpu-ci` |

`/run-ci`, `/run-base-ci`, and `/rerun-base-ci` with no argument are aliases of
`/tag-and-rerun-ci`: labeling and retrying each use their existing permission
checks. With hardware arguments, they, `/run-extra-ci`, `/run-full-ci`, and
`/run-<tier>-ci` add `run-ci` plus each selected tier's label, then rerun that
selection's workflows for the current commit, green runs included; they require
both permissions. Active runs are left alone.

Every run-ci command takes hardware arguments: `nv`, `amd`, `xeon`, `npu`,
`xpu`, `musa`, `arm64`, `mlx`, `mlu`, `ppu`, or a registered tier name such as
`nv-extra`, and accepts several of them. The command picks the layer: `/run-ci`
selects Base, `/run-extra-ci` Extra, and `/run-full-ci` both. NVIDIA and AMD
have a Base tier (`nv`, `amd`, `label: null`, selecting existing runs and
skipping missing ones) and an Extra tier. Every other hardware has only an
Extra tier, `<hardware>-extra` with label `run-ci-<hardware>-extra`, which all
three commands select. `/tag-run-ci-label` and `/tag-and-rerun-ci` resolve their
arguments as `/run-ci` does; `/tag-and-rerun-ci` also reruns the opt-in
workflows whose label it just added. Unknown arguments are rejected
before any label is added, and the retired `extra` argument gets migration
guidance.

Each opt-in workflow passes `require-label: run-ci-<tier>` to `pr-gate.yml`,
in addition to `run-ci`. Without the label the gate passes with
`opted-in=false` and the caller's test jobs skip, so the run is green rather
than failed; `main` pushes and schedules are not gated. Register each opt-in
workflow under its matching tier in `scripts/ci/opt_in_ci_tiers.json`;
`scripts/lint/check_opt_in_ci_tiers.py` checks the two agree.

Commands preserve existing labels and do not cancel queued or running workflows.
Rerunning an older run keeps its original workflow revision and gate.

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
