# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci`, `/run-ci <hardware...>`, `/run-extra-ci`, `/run-full-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/tag-and-rerun-ci`, `/run-ci <hardware...>`, `/run-extra-ci`, `/run-full-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group`, `/run-cpu-ci` |

Command syntax and the per-hardware tiers are documented in the
[contribution guide](https://docs.sglang.io/docs/developer_guide/contribution_guide#how-to-trigger-ci-tests).
`/run-ci`, `/run-extra-ci` and `/run-full-ci` with hardware arguments add
labels and rerun runs, so they need both keys. The
labeling commands and the no-argument `/run-ci` alias of `/tag-and-rerun-ci`
check each key for the step it gates.

Each opt-in workflow passes `require-label: run-ci-<tier>` to `pr-gate.yml` and
is registered under that tier in `scripts/ci/opt_in_ci_tiers.json`;
`scripts/lint/check_opt_in_ci_tiers.py` checks the two agree. Without the label,
without `run-ci`, or on a draft PR, the gate passes with `opted-in=false` and
the caller's test jobs skip. `pr-test-finish` then fails, so a PR cannot merge
without running Base CI.

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
