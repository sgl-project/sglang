# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/tag-and-rerun-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group` |

`/rerun-test` and `/rerun-group` are gated on the commenter alone: either
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
