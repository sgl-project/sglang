# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions of users without write access; write access grants every CI permission, so those users are not listed.
Maintainers add entries by hand and run `update_ci_permission.py` to clean the file: it drops write-access users and entries whose user has had no commit on `main` for 90 days. An entry is kept through inactivity only if its `reason` names why (anything other than `"top contributor"` or `"custom override"`).
`update_ci_permission.py --check` lists the entries the rules would change; the `CI Permissions Check` workflow runs it weekly and on PRs that edit the file, and only warns.

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/tag-and-rerun-ci`, `/run-ci <hardware...>`, `/run-extra-ci`, `/run-full-ci`, and approving a commit for NVIDIA PR Test Base and Extra; write access to the repo grants the same |
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

The workflows in `run_ci_approval.APPROVAL_WORKFLOWS` (`pr-test.yml`, `pr-test-extra.yml`,
`pr-test-sgl-router.yml`, `pr-test-mlu.yml`) call the gate with `approval-mode: commit`:
unless the PR's author can approve, besides the `run-ci` label they need a
`ci/run-ci-approved` status on the PR's head commit, so every push needs a new
approval. Only the slash command handler records it, when an approver runs a
command that adds `run-ci`, and only for a head pushed before the comment;
`scripts/ci/utils/run_ci_approval.py` holds the approver rule.

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
