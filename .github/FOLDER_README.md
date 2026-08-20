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
| `can_rerun_stage` | `/rerun-test`, `/rerun-group` -- on same-repo PRs only |
| `can_rerun_test` | `/rerun-test`, `/rerun-group` -- including PRs from forks |

`can_rerun_test` is the only key that clears a fork PR. Grant it deliberately:
these commands check out the PR head and execute it on the self-hosted GPU
runners, so it lets the holder run fork code on CI hardware. Everyone else needs
write permission on the repo to use these commands on a fork PR.

PR authors can always use `/rerun-failed-ci`, `/rerun-test`, and `/rerun-group`
on their own same-repo PRs without an entry here, but authorship alone never
clears a fork PR.

## Others
- `MAINTAINER.md` defines the code maintenance model.
