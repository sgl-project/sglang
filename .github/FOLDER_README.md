# Maintenance Tools

This folder contains tools and workflows for automating maintenance tasks.

## CI Permissions

`CI_PERMISSIONS.json` defines the CI permissions granted to each user.
Maintainers can directly edit the file to add entries with `"reason": "custom override"`.
Maintainers can also run `update_ci_permission.py` to update it with some auto rules (e.g., top contributors in the last 90 days get full permissions).

Recognized permission keys:

| Key | Grants |
| --- | --- |
| `can_tag_run_ci_label` | `/tag-run-ci-label`, `/run-ci` (and base aliases), `/tag-and-rerun-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `can_rerun_failed_ci` | `/rerun-failed-ci`, `/run-ci` (and base aliases), `/tag-and-rerun-ci`, `/run-full-ci`, `/run-<tier>-ci` |
| `cooldown_interval_minutes` | rate limit in `pr-gate.yml`; `0` also grants `/rerun-test`, `/rerun-group`, `/run-cpu-ci` |

`/run-<tier>-ci`, `/run-ci <tier>`, and `/run-extra-ci <hardware>` require both label and rerun permissions. `/tag-and-rerun-ci`, the no-argument forms of `/run-ci`, `/run-base-ci`, and `/rerun-base-ci`, and `/run-full-ci` check each step independently: a user who can only retry, including the PR author, may retry existing workflows but cannot add labels.

With no arguments, `/run-ci`, `/run-base-ci`, and `/rerun-base-ci` are aliases of `/tag-and-rerun-ci`: they add `run-ci` when permitted and retry unsuccessful NVIDIA base workflows and shared checks such as lint and router tests. With one registered tier argument, each instead labels and reruns only that tier, including successful runs: `/run-ci amd` equals `/run-amd-ci`. `/run-ci extra` remains an alias of `/run-extra-ci` for NVIDIA extra, and `/run-ci amd-extra` remains an alias of `/run-amd-extra-ci`.

`/run-extra-ci` selects NVIDIA extra with no arguments. With one hardware argument, it selects that hardware's registered extra tier: `/run-extra-ci amd` equals `/run-amd-extra-ci`. `/rerun-extra-ci amd` is equivalent too. For extra coverage, use the structured `/run-extra-ci amd` spelling; existing `/run-amd-extra-ci`, `/run-ci amd-extra`, and base-alias forms remain equivalent. Base commands and `/run-extra-ci` accept at most one argument; fully specified tier commands such as `/run-amd-extra-ci` accept none. An unregistered extra tier, such as `npu-extra`, is rejected without adding labels or rerunning workflows.

`/tag-and-rerun-ci <tiers>` and `/run-full-ci <tiers>` retain their broader unsuccessful-run retry scope: NVIDIA base, shared checks, and all named tiers. `/run-full-ci` always includes NVIDIA extra. Default `/rerun-failed-ci` retries NVIDIA base and extra plus shared checks without adding labels; its tier arguments select NVIDIA base, shared checks, and the named tiers instead.

Opt-in tiers in `scripts/ci/opt_in_ci_tiers.json` are `extra` (NVIDIA extra), `amd` (AMD base), `amd-extra` (AMD extra), `npu`, `xpu`, `musa`, `mlx`, `xeon`, and `arm64`. Their labels are independent; `extra` does not enable extra coverage on other hardware. Existing labels never expand a command's selection. Equivalent tier spellings require a registered tier; MLU and PPU remain outside this registry.

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
