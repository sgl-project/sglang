#!/bin/bash
set -uo pipefail

# Job-started hook for the Intel XPU self-hosted runners. A non-zero exit fails the
# job before any of its steps run.
#
# A fork PR can add its own workflow file and target any self-hosted label; GitHub runs
# it as <file>@refs/pull/<n>/merge. No gate inside our workflows can stop that, so the
# runner refuses any job whose workflow file is not on a branch or tag of the repo.
# pull_request_target and pr-test-xpu.yml's commit gate still run approved PR code.
#
# Install a root-owned copy outside the runner's work directory (the job controls the
# checkout), then add to the runner's .env and restart it:
#   sudo install -o root -g root -m 755 runner_job_started_guard.sh /usr/local/sbin/
#   ACTIONS_RUNNER_HOOK_JOB_STARTED=/usr/local/sbin/runner_job_started_guard.sh
#
# Settings (runner .env):
#   XPU_RUNNER_GUARD_MODE       log (default): log refusals only; enforce: fail the job.
#   XPU_RUNNER_GUARD_WORKFLOWS  optional space-separated workflow file names allowed on
#                               this runner, e.g. "nightly-test-intel.yml nightly-test-intel-wheel.yml".
# A missing GITHUB_* variable counts as a refusal, so the hook fails closed.

repo="sgl-project/sglang"
mode="${XPU_RUNNER_GUARD_MODE:-log}"
allowed="${XPU_RUNNER_GUARD_WORKFLOWS:-}"
event="${GITHUB_EVENT_NAME:-}"
workflow_ref="${GITHUB_WORKFLOW_REF:-}"
# <owner>/<repo>/.github/workflows/<file>@<ref>; split at the first '@' so a ref can never
# smuggle in a second '@refs/heads/...'.
workflow_path="${workflow_ref%%@*}"
workflow_git_ref="${workflow_ref#*@}"
workflow_file="${workflow_path##*/}"

reason=""
if [[ "${GITHUB_REPOSITORY:-}" != "${repo}" ]]; then
  reason="repository '${GITHUB_REPOSITORY:-unset}' is not ${repo}"
elif [[ "${event}" == "pull_request" ]]; then
  reason="pull_request runs the PR's own workflow files"
elif [[ "${workflow_ref}" != *@* || "${workflow_path}" != "${repo}/.github/workflows/${workflow_file}" ]]; then
  reason="workflow ref '${workflow_ref:-unset}' is not a workflow file of ${repo}"
else
  case "${workflow_git_ref}" in
    refs/heads/* | refs/tags/*) ;;
    *) reason="workflow ref '${workflow_ref}' is not a branch or tag of ${repo}" ;;
  esac
fi
if [[ -z "${reason}" && -n "${allowed}" && " ${allowed} " != *" ${workflow_file} "* ]]; then
  reason="workflow '${workflow_file}' is not in XPU_RUNNER_GUARD_WORKFLOWS"
fi

summary="event=${event:-unset} workflow_ref=${workflow_ref:-unset} actor=${GITHUB_ACTOR:-unset} run=${GITHUB_RUN_ID:-unset}"
if [[ -z "${reason}" ]]; then
  echo "XPU runner guard: allowed (${summary})"
  exit 0
fi

logger -t sglang-xpu-runner-guard -p auth.warning "refused (${mode}): ${reason}; ${summary}" 2>/dev/null || true
if [[ "${mode}" == "enforce" ]]; then
  echo "::error title=XPU runner guard::Refusing this job: ${reason} (${summary})"
  exit 1
fi
echo "::warning title=XPU runner guard::Would refuse this job: ${reason} (${summary})"
exit 0
