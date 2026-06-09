#!/usr/bin/env bash
set -euo pipefail

# Submit an MLU CI request to the Cambricon MLU CI bridge. The real MLU test
# execution is handled by the bridge/master/Jenkins side, similar to mlu-ops.
repo_name="${SGLANG_MLU_CI_REPO_NAME:-sglang}"
trigger_type="${SGLANG_MLU_CI_TRIGGER_TYPE:-ci}"
repeat_times="${SGLANG_MLU_CI_REPEAT_TIMES:-3}"

# Pull request refs look like refs/pull/<id>/merge. For push/manual runs,
# prefer the event payload and fall back to an empty PR id.
pr_id=""
repo_url="${SGLANG_MLU_CI_REPO_URL:-${GITHUB_SERVER_URL:-https://github.com}/${GITHUB_REPOSITORY:-sgl-project/sglang}.git}"
git_ref="${SGLANG_MLU_CI_GIT_REF:-${GITHUB_REF_NAME:-${GITHUB_REF:-}}}"
commit_sha="${SGLANG_MLU_CI_COMMIT_SHA:-${GITHUB_SHA:-}}"
if [[ "${GITHUB_REF:-}" =~ refs/pull/([0-9]+)/ ]]; then
  pr_id="${BASH_REMATCH[1]}"
fi

if [[ -n "${GITHUB_EVENT_PATH:-}" && -f "${GITHUB_EVENT_PATH}" ]]; then
  event_info="$(python3 - <<'PY'
import json
import os

path = os.environ.get("GITHUB_EVENT_PATH")
try:
    with open(path, "r", encoding="utf-8") as f:
        event = json.load(f)
    pr = event.get("pull_request") or {}
    head = pr.get("head") or {}
    repo = head.get("repo") or {}
    clone_url = repo.get("clone_url") or ""
    if not clone_url and repo.get("html_url"):
        clone_url = repo["html_url"].rstrip("/") + ".git"
    print(pr.get("number", ""))
    print(clone_url)
    print(head.get("ref", ""))
    print(head.get("sha", ""))
except Exception:
    print("")
    print("")
    print("")
    print("")
PY
)"
  mapfile -t event_lines <<< "${event_info}"
  if [[ -z "${pr_id}" ]]; then
    pr_id="${event_lines[0]:-}"
  fi
  if [[ -n "${event_lines[1]:-}" ]]; then
    repo_url="${event_lines[1]}"
  fi
  if [[ -n "${event_lines[2]:-}" ]]; then
    git_ref="${event_lines[2]}"
  fi
  if [[ -n "${event_lines[3]:-}" ]]; then
    commit_sha="${event_lines[3]}"
  fi
fi

# Millisecond timestamp keeps task ids unique across reruns.
timestamp="$(python3 - <<'PY'
import time
print(int(time.time() * 1000))
PY
)"

trigger_id="${GITHUB_ACTOR:-unknown}"

python3 scripts/ci/mlu/run_mlu_ci.py \
  --repo "${repo_name}" \
  --trigger-id "${trigger_id}" \
  --pr-id "${pr_id}" \
  --repo-url "${repo_url}" \
  --git-ref "${git_ref}" \
  --commit-sha "${commit_sha}" \
  --timestamp "${timestamp}" \
  --trigger-type "${trigger_type}" \
  --repeat-times "${repeat_times}"
