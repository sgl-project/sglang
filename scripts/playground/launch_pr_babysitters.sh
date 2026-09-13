#!/usr/bin/env bash
# Launch one interactive Codex PR babysitter per tmux window.
#
# Usage:
#   scripts/playground/launch_pr_babysitters.sh PR_NUMBER...
#
# Environment variables:
#   SGLANG_PR_TMUX_SESSION   tmux session name. Default: sglang-prs.
#   SGLANG_PR_WORKTREE_ROOT  directory for PR worktrees. Default: a sibling
#                            directory named sglang-pr-babysitters.
#
# Examples:
#   scripts/playground/launch_pr_babysitters.sh 35002 35001 35000
#   SGLANG_PR_TMUX_SESSION=release-prs \
#       scripts/playground/launch_pr_babysitters.sh 35002 35001
#
# Each babysitter monitors exactly lint.yml and pr-test.yml. The script creates
# or reuses a clean tracking worktree for every PR, starts the yolo2 Bash alias
# in a PR-numbered tmux window, and leaves the session detached. Here yolo2 is:
#
#   with-proxy codex --dangerously-bypass-approvals-and-sandbox \
#       --dangerously-enable-internet-mode
#
# Attach with:
#
#   tmux attach -t sglang-prs

set -euo pipefail

readonly SOURCE_REPO="sgl-project/sglang"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd -P)"
SKILL_PATH="${REPO_ROOT}/.claude/skills/babysit-pr-to-pass-ci/SKILL.md"
SESSION_NAME="${SGLANG_PR_TMUX_SESSION:-sglang-prs}"
WORKTREE_ROOT="${SGLANG_PR_WORKTREE_ROOT:-$(dirname "${REPO_ROOT}")/sglang-pr-babysitters}"
PRS=("$@")

declare -A WORKTREE_BY_PR=()
declare -A WINDOW_BY_PR=()
declare -A LOCAL_BRANCH_BY_PR=()
declare -A HEAD_REF_BY_PR=()
declare -A HEAD_REPO_BY_PR=()
declare -A HEAD_SHA_BY_PR=()
declare -A REMOTE_BY_PR=()

usage() {
    sed -n '2,/^$/s/^# \{0,1\}//p' "${BASH_SOURCE[0]}"
}

die() {
    echo "error: $*" >&2
    exit 1
}

validate_args() {
    local -A seen_prs=()
    local pr

    if [[ "${#PRS[@]}" -eq 1 && ("${PRS[0]}" == "-h" || "${PRS[0]}" == "--help") ]]; then
        usage
        exit 0
    fi
    [[ "${#PRS[@]}" -gt 0 ]] || {
        usage >&2
        exit 2
    }
    [[ "${SESSION_NAME}" =~ ^[A-Za-z0-9_.-]+$ ]] || \
        die "invalid tmux session name: ${SESSION_NAME}"

    for pr in "${PRS[@]}"; do
        [[ "${pr}" =~ ^[0-9]+$ ]] || die "invalid PR number: ${pr}"
        [[ -z "${seen_prs[${pr}]:-}" ]] || die "duplicate PR number: ${pr}"
        seen_prs["${pr}"]=1
    done
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

normalize_github_repo() {
    local value="${1%.git}"

    value="${value#https://github.com/}"
    value="${value#ssh://git@github.com/}"
    value="${value#git@github.com:}"
    printf '%s\n' "${value}"
}

worktree_for_branch() {
    local branch_ref="refs/heads/$1"

    git -C "${REPO_ROOT}" worktree list --porcelain | awk -v branch_ref="${branch_ref}" '
        /^worktree / {
            worktree_path = substr($0, 10)
        }
        !found && /^branch / && substr($0, 8) == branch_ref {
            print worktree_path
            found = 1
        }
    '
}

ensure_remote() {
    local remote_name="$1"
    local remote_repo="$2"
    local existing_url
    local existing_push_url

    if git -C "${REPO_ROOT}" config --get "remote.${remote_name}.url" >/dev/null 2>&1; then
        existing_url="$(git -C "${REPO_ROOT}" remote get-url "${remote_name}")"
        [[ "$(normalize_github_repo "${existing_url}")" == "${remote_repo}" ]] || \
            die "remote ${remote_name} points to ${existing_url}, expected ${remote_repo}"
    else
        git -C "${REPO_ROOT}" remote add \
            "${remote_name}" "https://github.com/${remote_repo}.git"
    fi

    existing_push_url="$(git -C "${REPO_ROOT}" remote get-url --push "${remote_name}")"
    [[ "$(normalize_github_repo "${existing_push_url}")" == "${remote_repo}" ]] || \
        die "remote ${remote_name} pushes to ${existing_push_url}, expected ${remote_repo}"
}

fetch_pr_head() {
    local pr="$1"
    local pr_data
    local state
    local is_draft
    local maintainer_can_modify
    local fetched_sha
    local attempt

    for ((attempt = 1; attempt <= 3; attempt++)); do
        pr_data="$(
            gh pr view "${pr}" \
                --repo "${SOURCE_REPO}" \
                --json state,isDraft,headRefOid,headRefName,headRepositoryOwner,headRepository,maintainerCanModify \
                --jq '[.state, (.isDraft | tostring), .headRefOid, .headRefName, .headRepositoryOwner.login, .headRepository.name, (.maintainerCanModify | tostring)] | @tsv'
        )"

        IFS=$'\t' read -r state is_draft PR_HEAD_SHA PR_HEAD_REF \
            PR_HEAD_OWNER PR_HEAD_REPO_NAME maintainer_can_modify <<<"${pr_data}"
        [[ "${state}" == "OPEN" ]] || die "PR ${pr} is not open"
        [[ "${is_draft}" == "false" ]] || die "PR ${pr} is still a draft"
        [[ -n "${PR_HEAD_OWNER}" && -n "${PR_HEAD_REPO_NAME}" ]] || \
            die "PR ${pr} has no accessible head repository"

        PR_HEAD_REPO="${PR_HEAD_OWNER}/${PR_HEAD_REPO_NAME}"
        if [[ "${PR_HEAD_REPO}" == "${SOURCE_REPO}" ]]; then
            PR_REMOTE="origin"
        else
            [[ "${maintainer_can_modify}" == "true" ]] || \
                die "PR ${pr} is from ${PR_HEAD_REPO}, but maintainer edits are disabled"
            PR_REMOTE="pr-${pr}"
            ensure_remote "${PR_REMOTE}" "${PR_HEAD_REPO}"
        fi

        PR_REMOTE_REF="refs/remotes/${PR_REMOTE}/${PR_HEAD_REF}"
        git -C "${REPO_ROOT}" fetch --no-tags "${PR_REMOTE}" \
            "+refs/heads/${PR_HEAD_REF}:${PR_REMOTE_REF}"
        fetched_sha="$(git -C "${REPO_ROOT}" rev-parse "${PR_REMOTE_REF}")"

        if [[ "${fetched_sha}" == "${PR_HEAD_SHA}" ]]; then
            return
        fi

        echo "PR ${pr} moved while it was being fetched; retrying (${attempt}/3)" >&2
    done

    die "could not fetch the current head of PR ${pr}"
}

prepare_worktree() {
    local pr="$1"
    local worktree="${WORKTREE_ROOT}/pr-${pr}"
    local local_branch="babysit/pr-${pr}"
    local existing_worktree
    local current_branch
    local current_sha
    local upstream

    fetch_pr_head "${pr}"
    existing_worktree="$(worktree_for_branch "${local_branch}")"

    if [[ -n "${existing_worktree}" && "${existing_worktree}" != "${worktree}" ]]; then
        die "branch ${local_branch} is already checked out at ${existing_worktree}"
    fi

    if [[ -n "${existing_worktree}" ]]; then
        [[ -d "${worktree}" ]] || die "registered worktree is missing: ${worktree}"
    elif [[ -e "${worktree}" ]]; then
        die "path exists but is not a registered worktree: ${worktree}"
    elif git -C "${REPO_ROOT}" show-ref --verify --quiet "refs/heads/${local_branch}"; then
        git -C "${REPO_ROOT}" worktree add "${worktree}" "${local_branch}"
    else
        git -C "${REPO_ROOT}" worktree add -b "${local_branch}" \
            "${worktree}" "${PR_REMOTE}/${PR_HEAD_REF}"
    fi

    current_branch="$(git -C "${worktree}" branch --show-current)"
    [[ "${current_branch}" == "${local_branch}" ]] || \
        die "${worktree} is on ${current_branch}, expected ${local_branch}"
    [[ -z "$(git -C "${worktree}" status --porcelain)" ]] || \
        die "worktree has uncommitted changes: ${worktree}"

    upstream="$(git -C "${worktree}" for-each-ref \
        --format='%(upstream:short)' "refs/heads/${local_branch}")"
    if [[ -z "${upstream}" ]]; then
        git -C "${worktree}" branch \
            --set-upstream-to="${PR_REMOTE}/${PR_HEAD_REF}" "${local_branch}"
    elif [[ "${upstream}" != "${PR_REMOTE}/${PR_HEAD_REF}" ]]; then
        die "branch ${local_branch} tracks ${upstream}, expected ${PR_REMOTE}/${PR_HEAD_REF}"
    fi

    git -C "${worktree}" merge --ff-only "${PR_REMOTE}/${PR_HEAD_REF}"
    current_sha="$(git -C "${worktree}" rev-parse HEAD)"
    [[ "${current_sha}" == "${PR_HEAD_SHA}" ]] || \
        die "${worktree} is not exactly at PR ${pr} head ${PR_HEAD_SHA}"

    WORKTREE_BY_PR["${pr}"]="${worktree}"
    LOCAL_BRANCH_BY_PR["${pr}"]="${local_branch}"
    HEAD_REF_BY_PR["${pr}"]="${PR_HEAD_REF}"
    HEAD_REPO_BY_PR["${pr}"]="${PR_HEAD_REPO}"
    HEAD_SHA_BY_PR["${pr}"]="${PR_HEAD_SHA}"
    REMOTE_BY_PR["${pr}"]="${PR_REMOTE}"
    echo "prepared PR ${pr}: ${PR_HEAD_REPO}:${PR_HEAD_REF} -> ${worktree}"
}

prompt_for_pr() {
    local pr="$1"
    local worktree="$2"
    local head_ref="${HEAD_REF_BY_PR[${pr}]}"
    local head_repo="${HEAD_REPO_BY_PR[${pr}]}"
    local head_sha="${HEAD_SHA_BY_PR[${pr}]}"
    local local_branch="${LOCAL_BRANCH_BY_PR[${pr}]}"
    local remote="${REMOTE_BY_PR[${pr}]}"

    printf '%s\n' \
        "Before taking any action, read ${SKILL_PATH} completely and follow it as the babysit-pr-to-pass-ci skill for this task." \
        "" \
        "\$babysit-pr-to-pass-ci https://github.com/${SOURCE_REPO}/pull/${pr} --only lint.yml pr-test.yml" \
        "" \
        "Use the dedicated worktree ${worktree} on local branch ${local_branch}. The current PR head is ${head_repo}:${head_ref} at ${head_sha}." \
        "" \
        "For any fix authorized by the skill, commit only in this worktree and push non-forcibly to the actual PR head with: git push ${remote} HEAD:refs/heads/${head_ref}" \
        "" \
        "The launcher also gives this process push.default=upstream, so a plain git push targets that same PR head branch. After every push, re-read the PR head SHA before monitoring new runs. Do not merge the PR. Do not touch any other worktree. If the skill requires user review, stop and clearly explain what is needed."
}

launch_window() {
    local pr="$1"
    local worktree="${WORKTREE_BY_PR[${pr}]}"
    local prompt
    local launch_command
    local window_id

    prompt="$(prompt_for_pr "${pr}" "${worktree}")"
    # yolo2 expands to:
    #   with-proxy codex --dangerously-bypass-approvals-and-sandbox
    #       --dangerously-enable-internet-mode
    # The GIT_CONFIG_* variables make a plain `git push` update the differently
    # named upstream PR branch without changing repository or user config.
    # $1 is intentionally expanded by the inner interactive Bash process.
    # shellcheck disable=SC2016
    printf -v launch_command 'env GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=push.default GIT_CONFIG_VALUE_0=upstream bash -ic %q bash %q' \
        'yolo2 "$1"; exec bash -i' "${prompt}"

    if [[ -z "${TMUX_CREATED:-}" ]]; then
        window_id="$(
            tmux new-session -d -P -F '#{window_id}' \
                -s "${SESSION_NAME}" -n "pr${pr}" -c "${worktree}" \
                "${launch_command}"
        )"
        TMUX_CREATED=1
    else
        window_id="$(
            tmux new-window -d -P -F '#{window_id}' \
                -t "${SESSION_NAME}:" -n "pr${pr}" -c "${worktree}" \
                "${launch_command}"
        )"
    fi

    tmux set-window-option -t "${window_id}" automatic-rename off >/dev/null
    tmux set-window-option -t "${window_id}" allow-rename off >/dev/null
    tmux rename-window -t "${window_id}" "pr${pr}"
    WINDOW_BY_PR["${pr}"]="${window_id}"
    echo "launched PR ${pr} in tmux window pr${pr}"
}

verify_window() {
    local pr="$1"
    local window_id="${WINDOW_BY_PR[${pr}]}"
    local pane_command=""
    local attempt

    for ((attempt = 1; attempt <= 30; attempt++)); do
        pane_command="$(tmux display-message -p -t "${window_id}" '#{pane_current_command}')"
        if [[ "${pane_command}" == codex* ]]; then
            return
        fi
        sleep 1
    done

    echo "warning: PR ${pr} window is running ${pane_command}, not codex" >&2
    tmux capture-pane -p -t "${window_id}" -S -30 >&2 || true
    return 1
}

main() {
    local command
    local origin_url
    local pr
    local first_pr
    local failed=0

    validate_args
    first_pr="${PRS[0]}"

    for command in bash git gh tmux; do
        require_command "${command}"
    done

    bash -ic 'type yolo2 >/dev/null 2>&1' >/dev/null 2>&1 || \
        die "the yolo2 Bash alias is not available (expected: with-proxy codex --dangerously-bypass-approvals-and-sandbox --dangerously-enable-internet-mode)"
    gh auth status -h github.com >/dev/null 2>&1 || die "GitHub CLI authentication failed"
    [[ -f "${SKILL_PATH}" ]] || die "babysitting skill not found: ${SKILL_PATH}"

    origin_url="$(git -C "${REPO_ROOT}" config --get remote.origin.url || true)"
    [[ "$(normalize_github_repo "${origin_url}")" == "${SOURCE_REPO}" ]] || \
        die "origin points to ${origin_url:-nothing}, not ${SOURCE_REPO}"
    ensure_remote origin "${SOURCE_REPO}"

    mkdir -p "${WORKTREE_ROOT}"
    WORKTREE_ROOT="$(cd "${WORKTREE_ROOT}" && pwd -P)"
    if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
        die "tmux session ${SESSION_NAME} already exists"
    fi

    for pr in "${PRS[@]}"; do
        prepare_worktree "${pr}"
    done

    for pr in "${PRS[@]}"; do
        launch_window "${pr}"
        sleep 1
    done

    for pr in "${PRS[@]}"; do
        if ! verify_window "${pr}"; then
            failed=1
        fi
    done

    tmux select-window -t "${WINDOW_BY_PR[${first_pr}]}"
    echo
    echo "tmux session ${SESSION_NAME} is ready with ${#PRS[@]} Codex windows."
    echo "Attach with: tmux attach -t ${SESSION_NAME}"
    echo "Use Ctrl-b w to select a PR window and Ctrl-b d to detach."

    if [[ "${failed}" -ne 0 ]]; then
        die "one or more Codex windows did not start correctly"
    fi
}

main
