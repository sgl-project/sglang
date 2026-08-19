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
# in a PR-numbered tmux window, and leaves the session detached. Attach with:
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

    if git -C "${REPO_ROOT}" config --get "remote.${remote_name}.url" >/dev/null 2>&1; then
        existing_url="$(git -C "${REPO_ROOT}" config --get "remote.${remote_name}.url")"
        [[ "$(normalize_github_repo "${existing_url}")" == "${remote_repo}" ]] || \
            die "remote ${remote_name} points to ${existing_url}, expected ${remote_repo}"
    else
        git -C "${REPO_ROOT}" remote add \
            "${remote_name}" "https://github.com/${remote_repo}.git"
    fi
}

fetch_pr_head() {
    local pr="$1"
    local pr_data
    local state
    local is_draft
    local fetched_sha
    local attempt

    for ((attempt = 1; attempt <= 3; attempt++)); do
        pr_data="$(
            gh pr view "${pr}" \
                --repo "${SOURCE_REPO}" \
                --json state,isDraft,headRefOid,headRefName,headRepositoryOwner,headRepository \
                --jq '[.state, (.isDraft | tostring), .headRefOid, .headRefName, .headRepositoryOwner.login, .headRepository.name] | @tsv'
        )"

        IFS=$'\t' read -r state is_draft PR_HEAD_SHA PR_HEAD_REF \
            PR_HEAD_OWNER PR_HEAD_REPO_NAME <<<"${pr_data}"
        [[ "${state}" == "OPEN" ]] || die "PR ${pr} is not open"
        [[ "${is_draft}" == "false" ]] || die "PR ${pr} is still a draft"
        [[ -n "${PR_HEAD_OWNER}" && -n "${PR_HEAD_REPO_NAME}" ]] || \
            die "PR ${pr} has no accessible head repository"

        PR_HEAD_REPO="${PR_HEAD_OWNER}/${PR_HEAD_REPO_NAME}"
        if [[ "${PR_HEAD_REPO}" == "${SOURCE_REPO}" ]]; then
            PR_REMOTE="origin"
        else
            PR_REMOTE="pr-${pr}"
            ensure_remote "${PR_REMOTE}" "${PR_HEAD_REPO}"
        fi

        PR_REMOTE_REF="refs/remotes/${PR_REMOTE}/${PR_HEAD_REF}"
        if [[ "${PR_HEAD_REPO}" == "${SOURCE_REPO}" ]]; then
            git -C "${REPO_ROOT}" fetch --no-tags "${PR_REMOTE}" \
                "+refs/heads/${PR_HEAD_REF}:${PR_REMOTE_REF}"
        else
            git -C "${REPO_ROOT}" fetch --no-tags origin \
                "+refs/pull/${pr}/head:${PR_REMOTE_REF}"
        fi
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
    local existing_worktree
    local current_branch
    local current_sha
    local upstream

    fetch_pr_head "${pr}"
    existing_worktree="$(worktree_for_branch "${PR_HEAD_REF}")"

    if [[ -n "${existing_worktree}" && "${existing_worktree}" != "${worktree}" ]]; then
        die "branch ${PR_HEAD_REF} is already checked out at ${existing_worktree}"
    fi

    if [[ -n "${existing_worktree}" ]]; then
        [[ -d "${worktree}" ]] || die "registered worktree is missing: ${worktree}"
    elif [[ -e "${worktree}" ]]; then
        die "path exists but is not a registered worktree: ${worktree}"
    elif git -C "${REPO_ROOT}" show-ref --verify --quiet "refs/heads/${PR_HEAD_REF}"; then
        git -C "${REPO_ROOT}" worktree add "${worktree}" "${PR_HEAD_REF}"
    else
        git -C "${REPO_ROOT}" worktree add --track -b "${PR_HEAD_REF}" \
            "${worktree}" "${PR_REMOTE}/${PR_HEAD_REF}"
    fi

    current_branch="$(git -C "${worktree}" branch --show-current)"
    [[ "${current_branch}" == "${PR_HEAD_REF}" ]] || \
        die "${worktree} is on ${current_branch}, expected ${PR_HEAD_REF}"
    [[ -z "$(git -C "${worktree}" status --porcelain)" ]] || \
        die "worktree has uncommitted changes: ${worktree}"

    upstream="$(git -C "${worktree}" for-each-ref \
        --format='%(upstream:short)' "refs/heads/${PR_HEAD_REF}")"
    if [[ -z "${upstream}" ]]; then
        git -C "${worktree}" branch \
            --set-upstream-to="${PR_REMOTE}/${PR_HEAD_REF}" "${PR_HEAD_REF}"
    elif [[ "${upstream}" != "${PR_REMOTE}/${PR_HEAD_REF}" ]]; then
        die "branch ${PR_HEAD_REF} tracks ${upstream}, expected ${PR_REMOTE}/${PR_HEAD_REF}"
    fi

    git -C "${worktree}" merge --ff-only "${PR_REMOTE}/${PR_HEAD_REF}"
    current_sha="$(git -C "${worktree}" rev-parse HEAD)"
    [[ "${current_sha}" == "${PR_HEAD_SHA}" ]] || \
        die "${worktree} is not exactly at PR ${pr} head ${PR_HEAD_SHA}"

    WORKTREE_BY_PR["${pr}"]="${worktree}"
    echo "prepared PR ${pr}: ${PR_HEAD_REF} -> ${worktree}"
}

prompt_for_pr() {
    local pr="$1"
    local worktree="$2"

    printf '%s\n' \
        "Before taking any action, read ${SKILL_PATH} completely and follow it as the babysit-pr-to-pass-ci skill for this task." \
        "" \
        "\$babysit-pr-to-pass-ci https://github.com/${SOURCE_REPO}/pull/${pr} --only lint.yml pr-test.yml" \
        "" \
        "Use the dedicated worktree ${worktree}. Follow the skill exactly and pursue its durable goal until both selected workflows pass on the latest PR head. Do not merge the PR. Do not touch any other worktree. If the skill requires user review, stop and clearly explain what is needed."
}

launch_window() {
    local pr="$1"
    local worktree="${WORKTREE_BY_PR[${pr}]}"
    local prompt
    local launch_command
    local window_id

    prompt="$(prompt_for_pr "${pr}" "${worktree}")"
    # $1 is intentionally expanded by the inner interactive Bash process.
    # shellcheck disable=SC2016
    printf -v launch_command 'bash -ic %q bash %q' \
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
        die "the yolo2 Bash alias is not available"
    gh auth status -h github.com >/dev/null 2>&1 || die "GitHub CLI authentication failed"
    [[ -f "${SKILL_PATH}" ]] || die "babysitting skill not found: ${SKILL_PATH}"

    origin_url="$(git -C "${REPO_ROOT}" config --get remote.origin.url || true)"
    [[ "$(normalize_github_repo "${origin_url}")" == "${SOURCE_REPO}" ]] || \
        die "origin points to ${origin_url:-nothing}, not ${SOURCE_REPO}"

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
