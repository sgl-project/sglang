#!/usr/bin/env bash
set -euo pipefail

# INPUT_REF, DEFAULT_SHA, REPO_URL and GITHUB_OUTPUT are supplied by the workflow.
ls_remote() {
    git ls-remote "${REPO_URL}" "$1" | cut -f1
}

# Resolve a branch or tag name to a commit SHA. Accepts bare names
# (`main`, `v1.2.0`) as well as fully qualified refs (`refs/heads/main`,
# `refs/tags/v1.2.0`), which is what `github.ref` looks like.
resolve_named_ref() {
    local ref="$1"
    local sha=""
    if [[ "${ref}" == refs/heads/* ]]; then
        sha="$(ls_remote "${ref}")"
    elif [[ "${ref}" == refs/tags/* ]]; then
        # Prefer the peeled commit of an annotated tag; fall back to the tag
        # object for a lightweight tag.
        sha="$(ls_remote "${ref}^{}")"
        if [[ -z "${sha}" ]]; then
            sha="$(ls_remote "${ref}")"
        fi
    else
        sha="$(ls_remote "refs/heads/${ref}")"
        if [[ -z "${sha}" ]]; then
            sha="$(ls_remote "refs/tags/${ref}^{}")"
        fi
        if [[ -z "${sha}" ]]; then
            sha="$(ls_remote "refs/tags/${ref}")"
        fi
    fi
    printf '%s' "${sha}"
}

if [[ -z "${INPUT_REF}" ]]; then
    resolved_sha="${DEFAULT_SHA}"
elif [[ "${INPUT_REF}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    resolved_sha="${INPUT_REF}"
else
    resolved_sha="$(resolve_named_ref "${INPUT_REF}")"
    if [[ -z "${resolved_sha}" ]]; then
        echo "::error::Could not resolve Git ref '${INPUT_REF}' in ${REPO_URL}"
        exit 1
    fi
fi

echo "Resolved ${INPUT_REF} to ${resolved_sha}"
echo "commit_sha=${resolved_sha}" >> "${GITHUB_OUTPUT}"
