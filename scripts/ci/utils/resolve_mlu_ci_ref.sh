#!/usr/bin/env bash
set -euo pipefail

# INPUT_REF, DEFAULT_SHA, REPO_URL and GITHUB_OUTPUT are supplied by the workflow.
if [[ -z "${INPUT_REF}" ]]; then
    resolved_sha="${DEFAULT_SHA}"
elif [[ "${INPUT_REF}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    resolved_sha="${INPUT_REF}"
else
    resolved_sha="$(git ls-remote "${REPO_URL}" "refs/heads/${INPUT_REF}" | cut -f1)"
    if [[ -z "${resolved_sha}" ]]; then
        resolved_sha="$(git ls-remote "${REPO_URL}" "refs/tags/${INPUT_REF}^{}" | cut -f1)"
    fi
    if [[ -z "${resolved_sha}" ]]; then
        resolved_sha="$(git ls-remote "${REPO_URL}" "refs/tags/${INPUT_REF}" | cut -f1)"
    fi
    if [[ -z "${resolved_sha}" ]]; then
        echo "::error::Could not resolve Git ref '${INPUT_REF}' in ${REPO_URL}"
        exit 1
    fi
fi

echo "Resolved ${INPUT_REF} to ${resolved_sha}"
echo "commit_sha=${resolved_sha}" >> "${GITHUB_OUTPUT}"
