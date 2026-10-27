#!/bin/bash
set -euo pipefail

# Script to commit an sgl-amd-mono pin bump to SGLang and create or update its pull request
# Usage: commit_and_pr_sgl_amd_mono.sh <old_commit> <new_commit> <branch_name>
#
# Arguments:
#   old_commit: The sgl-amd-mono commit main pins now
#   new_commit: The sgl-amd-mono commit to pin
#   branch_name: The git branch name to push to, reused by every bump

OLD_COMMIT="${1:-}"
NEW_COMMIT="${2:-}"
BRANCH_NAME="${3:-}"

if [ -z "$OLD_COMMIT" ] || [ -z "$NEW_COMMIT" ] || [ -z "$BRANCH_NAME" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <old_commit> <new_commit> <branch_name>"
    exit 1
fi

MONO_REPO="sgl-project/sgl-amd-mono"
OLD_SHORT_SHA="${OLD_COMMIT:0:8}"
NEW_SHORT_SHA="${NEW_COMMIT:0:8}"

OPEN_PR=$(gh pr list --head "$BRANCH_NAME" --base main --state open --json number --jq '.[].number')
# A force-push restarts the PR's CI, so skip it when the PR already pins this commit.
if [ -n "$OPEN_PR" ] && git fetch -q origin "$BRANCH_NAME" \
    && git show FETCH_HEAD:python/pyproject_other.toml | grep -qF "sgl-amd-mono.git@${NEW_COMMIT}"; then
    echo "#${OPEN_PR} already pins sgl-amd-mono at ${NEW_SHORT_SHA}; nothing to do."
    exit 0
fi

# Commit titles cite their PRs as (#N), which would link to SGLang PRs here.
CHANGES=$(gh api "repos/${MONO_REPO}/compare/${OLD_COMMIT}...${NEW_COMMIT}" \
    --jq '.commits[] | "- [`\(.sha[0:8])`](\(.html_url)) \(.commit.message | split("\n")[0])"' \
    | sed -E "s|\(#([0-9]+)\)|(${MONO_REPO}#\1)|g")
if [ -z "$CHANGES" ]; then
    CHANGES="None: \`${NEW_SHORT_SHA}\` does not come after \`${OLD_SHORT_SHA}\`."
fi

PR_TITLE="[Chore] [AMD] monokernels: Pin sgl-amd-mono at ${NEW_SHORT_SHA}"
PR_BODY="## Summary

This PR pins \`sgl-amd-mono\` at [\`${NEW_SHORT_SHA}\`](https://github.com/${MONO_REPO}/commit/${NEW_COMMIT}), up from [\`${OLD_SHORT_SHA}\`](https://github.com/${MONO_REPO}/commit/${OLD_COMMIT}). The pin is the \`monokernels_hip\` extra in \`python/pyproject_other.toml\`, and the gfx950 ROCm image checks out the same commit.

## sgl-amd-mono Changes
${CHANGES}

Full diff: https://github.com/${MONO_REPO}/compare/${OLD_COMMIT}...${NEW_COMMIT}

🤖 Generated with GitHub Actions"

echo "Committing changes..."
git add -A
git commit -m "chore: pin sgl-amd-mono at ${NEW_SHORT_SHA}

This commit pins sgl-amd-mono (the monokernels_hip extra in
python/pyproject_other.toml) at ${NEW_COMMIT},
up from ${OLD_COMMIT}.

🤖 Generated with GitHub Actions"

echo "Pushing to ${BRANCH_NAME}..."
git push --force origin "${BRANCH_NAME}"

if [ -n "$OPEN_PR" ]; then
    echo "Updating pull request #${OPEN_PR}..."
    gh pr edit "$OPEN_PR" --title "$PR_TITLE" --body "$PR_BODY"
    PR_URL=$(gh pr view "$OPEN_PR" --json url --jq .url)
else
    echo "Creating pull request..."
    PR_URL=$(gh pr create \
      --title "$PR_TITLE" \
      --body "$PR_BODY" \
      --label run-ci \
      --base main \
      --head "${BRANCH_NAME}")
fi

echo "✓ Pull request ready: ${PR_URL}"

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## ✅ sgl-amd-mono Pin Bump Complete

**sgl-amd-mono:** \`${OLD_SHORT_SHA}\` → \`${NEW_SHORT_SHA}\`

### 📝 Pull Request
${PR_URL}
EOF
fi
