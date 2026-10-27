#!/bin/bash
set -euo pipefail

# Script to commit an sgl-amd-mono pin bump to SGLang and create a pull request
# Usage: commit_and_pr_sgl_amd_mono.sh <old_commit> <new_commit>
#
# Arguments:
#   old_commit: The sgl-amd-mono commit main pins now
#   new_commit: The sgl-amd-mono commit to pin
#
# Environment:
#   GH_TOKEN: opens and closes the PRs
#   CANCEL_RUNS_TOKEN: cancels a superseded PR's runs; needs actions: write

OLD_COMMIT="${1:-}"
NEW_COMMIT="${2:-}"

if [ -z "$OLD_COMMIT" ] || [ -z "$NEW_COMMIT" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <old_commit> <new_commit>"
    exit 1
fi

MONO_REPO="sgl-project/sgl-amd-mono"
OLD_SHORT_SHA="${OLD_COMMIT:0:8}"
NEW_SHORT_SHA="${NEW_COMMIT:0:8}"
# One branch per target commit, so a closed bump PR is not reopened for it.
BRANCH_PREFIX="bot/bump-sgl-amd-mono-"
BRANCH_NAME="${BRANCH_PREFIX}${NEW_SHORT_SHA}"

if [ -n "$(gh pr list --head "$BRANCH_NAME" --state all --json number --jq '.[].number')" ]; then
    echo "A PR from ${BRANCH_NAME} already exists; nothing to do."
    exit 0
fi

OPEN_BUMP_PRS=$(gh pr list --author "@me" --state open --limit 100 --json number,headRefName \
    --jq ".[] | select(.headRefName | startswith(\"${BRANCH_PREFIX}\")) | \"\(.number) \(.headRefName)\"")

# Commit titles cite their PRs as (#N), which would link to SGLang PRs here.
CHANGES=$(gh api "repos/${MONO_REPO}/compare/${OLD_COMMIT}...${NEW_COMMIT}" \
    --jq '.commits[] | "- [`\(.sha[0:8])`](\(.html_url)) \(.commit.message | split("\n")[0])"' \
    | sed -E "s|\(#([0-9]+)\)|(${MONO_REPO}#\1)|g")
if [ -z "$CHANGES" ]; then
    CHANGES="None: \`${NEW_SHORT_SHA}\` does not come after \`${OLD_SHORT_SHA}\`."
fi

SUPERSEDES=""
if [ -n "$OPEN_BUMP_PRS" ]; then
    SUPERSEDES="Supersedes $(cut -d' ' -f1 <<< "$OPEN_BUMP_PRS" | sed 's/^/#/' | paste -sd' ' -)."
fi

echo "Committing changes..."
git config user.name "sglang-bot"
git config user.email "sglang-bot@users.noreply.github.com"
git checkout -b "$BRANCH_NAME"
git add -A
git commit -m "chore: pin sgl-amd-mono at ${NEW_SHORT_SHA}

This commit pins sgl-amd-mono (the monokernels_hip extra in
python/pyproject_other.toml) at ${NEW_COMMIT},
up from ${OLD_COMMIT}.

🤖 Generated with GitHub Actions"

# --force: an earlier run may have pushed this branch without opening its PR.
echo "Pushing to ${BRANCH_NAME}..."
git push --force origin "${BRANCH_NAME}"

echo "Creating pull request..."
PR_URL=$(gh pr create \
  --title "[Chore] [AMD] monokernels: Pin sgl-amd-mono at ${NEW_SHORT_SHA}" \
  --body "## Summary

This PR pins \`sgl-amd-mono\` at [\`${NEW_SHORT_SHA}\`](https://github.com/${MONO_REPO}/commit/${NEW_COMMIT}), up from [\`${OLD_SHORT_SHA}\`](https://github.com/${MONO_REPO}/commit/${OLD_COMMIT}). The pin is the \`monokernels_hip\` extra in \`python/pyproject_other.toml\`, and the gfx950 ROCm image checks out the same commit.

## sgl-amd-mono Changes
${CHANGES}

Full diff: https://github.com/${MONO_REPO}/compare/${OLD_COMMIT}...${NEW_COMMIT}

${SUPERSEDES}

🤖 Generated with GitHub Actions" \
  --label run-ci \
  --base main \
  --head "${BRANCH_NAME}")

echo "✓ Pull request created successfully: ${PR_URL}"

if [ -n "$OPEN_BUMP_PRS" ]; then
    while read -r OLD_PR OLD_BRANCH; do
        echo "Closing #${OLD_PR}, superseded by ${PR_URL}..."
        gh pr close "$OLD_PR" --comment "Superseded by ${PR_URL}."
        for RUN_ID in $(gh run list --branch "$OLD_BRANCH" --limit 100 --json databaseId,status \
            --jq '.[] | select(.status != "completed") | .databaseId'); do
            # A run can finish between listing and cancelling.
            GH_TOKEN="$CANCEL_RUNS_TOKEN" gh run cancel "$RUN_ID" || echo "Could not cancel run ${RUN_ID}"
        done
        git push origin --delete "$OLD_BRANCH"
    done <<< "$OPEN_BUMP_PRS"
fi

if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## ✅ sgl-amd-mono Pin Bump Complete

**sgl-amd-mono:** \`${OLD_SHORT_SHA}\` → \`${NEW_SHORT_SHA}\`

### 📝 Pull Request Created
${PR_URL}

${SUPERSEDES}
EOF
fi
