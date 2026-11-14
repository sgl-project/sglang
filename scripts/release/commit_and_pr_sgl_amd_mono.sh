#!/bin/bash
set -euo pipefail

# Script to commit an sgl-amd-mono pin bump to SGLang and create a pull request
# Usage: commit_and_pr_sgl_amd_mono.sh <old_commit> <new_commit>

OLD_COMMIT="${1:-}"
NEW_COMMIT="${2:-}"

if [ -z "$OLD_COMMIT" ] || [ -z "$NEW_COMMIT" ]; then
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

git config user.name "sglang-bot"
git config user.email "sglang-bot@users.noreply.github.com"
git checkout -b "$BRANCH_NAME"
git commit -am "chore: pin sgl-amd-mono at ${NEW_SHORT_SHA}"
# --force: an earlier run may have pushed this branch without opening its PR.
git push --force origin "$BRANCH_NAME"

PR_URL=$(gh pr create \
  --title "[Chore] [AMD] monokernels: Pin sgl-amd-mono at ${NEW_SHORT_SHA}" \
  --body "Pins \`sgl-amd-mono\` (the \`monokernels_hip\` extra) at \`${NEW_SHORT_SHA}\`, up from \`${OLD_SHORT_SHA}\` ([full diff](https://github.com/${MONO_REPO}/compare/${OLD_COMMIT}...${NEW_COMMIT})).

${CHANGES}

🤖 Generated with GitHub Actions" \
  --base main \
  --head "$BRANCH_NAME")
echo "Created ${PR_URL}"

while read -r OLD_PR OLD_BRANCH; do
    [ -n "$OLD_PR" ] || continue
    # Closed with the PAT, so cancel-pr-workflows-on-close.yml runs and cancels its CI.
    gh pr close "$OLD_PR" --comment "Superseded by ${PR_URL}."
    git push origin --delete "$OLD_BRANCH"
done <<< "$OPEN_BUMP_PRS"
