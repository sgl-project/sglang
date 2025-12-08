#!/bin/bash
set -e

# Script to commit version bump changes and create a pull request
# Usage: commit_and_pr.sh <version_type> <new_version> <branch_name>
#
# Arguments:
#   version_type: "SGLang" or "sgl-kernel"
#   new_version: The new version number
#   branch_name: The git branch name to push to

VERSION_TYPE="$1"
NEW_VERSION="$2"
BRANCH_NAME="$3"

if [ -z "$VERSION_TYPE" ] || [ -z "$NEW_VERSION" ] || [ -z "$BRANCH_NAME" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <version_type> <new_version> <branch_name>"
    exit 1
fi

# Get changed files and format them
echo "Getting changed files..."
FILES_LIST=$(git diff --name-only | sed 's/^/- /')
COMMIT_FILES=$(git diff --name-only | sed 's/^/          - /')

# Commit changes
echo "Committing changes..."
git add -A
git commit -m "chore: bump ${VERSION_TYPE} version to ${NEW_VERSION}

This commit updates the ${VERSION_TYPE} version across all relevant files:
${COMMIT_FILES}

🤖 Generated with GitHub Actions"

# Push changes
echo "Pushing to ${BRANCH_NAME}..."
git push origin "${BRANCH_NAME}"

# Create pull request
echo "Creating pull request..."
PR_URL=$(gh pr create \
  --title "chore: bump ${VERSION_TYPE} version to ${NEW_VERSION}" \
  --body "## Summary

This PR bumps the ${VERSION_TYPE} version to \`${NEW_VERSION}\` across all relevant files.

## Files Updated
${FILES_LIST}

🤖 Generated with GitHub Actions" \
  --base main \
  --head "${BRANCH_NAME}")

echo "✓ Pull request created successfully"

# Add GitHub Actions job summary
if [ -n "$GITHUB_STEP_SUMMARY" ]; then
  cat >> "$GITHUB_STEP_SUMMARY" <<EOF
## ✅ Version Bump Complete

**Version Type:** ${VERSION_TYPE}
**New Version:** \`${NEW_VERSION}\`

### 📝 Pull Request Created
${PR_URL}

### 📦 Files Updated
${FILES_LIST}
EOF
fi
