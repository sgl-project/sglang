#!/bin/bash

# This script is used for release.
# It tags all remote branches starting with 'v' with the same name as the branch,
# deletes the corresponding branches from the remote, and pushes the tags to the remote repository.

git fetch origin --prune

# List all branches starting with 'v'
branches=$(git branch -r | grep 'origin/v' | sed 's/origin\///')

# Loop through each branch
for branch in $branches; do
    echo "Processing branch: $branch"

    # Get the commit hash for the branch
    commit_hash=$(git rev-parse origin/$branch)

    # Create a tag with the same name as the branch using the commit hash
    git tag $branch $commit_hash

    # Delete the branch from the remote
    git push origin --delete $branch
done

# Push all tags to the remote repository
git push --tags

echo "All branches starting with 'v' have been tagged, deleted from remote, and pushed to the remote repository."
