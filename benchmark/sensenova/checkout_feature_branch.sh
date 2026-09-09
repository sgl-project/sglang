#!/usr/bin/env bash
# Clone or fast-forward a clean checkout for SenseNova attention experiments.
set -euo pipefail

REPO_URL="${SGLANG_REPO_URL:-https://github.com/syd520zy/sglang.git}"
BRANCH="${SGLANG_BRANCH:-feat/sensenova-neo-unify-attention}"
CHECKOUT_DIR="${SGLANG_CHECKOUT_DIR:-$(pwd)/sglang-neo-unify}"
REMOTE_NAME="neo-unify"

if [[ -e "$CHECKOUT_DIR" && ! -d "$CHECKOUT_DIR/.git" ]]; then
    echo "Refusing to use non-git path: $CHECKOUT_DIR" >&2
    exit 1
fi

if [[ ! -e "$CHECKOUT_DIR" ]]; then
    git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CHECKOUT_DIR"
else
    cd "$CHECKOUT_DIR"
    if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
        git remote set-url "$REMOTE_NAME" "$REPO_URL"
    else
        git remote add "$REMOTE_NAME" "$REPO_URL"
    fi
    git fetch "$REMOTE_NAME" "$BRANCH"
    if git show-ref --verify --quiet "refs/heads/$BRANCH"; then
        git switch "$BRANCH"
    else
        git switch --track -c "$BRANCH" "$REMOTE_NAME/$BRANCH"
    fi
    git merge --ff-only "$REMOTE_NAME/$BRANCH"
fi

cd "$CHECKOUT_DIR"
git status --short
printf 'Checkout: %s\nRevision: %s\n' "$PWD" "$(git rev-parse --short HEAD)"
