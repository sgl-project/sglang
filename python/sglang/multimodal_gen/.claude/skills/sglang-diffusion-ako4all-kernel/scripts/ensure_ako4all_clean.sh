#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${1:-$PWD}"
BASE_DIR="$(cd "$BASE_DIR" && pwd)"
AKO_DIR="${BASE_DIR}/AKO4ALL"
CANONICAL_UPSTREAM_URL="https://github.com/TongmingLAIC/AKO4ALL.git"
UPSTREAM_URL="${AKO4ALL_UPSTREAM_URL:-$CANONICAL_UPSTREAM_URL}"
CLONE_URL="${AKO4ALL_URL:-$UPSTREAM_URL}"

say() {
  printf '[ako4all] %s\n' "$*"
}

fail() {
  printf '[ako4all] ERROR: %s\n' "$*" >&2
  exit 1
}

if [[ ! -d "$AKO_DIR/.git" ]]; then
  say "AKO4ALL not found under ${BASE_DIR}; cloning ${CLONE_URL}"
  git clone "$CLONE_URL" "$AKO_DIR"
fi

cd "$AKO_DIR"

if ! git remote get-url origin >/dev/null 2>&1; then
  fail "AKO4ALL exists but has no origin remote."
fi

if ! git remote get-url upstream >/dev/null 2>&1; then
  say "Adding missing upstream remote -> ${UPSTREAM_URL}"
  git remote add upstream "$UPSTREAM_URL"
fi

git fetch upstream --prune
git remote set-head upstream -a >/dev/null 2>&1 || true

default_branch="${AKO4ALL_BRANCH:-}"
if [[ -z "$default_branch" ]]; then
  if upstream_head="$(git symbolic-ref --quiet --short refs/remotes/upstream/HEAD 2>/dev/null)"; then
    default_branch="${upstream_head#upstream/}"
  else
    default_branch="main"
  fi
fi

if [[ -n "$(git status --porcelain)" ]]; then
  fail "AKO4ALL worktree is dirty. Clean all local changes before using this skill."
fi

if git show-ref --verify --quiet "refs/heads/${default_branch}"; then
  git switch "$default_branch" >/dev/null
else
  git switch -c "$default_branch" --track "upstream/${default_branch}" >/dev/null
fi

git fetch upstream --prune

local_head="$(git rev-parse HEAD)"
upstream_head="$(git rev-parse "upstream/${default_branch}")"

if [[ "$local_head" != "$upstream_head" ]]; then
  if git merge-base --is-ancestor "$local_head" "$upstream_head"; then
    say "Fast-forwarding ${default_branch} to upstream/${default_branch}"
    git merge --ff-only "upstream/${default_branch}" >/dev/null
  else
    fail "Local ${default_branch} diverges from upstream/${default_branch}. Reset or re-clone AKO4ALL before continuing."
  fi
fi

if [[ -n "$(git status --porcelain)" ]]; then
  fail "AKO4ALL became dirty after sync; stop and inspect the repo."
fi

final_head="$(git rev-parse HEAD)"
expected_head="$(git rev-parse "upstream/${default_branch}")"
if [[ "$final_head" != "$expected_head" ]]; then
  fail "AKO4ALL is not exactly at upstream/${default_branch}."
fi

say "Ready: ${AKO_DIR}"
say "Branch: ${default_branch}"
say "Commit: ${final_head}"
