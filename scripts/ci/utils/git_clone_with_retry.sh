#!/bin/bash
# Shared `git clone` helper for CI install scripts.
#
# - Retries transient network failures (3 attempts, shallow clone).
# - When GH_TOKEN or GITHUB_TOKEN is set, authenticates github.com requests.
#   Anonymous git-over-HTTPS is rate limited per source IP, and shared
#   self-hosted runner egress trips it as "fatal: could not read Username
#   for 'https://github.com'". Authenticated requests are limited per token
#   instead. The token is passed via GIT_CONFIG_* env so it never lands in
#   `set -x` traces or in the cloned repo's .git/config.
#
# Usage: git_clone_with_retry <repo_url> <dest_dir> ["--branch <ref>"]

_git_with_github_auth() {
  # Disable xtrace before touching the token so neither it nor its base64
  # form is echoed.
  local xtrace=0
  [[ $- == *x* ]] && xtrace=1
  { set +x; } 2>/dev/null

  local rc=0
  local token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  if [ -z "$token" ]; then
    "$@" || rc=$?
  else
    GIT_CONFIG_COUNT=1 \
    GIT_CONFIG_KEY_0="http.https://github.com/.extraheader" \
    GIT_CONFIG_VALUE_0="AUTHORIZATION: basic $(printf 'x-access-token:%s' "$token" | base64 | tr -d '\n')" \
      "$@" || rc=$?
  fi

  [ "$xtrace" = 1 ] && set -x
  return $rc
}

# Helper function to git clone with retries
git_clone_with_retry() {
  local repo_url="$1"
  local dest_dir="${2:-}"
  local branch_args="${3:-}"
  local max_attempts=3

  for attempt in $(seq 1 $max_attempts); do
    echo "Git clone attempt $attempt/$max_attempts: $repo_url"

    # prevent from partial clone
    if [ -n "$dest_dir" ] && [ -d "$dest_dir" ]; then
      rm -rf "$dest_dir"
    fi

    if _git_with_github_auth git \
      -c http.lowSpeedLimit=1000 \
      -c http.lowSpeedTime=30 \
      clone --depth 1 ${branch_args:+$branch_args} "$repo_url" "$dest_dir"; then
      echo "Git clone succeeded."
      return 0
    fi

    if [ $attempt -lt $max_attempts ]; then
      echo "Git clone failed, retrying in 5 seconds..."
      sleep 5
    fi
  done

  echo "Git clone failed after $max_attempts attempts: $repo_url"
  return 1
}
