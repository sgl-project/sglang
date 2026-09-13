#!/bin/bash
# Delete AMD HuggingFace checkpoints that no longer have a registered nightly.
#
# Allowlisted directories only. Intended to run on an AMD CI runner that
# mounts /home/runner/sglang-data (the shared /sgl-data HF cache).
#
# Usage:
#   REQUIRE_HF_CACHE=1 bash scripts/ci/amd/cleanup_obsolete_hf_checkpoints.sh
#
# Env:
#   REQUIRE_HF_CACHE=1  Fail if no hub directory is found (CI).
#   HF_HOME             Extra cache root whose ./hub is also considered.

set -euo pipefail

ALLOWED_DIRS=(
  models--Qwen--Qwen3-235B-A22B-Instruct-2507
  models--Qwen--Qwen3-235B-A22B-Instruct-2507-FP8
  models--amd--Qwen3-235B-A22B-Instruct-2507-mxfp4
  models--deepseek-ai--DeepSeek-V3.1
)

is_allowlisted() {
  local name="$1"
  local allowed
  for allowed in "${ALLOWED_DIRS[@]}"; do
    if [[ "$name" == "$allowed" ]]; then
      return 0
    fi
  done
  return 1
}

collect_hubs() {
  local -a candidates=()
  if [[ -n "${HF_HOME:-}" ]]; then
    candidates+=("${HF_HOME}/hub")
  fi
  candidates+=(
    /home/runner/sglang-data/hf-cache/hub
    /sgl-data/hf-cache/hub
  )

  local -A seen=()
  local path real
  for path in "${candidates[@]}"; do
    if [[ -d "$path" ]]; then
      real=$(cd "$path" && pwd -P)
      if [[ -z "${seen[$real]:-}" ]]; then
        seen[$real]=1
        printf '%s\n' "$real"
      fi
    fi
  done
}

dir_size() {
  du -sh "$1" 2>/dev/null | awk '{print $1}'
}

echo "=== AMD obsolete HF checkpoint cleanup ==="
echo "hostname=$(hostname)"
echo "RUNNER_NAME=${RUNNER_NAME:-}"
echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

mapfile -t HUBS < <(collect_hubs)
if [[ ${#HUBS[@]} -eq 0 ]]; then
  echo "No HuggingFace hub directory found."
  echo "Looked for HF_HOME/hub, /home/runner/sglang-data/hf-cache/hub, /sgl-data/hf-cache/hub."
  if [[ "${REQUIRE_HF_CACHE:-}" == "1" ]]; then
    exit 1
  fi
  exit 0
fi

for hub in "${HUBS[@]}"; do
  echo "--- hub: ${hub} ---"
  df -h "$hub" || true
done
echo

removed=0
missing=0
failed=0
{
  echo "### Obsolete AMD HF checkpoint cleanup"
  echo
  echo "- hostname: \`$(hostname)\`"
  echo "- runner: \`${RUNNER_NAME:-unknown}\`"
  echo
  echo "| Hub | Checkpoint | Before | After |"
  echo "| --- | ---------- | ------ | ----- |"
} > github_summary.md

for hub in "${HUBS[@]}"; do
  for name in "${ALLOWED_DIRS[@]}"; do
    if ! is_allowlisted "$name"; then
      echo "Refusing to touch non-allowlisted name: ${name}" >&2
      exit 1
    fi
    if [[ "$name" == *"/"* || "$name" == "."* ]]; then
      echo "Refusing unsafe checkpoint name: ${name}" >&2
      exit 1
    fi

    target="${hub}/${name}"
    lock="${hub}/${name}.lock"
    if [[ ! -e "$target" && ! -e "$lock" ]]; then
      echo "absent  ${target}"
      echo "| \`${hub}\` | \`${name}\` | absent | absent |" >> github_summary.md
      missing=$((missing + 1))
      continue
    fi

    before="unknown"
    if [[ -e "$target" ]]; then
      before=$(dir_size "$target")
    fi
    echo "removing ${target} (${before})"
    if rm -rf -- "$target" "$lock"; then
      if [[ -e "$target" || -e "$lock" ]]; then
        echo "FAILED  ${target} still exists" >&2
        echo "| \`${hub}\` | \`${name}\` | ${before} | FAILED |" >> github_summary.md
        failed=$((failed + 1))
      else
        echo "removed ${target}"
        echo "| \`${hub}\` | \`${name}\` | ${before} | gone |" >> github_summary.md
        removed=$((removed + 1))
      fi
    else
      echo "FAILED  rm ${target}" >&2
      echo "| \`${hub}\` | \`${name}\` | ${before} | FAILED |" >> github_summary.md
      failed=$((failed + 1))
    fi
  done
  echo
  echo "--- hub after: ${hub} ---"
  df -h "$hub" || true
  echo
done

{
  echo
  echo "removed=${removed} missing=${missing} failed=${failed}"
} >> github_summary.md

echo "removed=${removed} missing=${missing} failed=${failed}"
if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
exit 0
