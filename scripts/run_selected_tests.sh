#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CACHE_ROOT="/root/.cache/tests/precise-test"

# Per-CI-run, per-attempt unique directory (no cross-run residue, no overwrite on re-run)
RUN_ID="${GITHUB_RUN_ID:-local}"
RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"
RUN_DIR="${RUN_ID}-attempt-${RUN_ATTEMPT}"

# Date tag for grouping coverage data. In CI, prefer GITHUB_RUN_STARTED_AT
# (same value across all jobs in one run, immune to midnight rollover).
# Fall back to local date for non-CI execution.
if [ -n "${GITHUB_RUN_STARTED_AT:-}" ]; then
  COV_DATE_TAG="${GITHUB_RUN_STARTED_AT:0:10}"
  COV_DATE_TAG="${COV_DATE_TAG//-/}"
else
  COV_DATE_TAG="$(date +%Y%m%d)"
fi
COV_ROOT="${CACHE_ROOT}/${RUN_DIR}/outputs/sglang@${COV_DATE_TAG}"

mkdir -p "${COV_ROOT}"

# --- Subprocess coverage wiring -------------------------------------------
# Tests launch sglang servers via subprocess.Popen; the parent's tracer cannot
# see them. Two things make Popen'd processes (and their mp children) measured:
#   1. COVERAGE_PROCESS_START: read by the .pth hook below in every child
#      python process to auto-start coverage. No-op in processes where the
#      variable is absent, so the hook is safe to leave installed.
#   2. The .pth hook in site-packages: coverage.process_startup().
# popen_launch_server merges os.environ into the child env, so both the
# variable and COVERAGE_FILE propagate to the server.
export COVERAGE_PROCESS_START="${SCRIPT_DIR}/coveragerc"
site_packages="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo 'import coverage; coverage.process_startup()' \
  > "${site_packages}/sglang_coverage_startup.pth"

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  echo "Usage: $0 <test> [test ...]"
  exit 1
fi

overall_status=0

results=()

# Derive a filesystem-safe directory name from a test target:
#   1. strip the CI workspace prefix (/__w/sglang/sglang)
#   2. strip the trailing ".py"
#   3. flatten path separators:  /  ->  __
#   4. flatten pytest separators: ::  ->  --
#   5. replace any remaining unsafe character with "_"
# Example:
#   /__w/sglang/sglang/test/registered/npu/basic_function/HiCache/test_npu_hicache_mha.py
#     -> __test__registered__npu__basic_function__HiCache__test_npu_hicache_mha
# Each test gets its own COVERAGE_FILE so results never collide.
setup_coverage() {
  local target="$1"
  local name="${target#/__w/sglang/sglang}"
  name="${name%.py}"
  name="${name//\//__}"
  name="${name//::/--}"
  name="${name//[^a-zA-Z0-9_.-]/_}"
  local covdir="${COV_ROOT}/${name}"
  mkdir -p "${covdir}"
  export COVERAGE_FILE="${covdir}/coverage"
}

run_one() {
  local target="$1"

  echo "=== Running: ${target} ==="
  setup_coverage "${target}"

  set +e
  python -m coverage run --rcfile="${SCRIPT_DIR}/coveragerc" -m pytest -sv --color=yes "${target}" 2>&1
  local status=$?
  set -e

  if [ "${status}" -ne 0 ]; then
    echo "1" > "$(dirname "${COVERAGE_FILE}")/FAILED"
    echo "=== FAILED: ${target} ==="
    overall_status=1
    results+=("${target}|FAILED")
  else
    echo "=== PASSED: ${target} ==="
    results+=("${target}|PASSED")
  fi

  # Merge parallel data files (main + server + mp children) into one.
  set +e
  python -m coverage combine --rcfile="${SCRIPT_DIR}/coveragerc" 2>/dev/null
  set -e
}

for target in "${targets[@]}"; do
  run_one "${target}"
done

# ====================
# Test result summary
# ====================
passed_list=()
failed_list=()

for entry in "${results[@]}"; do
  test_name="${entry%%|*}"
  test_status="${entry##*|}"
  if [ "${test_status}" = "PASSED" ]; then
    passed_list+=("${test_name}")
  else
    failed_list+=("${test_name}")
  fi
done

passed_count="${#passed_list[@]}"
failed_count="${#failed_list[@]}"
total_count=$((passed_count + failed_count))

echo
echo "============================================================"
echo "Test Summary: total ${total_count}, passed ${passed_count}, failed ${failed_count}"
echo "============================================================"

if [ "${passed_count}" -gt 0 ]; then
  echo "✓ PASSED:"
  for t in "${passed_list[@]}"; do
    echo "  ${t}"
  done
fi

if [ "${failed_count}" -gt 0 ]; then
  echo
  echo "✗ FAILED:"
  for t in "${failed_list[@]}"; do
    echo "  ${t}"
  done
fi

echo "============================================================"

if [ "${failed_count}" -gt 0 ]; then
  echo "ERROR: Some tests failed."
fi
echo "Coverage: ${COV_ROOT}/"

exit "${overall_status}"
