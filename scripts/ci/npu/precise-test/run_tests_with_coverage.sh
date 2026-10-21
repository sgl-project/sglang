#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Overridable so jobs without shared-disk access (e.g. hosted ubuntu runners)
# can stage coverage data locally and ship it to setup-covstub via an artifact.
CACHE_ROOT="${PRECISE_TEST_CACHE_ROOT:-/root/.cache/tests/precise-test}"

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

# ====================
# Subprocess coverage: server / scheduler / detokenizer processes
# ====================
# Tests launch servers as separate subprocesses (sglang serve console
# script / python -m sglang.launch_server), outside the traced pytest
# process. COVERAGE_PROCESS_START + a
# site-packages .pth makes every python process started from here
# (popen'd servers and their mp children) auto-start coverage via
# coverage.process_startup(). The .pth line is a no-op whenever the env
# var is unset, so it stays inert for later non-coverage jobs on the same
# runner. Subprocesses inherit the per-test COVERAGE_FILE, and
# parallel=True suffixes each file with host.pid.random, so all processes
# of one test land in the same per-test directory and are attributed to
# that test by the existing find/combine/map logic unchanged.
export COVERAGE_PROCESS_START="${SCRIPT_DIR}/coveragerc"
# Install the .pth next to the installed coverage package: that directory
# is guaranteed writable (pip put coverage there) and on sys.path for the
# same interpreter that runs the servers below.
SITE_PACKAGES="$(python -c 'import coverage, os; print(os.path.dirname(os.path.dirname(coverage.__file__)))')"
echo "import os; os.environ.get('COVERAGE_PROCESS_START') and __import__('coverage').process_startup()" \
  > "${SITE_PACKAGES}/sglang_ci_subprocess_coverage.pth"

# Traced subprocesses can only flush on a graceful stop (SIGKILL skips
# atexit). Load the CI-owned pytest plugin that wraps kill_process_tree
# with SIGTERM-first + bounded wait (the original SIGKILL still
# backstops). Product code (python/sglang) is NOT modified: the patch
# lives only in the pytest process. Scheduler/detokenizer flush via
# coveragerc sigterm=True; the server drains via its own SIGTERM path.
export SGLANG_KILL_PROCESS_TREE_GRACEFUL=1
export SGLANG_KILL_PROCESS_TREE_GRACEFUL_TIMEOUT="${SGLANG_KILL_PROCESS_TREE_GRACEFUL_TIMEOUT:-20}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTEST_PLUGINS="precise_test_coverage_plugin"

targets=("$@")
if [ "${#targets[@]}" -eq 0 ]; then
  echo "Usage: $0 <test> [test ...]"
  exit 1
fi

overall_status=0

results=()

# Derive a filesystem-safe directory name from a test target:
#   1. strip the trailing ".py"
#   2. flatten path separators:  /  ->  __
#   3. flatten pytest separators:  ::  ->  --
#   4. replace any remaining unsafe character with "_"
#      (keep "[" and "]" so parametrized pytest node ids
#       ...::test_x[case_id] stay reversible: "]"+first / "["+in-class
#       / "-"-last are all literals inside the bracket expression)
# Each test gets its own COVERAGE_FILE so results never collide.
setup_coverage() {
  local target="$1"
  local name="${target%.py}"
  name="${name//\//__}"
  name="${name//::/--}"
  name="${name//[^]a-zA-Z0-9_.[-]/_}"
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
