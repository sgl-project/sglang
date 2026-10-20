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
# Install a guard module + .pth next to the installed coverage package:
# that directory is guaranteed writable (pip put coverage there) and on
# sys.path for the same interpreter that runs the servers below. The
# .pth imports the guard in every python process; the guard no-ops
# unless COVERAGE_PROCESS_START is set, so it stays inert for later
# non-coverage jobs on the same runner.
#
# Besides calling process_startup(), the guard hardens coverage's
# SIGTERM shutdown (coverage 7.8): mp children (forkserver/pool
# workers) can receive the teardown SIGTERM after their coverage was
# already stopped, and the stock handler then dies on
# `assert self._collectors` (collector.py:348) -- a scary multi-process
# traceback lands in the test log (tests still pass) and the worker
# survives until the SIGKILL backstop. Patched handler: flush via the
# stock _atexit path (with the assert defused into a no-op), then
# always restore SIG_DFL and re-raise so the worker dies quietly on
# the first SIGTERM.
SITE_PACKAGES="$(python -c 'import coverage, os; print(os.path.dirname(os.path.dirname(coverage.__file__)))')"
cat > "${SITE_PACKAGES}/sglang_ci_cov_guard.py" <<'PYEOF'
"""Loaded by sglang_ci_subprocess_coverage.pth in every python process.

No-op unless COVERAGE_PROCESS_START is set. Starts subprocess coverage
and hardens coverage's SIGTERM shutdown for mp workers that already
stopped collecting (see run_tests_with_coverage.sh for the rationale).
"""
import os

if os.environ.get("COVERAGE_PROCESS_START"):
    import signal

    import coverage
    import coverage.collector

    try:
        # Second stop on an empty collector stack is a no-op, not an
        # assert: mp workers SIGTERMed after their coverage already
        # stopped must not die on `assert self._collectors`.
        _collector_stop = coverage.collector.Collector.stop

        def _idempotent_stop(self):
            if self._collectors:
                return _collector_stop(self)

        coverage.collector.Collector.stop = _idempotent_stop

        # Flush exactly like the stock handler, then ALWAYS die by the
        # default disposition: no traceback, no handler re-entry, no
        # lingering until the SIGKILL backstop.
        def _on_sigterm(self, signum=None, frame=None):
            try:
                self._atexit("sigterm")
            finally:
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signal.SIGTERM)

        coverage.Coverage._on_sigterm = _on_sigterm
    except Exception:
        # Unexpected coverage version: keep stock behavior.
        pass

    coverage.process_startup()
PYEOF
echo "import sglang_ci_cov_guard" > "${SITE_PACKAGES}/sglang_ci_subprocess_coverage.pth"

# Traced subprocesses can only flush on a graceful stop (SIGKILL skips
# atexit). Load the CI-owned pytest plugin that wraps kill_process_tree
# with SIGTERM-first + bounded wait (the original SIGKILL still
# backstops). Product code (python/sglang) is NOT modified: the patch
# lives only in the pytest process. Scheduler/detokenizer flush via
# coveragerc sigterm=True; the server drains via its own SIGTERM path.
export SGLANG_KILL_PROCESS_TREE_GRACEFUL=1
export SGLANG_KILL_PROCESS_TREE_GRACEFUL_TIMEOUT="${SGLANG_KILL_PROCESS_TREE_GRACEFUL_TIMEOUT:-5}"
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
