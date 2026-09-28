#!/bin/bash
# Manually trigger the registered MI455x (gfx1250) accuracy suites on bare metal.
#
# This is the hand-driven stand-in for the `nightly-1-gpu-mi45x-*` jobs in
# .github/workflows/nightly-test-amd.yml, for use before a CI runner exists.
# It reproduces the environment those jobs hand to the tests -- the CI env vars
# from scripts/ci/amd/amd_ci_exec.sh plus GPU_ARCHS=gfx1250 -- and invokes the
# same `run_suite.py` entrypoint, so a local result and a future nightly result
# mean the same thing. What it deliberately does not do is start a container:
# run this from inside whichever ROCm image already has SGLang importable.
#
# Every run lands in its own timestamped directory with the logs, the step
# summary the tests emit, per-file metrics, and a machine-readable environment
# snapshot; collect_results.py turns that into results.json + report.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# The cookbook suites, in the cookbook's own order. These gate against the
# accuracy the cookbook measured on MI455 A0; the older
# nightly-amd-1-gpu-mi45x* suites under test/registered/amd/accuracy/mi45x/ run
# the same models under different flags and are not selected by default. Keep in
# sync with the `register_amd_ci(suite=...)` calls under
# test/registered/accuracy/mi45x/cookbook/.
#
# Ordered small-TP first for every model, so a run that is going to fail on a
# model fails on its cheapest configuration rather than after a 4-card launch.
DEFAULT_SUITES="nightly-amd-1-gpu-mi45x-cookbook-gpt-oss,nightly-amd-4-gpu-mi45x-cookbook-gpt-oss,nightly-amd-2-gpu-mi45x-cookbook-dsr1,nightly-amd-4-gpu-mi45x-cookbook-dsr1,nightly-amd-1-gpu-mi45x-cookbook-dsv4-flash,nightly-amd-4-gpu-mi45x-cookbook-dsv4-flash,nightly-amd-1-gpu-mi45x-cookbook-qwen35,nightly-amd-4-gpu-mi45x-cookbook-qwen35"

SUITES="${DEFAULT_SUITES}"
TAG=""
RESULTS_ROOT="${REPO_ROOT}/local-ci-results/mi45x"
# 3600s matches the --timeout-per-file the nightly MI455x jobs pass, which is
# above the 900-1800s server-launch timeouts inside the tests so a slow cold
# start cannot get a file killed mid-eval.
TIMEOUT_PER_FILE=3600
GPU_ARCHS_VALUE="gfx1250"
CONTINUE_ON_ERROR=1
CLEAN_VRAM=0
DRY_RUN=0
LIST_ONLY=0
declare -a EXTRA_ENV=()

usage() {
  cat <<'EOF'
Usage: run_mi45x_local.sh [options]

Options:
  -s, --suite <list>        Comma-separated suites to run.
                            (default: all registered mi45x accuracy suites)
  -t, --tag <name>          Label appended to the run directory name.
  -o, --results-dir <dir>   Root for run directories.
                            (default: <repo>/local-ci-results/mi45x)
      --timeout-per-file <s>  Per-file time limit passed to run_suite.py. (default: 3600)
      --gpu-archs <arch>    Value for GPU_ARCHS. (default: gfx1250)
  -e KEY=VAL                Extra env var for the suite process. Repeatable.
                            Use this to shrink a run, e.g.
                              -e SGLANG_MI45X_NUM_QUESTIONS=32
                            (a short run is a smoke test, not a reproduction:
                            the accuracy gate is meaningless below 1319)
      --fail-fast           Stop at the first failing file. (default: continue)
      --clean-vram          Run ensure_vram_clear.sh first. This stops every
                            GPU-attached container on the host, so it is off by
                            default -- only use it on a dedicated box.
  -l, --list                Print the resolved plan and the tests each suite
                            would run, then exit.
  -n, --dry-run             Set up the run directory and print the commands
                            without executing the suites.
  -h, --help                This message.

Examples:
  # Does the plumbing work? Cheapest suite, 32 questions. Minutes, not hours.
  ./run_mi45x_local.sh -s nightly-amd-1-gpu-mi45x-cookbook-gpt-oss \
      -e SGLANG_MI45X_NUM_QUESTIONS=32 -t smoke

  # One model, both its TPs, gated for real.
  ./run_mi45x_local.sh -t gpt-oss \
      -s nightly-amd-1-gpu-mi45x-cookbook-gpt-oss,nightly-amd-4-gpu-mi45x-cookbook-gpt-oss

  # Full sweep of all eight cookbook suites, compared against the last run.
  ./run_mi45x_local.sh -t baseline
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--suite)           SUITES="$2"; shift 2;;
    -t|--tag)             TAG="$2"; shift 2;;
    -o|--results-dir)     RESULTS_ROOT="$2"; shift 2;;
    --timeout-per-file)   TIMEOUT_PER_FILE="$2"; shift 2;;
    --gpu-archs)          GPU_ARCHS_VALUE="$2"; shift 2;;
    -e)                   EXTRA_ENV+=("$2"); shift 2;;
    --fail-fast)          CONTINUE_ON_ERROR=0; shift;;
    --clean-vram)         CLEAN_VRAM=1; shift;;
    -l|--list)            LIST_ONLY=1; shift;;
    -n|--dry-run)         DRY_RUN=1; shift;;
    -h|--help)            usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

for kv in "${EXTRA_ENV[@]}"; do
  if [[ "$kv" != *=* ]]; then
    echo "Error: -e expects KEY=VAL, got '$kv'" >&2
    exit 2
  fi
done

IFS=',' read -r -a SUITE_LIST <<< "${SUITES}"
if [[ ${#SUITE_LIST[@]} -eq 0 ]]; then
  echo "Error: no suites selected" >&2
  exit 2
fi

# ---------------------------------------------------------------- preflight --
# A wrong-arch run is worse than no run: it burns an hour and produces numbers
# that look real. Refuse to start unless the GPUs actually are gfx1250.
detect_gfx_arch() {
  if command -v rocm_agent_enumerator >/dev/null 2>&1; then
    rocm_agent_enumerator 2>/dev/null | grep -m1 -E '^gfx[0-9]' && return 0
  fi
  if command -v rocminfo >/dev/null 2>&1; then
    rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9]+[a-z]*' && return 0
  fi
  return 1
}

DETECTED_ARCH="$(detect_gfx_arch || true)"

# --list only reads the registry, so it stays usable on a box with no GPU.
if [[ ${LIST_ONLY} -eq 0 ]]; then
  if [[ -z "${DETECTED_ARCH}" ]]; then
    echo "Error: could not detect a GPU architecture (no rocm_agent_enumerator/rocminfo output)." >&2
    echo "       Are you inside a container started with /dev/kfd and /dev/dri?" >&2
    exit 1
  fi
  if [[ "${DETECTED_ARCH}" != "${GPU_ARCHS_VALUE}" ]]; then
    echo "Error: detected ${DETECTED_ARCH} but --gpu-archs says ${GPU_ARCHS_VALUE}." >&2
    echo "       These suites are MI455x/gfx1250 only. Pass --gpu-archs ${DETECTED_ARCH}" >&2
    echo "       if you really mean to run them elsewhere." >&2
    exit 1
  fi
fi

if [[ ${LIST_ONLY} -eq 0 ]] && ! python3 -c "import sglang" >/dev/null 2>&1; then
  echo "Error: 'import sglang' failed for $(command -v python3)." >&2
  echo "       This script does not build or containerise anything; run it from" >&2
  echo "       inside a ROCm image that already has SGLang installed." >&2
  exit 1
fi

# ----------------------------------------------------------------- run setup --
RUN_ID="$(date +%Y%m%d-%H%M%S)"
[[ -n "${TAG}" ]] && RUN_ID="${RUN_ID}-${TAG}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"

if [[ ${LIST_ONLY} -eq 1 ]]; then
  # Read the registry directly. run_suite.py has no list-only mode -- it prints
  # its plan and then immediately starts executing -- so shelling out to it here
  # would launch a real server and load a real model just to answer "what would
  # run?". Importing it and stopping before run_unittest_files touches no GPU.
  ( cd "${REPO_ROOT}/test" && python3 - "${SUITES}" <<'PY'
import glob
import importlib.util
import os
import sys

# Load ci_register straight from its path instead of `import run_suite`. The
# registration markers are parsed out of the source by AST, so this needs
# nothing but the stdlib -- which keeps --list working outside the ROCm
# container, where `import sglang` fails on orjson/torch.
spec = importlib.util.spec_from_file_location(
    "ci_register", "../python/sglang/test/ci/ci_register.py"
)
ci_register = importlib.util.module_from_spec(spec)
sys.modules["ci_register"] = ci_register
spec.loader.exec_module(ci_register)

suites = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
files = [
    f
    for f in glob.glob("registered/**/*.py", recursive=True)
    if os.path.basename(f) not in ("conftest.py", "__init__.py")
]
all_tests = ci_register.collect_tests(files, sanity_check=False)

print(f"Suites ({len(suites)}):")
grand_total = 0.0
for suite in suites:
    matched = [
        t
        for t in all_tests
        if t.backend == ci_register.HWBackend.AMD
        and t.effective_suite == suite
        and t.nightly
    ]
    enabled = [t for t in matched if t.disabled is None]
    skipped = [t for t in matched if t.disabled is not None]
    total = sum(t.est_time for t in enabled)
    grand_total += total
    print(f"\n=== {suite} === ({len(enabled)} test(s), est {total / 60:.0f} min)")
    for t in enabled:
        print(f"  - {t.filename} (est_time={t.est_time})")
    for t in skipped:
        print(f"  - [disabled: {t.disabled}] {t.filename}")
    if not matched:
        print("  (no tests registered to this suite)")

print(f"\nTotal estimated runtime: {grand_total / 60:.0f} min")
PY
  )
  exit 0
fi

mkdir -p "${RUN_DIR}/logs"

STEP_SUMMARY="${RUN_DIR}/step_summary.md"
METRICS_FILE="${RUN_DIR}/metrics.jsonl"
: > "${STEP_SUMMARY}"
: > "${METRICS_FILE}"

# The tests only call write_github_step_summary() under is_in_ci(), which reads
# SGLANG_IS_IN_CI. Setting both that and GITHUB_STEP_SUMMARY is what makes the
# accuracy tables land in a file locally instead of being print-only. These are
# the same variables amd_ci_exec.sh exports, so the tests take their CI path.
export SGLANG_IS_IN_CI=1
export SGLANG_IS_IN_CI_AMD=1
export GITHUB_STEP_SUMMARY="${STEP_SUMMARY}"
export SGLANG_TEST_METRICS_FILE="${METRICS_FILE}"
# Disabled on AMD in CI: the async-assert probes fire torch._assert_async in the
# MXFP4 EAGLE-MTP decode path and abort the queue with an HSA hardware exception.
export SGLANG_ENABLE_ASYNC_ASSERT=0
export SGLANG_USE_AITER=1
export GPU_ARCHS="${GPU_ARCHS_VALUE}"
# gfx1250 has no CK kernels yet; AITER must take the non-CK path.
export ENABLE_CK=0
# Coredumps on these images are multi-GB and will fill the disk.
export HSA_COREDUMP_PATTERN=/dev/null

for kv in "${EXTRA_ENV[@]}"; do
  export "${kv?}"
done

# ------------------------------------------------------- environment capture --
# Written before the first suite so a run that dies halfway still says what it
# was running. Accuracy numbers are only comparable against a known image and
# commit, and "which ROCm was that?" is unanswerable after the fact.
collect_env_json() {
  python3 - "$@" <<'PY'
import json
import os
import subprocess
import sys


def sh(cmd):
    try:
        out = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        return None
    out = out.stdout.strip()
    return out or None


def version(module):
    try:
        mod = __import__(module)
    except Exception as exc:
        return f"<unavailable: {type(exc).__name__}>"
    return getattr(mod, "__version__", "<no __version__>")


info = {
    "run_id": sys.argv[1],
    "hostname": sh("hostname"),
    "kernel": sh("uname -r"),
    "in_container": os.path.exists("/.dockerenv"),
    "git": {
        "commit": sh("git rev-parse HEAD"),
        "branch": sh("git rev-parse --abbrev-ref HEAD"),
        "describe": sh("git describe --tags --always --dirty"),
        # Accuracy deltas are routinely caused by uncommitted local edits; a
        # run with a dirty tree is not reproducible and should say so.
        "dirty_files": (sh("git status --porcelain") or "").splitlines(),
    },
    "rocm": {
        "version": sh("cat /opt/rocm/.info/version"),
        "path": sh("readlink -f /opt/rocm"),
        "agents": sorted(set((sh("rocm_agent_enumerator") or "").split())),
    },
    "gpus": (sh("rocm-smi --showproductname --csv") or "").splitlines(),
    "python": sys.version.split()[0],
    "packages": {
        name: version(name) for name in ("torch", "sglang", "aiter", "triton")
    },
    "env": {
        key: os.environ[key]
        for key in sorted(os.environ)
        if key.startswith(
            ("SGLANG_", "AITER_", "HSA_", "HIP_", "GPU_ARCHS", "ENABLE_CK", "ROCM_")
        )
    },
}
json.dump(info, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY
}

( cd "${REPO_ROOT}" && collect_env_json "${RUN_ID}" ) > "${RUN_DIR}/env.json"

python3 - "${RUN_DIR}/run.json" "${RUN_ID}" "${SUITES}" "${TIMEOUT_PER_FILE}" \
        "${CONTINUE_ON_ERROR}" "${EXTRA_ENV[@]}" <<'PY'
import json
import sys
import time

out, run_id, suites, timeout, continue_on_error, *extra_env = sys.argv[1:]
with open(out, "w") as f:
    json.dump(
        {
            "run_id": run_id,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "suites": suites.split(","),
            "timeout_per_file": int(timeout),
            "continue_on_error": continue_on_error == "1",
            "extra_env": extra_env,
        },
        f,
        indent=2,
    )
PY

echo "=============================================================="
echo " MI455x local accuracy run"
echo "   run dir : ${RUN_DIR}"
echo "   arch    : ${DETECTED_ARCH}"
echo "   suites  : ${SUITES}"
echo "   timeout : ${TIMEOUT_PER_FILE}s per file"
[[ ${#EXTRA_ENV[@]} -gt 0 ]] && echo "   env     : ${EXTRA_ENV[*]}"
echo "=============================================================="

if [[ ${CLEAN_VRAM} -eq 1 ]]; then
  echo "--- Clearing VRAM ---"
  bash "${REPO_ROOT}/scripts/ci/amd/ensure_vram_clear.sh" rocm
fi

# ---------------------------------------------------------------- suite loop --
declare -a SUITE_STATUS=()
OVERALL_RC=0

RUN_FLAGS=(--hw amd --nightly --timeout-per-file "${TIMEOUT_PER_FILE}")
[[ ${CONTINUE_ON_ERROR} -eq 1 ]] && RUN_FLAGS+=(--continue-on-error)

for suite in "${SUITE_LIST[@]}"; do
  log="${RUN_DIR}/logs/${suite}.log"
  echo
  echo "--- Suite: ${suite} -> ${log}"

  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "    (dry run) cd ${REPO_ROOT}/test && python3 run_suite.py ${RUN_FLAGS[*]} --suite ${suite}"
    SUITE_STATUS+=("${suite}:skipped-dry-run")
    continue
  fi

  # Each suite marks its own slice of the shared step summary, otherwise the
  # collector cannot tell which suite produced which accuracy table.
  echo "<!-- suite: ${suite} -->" >> "${STEP_SUMMARY}"

  # PIPESTATUS, not $?: the pipe into tee would otherwise report tee's status
  # and every suite would look green. A failing suite must also not abort the
  # script, so that the remaining suites still run and the collector still gets
  # to write a report.
  set +e
  ( cd "${REPO_ROOT}/test" && python3 run_suite.py "${RUN_FLAGS[@]}" --suite "${suite}" ) \
    2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e

  SUITE_STATUS+=("${suite}:${rc}")
  if [[ ${rc} -ne 0 ]]; then
    OVERALL_RC=1
    echo "    suite ${suite} exited ${rc}"
    if [[ ${CONTINUE_ON_ERROR} -eq 0 ]]; then
      echo "    --fail-fast set; stopping."
      break
    fi
  fi
done

printf '%s\n' "${SUITE_STATUS[@]}" > "${RUN_DIR}/suite_status.txt"

# ------------------------------------------------------------------ collect --
echo
python3 "${SCRIPT_DIR}/collect_results.py" "${RUN_DIR}" --compare latest || true

echo
echo "Results: ${RUN_DIR}"
echo "  report.md    human-readable summary"
echo "  results.json machine-readable results"
exit ${OVERALL_RC}
