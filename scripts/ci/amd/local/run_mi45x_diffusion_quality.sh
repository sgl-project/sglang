#!/bin/bash
# Run the MI455x (gfx1250) diffusion output-quality gate on bare metal.
#
# This is the hand-driven stand-in for a `nightly-amd-1-gpu-mi45x-diffusion-quality`
# job, in the same spirit as run_mi45x_local.sh: it hands the suite the CI env
# vars so a local result and a future nightly result mean the same thing, and it
# does not start a container -- run it from inside a ROCm image that already has
# SGLang importable.
#
# What makes this one different from the accuracy suites is the ground truth.
# The frames the gate scores against are gfx1250's own, and there is no gfx1250
# directory in sgl-project/ci-data-diffusion yet, so the first run on a machine
# has to record one:
#
#   ./run_mi45x_diffusion_quality.sh --gen-gt      # record reference frames
#   ./run_mi45x_diffusion_quality.sh               # score a run against them
#
# The GT directory is reused across runs (default <repo>/local-ci-results/mi45x-diffusion-gt),
# so step 2 is the one to repeat after a kernel or scheduler change.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

TEST_FILE="registered/accuracy/mi45x/diffusion/test_wan22_t2v_a14b_quality_mi45x.py"
GT_DIR="${REPO_ROOT}/local-ci-results/mi45x-diffusion-gt"
RESULTS_ROOT="${REPO_ROOT}/local-ci-results/mi45x-diffusion-quality"
TAG=""
GEN_GT=0
GPU_ARCHS_VALUE="gfx1250"
# The generation is the CI's default T2V shape; what dominates is loading the
# 118 GB A14B checkpoint off a cold page cache onto a single card.
TIMEOUT=5400
declare -a EXTRA_ENV=()

usage() {
  cat <<'EOF'
Usage: run_mi45x_diffusion_quality.sh [options]

Options:
      --gen-gt              Record this run's frames as the ground truth
                            instead of scoring against an existing one.
      --gt-dir <dir>        Ground-truth directory to write or read.
                            (default: <repo>/local-ci-results/mi45x-diffusion-gt)
  -t, --tag <name>          Label appended to the run directory name.
  -o, --results-dir <dir>   Root for run directories.
                            (default: <repo>/local-ci-results/mi45x-diffusion-quality)
      --test-file <path>    Test to run, relative to <repo>/test.
                            (default: registered/accuracy/mi45x/diffusion/
                            test_wan22_t2v_a14b_quality_mi45x.py)
      --timeout <s>         Wall-clock limit for the pytest process. (default: 5400)
      --gpu-archs <arch>    Value for GPU_ARCHS. (default: gfx1250)
  -e KEY=VAL                Extra env var for the test process. Repeatable.
                            Use this to shrink a run, e.g.
                              -e SGLANG_TEST_NUM_INFERENCE_STEPS=2
  -h, --help                This message.

Examples:
  # First time on a box: record the reference frames.
  ./run_mi45x_diffusion_quality.sh --gen-gt -t baseline

  # After a kernel change: score against them.
  ./run_mi45x_diffusion_quality.sh -t after-aiter-bump
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gen-gt)           GEN_GT=1; shift;;
    --gt-dir)           GT_DIR="$2"; shift 2;;
    -t|--tag)           TAG="$2"; shift 2;;
    -o|--results-dir)   RESULTS_ROOT="$2"; shift 2;;
    --test-file)        TEST_FILE="$2"; shift 2;;
    --timeout)          TIMEOUT="$2"; shift 2;;
    --gpu-archs)        GPU_ARCHS_VALUE="$2"; shift 2;;
    -e)                 EXTRA_ENV+=("$2"); shift 2;;
    -h|--help)          usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

for kv in "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}"; do
  if [[ "$kv" != *=* ]]; then
    echo "Error: -e expects KEY=VAL, got '$kv'" >&2
    exit 2
  fi
done

# ---------------------------------------------------------------- preflight --
# A wrong-arch run is worse than no run: the thresholds and the ground truth are
# both gfx1250's, so scoring anything else against them is meaningless.
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
if [[ -z "${DETECTED_ARCH}" ]]; then
  echo "Error: could not detect a GPU architecture (no rocm_agent_enumerator/rocminfo output)." >&2
  echo "       Are you inside a container started with /dev/kfd and /dev/dri?" >&2
  exit 1
fi
if [[ "${DETECTED_ARCH}" != "${GPU_ARCHS_VALUE}" ]]; then
  echo "Error: detected ${DETECTED_ARCH} but --gpu-archs says ${GPU_ARCHS_VALUE}." >&2
  echo "       This gate is MI455x/gfx1250 only." >&2
  exit 1
fi

if ! python3 -c "import sglang" >/dev/null 2>&1; then
  echo "Error: 'import sglang' failed for $(command -v python3)." >&2
  echo "       This script does not build or containerise anything; run it from" >&2
  echo "       inside a ROCm image that already has SGLang installed." >&2
  exit 1
fi

if [[ ${GEN_GT} -eq 0 && ! -d "${GT_DIR}" ]]; then
  echo "Error: no ground truth at ${GT_DIR}." >&2
  echo "       Record one first:  $0 --gen-gt" >&2
  exit 1
fi

# ----------------------------------------------------------------- run setup --
RUN_ID="$(date +%Y%m%d-%H%M%S)"
[[ ${GEN_GT} -eq 1 ]] && RUN_ID="${RUN_ID}-genGT"
[[ -n "${TAG}" ]] && RUN_ID="${RUN_ID}-${TAG}"
RUN_DIR="${RESULTS_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"

export SGLANG_IS_IN_CI=1
export SGLANG_IS_IN_CI_AMD=1
export GITHUB_STEP_SUMMARY="${RUN_DIR}/step_summary.md"
# On a consistency failure the harness writes the produced frames, the GT frames
# and a side-by-side HTML index here. Without it a failure is just four numbers.
export SGLANG_DIFFUSION_ARTIFACT_DIR="${RUN_DIR}/artifacts"
export GPU_ARCHS="${GPU_ARCHS_VALUE}"
# amdsmi-based autodetection already resolves ROCm on this box, but gfx1250
# reports compute capability (12, 5), so anything that falls back to the
# capability check reads it as sm120. Pin it.
export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=rocm
export ENABLE_CK=0
export HSA_COREDUMP_PATTERN=/dev/null

if [[ ${GEN_GT} -eq 1 ]]; then
  mkdir -p "${GT_DIR}"
  export SGLANG_GEN_GT=1
  export SGLANG_GT_OUTPUT_DIR="${GT_DIR}"
  MODE_DESC="recording ground truth into ${GT_DIR}"
else
  export SGLANG_CONSISTENCY_GT_DIR="${GT_DIR}"
  MODE_DESC="scoring against ground truth in ${GT_DIR}"
fi

for kv in "${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"}"; do
  export "${kv?}"
done

LOG_FILE="${RUN_DIR}/pytest.log"

{
  echo "run_id:      ${RUN_ID}"
  echo "mode:        ${MODE_DESC}"
  echo "test:        ${TEST_FILE}"
  echo "arch:        ${DETECTED_ARCH}"
  echo "commit:      $(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty:       $(git -C "${REPO_ROOT}" status --porcelain 2>/dev/null | wc -l) file(s)"
  echo "rocm:        $(cat /opt/rocm/.info/version 2>/dev/null || echo unknown)"
  echo "torch:       $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
} | tee "${RUN_DIR}/run_info.txt"

echo
echo "=== ${MODE_DESC} ==="
echo "logging to ${LOG_FILE}"
echo

set +e
(
  cd "${REPO_ROOT}/test"
  timeout "${TIMEOUT}" python3 -m pytest "${TEST_FILE}" -v -s \
    --junitxml="${RUN_DIR}/junit.xml"
) 2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

echo
if [[ ${STATUS} -eq 0 ]]; then
  echo "PASS (${RUN_DIR})"
else
  echo "FAIL exit=${STATUS} (${RUN_DIR})"
  [[ -d "${SGLANG_DIFFUSION_ARTIFACT_DIR}" ]] &&
    echo "      artifacts: ${SGLANG_DIFFUSION_ARTIFACT_DIR}"
fi

# The consistency metrics are only ever printed, so lift them out of the log
# into something greppable across runs.
grep -E 'Consistency|clip_similarity|ssim|psnr|mean_abs_diff' "${LOG_FILE}" \
  > "${RUN_DIR}/consistency_metrics.txt" 2>/dev/null || true

exit "${STATUS}"
