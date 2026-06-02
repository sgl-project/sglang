#!/usr/bin/env bash
# Run the sgl-project/DeepGEMM test suite against an *installed* sgl-deep-gemm
# wheel, as a pre-release gate for release-whl-deepgemm.yml.
#
# Why "installed": the released artifact is the `sgl-deep-gemm` distribution,
# which imports as `deep_gemm` and ships a *pre-compiled* tvm-ffi `_C.so`
# (from csrc/tvm_ffi_api.cpp) plus the fork's Python wrapper layer
# (fp8_fp4_gemm_nt, bf16_gemm_nt, fp8_paged_mqa_logits, the mega_moe APIs, ...).
# The tests `import deep_gemm` and call exactly those top-level functions, so
# running them against the installed wheel validates both the compiled binding
# and the wrapper — i.e. the fork's added surface, not just upstream.
#
# CRITICAL: tests must run from <DEEPGEMM_SRC>/tests so that `import deep_gemm`
# resolves to the installed wheel in site-packages, NOT the source tree's
# deep_gemm/ directory (whose __init__.py differs and JIT-rebuilds _C). The
# tests/ dir has no deep_gemm/ subdir, and `import generators` still resolves
# to the local tests/generators.py.
#
# Usage: test_sgl_deep_gemm.sh <DEEPGEMM_SRC> [options]
#   DEEPGEMM_SRC        path to a checkout of sgl-project/DeepGEMM
# Options:
#   --max-procs N       cap multi-GPU process count (default: detected GPUs)
#   --skip-sanitizer    skip the (slow) compute-sanitizer pass
#   --skip-mega-moe     skip the heavy mega_moe family even on SM100
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <DEEPGEMM_SRC> [--max-procs N] [--skip-sanitizer] [--skip-mega-moe]" >&2
  exit 1
fi

DEEPGEMM_SRC="$(cd "$1" && pwd)"; shift
MAX_PROCS=""
SKIP_SANITIZER=0
SKIP_MEGA_MOE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --max-procs) MAX_PROCS="$2"; shift 2 ;;
    --skip-sanitizer) SKIP_SANITIZER=1; shift ;;
    --skip-mega-moe) SKIP_MEGA_MOE=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

TESTS_DIR="${DEEPGEMM_SRC}/tests"
PYTHON="${PYTHON:-python3}"

if [ ! -d "${TESTS_DIR}" ]; then
  echo "No tests/ directory under ${DEEPGEMM_SRC}" >&2
  exit 1
fi

# --- Detect GPU count and architecture ------------------------------------
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -eq 0 ]; then
  echo "No GPUs visible to nvidia-smi — DeepGEMM tests require a GPU." >&2
  exit 1
fi
# compute_cap like "9.0" / "10.3"; arch major 9 == Hopper (SM90), 10 == Blackwell (SM100).
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
ARCH_MAJOR=${COMPUTE_CAP%%.*}

NPROC=${NUM_GPUS}
if [ -n "${MAX_PROCS}" ] && [ "${MAX_PROCS}" -lt "${NPROC}" ]; then
  NPROC=${MAX_PROCS}
fi

# --- Verify we are testing the INSTALLED wheel, not the source tree -------
DG_FILE=$(cd "${TESTS_DIR}" && "${PYTHON}" -c "import deep_gemm, sys; sys.stdout.write(deep_gemm.__file__)" 2>/dev/null)
if [ -z "${DG_FILE}" ]; then
  echo "Failed to import deep_gemm — is the wheel installed?" >&2
  exit 1
fi
case "${DG_FILE}" in
  "${DEEPGEMM_SRC}"/*)
    echo "ERROR: 'import deep_gemm' resolved to the source tree (${DG_FILE})," >&2
    echo "       not the installed wheel. Aborting to avoid a false-positive test." >&2
    exit 1 ;;
esac

echo "=============================================================="
echo " DeepGEMM wheel test run"
echo "   deep_gemm:    ${DG_FILE}"
echo "   GPUs:         ${NUM_GPUS} (compute_cap ${COMPUTE_CAP}, arch major ${ARCH_MAJOR})"
echo "   processes:    ${NPROC}"
echo "   skip-mega:    ${SKIP_MEGA_MOE}   skip-sanitizer: ${SKIP_SANITIZER}"
echo "=============================================================="

PASSED=()
FAILED=()
SKIPPED=()

# run_test <relative-test-path> [extra args...]
run_test() {
  local name="$1"; shift
  echo ""
  echo "----- RUN ${name} $* -----"
  if (cd "${TESTS_DIR}" && "${PYTHON}" "${name}" "$@"); then
    echo "----- PASS ${name} -----"
    PASSED+=("${name}")
  else
    echo "----- FAIL ${name} (exit $?) -----"
    FAILED+=("${name}")
  fi
}

skip_test() {
  echo ""
  echo "----- SKIP $1 ($2) -----"
  SKIPPED+=("$1 ($2)")
}

# --- Single-GPU correctness (both SM90 and SM100; arch-gated internally) ---
SINGLE_GPU_TESTS=(
  test_bf16.py
  test_einsum.py
  test_fp8_fp4.py
  test_hyperconnection.py
  test_layout.py
  test_legacy.py
  test_attention.py
)
for t in "${SINGLE_GPU_TESTS[@]}"; do
  if [ -f "${TESTS_DIR}/${t}" ]; then
    run_test "${t}"
  else
    skip_test "${t}" "not present in this branch"
  fi
done

# --- Lazy init: multi-process, trivial, both archs ------------------------
if [ -f "${TESTS_DIR}/test_lazy_init.py" ]; then
  run_test test_lazy_init.py --num-processes "${NPROC}"
fi

# --- mega_moe family: SM100 (Blackwell) only ------------------------------
# These use SM100 fp4 + symmetric-memory kernels and carry an upstream
# "TODO: skip the test for SM90" note, so they are gated to arch major >= 10.
MEGA_MOE_MULTI=(
  test_mega_moe.py
  test_mega_moe_l1_fp4_accuracy.py
  test_mega_moe_l1_sentinel.py
)
if [ "${SKIP_MEGA_MOE}" -eq 1 ]; then
  for t in "${MEGA_MOE_MULTI[@]}" test_mega_moe_pre_dispatch.py; do
    [ -f "${TESTS_DIR}/${t}" ] && skip_test "${t}" "--skip-mega-moe"
  done
elif [ "${ARCH_MAJOR}" -ge 10 ]; then
  for t in "${MEGA_MOE_MULTI[@]}"; do
    [ -f "${TESTS_DIR}/${t}" ] && run_test "${t}" --num-processes "${NPROC}"
  done
  # pre_dispatch runs in a single process (no --num-processes flag).
  [ -f "${TESTS_DIR}/test_mega_moe_pre_dispatch.py" ] && run_test test_mega_moe_pre_dispatch.py
else
  for t in "${MEGA_MOE_MULTI[@]}" test_mega_moe_pre_dispatch.py; do
    [ -f "${TESTS_DIR}/${t}" ] && skip_test "${t}" "SM100-only, arch major ${ARCH_MAJOR}"
  done
fi

# --- compute-sanitizer pass (re-runs the others under memcheck/synccheck) -
if [ -f "${TESTS_DIR}/test_sanitizer.py" ]; then
  if [ "${SKIP_SANITIZER}" -eq 1 ]; then
    skip_test test_sanitizer.py "--skip-sanitizer"
  elif [ ! -x /usr/local/cuda/bin/compute-sanitizer ] && ! command -v compute-sanitizer >/dev/null 2>&1; then
    skip_test test_sanitizer.py "compute-sanitizer not found"
  else
    run_test test_sanitizer.py
  fi
fi

# --- Summary --------------------------------------------------------------
echo ""
echo "=============================================================="
echo " Summary: ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#PASSED[@]}  -gt 0 ] && printf '   PASS  %s\n'  "${PASSED[@]}"
[ ${#SKIPPED[@]} -gt 0 ] && printf '   SKIP  %s\n'  "${SKIPPED[@]}"
[ ${#FAILED[@]}  -gt 0 ] && printf '   FAIL  %s\n'  "${FAILED[@]}"
echo "=============================================================="

[ ${#FAILED[@]} -eq 0 ]
