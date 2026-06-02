#!/usr/bin/env bash
# Run the sgl-project/DeepGEMM test suite against an installed sgl-deep-gemm
# wheel, as a pre-release gate for release-whl-deepgemm.yml.
#
# Tests must run from <DEEPGEMM_SRC>/tests so `import deep_gemm` resolves to the
# installed wheel (which ships the pre-compiled tvm-ffi _C.so) rather than the
# source tree's deep_gemm/ package, which differs and JIT-rebuilds _C. The guard
# below aborts if that resolution is wrong, to avoid a false-positive gate.
#
# Usage: test_sgl_deep_gemm.sh <DEEPGEMM_SRC> [--max-procs N] [--skip-sanitizer] [--skip-mega-moe]
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

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -eq 0 ]; then
  echo "No GPUs visible to nvidia-smi — DeepGEMM tests require a GPU." >&2
  exit 1
fi
# arch major: 9 == Hopper (SM90), 10 == Blackwell (SM100).
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
ARCH_MAJOR=${COMPUTE_CAP%%.*}

NPROC=${NUM_GPUS}
if [ -n "${MAX_PROCS}" ] && [ "${MAX_PROCS}" -lt "${NPROC}" ]; then
  NPROC=${MAX_PROCS}
fi

DG_FILE=$(cd "${TESTS_DIR}" && "${PYTHON}" -c "import deep_gemm, sys; sys.stdout.write(deep_gemm.__file__)" 2>/dev/null)
if [ -z "${DG_FILE}" ]; then
  echo "Failed to import deep_gemm — is the wheel installed?" >&2
  exit 1
fi
case "${DG_FILE}" in
  "${DEEPGEMM_SRC}"/*)
    echo "ERROR: 'import deep_gemm' resolved to the source tree (${DG_FILE})," >&2
    echo "       not the installed wheel. Aborting." >&2
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

if [ -f "${TESTS_DIR}/test_lazy_init.py" ]; then
  run_test test_lazy_init.py --num-processes "${NPROC}"
fi

# mega_moe family uses SM100 fp4 + symmetric-memory kernels (SM100-only).
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
  [ -f "${TESTS_DIR}/test_mega_moe_pre_dispatch.py" ] && run_test test_mega_moe_pre_dispatch.py
else
  for t in "${MEGA_MOE_MULTI[@]}" test_mega_moe_pre_dispatch.py; do
    [ -f "${TESTS_DIR}/${t}" ] && skip_test "${t}" "SM100-only, arch major ${ARCH_MAJOR}"
  done
fi

if [ -f "${TESTS_DIR}/test_sanitizer.py" ]; then
  if [ "${SKIP_SANITIZER}" -eq 1 ]; then
    skip_test test_sanitizer.py "--skip-sanitizer"
  elif [ ! -x /usr/local/cuda/bin/compute-sanitizer ] && ! command -v compute-sanitizer >/dev/null 2>&1; then
    skip_test test_sanitizer.py "compute-sanitizer not found"
  else
    run_test test_sanitizer.py
  fi
fi

echo ""
echo "=============================================================="
echo " Summary: ${#PASSED[@]} passed, ${#FAILED[@]} failed, ${#SKIPPED[@]} skipped"
[ ${#PASSED[@]}  -gt 0 ] && printf '   PASS  %s\n'  "${PASSED[@]}"
[ ${#SKIPPED[@]} -gt 0 ] && printf '   SKIP  %s\n'  "${SKIPPED[@]}"
[ ${#FAILED[@]}  -gt 0 ] && printf '   FAIL  %s\n'  "${FAILED[@]}"
echo "=============================================================="

[ ${#FAILED[@]} -eq 0 ]
