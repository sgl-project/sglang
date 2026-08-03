#!/usr/bin/env bash
set -euo pipefail

BAD_NODE="smci355-ccs-aus-n08-21"
if [[ "$(hostname -s)" == "$BAD_NODE" ]]; then
    echo "Refusing to run ROCm GPU tests on known-bad node $BAD_NODE" >&2
    exit 97
fi

ARM="${DWDP_TEST_ARM:?DWDP_TEST_ARM is required}"
IMAGE="${IMAGE:-rocm/pytorch-private:sglang-main-20260731-hisparse32368-rebased-e776c8a9}"
SGLANG_WORKTREE="${SGLANG_WORKTREE:-/home/yanfwang/workspace/sglang-dwdp-rocm}"
AITER_WORKTREE="${AITER_WORKTREE:-/home/yanfwang/workspace/aiter-dwdp-rocm}"
RESULT_ROOT="${RESULT_ROOT:-/home/yanfwang/workspace/dwdp_rocm_slurm_results}"
RESULT_DIR="${RESULT_ROOT}/${SLURM_JOB_ID}-${ARM}"

mkdir -p "$RESULT_DIR"
exec > >(tee "${RESULT_DIR}/stdout.log") 2> >(tee "${RESULT_DIR}/stderr.log" >&2)

python3 "${SGLANG_WORKTREE}/scripts/ci/amd/check_dwdp_rocm_vram.py" \
    --expected-gpus "${EXPECTED_GPU_COUNT:-8}" \
    --max-used-gib "${MAX_USED_VRAM_GIB:-4}" \
    --max-skew-gib "${MAX_VRAM_SKEW_GIB:-2}"

postflight_vram() {
    local job_status=$?
    local attempt
    local clean=0
    trap - EXIT
    set +e
    for ((attempt = 1; attempt <= ${VRAM_POSTFLIGHT_ATTEMPTS:-6}; attempt++)); do
        echo "VRAM postflight attempt ${attempt}/${VRAM_POSTFLIGHT_ATTEMPTS:-6}"
        if python3 "${SGLANG_WORKTREE}/scripts/ci/amd/check_dwdp_rocm_vram.py" \
            --expected-gpus "${EXPECTED_GPU_COUNT:-8}" \
            --max-used-gib "${MAX_USED_VRAM_GIB:-4}" \
            --max-skew-gib "${MAX_VRAM_SKEW_GIB:-2}"; then
            clean=1
            break
        fi
        sleep "${VRAM_POSTFLIGHT_INTERVAL_SECONDS:-10}"
    done
    if [[ "$clean" -ne 1 ]]; then
        echo "VRAM postflight failed; leaving the node untouched and marking it unsafe" >&2
        if [[ "$job_status" -eq 0 ]]; then
            job_status=99
        fi
    fi
    exit "$job_status"
}
trap postflight_vram EXIT

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname -s)"
echo "arm=${ARM}"
echo "image=${IMAGE}"
echo "rocm=$(cat /opt/rocm/.info/version 2>/dev/null || true)"
git -C "$SGLANG_WORKTREE" log -1 --format='sglang=%H %s'
git -C "$AITER_WORKTREE" log -1 --format='aiter=%H %s'

case "$ARM" in
    vmm-poc)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 -m pytest -q test/registered/unit/layers/test_hip_vmm.py test/registered/amd/moe/test_dwdp_vmm_manager_amd.py && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 -m pytest -q test/registered/amd/moe/test_dwdp_hsa_copy_amd.py"
        ;;
    vmm-access-probe)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 benchmark/dwdp/probe_hip_vmm_access.py"
        ;;
    ipc-poc)
        TEST_COMMAND="python3 -m pytest -q test/registered/amd/moe/test_dwdp_ipc_amd.py"
        ;;
    aiter-multib)
        TEST_COMMAND="cd /sgl-workspace/aiter && python3 -m pytest -q op_tests/test_moe_multi_b_contract.py op_tests/test_moe_multi_b_rocm.py && cd /sgl-workspace/sglang && python3 -m pytest -q test/registered/unit/layers/moe/test_aiter_runner.py test/registered/unit/layers/moe/test_dwdp_hsa_fallback.py test/registered/unit/layers/moe/test_dwdp_page_pool.py test/registered/unit/layers/moe/test_dwdp_tensor_schema.py test/registered/unit/layers/moe/test_dwdp_layout.py test/registered/unit/server_args/test_server_args.py::TestDwdpWeightBackend test/registered/unit/test_runtime_context.py::TestRuntimeContextSingletons::test_reset_context_cleans_up_dwdp_manager"
        ;;
    synthetic-2)
        TEST_COMMAND="DWDP_TEST_WORLD_SIZE=2 python3 -m pytest -q test/registered/amd/moe/test_dwdp_prefetch_amd.py"
        ;;
    synthetic-4)
        TEST_COMMAND="DWDP_TEST_WORLD_SIZE=4 python3 -m pytest -q test/registered/amd/moe/test_dwdp_prefetch_amd.py"
        ;;
    synthetic-8)
        TEST_COMMAND="DWDP_TEST_WORLD_SIZE=8 python3 -m pytest -q test/registered/amd/moe/test_dwdp_prefetch_amd.py"
        ;;
    hsa-benchmark)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 benchmark/dwdp/benchmark_hsa_copy.py"
        ;;
    model-smoke)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH bash test/manual/dwdp/run_model_smoke_rocm.sh"
        ;;
    standalone)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 -m pytest -q test/registered/amd/test_dwdp_standalone_amd.py"
        ;;
    pd)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH python3 -m pytest -q test/registered/amd/disaggregation/test_disaggregation_dwdp_amd.py"
        ;;
    benchmark)
        TEST_COMMAND="rm -rf /tmp/dwdp-aot && cp -a python/sglang/kernels/aot /tmp/dwdp-aot && cd /tmp/dwdp-aot && rm -rf build && python3 setup_rocm.py build_ext --inplace && cd /sgl-workspace/sglang && PYTHONPATH=/tmp/dwdp-aot/python:\$PYTHONPATH bash benchmark/dwdp/run_rocm_backend_comparison.sh"
        ;;
    *)
        echo "Unknown DWDP_TEST_ARM: $ARM" >&2
        exit 2
        ;;
esac

if [[ -n "${DWDP_TEST_COMMAND:-}" ]]; then
    TEST_COMMAND="$DWDP_TEST_COMMAND"
fi
printf 'command=%q\n' "$TEST_COMMAND"
printf '%s\n' "$TEST_COMMAND" > "${RESULT_DIR}/command.txt"

docker run --rm \
    --privileged \
    --user "$(id -u):$(id -g)" \
    --ulimit memlock=-1:-1 \
    --network=host \
    --ipc=host \
    --shm-size=256g \
    --tmpfs /tmp:rw,exec,nosuid,size=64g \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e HOME=/tmp/dwdp-home \
    -e USER="$USER" \
    -e LOGNAME="$USER" \
    -e SGLANG_USE_AITER=1 \
    -e AMDGPU_TARGET=gfx950 \
    -e BACKENDS="${BACKENDS:-}" \
    -e MODEL_PATH="${MODEL_PATH:-}" \
    -e DWDP_SIZE="${DWDP_SIZE:-}" \
    -e TP_SIZE="${TP_SIZE:-}" \
    -e INPUT_LENGTHS="${INPUT_LENGTHS:-}" \
    -e NUM_PROMPTS="${NUM_PROMPTS:-}" \
    -e MAX_CONCURRENCY="${MAX_CONCURRENCY:-}" \
    -e CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-}" \
    -e BASE_PORT="${BASE_PORT:-}" \
    -e EXPECTED_GPU_COUNT="${EXPECTED_GPU_COUNT:-8}" \
    -e MAX_USED_VRAM_GIB="${MAX_USED_VRAM_GIB:-4}" \
    -e MAX_VRAM_SKEW_GIB="${MAX_VRAM_SKEW_GIB:-2}" \
    -e SGLANG_DWDP_TEST_BACKEND="${SGLANG_DWDP_TEST_BACKEND:-ipc}" \
    -e SGLANG_DWDP_DISABLE_HSA_COPY="${SGLANG_DWDP_DISABLE_HSA_COPY:-0}" \
    -e NIXL_LOG_LEVEL="${NIXL_LOG_LEVEL:-ERROR}" \
    -e UCX_LOG_LEVEL="${UCX_LOG_LEVEL:-error}" \
    -e DWDP_TEST_WORLD_SIZE="${DWDP_TEST_WORLD_SIZE:-2}" \
    -e DWDP_TEST_NUM_LAYERS="${DWDP_TEST_NUM_LAYERS:-3}" \
    -e DWDP_TEST_EXPERT_BYTES="${DWDP_TEST_EXPERT_BYTES:-32}" \
    -e PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/sglang/python:/sgl-workspace/sglang/python/sglang/kernels/aot/python \
    -v "${SGLANG_WORKTREE}:/sgl-workspace/sglang" \
    -v "${AITER_WORKTREE}:/sgl-workspace/aiter" \
    -v "${RESULT_DIR}:/results" \
    -v /apps/data/models:/models:ro \
    -w /sgl-workspace/sglang \
    "$IMAGE" \
    bash -lc "set -euo pipefail; mkdir -p \"\$HOME\"; memlock=\$(ulimit -l); echo \"memlock=\${memlock}\"; [[ \"\$memlock\" == unlimited ]] || { echo \"DWDP GPU jobs require unlimited memlock for UCX/NIXL (got \$memlock KiB)\" >&2; exit 98; }; ${TEST_COMMAND}"
