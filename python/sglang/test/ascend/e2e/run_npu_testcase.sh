test_case=$1

sglang_source_path=/root/sglang
if [ ! -f "${sglang_source_path}/${test_case}" ];then
  echo "The test case file is not exist: $test_case"
  exit 0
fi

# Optional single test method ("Class.method") injected as NPU_TEST_METHOD;
# empty runs every test defined in the test case file.
test_method="${NPU_TEST_METHOD:-}"
test_method_args=()
if [ -n "${test_method}" ]; then
  echo "Restricting run to test method: ${test_method}"
  test_method_args=("${test_method}")
fi

# Leave the image WORKDIR (/sgl-workspace): the sglang repo cloned there can
# shadow the installed package on sys.path.
cd "${sglang_source_path}"

# CI hosts mount /root/.cache/huggingface read-only, but --trust-remote-code
# needs a writable transformers dynamic-module cache.
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf_modules}"

echo "NPU info:"
npu-smi info

# Apply the declarative container patch sets requested by the workflow
# (SGLANG_NPU_CONTAINER_PATCHES, comma separated set names). Patch sets live in
# scripts/ci/npu/patches/<name>.sh and an empty value is a no-op.
#
# Notes:
#   * The set name comes from a workflow input, so only names matching
#     ^[a-z0-9_]+(\.[a-z0-9_]+)*$ (dots allowed for version suffixes, e.g.
#     mooncake_memfabric_1.3.1) that resolve to a file in the patch directory
#     are executed.
#   * Patch resources are cached in the pod-local filesystem (PATCH_CACHE_DIR,
#     /tmp by default) instead of the shared disk, so the pods never race on
#     the same file. Each pod downloads once and reuses it for re-runs.
#   * A per-set marker keyed by the file checksum makes re-runs idempotent.
#   * Patch sets download packages from the public internet, which is slow from
#     the cluster; pip defaults to the domestic mirror (override with
#     PIP_INDEX_URL) while apt mirrors are handled per set.
#   * Any failure aborts the run: a half patched cluster produces results that
#     cannot be trusted.
#
# The block runs in a subshell so set -euo pipefail applies only here.
if [ -n "${SGLANG_NPU_CONTAINER_PATCHES:-}" ]; then
    echo "===== Apply container patches - Begin ====="
    (
        set -euo pipefail

        PATCH_DIR="${sglang_source_path}/scripts/ci/npu/patches"
        PATCH_CACHE_DIR="${PATCH_CACHE_DIR:-/tmp/sglang_npu_patch_cache}"
        PATCH_STATE_FILE="${PATCH_STATE_FILE:-/tmp/sglang_npu_patches_applied.json}"
        REQUESTED="${SGLANG_NPU_CONTAINER_PATCHES}"

        # Pods reach PyPI over a slow international link; every patch set
        # downloads wheels, so default to the domestic mirror.
        export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple}"

        APPLIED_JSON=""
        NORMALIZED=""
        while IFS= read -r set_name; do
            set_name="$(echo "${set_name}" | tr -d '[:space:]')"
            if [ -z "${set_name}" ]; then
                continue
            fi

            if ! echo "${set_name}" | grep -Eq '^[a-z0-9_]+(\.[a-z0-9_]+)*$'; then
                echo "Invalid container patch set name: ${set_name}" >&2
                exit 1
            fi

            patch_file="${PATCH_DIR}/${set_name}.sh"
            if [ ! -f "${patch_file}" ]; then
                echo "Container patch set not found: ${patch_file}" >&2
                exit 1
            fi

            patch_sha="$(sha256sum "${patch_file}" | awk '{print $1}')"
            patch_version="$(grep -m1 '^# PATCH_VERSION:' "${patch_file}" | sed 's/^# PATCH_VERSION:[[:space:]]*//' || true)"
            patch_version="${patch_version:-unknown}"
            marker="/tmp/.sglang_npu_patch_${set_name}_${patch_sha}"

            if [ -f "${marker}" ]; then
                echo "Container patch set already applied, skip: ${set_name}"
                patch_status="skipped"
            else
                echo "===== Apply container patch set: ${set_name} - Begin ====="
                mkdir -p "${PATCH_CACHE_DIR}/${set_name}"
                PATCH_CACHE_DIR="${PATCH_CACHE_DIR}/${set_name}" bash "${patch_file}"
                echo "===== Apply container patch set: ${set_name} - End ====="
                touch "${marker}"
                patch_status="applied"
            fi

            if [ -n "${APPLIED_JSON}" ]; then
                APPLIED_JSON="${APPLIED_JSON},"
                NORMALIZED="${NORMALIZED},"
            fi
            NORMALIZED="${NORMALIZED}${set_name}"
            APPLIED_JSON="${APPLIED_JSON}{\"set\": \"${set_name}\", \"version\": \"${patch_version}\", \"sha256\": \"${patch_sha}\", \"status\": \"${patch_status}\"}"
        done <<< "$(echo "${REQUESTED}" | tr ',' '\n')"

        cat > "${PATCH_STATE_FILE}" <<EOF
{
  "requested": "${NORMALIZED}",
  "host": "${HOSTNAME:-unknown}",
  "recorded_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "patches": [${APPLIED_JSON}]
}
EOF

        echo "Container patch state written to ${PATCH_STATE_FILE}"
        cat "${PATCH_STATE_FILE}"
    ) || {
        echo "Failed to apply the requested container patches."
        exit 1
    }
    echo "===== Apply container patches - End ====="
else
    echo "No container patch set requested, skip."
fi

echo "===== Install kubernetes - Begin ====="
KUBERNETES_PKG_PATH_SOURCE=/root/.cache/.cache/kubernetes
if [ ! -d "${KUBERNETES_PKG_PATH_SOURCE}" ]; then
  echo "Install kubernetes online."
  pip install kubernetes -i -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
else
  echo "Install kubernetes locally."
  cp -r ${KUBERNETES_PKG_PATH_SOURCE} /tmp/
  pip install --no-index --find-links=/tmp/kubernetes/ kubernetes
fi
echo "===== Install kubernetes - End ====="

PYTHON_FOR_SGLANG="python"
PIP_FOR_SGLANG="pip"
if [ -n "${TRANSFORMERS_VERSION_FOR_SGLANG}" ];then
  echo "===== Install transformers for sglang - Begin ====="
  TRANSFORMERS_PKG_PATH_SOURCE=/root/.cache/.cache/transformers/${TRANSFORMERS_VERSION_FOR_SGLANG}
  if [ ! -d "${TRANSFORMERS_PKG_PATH_SOURCE}" ]; then
    echo "The dependent transformers package does not exist: ${TRANSFORMERS_PKG_PATH_SOURCE}."
    echo "Install transformers ${TRANSFORMERS_VERSION_FOR_SGLANG} online."
    pip install transformers=="${TRANSFORMERS_VERSION_FOR_SGLANG}" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  else
    echo "Install transformers ${TRANSFORMERS_VERSION_FOR_SGLANG} locally."
    TRANSFORMERS_PKG_PATH_TARGET=/tmp/transformers/${TRANSFORMERS_VERSION_FOR_SGLANG}
    mkdir -p "${TRANSFORMERS_PKG_PATH_TARGET}"
    cp "${TRANSFORMERS_PKG_PATH_SOURCE}/"* "${TRANSFORMERS_PKG_PATH_TARGET}/"
    pip install --no-index --find-links="${TRANSFORMERS_PKG_PATH_TARGET}" transformers=="${TRANSFORMERS_VERSION_FOR_SGLANG}"
  fi
  echo "===== Install transformers for sglang in virtual env - End ====="
fi

if [ -n "${TRANSFORMERS_VERSION_FOR_TEST_TOOL}" ]; then
  # Example: TRANSFORMERS_VERSION_FOR_TEST_TOOL=4.57.6
  echo "===== Install transformers in virtual env for test tools - Begin ====="
  PYTHON_ENV_FOR_TEST_TOOL=python_venv_for_test_tool
  PIP_FOR_TEST_TOOL=${PYTHON_ENV_FOR_TEST_TOOL}/bin/pip
  python -m venv ${PYTHON_ENV_FOR_TEST_TOOL} --system-site-packages
  TRANSFORMERS_PKG_PATH_SOURCE=/root/.cache/.cache/transformers/${TRANSFORMERS_VERSION_FOR_TEST_TOOL}
  if [ ! -d "${TRANSFORMERS_PKG_PATH_SOURCE}" ]; then
    echo "The dependent transformers package does not exist: ${TRANSFORMERS_PKG_PATH_SOURCE}."
    echo "Install transformers ${TRANSFORMERS_VERSION_FOR_TEST_TOOL} online."
    ${PIP_FOR_TEST_TOOL} install transformers==${TRANSFORMERS_VERSION_FOR_TEST_TOOL} -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
  else
    echo "Install transformers ${TRANSFORMERS_VERSION_FOR_TEST_TOOL} locally."
    TRANSFORMERS_PKG_PATH_TARGET=/tmp/transformers/${TRANSFORMERS_VERSION_FOR_TEST_TOOL}
    mkdir -p ${TRANSFORMERS_PKG_PATH_TARGET}
    cp ${TRANSFORMERS_PKG_PATH_SOURCE}/* ${TRANSFORMERS_PKG_PATH_TARGET}/
    ${PIP_FOR_TEST_TOOL} install --no-index --find-links=${TRANSFORMERS_PKG_PATH_TARGET} transformers==${TRANSFORMERS_VERSION_FOR_TEST_TOOL}
  fi
  echo "===== Install transformers in virtual env for test tools - End ====="
  echo "Transformers version for test tools: $(${PIP_FOR_TEST_TOOL} show transformers | grep Version | cut -d: -f2)"
fi

echo "Transformers version for sglang: $(${PIP_FOR_SGLANG} show transformers | grep Version | cut -d: -f2)"

# copy or download required file
cp /root/.cache/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
#curl -o /tmp/test.jsonl -L https://gh-proxy.test.osinfra.cn/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
cp /root/.cache/modelscope/hub/datasets/grade_school_math/test.jsonl /tmp

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

# Prefer the sgl-router built for this commit (CI artifact), then whatever PATH
# resolves, and finally a prebuilt aarch64 package published on OBS. The image
# ships neither, and the pip-installed `sglang-router` is a different CLI.
SGL_ROUTER_OBS_URL="https://sglang-npu.obs.cn-southwest-2.myhuaweicloud.com/sgl-router/bin/sgl-router-aarch64-npu.zip"

download_sgl_router() {
    local archive=/tmp/sgl-router-aarch64-npu.zip
    local dest_dir=/tmp/sgl-router-bin
    mkdir -p "${dest_dir}"
    echo "sgl-router not found locally, downloading ${SGL_ROUTER_OBS_URL}"
    # OBS lives in the same region as the cluster; bypass any image proxy.
    if command -v curl >/dev/null 2>&1; then
        (
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
            curl -fsSL -o "${archive}" "${SGL_ROUTER_OBS_URL}"
        ) || return 1
    elif command -v wget >/dev/null 2>&1; then
        (
            unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
            wget -q -O "${archive}" "${SGL_ROUTER_OBS_URL}"
        ) || return 1
    else
        echo "Neither curl nor wget is available to download sgl-router"
        return 1
    fi
    python -m zipfile -e "${archive}" "${dest_dir}" || return 1
    local bin
    bin="$(find "${dest_dir}" -type f -name 'sgl-router' -print -quit)"
    if [ -z "${bin}" ]; then
        echo "sgl-router binary not found in ${archive}"
        return 1
    fi
    chmod +x "${bin}"
    DOWNLOADED_SGL_ROUTER="${bin}"
}

sgl_router_bin="${sglang_source_path}/sgl-router-bin/sgl-router"
if "${sgl_router_bin}" --help >/dev/null 2>&1; then
    export SGLANG_TEST_PD_ROUTER_BINARY="${sgl_router_bin}"
elif command -v sgl-router >/dev/null 2>&1; then
    export SGLANG_TEST_PD_ROUTER_BINARY="$(command -v sgl-router)"
elif download_sgl_router; then
    export SGLANG_TEST_PD_ROUTER_BINARY="${DOWNLOADED_SGL_ROUTER}"
else
    echo "WARNING: sgl-router is unavailable, PD router tests will fail"
fi
echo "sgl-router: ${SGLANG_TEST_PD_ROUTER_BINARY:-not found}"

export SGLANG_TEST_MAX_RETRY=0
export SGLANG_SET_CPU_AFFINITY=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

visibe_devices=$ASCEND_VISIBLE_DEVICES
echo "ASCEND_VISIBLE_DEVICES=$ASCEND_VISIBLE_DEVICES"
if [ "${visibe_devices}" != "" ];then
    ASCEND_RT_VISIBLE_DEVICES=$(echo "$ASCEND_VISIBLE_DEVICES" | tr ',' '\n' | sort -n | tr '\n' ',')
    export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES%,}
    echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
    export ASCEND_VISIBLE_DEVICES=""
fi

unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

# The sglang package comes from the image's site-packages (PYTHONPATH below is intentionally NOT prepended).
# However, test helper modules shipped in the PR  must be available on top of whatever the image bundled,
# so we rsync the checkout's python/sglang/test tree into site-packages before anything imports it.
IMAGE_SGLANG_PKG="$(python -c 'import sglang, os; print(os.path.dirname(sglang.__file__))')"
if [ -d "${IMAGE_SGLANG_PKG}/test" ] && [ -d "${sglang_source_path}/python/sglang/test" ]; then
    echo "Syncing PR sglang/test → ${IMAGE_SGLANG_PKG}/test"
    cp -r "${sglang_source_path}/python/sglang/test/ascend/"* "${IMAGE_SGLANG_PKG}/test/ascend/"
fi

# set environment of cann
. /usr/local/Ascend/cann/set_env.sh
. /usr/local/Ascend/nnal/atb/set_env.sh
# Adapt Deepseek-V4-Flash test cases with additional environment variables.
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash || true
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash || true
# Additional environment variables required to enable DSPARK for Deepseek-V4 in the 950 environment
export SGLANG_DSPARK_A5_EXTRA_OPS_SO=/usr/local/Ascend/extra-ops/vllm_ascend_C.cpython-312-x86_64-linux-gnu.so
export ASCEND_CUSTOM_OPP_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/:$ASCEND_CUSTOM_OPP_PATH
export LD_LIBRARY_PATH=/usr/local/Ascend/extra-ops/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH

export TORCH_EXTENSIONS_DIR=/tmp/torch_extensions

echo "Running test case ${test_case}"
tc_name=${test_case##*/}
tc_name=${tc_name%.*}
run_label="${RUN_LABEL:-unknown}"
log_path="/root/sglang/debug/logs/log/${run_label}/${tc_name}/${HOSTNAME}"
if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
    # In CI, persist logs under /root/.cache/tests/logs so they can be collected
    log_path="/root/.cache/tests/logs/log/${run_label}/${tc_name}/${HOSTNAME}"
fi
rm -rf "${log_path}"
mkdir -p "${log_path}"
echo "Log path: ${log_path}"

if [ "${TROUBLE_SHOTTING}" = "true" ] || [ "${TROUBLE_SHOTTING}" = "True" ];then
    echo "TROUBLE_SHOTTING=true, the pod will keep alive for four hour."
    ( ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" "${test_method_args[@]}" 2>&1 || true ) | tee -a "${log_path}/${tc_name}.log"
    sleep 14400
else
    ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" "${test_method_args[@]}" 2>&1 | tee -a "${log_path}/${tc_name}.log"
fi
echo "Finished test case ${test_case}"

if [ -n "${METRICS_DATA_FILE}" ]; then
    mkdir -p "${METRICS_DATA_FILE}"
    # Archive the test log into the output directory for result collection
    cp "${log_path}/${tc_name}.log" "${METRICS_DATA_FILE}/test_output.log"
    echo "Metrics log saved to ${METRICS_DATA_FILE}/test_output.log"
    # Record which container patch sets (if any) produced this result.
    if [ -f "/tmp/sglang_npu_patches_applied.json" ]; then
        cp /tmp/sglang_npu_patches_applied.json "${METRICS_DATA_FILE}/patch_info.json"
        echo "Container patch info saved to ${METRICS_DATA_FILE}/patch_info.json"
    fi
fi

source_plog_path="/root/ascend/log/debug/plog"
if [ -d "$source_plog_path" ];then
    echo "Plog files found. Begin to backup them."
    target_plog_path="/root/sglang/debug/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
    if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
        target_plog_path="/root/.cache/tests/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
    fi
    rm -rf "${target_plog_path}"
    mkdir -p "${target_plog_path}"
    cp ${source_plog_path}/* "${target_plog_path}"
fi
