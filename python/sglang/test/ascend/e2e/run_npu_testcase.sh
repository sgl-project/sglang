# Accept one or more test_case paths (space separated). Loop over them so a single
# Pod allocation can run a batch of cases that share the same resource profile.
TEST_CASES=("$@")
if [ ${#TEST_CASES[@]} -eq 0 ];then
  echo "No test case provided. Usage: run_npu_testcase.sh <case1.py> [case2.py ...]"
  exit 0
fi

sglang_source_path=/root/sglang
# Validate all cases exist up front so we fail fast before allocating NPU time.
for tc in "${TEST_CASES[@]}";do
  if [ ! -f "${sglang_source_path}/${tc}" ];then
    echo "The test case file is not exist: $tc"
    exit 0
  fi
done

echo "NPU info:"
npu-smi info

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

# use sglang from source or from image
if [ "${INSTALL_SGLANG_FROM_SOURCE}" = "true" ] || [ "${INSTALL_SGLANG_FROM_SOURCE}" = "True" ];then
    echo "Use sglang from source: ${sglang_source_path}"
    export PYTHONPATH=${sglang_source_path}/python:$PYTHONPATH
else
    echo "Use sglang from docker image"
    sglang_pkg_path=/sgl-workspace/sglang/python
    ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend
    mkdir -p "${ascend_test_util_path}"
    mv "${ascend_test_util_path}" "${ascend_test_util_path}_bak"
    cp -r ${sglang_source_path}/python/sglang/test/ascend "${ascend_test_util_path}"
fi

# set environment of cann
. /usr/local/Ascend/cann/set_env.sh
. /usr/local/Ascend/nnal/atb/set_env.sh
# Adapt DeepSeek-V4-Flash test cases with additional environment variables.
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash || true
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash || true

echo "Running ${#TEST_CASES[@]} test case(s): ${TEST_CASES[*]}"
OVERALL_SUCCESS=true
case_index=0
total_cases=${#TEST_CASES[@]}
# run_label is injected as RUN_LABEL (derived from the metrics output path) and
# replaces the old date-based prefix so logs can be grouped per workflow run.
run_label="${RUN_LABEL:-unknown}"

for test_case in "${TEST_CASES[@]}";do
    case_index=$((case_index+1))
    tc_name=${test_case##*/}
    tc_name=${tc_name%.*}
    log_path="/root/sglang/debug/logs/log/${run_label}/${tc_name}/${HOSTNAME}"
    if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
        # In CI, persist logs under /root/.cache/tests/logs so they can be collected
        log_path="/root/.cache/tests/logs/log/${run_label}/${tc_name}/${HOSTNAME}"
    fi
    rm -rf "${log_path}"
    mkdir -p "${log_path}"
    echo "===== [$(date)] Case ${case_index}/${total_cases}: ${tc_name} (log: ${log_path}) ====="

    if [ "${TROUBLE_SHOTTING}" = "true" ] || [ "${TROUBLE_SHOTTING}" = "True" ];then
        echo "TROUBLE_SHOTTING=true, the pod will keep alive for four hour."
        ( ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" 2>&1 || true ) | tee -a "${log_path}/${tc_name}.log"
        sleep 14400
    else
        # Use process substitution so bash only waits for the python process.
        # A plain `python | tee` pipeline hangs forever when a grandchild
        # (e.g. the smg router spawned via Popen without stdout redirection)
        # inherits and holds the pipe write end after python exits.
        set +e
        ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" > >(tee -a "${log_path}/${tc_name}.log") 2>&1
        case_exit=$?
        set -e
        if [ "${case_exit}" != "0" ];then
            OVERALL_SUCCESS=false
            echo "===== [$(date)] Case ${tc_name} FAILED (exit ${case_exit}) ====="
        else
            echo "===== [$(date)] Case ${tc_name} OK ====="
        fi
    fi
    echo "Finished test case ${test_case}"

    # per-case metrics: each case writes into its own subdirectory so concurrent
    # cases in the same batch do not overwrite each other's test_output.log.
    if [ -n "${METRICS_DATA_FILE}" ]; then
        case_metrics_dir="${METRICS_DATA_FILE}/${tc_name}"
        mkdir -p "${case_metrics_dir}"
        cp "${log_path}/${tc_name}.log" "${case_metrics_dir}/test_output.log"
        echo "Metrics log saved to ${case_metrics_dir}/test_output.log"
    fi

    # per-case plog backup into its own tc_name subdir.
    source_plog_path="/root/ascend/log/debug/plog"
    if [ -d "$source_plog_path" ];then
        echo "Plog files found. Begin to backup them for ${tc_name}."
        target_plog_path="/root/sglang/debug/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
        if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
            target_plog_path="/root/.cache/tests/logs/plog/${run_label}/${tc_name}/${HOSTNAME}"
        fi
        rm -rf "${target_plog_path}"
        mkdir -p "${target_plog_path}"
        cp ${source_plog_path}/* "${target_plog_path}" || true
        # Clear plog buffer so the next case starts fresh.
        rm -f ${source_plog_path}/* 2>/dev/null || true
    fi

    # Inter-case cleanup: kill lingering sglang/python test processes and wait
    # for NPU memory + ports to be released before launching the next case.
    # Skip after the last case (Volcano Job teardown will reclaim everything).
    if [ "${case_index}" -lt "${total_cases}" ];then
        echo "===== Cleaning up sglang processes before next case ====="
        pkill -9 -f "sglang" 2>/dev/null || true
        pkill -9 -f "python.*launch_server" 2>/dev/null || true
        pkill -9 -f "python.*test_npu" 2>/dev/null || true
        sleep 10

        # Wait for NPU memory to be released (max 120s).
        wait_npu_idle=0
        while [ $wait_npu_idle -lt 120 ]; do
            # npu-smi lists one summary block per card; "0NPU" markers indicate idle.
            # If no line shows real usage, consider NPU idle.
            npu_busy=$(npu-smi info 2>/dev/null | grep -E "^\|" | grep -v "0NPU" | grep -v "^$" | wc -l)
            if [ "${npu_busy}" -le 0 ];then
                echo "NPU memory all released."
                break
            fi
            echo "Waiting NPU memory release... (${wait_npu_idle}s, busy cards: ${npu_busy})"
            sleep 5
            wait_npu_idle=$((wait_npu_idle + 5))
        done
        if [ $wait_npu_idle -ge 120 ];then
            echo "WARNING: NPU memory not fully released after 120s, proceeding anyway."
        fi

        # Wait for common sglang ports to be released (max 30s per port).
        for port in 6677 8000 8995; do
            for i in $(seq 1 30); do
                if ! ss -tln 2>/dev/null | grep -q ":${port} "; then
                    break
                fi
                sleep 1
            done
        done
    fi
done

if [ "$OVERALL_SUCCESS" = "true" ];then
    echo "All ${total_cases} case(s) OK."
    exit 0
else
    echo "Some case(s) failed in batch."
    exit 1
fi
