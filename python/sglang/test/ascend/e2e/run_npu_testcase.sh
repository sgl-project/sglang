test_case=$1

sglang_source_path=/root/sglang
if [ ! -f "${sglang_source_path}/${test_case}" ];then
  echo "The test case file is not exist: $test_case"
  exit 0
fi

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
    cp "${TRANSFORMERS_PKG_PATH_SOURCE}/*" "${TRANSFORMERS_PKG_PATH_TARGET}/"
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

echo "Running test case ${test_case}"
tc_name=${test_case##*/}
tc_name=${tc_name%.*}
current_date=$(date +%Y%m%d)
log_path="/root/sglang/debug/logs/log/${current_date}/${tc_name}/${HOSTNAME}"
if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
    log_path="/root/.cache/tests/logs/log/${current_date}/${tc_name}/${HOSTNAME}"
fi
rm -rf "${log_path}"
mkdir -p "${log_path}"
echo "Log path: ${log_path}"

if [ "${TROUBLE_SHOTTING}" = "true" ] || [ "${TROUBLE_SHOTTING}" = "True" ];then
    echo "TROUBLE_SHOTTING=true, the pod will keep alive for four hour."
    ( ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" 2>&1 || true ) | tee -a "${log_path}/${tc_name}.log"
    sleep 14400
else
    ${PYTHON_FOR_SGLANG} -u "${sglang_source_path}/${test_case}" 2>&1 | tee -a "${log_path}/${tc_name}.log"
fi
echo "Finished test case ${test_case}"

source_plog_path="/root/ascend/log/debug/plog"
if [ -d "$source_plog_path" ];then
    echo "Plog files found. Begin to backup them."
    target_plog_path="/root/sglang/debug/logs/plog/${tc_name}/${HOSTNAME}"
    if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
        target_plog_path="/root/.cache/tests/logs/plog/${tc_name}/${HOSTNAME}"
    fi
    rm -rf "${target_plog_path}"
    mkdir -p "${target_plog_path}"
    cp ${source_plog_path}/* "${target_plog_path}"
fi
