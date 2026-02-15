test_case=$1

sglang_source_path=/root/sglang
cd ${sglang_source_path}
if [ ! -f "${test_case}" ];then
  echo "The test case file is not exist: $test_case"
  exit 0
fi

echo "NPU info:"
npu-smi info

# set dns
cp /etc/resolv.conf /etc/resolv.conf_bak
echo -e "nameserver 223.5.5.5\nnameserver 223.6.6.6" > /etc/resolv.conf
cat /etc/resolv.conf_bak >> /etc/resolv.conf
echo "DNS info:"
cat /etc/resolv.conf

pip config set global.index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn"

pip3 install kubernetes

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
ASCEND_RT_VISIBLE_DEVICES=$(echo $ASCEND_VISIBLE_DEVICES | tr ',' '\n' | sort -n | tr '\n' ',')
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES%,}
echo "ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES"
export ASCEND_VISIBLE_DEVICES=""
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"

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
    sglang_pkg_path=$(pip show sglang | grep Location | awk '{print $2}')
    ascend_test_util_path=${sglang_pkg_path}/sglang/test/ascend
    mkdir -p ${ascend_test_util_path}
    mv ${ascend_test_util_path} ${ascend_test_util_path}_bak
    cp -r ${sglang_source_path}/python/sglang/test/ascend ${ascend_test_util_path}
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
rm -rf ${log_path}
mkdir -p ${log_path}
echo "Log path: ${log_path}"
python3 -u ${test_case} 2>&1 | tee -a ${log_path}/${tc_name}.log
echo "Finished test case ${test_case}"

source_plog_path="/root/ascend/log/debug/plog"
if [ -d "$source_plog_path" ];then
    echo "Plog files found. Begin to backup them."
    target_plog_path="/root/sglang/debug/logs/plog/${tc_name}/${HOSTNAME}"
    if [ "${SGLANG_IS_IN_CI}" = "true" ] || [ "${SGLANG_IS_IN_CI}" = "True" ];then
        target_plog_path="/root/.cache/tests/logs/plog/${tc_name}/${HOSTNAME}"
    fi
    rm -rf ${target_plog_path}
    mkdir -p ${target_plog_path}
    cp ${source_plog_path}/* ${target_plog_path}
fi

