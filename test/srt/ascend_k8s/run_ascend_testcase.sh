test_case=$1
cd $SGLANG_SOURCE_PATH
if [ ! -f "${test_case}" ];then
  echo "The test case file is not exist: $test_case"
  exit 0
fi

# speed up by using infra cache services
CACHING_URL="cache-service.nginx-pypi-cache.svc.cluster.local"
sed -Ei "s@(ports|archive).ubuntu.com@${CACHING_URL}:8081@g" /etc/apt/sources.list
pip config set global.index-url http://${CACHING_URL}/pypi/simple
pip config set global.extra-index-url "https://pypi.tuna.tsinghua.edu.cn/simple"
pip config set global.trusted-host "${CACHING_URL} pypi.tuna.tsinghua.edu.cn"

pip install kubernetes
pip3 install xgrammar==0.1.25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# copy or download required file
cp /data/ascend-ci-share-pkking-sglang/huggingface/hub/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json /tmp
curl -o /tmp/test.jsonl -L https://gh-proxy.test.osinfra.cn/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

export PYTHONPATH=$SGLANG_SOURCE_PATH/python:$PYTHONPATH

echo "Running test case ${test_case}"
python3 -u ${test_case}
echo "Finished test case ${test_case}"
