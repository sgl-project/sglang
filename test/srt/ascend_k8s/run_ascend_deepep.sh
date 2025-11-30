test_case=$1
if [ ! -f "${test_case}" ];then
  echo "The test case file is not exist!"
  exit 0
fi

pip install kubernetes
pip3 install xgrammar==0.1.25
source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_goveror
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0

export PYTHONPATH=/data/.cache/sglang/python:$PYTHONPATH

echo "Running test case ${test_case}"
python3 -u ${test_case}
echo "Finished test case ${test_case}"
