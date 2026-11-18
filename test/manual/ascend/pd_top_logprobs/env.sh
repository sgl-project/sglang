#!/usr/bin/env bash
# Source the installed Ascend environment before launching either PD worker.
set -eo pipefail
ascend_env_script=${ASCEND_ENV_SCRIPT:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
if [[ -f "$ascend_env_script" ]]; then
    source "$ascend_env_script"
fi
for env_file in \
    /usr/local/Ascend/nnal/atb/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash \
    /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/custom_transformer/bin/set_env.bash; do
    if [[ -f "$env_file" ]]; then
        source "$env_file"
    fi
done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export STREAMS_PER_DEVICE=${STREAMS_PER_DEVICE:-32}
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-128}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export ASCEND_MF_STORE_URL=${ASCEND_MF_STORE_URL:-tcp://127.0.0.1:27730}
export ASCEND_MF_TRANSFER_PROTOCOL=${ASCEND_MF_TRANSFER_PROTOCOL:-sdma}
