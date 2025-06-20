#!/bin/bash
set -x

# GPUS=${2:-8}
# GPUS_PER_NODE=${3:-8}
GPUS=${2:-1}
GPUS_PER_NODE=${3:-1}
PARTITION=${4:-"INTERN2"}
QUOTA_TYPE=${5:-"auto"}
JOB_NAME=${6:-"vl_sj"}

CPUS_PER_TASK=${CPUS_PER_TASK:-10}

if [ $GPUS -lt 8 ]; then
    NODES=1
else
    NODES=$((GPUS / GPUS_PER_NODE))
fi

SRUN_ARGS=${SRUN_ARGS:-" --jobid=5085910 "} # 3794248 3792227

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=${MASTER_PORT:-32436}
export LAUNCHER=slurm
export BATCH_SIZE

export LOGGING_GRAD_NORM
export DEBUG
export SKIP_LOG_IMAGE
export DEEPSPEED_TIMEOUT=60
export TEXT_NUM_IMAGE

SUFFIX=$(date '+%Y%m%d%H%M')
srun -p ${PARTITION} \
  --job-name=${JOB_NAME} \
  --gres=gpu:${GPUS_PER_NODE} \
  --nodes=${NODES} \
  --ntasks=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=${CPUS_PER_TASK} \
  --kill-on-bad-exit=1 \
  --quotatype=${QUOTA_TYPE} \
  ${SRUN_ARGS} \
  python my_scripts/sglang_offline2.py