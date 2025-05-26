#!/bin/bash

if [ -z "$REGION" ]; then
  # auto detect region
  DEFAULT_REGION=$(curl --connect-timeout 5 --max-time 10 -f -s 100.96.0.96/latest/region_id)
  if [ $? -ne 0 ]; then
    echo "auto detect region failed, failback use cn-beijing"
    DEFAULT_REGION="cn-beijing"
  else
    echo "auto detected region: ${DEFAULT_REGION}"
  fi
else
  DEFAULT_REGION=$REGION
fi
DEFAULT_BUCKET="iaas-public-model-$DEFAULT_REGION"


MODEL_PATH=${MODEL_PATH:-"/data/models"}
MODEL_NAME=${MODEL_NAME:-"DeepSeek-R1"}
MODEL_LENGTH=${MODEL_LENGTH:-131072}
TP=${TP:-8}
RANK0_ADDR=${RANK0_ADDR:-""}
RANKS=${RANKS:-0}
TOTAL_RANKS=${TOTAL_RANKS:-1}
PORT=${PORT:-8080}
CMD_ARGS=${CMD_ARGS:-""}
BUCKET=${BUCKET:-"$DEFAULT_BUCKET"}
REGION=${REGION:-"$DEFAULT_REGION"}

echo "REGION=$REGION"
echo "BUCKET=$BUCKET"

# check if MODE_PATH and MODEL_NAME are set
if [ -z "$MODEL_PATH" ] || [ -z "$MODEL_NAME" ]; then
  echo "MODE_PATH and MODEL_NAME must be set"
  exit 1
fi

# check if MODE_PATH not exists, create it
if [ ! -d "$MODEL_PATH" ]; then
  mkdir -p $MODEL_PATH
fi

# check if MODE_PATH/MODEL_NAME not exists, download it
if [ ! -d "$MODEL_PATH/$MODEL_NAME" ]; then
  cd $MODEL_PATH
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] downloading model $MODEL_NAME"
  BUCKET=$BUCKET REGION=$REGION oniond download model $MODEL_NAME --turbo
  if [ $? -ne 0 ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Failed to download model $MODEL_NAME"
    exit 1
  fi
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] model download success"
  cd -
fi

# check if speculative-algo is set to NEXTN
if [[ $CMD_ARGS =~ --speculative-algo[[:space:]]+NEXTN ]]; then
    echo "speculative-algo = NEXTN"
    # extra speculative-draft path
    if [[ $CMD_ARGS =~ --speculative-draft[[:space:]]+([^[:space:]]+) ]]; then
        draft_path="${BASH_REMATCH[1]}"
        echo "speculative-draft = $draft_path"
        # check if draft_path exsits
        if [ ! -d "$draft_path" ]; then
          draft_parent_path=$(dirname "$draft_path")
          draft_name=$(basename "$draft_path")
          cd $draft_parent_path
          BUCKET=$BUCKET REGION=$REGION oniond download model $draft_name --turbo
          if [ $? -ne 0 ]; then
            echo "Failed to download speculative model $draft_name"
            exit 1
          fi
          cd -
        fi
    fi
fi

# check if it is multiple ranks
if [ $TOTAL_RANKS -gt 1 ]; then
    # check if RANK0_ADDR is set
    if [ -z "$RANK0_ADDR" ]; then
        echo "RANK0_ADDR must be set"
        exit 1
    fi

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] starting sglang server"
    GLOO_SOCKET_IFNAME=eth0 NCCL_IB_HCA=mlx5_ NCCL_IB_DISABLE=0 NCCL_SOCKET_IFNAME=eth0 NCCL_IB_GID_INDEX=3 python3 -m sglang.launch_server --model-path $MODEL_PATH/$MODEL_NAME --tp $TP --dist-init-addr $RANK0_ADDR --nnodes $TOTAL_RANKS --node-rank $RANKS --trust-remote-code --host 0.0.0.0 --port $PORT $CMD_ARGS
else

    if [ $TP -gt 8 ]; then
        TP=8
    fi

    echo "[$(date +'%Y-%m-%d %H:%M:%S')] starting sglang server"
    python3 -m sglang.launch_server --model-path $MODEL_PATH/$MODEL_NAME --context-length $MODEL_LENGTH --tp $TP --trust-remote-code --host 0.0.0.0 --port $PORT --mem-fraction-static 0.9 --disable-radix-cache $CMD_ARGS
fi
