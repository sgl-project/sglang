#!/usr/bin/env bash
# Same-host two-NPU test, with one worker per idle device.
#   PD_PREFILL_NPU=0 bash launch_worker.sh prefill /path/to/Qwen3-0.6B
#   PD_DECODE_NPU=1 bash launch_worker.sh decode /path/to/Qwen3-0.6B
# Then run launch_router.sh and client.py in separate terminals.
# Set PD_TOP_LOGPROBS_OUTPUT_DIR and client.py --output-dir to the same directory.
set -eo pipefail
if [[ $# -lt 2 || ( "$1" != prefill && "$1" != decode ) ]]; then
    echo "Usage: $0 {prefill|decode} MODEL_PATH [extra server arguments...]" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
source "$script_dir/env.sh"
export PYTHONPATH="$repo_root/python:${PYTHONPATH:-}"
role=$1
model_path=$2
shift 2
if [[ "$role" == prefill ]]; then
    physical_device=${PD_PREFILL_NPU:-0}
    http_port=${PD_PREFILL_PORT:-17710}
    encoder_port=${PD_PREFILL_ENCODER_PORT:-17711}
    dist_port=${PD_PREFILL_DIST_PORT:-17712}
    init_port=${PD_PREFILL_INIT_PORT:-27710}
else
    physical_device=${PD_DECODE_NPU:-1}
    http_port=${PD_DECODE_PORT:-17720}
    encoder_port=${PD_DECODE_ENCODER_PORT:-17721}
    dist_port=${PD_DECODE_DIST_PORT:-17722}
    init_port=${PD_DECODE_INIT_PORT:-27720}
fi
export ASCEND_RT_VISIBLE_DEVICES=$physical_device
export ASCEND_VISIBLE_DEVICES=$physical_device
export ASCEND_DEVICES=$physical_device
exec python3 "$script_dir/serve.py" \
    --model-path "$model_path" --served-model-name pd-top-logprobs --trust-remote-code \
    --device npu --base-gpu-id 0 --tp-size 1 \
    --attention-backend ascend --sampling-backend pytorch \
    --dtype bfloat16 --kv-cache-dtype bfloat16 --mem-fraction-static 0.20 \
    --context-length 2048 --max-total-tokens 4096 --max-running-requests 8 \
    --chunked-prefill-size 512 --max-prefill-tokens 2048 --page-size 128 \
    --host 127.0.0.1 --port "$http_port" --encoder-bootstrap-port "$encoder_port" \
    --nccl-port "$dist_port" --dist-init-addr "127.0.0.1:$init_port" \
    --disaggregation-mode "$role" --disaggregation-transfer-backend ascend \
    --disaggregation-bootstrap-port "${PD_BOOTSTRAP_PORT:-17730}" \
    --disaggregation-max-top-logprobs-num 256 --num-reserved-decode-tokens 64 \
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
    --enable-metrics --return-output-ids --random-seed 1 --watchdog-timeout 180 \
    "$@"
