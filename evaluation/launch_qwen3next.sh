export HIP_VISIBLE_DEVICES=0,1,2,3

export SGLANG_USE_AITER=1
export SGLANG_ROCM_USE_AITER_LINEAR_SHUFFLE=1 
export HIPBLASLT_TUNING_OVERRIDE_FILE=/mnt/md0/yixiongh/Qwen3_next/unittest/hipblaslt_tuning/file/output/tuning_result.csv


pkill -9 python3
rm -rf gpucore.*

TP=4
EP=1
 
echo "launching ${model}"
echo "TP=${TP}"
echo "EP=${EP}"
 
python3 -m sglang.launch_server \
    --model-path /data/models/Qwen/Qwen3-Next-80B-A3B-Instruct \
    --host localhost \
    --port 8080 \
    --tp-size ${TP} \
    --ep-size ${EP} \
    --trust-remote-code \
    --chunked-prefill-size 32768 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --max-prefill-tokens 32768 \
    --cuda-graph-max-bs 256 \
    --page-size 64 \
    --max-running-requests 128 \
    --attention-backend triton

