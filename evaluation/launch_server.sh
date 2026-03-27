export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export OPTFLAG="w8a8_gemm,moe"
python -m sglang.launch_server \
    --model-path /models/Qwen3.5-397B-A17B \
    --port 9000 \
    --tp-size 8 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --attention-backend triton \
    --disable-radix-cache \
    --cuda-graph-max-bs 64
