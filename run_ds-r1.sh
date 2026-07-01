#PYTHONPATH=/sgl-workspace/aiter_fbd6:/sgl-workspace/sglang/python:$PYTHONPATH \
HIP_VISIBLE_DEVICES=3 \
ENABLE_CK=0 \
SGLANG_USE_AITER=1 \
ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
SGLANG_AITER_FP8_PREFILL_ATTN=0 \
SGLANG_AITER_MLA_PERSIST=0 \
SGLANG_INT4_WEIGHT=0 \
SGLANG_MOE_PADDING=1 \
SGLANG_SET_CPU_AFFINITY=1 \
SGLANG_ROCM_FUSED_DECODE_MLA=0 \
SGLANG_USE_ROCM700A=1 \
python3 -m sglang.launch_server \
  --model-path /dockerx/data/models/DeepSeek-R1-MXFP4 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 \
  --attention-backend triton \
  --max-running-requests 32 \
  --kv-cache-dtype fp8_e4m3 \
  --page-size 64
  #--context-length 200000 \
