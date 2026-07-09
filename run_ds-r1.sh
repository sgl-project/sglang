# Disable (GPU) core dumps: systemd-coredump is absent in this container, so
# ROCr falls back to writing a huge gpucore.<pid>.gpu (hundreds of GB) in cwd on
# any GPU fault, which fills the disk. RLIMIT_CORE=0 stops the file-based dump.
ulimit -c 0

#PYTHONPATH=/sgl-workspace/aiter_fbd6:/sgl-workspace/sglang/python:$PYTHONPATH \
HIP_VISIBLE_DEVICES=3 \
HSA_ENABLE_COREDUMP=0 \
HSA_COREDUMP_PATTERN=/dev/null \
AMD_COREDUMP=0 \
ENABLE_CK=0 \
SGLANG_USE_AITER=1 \
AITER_FORCE_A8W4=1 \
AITER_GROUPED_FORCE_SPLIT_K1=1 \
SGLANG_MOE_SHUFFLE_GFX1250=1 \
ROCM_QUICK_REDUCE_QUANTIZATION=NONE \
SGLANG_AITER_FP8_PREFILL_ATTN=0 \
SGLANG_AITER_MLA_PERSIST=0 \
SGLANG_INT4_WEIGHT=0 \
SGLANG_MOE_PADDING=1 \
SGLANG_SET_CPU_AFFINITY=1 \
SGLANG_ROCM_FUSED_DECODE_MLA=0 \
SGLANG_USE_ROCM700A=1 \
python3 -m sglang.launch_server \
  --model-path /dockerx/data/models/DeepSeek-R1-0528-MXFP4 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000 \
  --mem-fraction-static 0.90 \
  --chunked-prefill-size 16384 \
  --attention-backend triton \
  --max-running-requests 32 \
  --kv-cache-dtype auto \
  --page-size 64
  #--context-length 200000 \
