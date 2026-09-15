# real-acceptance additions for launch_v2.sh: source this, then run launch_v2.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export ENVS2="SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4 SGLANG_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_TIMEOUT_KEEP_ALIVE=3600"
# 64 kv-splits for the draft's dense attention; prefill/extend graph replay
export EXTRA2="--triton-attention-num-kv-splits 64 --cuda-graph-backend-prefill breakable"
export STEPS=3 DRAFT=4 MEMFRAC=0.85

# INT4 quick-reduce only reaches the >=64 MB prefill all-reduce messages, so decode is unaffected
export ENVS2="$ENVS2 ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
# aiter paged batch-prefill for large extends over a long cached prefix
export ENVS2="$ENVS2 SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1"

# short waiting extends share the chunk budget with a long in-flight prefill
export ENVS2="$ENVS2 SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5"

# quark-excluded bf16 linears run as online per-token FP8; gate and lm_head stay bf16
export ENVS2="$ENVS2 SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1 SGLANG_QUARK_ONLINE_FP8_SKIP_MODULES=gate,lm_head,index_qkv_proj SGLANG_FUSED_NORM_FP8_QUANT_MAX_M=16384"
# aiter merges the colon-separated GEMM config files, lowest us wins; its stock file comes from the installed package
AITER_PKG=$(aiter_pkg_dir)
export ENVS2="$ENVS2 AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=${AITER_PKG:+$AITER_PKG/configs/a8w8_bpreshuffle_tuned_gemm.csv:}$M3_ROOT/tuned_a8w8_bpreshuffle_m3_gfx950.csv"
