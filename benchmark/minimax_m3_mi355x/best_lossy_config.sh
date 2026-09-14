# PERFORMANCE-ONLY: forced acceptance commits unchecked draft tokens; never evaluate accuracy on it
source /scratch/run/best_config.sh
# mean acceptance length 2.78 over 3 draft tokens is ATOM's --spec-decode-acceptance-rate 0.5933
export STEPS=2 DRAFT=3
export MEMFRAC=0.9 CHUNK=8192
export ENVS2="$ENVS2 SGLANG_SIMULATE_ACC_LEN=2.78 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token"

export ENVS2="$ENVS2 SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5"

# fp4 MoE activations at decode too (fused SwiGLU path); costs accuracy, so performance-only
export ENVS2="$ENVS2 GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0"
