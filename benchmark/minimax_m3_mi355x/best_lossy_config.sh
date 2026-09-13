# Track J: ATOM-parity PERFORMANCE-ONLY config for MiniMax-M3 on 4x MI350X (TP4), MXFP4 quark target + EAGLE3.
# Source /scratch/run/best_config.sh first (this file does that), then run /scratch/run/launch_v2.sh.
#
# *** OUTPUTS ARE NOT THE MODEL'S REAL OUTPUT. ***  SGLANG_SIMULATE_ACC_LEN forces which draft tokens commit
# (mean acceptance length 2.78 = 1 + 0.5933 x 3, the same floor/ceil schedule ATOM's --spec-decode-acceptance-rate 0.5933
# resolves to with 3 draft tokens); draft and verify still run, but accepted tokens are never checked against the target.
# Never run GSM8K or any accuracy evaluation with this file sourced. Report numbers from it as "synthetic acceptance".
#
# Measured on GPUs 4-7 at ~195K-token contexts (steady.py 60 s): forced 2.78 with STEPS=2/DRAFT=3 -> N=16 2175 tok/s, N=24 2704 tok/s
# vs the real-acceptance best_config (STEPS=3/DRAFT=4, accept 2.3-2.4 on this workload) 1610 / 1990 tok/s (+35% at N=24).
# CHUNK=16384 (ATOM's prefill chunk): fresh 197K prefill 5.75 -> 5.5 s; MEMFRAC=0.9 (ATOM gpu-mem 0.9): KV pool 7.9M -> 8.9M tokens, no OOM.
# Rejected: PTPC-FP8 dense (SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED, no gain), --max-running-requests 64 (neutral; keep 48).
source /scratch/run/best_config.sh
export STEPS=2 DRAFT=3
export MEMFRAC=0.9 CHUNK=8192
export ENVS2="$ENVS2 SGLANG_SIMULATE_ACC_LEN=2.78 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token"
# Launch example (coordinator GPUs 0-3):
#   source /scratch/run/best_lossy_config.sh; TAG=v9_lossy GPUS=0,1,2,3 PORT=30000 SPEC_ATTN=decode \
#     EXTRA2="--max-running-requests 48 $EXTRA2" ENVS2="NCCL_MIN_NCHANNELS=112 HIP_FORCE_DEV_KERNARG=1 $ENVS2" bash /scratch/run/launch_v2.sh

# Track L: chunked-prefill fairness (waiting short extends share the chunk budget with a long in-flight prefill);
# c=24 900 s windows: TTFT p90 4.50->3.96 s, p99 12.7->10.3 s, throughput within noise; gsm8k-500 0.870 on the real-acceptance cfg.
export ENVS2="$ENVS2 SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5"

# Track N: fp4 activations for the decode MoE too (aiter FlyDSL path with fused SwiGLU; default keeps bf16 activations below 256 rows).
# Performance-only: GSM8K-500 0.834/0.834/0.820 vs 0.85-0.89 with bf16 activations. Steady decode ~195K: N=16 1604->1839, N=24 2016->2094 tok/s.
export ENVS2="$ENVS2 GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0"
