# Track B best-known additions for /scratch/run/launch_v2.sh (source this, then run launch_v2.sh).
# Code: M3-perf @ 65958e1554 or later (0e7ce8a7af padded mxfp8 quant, 3aae177df5 fused fp8 KV store,
#       09abe2981d split-KV/shared-KV draft extend + dense verify, 0760f29389 eager verify meta,
#       011ea829fd shared-KV verify kernel long-context config, 3f5c2b3bc8 small-extend decode-style path, 65958e1554 kv-splits hook).
# Measured at ~195K-token contexts (closed-loop steady state, GPUs 4-7): N=24 1226 -> 1595 tok/s, N=16 1173 -> 1417 tok/s
# vs the same code without ENVS2 (kernel fix + TOPK_FREQ=4 together; bs=24 step 37.5 -> 29.4 ms).
# Needle (thinking off) coherent at 5K / 79K / 132K prompt tokens.
export ENVS2="SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4 SGLANG_TRITON_EXTEND_LONG_PREFIX=1 SGLANG_TIMEOUT_KEEP_ALIVE=3600"
# --triton-attention-num-kv-splits 64: needs 65958e1554 (the AMD hook used to pin 16); draft dense decode attention 2.9 -> 2.3 ms/step, steady N=24 @195K 1675 -> 1752 tok/s.
# --speculative-draft-model-quantization unquant (bf16 draft) is what Track B measured; v2 default also works.
# --cuda-graph-backend-prefill breakable: prefill/extend graph replay (needs e0da9ebec1: M3 sparse attention as an eager break).
#   Extend forward at ~185K prefix: 20 tok 75->29 ms, 409 tok 118->38, 1231 tok 92->65; steady N=24 @195K 1734->1880 tok/s; GSM8K-500 0.854.
export EXTRA2="--triton-attention-num-kv-splits 64 --cuda-graph-backend-prefill breakable"
export STEPS=3 DRAFT=4 MEMFRAC=0.85

# SGLANG_TIMEOUT_KEEP_ALIVE=3600: two c=32 AIPerf warmups aborted on a single 'Connection reset by peer' (uvicorn default keep-alive 5 s vs aiohttp pooled connections).

# Track G: INT4 quick-reduce for the >=64 MB prefill all-reduce messages (198K fresh prefill 36.2 -> 31.1 us/token; gsm8k-500 0.886, needles 20K/200K and cached-restart parity OK; decode unaffected).
export ENVS2="$ENVS2 ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
export ENVS2="$ENVS2 SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1"
# SGLANG_USE_AITER_EXTEND_LONG_PREFIX=1 (Track H): aiter CK paged batch-prefill for >=2048-row extends over long prefixes; fresh 197K prefill 6.2 -> 5.8 s; gsm8k-500 0.864.

# Track L: chunked-prefill fairness (waiting short extends share the chunk budget with a long in-flight prefill);
# c=24 900 s windows: TTFT p90 4.50->3.96 s, p99 12.7->10.3 s, throughput within noise; gsm8k-500 0.870 on the real-acceptance cfg.
export ENVS2="$ENVS2 SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE=0.5"

# Track P: ATOM-style online PTPC-FP8 for the quark-excluded bf16 linears (attention q/k/v/o, index projections, 3 dense MLPs;
# router gate + lm_head stay bf16) with tuned aiter a8w8_bpreshuffle rows for the M3 shapes and the fused add-RMSNorm fp8
# emission consumed by the following GEMM (51e758ee57). Quality: GSM8K-1000 0.863 (bf16 dense: 0.854), GSM8K-500 0.862/0.864.
# Speed: fresh 197K prefill 5.62-5.73 s (bf16 dense 5.7-5.8), steady decode unchanged (dense GEMMs are ~1% of the verify step).
# Tuned rows: /scratch/run/tuned_a8w8_bpreshuffle_m3_gfx950.csv (repo copy: benchmark/minimax_m3_mi355x/); aiter merges the
# colon-separated files, lowest us wins. Draft-model fp8 (--speculative-draft-model-quantization fp8) was neutral: not enabled.
export ENVS2="$ENVS2 SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1 SGLANG_QUARK_ONLINE_FP8_SKIP_MODULES=gate,lm_head SGLANG_FUSED_NORM_FP8_QUANT_MAX_M=16384"
export ENVS2="$ENVS2 AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE=/sgl-workspace/aiter/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv:/scratch/run/tuned_a8w8_bpreshuffle_m3_gfx950.csv"
