# MiniMax-M3 on 4x MI350X: SGLang AgentX results and how to reproduce them

Benchmark: AIPerf `inferencex-agentx-mvp` (393 SemiAnalysis coding-agent traces, seed 42), SemiAnalysis AIPerf fork at InferenceX's pinned commit 754356e9 with InferenceX/ATOM's client flags (idle cap 300 s, trajectory start 0.25-0.75, 10 warmup requests per lane), TP4 on 4 GPUs. Score = total token throughput (prompt tokens incl. cache hits + completion) per second / 4. Hardware here is MI350X (gfx950, 288 GB, ROCm 7.2.4); ATOM's published figures are MI355X.

## Results (tok/s per GPU; 3600 s runs except the 1800 s real c=1/8 and forced c=1 points)

| c | SGLang real acceptance | SGLang forced acceptance (ATOM parity) | ATOM published (forced) |
|---:|---:|---:|---:|
| 1 | 3,565 | 4,194 | 4,845 |
| 8 | 11,021 | 13,029 | 14,047 |
| 24 | 34,785 | 38,457 | 39,680 |
| 32 | 39,083 | 43,956 | 42,476 |

SGLang numbers are the in-window rate (tokens of the profiling records over the profiling span); AIPerf's reported value is lower when cancelled requests stall its drain (e.g. 35,935 and 40,598 at c=32, 12,961 at forced c=8). ATOM's recipe runs every concurrency for 3600 s and the low-concurrency score is trace-sample bound, so only 3600 s points are comparable there. Quality of the real-acceptance config: GSM8K-1000 0.854-0.863, needles coherent to 257K tokens. The forced-acceptance config commits unchecked draft tokens (ATOM's `--spec-decode-acceptance-rate 0.5933`); its outputs are not the model's. Chart: `agentx_sglang_vs_atom.png`.

## Reproduce

Inside the sglang ROCm 7.2.4 container for gfx950 (`docker/rocm.Dockerfile`, `GPU_ARCH=gfx950-rocm724`; Ubuntu 24.04, Python 3.12, torch 2.11.0+rocm7.2) on a node with 4 free GPUs:

```bash
git clone -b M3-perf https://github.com/kevin-mii/sglang && cd sglang/benchmark/minimax_m3_mi355x
bash reproduce.sh real      # recommended config: c=1 8 24 32, 3600 s each; summary per point
bash reproduce.sh lossy     # ATOM-parity performance-only config (forced acceptance); outputs are not the model's
```

The first run installs sglang from the checkout, aiter 4ad99832 with the FlyDSL swizzle fix, the tuned MoE rows, both models and the SemiAnalysis AIPerf fork (754356e9) under `M3_WORK` (default `/scratch` if writable, else `~/m3-agentx`; ~300 GB). Each point prints AIPerf's `total_token_throughput` per GPU, the in-window rate, TTFT/ITL and p90 interactivity; raw artifacts land in `$M3_WORK/results/aiperf_<mode>_c<N>/`. `GPUS`, `PORT`, `DURATION` and concurrencies as arguments override the defaults. The script's `serve` function holds the full `sglang.launch_server` command and every env var of both configs; `bench` holds ATOM's client environment and flags. Quality gate for the real config: `python -m sglang.test.few_shot_gsm8k --port 30000 --num-questions 1000 --num-shots 5 --parallel 48` (expect 0.85-0.87).

---

# MiniMax-M3 on 4x MI355X/MI350X: every optimization ported for ATOM parity

SGLang `M3-perf` (`github.com/kevin-mii/sglang`), TP4, MXFP4 quark checkpoint, fp8 KV, EAGLE3 draft. Metric: AIPerf `inferencex-agentx-mvp` total token throughput per GPU, SemiAnalysis client, ATOM's flags, 3600 s.

| | c=24 | c=32 |
|---|---:|---:|
| Baseline `main` (start of work) | 6,815 | 10,019 |
| ATOM published (forced acceptance) | 39,680 | 42,476 |
| SGLang, real acceptance (GSM8K-1000 0.854-0.863) | 34,720 | 35,935 reported / 39,229 in-window |
| SGLang, ATOM-parity forced acceptance (performance-only) | 38,196 | 40,598 reported / 44,320 in-window |

## 1. Upstream PRs cherry-picked into M3-perf (by zcnrex; open unless noted)

| PR | Feature | Why it mattered |
|---|---|---|
| 36546 | Gluon paged-attention sparse prefill (AITER `pa_decode_gluon`) | sparse-attention prefill step 441 -> 173 ms once the fp8-KV fallback was fixed (our follow-up) |
| 36549 | fp8 lightning-indexer K cache on ROCm | halves index-cache traffic; needed for the fused KV/index store |
| 36559 | MoE small-batch sorting with fused mxfp8 quant (gfx950 aiter small-sort / SwiGLU path) | **load-bearing**: without it main produces garbage for M3 MXFP4 on gfx950 |
| 36560 | wave64 histogram-select decode top-k, higher kMaxNumBlocks for graphs | decode top-k at 1M context under CUDA graphs |
| 36574 | MXFP8 dense-only block convert, `torch._scaled_mm` 1x32 path, bf16 fp8-gemm backend | MXFP8 dense projections on gfx950 |
| 36575 | fused add-RMSNorm + per-token fp8 quant, deferred MoE all-reduce | fewer launches per layer (fusion flag itself measured neutral, see below) |
| 36576 | shared-experts fusion on ROCm gfx942+ | +33-45% decode once the double-append bug was fixed (our follow-up) |
| 36527 (merged) | index top-k shared across layers, decode top-k buffer reuse | basis of `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4` (ATOM's index_topk_freq 4): step 37.5 -> ~33 ms at 24 x 195K |
| 36557 (merged) | Triton split-K router GEMV | router at 128-row verify shapes (our follow-up extends it) |

## 2. Our branches (cleaned up per the pr-cleanup skill, pushed to `kevin-mii/sglang`, no PRs opened)

Follow-ups that stack on the open upstream PRs:

| Branch | Base PR | Feature | Effect |
|---|---|---|---|
| `pr/36576-followup-shared-expert-slot` | 36576 | fused shared expert was appended twice and one routed expert dropped on aiter paths (rows `[3 routed, 128, 128]`); Triton gate now fills the slot itself | GSM8K-500 0.81 -> 0.896; MXFP4 output no longer garbage on multi-chunk prompts; -57 launches/step |
| `pr/36546-followup-gluon-fp8-and-scratch` | 36546 | Gluon prefill on fp8 K/V pools (PR silently fell back to Triton); page layout built on device (no pageable H2D sync); dtype-aware configurable scratch cap; score-only index kernel split over KV blocks (M3 has 1 index head per rank) | sparse prefill 441 -> 173 ms; index scoring for a turn's new tokens 2.3 -> 0.08 ms/layer at 198K |
| `pr/36574-followup-mxfp8-aiter-backend` | 36574 | `--moe-runner-backend aiter` no longer overridden under the gfx95 mxfp8 override (FlyDSL fp8/fp4 fused MoE); quant kernel emits row-aligned scaled_mm operands | 257K prefill 18.5 -> 14.7 s; -4 launches per dense GEMM (-770/step) |
| `pr/36575-followup-fp8-linear-consumes-fused-quant` | 36575 | per-token fp8 linear consumes the fused add-RMSNorm's pre-quantized activation (version-guarded), `SGLANG_FUSED_NORM_FP8_QUANT_MAX_M` | removes the separate activation quant for qkv/indexer/MLP inputs |

Standalone branches on `origin/main`:

| Branch | Feature | Effect |
|---|---|---|
| `pr/quark-packed-shard-exclude-fix` | quark `should_ignore_layer` ignores packed shards absent from the exclude list (`index_v_proj` not in the checkpoint) | the MXFP4 checkpoint loads at all on current main |
| `pr/m3-eagle3-chain-verify` | EAGLE chain target-verify for the MiniMax sparse attention on CUDA/ROCm (was NPU-only); cross-layer top-k reuse in verify; packed per-request index scoring (with the local-block fix found in review); small extends over long cached prefixes served by the decode-style kernels | EAGLE3 usable on AMD: 1 stream 57 -> 84 tok/s at 70K (MXFP8); turn-restart extend at 195K 90 -> 67 ms |
| `pr/triton-verify-shared-kv-long-context` | split-KV / grouped-head verify kernels for the EAGLE draft extend and M3 dense layers; head_dim-128 long-context split config; grouped-head decode via the shared-KV kernel; small constant-length dense extends routed to the verify kernels | 24 x 195K verify 2.02 -> 0.38 ms/call; draft extend 2.6 -> 0.12 ms at 100K; draft dense attention 2 x 1.08 -> 6 x 0.34 ms/step |
| `pr/triton-extend-long-prefix` | split-prefix extend attention for long cached prefixes (`SGLANG_ENABLE_TRITON_EXTEND_LONG_PREFIX`); aiter CK paged batch-prefill for extends >= 2048 rows (`SGLANG_USE_AITER_EXTEND_LONG_PREFIX`); AMD hook keeps an explicit `--triton-attention-num-kv-splits` | dense attention per 8192-token chunk at 198K 41 -> 16 ms; 197K fresh prefill 6.2 -> 5.8 s; draft dense decode 2.9 -> 2.3 ms/step with 64 splits |
| `pr/spec-page-table-parallel-copies` | token-block-parallel page-table copies in the Triton draft backend and EAGLE verify/draft-extend metadata | 419 -> 30 us and 101 -> 13 us per call at 24 x 200K |
| `pr/m3-fused-kv-index-store` | fused scale+cast+scatter store for fp8 main and index KV caches | 7-8 launches per layer -> 1 (-360/step) |
| `pr/m3-sparse-attention-eager-break-bcg` | M3 sparse attention runs as an eager break under `--cuda-graph-backend-prefill breakable` (it was captured into the graph: garbage and faults on replay) | prefill graph replay works: extend forward at 185K 75 -> 29 ms (20 tok), 118 -> 38 ms (409 tok); steady decode at 24 streams +8% |
| `pr/scheduler-chunked-prefill-fairness` | `SGLANG_CHUNKED_PREFILL_FAIRNESS_RESERVE`: waiting short extends ride along with an in-flight chunked prefill | TTFT p99 -19% at c=24, throughput neutral |
| `pr/quark-online-fp8-excluded-layers` | opt-in load-time per-channel FP8 for quark-excluded bf16 linear layers (ATOM's PTPC-FP8 attention/dense), with tuned aiter a8w8 rows for the M3 shapes (aiter-side data) | GSM8K-1000 0.863 vs 0.854 bf16; decode GEMMs 1.4-1.7x faster, prefill GEMMs 1.5-2x; whole-run effect within noise (~1% of the step) |
| `pr/router-gemv-128-rows` | `router_gemv` covers 65-128-row batches (EAGLE verify shapes) | router at 96 rows stays on the Triton kernel |
| `pr/openai-chat-log-rejected-stream` | log the validation error (with request id) when a streaming chat request is rejected before its first chunk | diagnostics for client-side connection resets |

Also on `M3-perf` only (not upstreamed): `benchmark/minimax_m3_mi355x/` launch scripts, AIPerf wrappers, both configs, tuned CSVs, `M3_MI350X_STATUS.md`, `PR_INDEX.md`.

## 3. Configuration knobs adopted from ATOM's recipe (no code, in `best_config.sh` / `best_lossy_config.sh`)

| Knob | ATOM | Ours | Effect / note |
|---|---|---|---|
| Quantization | MXFP4 (quark), fp8 KV, fp8 index cache | same | MXFP4 decode 83.5 vs 60 tok/s MXFP8 at 1 stream |
| Speculative | EAGLE3, 3 draft tokens | EAGLE3, 3 steps / 4 tokens (real, accept 2.7-2.9); 2 steps / 3 tokens with forced 2.78 for parity | 4 tokens real: +0-4% over 3; forced 2.78: +35% steady decode |
| `--spec-decode-acceptance-rate 0.5933` | forced acceptance (performance-only) | `SGLANG_SIMULATE_ACC_LEN=2.78 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token` | same floor/ceil schedule; outputs are not the model's |
| Quick-reduce INT4 | yes | `ROCM_QUICK_REDUCE_QUANTIZATION=INT4` (applies to >= 64 MB messages, i.e. prefill only) | 197K fresh prefill 7.2 -> 6.1 s; GSM8K 0.886 |
| Custom all-reduce | yes | `SGLANG_M3_ALLOW_CUSTOM_AR=1` (aiter 2-stage; fastest at decode sizes, 17 us vs NCCL 48) | +37% at 1 stream, +7-10% at 8 |
| index_topk_freq 4 | yes | `SGLANG_MINIMAX_M3_INDEX_TOPK_FREQ=4` | 57 -> 15 full-context index scorings per step; GSM8K unchanged |
| gpu-mem 0.9, max-num-seqs 2 x conc | yes | `MEMFRAC=0.9` (lossy config), `--max-running-requests 48` | more KV; 64 running was neutral |
| prefill chunk 16384 | yes | 8192 kept | 16384 neutral per token under prefill graphs and doubles the decode stall per chunk |
| PTPC-FP8 attention/dense | yes | `SGLANG_QUARK_USE_ONLINE_FP8_FOR_EXCLUDED=1 SGLANG_USE_AITER_FP8_PER_TOKEN=1` + tuned rows (branch above) | quality-neutral, speed within noise |
| fp4 MoE activations at decode | (aiter default at prefill only) | `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=0`, lossy config only | +4-15% steady decode, GSM8K -3 pts |
| Prefill graphs | ATOM extends run eager | `--cuda-graph-backend-prefill breakable` (with the eager-break branch) | see above |
| HTTP keep-alive | n/a | `SGLANG_TIMEOUT_KEEP_ALIVE=3600` | stops the client connection resets that aborted three warmups |

## 4. Tested and rejected (numbers in `M3_MI350X_STATUS.md`)

`--enable-aiter-allreduce-fusion` (neutral, +3.4 ms memset/step); quick-reduce below 64 MB (custom AR takes those messages); chunk 16384; `--max-running-requests 64`; every scheduler knob (lpm, conservativeness, prefill-decode interval, max-prefill-tokens; all within +-2%); draft-model FP8 (neutral); expert parallelism (EP4 -15..-20%, fp4 atomic stage-2 broken in aiter); TP2 x DP2 with cache-aware routing (-40%); no-copy Gluon page layout (~1% of prefill for days of work); aiter MoE glue fusion (needs an upstream aiter kernel change, ~0.76 ms/step).

## 5. What ATOM still does that we do not

Mixed prefill and decode in one forward under speculative decoding (vLLM-V1 scheduler; SGLang disables mixed chunk with spec decode, so every prefill chunk stalls all decode streams: the largest remaining structural gap), block-128 native KV layout for the sparse attention, and MI355X clocks. Remaining fixed costs of the 26.5 ms verify step at 24 streams / 195K: MoE weight read 8.6 ms at 80% of HBM bandwidth, custom all-reduce 2.6 ms (count fixed by TP4).
