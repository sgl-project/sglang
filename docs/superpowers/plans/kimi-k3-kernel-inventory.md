# Kimi-K3 Standalone Kernel Inventory

Source revision: `578edb240a6d6f6f2fa4c31497276955d7f73432`

Base revision: `f4e0ac382e4e5d644f2fbe4a15c20da53500bbca`

## Generic JIT

| Family | Implementation | Direct test |
|---|---|---|
| JIT compilation | `kernels/jit/utils/compile.py` | exercised by all CUDA JIT tests |
| distributed JIT communicator | `kernels/jit/include/sgl_kernel/distributed/communicator.cuh` | K3 distributed tests |
| math and warp helpers | `kernels/jit/include/sgl_kernel/{math,warp}.cuh` | compilation plus consuming-kernel tests |

## Generic Attention, MLA, and VLM

| Family | Implementation | Direct test |
|---|---|---|
| MLA KV concatenate + Q | `ops/attention/set_mla_kv_concat_q.py` and two CUDA headers | `registered/jit/test_set_mla_kv_concat_q*.py` |
| MLA concatenate/cache changes | `ops/attention/concat_mla.py` and CUDA headers | existing concatenate tests plus JIT tests |
| vision RoPE | `ops/attention/vision_rope.py` | `registered/kernels/ops/test_vision_rope.py` |
| image preprocessing | `ops/mm/process/image.py` | direct processor-kernel test to add if retained |
| split-KV and FA4 support edits | existing attention wrappers | existing attention tests |

## Generic Elementwise, GEMM, MoE, and Sampling

| Family | Implementation | Direct test |
|---|---|---|
| add3 | `ops/elementwise/add3.py` | `registered/kernels/ops/test_add3.py` |
| tiny BF16 GEMM | `ops/gemm/tiny_gemm.py` | `registered/kernels/ops/test_tiny_gemm.py` |
| single-token MoE align | `ops/moe/moe_align_single_token.py` | `test_moe_auxiliary.py` |
| MoE front | `ops/moe/moe_front.py` | `test_moe_front.py` |
| route radix | `ops/moe/moe_route_radix.py` | `test_moe_route_radix.py` |
| fused route + quant | `ops/moe/moe_route_quant_fused.py` | `test_moe_route_quant_fused.py` |
| MoE top-k sum | `ops/moe/moe_topk_sum.py` | `test_moe_auxiliary.py` |
| TRT-LLM-generated MoE loader | `ops/moe/trtllm_gen_moe.py` | `ops/kimi_k3/test_trtllm_gen_moe.py` |
| top-p renormalization fallback | `ops/sampling/top_p_renorm_triton.py` | `test_top_p_renorm_triton.py` |

## KDA Decode, Prefill, and State

| Family | Implementation | Direct test |
|---|---|---|
| fused KDA decode | `ops/attention/kda_fused_decode.py` + CUDA header | KDA state/strided tests |
| packed KDA decode | `ops/attention/kda_packed_decode.py` + CUDA header | KDA decode parity test to retain/add |
| ReplaySSM speculative state | `ops/attention/fla/kda_replayssm_spec_decode.py` | five ReplaySSM kernel tests |
| CuTeDSL prefill | `ops/attention/linear/kda_nvidia_prefill/` | `unit/layers/attention/linear/kernels/test_kda_nvidia.py` |
| PTX prefill | `ops/attention/linear/kda_ptx_prefill/` + `kda_prefill.cu` | KDA prefill parity test |
| KDA MTP decode | `ops/kimi_k3/kda_decode_mtp.py` | ReplaySSM MTP and production-shape parity tests |
| state scatter/cache indices | existing Mamba/KV wrappers | state-stride and cache-index tests |

## Kimi-K3 Fused Compute

| Family | Implementation | Direct test |
|---|---|---|
| SiTU activation | `ops/kimi_k3/activation.py` | `test_situ_mul_quant.py` |
| masked SiTU + quant | `ops/kimi_k3/moe.py` | `test_situ_mul_quant.py` |
| MLA output gate | `ops/kimi_k3/mla_output_gate.py` | `test_mla_output_gate.py` |
| attention residual TMA | `ops/kimi_k3/attn_res.py` | `test_attn_res_fused_tma.py` |
| attention residual stream aggregation | `ops/kimi_k3/attn_res.py` | `test_attn_res_aggregate_stream.py` |
| K3 tiny-GEMM dispatch | `ops/kimi_k3/__init__.py` | tiny-GEMM production-shape cases |

## Kimi-K3 Distributed

| Family | Implementation | Direct test |
|---|---|---|
| fused all-reduce | `ops/kimi_k3/all_reduce.py` | `test_ar_fusion.py` |
| GEMM + all-gather | `ops/kimi_k3/gemm_ag.py` | `test_gemm_ag.py` |
| GEMM + all-reduce | `ops/kimi_k3/gemm_ar.py` | direct GEMM-AR parity test to add |
| SP collectives | `ops/kimi_k3/sp_collective.py` | direct SP collective parity test to add |
| persistent symmetric buffers | generic custom-all-reduce-v2 support | direct communicator/buffer test to add |

## Excluded Runtime Integration

The port excludes Kimi model/configuration files, attention backend wiring,
model-runner hooks, scheduler and cache changes, speculative-decoding
orchestration, disaggregated serving, parsers, HTTP/OpenAI serving code, and
model-level end-to-end tests.

## Excluded Development Artifacts

The following source-branch artifacts remain excluded:

- `benchmark/bench_linear_attention/bench_kda_fold_batched.py`
- `benchmark/bench_linear_attention/bench_kda_verify_ringwrite.py`
- `benchmark/hicache/bench_hicache_rw_cycles.py`
- `benchmark/kernels/kimi_k3/bench_sp_attn_res.py`
- `benchmark/kernels/kimi_k3/bench_sp_collective.py`
- all Kimi-K3 files under `test/registered/**/benchmark/`
- the runtime-coupled `test_symm_buffers.py`
