# Cake kernels

## Purpose and layering

This directory owns thin adapters to Cake-generated kernels distributed by FlashInfer.
Generated CUDA sources and JIT caches remain owned by FlashInfer. Each adapter module
here exposes `supports_<name>(...) -> bool` plus a forwarder; the matching `KernelSpec`
registrations and the explicit `cake_<name>` entry points live in
`sglang/kernels/ops/<group>/cake.py`, which the group facade imports after its
`__all__` (metadata only, see `python/sglang/kernels/README.md`). Runtime callers
import the facade from `sglang.kernels.ops.<group>` and gate every Cake call on the
adapter's `supports_<name>`. Nothing in this package imports FlashInfer or `torch` at
import time, and no JIT build is triggered until a kernel is actually called.

Backend selection must preserve the operator's dtype, layout, state, mutation,
and numerical contract. In particular, Cake top-k-then-top-p sampling is not a
drop-in replacement for joint top-k/top-p filtering. Prepared plans must remain
valid when input contents and buffers change across calls and CUDA Graph replay.

## Adapter contract checklist

Every adapter module in this directory provides the following; `kvcache.py` together
with `ops/kvcache/cake.py` is the canonical example.

| item | rule |
|---|---|
| `FI_MODULE`, `FI_JIT_MODULE` | Fully qualified FlashInfer module names probed with `importlib.util.find_spec` through `_support.flashinfer_module_available`; further constants (e.g. `FI_SILU_MODULE`) when an entry needs more modules. |
| `ARCHS` | Tuple of `(major, minor)` the kernel was built for, from `_support` (`SM90`, `SM100`, `SM103`, `SM120`, `SM121`; `attention_common` adds `SM107`, `SM110`). |
| `supports_<name>(...)` | Never raises. Mirrors the FlashInfer admission: module present, tensor on a CUDA device in `ARCHS` (`cuda_tensor_on`, never ROCm), dtype/shape/layout contract, and the physical SM count where FlashInfer pins it (`multi_processor_count`, 148/152 for the DeepGEMM-family and MQA plans). Where FlashInfer exposes a generated-program table, the check consults it. |
| forwarder | Imports FlashInfer inside the function body and passes the **full** FlashInfer signature keyword for keyword, fixing `backend="cake"` where the entry is a `backend=` branch of a public function. Documented exceptions only (`fallback=False` for BGMV, `sf_layout=None` meaning the FlashInfer default). |
| module docstring | Names the FlashInfer entry and JIT module, states the contract at the baseline commit (dtypes, shapes, head configs, ranks), the CUDA-graph rules (what runs before capture, what is not capturable) and the unsupported configurations for which callers keep the existing SGLang path. |
| registration | One `KernelSpec` per op id in `ops/<group>/cake.py`: `op="<group>.<name>"`, `backend=KernelBackend.FLASHINFER`, `target="sglang.kernels.cake_kernels.<module>:<function>"`, `CapabilityRequirement.cuda(min_sm, max_sm)` covering `ARCHS` (inclusive range; the adapter checks exact membership) and `FormatSignature` dtypes, plus a `cake_<name>` wrapper resolving through `get_kernel(op, KernelBackend.FLASHINFER)`. |

## Inventory

167 op ids: 162 at the FlashInfer baseline (main commit `46340689a5ab`, 2026-10-02) plus
5 post-baseline entries read at main `e4f94f948` (noted `post-baseline` in the row). Columns:
op id without the group prefix, FlashInfer entry (module path after `flashinfer.`, then
`:name`), admitted compute capabilities (inclusive `KernelSpec` range; `any CUDA` marks
pure-torch or FlashInfer-quantizer weight preparation) and graph/prepare notes.

### `attention` (52)

Dense/MLA/sparse rows register in `ops/attention/cake.py`; KDA/GDN rows in
`ops/attention/cake_linear.py`.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `bsa_attn_sm100_blk64_sage_fwd` | `cute_dsl.sparse.bsa_attn_sm100_blk64:bsa_attn_sm100_blk64_fwd` | 10.0-10.3 | one-shot; pairs with `quantization.sage_fp8_quantize` |
| `bsa_attn_sm120_blk64_sage_fwd` | `cute_dsl.sparse.bsa_attn_sm120:bsa_attn_sm120_blk64_sage_fwd` | 12.0 | one-shot; registry-only CI coverage (no SM120 runner) |
| `concat_mla_k` | `concat_ops:concat_mla_k` | 10.0-10.3 | in-place K assembly; `backend="cake"` |
| `create_block_sparse_attention_wrapper` | `sparse:BlockSparseAttentionWrapper` | 10.0-10.3 | wrapper; `plan()` before capture |
| `create_sparse_mla_sm120_wrapper` | `mla:SparseMLASm120Wrapper` | 12.0-12.1 | wrapper; `plan()` before capture; SM120 registry-only in CI |
| `create_sparse_mla_sm120_dsv41_mixed_wrapper` | `mla:SparseMLASm120Wrapper` (`kv_cache_format='fp8'`, `kv_scale_format='ue8m0_g32'`, `extra_kv_fp4=True`) | 12.0-12.1 | post-baseline (#5983); wrapper, decode-only; warm every shape before capture; SM120 registry-only in CI |
| `create_variable_block_sparse_attention_wrapper_sm90` | `sparse:VariableBlockSparseAttentionWrapper` | 9.0 | wrapper, engine cuda/cute; plan before capture |
| `dcp_spec_decode` | `decode:trtllm_batch_decode_with_kv_cache` | 10.0-10.7 | `backend="cake"`; workspace sized by host helpers |
| `dsa_indexer_topk` | `dsa_indexer:dsa_indexer_topk` | 10.0-10.7 | one-shot; workspace via host helper |
| `dsv41_fp8_quantize_append_sparse_mla_cache` | `mla:dsv41_fp8_quantize_append_sparse_mla_cache` | 12.0-12.1 | post-baseline (#5983); in-place slot append into the 528 B/token main cache; capturable |
| `dsv41_fp8_quantize_pack_sparse_mla_cache` | `mla:dsv41_fp8_quantize_pack_sparse_mla_cache` | 12.0-12.1 | post-baseline (#5983); allocating full-page pack (HND/NHD) |
| `fmha_batch_context_with_kv_cache` | `prefill:trtllm_batch_context_with_kv_cache` | 10.0-10.3 | `backend="cake"`; one-shot |
| `fmha_batch_decode_with_kv_cache` | `decode:trtllm_batch_decode_with_kv_cache` | 10.0-10.3 | `backend="cake"`; one-shot |
| `gdn_chunk_gated_delta_rule` | `gdn_prefill:chunk_gated_delta_rule` | 10.0-10.3 | `backend="cake_gdn"`; manifest rows only (skip otherwise) |
| `gdn_cp_prefill_prepare` | `gdn_kernels.blackwell.cake_gdn_cp_backend:prepare_gdn_cp_prefill` | 10.0-10.3 | prepare once; FP32 checkpoint rows |
| `gdn_decode` | `gdn_decode:gated_delta_rule_decode` | 10.0-10.3 | `backend="cake_gdn"`; K-major state |
| `gdn_decode_pretranspose` | `gdn_decode:gated_delta_rule_decode_pretranspose` | 10.0-10.3 | `backend="cake_gdn"` default (auto via arg) |
| `kda_fused_decode` | `kda_decode:fused_kda_decode` | 10.0-10.3 | `backend="cake"`; manifest variants |
| `kda_packed_decode` | `kda_decode:packed_kda_decode` | 10.0-10.3 | one-shot |
| `kda_prefill_plan_cache` | `kda_prefill:KDAPrefillPlanCache` | 10.0-10.3 | host plan cache object |
| `kda_prefill_prepare_bf16` | `kda_prefill:prepare_bf16_kda_prefill` | 10.0-10.3 | prepare once; checkpoint every 16n tokens |
| `kda_prefill_prepare_tf32` | `kda_prefill:prepare_tf32_kda_prefill` | 10.0-10.3 | prepare once; TF32 tensor cores |
| `kda_prefill_supports_fp32_checkpoints` | `kda_prefill:kda_prefill_supports_fp32_checkpoints` | 10.0-10.3 | host capability query |
| `kda_recurrent` | `kda:recurrent_kda` | 10.0-10.3 | `backend="cake"` |
| `kimi_k3_attn_res` | `kimi_k3_attn_res:kimi_k3_attn_res` | 10.0-10.3 | one-shot; per-(M,K) program table |
| `kimi_k3_mla_fp8_paged_attention` | `mla:run_cake_kimi_k3_mla_fp8_paged_attention` | 10.0-10.3 | one-shot |
| `launch_sm110_gqa_decode_prepared` | `decode:launch_sm110_gqa_decode_prepared` | 11.0 | Thor; registry-only (no runner) |
| `minimax_h3_varlen_attention` | `prefill:minimax_h3_varlen_attention` | 10.0-10.3 | one-shot (packed varlen) |
| `minimax_h3_varlen_nvfp4_attention` | `prefill:minimax_h3_varlen_nvfp4_attention` | 10.0-10.3 | one-shot (packed varlen) |
| `mla_varq_dcp_decode` | `mla:cake_mla_varq_dcp_decode` | 10.0-10.3 | one-shot; workspace via host helper |
| `prepare_balanced_batch_decode_with_kv_cache` | `decode:prepare_balanced_batch_decode_with_kv_cache` | 10.0-10.3 | prepare once; on-device load balance |
| `prepare_dense_mqa_logits` | `dense_mqa:prepare_dense_mqa_logits` | 10.0-10.3 | prepare once; pins 148/152 SMs |
| `prepare_dsa_indexer_topk` | `experimental.cake_dsa_indexer.cake_backend:prepare_dsa_indexer_topk` | 10.0-10.7 | prepare once |
| `prepare_kimi_k3_attn_res` | `kimi_k3_attn_res:prepare_kimi_k3_attn_res` | 10.0-10.3 | prepare once |
| `prepare_kimi_k3_mla_fp8_paged_attention` | `mla:KimiK3MlaFp8PagedAttention` | 10.0-10.3 | prepare once (runner class) |
| `prepare_minimax_h3_varlen_attention` | `experimental.minimax_h3_varlen_attention.cake_backend:prepare_minimax_h3_varlen_attention` | 10.0-10.3 | prepare once |
| `prepare_minimax_h3_varlen_nvfp4_attention` | `experimental.minimax_h3_varlen_attention.cake_backend:prepare_minimax_h3_varlen_nvfp4_attention` | 10.0-10.3 | prepare once |
| `prepare_mla_varq_dcp_decode` | `mla:prepare_cake_mla_varq_dcp_decode` | 10.0-10.3 | prepare once |
| `prepare_msa_nvfp4_sparse_decode` | `msa_ops:prepare_msa_nvfp4_sparse_decode` | 10.0-10.7 | prepare once; inputs need FI packers |
| `prepare_nvfp4_attention` | `prefill:prepare_nvfp4_attention` | 10.3 | prepare once; SM103 only |
| `prepare_nvfp4_batch_decode_with_kv_cache_mla` | `mla:prepare_nvfp4_batch_decode_with_kv_cache_mla` | 10.0-10.3 | prepare once (runner class) |
| `prepare_sm110_gqa_decode` | `decode:prepare_sm110_gqa_decode` | 11.0 | Thor; registry-only (no runner) |
| `prepare_sparse_mqa_logits` | `sparse_mqa:prepare_sparse_mqa_logits` | 10.0-10.3 | prepare once; pins 148/152 SMs |
| `prepare_sparse_mqa_metadata` | `sparse_mqa:prepare_sparse_mqa_metadata` | 10.0-10.3 | prepare once (int32 metadata plan) |
| `sm110_gqa_decode` | `decode:sm110_gqa_decode` | 11.0 | Thor; registry-only (no runner) |
| `sm110_xqa_attention` | `sm110_xqa:attention` | 11.0 | Thor; registry-only (no runner) |
| `sm110_xqa_prepare` | `sm110_xqa:prepare` | 11.0 | Thor; registry-only (no runner) |
| `sparse_mla_sm120_dsv4_nvfp4_decode` | `mla:cake_sparse_mla_sm120_dsv4_nvfp4_decode` | 12.0-12.1 | SM120 registry-only in CI |
| `sparse_mla_sm120_dsv4_nvfp4_prefill` | `mla:cake_sparse_mla_sm120_dsv4_nvfp4_prefill` | 12.0-12.1 | SM120 registry-only in CI |
| `sparse_mla_sm120_dsv41_mixed_decode` | `mla:cake_sparse_mla_sm120_dsv41_mixed_decode` | 12.0-12.1 | post-baseline (#5983); allocation-free, caller-owned `mid_out`/`mid_lse` when the plan splits; `compute_precision` bf16/fp8; SM120 registry-only in CI |
| `trtllm_batch_decode_sparse_mla_dsv4` | `mla:trtllm_batch_decode_sparse_mla_dsv4` | 10.0-10.3, 12.0-12.1 | `backend="cake"`; `cake_dsv4_workspace_reset` before capture |
| `trtllm_batch_decode_with_kv_cache_mla` | `mla:trtllm_batch_decode_with_kv_cache_mla` | 10.0-10.3 | `backend="cake"`; 148/152-SM builds only |

### `communication` (16)

All rows are collectives or their workspace lifecycle; rank counts are part
of the contract.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `all_gather_matmul` | `comm.all_gather_matmul.all_gather_matmul:all_gather_matmul` | 10.0-10.3 | collective; host launcher nvcc-built at first use |
| `create_kimi_k3_tp12_tail_workspace` | `kimi_k3_tp12_tail:create_kimi_k3_tp12_tail_workspace` | 10.0-10.3 | MNNVL workspace factory (12 ranks) |
| `fused_norm_combine` | `comm.cake_fused_norm_combine:cake_fused_norm_combine` | 10.0-10.3 | 8-rank CUDA-IPC collective; workspace lifecycle |
| `fused_norm_combine_create_workspace` | `comm.cake_fused_norm_combine:cake_fused_norm_combine_create_workspace` | 10.0-10.3 | collective factory |
| `fused_norm_combine_destroy_workspace` | `comm.cake_fused_norm_combine:cake_fused_norm_combine_destroy_workspace` | 10.0-10.3 | collective teardown |
| `kimi_k3_tp12_tail` | `kimi_k3_tp12_tail:kimi_k3_tp12_tail` | 10.0-10.3 | one-shot; 12 ranks |
| `moe_a2a_combine` | `comm.trtllm_moe_alltoall:moe_a2a_combine` | 10.0-10.3 | `backend="cake"`; `sf_layout=None` = FI default |
| `moe_a2a_dispatch` | `comm.trtllm_moe_alltoall:moe_a2a_dispatch` | 10.0-10.3 | `backend="cake"` |
| `moe_a2a_get_workspace_size_per_rank` | `comm.trtllm_moe_alltoall:moe_a2a_get_workspace_size_per_rank` | 10.0-10.3 | host sizing |
| `moe_a2a_initialize` | `comm.trtllm_moe_alltoall:moe_a2a_initialize` | 10.0-10.3 | workspace init, outside capture |
| `moe_a2a_sanitize_expert_ids` | `comm.trtllm_moe_alltoall:moe_a2a_sanitize_expert_ids` | 10.0-10.3 | `backend="cake"` |
| `moe_ep_alltoall` | `moe_ep.backends.split.comm.cake.communication:CakeAlltoAll` | 10.0-10.3 | backend object (`CakeAlltoAll`) |
| `prepare_all_gather_matmul` | `comm.all_gather_matmul.all_gather_matmul:prepare_all_gather_matmul` | 10.0-10.3 | prepare once; `(world,N)` in {(8,1280),(4,2560 SM103)} |
| `prepare_kimi_k3_tp12_tail` | `kimi_k3_tp12_tail:prepare_kimi_k3_tp12_tail` | 10.0-10.3 | prepare once; 12 ranks |
| `trtllm_moe_allreduce_fusion` | `comm.trtllm_ar:trtllm_moe_allreduce_fusion` | 10.0-10.3 | collective on TRT-LLM AR workspace |
| `trtllm_moe_finalize_allreduce_fusion` | `comm.trtllm_ar:trtllm_moe_finalize_allreduce_fusion` | 10.0-10.3 | collective; `moe_finalize_backend="cake"` |

### `diffusion` (33)

MiniMax-H3 blocks. SM100/103 (tcgen05) and SM120 (mma.sync) routes use different
prepared weight layouts.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `minimax_h3_bf16_pre_attention` | `diffusion_ops.minimax_h3:minimax_h3_bf16_pre_attention` | 10.0-10.3 | one-shot |
| `minimax_h3_dense_attention` | `diffusion_ops.cake_minimax_h3_dense_attention:minimax_h3_dense_attention` | 9.0-10.3, 12.0-12.1 | one-shot; cc 12.1 admitted but unmeasured |
| `minimax_h3_fc1_swiglu` | `diffusion_ops.minimax_h3_fc1_swiglu:minimax_h3_fc1_swiglu` | 10.0-10.3 | one-shot |
| `minimax_h3_fc1_swiglu_fp8` | `diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu:minimax_h3_fc1_swiglu_fp8` | 12.0 | one-shot |
| `minimax_h3_fc1_swiglu_mxfp8` | `diffusion_ops.minimax_h3_fc1_swiglu:minimax_h3_fc1_swiglu_mxfp8` | 10.0-10.3 | one-shot |
| `minimax_h3_fc1_swiglu_nvfp4` | `diffusion_ops.minimax_h3_fc1_swiglu:minimax_h3_fc1_swiglu_nvfp4` | 10.0-10.3, 12.0 | arch dispatcher (SM120 route on cc 12.0) |
| `minimax_h3_fp8_out_proj` | `diffusion_ops.cake_minimax_h3_sm120_quant_out_proj:minimax_h3_fp8_out_proj` | 12.0 | one-shot |
| `minimax_h3_fp8_pre_attention` | `diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention:minimax_h3_fp8_pre_attention` | 12.0 | one-shot |
| `minimax_h3_nvfp4_out_proj` | `diffusion_ops.cake_minimax_h3_sm120_quant_out_proj:minimax_h3_nvfp4_out_proj` | 12.0 | one-shot |
| `minimax_h3_nvfp4_pre_attention` | `diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention:minimax_h3_nvfp4_pre_attention` | 12.0 | one-shot |
| `minimax_h3_out_proj` | `diffusion_ops.minimax_h3_out_proj:minimax_h3_out_proj` | 10.0-10.3 | one-shot |
| `minimax_h3_out_proj_mxfp8` | `diffusion_ops.minimax_h3_out_proj:minimax_h3_out_proj_mxfp8` | 10.0-10.3 | one-shot |
| `minimax_h3_out_proj_nvfp4` | `diffusion_ops.minimax_h3_out_proj:minimax_h3_out_proj_nvfp4` | 10.0-10.3 | one-shot |
| `minimax_h3_qkv_quantize_pack` | `diffusion_ops.cake_minimax_h3_qkv_pack:minimax_h3_qkv_quantize_pack` | 10.0-10.3 | one-shot |
| `minimax_h3_sm120_varlen_attention_fp8` | `diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention:minimax_h3_sm120_varlen_attention_fp8` | 12.0-12.1 | one-shot |
| `minimax_h3_sm120_varlen_attention_nvfp4` | `diffusion_ops.cake_minimax_h3_sm120_nvfp4_varlen_attention:minimax_h3_sm120_varlen_attention_nvfp4` | 12.0-12.1 | one-shot; experimental |
| `minimax_h3_varlen_attention` | `prefill:minimax_h3_varlen_attention` | 10.0-10.3 | one-shot |
| `minimax_h3_varlen_nvfp4_attention` | `prefill:minimax_h3_varlen_nvfp4_attention` | 10.0-10.3 | one-shot |
| `prepare_minimax_h3_fc1_weight_fp8` | `diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu:prepare_minimax_h3_fc1_weight_fp8` | any CUDA | offline weight prep |
| `prepare_minimax_h3_fc1_weight_mxfp8` | `diffusion_ops.minimax_h3_fc1_swiglu:prepare_minimax_h3_fc1_weight_mxfp8` | any CUDA | offline weight prep |
| `prepare_minimax_h3_fc1_weight_nvfp4` | `diffusion_ops.minimax_h3_fc1_swiglu:prepare_minimax_h3_fc1_weight_nvfp4` | any CUDA | offline weight prep; arch dispatcher |
| `prepare_minimax_h3_fc1_weight_nvfp4_sm120` | `diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu:prepare_minimax_h3_fc1_weight_nvfp4_sm120` | any CUDA | offline weight prep |
| `prepare_minimax_h3_mxfp8_pre_attention` | `diffusion_ops.cake_minimax_h3_mxfp8:prepare_minimax_h3_mxfp8_pre_attention` | 10.0-10.3 | prepare once, launch many |
| `prepare_minimax_h3_nvfp4_pre_attention` | `diffusion_ops.cake_minimax_h3_nvfp4:prepare_minimax_h3_nvfp4_pre_attention` | 10.0-10.3 | prepare once, launch many |
| `prepare_minimax_h3_o_weight_mxfp8` | `diffusion_ops.minimax_h3_out_proj:prepare_minimax_h3_o_weight_mxfp8` | any CUDA | offline weight prep |
| `prepare_minimax_h3_o_weight_nvfp4` | `diffusion_ops.minimax_h3_out_proj:prepare_minimax_h3_o_weight_nvfp4` | any CUDA | offline weight prep |
| `prepare_minimax_h3_qkv_quantize_pack` | `diffusion_ops.cake_minimax_h3_qkv_pack:prepare_minimax_h3_qkv_quantize_pack` | 10.0-10.3 | prepare once, launch many |
| `prepare_minimax_h3_varlen_attention` | `experimental.minimax_h3_varlen_attention.cake_backend:prepare_minimax_h3_varlen_attention` | 10.0-10.3 | prepare once, launch many |
| `prepare_minimax_h3_varlen_nvfp4_attention` | `experimental.minimax_h3_varlen_attention.cake_backend:prepare_minimax_h3_varlen_nvfp4_attention` | 10.0-10.3 | prepare once, launch many |
| `quantize_minimax_h3_o_weight_fp8` | `diffusion_ops.cake_minimax_h3_sm120_quant_out_proj:quantize_minimax_h3_o_weight_fp8` | any CUDA | offline weight prep |
| `quantize_minimax_h3_o_weight_nvfp4` | `diffusion_ops.cake_minimax_h3_sm120_quant_out_proj:quantize_minimax_h3_o_weight_nvfp4` | any CUDA | offline weight prep |
| `quantize_minimax_h3_qkv_weight_fp8` | `diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention:quantize_minimax_h3_qkv_weight_fp8` | any CUDA | offline weight prep |
| `quantize_minimax_h3_qkv_weight_nvfp4` | `diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention:quantize_minimax_h3_qkv_weight_nvfp4` | any CUDA | offline weight prep |

### `gemm` (23)

DeepGEMM-family plans and the dense/sparse MQA plans pin the physical SM count
(148 on SM100, 152 on SM103).

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `allocate_kimi_k3_fp8_projection_workspace` | `gemm.kimi_k3_fp8_projection:allocate_kimi_k3_fp8_projection_workspace` | 10.0-10.3 | host workspace allocation |
| `allocate_nvfp4_per_token_quantize_outputs` | `experimental.cake_nvfp4_per_token.cake_backend:allocate_nvfp4_per_token_quantize_outputs` | 10.0-10.3 | allocates quantizer outputs (outside capture) |
| `bmm_bf16` | `gemm:bmm_bf16` | 10.0-10.3 | `backend="cake"`; one-shot |
| `grouped_gemm_fwd` | `experimental.cake_moe_grouped_gemm:grouped_gemm_fwd` | 10.0-10.7 | one-shot; adapter pins {10.0,10.3,10.7} |
| `grouped_mm_bf16` | `grouped_mm:grouped_mm_bf16` | 10.0-10.7 | `backend="cake"`; adapter pins {10.0,10.3,10.7} |
| `kimi_k3_fp8_projection` | `gemm.kimi_k3_fp8_projection:kimi_k3_fp8_projection` | 10.0-10.3 | host workspace allocation |
| `mm_fp4_per_token` | `gemm:mm_fp4` | 10.0-10.3 | `backend="cake"`; tactic depends on SM count |
| `mm_m1_16_k6144_n256` | `gemm.routergemm:mm_M1_16_K6144_N256` | 10.0-10.3 | one-shot; operands on the current device |
| `mm_m1_16_k7168_n128` | `gemm.routergemm:mm_M1_16_K7168_N128` | 10.0-10.3 | one-shot; operands on the current device |
| `mm_m1_16_k7168_n256` | `gemm.routergemm:mm_M1_16_K7168_N256` | 10.0-10.3 | one-shot; operands on the current device |
| `mm_nvfp4_svdquant` | `gemm:mm_nvfp4_svdquant` | 10.0-10.3 | `backend="cake"`; one-shot |
| `prepare_fp4_gemm` | `fp4_gemm:prepare_fp4_gemm` | 10.0-10.3 | prepare once; pins 148/152 SMs |
| `prepare_fp4_k_grouped_gemm` | `fp4_k_grouped_gemm:prepare_fp4_k_grouped_gemm` | 10.0-10.3 | prepare once; runtime shapes (any group layout, `num_stages` 7) |
| `prepare_fp8_batched_gemm` | `fp8_batched_gemm:prepare_fp8_batched_gemm` | 10.0-10.3 | prepare once; pins 148/152 SMs |
| `prepare_fp8_fp4_gemm` | `fp8_fp4_gemm:prepare_fp8_fp4_gemm` | 10.0-10.3 | prepare once; runtime shapes (`gran_k_a` 32/128) |
| `prepare_fp8_gemm_1d1d` | `experimental.deepgemm_fp8_gemm:prepare_fp8_gemm_1d1d` | 10.0-10.3 | prepare once; any M/N, K % 128; JIT-built at first use |
| `prepare_group_gemm_fp8_nt_groupwise_contiguous` | `gemm.cake_grouped_fp8_gemm:prepare_group_gemm_fp8_nt_groupwise_contiguous` | 10.0 | prepare once; first launch NOT capturable |
| `prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant` | `gemm.cake_grouped_fp8_fused_silu_quant:prepare_group_gemm_fp8_nt_groupwise_contiguous_silu_quant` | 10.0 | prepare once; first launch NOT capturable |
| `prepare_grouped_gemm_fwd` | `experimental.cake_moe_grouped_gemm.cake_backend:prepare_grouped_gemm_fwd` | 10.0-10.7 | prepare once; adapter pins {10.0,10.3,10.7} |
| `prepare_kimi_k3_fp8_projection` | `gemm.kimi_k3_fp8_projection:prepare_kimi_k3_fp8_projection` | 10.0-10.3 | prepare once, launch many |
| `prepare_kimi_k3_fp8_projection_weights` | `gemm.kimi_k3_fp8_projection:prepare_kimi_k3_fp8_projection_weights` | 10.0-10.3 | offline weight prep |
| `prepare_mm_fp4_per_token` | `experimental.cake_nvfp4_per_token.cake_backend:prepare_mm_fp4_per_token` | 10.0-10.3 | prepare once, launch many |
| `prepare_nvfp4_per_token_chain` | `experimental.cake_nvfp4_per_token.cake_backend:prepare_nvfp4_per_token_chain` | 10.0-10.3 | prepare once; quantizer + GEMM chain |

### `kvcache` (2)

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `fused_qk_rmsnorm_rope_append_paged_kv_cache` | `cake_fused_qk_rope_append:cake_fused_qk_rmsnorm_rope_append_paged_kv_cache` | 9.0-10.3 | in-place NHD append; capturable |
| `fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache` | `cake_fused_qk_rope_fp8_append:cake_fused_qk_rmsnorm_rope_quantize_fp8_append_paged_kv_cache` | 9.0-10.3 | post-baseline (#5956); FP8 E4M3 Q + `q_scale` + `split_k_flag`, in-place NHD `float8_e4m3fn` append; capturable |

### `mamba` (3)

`mamba.selective_state_update` shares its op id with the vendored Triton kernel;
select the backend explicitly.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `selective_state_update` | `mamba:selective_state_update` | 10.0-10.3 | `backend="cake"`; dynamic checkpoint = host sync |
| `ssd_combined` | `mamba:SSDCombined` | 10.0-10.3 | runner class (`backend="cake"`) |
| `ssd_combined_fwd` | `mamba:ssd_combined_fwd` | 10.0-10.3 | one-shot |

### `mm` (3)

Kimi-K3 vision tower (multimodal input encoding, hence `mm`).

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `kimi_k3_vision_tower` | `kimi_k3_vision:kimi_k3_vision_tower` | 10.0-10.3 | one-shot; `plan=`/`pos_rows=` pass-through |
| `prepare_kimi_k3_vision_tower` | `kimi_k3_vision:prepare_kimi_k3_vision_tower` | 10.0-10.3 | prepare once; graph-capturable |
| `prepare_kimi_k3_vision_weights` | `experimental.kimi_k3_vision_tower.cake_backend:prepare_kimi_k3_vision_weights` | any CUDA | offline weight prep |

### `moe` (29)

MegaMoE / warp-decode rows are plan or runner objects; composition stays in `srt`.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `allocate_kimi_k3_route_plan` | `fused_moe:allocate_kimi_k3_route_plan` | 10.0-10.3 | route plan allocation (outside capture) |
| `bind_mega_moe_prepared` | `mega_moe_v3:bind_prepared` | 10.0-10.3 | low-level route binding |
| `create_mxfp8_megamoe_ep16_session` | `moe_ep:CakeMxfp8MegaMoeEp16` | 10.3 | session; 16 NVSHMEM ranks |
| `fused_topk_deepseek` | `fused_moe:fused_topk_deepseek` | 10.0-10.3 | `backend="cake"`; in-place outputs |
| `kimi_k3_fused_router` | `fused_moe:kimi_k3_fused_router` | 10.0-10.3 | one-shot |
| `kimi_k3_latent_moe_front` | `kimi_k3_latent_moe:kimi_k3_latent_moe_front` | 10.0-10.3 | one-shot |
| `kimi_k3_latent_moe_tail` | `kimi_k3_latent_moe:kimi_k3_latent_moe_tail` | 10.0-10.3 | one-shot |
| `kimi_k3_situ_fused_moe` | `fused_moe:cutlass_fused_moe` | 10.0-10.3 | `backend="cake"`; fixed H3584/I384/E896/top16 |
| `kimi_k3_situ_fused_moe_prepare_workspace` | `fused_moe:cake_fused_moe_prepare_workspace` | 10.0-10.3 | outside capture, once per token count |
| `kimi_k3_situ_fused_moe_workspace_size` | `fused_moe:cutlass_fused_moe_workspace_size` | 10.0-10.3 | host sizing (`backend="cake"`) |
| `load_megamoe_topk_reduce_module` | `jit.cake_megamoe_topk_reduce:get_cake_megamoe_topk_reduce_module` | 10.0-10.3 | JIT module loader (not import-time) |
| `megamoe_topk_reduce` | `jit.cake_megamoe_topk_reduce:run_cake_megamoe_topk_reduce` | 10.0-10.3 | one-shot reducer |
| `prepare_bgmv_moe` | `fused_moe:prepare_bgmv_moe` | 9.0-10.3 | `backend="cake"`, `fallback=False` default |
| `prepare_kimi_k3_fused_router` | `fused_moe:prepare_kimi_k3_fused_router` | 10.0-10.3 | prepare once, launch many |
| `prepare_kimi_k3_latent_moe_front` | `kimi_k3_latent_moe:prepare_kimi_k3_latent_moe_front` | 10.0-10.3 | prepare once, launch many |
| `prepare_kimi_k3_latent_moe_tail` | `kimi_k3_latent_moe:prepare_kimi_k3_latent_moe_tail` | 10.0-10.3 | prepare once, launch many |
| `prepare_mega_gate` | `mega_gate:prepare_mega_gate` | 10.0-10.3 | prepare once, launch many |
| `prepare_mega_moe_grouped_fused` | `mega_moe_v3:prepare_grouped_fused` | 10.0-10.3 | prepare once, launch many |
| `prepare_mega_moe_grouped_l1` | `mega_moe_v3:prepare_grouped_l1` | 10.0-10.3 | prepare once, launch many |
| `prepare_mega_moe_grouped_l2` | `mega_moe_v3:prepare_grouped_l2` | 10.0-10.3 | prepare once, launch many |
| `prepare_mega_moe_pipeline` | `mega_moe_v3:prepare_pipeline` | 10.0-10.3 | prepare once, launch many |
| `prepare_source_mega_moe` | `source_mega_moe:prepare_mega_moe` | 10.0-10.3 | prepare once, launch many |
| `preprocess_mxfp8_megamoe_ep16_weights` | `moe_ep:preprocess_cake_mxfp8_megamoe_ep16_weights` | 10.3 | offline weight prep |
| `preprocess_sm90_push_cake_bf16_mega_weights` | `moe_ep:preprocess_sm90_push_cake_bf16_mega_weights` | 9.0 | offline weight prep; EP group <= 32 ranks |
| `sm90_push_cake_megamoe_config` | `moe_ep:Sm90_Bf16_Bf16_Bf16_PushCake_MegaMoeConfig` | 9.0 | config object; EP group <= 32 ranks |
| `warp_decode_config` | `fused_moe:CakeWarpDecodeConfig` | 10.0-10.3 | config object |
| `warp_decode_prepare_activations` | `fused_moe:CakeWarpDecodeConfig.prepare_activations` | 10.0-10.3 | per-call activation prep (`CakeWarpDecodeConfig`) |
| `warp_decode_prepare_weights` | `fused_moe:CakeWarpDecodeConfig.prepare_weights` | 10.0-10.3 | offline weight prep (`CakeWarpDecodeConfig`) |
| `warp_decode_runner` | `fused_moe:CakeWarpDecodeRunner` | 10.0-10.3 | runner; compose via FI `MoELayer` |

### `quantization` (3)

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `mxfp8_grouped_quantize` | `quantization.fp8_quantization:mxfp8_grouped_quantize` | 10.0-10.3 | `backend="cake"`; per-dtype profile |
| `nvfp4_quantize_per_token` | `quantization.fp4_quantization:nvfp4_quantize` | 10.0-10.3 | `backend="cake"`, `per_token_activation=True` |
| `sage_fp8_quantize` | `cute_dsl.sparse.bsa_sage_sm100_cake:sage_fp8_quantize_sm100` | 10.0-10.3 | one-shot; feeds Sage BSA |

### `sampling` (3)

Explicit opt-in only; see the sampler caveats under Integration guidance.

| op id | FlashInfer entry | SM | notes (graph/prepare) |
|---|---|---|---|
| `softmax` | `sampling:softmax` | 10.3 | FP32 [<=64, 128256..262144]; FI auto-routes |
| `top_k_probs_to_slab` | `cake_sampling:top_k_probs_to_slab` | 9.0-12.1 | stage 1 only; raises when unservable |
| `top_k_top_p_sampling_from_probs_top_k_first` | `cake_sampling:top_k_top_p_sampling_from_probs` | 9.0-12.1 | top-k first (NOT joint); deterministic |

## Not forwarded

FlashInfer entries that reference Cake but have no op id here, by category:

- **Host helpers**: route selectors, module getters, manifests, plan builders, workspace
  sizing (`*_workspace_bytes`, `*_workspace_size`) and variant rankers. FlashInfer applies
  them inside the public entry. A few are re-exported as unregistered adapter helpers
  (e.g. `attention_mla.cake_dsv4_workspace_reset`, `attention_fmha.balanced_workspace_bytes`)
  because a `supports_*` check or a caller needs them.
- **Training-only**: grouped GEMM dgrad/wgrad and the autograd `CakeGroupedMm`, chunked
  LM-head loss/logprob (forward + backward), DSA training launches.
- **Non-Cake**: entries inventoried as `is_cake=no` (cuDNN linear attention, CuTe KDA
  prefill, `chunk_gated_delta_rule2`, `RecurrentKDAPrefillWrapper`, portable `bgmv_moe*`,
  `tinygemm_bf16`, `mm_bf16_fp4`, and the `"default"` branch next to a `backend="cake"`
  branch).
- **Internal raw ops**: `get_*_backend(backend="cake")` raw ops, private SM120 routes
  reached through an arch dispatcher, plan and return-type objects, stable wrapper classes
  duplicating a `prepare_*` entry, drop-level shims under `moe_ep.kernel_src`, pure-torch
  test references.
- **Post-baseline modules**: added to FlashInfer after `46340689a5ab`. The FP8 fused QK
  RoPE + paged append (#5956) and the DSv4.1 mixed-cache SM120 sparse MLA with its FP8
  main-cache writers (#5983) are forwarded as of the `e4f94f948` re-pin (rows marked
  `post-baseline`). Still not forwarded: the DeepGEMM FP8 JIT helper
  (`experimental/deepgemm_fp8_gemm/cake_jit.py`, a loader) and the DSA training launch
  helper (`experimental/cake_dsa_train/cake_launch.py`, training-only). The pre-baseline
  `dsv41_fp4_quantize_{pack,append}_sparse_mla_cache` extra-cache writers are hand-written
  (`is_cake=no`) and stay unforwarded; the mixed-cache test reaches them directly.
- **Dispatch-only rows**: `allreduce_fusion(moe_finalize_backend="cake")` (SGLang's
  unified all-reduce workspace path passes the kwarg itself), the deprecated
  `MoeAlltoAll(backend="cake")` (use `communication.moe_ep_alltoall`) and package
  `__init__` re-exports.

Rows owned by another group are listed once under their owner: `gemm` is canonical for
grouped GEMM, `communication` for the fused norm-combine and the TP12 tail, `quantization`
for the standalone quantizers. The full per-row inventory with reasons lives in the Cake
project, not in this tree.

## Versioning

- **Baseline**: every docstring contract, `supports_*` check and test was read from
  FlashInfer main commit `46340689a5ab` (2026-10-02).
- **Re-pin to `e4f94f948`**: the five post-baseline op ids (`kvcache.fused_qk_rmsnorm_rope_
  quantize_fp8_append_paged_kv_cache`; `attention.sparse_mla_sm120_dsv41_mixed_decode`,
  `attention.create_sparse_mla_sm120_dsv41_mixed_wrapper`,
  `attention.dsv41_fp8_quantize_{pack,append}_sparse_mla_cache`) were read from FlashInfer
  main `e4f94f948` (PRs #5956 and #5983); their docstrings cite that commit.
- **Pinned release**: SGLang pins `flashinfer_python 0.7.0.post1`, which ships 11 of the
  117 Cake Python modules present at the baseline. For the other 106, `find_spec` returns
  `None`, `supports_*` returns `False` and callers keep their existing backend: no import
  error, no JIT attempt, no warning. Registry entries are metadata and still exist, so
  `select_kernel(op, backend=KernelBackend.FLASHINFER)` resolves; only `.load()` or a
  forwarder call reaches the missing module. Degradation is per op (module presence), not
  per package.
- **When the pin moves**: re-read every forwarded entry against the new FlashInfer commit
  (signature, admitted architectures, graph rules), update the docstring baseline line and
  run the registered tests on SM100/SM103 hardware. Post-baseline Cake modules become
  candidates for new adapters; removed or renamed entries lose their `KernelSpec` (no
  aliases).
- This directory does not change SGLang's dependency pins.

## Testing

Tests live in `test/registered/kernels/ops/<group>/test_cake_<name>.py`, registered with
`register_cuda_ci(stage="base-b-kernel-unit", runner_config="4-gpu-b200")` (SM90-only and
SM120-only files on `1-gpu-large`; 8-rank collectives on `8-gpu-b200`, stage `nightly`).
Each file contains:

- a GPU-free registry test per op id (backend resolves to `FLASHINFER`, target points at
  the adapter);
- `supports_*` rejection cases (dtype, shape, device, monkeypatched module absence);
- GPU parity: facade vs direct FlashInfer call (bitwise), then vs a pure-torch reference.

Skip semantics: GPU tests `pytest.skip` with a reason when CUDA is absent, when the
installed FlashInfer lacks the Cake module (`flashinfer_module_available`), when the
device is outside `ARCHS` (or has an SM count the plans were not frozen for), or when
FlashInfer registers no generated program / manifest row for the shape. Multi-rank tests
(`communication`; EP16 and the SM90 push-MoE in `moe`) run under `torchrun` /
`multigpu_pytest_main` and skip with "needs N ranks" when not launched with the matching
world size; the 12-rank Kimi-K3 tail has no CI runner and self-skips.

Tolerances (never looser):

| precision | `atol` / `rtol` |
|---|---|
| BF16 / FP16 outputs | 1e-2 / 1e-2 (2e-2 only where the kernel applies several 16-bit roundings, e.g. the eight-peer rank-ordered sum) |
| FP8 (e4m3) operands | 0.1 / 0.1 |
| FP4 (e2m1) block-scaled | 1.0 / 0.1 |
| FP32-compute state paths (GDN K-major decode, SSU FP32 identity, fused KDA decode on an FP32 pool) | 1e-3 |

Two files currently carry FlashInfer's own looser bound and flag it (AttnRes BF16 at
8e-2 / 3e-2; prepared KDA prefill FP32 pool at 1e-2). Tightening them is a maintainer
decision, not a test-side change.

## Integration guidance

- Import the facade (`from sglang.kernels.ops.<group> import cake_<name>`) and the gate
  (`from sglang.kernels.cake_kernels.<module> import supports_<name>`); call the Cake
  entry only when the gate is true, otherwise keep the existing backend. Do not re-derive
  the admission at the call site.
- Prepare plans, workspaces and prepared runners (`prepare_*`, `*_initialize`,
  `*_prepare_workspace`, wrapper `plan()`) before CUDA-graph capture, once per shape
  class; replay launches only. Weight preparation is offline.
- Anything outside the documented contract (other head configs, HND caches, FP8 caches,
  other dtypes, other rank counts, unlisted shapes) stays on the current SGLang path.

Semantic caveats:

- **Sampler**: `sampling.top_k_top_p_sampling_from_probs_top_k_first` applies top-k and
  then top-p (`filter_apply_order="top_k_first"`); it is not a drop-in for SGLang's joint
  filter and must be requested explicitly. The sampler's softmax fast path accepts FP32
  contiguous logits on SM103, small batches and large vocabularies. The sampler retains
  its existing filtering and random-number generation, and deterministic inference
  retains `torch.softmax`. The fast path also requires FlashInfer's
  `cake_blackwell_softmax` module; older FlashInfer installations retain Torch softmax.
- **MiniMax-H3 weight layouts**: SM100/103 (tcgen05) prepared weights
  (`prepare_minimax_h3_*_weight_{mxfp8,nvfp4}`, `prepare_minimax_h3_o_weight_*`) are not
  interchangeable with the SM120 layouts (`*_weight_fp8`, `*_nvfp4_sm120`,
  `quantize_minimax_h3_*`). Prepare on the architecture that runs, or use the
  `prepare_minimax_h3_fc1_weight_nvfp4` dispatcher.
- **DSv4 sparse MLA**: reset the workspace once (`cake_dsv4_workspace_reset` or one
  eager call) before capturing `trtllm_batch_decode_sparse_mla_dsv4`.
- **Grouped FP8 GEMM**: the first `launch()` of a prepared
  `prepare_group_gemm_fp8_nt_groupwise_contiguous[_silu_quant]` plan fills descriptor
  storage synchronously and is not capturable; warm it up eagerly, then capture.
- **Kimi-K3 SiTU MoE**: `kimi_k3_situ_fused_moe_prepare_workspace` runs outside capture,
  once per token count.
- Other operators require their own numerical and real-model performance evidence
  before being added here.
