# vLLM Torch Compile Fusion Patterns

Refresh: `2026-05-01`.
Source tree: vLLM `origin/main` at `7075df79b`.

Use this file when the fuse-pattern table reports split kernels in a trace and
you need to decide whether the shape is already covered by vLLM's
`torch.compile` pattern matcher. Treat every row here as an upstream precedent
before calling a similar SGLang opportunity novel.

## Pass Registration

vLLM registers these passes from
`vllm/compilation/passes/pass_manager.py` through `PassConfig`.

| Toggle | Pass | Target shape |
| --- | --- | --- |
| `enable_sp` | `SequenceParallelismPass` | all-reduce around residual/norm blocks becomes reduce-scatter, local work, and all-gather |
| `fuse_gemm_comms` | `AsyncTPPass` | GEMM plus reduce-scatter / all-gather overlap through symmetric-memory collectives |
| `fuse_allreduce_rms` | `AllReduceFusionPass` | all-reduce followed by RMSNorm, optional residual add, optional FP8 / NVFP4 quant |
| `fuse_minimax_qk_norm` | `MiniMaxQKNormPass` | MiniMax Q/K all-reduce plus RMSNorm decode path |
| `fuse_norm_quant` | `RMSNormQuantFusionPass` | RMSNorm or fused-add-RMSNorm followed by FP8 / FP4 quant |
| `fuse_norm_quant` + AITER | `RocmAiterRMSNormQuantFusionPass` | ROCm AITER RMSNorm / fused-add-RMSNorm followed by AITER or vLLM quant |
| `fuse_act_quant` | `ActivationQuantFusionPass` | SiLU-and-mul followed by FP8 / NVFP4 / block quant |
| `fuse_act_quant` + AITER | `RocmAiterSiluMulFp8GroupQuantFusionPass` | AITER SiLU-and-mul followed by FP8 group quant |
| `fuse_act_padding` + AITER | `RocmAiterTritonAddRMSNormPadFusionPass` | AITER fused-add-RMSNorm followed by padding into the next layout |
| `fuse_mla_dual_rms_norm` + AITER | `MLADualRMSNormFusionPass` | MLA paired Q and KV RMSNorms become `fused_mla_dual_rms_norm` |
| `fuse_rope_kvcache` | `RopeKVCacheFusionPass` | RoPE plus paged KV-cache update, after split cleanup passes |
| `fuse_attn_quant` | `AttnQuantFusionPass` | attention output followed by FP8 / NVFP4 quant |
| `fuse_attn_quant` | `MLAAttnQuantFusionPass` | MLA attention output followed by FP8 / NVFP4 / FP8 group quant |
| `enable_qk_norm_rope_fusion` | `QKNormRoPEFusionPass` | Q/K RMSNorm plus RoPE on packed QKV tensors |

## Pattern Inventory

| Source file | Pattern classes | Trace clue | Replacement |
| --- | --- | --- | --- |
| `fusion/allreduce_rms_fusion.py` | `AllReduceRMSNormPattern`, `AllReduceFusedAddRMSNormPattern`, `AllReduceFusedRMSNormStaticQuantFP8Pattern`, `AllReduceFusedAddRMSNormStaticQuantFP8Pattern`, `AllReduceFusedRMSNormStaticQuantNVFP4Pattern`, `AllReduceFusedAddRMSNormStaticQuantNVFP4Pattern` | TP all-reduce directly before RMSNorm, residual-add RMSNorm, or quant | `flashinfer_trtllm_fused_allreduce_norm` with FlashInfer allreduce fusion pattern codes |
| `fusion/rms_quant_fusion.py` | `RMSNormStaticQuantPattern`, `FusedAddRMSNormStaticQuantPattern`, `RMSNormDynamicQuantPattern`, `FusedAddRMSNormDynamicQuantPattern`, `RMSNormGroupQuantPattern`, `FusedAddRMSNormGroupQuantPattern` | RMSNorm or fused-add-RMSNorm followed by static FP8, dynamic per-token FP8, FP8 group quant, or NVFP4 quant | `_C.rms_norm_*_quant`, `_C.fused_add_rms_norm_*_quant`, or per-block quant custom op |
| `fusion/rocm_aiter_fusion.py` | `AiterRMSNormDynamicQuantPattern`, `AiterFusedAddRMSNormDynamicQuantPattern`, `AiterRMSFp8GroupQuantPattern`, `AiterFusedAddRMSFp8GroupQuantPattern` | AITER RMSNorm/fused-add-RMSNorm followed by AITER or vLLM FP8 quant | AITER fused RMSNorm-quant custom ops |
| `fusion/act_quant_fusion.py` | `SiluMulFp8StaticQuantPattern`, `SiluMulNvfp4QuantPattern`, `SiluMulBlockQuantPattern` | SiLU-and-mul activation output immediately quantized | fused activation-plus-quant custom op |
| `fusion/rocm_aiter_fusion.py` | `AiterSiluMulFp8GroupQuantPattern` | AITER SiLU-and-mul followed by FP8 group quant | AITER `act_mul_fused_fp8_group_quant` |
| `fusion/rocm_aiter_fusion.py` | `AddAiterRMSNormPadPattern` | AITER fused-add-RMSNorm output padded before the next op | AITER add-RMSNorm-pad op |
| `fusion/rocm_aiter_fusion.py` | `MLADualRMSNormPattern` | MLA Q branch and KV branch each run RMSNorm | `torch.ops.vllm.fused_mla_dual_rms_norm` backed by AITER fused QK RMSNorm |
| `fusion/qk_norm_rope_fusion.py` | `QkNormRopePattern` | Q/K RMSNorm, split/getitem reshapes, then RoPE | `_C.fused_qk_norm_rope` |
| `fusion/rope_kvcache_fusion.py` | `RopeReshapeKVCachePattern` | RoPE output followed by reshape/cache update | `vllm.fused_rope_and_unified_kv_cache_update` |
| `fusion/attn_quant_fusion.py` | `AttnFp8StaticQuantPattern`, `AttnNvfp4QuantPattern` | attention output followed by FP8 static quant or NVFP4 quant | backend attention op with fused output quant when supported |
| `fusion/mla_attn_quant_fusion.py` | `MLAAttnFp8StaticQuantPattern`, `MLAAttnNvfp4QuantPattern`, `MLAAttnFp8GroupQuantPattern` | MLA attention output followed by static FP8, NVFP4, or FP8 group quant | MLA attention op with fused output quant when supported |
| `fusion/minimax_qk_norm_fusion.py` | `MiniMaxQKNormPattern` | MiniMax `forward_qk`: Q/K variance all-reduce divided by TP world size, then RMS apply | `vllm.minimax_qk_norm_fused` / Lamport fused kernel |
| `fusion/sequence_parallelism.py` | `FirstAllReduceRMSNormPattern`, `MiddleAllReduceRMSNormPattern`, `FirstAllReduceRMSNormStaticFP8Pattern`, `MiddleAllReduceRMSNormStaticFP8Pattern` | all-reduce plus norm block in a full-graph TP model | sequence-parallel reduce-scatter, local norm, all-gather staging |
| `fusion/collective_fusion.py` | `GEMMReduceScatterPattern`, `AllGatherGEMMPattern`, `ScaledMMReduceScatterPattern`, `AllGatherScaledMMPattern`, `CutlassScaledMMReduceScatterPattern`, `AllGatherCutlassScaledMMPattern`, `FlashInferBMMFP8ReduceScatterPattern`, `FlashInferAllGatherBMMFP8Pattern` | matmul / scaled-mm / FlashInfer BMM adjacent to TP collectives | symmetric-memory fused matmul+reduce-scatter or all-gather+matmul |

## Triage Rules

- If the trace shows split norm/add/quant, compare first against
  `RMSNormQuantFusionPass`, AITER variants, and `AllReduceFusionPass`.
- If the trace shows attention output followed by quant kernels, compare against
  `AttnQuantFusionPass` or `MLAAttnQuantFusionPass`, not only handwritten
  attention kernels.
- If the trace shows Q/K norm followed by RoPE or cache update, compare both
  `QKNormRoPEFusionPass` and `RopeKVCacheFusionPass`; they are separate passes.
- If the trace is a TP decode trace with visible collectives, check whether
  `enable_sp` and `fuse_gemm_comms` would transform the same region into
  sequence-parallel or AsyncTP overlap.
- A missing vLLM compile fusion may be intentional when the graph range, backend
  support check, dtype, token count, or AITER / FlashInfer availability does not
  satisfy the pass-specific guard.
