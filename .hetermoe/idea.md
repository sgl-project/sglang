this is an engineering project that aims to explore the use of mixed precision for experts during batched online serving.

assigning different precisions to different experts have different benefits:
    1. a16w16 is the vanilla and provides best precision
    2. a16w4 is has the lowest memory footprint (in terms of HBM bandwidth)
    3. a8w8 has the fastest compute benefiting from the int8 tensor core

we observe load imbalancing in moe serving and the fluctuation of load imbalancing throughout serving of different/same requests
our aim is to improve the efficiency by switching expert precisions in this principle:
    1. cold experts are memory bounded, so a16w4 could save memory bandwidth and provide speedup
    2. hot experts are compute bounded, so a8w8 could save compute and provide speedup
    2.* caveat: we should also support a16w16 as compute bound scheme. in fact this should be our primary support; a8w8 as an option
    3. in one groupgemm, mixing cold and hot expert could potentially lead to speedup as well, but not our primary focus
we stick to the optimization method 1 and 2, and simply implement two groupgemms for each precision

as contrary to prior work where expert precision assignment is either fixed or only updated infrequently, our goal is to:
    1. upon each single batch, adapt the precision based on token count per expert
        (this is because experts classified as hot expert might be cold in the next iteration)
    2. assume colocation of memory weights of different precisions in VRAM, and we only focus on speeding up serving

---
## sglang kernel mapping (added during step 0.1 refinement)

| scheme   | sglang kernel                        | file                                                                        | notes                                                    |
|----------|--------------------------------------|-----------------------------------------------------------------------------|----------------------------------------------------------|
| a16w16   | fused_moe_kernel (Triton)            | python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py  | BF16 activations, BF16 weights; true fused group-GEMM    |
| a16w4    | fused_marlin_moe (JIT Marlin)        | python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py          | BF16 activations, INT4 packed weights; GPTQ/AWQ; separate kernel |
| a8w8     | fused_moe_kernel (Triton, use_int8_w8a8=True) | python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py | TRUE fused group-GEMM (same kernel as BF16, INT8 flag); per_token_quant_int8() for activation |

NOTE: a16w16 and a8w8 share the SAME Triton fused_moe_kernel — INT8 is a compile-time flag (tl.constexpr).
    a16w4 uses a completely DIFFERENT kernel (fused_marlin_moe with JIT Marlin).
    int8_scaled_mm from sgl-kernel is for DENSE linear layers only, NOT used in MoE path.

primary scheme pair for v1: {a16w16, a16w4} — simplest, no activation quantization needed for either
optional extension:         {a8w8, a16w4}   — requires per-token INT8 activation quantization for hot path
