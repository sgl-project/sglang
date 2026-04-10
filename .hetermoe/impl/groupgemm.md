specifically, the two groupgemms to use are fused moe (groupgemm) kernels for a8w8 and a16w4 (fused marlin)

implementation in the system:
    assume the input is already sorted and dispatched correctly for each groupgemm
    a8w8 has a quantization step for activation, you should be able to handle that elegantly
        if the input for the groupgemm is bf16, then no extra work is required
        if the input for the groupgemm is int8, then we need to quantize the weights accordingly
            if the a8w8 uses techniques like AWQ that requires scaling some columns, mark this work and remind me to impl later
    then just call these kernels

the reference (in reference.md) includes one mixed groupgemm usage of bf16 and nvfp4 with different tensor cores

---
## concrete sglang kernel call patterns (added during step 0.1 refinement)

### a16w4 cold-expert path (Marlin INT4)
file: python/sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py
function: fused_marlin_moe(
    hidden_states,          # [num_tokens, hidden_size], dtype=bf16
    w1=w13_qweight,         # [num_local_experts, 2*intermediate_size, hidden_size//8], dtype=uint8 (packed INT4)
    w2=w2_qweight,          # [num_local_experts, hidden_size, intermediate_size//8], dtype=uint8
    w1_scale=w13_scales,    # [num_local_experts, 2*intermediate_size, hidden_size//group_size], dtype=bf16
    w2_scale=w2_scales,     # [num_local_experts, hidden_size, intermediate_size//group_size], dtype=bf16
    gating_output,          # [num_tokens, num_experts] (router logits, before softmax)
    topk_weights,           # [num_tokens, top_k]
    topk_ids,               # [num_tokens, top_k]
    w1_zeros=...,           # optional, for AWQ zero-points
    w2_zeros=...,           # optional
    num_bits=4,
    is_k_full=True,
)
no activation quantization needed — activations stay bf16.

### a16w16 hot-expert path (BF16 Triton)
file: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py
entry: invoke_fused_moe_kernel() or via MoeRunner(MoeRunnerBackend.TRITON, ...).run(dispatch_output, quant_info)
    quant_info = TritonMoeQuantInfo(w13_weight=..., w2_weight=..., use_fp8_w8a8=False, use_int8_w8a8=False)
no activation quantization needed.

### a8w8 hot-expert path (INT8 Triton) — optional extension
file: python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py (SAME kernel as a16w16)
entry: invoke_fused_moe_kernel(use_int8_w8a8=True) via MoeRunner(MoeRunnerBackend.TRITON, ...)
    this is a TRUE fused group-GEMM — same fused_moe_kernel as BF16, with INT8 as a tl.constexpr flag.
    NOT per-expert int8_scaled_mm calls (that's dense-only).
call pattern:
    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,    # [E, 2*I, H], dtype=int8
        w2_weight=layer.w2_weight,      # [E, H, I], dtype=int8
        use_int8_w8a8=True,
        per_channel_quant=True,
        w13_scale=layer.w13_weight_scale,   # [E, 2*I, 1], dtype=float32
        w2_scale=layer.w2_weight_scale,     # [E, H, 1], dtype=float32
    )
    runner.run(dispatch_output, quant_info)
activation quantization: per_token_quant_int8(A) called inside invoke_fused_moe_kernel (L757)
    → produces (A_int8, A_scale) before launching the fused Triton kernel
    → inside kernel: accumulator += tl.dot(a, b); accumulator *= a_scale * b_scale

execution trace:
    W8A8Int8MoEMethod.apply()           → w8a8_int8.py:381
      TritonRunnerCore.run()            → triton.py:184 (w13), triton.py:261 (w2)
        invoke_fused_moe_kernel()       → fused_moe_triton_kernels.py:750 (quant), L860 (kernel launch)
          per_token_quant_int8(A)       → quantization/int8_kernel.py
          fused_moe_kernel[grid](...)   → fused_moe_triton_kernels.py:324 (single Triton kernel, all experts)

### key constraint for multi-group forward
    each group's kernel call expects full-sized topk_ids/topk_weights tensors
    for non-group experts: zero out their topk_weights so the kernel skips them
    OR: remap expert IDs to a local index space within each group's weight tensor
    the TRT-LLM reference uses the "zero scales + remap" approach — we should follow suit