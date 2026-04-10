we need to profile on both kernels with varying batch sizes
assume to use qwen3 30b a3b, use the expert's linear weight size and group size

we need two kinds of profiling:
    1. just test the kernel with uniform token distribution across experts 
    2. use routing information (less priority)

the ultimate goal of this profile is to show the (batch size) knee of memory bound and compute bound of different configs {a8w8, a16w4, a16w16}
you should create necessary plots for demostration

---
## concrete dimensions and profiling plan (added during step 0.1 refinement)

### Qwen3-30B-A3B expert GEMM dimensions
    gate_proj (w1):  M=num_tokens_per_expert, K=2048, N=768
    up_proj (w3):    M=num_tokens_per_expert, K=2048, N=768
    down_proj (w2):  M=num_tokens_per_expert, K=768,  N=2048
    fused w13:       M=num_tokens_per_expert, K=2048, N=1536  (gate+up fused)

### batch sizes to sweep (tokens per expert)
    M = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    this covers decode (M=1..8) through large-batch prefill (M=256..1024)

### kernels to profile
    1. a16w16: invoke_fused_moe_kernel with bf16 weights (Triton)
    2. a16w4:  fused_marlin_moe with GPTQ-INT4 packed weights (JIT Marlin, num_bits=4)
    3. a8w8:   Triton runner with use_int8_w8a8=True (INT8 tensor cores)

### profiling methodology
    - use torch.cuda.Event for timing (start/end events with synchronize)
    - warm-up: 10 iterations before measurement
    - measurement: 20 iterations, report median
    - working set: allocate fresh random tensors to avoid L2 caching artifacts
    - MOE instead of running single expert by expert
    - plot: x-axis = tokens_per_expert (log scale), y-axis = throughput (TFLOPS) or latency (us)
    - expected knee: a16w4 wins at small M (memory-bound), a16w16/a8w8 wins at large M (compute-bound)

### output
    save to /data/heter-moe/profiles/groupgemm/
    CSV: columns = [tokens_per_expert, kernel, latency_us, throughput_tflops]
    plots: groupgemm_latency.png, groupgemm_throughput.png
