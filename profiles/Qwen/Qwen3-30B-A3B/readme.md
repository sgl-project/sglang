# Qwen3-30B-A3B — Inductor Compilation Profile

## Setup

- **Model:** `Qwen/Qwen3-30B-A3B`
- **MoE backend:** auto (flashinfer\_trtllm)
- **Weights:** real (HuggingFace)
- **Dataset:** ShareGPT, output sequence length 8192
- **TP:** 1
- **Device:** GB200
- **SGLang commit:**: `cb8105fe282fc373b5baed63d5df38682418a373`
- **`sgl_kernel` version:**: `0.3.21` 
- **`torch` commit:**: `cb8105fe282fc373b5baed63d5df38682418a373` (version nightly `2.12`)

## Notes

- `inductor[moe]` total time is misleading: prefill still uses the `triton_kernel` MoE backend, not `flashinfer_trtllm`.
- `inductor[rope]` uses the fallback fused q,k norm + rope path compiled by Inductor. Inductor can fuse the KV-cache update into the rotary embedding graph, while standard SGLang must fire 2 separate kernels because the SWA KV-cache type prevents fusion. The `SWAKVPool` uses dual addressing — SWA layers write to `out_cache_loc_swa`, non-SWA layers to `out_cache_loc` — which the JIT rope kernel doesn't handle. Inductor compiles the pure-PyTorch `forward_native` path where this dual addressing is expressed as `index_put_` ops that get fused into the rope graph.
- `inductor[rope-rmsnorm]` does **not** use Inductor for the pre-attention q,k normalization — only the rotary embedding and the layer-level RMSNorm are compiled.
- **RMSNorm** is compiled with no dynamic shapes, so Inductor can specialize on the fixed decode batch sizes used by SGLang's CUDA graphs. This means efficient code with only slightly higher startup times.
- **RotaryEmbedding** is compiled with dynamic shapes due to the KV-cache update (`index_put_` with variable `cache_loc`), which limits Inductor's ability to specialize and adds overhead.
- `inductor[qvnormropekv-rmsnorm]` compiles the full `QKNormRope` region — q/k normalization, rotary embedding, and KV-cache write — as a single fused Inductor graph, alongside the layer-level RMSNorm. This is the most comprehensive compilation scope, but the larger graph with dynamic shapes adds overhead that only amortizes at high concurrency.
- `inductor[ropekv-rmsnorm]` compiles the rotary embedding with KV-cache write (fused into a single Inductor graph) and the layer-level RMSNorm. Q/k normalization uses the custom kernel. Same compilation scope as `inductor[rope-rmsnorm]` but with naming that makes the KV-cache fusion explicit.
- `inductor[qvnorm-ropekv-rmsnorm]` splits the `QKNormRope` region into two separate Inductor graphs: one for q/k normalization (no dynamic shapes) and one for rope + KV-cache write (dynamic shapes). This reduces the scope of the dynamic-shape graph compared to the fully-fused `qvnormropekv` variant.

## bench\_one\_batch Speedup Charts

The charts below were generated with `bench_one_batch.py`, which measures raw single-batch latency and throughput at various batch sizes (1, 4, 16, 32, 64) with input length 1024 and output length 8192. The baseline is `inductor[None]` (no Inductor compilation).

```bash
python profiles/plot_speedup.py profiles/Qwen/Qwen3-30B-A3B
```

![Speedup Charts](speedup_charts.png)

**Key observations:**
- `inductor[moe]` significantly hurts decode throughput (~0.70x at bs=1), likely due to the triton\_kernel prefill overhead bleeding into measured totals.
- `inductor[rmsnorm]`, `inductor[rope-rmsnorm]`, and `inductor[rope]` are all roughly at parity with the baseline for decode throughput (1.00–1.02x).
- Overall throughput shows similar trends: the non-moe configs hover around 1.00x, while `inductor[moe]` drags overall throughput to ~0.65x at bs=1.

## bench\_offline\_throughput (Real Engine)

These benchmarks use `bench_offline_throughput.py`, which runs the full SGLang engine (scheduler, radix cache, continuous batching) to better reflect production serving performance.

**Note:** Piecewise CUDA graphs are automatically disabled for this model due to the `flashinfer_trtllm` MoE backend, so baseline and Inductor configs run under the same conditions.

```bash
python3 -m sglang.bench_offline_throughput \
  --model-path Qwen/Qwen3-30B-A3B \
  --trust-remote-code \
  --cuda-graph-bs <cg-bs> \
  --tp-size 1 \
  --sharegpt-output-len 8192 \
  --num-prompts <N> \
  --dataset-name sharegpt \
  --result-filename "" \
  [--enable-torch-compile --torch-compile-override-layers <layers> --torch-compile-scope local]
```

### 1 prompt, cuda-graph-bs 1

| Config | Output tok/s | Total tok/s | Total tok/s vs Baseline |
|--------|-------------|-------------|------------------------|
| Baseline | 358 | 359 | — |
| Inductor — QKNormRopeKV + RMSNorm | 338 | 339 | −5.6% |
| Inductor — QKNorm + RopeKV + RMSNorm | 351 | 351 | −2.0% |
| Inductor — RotaryEmbedding + RMSNorm | 348 | 348 | −2.8% |
| Inductor — RotaryEmbedding | 360 | 361 | **+0.5%** |
| Inductor — RMSNorm | 346 | 347 | −3.3% |

### 32 prompts, cuda-graph-bs 32

| Config | Output tok/s | Total tok/s | Total tok/s vs Baseline |
|--------|-------------|-------------|------------------------|
| Baseline | 3,319 | 3,447 | — |
| Inductor — QKNormRopeKV + RMSNorm | 3,297 | 3,424 | −0.7% |
| Inductor — QKNorm + RopeKV + RMSNorm | 3,377 | 3,507 | **+1.7%** |
| Inductor — RotaryEmbedding + RMSNorm | 3,427 | 3,558 | **+3.2%** |
| Inductor — RotaryEmbedding | 3,401 | 3,532 | **+2.5%** |
| Inductor — RMSNorm | 3,368 | 3,497 | **+1.5%** |

### 128 prompts, cuda-graph-bs 128

| Config | Output tok/s | Total tok/s | Total tok/s vs Baseline |
|--------|-------------|-------------|------------------------|
| Baseline | 6,963 | 7,280 | — |
| Inductor — QKNormRopeKV + RMSNorm | 7,019 | 7,338 | **+0.8%** |
| Inductor — QKNorm + RopeKV + RMSNorm | 7,111 | 7,435 | **+2.1%** |
| Inductor — RotaryEmbedding + RMSNorm | 7,048 | 7,368 | **+1.2%** |
| Inductor — RotaryEmbedding | 7,030 | 7,350 | **+1.0%** |
| Inductor — RMSNorm | 6,958 | 7,274 | −0.1% |

### 256 prompts, cuda-graph-bs 256

| Config | Output tok/s | Total tok/s | Total tok/s vs Baseline |
|--------|-------------|-------------|------------------------|
| Baseline | 7,316 | 7,589 | — |
| Inductor — RopeKV + RMSNorm | 7,366 | 7,640 | **+0.7%** |
| Inductor — QKNorm + RopeKV + RMSNorm | 7,353 | 7,627 | **+0.5%** |

### 512 prompts, cuda-graph-bs 512

| Config | Output tok/s | Total tok/s | Total tok/s vs Baseline |
|--------|-------------|-------------|------------------------|
| Baseline | 7,450 | 7,736 | — |
| Inductor — QKNorm + RopeKV + RMSNorm | 7,566 | 7,857 | **+1.6%** |
| Inductor — RopeKV + RMSNorm | 7,521 | 7,810 | **+1.0%** |

### Summary

| Scenario | Config | Total tok/s | Total tok/s vs Baseline |
|----------|--------|-------------|------------------------|
| 1 prompt, cg-bs 1 | RotaryEmbedding | 361 | **+0.5%** |
| 1 prompt, cg-bs 1 | QKNorm + RopeKV + RMSNorm | 351 | −2.0% |
| 1 prompt, cg-bs 1 | RotaryEmbedding + RMSNorm | 348 | −2.8% |
| 1 prompt, cg-bs 1 | RMSNorm | 347 | −3.3% |
| 1 prompt, cg-bs 1 | QKNormRopeKV + RMSNorm | 339 | −5.6% |
| 32 prompts, cg-bs 32 | RotaryEmbedding + RMSNorm | 3,558 | **+3.2%** |
| 32 prompts, cg-bs 32 | RotaryEmbedding | 3,532 | **+2.5%** |
| 32 prompts, cg-bs 32 | QKNorm + RopeKV + RMSNorm | 3,507 | **+1.7%** |
| 32 prompts, cg-bs 32 | RMSNorm | 3,497 | **+1.5%** |
| 32 prompts, cg-bs 32 | QKNormRopeKV + RMSNorm | 3,424 | −0.7% |
| 128 prompts, cg-bs 128 | QKNorm + RopeKV + RMSNorm | 7,435 | **+2.1%** |
| 128 prompts, cg-bs 128 | RotaryEmbedding + RMSNorm | 7,368 | **+1.2%** |
| 128 prompts, cg-bs 128 | RotaryEmbedding | 7,350 | **+1.0%** |
| 128 prompts, cg-bs 128 | QKNormRopeKV + RMSNorm | 7,338 | **+0.8%** |
| 128 prompts, cg-bs 128 | RMSNorm | 7,274 | −0.1% |
| 256 prompts, cg-bs 256 | RopeKV + RMSNorm | 7,640 | **+0.7%** |
| 256 prompts, cg-bs 256 | QKNorm + RopeKV + RMSNorm | 7,627 | **+0.5%** |
| 512 prompts, cg-bs 512 | QKNorm + RopeKV + RMSNorm | 7,857 | **+1.6%** |
| 512 prompts, cg-bs 512 | RopeKV + RMSNorm | 7,810 | **+1.0%** |

At medium concurrency (B=32), the smaller-scope configs deliver clear gains: `RotaryEmbedding + RMSNorm` leads at **+3.2%** (3,447 → 3,558 tok/s), followed by `RotaryEmbedding` at **+2.5%**, `QKNorm + RopeKV + RMSNorm` at **+1.7%**, and `RMSNorm` at **+1.5%**. At high concurrency (B=128), `QKNorm + RopeKV + RMSNorm` leads at **+2.1%** (7,280 → 7,435 tok/s), followed by `RotaryEmbedding + RMSNorm` at **+1.2%**, `RotaryEmbedding` at **+1.0%**, and `QKNormRopeKV + RMSNorm` at **+0.8%**.

At B=256 both Inductor configs show modest gains: `RopeKV + RMSNorm` at **+0.7%** and `QKNorm + RopeKV + RMSNorm` at **+0.5%**. At B=512 the gains increase again, with `QKNorm + RopeKV + RMSNorm` leading at **+1.6%** (7,736 → 7,857 tok/s) and `RopeKV + RMSNorm` at **+1.0%**. The crossover between B=256 and B=512 — where `QKNorm + RopeKV + RMSNorm` overtakes `RopeKV + RMSNorm` — is consistent with the B=128 results and confirms that the broader compilation scope amortizes better at higher concurrency.

Splitting q/k normalization into a separate static-shape graph (`QKNorm + RopeKV + RMSNorm`) substantially improves over the fully-fused variant (`QKNormRopeKV + RMSNorm`): at B=1 the regression drops from −5.6% to −2.0%, at B=32 it flips from −0.7% to **+1.7%**, and at B=128+ it consistently leads. Nsys profiling confirms the mechanism: the fused Inductor graph reduces the number of kernel launches between the projection GEMM and the attention kernel by collapsing q/k normalization, rotary embedding, and KV-cache store into fewer kernels, shrinking the gap at higher batch sizes where launch overhead is more visible.

| Baseline | Inductor |
|---|---|
| ![Baseline nsys](qwen3-30B-A3B-[qvnorm-rope-kv]-base.png) | ![Inductor nsys](qwen3-30B-A3B-[qvnorm-rope-kv]-inductor.png) |

The baseline fires 3 kernels between the projection GEMM and `fmhaSm100fKernel`: one for fused q/k normalization, and two for rope + KV-cache store (split by an intermediate `contiguous` call). Inductor compiles this region into only 2 kernels, eliminating the extra launch.

`QKNorm + RopeKV + RMSNorm` is the best config at B=128 (+2.1%) and B=512 (+1.6%), and competitive at B=32, making it the recommended config for high-throughput serving. `RopeKV + RMSNorm` is a simpler alternative that leads at B=256 (+0.7%) and is close behind at B=512 (+1.0%). `RotaryEmbedding + RMSNorm` remains the best at B=32 and a strong all-round choice.