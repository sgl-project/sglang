# Kimi-K3 kernel benchmarks

## DEP16 KDA projection prologue

`bench_kda_dep16_projections.py` measures the current unfused BF16 projection
chain used by Kimi-K3 KDA layers under full DP attention (`tp_size=dp_size=16`,
so `attn_tp_size=1`):

```text
qkv:    [M, 7168] x [7168, 36864]
beta:   [M, 7168] x [7168,    96]
f_a:    [M, 7168] x [7168,   128]
f_b:    [M,  128] x [ 128, 12288]
gate:   [M, 7168] x [7168, 12288]
```

The benchmark captures the complete five-GEMM chain in a CUDA graph. It can
capture multiple independent weight sets and divides the measured replay time
by that rotation count. Rotating weights avoids measuring an unrealistically
warm weight cache.

At the serving campaign's global decode batch of 512, DEP16 normally has 32
local tokens per attention-DP rank, so `M=32` is the primary baseline point.
A useful sweep is:

```bash
python3 benchmark/kernels/kimi_k3/bench_kda_dep16_projections.py \
  --m 1,2,4,8,16,32,64,128 \
  --rotations 2 --warmup-replays 10 \
  --batches 7 --replays-per-batch 40 \
  --output baseline.json
```

For a narrow Nsight Systems capture of ten `M=32` graph replays:

```bash
nsys profile \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --output=dep16-projections-m32 \
  python3 benchmark/kernels/kimi_k3/bench_kda_dep16_projections.py \
    --m 32 --rotations 2 \
    --warmup-replays 2 --batches 1 --replays-per-batch 2 \
    --nvtx-replays 10
```

This is a focused GPU-step benchmark. Any production change still requires a
matched serving and Nsight A/B with the real checkpoint.

### Baseline

Job `517390` ran on an NVIDIA GB300 (SM103), CUDA 13.0, PyTorch
`2.11.0+cu130`, from source revision `1ef7882a5b76` using two weight rotations.
CUDA-graph replay medians were:

| Local tokens (M) | Unfused chain |
|---:|---:|
| 1 | 123.769 us |
| 2 | 124.242 us |
| 4 | 124.535 us |
| 8 | 124.848 us |
| 16 | 125.365 us |
| **32** | **129.364 us** |
| 64 | 127.372 us |
| 128 | 129.783 us |

The ten-replay M=32 Nsight capture contained, per chain, one wide QKV GEMM,
one small-K `f_b` GEMM, three split-K GEMMs, and three split-K reducers. This
confirms that the microbenchmark reproduces the serving kernel pattern targeted
by the fusion work.
