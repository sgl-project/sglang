## Benchmark FBGEMM Grouped GEMM

Benchmark FBGEMM Grouped GEMM in both Triton and CUDA version and SGLang Triton Grouped GEMM, it will be used to compare the bandwidth of different implementations.

### Requirements

```shell
pip install fbgemm-gpu-genai
```

### Usage

```bash
python3 benchmark/fbgemm/benchmark_fbgemm_grouped_gemm.py --model Qwen/Qwen2-57B-A14B-Instruct --tp-size 4 --use-fp8-w8a8
```

For example, in H200, the Qwen2-57B-A14B-Instruct TP4 fp8w8a8 grouped gemm bandwidth result is as follows:

```shell
grouped-gemm-performance:
   batch_size  FBGEMM Triton Grouped GEMM FP8  FBGEMM CUTLASS F8F8BF16 Rowwise  SGLang Grouped GEMM FP8
0       256.0                     3704.841339                      3042.626402              2254.725030
1       512.0                     3691.426346                      3029.065684              2269.504543
2      1024.0                     3653.938629                      2258.471467              2358.319020
3      2048.0                     3596.644313                      2271.611904              2476.895397
4      4096.0                     3468.496435                      2231.283986              2179.473910
```

The theoretical peak bandwidth of H200 is 4.8 TB/s. Taking batch_size 256 as an example, the bandwidth of FBGEMM Triton Grouped GEMM FP8 is 3704.841339 GB/s, the bandwidth of FBGEMM CUTLASS F8F8BF16 Rowwise is 3042.626402 GB/s, and the bandwidth of SGLang Grouped GEMM FP8 is 2254.725030 GB/s. Therefore, FBGEMM Triton Grouped GEMM FP8 achieves 77.9% of H200's theoretical peak bandwidth, FBGEMM CUTLASS F8F8BF16 Rowwise achieves 63.4% of H200's theoretical peak bandwidth, and SGLang Grouped GEMM FP8 achieves 46.9% of H200's theoretical peak bandwidth.
