# TileLang FP8 GEMM H20 and RTX 4090 Results

This directory contains the tuned TileLang FP8 GEMM configs and full benchmark
artifacts for the H20 and RTX 4090 runs used by the TileLang FP8 GEMM PR.

## Run Matrix

- Branch under test: `tilelang-fp8-gemm-modernization`
- Commit under test: `1a6b70949961ce6e79078b35daf2869c18381321`
- Benchmark backend: CUDA graph
- Benchmark repetitions: 100
- Shapes: 25 `(N, K)` pairs
- M values: 23 values
- Total rows per GPU: 575

## Summary

| GPU | Baseline | Rows | Allclose failures | TileLang faster | TileLang slower | Median speedup | Worst TileLang / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTX 4090 | Triton | 575 | 0 | 489 | 86 | 2.95x | 1.72x |
| H20 | Triton | 575 | 1 | 548 | 27 | 1.78x | 1.16x |
| H20 | DeepGEMM | 575 | 1 | 499 | 76 | 1.27x | 1.44x |

Notes:

- RTX 4090 does not include a DeepGEMM comparison because DeepGEMM is not
  available on SM89. DeepGEMM-related fields in the RTX 4090 summary are
  therefore recorded as `NaN`.
- The H20 run has one allclose failure shared by TileLang and DeepGEMM for
  `M=1, N=512, K=2048`, with `max_diff=0.5`.

## Configs

| GPU | Config | Notes |
|---|---|---|
| H20 | [tilelang_selected_configs.json](h20/tilelang_selected_configs.json) | Configs produced by the H20 tuning run. |
| H20 | [tilelang_benchmark_selected_configs.json](h20/tilelang_benchmark_selected_configs.json) | Configs loaded by the H20 benchmark run. |
| RTX 4090 | [tilelang_selected_configs.json](rtx4090/tilelang_selected_configs.json) | Configs used for the RTX 4090 benchmark run. |
| RTX 4090 | [tilelang_benchmark_selected_configs.json](rtx4090/tilelang_benchmark_selected_configs.json) | Configs loaded by the RTX 4090 benchmark run. |

## Full Benchmark Artifacts

| GPU | Summary | Full CSV | Full log | Slower-case details |
|---|---|---|---|---|
| H20 | [benchmark_summary.json](h20/benchmark_summary.json) | [tilelang_vs_tuned_triton_deepgemm.csv](h20/tilelang_vs_tuned_triton_deepgemm.csv) | [benchmark.log](h20/benchmark.log) | [vs Triton](h20/tilelang_slower_than_triton.csv), [vs DeepGEMM](h20/tilelang_slower_than_deepgemm.csv), [top 20 vs Triton](h20/top20_tilelang_slower_than_triton.csv), [top 20 vs DeepGEMM](h20/top20_tilelang_slower_than_deepgemm.csv) |
| RTX 4090 | [benchmark_summary.json](rtx4090/benchmark_summary.json) | [tilelang_vs_tuned_triton_deepgemm.csv](rtx4090/tilelang_vs_tuned_triton_deepgemm.csv) | [benchmark.log](rtx4090/benchmark.log) | [vs Triton](rtx4090/tilelang_slower_than_triton.csv), [top 20 vs Triton](rtx4090/top20_tilelang_slower_than_triton.csv) |

Additional H20 artifacts:

- [tune.log](h20/tune.log)
- [allclose_failures.csv](h20/allclose_failures.csv)
- [deepgemm_allclose_failures.csv](h20/deepgemm_allclose_failures.csv)
- [tune_elapsed.txt](h20/tune_elapsed.txt)
- [benchmark_elapsed.txt](h20/benchmark_elapsed.txt)
- [environment.txt](h20/environment.txt)
- [run_metadata.txt](h20/run_metadata.txt)
- [shapes.tsv](h20/shapes.tsv)
- [m_values.txt](h20/m_values.txt)

Additional RTX 4090 artifacts:

- [benchmark_elapsed.txt](rtx4090/benchmark_elapsed.txt)
- [environment.txt](rtx4090/environment.txt)
- [run_metadata.txt](rtx4090/run_metadata.txt)
- [shapes.tsv](rtx4090/shapes.tsv)
- [m_values.txt](rtx4090/m_values.txt)

## Source Artifact Locations

H20:

```text
/goosefsx/x-c60-2k48ac4x-proxy/data/linjunxian/InferScripts/sglang/myscripts/startup/tilelang_fp8_gemm_repro_outputs/20260615_093452
```

RTX 4090:

```text
/workspace/model/experiments/tilelang_fp8_gemm_repro/tilelang_fp8_gemm_repro_outputs/cu130_4090_10_no_deepgemm_20260629_061522
```
