# Blackwell Distributed GEMM

This example implements Tensor Parallel GEMMs for the Hopper architecture with the experimental
[Distributed GEMM](../../include/cutlass/experimental/distributed) API in CUTLASS.

This example requires Blackwell GPUs with an any-to-any NVLink network.
Please refer to [REQUIREMENTS.md](REQUIREMENTS.md) for more information.

By default, the example assumes 8 GPUs (TP=8) and runs an All Gather + GEMM operation, which rotates
operand A. To run with a different number of GPUs or schedule, please refer to
[82_blackwell_distributed_gemm.cu](82_blackwell_distributed_gemm.cu).


## Getting started

Command line arguments are mostly similar to other examples:

```
--m=<int>                   Sets the M extent of the GEMM
--n=<int>                   Sets the N extent of the GEMM
--k=<int>                   Sets the K extent of the GEMM
--l=<int>                   Sets the L extent (batch) of the GEMM (default: 1)
--alpha=<f32>               Epilogue scalar alpha (default: 1.0)
--beta=<f32>                Epilogue scalar beta (default: 0.0)
--iterations=<int>          Number of profiling iterations to perform (default: 100)
--warmup-iterations=<int>   Number of warmup iterations prior to profiling (default: 10)
--eps=<f32>                 Threshold for error compared to reference GEMM (default: 0.0)
```

Sample run command:

```bash
./82_blackwell_distributed_gemm --m=16384 --n=106496 --k=16384 --warmup-iterations=10 --iterations=100
```

This example follows the [Hopper example](../65_distributed_gemm/) very closely, and only differs in the base GEMM kernel. For
more information you can refer to [that example](../65_distributed_gemm/README.md).
