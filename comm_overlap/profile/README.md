# Profiling Comm-overlap Strategie

## Install

Please follow: [TransformerEngine](https://github.com/NVIDIA/TransformerEngine/tree/main) and [Flux](https://github.com/bytedance/flux) prerequsite for installation.

### TransformerEngine
For a quick pre-knwoledge, [TransformerEngine](https://github.com/NVIDIA/TransformerEngine/tree/main) is a library for accelerating Transformer Model on NVIDIA GPUs. The official document and API cookbook is [here](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html).

Useful code:
* [Distributed Test](https://github.com/NVIDIA/TransformerEngine/tree/main/tests/pytorch/distributed) is a good reference for a quick understanding the TE kernels for gemm, communication overlapping, and common TE APIs. (especially [test_comm_gemm_overlap.py](https://github.com/NVIDIA/TransformerEngine/blob/main/tests/pytorch/distributed/test_comm_gemm_overlap.py))
* [Comm-overlap Example](https://github.com/NVIDIA/TransformerEngine/tree/main/examples/pytorch/comm_gemm_overlap) is an introduction for NVIDIA's example for training a model within a node appling TP (+DP)
* Noted that for best performance, P2P communication via CUDA Multicast needs CUDA Toolkit 12.0+ and CUDA driver 535+ on devices with compute capability 9.0 or newer.

### Flux
[Flux](https://github.com/bytedance/flux) is a fast communication-overlapping library for tensor parallelism on GPUs.

Useful code:
* [Test](https://github.com/bytedance/flux/tree/main/test) provides some scripts for evaluting kernel-wise functionality and performance with native pytorch implementaion. It also provide cross-node with [nvshmem](https://docs.nvidia.com/nvshmem/index.html).
* [gemm_reduce_scatter.cc
](https://github.com/bytedance/flux/blob/main/src/reduce_scatter/ths_op/gemm_reduce_scatter.cc) and [all_gather_gemm_kernel.cc](https://github.com/bytedance/flux/blob/main/src/all_gather/ths_op/all_gather_gemm_kernel.cc) is their core op implementation.
* Noted that Flux now can only support GPUs with compute capacity: `arch_num == 80 || arch_num == 89 || arch_num == 90`

## Run profiling
```
# for AG eval
./launch.sh test_ag_kernel <M size, eg. 1024> <N, eg. 12288> <K, eg. 49152> --dtype=float16 --iters=10

# for RS eval
./launch.sh test_gemm_rs.py <M size, eg. 1024> <N, eg. 49152> <K, eg. 12288> --dtype=float16 --iters=10
```
