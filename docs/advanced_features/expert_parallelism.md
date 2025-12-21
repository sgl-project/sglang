# Expert Parallelism

Expert Parallelism (EP) in SGLang distributes expert weights across multiple devices in Mixture-of-Experts (MoE) models, addressing memory bottlenecks and enabling efficient scaling for high-performance inference. It is particularly vital for serving large-scale MoE models where tokens are dynamically routed to specialized experts across GPUs. By leveraging optimized all-to-all communication and grouped matrix multiplications (GEMMs), EP reduces latency, boosts throughput, and minimizes idle GPU time. SGLang's EP offers strong extensibility through its modular framework, allowing seamless integration of custom kernels, backends, and optimizations without refactoring core logic, supporting diverse hardware and quantization schemes.

## Supported Backends and Selection Guidance

SGLang's EP integrates diverse, highly efficient backends for different use cases, allowing fine-grained control over performance trade-offs. Users specify backends via command-line flags:
- `--moe-a2a-backend`: Selects the backend for all-to-all communication.
- `--moe-runner-backend`: Selects the backend for MoE computation.

### Backends for All-to-All Communication

| Backend      | Description                                                                 | Use Cases                          |
|--------------|-----------------------------------------------------------------------------|------------------------------------|
| **`none` (default)** | Disables all-to-all for EP. Uses All-Reduce or All-Gather for token dispatch. | Hybrid EP and TP setups.           |
| `deepep`     | DeepEP, a communication library for efficient token shuffling in MoE models. | Large-scale EP deployments.        |
| `mooncake`   | An extension of DeepEP for elastic inference, leveraging RDMA for high-performance data transfers. | Elastic EP serving. |

DeepEP and Mooncake backends support two modes for token dispatch: `normal` mode (optimized for prefill workloads with high throughput) and `low_latency` mode (optimized for decode workloads with low latency and CUDA Graph compatibility). Users are recommended to set `--deepep-mode auto` to enable automatic dispatch mode switching during runtime. Setting `--deepep-mode normal` or `--deepep-mode low_latency` is useful for debugging or development purposes.

Currently, DeepEP and Mooncake only support cases where `ep_size = tp_size`. For hybrid EP and TP (i.e., `ep_size < tp_size`), only the `none` backend (All-Reduce or All-Gather-based dispatching) is supported.

### Backends for MoE Computation

| Backend                  | Description                                                                 | Use Cases                          |
|--------------------------|-----------------------------------------------------------------------------|------------------------------------|
| **`auto` (default)**     | Automatically selects the optimal backend based on model architecture, hardware (e.g., NVIDIA architecture like Ampere, Hopper, Blackwell), quantization scheme (e.g., FP8, FP4), and runtime conditions. | General-purpose deployments; ensures compatibility and performance without user intervention. |
| `triton`                 | Triton-based implementation for grouped GEMMs. To achieve higher performance, it's highly recommended to create [tuned configurations](https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/fused_moe_triton/README.md). | Custom kernel development or scenarios requiring high extensibility with Torch compilation support. |
| `deep_gemm`              | DeepGEMM backend optimized for MoE matrix multiplications, supporting contiguous layouts for prefill and masked layouts for decode; often JIT-compiled for performance. | Large-scale EP deployments with FP8 block-wise quantization. |
| `cutlass`                | CUTLASS-based backend for efficient GEMMs. | NVIDIA architectures with CUTLASS support. |
| `flashinfer_trtllm`      | FlashInfer integrated with TensorRT-LLM for accelerated MoE computations, supporting FP4 communication operators and high-performance GEMMs. | Blackwell with TRT-LLM. |
| `flashinfer_cutlass`     | FlashInfer combined with CUTLASS for high-performance grouped GEMMs in MoE layers, handling FP4/FP8 quantization efficiently. | Blackwell with FP4/FP8 models. |
| `flashinfer_mxfp4`       | FlashInfer variant optimized for MXFP4 (mixed FP4) quantization in MoE runners, focusing on memory-efficient low-precision inference. | Low-precision models with MXFP4. |
| `flashinfer_cutedsl`     | FlashInfer with a custom DSL for flexible and efficient MoE kernel generation, integrated with ModelOpt FP4 quantization. | Low-precision models with NVFP4. |

### Examples

Launch with DeepEP and DeepGEMM for DeepSeek-V3:

```bash
python -m sglang.launch_server --model-path deepseek-ai/DeepSeek-V3 --moe-a2a-backend deepep --moe-runner-backend deep_gemm --tp 8 --ep 8
```

## Extensible EP Framework

SGLang's EP framework provides modular abstractions for easy integration of custom kernels, backends, and optimizations. It decouples the MoE forward pass into stages (dispatch → pre-permute → core runner → post-permute → combine), enabling seamless extensions without refactoring core logic.

### Framework Overview

The framework centers on `FusedMoE` as the unified entry point for a single, extensible structure. Key components include:
- **Dispatcher**: Manages dispatch/combine for backends like DeepEP (implements `BaseDispatcher` subclasses).
- **MoeRunner**: Orchestrates grouped-GEMM execution via `MoeRunnerCore` implementations (e.g., `TritonRunnerCore`).
- **PermuteMethodPool**: Auto-registers layout conversions (e.g., pre/post-permute via `register_pre_permute` and `register_post_permute` for dynamic modes, or `register_fused_func` for static, torch.compile-compatible fused operations).
- **TopK Router**: Backend-agnostic expert selection.

This design supports multiple backends via `--moe-a2a-backend` and `--moe-runner-backend`, with quantization integrated through a standardized `apply()` method. The computation flow ensures modularity:

```
[input_hidden_states]
          |
          v
     TopK.forward -> select_experts / triton_kernels.routing / bypass
          |
          v
     [TopKOutput]
          |
          v
   FusedMoE.forward -> Dispatcher.dispatch -> DeepEP / bypass
          |                     |
          |                     v
          |              [DispatchOutput]
          |                     |
          |                     v
          |             quant_method.apply -> MoeRunner.forward
          |                     |              |
          |                     |              v
          |                     | pre-permute + grouped_gemm + post-permute
          |                     |              |
          |                     |--------------
          |                     v
          |               [CombineInput]
          |                     |
          |                     v
          |            Dispatcher.combine -> DeepEP / bypass
          |                     |
          |---------------------
          v
[final_hidden_states]
```

For details, see the [MoE Refactor Roadmap](https://github.com/sgl-project/sglang/issues/8715).

### Implementing New Backends

To add a new backend:
1. For a new all-to-all dispatcher, implement a `BaseDispatcher` subclass with `dispatch` and `combine` methods.
2. For a new MoE runner backend, define a `MoeRunnerCore` subclass for core operations (e.g., grouped GEMMs).
3. Define new input/output formats for the dispatcher or model runner (e.g., `RunnerInput`, `RunnerOutput`).
4. Register permute/unpermute methods to ensure compatibility:
   - **Fused Mode** (static, torch.compile-compatible): Use `register_fused_func` for end-to-end operations.
   - **Permute Mode** (dynamic): Register `register_pre_permute` and `register_post_permute` for flexible layouts.

See the [MoE Refactor Implementation PR](https://github.com/sgl-project/sglang/pull/9269) for full changes, including type hints and config expansions.

### Examples

For an example implementation, see [moe_runner/triton.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/moe_runner/triton.py), which demonstrates Triton-based grouped GEMMs with registered fused and permutation functions.

## Computation and Communication Overlap

SGLang's EP employs advanced overlap techniques to hide communication latency behind computation, maximizing GPU utilization in MoE layers.

### Two-Batch Overlap (TBO)

TBO splits requests into micro-batches, interleaving attention computation with dispatch/combine operations. Yield points in the execution graph allow pausing for overlaps, increasing overall throughput without peak memory spikes:

```python
operations = [
    self._forward_attn,
    YieldOperation(),  # Overlap with dispatch of prior micro-batch
    self._forward_dispatch,
    self._forward_mlp,
    YieldOperation(),  # Overlap with combine
    self._forward_combine,
]
```

Users need to specify `--enable-two-batch-overlap` to unlock up to 2x throughput. For details, see the [Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/#two-batch-overlap).

### Single-Batch Overlap (SBO)

SGLang introduces a dispatcher-hook system for Single-Batch Overlap (SBO), enabling the overlap of operations within a single batch—such as shared experts computation with communication—while decentralizing logic to enhance modularity. These hooks execute before and after the `dispatch` and `combine` operations without modifying core MoE modules. This design simplifies interfaces, reduces coupling, and improves extensibility. For implementation details and an example of overlapping shared experts with DeepEP's combine operation, refer to [PR #13327](https://github.com/sgl-project/sglang/pull/13327). Users can set `--enable-single-batch-overlap` to enable this feature.


## Workload Balancer

SGLang integrates the [Expert Parallelism Load Balancer (EPLB)](https://github.com/deepseek-ai/EPLB) from DeepSeek to address routing imbalances in MoE models. By analyzing expert activation statistics, EPLB computes an optimal expert arrangement, strategically placing or replicating experts to minimize GPU utilization variance, reduce idle cycles, and enhance scalability.

To enable EPLB, use the flags `--enable-eplb true --load-balance-method eplb`. For optimal performance, increase batch sizes to stabilize activation statistics and configure periodic rebalancing (e.g., every 1000 requests) to adapt to evolving workloads. Simulations demonstrate significant improvements in load balancedness (ratio of mean to max computation time), correlating strongly with throughput gains.

For more details, refer to the [EPLB Section in the Large-Scale EP Blog](https://lmsys.org/blog/2025-05-05-large-scale-ep/#expert-parallelism-load-balancer) and the [EPLB Repository](https://github.com/deepseek-ai/eplb).
