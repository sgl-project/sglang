<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.ai to discuss further. -->

## Motivation

This PR enables config-driven, parameterized activations for MoE across Triton and Cutlass (bf16/fp8/fp4) while preserving full backward compatibility and performance fast paths. It introduces a single, unified activation entry point (register + get in one place) and a generic Triton GLU activation kernel that supports swish/swiglu/reglu/geglu (with alpha/limit/up_shift), while SiLU/GELU still use existing fused sgl-kernel ops. Activation type and parameters are read from config.json using getattr-style defaults (e.g., gpt-oss keeps swiglu with alpha=1.702 by default and honors existing swiglu_limit=7) so no changes to official configs are required. The implementation adds no new files, minimizes conditionals via name→builder mapping and compile-time “switch/case” mode branching in Triton, and wires the same entry to Cutlass paths.

## Modifications

# What I changed (first-person, detailed)

## High-level summary
I introduced a **unified activation registry + runtime apply helper** for MoE GLU-style activations and **threaded it through every MoE path** (Cutlass, Triton, native, and quantized variants). This lets me switch between `silu`, `gelu`, `swish/swiglu`, `geglu`, and `reglu` (with optional `alpha`, `limit` clamp, and `up_shift`) **without adding new files** and while preserving the existing fused fast paths for the common cases.

---

## File-by-file changes

### 1) `python/sglang/srt/layers/activation.py`
- A compact activation “spec” and registry:
  - `@dataclass(frozen=True) ActivationSpec` with fields:
    - `name: str`
    - `alpha: Optional[float] = None`
    - `limit: Optional[float] = None` (for optional clamping)
    - `up_shift: float = 0.0` (to support variants like SWIGLU with a bias on the `up` branch)
  - `ActivationSpec.is_fastpath` property to detect when I can fall back to the existing highly-optimized fused kernels (`silu`/`gelu` with no extras).
  - `_ACT_REG: Dict[str, Callable[..., ActivationSpec]]`, `register_activation(name, builder)`, and `get_activation(name, **kwargs)` so I can register and resolve activation specs by name at runtime.
  - Built-in registrations:
    - `"silu"`, `"gelu"` (no extra params)
    - `"swish"` / `"swiglu"` (support `alpha`, `limit`, `up_shift` (default `1.0` for `swiglu`))
    - `"geglu"`, `"reglu"` (with optional `limit`, default `up_shift=0.0`)

- A pure-PyTorch GLU fallback that respects the spec:
  - `_apply_glu_python(x2d, spec)` computes:
    - gate = first half; up = second half
    - optional clamp via `limit`
    - activation choices:
      - `silu/swish` (respect `alpha`)
      - `swiglu` (respect `alpha` + `up_shift`)
      - `gelu/geglu`, `reglu`
      - default fallback = `silu`
    - returns `activation(gate) * (up + up_shift)`

- A single runtime entry point used by all MoE impls:
  - `apply_glu_activation_for_moe(x2d, out2d, spec)`:
    - If `spec.is_fastpath` and backend supports it:
      - CUDA/XPU/HIP → call existing fused kernels `silu_and_mul` / `gelu_and_mul`
      - CPU w/ AMX → call `torch.ops.sgl_kernel.*_and_mul_cpu`
      - Otherwise → manual `F.silu`/`F.gelu` multiply
    - If not fastpath → call `_apply_glu_python`
    - Always writes into `out2d` to minimize extra allocations.

- Backend flags at the bottom:
  - `_is_cuda = is_cuda()`
  - `_is_npu = is_npu()`
  - `_is_cpu_amx_available = cpu_has_amx_support()`

### 2) `python/sglang/srt/layers/moe/cutlass_moe.py`
- Imports:
  - `from sglang.srt.layers.activation import apply_glu_activation_for_moe, get_activation`
- In `cutlass_fused_experts_fp8(...)`:
  - Replaced the hard-coded `silu_and_mul(c1, intermediate)` with:
    ```python
    apply_glu_activation_for_moe(
        c1, intermediate,
        get_activation(activation, alpha=gemm1_alpha, limit=gemm1_limit),
    )
    ```
  - This applies whatever activation spec is passed in (still fast-paths `silu/gelu` when possible).
- In `cutlass_moe_fp4(...)`:
  - Extended the signature to accept activation parameters:
    - `*, activation: str = "silu", gemm1_alpha: Optional[float] = None, gemm1_limit: Optional[float] = None`
  - Replaced the fixed `silu_and_mul(...)` with the same unified call as above.


### 3) `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- Imports the unified helpers (already available via activation module).
- Replaced:
  - `silu_and_mul(c1, intermediate)`
  - with:
    ```python
    apply_glu_activation_for_moe(c1, intermediate, get_activation("silu"))
    ```
  - This keeps behavior identical (SiLU) but routes through the new helper so the code path is consistent and future-proof.

### 4) `python/sglang/srt/layers/moe/fused_moe_native.py`
- Imports:
  - Added `apply_glu_activation_for_moe` and `get_activation` alongside existing `GeluAndMul`, `SiluAndMul`.
- In `fused_moe_forward_native(...)`:
  - Previously :
    - `x1 = activation(x1)` via `F.silu`/`F.gelu`
    - `x3 = ...` (the “up” branch)
    - then used `(x1 * x3)` downstream.
  - Now I build a spec from `moe_runner_config`:
    ```python
    spec = get_activation(
        moe_runner_config.activation,
        alpha=moe_runner_config.gemm1_alpha,
        limit=moe_runner_config.gemm1_clamp_limit,
    )
    tmp = x1.new_empty(x1.shape[:-1] + (x1.shape[-1] // 2,))
    apply_glu_activation_for_moe(torch.cat([x1, x3], dim=-1), tmp, spec)
    x1 = tmp
    ```
  - This centralizes the GLU math into the helper (including optional `limit`, `alpha`, and `up_shift`).

- In `moe_forward_native(...)`:
  - Previously chose between `SiluAndMul()` and `GeluAndMul()` and applied directly.
  - Now I construct a spec (as above) and call:
    ```python
    tmp = gate_up.new_empty(gate_up.shape[:-1] + (gate_up.shape[-1] // 2,))
    apply_glu_activation_for_moe(gate_up, tmp, spec)
    gate_up = tmp
    ```
  - This replaces ad hoc branching with the unified activation application.

### 5) `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`
- Imports:
  - `apply_glu_activation_for_moe`, `get_activation`
- Inside `fused_experts_impl(...)`:
  - I deleted the large `if activation == "silu" ... elif activation == "gelu" ... else` block that manually handled:
    - `swiglu_with_alpha_and_limit(...)`
    - device-specific calls to `silu_and_mul` / `gelu_and_mul` / `vllm_ops.*`
  - Replaced it with the unified call:
    ```python
    spec = get_activation(activation, alpha=gemm1_alpha, limit=gemm1_limit)
    apply_glu_activation_for_moe(
        intermediate_cache1.view(-1, N), intermediate_cache2, spec
    )
    ```
- This leaves the kernel invocation that follows unchanged.

### 6) `python/sglang/srt/layers/quantization/fp8.py`
- In one early-return spot (right after producing `output`), I added:
  ```python
  return StandardCombineInput(hidden_states=output)

## Accuracy Tests
(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/Hanrui/modifi
ed/sglang_fork/test# python3 -m sglang.test.few_shot_gsm8k --num-questions 200
100%|██████████████████████████████████████| 200/200 [00:23<00:00,  8.49it/s]
Accuracy: 0.520
Invalid: 0.180
Latency: 23.781 s
Output throughput: 3729.099 token/s (original)

(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/Hanrui/modifi
ed/sglang_fork/test# python3 -m sglang.test.few_shot_gsm8k --num-questions 200
100%|██████████████████████████████████████| 200/200 [00:19<00:00, 10.36it/s]
Accuracy: 0.510
Invalid: 0.215
Latency: 19.348 s
Output throughput: 4398.578 token/s(modified)

## Benchmarking and Profiling

(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/Hanrui/modified# CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=90 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   python -m sglang.bench_one_batch     --model-path /z_data/syxin/Hanrui/gpt-oss-20b     --tokenizer-path /z_data/syxin/Hanrui/gpt-oss-20b     --load-f
ormat dummy     --dtype bfloat16     --batch 1 --input-len 16 --output-len 1     --kv-cache-dtype fp8_e5m2     --max-total-tokens 4096     --mem-fraction-static 0.1
2     --disable-cuda-graph     --json-model-override-args '{"num_hidden_layers": 1}'
All deep_gemm operations loaded successfully!
WARNING:sglang.srt.server_args:Detected GPT-OSS model, enabling triton_kernels MOE kernel.
`torch_dtype` is deprecated! Use `dtype` instead!
[2025-09-28 17:07:09 TP0] Downcasting torch.float32 to torch.bfloat16.
[2025-09-28 17:07:09 TP0] mxfp4 quantization is not fully optimized yet. The speed can be slower than non-quantized models.
[2025-09-28 17:07:09 TP0] FlashAttention3 only supports fp8_e4m3 if using FP8; Setting attention backend to triton.
[2025-09-28 17:07:09 TP0] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2025-09-28 17:07:09 TP0] Init torch distributed ends. mem usage=0.00 GB
[2025-09-28 17:07:10 TP0] Load weight begin. avail mem=94.63 GB
[2025-09-28 17:07:10 TP0] Load weight end. type=GptOssForCausalLM, dtype=torch.bfloat16, avail mem=91.91 GB, mem usage=2.72 GB.
[2025-09-28 17:07:10 TP0] Using KV cache dtype: torch.float8_e5m2
[2025-09-28 17:07:10 TP0] KV Cache is allocated. #tokens: 4096, K size: 0.00 GB, V size: 0.00 GB
[2025-09-28 17:07:10 TP0] Memory pool end. avail mem=89.83 GB
max_total_num_tokens=4096
Warmup ...
Prefill. latency: 0.47594 s, throughput:     33.62 token/s
Total. latency:  0.476 s, throughput:     35.72 token/s
Benchmark ...
Prefill. latency: 0.00328 s, throughput:   4879.42 token/s
Total. latency:  0.003 s, throughput:   5184.39 token/s
[rank0]:[W928 17:07:13.511609939 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/Hanrui/modified# CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=90 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \                                                                                                                                                      python -m sglang.bench_one_batch \                                                                                                                                    --model-path /z_data/syxin/Hanrui/gpt-oss-20b \
  --tokenizer-path /z_data/syxin/Hanrui/gpt-oss-20b \
  --load-format dummy \
  --dtype bfloat16 \
  --batch 1 --input-len 16 --output-len 1 \
  --kv-cache-dtype fp8_e5m2 \
  --max-total-tokens 4096 \
  --mem-fraction-static 0.12 \
  --disable-cuda-graph \
  --log-level debug \
  --json-model-override-args '{"num_hidden_layers": 1, "hidden_act_alpha": 1.30}'
All deep_gemm operations loaded successfully!
WARNING:sglang.srt.server_args:Detected GPT-OSS model, enabling triton_kernels MOE kernel.
`torch_dtype` is deprecated! Use `dtype` instead!
[2025-09-28 17:11:37 TP0] Downcasting torch.float32 to torch.bfloat16.
[2025-09-28 17:11:37 TP0] mxfp4 quantization is not fully optimized yet. The speed can be slower than non-quantized models.
[2025-09-28 17:11:37 TP0] FlashAttention3 only supports fp8_e4m3 if using FP8; Setting attention backend to triton.
[2025-09-28 17:11:37 TP0] Init torch distributed begin.
[2025-09-28 17:11:37 TP0] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://127.0.0.1:30680 backend=nccl
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2025-09-28 17:11:37 TP0] Init torch distributed ends. mem usage=0.00 GB
[2025-09-28 17:11:38 TP0] Initializing MLIR with module: _site_initialize_0
[2025-09-28 17:11:38 TP0] Registering dialects from initializer <module 'cutlass._mlir._mlir_libs._site_initialize_0' from '/opt/conda/lib/python3.11/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/_mlir/_mlir_libs/_site_initialize_0.cpython-311-x86_64-linux-gnu.so'>
[2025-09-28 17:11:38 TP0] Load weight begin. avail mem=94.63 GB
[2025-09-28 17:11:38 TP0] Load weight end. type=GptOssForCausalLM, dtype=torch.bfloat16, avail mem=91.91 GB, mem usage=2.72 GB.
[2025-09-28 17:11:38 TP0] Using KV cache dtype: torch.float8_e5m2
[2025-09-28 17:11:38 TP0] KV Cache is allocated. #tokens: 4096, K size: 0.00 GB, V size: 0.00 GB
[2025-09-28 17:11:38 TP0] Memory pool end. avail mem=89.83 GB
max_total_num_tokens=4096
Warmup ...
[2025-09-28 17:11:39 TP0] Attempting to acquire lock 140350307116048 on /root/.cache/flashinfer/90a/cached_ops/tmp/norm.lock
[2025-09-28 17:11:39 TP0] Lock 140350307116048 acquired on /root/.cache/flashinfer/90a/cached_ops/tmp/norm.lock
[2025-09-28 17:11:39 TP0] Attempting to release lock 140350307116048 on /root/.cache/flashinfer/90a/cached_ops/tmp/norm.lock
[2025-09-28 17:11:39 TP0] Lock 140350307116048 released on /root/.cache/flashinfer/90a/cached_ops/tmp/norm.lock
Prefill. latency: 0.46774 s, throughput:     34.21 token/s
Total. latency:  0.468 s, throughput:     36.34 token/s
Benchmark ...
Prefill. latency: 0.00317 s, throughput:   5039.38 token/s
Total. latency:  0.003 s, throughput:   5354.34 token/s
[rank0]:[W928 17:11:41.573765977 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/Hanrui/modified# CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=90 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -m sglang.bench_one_batch   --model-path /z_data/syxin/Hanrui/gpt-oss-20b   --tokenizer-path /z_data/syxin/Hanrui/gpt-oss-20b   --load-format du
mmy   --dtype bfloat16   --batch 1 --input-len 16 --output-len 1   --kv-cache-dtype fp8_e5m2   --max-total-tokens 4096   --mem-fraction-static 0.12   --disable-cuda
-graph   --profile   --json-model-override-args '{"num_hidden_layers": 1}'
All deep_gemm operations loaded successfully!
WARNING:sglang.srt.server_args:Detected GPT-OSS model, enabling triton_kernels MOE kernel.
`torch_dtype` is deprecated! Use `dtype` instead!
[2025-09-28 17:13:48 TP0] Downcasting torch.float32 to torch.bfloat16.
[2025-09-28 17:13:48 TP0] mxfp4 quantization is not fully optimized yet. The speed can be slower than non-quantized models.
[2025-09-28 17:13:48 TP0] FlashAttention3 only supports fp8_e4m3 if using FP8; Setting attention backend to triton.
[2025-09-28 17:13:48 TP0] Init torch distributed begin.
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[Gloo] Rank 0 is connected to 0 peer ranks. Expected number of connected peer ranks is : 0
[2025-09-28 17:13:48 TP0] Init torch distributed ends. mem usage=0.00 GB
[2025-09-28 17:13:49 TP0] Load weight begin. avail mem=94.63 GB
[2025-09-28 17:13:49 TP0] Load weight end. type=GptOssForCausalLM, dtype=torch.bfloat16, avail mem=91.91 GB, mem usage=2.72 GB.
[2025-09-28 17:13:49 TP0] Using KV cache dtype: torch.float8_e5m2
[2025-09-28 17:13:49 TP0] KV Cache is allocated. #tokens: 4096, K size: 0.00 GB, V size: 0.00 GB
[2025-09-28 17:13:49 TP0] Memory pool end. avail mem=89.83 GB
max_total_num_tokens=4096
Warmup ...
Prefill. latency: 0.47269 s, throughput:     33.85 token/s
Total. latency:  0.473 s, throughput:     35.96 token/s
Benchmark ...
Prefill. latency: 0.00625 s, throughput:   2560.79 token/s
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                Activity Buffer Request        26.71%     830.978us        26.71%     830.978us     830.978us       0.000us         0.00%       0.000us       0.000us             1
                                             SortTokens         9.42%     293.005us        11.53%     358.779us     358.779us       5.247us         0.90%       5.247us       5.247us             1
                                       cudaLaunchKernel         6.33%     197.031us         6.33%     197.031us       5.185us       0.000us         0.00%       0.000us       0.000us            38
                                                   TopK         6.04%     188.084us         7.02%     218.531us     218.531us       1.536us         0.26%       1.536us       1.536us             1
                                  cudaDeviceSynchronize         5.30%     164.896us         5.30%     164.896us      54.965us       0.000us         0.00%       0.000us       0.000us             3
                                            aten::addmm         4.55%     141.430us         5.77%     179.635us      59.878us      34.848us         5.98%      34.848us      11.616us             3
                                       cuLaunchKernelEx         3.95%     122.817us         3.95%     122.817us       7.225us       0.832us         0.14%       0.832us       0.049us            17
                                            aten::copy_         3.49%     108.712us         9.02%     280.746us      11.230us      38.141us         6.55%      38.141us       1.526us            25
                                        cudaMemcpyAsync         2.98%      92.639us         2.98%      92.639us       6.617us       0.000us         0.00%       0.000us       0.000us            14
                                            aten::empty         2.86%      88.949us         2.86%      88.949us       2.695us       0.000us         0.00%       0.000us       0.000us            33
                                    aten::empty_strided         2.60%      80.987us         2.60%      80.987us       3.681us       0.000us         0.00%       0.000us       0.000us            22
                                            aten::slice         2.36%      73.313us         2.93%      91.030us       2.396us       0.000us         0.00%       0.000us       0.000us            38
                                           aten::cumsum         2.24%      69.674us         5.61%     174.465us      43.616us      11.936us         2.05%      18.240us       4.560us             4
                                         aten::_to_copy         1.53%      47.692us        11.59%     360.599us      18.030us       0.000us         0.00%      29.180us       1.459us            20
                                  cudaStreamSynchronize         1.46%      45.289us         1.46%      45.289us       5.661us       0.000us         0.00%       0.000us       0.000us             8
                                              aten::sub         1.39%      43.367us         2.47%      76.944us      19.236us       6.367us         1.09%       6.367us       1.592us             4
                                             aten::view         1.23%      38.325us         1.23%      38.325us       1.533us       0.000us         0.00%       0.000us       0.000us            25
                                       aten::as_strided         1.19%      36.946us         1.19%      36.946us       0.543us       0.000us         0.00%       0.000us       0.000us            68
                                               aten::mm         1.02%      31.666us         1.54%      47.995us      47.995us     313.695us        53.85%     313.695us     313.695us             1
                                               aten::to         0.94%      29.390us        12.53%     389.989us      13.000us       0.000us         0.00%      29.180us       0.973us            30
                                           aten::select         0.93%      28.906us         1.06%      32.996us       2.750us       0.000us         0.00%       0.000us       0.000us            12
                                              aten::sum         0.78%      24.318us         1.32%      41.107us      41.107us       2.784us         0.48%       4.287us       4.287us             1
                                    cudaLaunchKernelExC         0.67%      20.738us         0.67%      20.738us       6.913us       0.000us         0.00%       0.000us       0.000us             3
                                          aten::minimum         0.65%      20.368us         0.86%      26.798us      26.798us       1.152us         0.20%       1.152us       1.152us             1
                                 aten::_index_put_impl_         0.64%      19.928us         1.09%      33.908us      16.954us       7.712us         1.32%       7.712us       3.856us             2
                                            aten::index         0.57%      17.648us         0.82%      25.657us      25.657us       3.904us         0.67%       3.904us       3.904us             1
                                     aten::index_select         0.55%      17.109us         0.99%      30.907us      30.907us      12.256us         2.10%      12.256us      12.256us             1
                                           aten::argmax         0.49%      15.399us         0.83%      25.778us      25.778us       9.569us         1.64%       9.569us       9.569us             1
                              aten::_local_scalar_dense         0.47%      14.669us         1.38%      43.057us      21.529us       5.408us         0.93%       5.408us       2.704us             2
                                        cudaMemsetAsync         0.47%      14.648us         0.47%      14.648us       7.324us       0.000us         0.00%       0.000us       0.000us             2
                                            aten::fill_         0.40%      12.399us         0.88%      27.258us      13.629us       2.176us         0.37%       2.176us       1.088us             2
                                        aten::transpose         0.38%      11.678us         0.56%      17.318us       3.464us       0.000us         0.00%       0.000us       0.000us             5
                                          aten::detach_         0.33%      10.418us         0.54%      16.895us       1.408us       0.000us         0.00%       0.000us       0.000us            12
                                           aten::linear         0.33%      10.368us         6.74%     209.572us      69.857us       0.000us         0.00%      34.848us      11.616us             3
                                    sgl_kernel::rmsnorm         0.31%       9.759us         0.64%      19.988us      19.988us       2.113us         0.36%       2.113us       2.113us             1
                                norm::fused_add_rmsnorm         0.31%       9.700us         0.85%      26.308us      13.154us       5.376us         0.92%       5.376us       2.688us             2
                                                aten::t         0.31%       9.699us         0.63%      19.569us       6.523us       0.000us         0.00%       0.000us       0.000us             3
                                          aten::squeeze         0.28%       8.829us         0.34%      10.499us       5.249us       0.000us         0.00%       0.000us       0.000us             2
                                 aten::split_with_sizes         0.27%       8.490us         0.32%       9.900us       9.900us       0.000us         0.00%       0.000us       0.000us             1
           sgl_kernel::apply_rope_pos_ids_cos_sin_cache         0.23%       7.189us         0.56%      17.408us      17.408us       2.304us         0.40%       2.304us       2.304us             1
                                       aten::empty_like         0.22%       6.799us         0.71%      22.018us       5.505us       0.000us         0.00%       0.000us       0.000us             4
                                        aten::unsqueeze         0.21%       6.650us         0.29%       9.090us       4.545us       0.000us         0.00%       0.000us       0.000us             2
cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFla...         0.21%       6.610us         0.21%       6.610us       1.322us       0.000us         0.00%       0.000us       0.000us             5
                                                detach_         0.21%       6.477us         0.21%       6.477us       0.540us       0.000us         0.00%       0.000us       0.000us            12
                                   cudaFuncSetAttribute         0.20%       6.099us         0.20%       6.099us       2.033us       0.000us         0.00%       0.000us       0.000us             3
                                       aten::index_put_         0.19%       6.049us         1.28%      39.957us      19.978us       0.000us         0.00%       7.712us       3.856us             2
                                            aten::zeros         0.19%       5.940us         1.41%      43.726us      21.863us       0.000us         0.00%       2.176us       1.088us             2
                                        aten::embedding         0.19%       5.760us         1.18%      36.667us      36.667us       0.000us         0.00%      12.256us      12.256us             1
                                            aten::clone         0.17%       5.180us         0.95%      29.687us      14.843us       0.000us         0.00%       5.024us       2.512us             2
                                            aten::zero_         0.16%       5.060us         1.04%      32.318us      16.159us       0.000us         0.00%       2.176us       1.088us             2
                                             aten::item         0.15%       4.730us         1.54%      47.787us      23.893us       0.000us         0.00%       5.408us       2.704us             2
                                          aten::reshape         0.13%       4.190us         0.19%       5.950us       1.983us       0.000us         0.00%       0.000us       0.000us             3
                                          aten::permute         0.13%       4.180us         0.15%       4.809us       4.809us       0.000us         0.00%       0.000us       0.000us             1
                                           aten::matmul         0.11%       3.490us         1.65%      51.485us      51.485us       0.000us         0.00%     313.695us     313.695us             1
                                          aten::resize_         0.11%       3.410us         0.11%       3.410us       3.410us       0.000us         0.00%       0.000us       0.000us             1
                                       aten::contiguous         0.10%       3.190us         1.06%      32.877us      16.439us       0.000us         0.00%       5.024us       2.512us             2
                                 cudaDeviceGetAttribute         0.09%       2.670us         0.09%       2.670us       0.534us       0.000us         0.00%       0.000us       0.000us             5
                                            aten::alias         0.08%       2.509us         0.08%       2.509us       2.509us       0.000us         0.00%       0.000us       0.000us             1
                                          aten::numpy_T         0.07%       2.310us         0.23%       7.119us       7.119us       0.000us         0.00%       0.000us       0.000us             1
                                       aten::lift_fresh         0.06%       1.800us         0.06%       1.800us       0.150us       0.000us         0.00%       0.000us       0.000us            12
                                    cudaPeekAtLastError         0.03%       0.930us         0.03%       0.930us       0.058us       0.000us         0.00%       0.000us       0.000us            16
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us       8.894us         1.53%       8.894us       0.809us            11
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.824us         0.31%       1.824us       1.824us             1
                         write_req_to_token_pool_triton         0.00%       0.000us         0.00%       0.000us       0.000us       1.728us         0.30%       1.728us       1.728us             1
                       Memcpy DtoH (Device -> Pageable)         0.00%       0.000us         0.00%       0.000us       0.000us       2.656us         0.46%       2.656us       2.656us             1
                                compute_position_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       1.280us         0.22%       1.280us       1.280us             1
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       7.807us         1.34%       7.807us       1.561us             5
void at_cuda_detail::cub::DeviceScanInitKernel<at_cu...         0.00%       0.000us         0.00%       0.000us       0.000us       5.088us         0.87%       5.088us       1.272us             4
void at_cuda_detail::cub::DeviceScanKernel<at_cuda_d...         0.00%       0.000us         0.00%       0.000us       0.000us       6.848us         1.18%       6.848us       1.712us             4
void at::native::unrolled_elementwise_kernel<at::nat...         0.00%       0.000us         0.00%       0.000us       0.000us       3.937us         0.68%       3.937us       1.312us             3
void at::native::reduce_kernel<512, 1, at::native::R...         0.00%       0.000us         0.00%       0.000us       0.000us       2.784us         0.48%       2.784us       2.784us             1
                         Memcpy DtoH (Device -> Pinned)         0.00%       0.000us         0.00%       0.000us       0.000us       5.408us         0.93%       5.408us       2.704us             2
                    create_flashinfer_kv_indices_triton         0.00%       0.000us         0.00%       0.000us       0.000us       2.112us         0.36%       2.112us       1.056us             2
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.152us         0.20%       1.152us       1.152us             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       1.312us         0.23%       1.312us       1.312us             1
void at::native::(anonymous namespace)::indexSelectS...         0.00%       0.000us         0.00%       0.000us       0.000us      12.256us         2.10%      12.256us      12.256us             1
void flashinfer::norm::RMSNormKernel<8u, __nv_bfloat...         0.00%       0.000us         0.00%       0.000us       0.000us       2.113us         0.36%       2.113us       2.113us             1
         nvjet_tst_192x16_64x8_4x1_v_bz_splitK_bias_TNT         0.00%       0.000us         0.00%       0.000us       0.000us      11.680us         2.00%      11.680us      11.680us             1
void cublasLt::splitKreduce_kernel<32, 16, int, floa...         0.00%       0.000us         0.00%       0.000us       0.000us       7.456us         1.28%       7.456us       2.485us             3
void flashinfer::BatchQKApplyRotaryPosIdsCosSinCache...         0.00%       0.000us         0.00%       0.000us       0.000us       2.304us         0.40%       2.304us       2.304us             1
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       5.856us         1.01%       5.856us       2.928us             2
void at::native::index_elementwise_kernel<128, 4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       7.712us         1.32%       7.712us       3.856us             2
void at::native::elementwise_kernel<128, 4, at::nati...         0.00%       0.000us         0.00%       0.000us       0.000us       5.024us         0.86%       5.024us       2.512us             2
                                            _fwd_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      34.175us         5.87%      34.175us      34.175us             1
        nvjet_tst_128x16_64x11_2x1_v_bz_splitK_bias_TNT         0.00%       0.000us         0.00%       0.000us       0.000us      10.400us         1.79%      10.400us      10.400us             1
void flashinfer::norm::FusedAddRMSNormKernel<8u, __n...         0.00%       0.000us         0.00%       0.000us       0.000us       5.376us         0.92%       5.376us       2.688us             2
         nvjet_tst_64x16_64x16_1x1_v_bz_splitK_bias_TNT         0.00%       0.000us         0.00%       0.000us       0.000us       5.312us         0.91%       5.312us       5.312us             1
                                          _topk_forward         0.00%       0.000us         0.00%       0.000us       0.000us       1.536us         0.26%       1.536us       1.536us             1
                                    _sum_bitmatrix_rows         0.00%       0.000us         0.00%       0.000us       0.000us       1.407us         0.24%       1.407us       1.407us             1
                               _combined_routing_memset         0.00%       0.000us         0.00%       0.000us       0.000us       1.600us         0.27%       1.600us       1.600us             1
                              _combined_routing_compute         0.00%       0.000us         0.00%       0.000us       0.000us       2.240us         0.38%       2.240us       2.240us             1
    _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1_swiglu         0.00%       0.000us         0.00%       0.000us       0.000us      47.328us         8.12%      47.328us      47.328us             1
void at::native::vectorized_elementwise_kernel<4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       2.176us         0.37%       2.176us       1.088us             2
                                 _compute_writeback_idx         0.00%       0.000us         0.00%       0.000us       0.000us       4.288us         0.74%       4.288us       4.288us             1
           _matmul_ogs_NNT_bf16xbf16xmxfp4_16x256x128x1         0.00%       0.000us         0.00%       0.000us       0.000us      22.624us         3.88%      22.624us      22.624us             1
                          _finalize_matmul_scatter_bf16         0.00%       0.000us         0.00%       0.000us       0.000us       2.496us         0.43%       2.496us       2.496us             1
void at::native::vectorized_elementwise_kernel<2, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.231us         0.55%       3.231us       1.615us             2
void at::native::index_elementwise_kernel<128, 4, at...         0.00%       0.000us         0.00%       0.000us       0.000us       3.904us         0.67%       3.904us       3.904us             1
                                        Memset (Device)         0.00%       0.000us         0.00%       0.000us       0.000us       1.601us         0.27%       1.601us       0.800us             2
                      nvjet_tst_384x8_64x4_2x1_v_bz_TNT         0.00%       0.000us         0.00%       0.000us       0.000us     312.959us        53.72%     312.959us     312.959us             1
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 3.111ms
Self CUDA time total: 582.555us

torch profiler chrome trace for prefill saved to profile_batch1_input16_output1_prefill.trace.json.gz
Total. latency:  0.006 s, throughput:   2720.84 token/s
[rank0]:[W928 17:13:51.044569783 ProcessGroupNCCL.cpp:1538] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.ai/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.ai/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.ai/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.ai/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.ai/developer_guide/contribution_guide.html#benchmark-the-speed).
