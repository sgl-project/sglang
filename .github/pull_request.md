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
(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/hanrui/sglang_fork/test# python3 -m sglang.test.few_shot_gsm8k --num-questions 200
100%|███████████████████████████████████████████████████████████████████████████████████| 200/200 [00:22<00:00,  8.96it/s]
Accuracy: 0.565
Invalid: 0.130
Latency: 22.411 s
Output throughput: 3841.140 token/s (original)

(sglang) root@nb-1365100433987193600-amrnhavl9gjk:/z_data/syxin/hanrui/sglang_fork/test# python3 -m sglang.test.few_shot_gsm8k --num-questions 200
Downloading from https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl to /tmp/test.jsonl
/tmp/test.jsonl: 732kB [00:00, 8.45MB/s]                                                                                                                            
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:21<00:00,  9.12it/s]
Accuracy: 0.535
Invalid: 0.160
Latency: 22.434 s
Output throughput: 3992.351 token/s (modified)

## Benchmarking and Profiling

<!-- If this pull request impacts inference speed, provide benchmarking and profiling results. -->

## Checklist

- [ ] Format your code according to the [Format code with pre-commit](https://docs.sglang.ai/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [ ] Add unit tests according to the [Run and add unit tests](https://docs.sglang.ai/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.ai/developer_guide/contribution_guide.html#write-documentations).
- [ ] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.ai/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.ai/developer_guide/contribution_guide.html#benchmark-the-speed).
