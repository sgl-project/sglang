<!-- Thank you for your contribution! Please follow these guidelines to enhance your pull request. If anything is unclear, submit your PR and reach out to maintainers for assistance. Join our Slack community at https://slack.sglang.ai to discuss further. -->

## Motivation

Enable config-driven, parameterized GLU activations for MoE across Triton/CUTLASS/native paths (bf16/fp8/fp4), while preserving existing fast paths and backward compatibility. Provide a single, unified activation entry that supports `silu`, `gelu`, `swish/swiglu`, `geglu`, `reglu` with optional `alpha`, `limit` (clamp), and `up_shift`. Defaults match the current behavior (e.g., standard SiLU/GEGLU fast paths) and no official configs need changes.

## Scope

- Only MoE GLU activation unification and plumbed params.
- No unrelated refactors; defaults remain unchanged; fast paths retained.

## Implementation Outline

- Add a minimal ActivationSpec + registry and a unified runtime apply helper:
  - `ActivationSpec(name, alpha=None, limit=None, up_shift=0.0)` and `register_activation()` / `get_activation()`.
  - `_apply_glu_python(x2d, spec)` as the generic PyTorch fallback implementing GLU variants (silu/swish/swiglu/gelu/geglu/reglu) with optional clamp and up_shift.
  - `apply_glu_activation_for_moe(x2d, out2d, spec)` that routes to fused kernels (`silu_and_mul`, `gelu_and_mul`, or CPU AMX ops) when on the fast path (plain silu/gelu), else falls back to Python.
- Thread the unified entry through MoE Triton/CUTLASS/native implementations.
- Expose activation parameters via config (alpha/limit) and pass them through to CUTLASS FP8/FP4 paths.

## File-by-file Changes

1) `python/sglang/srt/layers/activation.py`
- Add the compact activation spec/registry and unified runtime apply helper:
  - `ActivationSpec`, `_ACT_REG`, `register_activation`, `get_activation`.
  - Built-ins: `"silu"`, `"gelu"`, `"swish"`, `"swiglu"` (default `up_shift=1.0`), `"geglu"`, `"reglu"`.
  - `_apply_glu_python(x2d, spec)` and `apply_glu_activation_for_moe(x2d, out2d, spec)` with fused fast paths on CUDA/HIP/XPU/CPU(AMX).
- Keep existing classes (`SiluAndMul`, `GeluAndMul`, etc.) unchanged.

2) `python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py`
- Imports: `apply_glu_activation_for_moe`, `get_activation`.
- Replace the manual activation branch after GEMM1 with the unified entry:
  ```python
  spec = get_activation(activation, alpha=gemm1_alpha, limit=gemm1_limit)
  apply_glu_activation_for_moe(intermediate_cache1.view(-1, N), intermediate_cache2, spec)
  ```
- Leaves the rest of the kernel plumbing unchanged.

3) `python/sglang/srt/layers/moe/fused_moe_native.py`
- Imports: `apply_glu_activation_for_moe`, `get_activation`.
- `fused_moe_forward_native(...)` and `moe_forward_native(...)` use the unified entry to compute GLU on the concatenated `[gate, up]` input, preserving prior behavior on default `silu`/`gelu` and enabling parameterization.

4) `python/sglang/srt/layers/moe/cutlass_moe.py`
- Imports: `apply_glu_activation_for_moe`, `get_activation`.
- `cutlass_fused_experts_fp8(...)`: replace `silu_and_mul(c1, intermediate)` with the unified entry and accept activation parameters through function args:
  ```python
  def cutlass_fused_experts_fp8(...,
      use_fp8_blockscale: bool = True,
      *,
      activation: str = "silu",
      gemm1_alpha: Optional[float] = None,
      gemm1_limit: Optional[float] = None,
  ) -> torch.Tensor:
      ...
      apply_glu_activation_for_moe(c1, intermediate,
          get_activation(activation, alpha=gemm1_alpha, limit=gemm1_limit))
  ```
- `cutlass_moe_fp4(...)`: likewise, extend the signature with the same activation parameters and use the unified entry.

5) `python/sglang/srt/layers/quantization/fp8.py`
- In the `use_cutlass_fused_experts_fp8` path, pass activation parameters to `cutlass_fused_experts_fp8(...)`:
  ```python
  output = cutlass_fused_experts_fp8(
      x, ...,
      use_fp8_blockscale=True,
      activation=self.moe_runner_config.activation,
      gemm1_alpha=self.moe_runner_config.gemm1_alpha,
      gemm1_limit=self.moe_runner_config.gemm1_clamp_limit,
  )
  ```
- No other changes (no additional early returns added or moved).

6) `python/sglang/srt/layers/moe/cutlass_w4a8_moe.py`
- Replace `silu_and_mul(c1, intermediate)` with:
  ```python
  apply_glu_activation_for_moe(c1, intermediate, get_activation("silu"))
  ```
- Behavior identical to SiLU fast path; routes through the unified entry for consistency.

## Backward Compatibility

- Defaults keep current behavior (e.g., `activation="silu"`), with fused fast paths preserved.
- Activation parameters are optional; configs do not need changes.

## Testing Plan

- Unit: existing MoE tests should pass; add small shape tests for each activation name and `alpha/limit/up_shift` options.
- E2E: sanity runs on Triton/CUTLASS/native MoE paths (bf16/fp8/fp4), comparing outputs to baselines; verify no perf regression for plain `silu`/`gelu`.

## Checklist
- [ ] Format with pre-commit
- [ ] Add/extend unit tests for activation coverage
- [ ] Update docs if needed
- [ ] Provide accuracy/speed benchmarks if impacted
