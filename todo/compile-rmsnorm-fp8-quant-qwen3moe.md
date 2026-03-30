# Plan: torch.compile RMSNorm + FP8 Per-Group Quantization (Qwen3MoE)

## Goal

Fuse `RMSNorm` (or `FusedAddRMSNorm`) with the subsequent
`per_token_group_quant_8bit` FP8 quantization into a single compiled kernel
via `torch.compile`. The fused kernel replaces two separate kernel launches
(norm + quant) with one, eliminating the intermediate bf16 materialization.

Target model: **Qwen3MoE** (`Qwen3MoeForCausalLM`, e.g. Qwen/Qwen3-30B-A3B).
Target hardware: **SM100 (Blackwell)** with DeepGEMM block-FP8 GEMMs.

---

## Why Qwen3MoE Is the Simplest Starting Point

| Aspect | Qwen3MoE | Qwen3.5 |
|---|---|---|
| Layer types | 1 (`Qwen3MoeDecoderLayer`) | 2 (GatedDeltaNet + Std Attention) |
| Norm type | Standard `RMSNorm` (`weight * x`) | `GemmaRMSNorm` (`(weight+1) * x`) + `RMSNormGated` |
| Fusion sites per layer | 2 (uniform, every layer identical) | 2-3 (varies by layer type) |
| AITER reference | `fused_rms_fp8_group_quant` works as-is | Needs Gemma `w+1` variant |

Every decoder layer has the exact same structure:
```
input_layernorm → attention → post_attention_layernorm → MoE
```

---

## Architecture: Single Decoder Layer

`Qwen3MoeDecoderLayer` (qwen3_moe.py:735):

```
hidden_states, residual
    │
    ├── layer_communicator.prepare_attn(hidden_states, residual)
    │       └── input_layernorm(hidden_states, residual)        ← RMSNorm + residual add
    │               └── returns bf16 hidden_states
    │
    ├── self_attn(positions, hidden_states, forward_batch)
    │       └── qkv_proj(hidden_states)                         ← quant + DeepGEMM
    │       └── qk_norm + rope + attention + o_proj
    │
    ├── layer_communicator.prepare_mlp(hidden_states, residual)
    │       └── post_attention_layernorm(hidden_states, residual) ← RMSNorm + residual add
    │               └── returns bf16 hidden_states
    │
    └── mlp(hidden_states)                                       ← Qwen3MoeSparseMoeBlock
            └── shared expert gate_up_proj                       ← quant + DeepGEMM
            └── MoE experts (TRT-LLM BMM)
```

---

## Fusion Sites (2 per layer, every layer)

### Site 1: Before Attention

**Code path:**
```
Qwen3MoeDecoderLayer.forward (line 831)
  └── layer_communicator.prepare_attn_and_capture_last_layer_outputs()
      └── communicator.py: self.input_layernorm(hidden_states, residual)
          └── RMSNorm.forward_cuda → fused_add_rmsnorm()    ← NORM KERNEL
              └── returns bf16 hidden_states
                      │
                      ▼
Qwen3MoeDecoderLayer.forward (line 842)
  └── self.self_attn(positions, hidden_states, forward_batch)
      └── Qwen3MoeAttention.forward_prepare_native (line 560)
          └── self.qkv_proj(hidden_states)
              └── Fp8LinearMethod.apply()
                  └── per_token_group_quant_8bit()            ← QUANT KERNEL
                  └── deep_gemm_fp8_fp8_bf16_nt()             ← GEMM
```

### Site 2: Before MoE

**Code path:**
```
Qwen3MoeDecoderLayer.forward (line 848)
  └── layer_communicator.prepare_mlp(hidden_states, residual, forward_batch)
      └── communicator.py: self.post_attention_layernorm(hidden_states, residual)
          └── RMSNorm.forward_cuda → fused_add_rmsnorm()    ← NORM KERNEL
              └── returns bf16 hidden_states
                      │
                      ▼
Qwen3MoeDecoderLayer.forward (line 863)
  └── self.mlp(hidden_states, forward_batch, ...)
      └── Qwen3MoeSparseMoeBlock.forward()
          └── shared_expert.gate_up_proj(hidden_states)
              └── Fp8LinearMethod.apply()
                  └── per_token_group_quant_8bit()            ← QUANT KERNEL
                  └── deep_gemm_fp8_fp8_bf16_nt()             ← GEMM
```

---

## Approach: PyTorch-native + torch.compile

Write the norm+quant as pure PyTorch in one function, then let
`torch.compile` fuse them into a single kernel.

### RMSNorm.forward_native (layernorm.py:261-309)

```python
def forward_native(self, x, residual=None, post_residual_addition=None):
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    x = (x * self.weight).to(orig_dtype)

    return x if residual is None else (x, residual)
```

Note: standard RMSNorm uses `x * weight`, not `x * (weight + 1)` like Gemma.

### Fused function (the compilation target)

```python
def _fused_add_rmsnorm_fp8_quant(x, residual, weight, eps, group_size=128):
    """FusedAddRMSNorm + per-group FP8 quantization in one pass.

    Inputs:
        x:        [M, hidden] bf16 — layer output
        residual: [M, hidden] bf16 — residual stream
        weight:   [hidden]         — RMSNorm weight
        eps:      float            — variance epsilon
        group_size: int            — FP8 quantization group size (128)

    Returns:
        x_q:      [M, hidden] fp8_e4m3 — quantized normalized activation
        x_s:      [M, hidden // group_size] float32 — per-group scales
        residual: [M, hidden] bf16 — updated residual (= x + old_residual)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Step 1: Residual add (in float32 for accuracy)
    x = x.float() + residual.float()
    residual = x.to(torch.bfloat16)

    # Step 2: Standard RMSNorm
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    x_normed = x_normed * weight.float()

    # Step 3: Per-token-group FP8 quantization
    M, K = x_normed.shape
    x_grouped = x_normed.reshape(M, K // group_size, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / fp8_max
    x_q = (x_grouped / scale).to(torch.float8_e4m3fn).reshape(M, K)
    x_s = scale.squeeze(-1)

    return x_q, x_s, residual
```

No-residual variant (`_rmsnorm_fp8_quant`) is the same without step 1.

---

## Integration via CompilableRegionMixin

Register two new regions on `Qwen3MoeDecoderLayer`:

```python
class Qwen3MoeDecoderLayer(nn.Module, CompilableRegionMixin):

    _REGION_DYNAMIC = {"PreAttnNormQuant": None, "PreMlpNormQuant": None}

    def get_compilable_regions(self) -> dict[str, str]:
        return {
            "PreAttnNormQuant": "_pre_attn_norm_quant",
            "PreMlpNormQuant": "_pre_mlp_norm_quant",
        }

    def _pre_attn_norm_quant(self, hidden_states, residual):
        return _fused_add_rmsnorm_fp8_quant(
            hidden_states, residual,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )

    def _pre_mlp_norm_quant(self, hidden_states, residual):
        return _fused_add_rmsnorm_fp8_quant(
            hidden_states, residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
```

Note: `Qwen3MoeDecoderLayer` currently does NOT mix in
`CompilableRegionMixin`. Only `Qwen3MoeAttention` does (for QKNorm/RopeKV).
Adding the mixin to the decoder layer is straightforward.

### Modified forward (sketch)

```python
def forward(self, positions, hidden_states, forward_batch, residual, ...):

    # --- Pre-attention ---
    if self.is_region_compiled("PreAttnNormQuant"):
        x_q, x_s, residual = self._pre_attn_norm_quant(hidden_states, residual)
        # Pass pre-quantized input to attention
        hidden_states = self.self_attn(
            positions, (x_q, x_s), forward_batch,
        )
    else:
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.self_attn(positions, hidden_states, forward_batch)

    # --- Pre-MLP ---
    if self.is_region_compiled("PreMlpNormQuant"):
        x_q, x_s, residual = self._pre_mlp_norm_quant(hidden_states, residual)
        # Pass pre-quantized input to MoE shared expert
        hidden_states = self.mlp(
            (x_q, x_s), forward_batch, ...
        )
    else:
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        hidden_states = self.mlp(hidden_states, forward_batch, ...)

    # ... postprocess_layer ...
    return hidden_states, residual
```

### Pre-quantized linear input path

When the fused region is active, `qkv_proj` and `gate_up_proj` receive
`(fp8_tensor, scales)` instead of bf16. The `Fp8LinearMethod.apply` needs
a fast path to skip quantization and go directly to the GEMM.

Options (same as in compile-gemma-rmsnorm-fp8-quant.md):
1. Flag on `Fp8LinearMethod.apply` to skip quant when input is already FP8.
2. Direct GEMM call from the decoder layer, bypassing the linear module.
3. Wrapper that calls DeepGEMM with pre-quantized activations.

Option 1 is cleanest for maintainability.

---

## Interaction with TP Communication

When the fused region is active, we bypass `layer_communicator.prepare_attn`
/ `prepare_mlp` for the norm step. The allreduce still needs to happen.

For the simple TP=1 case, there is no allreduce — the communicator just
calls the layernorm. The fused region directly replaces this.

For TP > 1, the sequence is:
1. Allreduce on `hidden_states` (from previous o_proj / MoE down_proj)
2. Norm(hidden_states, residual)
3. Quant

The allreduce must happen before the fused region. Options:
- **Split**: do allreduce in the communicator, then call the fused region
  on the reduced tensor. This is the simplest approach.
- **Fuse allreduce+norm+quant**: would require NCCL ops inside
  `torch.compile`, likely causing graph breaks. Not recommended.

---

## CLI Usage

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3-30B-A3B \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers PreAttnNormQuant PreMlpNormQuant \
    --torch-compile-max-bs 1024
```

Can be combined with existing attention compile regions:
```bash
    --torch-compile-override-layers QKNorm RopeKV PreAttnNormQuant PreMlpNormQuant
```

---

## Compile Boundary Concerns

### 1. FP8 dtype in torch.compile

`.to(torch.float8_e4m3fn)` is supported in PyTorch 2.x Dynamo/Inductor.
Needs empirical verification that Inductor generates a fused kernel
(not separate cast + store).

### 2. Scale tensor layout

DeepGEMM block-FP8 expects scales as `[M, K // 128]` float32 row-major.
The fused function produces exactly this layout. Must verify alignment
and contiguity requirements match.

### 3. No Gemma variant needed

Standard RMSNorm uses `x * weight` — no `+ 1` adjustment. This means:
- Existing `fused_rms_fp8_group_quant` (AITER) works as a validation
  reference on ROCm.
- The fused PyTorch function is simpler than the Gemma variant.

### 4. variance_size_override / cast_x_before_out_mul

`RMSNorm` has optional `variance_size_override` and `cast_x_before_out_mul`
flags. For Qwen3MoE these are not used (defaults: `None` and `False`).
The fused function can ignore them. If generalized later, they need handling.

---

## Validation Reference

AITER's `fused_rms_fp8_group_quant` on ROCm does exactly this fusion for
standard RMSNorm. Test file:
`test/registered/quant/test_fused_rms_fp8_group_quant.py`.

The numerical reference for validation:
```python
# Reference: separate norm + quant
hidden_states, residual = rmsnorm(hidden_states, residual)  # forward_native
x_q, x_s = sglang_per_token_group_quant_fp8(hidden_states, group_size=128)

# Fused: should produce identical x_q, x_s, residual
x_q_fused, x_s_fused, residual_fused = _fused_add_rmsnorm_fp8_quant(
    hidden_states_orig, residual_orig, weight, eps
)

torch.testing.assert_close(x_q, x_q_fused)
torch.testing.assert_close(x_s, x_s_fused)
torch.testing.assert_close(residual, residual_fused)
```

---

## Implementation Steps

### Phase 1: Fused PyTorch function

1. Implement `_fused_add_rmsnorm_fp8_quant` and `_rmsnorm_fp8_quant`
   as standalone pure PyTorch functions.
2. Unit test: verify numerical equivalence against separate
   `RMSNorm.forward_native` + `per_token_group_quant_fp8`.
3. Verify `torch.compile` produces a single fused kernel (no graph breaks).
   Check with `TORCH_LOGS="graph_breaks"`.

### Phase 2: CompilableRegion registration

4. Add `CompilableRegionMixin` to `Qwen3MoeDecoderLayer`.
5. Register `PreAttnNormQuant` and `PreMlpNormQuant` regions.
6. Add `is_region_compiled(...)` branches in `forward()`.

### Phase 3: Pre-quantized input path

7. Add a fast path in `Fp8LinearMethod.apply` that accepts `(fp8, scales)`
   and skips the quant kernel.
8. Wire `Qwen3MoeAttention.forward_prepare_native` to accept pre-quantized
   `hidden_states` and pass them directly to the weight GEMM.
9. Wire `Qwen3MoeSparseMoeBlock` shared expert to accept pre-quantized input.

### Phase 4: Validation

10. Correctness: compare logits with and without fused regions.
11. Performance: nsys profile decode, confirm 2 kernel launches eliminated
    per layer (norm + quant → single compiled kernel).
12. Test at multiple batch sizes (1, 32, 128, 256).

---

## Expected Outcome

Before (per layer, 2 sites):
```
FusedAddRMSNormKernel          ~2.5 μs
per_token_group_quant_8bit     ~2.5 μs
DeepGEMM                       varies
```

After (per layer, 2 sites):
```
compiled_norm_quant_kernel     ~3.0 μs  (estimate: one pass over data)
DeepGEMM                       varies
```

Estimated saving: ~2 μs × 2 sites × N_layers, plus reduced memory traffic
from not materializing the intermediate bf16 normalized tensor.

---

## Generalization Path

Once validated on Qwen3MoE, the same pattern generalizes to:
- **Qwen3.5**: swap `_fused_add_rmsnorm_fp8_quant` for a Gemma variant
  with `(1 + weight)`. See `todo/compile-gemma-rmsnorm-fp8-quant.md`.
- **DeepSeek-V3/R1**: uses standard `RMSNorm`, same as Qwen3MoE.
- **Llama4**: uses standard `RMSNorm`.
- Any model with `RMSNorm` → FP8 linear pattern.
