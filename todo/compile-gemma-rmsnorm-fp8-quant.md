# Plan: torch.compile GemmaRMSNorm + FP8 Per-Group Quantization

## Goal

Fuse `GemmaRMSNorm` (or `GemmaFusedAddRMSNorm`) with the subsequent
`per_token_group_quant_8bit` FP8 quantization into a single compiled kernel
via `torch.compile`. The fused kernel replaces two separate kernel launches
(norm + quant) with one, eliminating the intermediate bf16 materialization.

Target model: **Qwen3.5-35B-A3B-FP8** (`Qwen3_5MoeForConditionalGeneration`).
Target hardware: **SM100 (Blackwell)** with DeepGEMM block-FP8 GEMMs.

---

## Context: nsys Trace Analysis

A decode step of Qwen3.5-35B-A3B-FP8 with `--attention-backend trtllm_mha`
shows the following norm→quant adjacencies per layer pair:

| Site | Layer Type         | Norm                        | Feeds Into                     |
|------|--------------------|-----------------------------|--------------------------------|
| 1    | Std Attention      | GemmaFusedAddRMSNorm        | `qkv_proj` (attention)         |
| 2    | GatedDeltaNet      | GemmaFusedAddRMSNorm        | shared expert `gate_up_proj`   |
| 3    | Std Attention      | GemmaFusedAddRMSNorm        | shared expert `gate_up_proj`   |

Each site has the pattern:
```
flashinfer::FusedAddRMSNormKernel  (~2.5 μs)
per_token_group_quant_8bit_kernel  (~2.5 μs)
deep_gemm::sm100_fp8_gemm         (varies)
```

Fusing the first two kernels eliminates ~2.5 μs per site, 3 sites per
layer-pair, across all layers.

Additional norm→quant sites exist but are less frequent:
- `GemmaRMSNorm` (no residual) → quant: 1x per GatedDeltaNet layer
- `RMSNormGated` → quant: 1x per GatedDeltaNet layer (more complex, lower priority)

---

## Approach: PyTorch-native + torch.compile

Write the norm+quant logic as a pure PyTorch function using
`GemmaRMSNorm.forward_native` style ops plus per-group FP8 quantization.
Then compile it with `torch.compile` so the compiler fuses them into a
single GPU kernel.

### Why torch.compile (not a hand-written kernel)?

1. **Maintainability** — pure PyTorch, no Triton/CUDA to maintain.
2. **Extensibility** — easy to add variants (with/without residual,
   different norm types) without writing new kernels.
3. **Consistency** — follows the same `CompilableRegionMixin` +
   `--torch-compile-override-layers` pattern already used for
   QKNorm+RoPE fusion (see `todo/compile-qk-norm-rope-kv.md`).

---

## Reference Code

### GemmaRMSNorm.forward_native (layernorm.py:461-479)

```python
def forward_native(self, x, residual=None, post_residual_addition=None):
    orig_dtype = x.dtype
    if residual is not None:
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        x = x + residual
        residual = x

    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    x = x * (1.0 + self.weight.float())
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)
```

### Per-token-group FP8 quantization (pure PyTorch equivalent)

```python
def per_token_group_quant_fp8_native(x, group_size=128):
    """Per-token-group dynamic FP8 E4M3 quantization.

    x: [M, K] bf16 tensor, K divisible by group_size
    Returns: (x_q: [M, K] fp8_e4m3, x_s: [M, K//group_size] float32)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max  # 448.0

    M, K = x.shape
    x_grouped = x.reshape(M, K // group_size, group_size)
    amax = x_grouped.abs().float().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / fp8_max
    x_scaled = x_grouped.float() / scale
    x_q = x_scaled.to(torch.float8_e4m3fn).reshape(M, K)
    x_s = scale.squeeze(-1)  # [M, K // group_size]
    return x_q, x_s
```

### Fused function (the compilation target)

```python
def _gemma_fused_add_rmsnorm_fp8_quant(
    x, residual, weight, eps, group_size=128
):
    """GemmaFusedAddRMSNorm + per-group FP8 quantization in one pass.

    Inputs:
        x:        [M, hidden] bf16 — layer output (e.g. from attention o_proj)
        residual: [M, hidden] bf16 — residual stream
        weight:   [hidden]    bf16 — GemmaRMSNorm weight
        eps:      float            — variance epsilon
        group_size: int            — FP8 quantization group size (128)

    Returns:
        x_q:      [M, hidden] fp8_e4m3 — quantized normalized activation
        x_s:      [M, hidden // group_size] float32 — per-group scales
        residual: [M, hidden] bf16 — updated residual (= x + old_residual)
    """
    fp8_max = torch.finfo(torch.float8_e4m3fn).max

    # Step 1: Residual add
    x = x + residual
    residual = x

    # Step 2: Gemma RMSNorm (weight + 1)
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    x_normed = x_normed * (1.0 + weight.float())

    # Step 3: Per-token-group FP8 quantization
    M, K = x_normed.shape
    x_grouped = x_normed.reshape(M, K // group_size, group_size)
    amax = x_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-10)
    scale = amax / fp8_max
    x_scaled = x_grouped / scale
    x_q = x_scaled.to(torch.float8_e4m3fn).reshape(M, K)
    x_s = scale.squeeze(-1)

    return x_q, x_s, residual
```

Similarly for the no-residual variant (`_gemma_rmsnorm_fp8_quant`), just
drop the residual add and return `(x_q, x_s)`.

---

## Integration via CompilableRegionMixin

### Option A: Region on the decoder layer (preferred)

Register fused norm+quant regions on `Qwen3_5AttentionDecoderLayer` and
`Qwen3_5LinearDecoderLayer`, alongside the existing QKNorm/RopeKV regions.

```python
class Qwen3_5AttentionDecoderLayer(nn.Module, CompilableRegionMixin):

    _REGION_DYNAMIC = {
        "QKNorm": False,
        "RopeKV": None,
        "PreAttnNormQuant": None,   # NEW
        "PreMlpNormQuant": None,    # NEW
    }

    def get_compilable_regions(self) -> dict[str, str]:
        return {
            "QKNorm": "_qk_norm",
            "RopeKV": "_rope_kv",
            "PreAttnNormQuant": "_pre_attn_norm_quant",    # NEW
            "PreMlpNormQuant": "_pre_mlp_norm_quant",      # NEW
        }

    def _pre_attn_norm_quant(self, hidden_states, residual):
        """Compilable: input_layernorm + FP8 quant."""
        return _gemma_fused_add_rmsnorm_fp8_quant(
            hidden_states, residual,
            self.input_layernorm.weight,
            self.input_layernorm.variance_epsilon,
        )

    def _pre_mlp_norm_quant(self, hidden_states, residual):
        """Compilable: post_attention_layernorm + FP8 quant."""
        return _gemma_fused_add_rmsnorm_fp8_quant(
            hidden_states, residual,
            self.post_attention_layernorm.weight,
            self.post_attention_layernorm.variance_epsilon,
        )
```

### Calling the fused region

In `Qwen3_5AttentionDecoderLayer.forward`, the pre-attention path becomes:

```python
# Before (today):
#   hidden_states, residual = self.layer_communicator.prepare_attn(...)
#   hidden_states = self.self_attention(...)
#     └── inside self_attention: Fp8LinearMethod.apply does quant + GEMM

# After (when PreAttnNormQuant is compiled):
if self.is_region_compiled("PreAttnNormQuant"):
    x_q, x_s, residual = self._pre_attn_norm_quant(hidden_states, residual)
    # Skip the quant inside qkv_proj — pass pre-quantized input
    hidden_states = self.self_attention(
        positions=positions,
        hidden_states=(x_q, x_s),  # pre-quantized
        forward_batch=forward_batch,
    )
else:
    # Existing path: communicator does norm, linear does quant
    hidden_states, residual = self.layer_communicator.prepare_attn(...)
    hidden_states = self.self_attention(...)
```

The same pattern applies to `PreMlpNormQuant` before the MoE block.

### Passing pre-quantized activations to the linear layer

The `Fp8LinearMethod.apply` path currently quantizes the bf16 input. With
the fused region, the input arrives already as `(fp8_tensor, scales)`. The
linear layer needs a way to accept pre-quantized input and skip the quant
step, going directly to the DeepGEMM call.

Options:
1. **Wrapper approach**: create a thin wrapper that calls the GEMM with
   pre-quantized input, bypassing `Fp8LinearMethod.apply`.
2. **Flag on Fp8LinearMethod**: add an `already_quantized` code path to
   `apply()` that skips the quant kernel when input is already FP8.
3. **Direct GEMM call**: call `deep_gemm_fp8_fp8_bf16_nt` directly from
   the decoder layer, passing the pre-quantized activations and the
   weight from the linear layer.

Option 2 is cleanest — it keeps the existing module structure and just
adds a fast path.

---

## CLI Usage

```bash
python -m sglang.launch_server \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --enable-torch-compile \
    --torch-compile-scope local \
    --torch-compile-override-layers PreAttnNormQuant PreMlpNormQuant \
    --torch-compile-max-bs 1024
```

Can be combined with existing compiled regions:
```bash
    --torch-compile-override-layers QKNorm RopeKV PreAttnNormQuant PreMlpNormQuant
```

---

## Compile Boundary Concerns

### 1. FP8 dtype support in torch.compile

`torch.float8_e4m3fn` is supported in PyTorch 2.x Dynamo/Inductor. The
`.to(torch.float8_e4m3fn)` cast and FP8 tensor creation should trace
without graph breaks on recent PyTorch builds. Needs empirical verification.

### 2. Output is FP8 — crossing the compile boundary

The compiled function returns `(fp8_tensor, float32_scales, bf16_residual)`.
Returning mixed dtypes from a compiled region is fine — Dynamo handles
multiple return values of different dtypes.

### 3. Residual must remain bf16

The residual stream stays in bf16 throughout. Only the normalized activation
gets quantized to FP8 — the residual is returned as-is for the next layer.

### 4. Interaction with LayerCommunicator

When the fused region is active, we bypass `layer_communicator.prepare_attn`
/ `prepare_mlp` for the norm step. The allreduce/TP communication still
needs to happen. Two options:

- **Do allreduce before the fused region**, then call the region on the
  already-reduced hidden states. This keeps the compile boundary clean.
- **Include allreduce in the region** — likely causes graph breaks with
  NCCL ops. Not recommended.

The simple path: keep the allreduce in the communicator, replace only the
norm call with the fused norm+quant region.

### 5. Group size must match DeepGEMM expectations

DeepGEMM block-FP8 uses group_size=128 with specific scale tensor layouts.
The fused quant must produce scales in the same format as
`sglang_per_token_group_quant_fp8` (row-major, float32,
shape `[M, K // 128]`). The reference implementation above does this.

---

## Scope and Non-Goals

### In scope
- GemmaFusedAddRMSNorm + FP8 group quant (3 sites, highest value)
- GemmaRMSNorm + FP8 group quant (1 site, GatedDeltaNet pre-attention)
- Qwen3.5 model family on CUDA SM100

### Not in scope (future work)
- RMSNormGated + FP8 quant (complex gated norm, low frequency)
- SiLU+mul + FP8 quant (already exists as `fuse_silu_and_mul` in sgl-kernel v2)
- Sigmoid+mul + FP8 quant (attn output gate → o_proj, 1 site)
- DeepGEMM epilogue fusion (requires upstream DeepGEMM changes)
- ROCm / AITER path (already has `fused_rms_fp8_group_quant`)

---

## Implementation Steps

### Phase 1: Fused PyTorch function

1. Add `_gemma_fused_add_rmsnorm_fp8_quant` and `_gemma_rmsnorm_fp8_quant`
   as standalone functions (pure PyTorch, no custom ops).
2. Verify numerical equivalence: `forward_native` + `per_token_group_quant`
   vs the fused function, across representative input shapes.

### Phase 2: CompilableRegion registration

3. Register `PreAttnNormQuant` and `PreMlpNormQuant` regions on
   `Qwen3_5AttentionDecoderLayer`.
4. Register `PreAttnNormQuant` and `PreMlpNormQuant` on
   `Qwen3_5LinearDecoderLayer` (same pattern, different norm modules).
5. Add the `is_region_compiled(...)` check in each layer's `forward` to
   select the fused path vs the existing path.

### Phase 3: Pre-quantized linear input path

6. Add a fast path in `Fp8LinearMethod.apply` (or an alternative entry
   point) that accepts `(fp8_input, scales)` and skips the quant kernel,
   going directly to the block-FP8 GEMM.
7. Wire the decoder layer forward to pass pre-quantized activations to
   `qkv_proj` / shared expert `gate_up_proj` when the fused region is active.

### Phase 4: Validation

8. Correctness: compare logits with and without the fused regions.
9. Compile cleanliness: run with `TORCH_LOGS="graph_breaks"` to verify
   no graph breaks in the fused function.
10. Performance: nsys profile decode at bs=1 and bs=32, confirm the
    norm+quant kernels are fused into a single kernel launch.

---

## Expected Outcome

Before (per layer-pair, 3 sites):
```
FusedAddRMSNormKernel          ~2.5 μs
per_token_group_quant_8bit     ~2.5 μs
DeepGEMM                       varies
```

After (per layer-pair, 3 sites):
```
compiled_norm_quant_kernel     ~3.0 μs  (estimate: one pass over data)
DeepGEMM                       varies
```

Estimated saving: ~2 μs × 3 sites × N_layers, plus reduced memory traffic
from not materializing the intermediate bf16 normalized tensor.
