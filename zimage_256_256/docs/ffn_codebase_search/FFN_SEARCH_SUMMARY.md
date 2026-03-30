# SGLang FFN Implementation Search Summary

## Overview
Comprehensive analysis of Feed-Forward Network (FFN) implementations in SGLang codebase, with focus on SwiGLU variants, weight structures (w13, w2, gate_up_proj, down_proj), and model-specific implementations.

---

## 1. Core FFN Patterns Found

### Pattern 1.1: Z-Image Model (Diffusion) - SwiGLU with w13/w2
**Location**: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py` (lines 112-144)

```python
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, quant_config=None, prefix: str = ""):
        super().__init__()
        # Use MergedColumnParallelLinear for gate and up projection (fused)
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim, hidden_dim],  # Two outputs: gate and up
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=f"{prefix}.w13",
        )
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.w2",
        )
        self.act = SiluAndMul()  # SwiGLU activation

    def forward(self, x):
        x13, _ = self.w13(x)      # Gate and Up projection
        x = self.act(x13)          # Element-wise multiplication (SwiGLU)
        out, _ = self.w2(x)        # Down projection
        return out
```

**Key Characteristics**:
- **w13**: Fused gate + up projection (MergedColumnParallelLinear)
- **w2**: Down projection (RowParallelLinear)
- **Activation**: SiluAndMul (element-wise gating)
- **Used in**: Z-Image transformer blocks (lines 406-411)

### Pattern 1.2: LLaM-based Models (LLM) - SwiGLU with gate_up_proj/down_proj
**Location**: `python/sglang/srt/models/llama.py` and variants (xverse.py, etc.)

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, ...):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # Gate and Up
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
```

**Key Characteristics**:
- **gate_up_proj**: Fused gate + up projection (MergedColumnParallelLinear)
- **down_proj**: Down projection (RowParallelLinear)
- **Activation**: SiluAndMul
- **Used in**: Llama, Xverse, and other LLM variants

---

## 2. Weight Structure Mapping

### Z-Image Config: `python/sglang/multimodal_gen/configs/models/dits/zimage.py`

The model has sophisticated parameter mappings to handle different checkpoint formats:

```python
param_names_mapping: dict = {
    # Weight mappings: w1/w3 -> w13 (fused weight format)
    r"(.*)\.feed_forward\.w1\.weight$": (r"\1.feed_forward.w13.weight", 0, 2),
    r"(.*)\.feed_forward\.w3\.weight$": (r"\1.feed_forward.w13.weight", 1, 2),
    
    # FP8 block-quantized scale mappings
    r"(.*)\.feed_forward\.w1\.weight_scale_inv$": (
        r"\1.feed_forward.w13.weight_scale_inv", 0, 2),
    r"(.*)\.feed_forward\.w3\.weight_scale_inv$": (
        r"\1.feed_forward.w13.weight_scale_inv", 1, 2),
    
    # FP8 per-tensor scale mappings
    r"(.*)\.feed_forward\.w1\.weight_scale$": (
        r"\1.feed_forward.w13.weight_scale", 0, 2),
    r"(.*)\.feed_forward\.w3\.weight_scale$": (
        r"\1.feed_forward.w13.weight_scale", 1, 2),
    
    # LoRA mappings
    r"(.*)\.feed_forward\.w1\.(lora_A|lora_B)$": (
        r"\1.feed_forward.w13.\2", 0, 2),
    r"(.*)\.feed_forward\.w3\.(lora_A|lora_B)$": (
        r"\1.feed_forward.w13.\2", 1, 2),
}

stacked_params_mapping: list = [
    (".feed_forward.w13", ".feed_forward.w1", "gate"),
    (".feed_forward.w13", ".feed_forward.w3", "up"),
]
```

**Interpretation**:
- **w1**: Gate projection (dim -> hidden_dim)
- **w3**: Up projection (dim -> hidden_dim)  
- **w13**: Fused w1+w3 (both projections combined)
- Mapping tuple format: `(target_name, shard_id, total_shards)`

---

## 3. File Locations Summary

### Diffusion Models (Z-Image, Turbo, etc.)
- **Model Implementation**: 
  - `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`
  - FeedForward class with w13/w2
  
- **Config**:
  - `python/sglang/multimodal_gen/configs/models/dits/zimage.py`
  - ZImageArchConfig with param mappings
  
- **Generic MLP Layer**:
  - `python/sglang/multimodal_gen/runtime/layers/mlp.py`
  - MLP class and FeedForward class (generic)

- **Turbo/Sparse Attention**:
  - `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py`
  - MinimalA2AAttnOp for TurboWan
  - Context parallelism support

### LLM Models
- **Llama variants**:
  - `python/sglang/srt/models/llama.py` - LlamaMLP with gate_up_proj/down_proj
  
- **Other models with gate_up_proj**:
  - `python/sglang/srt/models/xverse.py`
  - `python/sglang/srt/models/grok.py`
  - `python/sglang/srt/models/nemotron_nas.py`
  - `python/sglang/srt/models/phimoe.py`
  - `python/sglang/srt/models/qwen3_5_mtp.py`
  - And 5+ more variants

---

## 4. FFN Implementation Variants

### Variant 1: Standard SwiGLU (Z-Image)
```
Input -> w13 (fused gate+up) -> SiluAndMul -> w2 -> Output
         [dim -> 2*hidden_dim]   [element-wise]  [hidden_dim -> dim]
```

### Variant 2: Llama-style SwiGLU (LLM models)
```
Input -> gate_up_proj -> SiluAndMul -> down_proj -> Output
         [hidden -> 2*inter]  [mul]    [inter -> hidden]
```

### Variant 3: Generic FeedForward (mlp.py)
Supports multiple activation types:
- GELU, ApproximateGELU
- GEGLU, ApproximateGELU
- **SwiGLU** ← Key for diffusion models
- LinearActivation with SiLU

```python
# From mlp.py lines 97-108
if activation_fn == "swiglu":
    act_fn = SwiGLU(dim, inner_dim, bias=bias)
elif activation_fn == "geglu":
    act_fn = GEGLU(dim, inner_dim, bias=bias)
# ... etc
```

### Variant 4: TurboWan (Minimal A2A Attention)
Location: `turbo_layer.py` (lines 229-275)
- Uses sparse linear attention backend
- All-to-all communication for context parallelism
- Supports topk-based sparsity

---

## 5. Quantization Integration

### FP8 Support in Z-Image
Both block-quantized and per-tensor scales are mapped:
```python
# Block-quantized scales
"weight_scale_inv" → maps between w1/w3 and w13

# Per-tensor scales  
"weight_scale" → maps between w1/w3 and w13
```

### LoRA Support
LoRA adapters mapped to fused weights:
```python
"w1.lora_A" / "w1.lora_B" → "w13.lora_A/B" (shard 0)
"w3.lora_A" / "w3.lora_B" → "w13.lora_A/B" (shard 1)
```

---

## 6. Key Classes & Utilities

| Class | Location | Purpose |
|-------|----------|---------|
| `FeedForward` (Z-Image) | `zimage.py:112` | SwiGLU with w13/w2 |
| `FeedForward` (Generic) | `mlp.py:71` | Generic gated linear units |
| `MLP` | `mlp.py:26` | Non-gated MLP |
| `LlamaMLP` | `llama.py` | Llama SwiGLU variant |
| `MinimalA2AAttnOp` | `turbo_layer.py:229` | TurboWan attention |
| `MergedColumnParallelLinear` | `linear.py` | Fused projections |
| `RowParallelLinear` | `linear.py` | Row-parallel down proj |
| `SiluAndMul` | `activation.py` | SwiGLU gating |

---

## 7. Activation Functions

Location: `python/sglang/multimodal_gen/runtime/layers/activation.py`

```python
class SiluAndMul(nn.Module):
    """Element-wise multiplication with SiLU gating"""
    def forward(self, x):
        # Splits x into two halves and multiplies: x[:d] * silu(x[d:])
        ...

# Imported from diffusers:
from diffusers.models.activations import SwiGLU, GEGLU, GELU
```

---

## 8. Testing & Examples

### Z-Image Test
- `test/registered/amd/test_zimage_turbo.py` - AMD backend test
- Tests Z-Image Turbo generation

### MLP Tests
- Found references in `test/srt/cpu/test_qkv_proj_with_rope.py`
- Tests quantization with w2 variables

---

## 9. Common Weight Variable Names

| Variable | Dimension | Purpose | Models |
|----------|-----------|---------|--------|
| `w13` | [dim, 2*hidden_dim] | Fused gate+up | Z-Image, FLUX, etc. |
| `w2` | [hidden_dim, dim] | Down projection | Z-Image, FLUX, etc. |
| `gate_up_proj` | [hidden, 2*inter] | Fused gate+up | Llama, Xverse, etc. |
| `down_proj` | [inter, hidden] | Down projection | Llama, Xverse, etc. |
| `w1` (original) | [dim, hidden_dim] | Gate projection | HuggingFace format |
| `w3` (original) | [dim, hidden_dim] | Up projection | HuggingFace format |

---

## 10. Models Using gate_up_proj/down_proj (10+ identified)

1. `llama.py` - LlamaMLP
2. `xverse.py` - XverseMLP  
3. `grok.py` - GrokMLP
4. `nemotron_nas.py` - NemotronNasMLP
5. `phimoe.py` - PhiMoeMLP
6. `qwen3_5_mtp.py` - Qwen3.5 MTP
7. `sarashina2_vision.py`
8. `glm4v_moe.py`
9. `gemma3_mm.py`
10. `dots_vlm_vit.py`

---

## 11. Parameter Loading & Weight Sharding

From Z-Image config:
```python
stacked_params_mapping = [
    (".feed_forward.w13", ".feed_forward.w1", "gate"),  # shard_id=0
    (".feed_forward.w13", ".feed_forward.w3", "up"),    # shard_id=1
]
```

This enables:
- Loading from separate w1/w3 checkpoints into merged w13
- FP8 quantization scale migration
- LoRA adapter mapping
- Distributed sharding across TPUs/GPUs

---

## 12. ZImage/Turbo Specific Resources

### Configuration Files
- `python/sglang/multimodal_gen/configs/models/dits/zimage.py`
- `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py`

### Model Files  
- `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`

### Test Files
- `test/registered/amd/test_zimage_turbo.py`
- `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_zimage_pipeline.py`

### Attention Backends
- Supports Turbo (sparse linear attention)
- `turbo_layer.py` - Context parallel all-to-all communication

