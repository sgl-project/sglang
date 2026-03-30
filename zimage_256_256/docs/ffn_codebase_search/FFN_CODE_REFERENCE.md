# SGLang FFN Code Reference Guide

## Quick Reference: Where to Find FFN Code

### Diffusion/Multimodal Gen (Preferred for SwiGLU)
```
python/sglang/multimodal_gen/
├── runtime/
│   ├── models/dits/
│   │   ├── zimage.py ...................... Z-Image FFN with w13/w2
│   │   ├── flux_2.py ...................... FLUX2 FFN
│   │   └── helios.py ...................... Helios FFN
│   ├── layers/
│   │   ├── mlp.py ......................... Generic MLP, FeedForward classes
│   │   ├── activation.py .................. SiluAndMul, other activations
│   │   ├── linear.py ...................... MergedColumnParallelLinear
│   │   └── attention/turbo_layer.py ....... TurboWan sparse attention
│   └── loader/
│       └── model_loader.py ................ Weight loading, param mapping
└── configs/
    ├── models/dits/
    │   └── zimage.py ...................... Z-Image config with param_names_mapping
    └── pipeline_configs/
        └── zimage.py ...................... Pipeline-level config
```

### LLM/SRT (Alternative naming convention)
```
python/sglang/srt/models/
├── llama.py ............................. LlamaMLP (gate_up_proj/down_proj)
├── xverse.py ............................ XverseMLP variant
├── grok.py .............................. GrokMLP variant
└── [10+ more models using gate_up_proj] .. See section 10 in summary
```

---

## Detailed Code Examples

### Example 1: Z-Image FeedForward (Preferred Implementation)

**File**: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py` (lines 112-144)

```python
class FeedForward(nn.Module):
    """SwiGLU feedforward for diffusion models.
    
    Structure:
        input [B, S, D]
          ↓ (w13: fused gate+up)
        [B, S, 2*H]
          ↓ (SiluAndMul: element-wise gating)
        [B, S, H]
          ↓ (w2: down projection)
        output [B, S, D]
    
    Parameters:
        dim: Input/output dimension (e.g., 3840 for Z-Image)
        hidden_dim: Intermediate dimension (e.g., 10240 for Z-Image = 3840 * 8/3)
        quant_config: Optional quantization config (FP8, LoRA, etc.)
        prefix: Parameter name prefix for weight loading
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        
        # Fused gate+up projection: [dim] -> [hidden_dim, hidden_dim]
        # This is more efficient than two separate linear layers
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim, hidden_dim],  # Two outputs: [0]=gate, [1]=up
            bias=False,
            gather_output=False,  # Output remains distributed
            quant_config=quant_config,
            prefix=f"{prefix}.w13",  # Weight file: "w13.weight"
        )
        
        # Down projection: [hidden_dim] -> [dim]
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,  # Input is distributed from w13
            quant_config=quant_config,
            prefix=f"{prefix}.w2",  # Weight file: "w2.weight"
        )
        
        # SwiGLU activation: x_gate * silu(x_up)
        # Takes [B, S, 2*H] and produces [B, S, H] via element-wise gating
        self.act = SiluAndMul()
    
    def forward(self, x):
        """
        Args:
            x: [B, S, D] input tensor
            
        Returns:
            [B, S, D] output tensor
        """
        # Step 1: Fused gate+up projection
        x13, _ = self.w13(x)  # [B, S, 2*H]
        
        # Step 2: SwiGLU gating (splits [B, S, 2*H] -> [B, S, H] * silu([B, S, H]))
        x = self.act(x13)  # [B, S, H]
        
        # Step 3: Down projection to original dimension
        out, _ = self.w2(x)  # [B, S, D]
        
        return out
```

**Key Differences from Llama FFN**:
- Uses **w13** and **w2** naming (from FLUX/diffusers convention)
- Explicitly uses `MergedColumnParallelLinear` for distributed training
- Returns two values (output + gradient info) for distributed training

---

### Example 2: Llama-Style MLP (LLM Convention)

**File**: `python/sglang/srt/models/llama.py` (snippet)

```python
class LlamaMLP(nn.Module):
    """Llama-style SwiGLU feedforward.
    
    Structure:
        input [B, S, H]
          ↓ (gate_up_proj: fused gate+up)
        [B, S, 2*I] (I = intermediate_size)
          ↓ (SiluAndMul: element-wise gating)
        [B, S, I]
          ↓ (down_proj: down projection)
        output [B, S, H]
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        reduce_results: bool = True,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        
        # Fused gate+up projection
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,  # Same as [intermediate_size, intermediate_size]
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        
        # Down projection
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
            reduce_results=reduce_results,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        
        # Activation: only SiLU is supported for now
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported.")
        self.act_fn = SiluAndMul()
    
    def forward(self, x):
        """Forward pass."""
        gate_up, _ = self.gate_up_proj(x)      # [B, S, 2*I]
        x = self.act_fn(gate_up)                # [B, S, I]
        x, _ = self.down_proj(x)                # [B, S, H]
        return x
```

**Key Differences from Z-Image FFN**:
- Uses **gate_up_proj** and **down_proj** naming (from Llama convention)
- Typically has `reduce_results` and `tp_rank`/`tp_size` parameters
- Often adds validation for `hidden_act` parameter

---

### Example 3: Generic FeedForward Layer

**File**: `python/sglang/multimodal_gen/runtime/layers/mlp.py` (lines 71-121)

```python
class FeedForward(nn.Module):
    """Generic feed-forward layer supporting multiple activation functions.
    
    This is a more flexible implementation that works with any gated activation,
    including SwiGLU, GEGLU, GELU, and others. Not distributed-aware.
    
    Parameters:
        dim: Input/output dimension
        dim_out: Output dimension (defaults to input dim)
        mult: Multiplier for hidden dimension (default: 4)
        activation_fn: Type of activation ("swiglu", "geglu", "gelu", etc.)
        inner_dim: Explicit hidden dimension (overrides mult if provided)
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        # Choose activation function
        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":  # ← For diffusion models
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")
        else:
            raise ValueError(f"Unknown activation: {activation_fn}")

        # Build module list: activation → dropout → output
        self.net = nn.ModuleList([])
        self.net.append(act_fn)
        self.net.append(nn.Dropout(0.0))
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
```

**Use Cases**:
- Single-GPU inference/training
- Research experiments
- Models that don't use distributed training

---

## Weight Loading & Parameter Mapping

### Z-Image Config Mappings

**File**: `python/sglang/multimodal_gen/configs/models/dits/zimage.py`

```python
@dataclass
class ZImageArchConfig(DiTArchConfig):
    # ... other config ...
    
    # How to shard w13 (for distributed training)
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            # (param_name, shard_name, shard_id)
            (".feed_forward.w13", ".feed_forward.w1", "gate"),  # shard 0/2
            (".feed_forward.w13", ".feed_forward.w3", "up"),    # shard 1/2
        ]
    )
    
    # How to load weights from different checkpoint formats
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # Load separate w1/w3 weights from HF checkpoint into merged w13
            r"(.*)\.feed_forward\.w1\.weight$": (
                r"\1.feed_forward.w13.weight",  # target name
                0,                              # shard_id
                2,                              # total_shards
            ),
            r"(.*)\.feed_forward\.w3\.weight$": (
                r"\1.feed_forward.w13.weight",
                1,  # shard_id = 1 means it's the second half
                2,
            ),
            
            # FP8 scales: block-quantized
            r"(.*)\.feed_forward\.w1\.weight_scale_inv$": (
                r"\1.feed_forward.w13.weight_scale_inv", 0, 2),
            r"(.*)\.feed_forward\.w3\.weight_scale_inv$": (
                r"\1.feed_forward.w13.weight_scale_inv", 1, 2),
            
            # FP8 scales: per-tensor
            r"(.*)\.feed_forward\.w1\.weight_scale$": (
                r"\1.feed_forward.w13.weight_scale", 0, 2),
            r"(.*)\.feed_forward\.w3\.weight_scale$": (
                r"\1.feed_forward.w13.weight_scale", 1, 2),
            
            # LoRA adapters
            r"(.*)\.feed_forward\.w1\.(lora_A|lora_B)$": (
                r"\1.feed_forward.w13.\2", 0, 2),
            r"(.*)\.feed_forward\.w3\.(lora_A|lora_B)$": (
                r"\1.feed_forward.w13.\2", 1, 2),
        }
    )
```

**How Mapping Works**:
1. Load checkpoint with `w1.weight` and `w3.weight`
2. Map them to `w13.weight[0]` and `w13.weight[1]` respectively
3. Concatenate: `w13.weight = cat([w1.weight, w3.weight], dim=0)`
4. Same process for FP8 scales and LoRA adapters

---

## Activation Functions: SiluAndMul

**File**: `python/sglang/multimodal_gen/runtime/layers/activation.py`

```python
class SiluAndMul(nn.Module):
    """SwiGLU activation: gated linear unit using SiLU gating.
    
    Takes a tensor [B, S, 2*H] and splits it into two halves:
    - First half [B, S, H]: multiplied with SiLU of second half
    - Second half [B, S, H]: passed through SiLU activation
    
    Output: [B, S, H] = x[:, :, :H] * SiLU(x[:, :, H:])
    
    This is the activation function used in SwiGLU layers.
    """
    
    def forward(self, x):
        """
        Args:
            x: [B, S, 2*H] tensor (concatenated gate and up projections)
            
        Returns:
            [B, S, H] tensor after gating
        """
        # Split into two halves
        x, gates = x.chunk(2, dim=-1)  # Each is [B, S, H]
        
        # Apply SiLU to gates and multiply with main projection
        return x * torch.nn.functional.silu(gates)
```

---

## Common Usage Patterns

### Pattern 1: Creating Z-Image Model

```python
from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel
from sglang.multimodal_gen.configs.models.dits.zimage import ZImageDitConfig

# Create config
config = ZImageDitConfig()

# Create model
model = ZImageTransformer2DModel(
    config=config,
    hf_config={...},
    quant_config=None,  # or NunchakuConfig(...) for quantization
)

# The model contains multiple FeedForward layers in:
# - model.layers[i].feed_forward
# - model.noise_refiner[i].feed_forward
# - model.context_refiner[i].feed_forward
```

### Pattern 2: Creating Llama Model

```python
from sglang.srt.model_loader import load_model

# Load pretrained model (auto-detects architecture)
model = load_model(model_path="meta-llama/Llama-2-7b")

# Access MLP layers
for layer in model.model.layers:
    mlp = layer.mlp  # Instance of LlamaMLP
    # mlp.gate_up_proj and mlp.down_proj are the FFN layers
```

### Pattern 3: Using Generic FeedForward

```python
from sglang.multimodal_gen.runtime.layers.mlp import FeedForward

# Create SwiGLU feedforward
ff = FeedForward(
    dim=768,
    dim_out=768,
    mult=4,  # hidden_dim = 768 * 4 = 3072
    activation_fn="swiglu",
)

# Forward pass
output = ff(input_tensor)  # [B, S, 768]
```

---

## Quantization Support

### Z-Image with FP8

```python
from sglang.multimodal_gen.runtime.layers.quantization import NunchakuConfig

# Enable FP8 quantization
quant_config = NunchakuConfig(
    precision="int4",
    rank="r16",
    act_unsigned=False,
)

# Create model with quantization
model = ZImageTransformer2DModel(
    config=config,
    hf_config={...},
    quant_config=quant_config,
)

# The FFN layers will automatically quantize:
# - w13 weights
# - w2 weights
# - Weight scales (both block-quantized and per-tensor)
```

---

## Performance Optimization Tips

### 1. Tensor Parallelism
```python
# For distributed training/inference
# Set TP size to split across GPUs
```

### 2. Sequence Parallelism
```python
# Used in Z-Image for video generation
# Splits sequence dimension across GPUs
```

### 3. Quantization
```python
# FP8 quantization reduces memory and improves throughput
# Especially beneficial for large models like Z-Image
```

### 4. Sparse Attention (TurboWan)
```python
# Uses sparse linear attention instead of dense
# Reduces computation for video generation
```

---

## References

- **Z-Image Model**: https://github.com/Chaofan-Zheng/Z-Image
- **FLUX Model**: https://github.com/black-forest-labs/flux
- **Llama Model**: https://github.com/meta-llama/llama
- **Diffusers**: https://github.com/huggingface/diffusers

---

## Key Takeaways

1. **w13 vs gate_up_proj**: Same thing, different naming conventions
   - Z-Image/FLUX: `w13` + `w2`
   - Llama/Xverse: `gate_up_proj` + `down_proj`

2. **Fused projections**: Always use `MergedColumnParallelLinear` for efficiency
   - Reduces memory footprint
   - Improves cache locality
   - Faster training/inference

3. **SwiGLU activation**: Standard for modern diffusion models
   - Output gate: `x[:, :, :H] * silu(x[:, :, H:])`
   - Used in Z-Image, FLUX, Qwen, and others

4. **Parameter mapping**: Critical for loading checkpoints
   - Maps `w1/w3` → `w13` during loading
   - Handles FP8 scales and LoRA adapters
   - Enables distributed sharding

5. **Quantization support**: Essential for production
   - FP8 block-quantized and per-tensor scales
   - LoRA fine-tuning
   - Automatic scale migration
