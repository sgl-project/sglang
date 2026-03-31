# ZImage-Turbo Architecture Analysis

## 1. ZImageTransformerBlock Class Definition

**Location:** `/data/home/rhyshen/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/zimage.py`, lines 366-516

### Class Structure
```python
class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,  # Critical: controls AdaLN vs standard LN
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
```

### Component Breakdown
- **self.attention**: `ZImageAttention` instance
  - Contains `USPAttention` (Ulysses Sequence Parallelism with Ring Attention)
  - Also contains `UlyssesAttention` for mixed cases
  
- **self.feed_forward**: `FeedForward` or `NunchakuFeedForward`
  - Hidden dimension: `int(dim / 3 * 8)` (expands by 8/3)
  
- **self.attention_norm1, self.attention_norm2**: `RMSNorm` layers
  
- **self.ffn_norm1, self.ffn_norm2**: `RMSNorm` layers
  
- **self.adaLN_modulation** (only if `modulation=True`):
  ```python
  nn.Sequential(
      ReplicatedLinear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True)
  )
  ```
  - Produces 4 scaling/gating parameters (scale_msa, gate_msa, scale_mlp, gate_mlp)

---

## 2. Forward Method Signature & Implementation

### Signature (lines 443-450)
```python
def forward(
    self,
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    adaln_input: Optional[torch.Tensor] = None,
    num_replicated_prefix: int = 0,
    num_replicated_suffix: int = 0,
) -> torch.Tensor:
```

### Implementation Details

#### With Modulation (`modulation=True`)
**Lines 451-487:**
1. **AdaLN Modulation** (line 454):
   - Takes `adaln_input` (required, asserted at line 452)
   - Outputs `scale_msa_gate` of shape `[B, 1, 4*dim]`
   - Splits into 4 components via `unsqueeze(1).chunk(4, dim=2)`:
     - `scale_msa`: [B, 1, dim]
     - `gate_msa`: [B, 1, dim]  
     - `scale_mlp`: [B, 1, dim]
     - `gate_mlp`: [B, 1, dim]
   - Applies: `gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()`
   - Applies: `scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp`

2. **Attention Block**:
   ```python
   attn_in = self.attention_norm1(x) * scale_msa
   attn_out = self.attention(
       attn_in,
       freqs_cis=freqs_cis,
       num_replicated_prefix=num_replicated_prefix,
       num_replicated_suffix=num_replicated_suffix,
   )
   x = x + gate_msa * self.attention_norm2(attn_out)
   ```

3. **FFN Block**:
   ```python
   ffn_in = self.ffn_norm1(x) * scale_mlp
   ffn_out = self.feed_forward(ffn_in)
   x = x + gate_mlp * self.ffn_norm2(ffn_out)
   ```

#### Without Modulation (`modulation=False`)
**Lines 488-514:**
- No scaling/gating parameters
- Pre-norm without multiplication
- Simple residual connections
- Used in `context_refiner` blocks

---

## 3. `attention.attn.skip_sequence_parallel` Usage

### Location & Purpose
**In `ZImageTransformerBlock.__init__` (lines 394-397):**
```python
if not modulation:
    # Context refiner runs on fully replicated caption tokens only.
    # Bypass Ulysses here to preserve the single-GPU attention semantics.
    self.attention.attn.skip_sequence_parallel = True
```

### What It Does
When `skip_sequence_parallel = True`:
- **Located in**: `USPAttention.forward()` line 383
- **Effect**: Bypasses all sequence parallelism communication
- **Condition**:
  ```python
  if self.skip_sequence_parallel or get_sequence_parallel_world_size() == 1:
      out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
      return out
  ```
- **Use Case**: Caption tokens (context_refiner) are replicated across all SP ranks, so no need for Ulysses all-to-all communication

### Additional Dynamic Setting (lines 958-968)
In main forward loop, `skip_sequence_parallel` is also set dynamically:
```python
if use_full_unified_sequence:
    # When unified sequence is fully gathered, no need for SP
    layer.attention.attn.skip_sequence_parallel = use_full_unified_sequence
else:
    # When using local sharding, enable SP
    layer.attention.attn.skip_sequence_parallel = False
```

---

## 4. `noise_refiner` and `context_refiner` Structures

### Architecture
Both are **lists of identical `ZImageTransformerBlock` instances**

**Lines 692-723:**
```python
self.noise_refiner = nn.ModuleList([
    ZImageTransformerBlock(
        1000 + layer_id,  # layer_id starts at 1000
        self.dim,
        self.n_heads,
        arch_config.n_kv_heads,
        arch_config.norm_eps,
        arch_config.qk_norm,
        modulation=True,  # USES AdaLN
        quant_config=quant_config,
        prefix=f"noise_refiner.{layer_id}",
    )
    for layer_id in range(arch_config.n_refiner_layers)
])

self.context_refiner = nn.ModuleList([
    ZImageTransformerBlock(
        layer_id,  # layer_id starts at 0
        self.dim,
        self.n_heads,
        arch_config.n_kv_heads,
        arch_config.norm_eps,
        arch_config.qk_norm,
        modulation=False,  # NO AdaLN, has skip_sequence_parallel=True
        quant_config=quant_config,
        prefix=f"context_refiner.{layer_id}",
    )
    for layer_id in range(arch_config.n_refiner_layers)
])
```

### Key Differences
| Aspect | noise_refiner | context_refiner |
|--------|---------------|-----------------|
| **modulation** | True | False |
| **adaln_input** | Required (passed from timestep_embedder) | Not passed (None) |
| **skip_sequence_parallel** | Not set in __init__ (dynamic later) | Set to True in __init__ |
| **Purpose** | Refine noisy image embeddings | Refine caption embeddings |

### Configuration
- **Number of layers per refiner**: `n_refiner_layers = 2` (from config, line 29)
- **Total refiner layers**: 4 (2 noise + 2 context)

---

## 5. FinalLayer Implementation

### Class Definition (lines 519-544)
```python
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = ColumnParallelLinear(
            hidden_size, out_channels, bias=True, gather_output=True
        )
        self.act = nn.SiLU()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            ReplicatedLinear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        # x: unified output from main transformer layers
        # c: adaln_input (timestep embedding)
        
        scale, _ = self.adaLN_modulation(c)
        scale = 1.0 + scale
        
        x = self.norm_final(x) * scale.unsqueeze(1)
        x, _ = self.linear(x)
        return x
```

### Inputs Required
- **x**: Shape `[B, seq_len, hidden_size]` - output from final main transformer block
- **c**: Shape `[B, 1, min(hidden_size, ADALN_EMBED_DIM)]` - `adaln_input` from `t_embedder`

### Key Points
- **DOES use `adaln_input`** (parameter `c`)
- **NO skip gate** (unlike transformer blocks which have gate_msa, gate_mlp)
- Only uses **scale** from AdaLN modulation
- `self.act = nn.SiLU()` is **defined but never used** (dead code)
- Output channels vary per resolution (set at lines 684-686)

### Usage (lines 978-982)
```python
unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](
    unified, adaln_input
)
```

---

## 6. `unpatchify` Method

### Implementation (lines 761-778)
```python
def unpatchify(
    self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size
) -> List[torch.Tensor]:
    pH = pW = patch_size
    pF = f_patch_size
    bsz = len(x)
    assert len(size) == bsz
    for i in range(bsz):
        F, H, W = size[i]
        ori_len = (F // pF) * (H // pH) * (W // pW)
        # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
        x[i] = (
            x[i][:ori_len]
            .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
            .permute(6, 0, 3, 1, 4, 2, 5)
            .reshape(self.out_channels, F, H, W)
        )
    return x
```

### Process
1. Processes each batch element in-place (modifies `x[i]`)
2. **Extracts valid length**: `ori_len = (F // pF) * (H // pH) * (W // pW)`
   - Removes padding tokens
3. **Reconstructs spatial dimensions**:
   - Input format: `[ori_len, pf*ph*pw*c]` (flattened patches)
   - View as: `[F//pF, H//pH, W//pW, pF, pH, pW, c]` (unroll patches)
   - Permute to: `[c, F//pF, pF, H//pH, pH, W//pW, pW]`
   - Reshape to: `[c, F, H, W]` (final video tensor)

### Output Format
- **Input**: List of flattened patch sequences
- **Output**: List of video tensors with shape `[out_channels, F, H, W]`

---

## 7. Total Main Transformer Blocks Count

### Configuration
**From** `ZImageArchConfig` (config file, line 28):
```python
num_layers: int = 30
```

### Construction (lines 736-750)
```python
self.layers = nn.ModuleList([
    ZImageTransformerBlock(
        layer_id,
        self.dim,
        self.n_heads,
        arch_config.n_kv_heads,
        arch_config.norm_eps,
        arch_config.qk_norm,
        quant_config=quant_config,
        prefix=f"layers.{layer_id}",
    )
    for layer_id in range(arch_config.num_layers)
])
```

### Total Layer Count Summary
- **Main transformer blocks** (`self.layers`): **30**
- **Noise refiner blocks** (`self.noise_refiner`): **2**
- **Context refiner blocks** (`self.context_refiner`): **2**
- **Total `ZImageTransformerBlock` instances**: **34**
- **Plus 1 FinalLayer**
- **Grand total**: **35 transformer/final blocks**

---

## 8. `t_embedder` Return Values

### Class Definition (lines 66-112)
```python
class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        # out_size = min(self.dim, ADALN_EMBED_DIM) = min(3840, 256) = 256
        # mid_size = 1024
        # frequency_embedding_size = 256 (default)
        
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(
            self.mlp[0].weight.dtype
        )  # [B, 256]
        t_emb, _ = self.mlp[0](t_freq)  # [B, 1024] (with TP, parallel output)
        t_emb = self.mlp[1](t_emb)       # [B, 1024] (SiLU activation)
        t_emb, _ = self.mlp[2](t_emb)    # [B, 256] (final output)
        return t_emb
```

### Construction (lines 724-726)
```python
self.t_embedder = TimestepEmbedder(
    min(self.dim, ADALN_EMBED_DIM),  # out_size = 256
    mid_size=1024
)
```

### Return Shape and Type
**Line 876-877:**
```python
t = self.t_embedder(t)
adaln_input = t.type_as(x)
```

| Aspect | Value |
|--------|-------|
| **Shape** | `[B, out_size]` = `[batch_size, 256]` |
| **Dtype** | Same as `mlp[0].weight.dtype` (typically BF16 or FP32 depending on autocast) |
| **Constant** | `ADALN_EMBED_DIM = 256` (line 54) |

---

## 9. `type_as(x)` Behavior

### Standard PyTorch Behavior
**Method**: `Tensor.type_as(other_tensor) -> Tensor`
- Casts the tensor to the same dtype as `other_tensor`

### Usage in ZImage (line 877)
```python
adaln_input = t.type_as(x)
```

### Problem & Reality
**⚠️ ISSUE**: At this point in forward(), `x` is a **`List[torch.Tensor]`** (from line 868)
```python
x = hidden_states  # List[torch.Tensor]
...
adaln_input = t.type_as(x)  # Calling .type_as() on a list!
```

**Behavior**:
- **If `x` is a list**: PyTorch's `type_as()` will raise `TypeError`
  - PyTorch expects a tensor, not a sequence
- **In actual execution**: This suggests either:
  1. There's a custom `type_as()` implementation (not found)
  2. This is **dead code** that should be `t.type_as(x[0])`
  3. The actual execution handles this differently

**Likely Fix**: Should be:
```python
adaln_input = t.type_as(x[0])  # Use first element of list
```

---

## 10. Segment Boundaries for V2 Spec Validation

### Architecture Segments
Based on execution flow and forward range pushes:

**Segment 1: Input Embedding & Refiners**
- Time embedding: `TimestepEmbedder`
- Patchify & embed: x_embedder + noise_refiner (2 blocks)
- Caption embedding: cap_embedder + context_refiner (2 blocks)
- **Total blocks**: 2 + 2 = 4

**Segment 2: Main Transformer**
- Unified concatenation: caption + image
- Main layers: 30 transformer blocks
- **Total blocks**: 30

**Segment 3: Output**
- Final layer: 1 FinalLayer
- Unpatchify
- **Total blocks**: 1

### Summary Table
| Component | Layer Count | Notes |
|-----------|-------------|-------|
| noise_refiner | 2 | Modulation=True, layer_id 1000-1001 |
| context_refiner | 2 | Modulation=False, skip_sp=True, layer_id 0-1 |
| main layers | 30 | Modulation=True, layer_id 0-29 |
| final_layer | 1 | Uses adaln_input, no gate |
| **TOTAL** | **35** | Architecture complete |

---

## Constants
- `ADALN_EMBED_DIM = 256` (line 54)
- `SEQ_MULTI_OF = 32` (line 55) - Sequence length multiple for padding
- Default config:
  - `dim = 3840`
  - `num_layers = 30`
  - `n_refiner_layers = 2`
  - `num_heads = 30`
  - `axes_dims = (32, 48, 48)` - RoPE dimensions
  - `axes_lens = (1024, 512, 512)` - RoPE sequence lengths

