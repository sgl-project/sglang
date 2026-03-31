# ZImage-Turbo Code References & Line Numbers

## File Location
`/data/home/rhyshen/sgl-workspace/sglang/python/sglang/multimodal_gen/runtime/models/dits/zimage.py`

## Key Classes and Their Implementations

### 1. ZImageTransformerBlock Class
- **Definition**: Lines 366-516
- **__init__**: Lines 367-441
  - Attention setup: Line 385-393
  - skip_sequence_parallel for context_refiner: Lines 394-397
  - FeedForward setup: Lines 399-430
  - RMSNorm setup: Lines 432-436
  - AdaLN modulation setup (modulation=True only): Lines 438-441

- **forward()**: Lines 443-516
  - Method signature: Lines 443-450
  - Modulation path (modulation=True): Lines 451-487
    - AdaLN processing: Lines 453-460
    - Attention block: Lines 462-476
    - FFN block: Lines 478-487
  - Non-modulation path (modulation=False): Lines 488-514
    - Attention block: Lines 489-503
    - FFN block: Lines 505-514

### 2. FinalLayer Class
- **Definition**: Lines 519-544
- **__init__**: Lines 520-531
- **forward()**: Lines 533-544
  - AdaLN modulation: Lines 535-538
  - Normalization & linear: Lines 539-542

### 3. TimestepEmbedder Class
- **Definition**: Lines 66-112
- **__init__**: Lines 67-84
- **timestep_embedding() static method**: Lines 86-101
  - Frequency embedding logic: Lines 87-101
- **forward()**: Lines 103-112
  - Timestep processing: Lines 104-111
  - Returns: [B, out_size] tensor (out_size=256)

### 4. ZImageAttention Class
- **Definition**: Lines 156-363
- **__init__**: Lines 157-256
  - USPAttention setup: Lines 242-256
- **forward()**: Lines 258-363
  - QKV projection: Lines 265-286
  - QK norm: Lines 288-298
  - RoPE application: Lines 300-317
  - Attention computation: Lines 319-357
  - Output projection: Lines 359-361

### 5. RopeEmbedder Class
- **Definition**: Lines 547-612
- **__init__**: Lines 548-562
- **precompute_freqs() static method**: Lines 564-580
- **__call__()**: Lines 582-612

### 6. ZImageTransformer2DModel Class
- **Definition**: Lines 615-996
- **get_nunchaku_quant_rules()**: Lines 626-645
- **__init__()**: Lines 647-759
  - noise_refiner setup: Lines 692-707
  - context_refiner setup: Lines 708-723
  - t_embedder setup: Lines 724-726
  - cap_embedder setup: Lines 728-731
  - main layers setup: Lines 736-750

- **unpatchify()**: Lines 761-778
  - Spatial reconstruction: Lines 768-777

- **forward()**: Lines 853-995
  - Timestep embedding: Lines 875-878
  - **ISSUE**: Line 877 - `t.type_as(x)` where x is List[Tensor]
  - Patchify & embed: Lines 880-896
  - Noise refiner loop: Lines 898-905
  - Caption embed: Lines 907-915
  - Context refiner loop: Lines 917-927
  - Unified concatenation: Lines 929-947
  - Main transformer loop: Lines 949-976
  - Final layer: Lines 978-982
  - Unpatchify: Lines 984-992

## Key Methods & Functions

### Layer Construction
- **noise_refiner**: Lines 692-707 (ModuleList of 2 ZImageTransformerBlock)
- **context_refiner**: Lines 708-723 (ModuleList of 2 ZImageTransformerBlock)
- **layers**: Lines 736-750 (ModuleList of 30 ZImageTransformerBlock)

### Forward Flow Control
- **skip_sequence_parallel assignment (init)**: Line 397
- **skip_sequence_parallel assignment (dynamic)**: Lines 961-962, 966
- **use_full_unified_sequence check**: Line 931-933

## Constants

| Name | Value | Line |
|------|-------|------|
| ADALN_EMBED_DIM | 256 | 54 |
| SEQ_MULTI_OF | 32 | 55 |

## Configuration Keys

From `ZImageArchConfig` (zimage.py config file):

| Parameter | Default | Type | Line |
|-----------|---------|------|------|
| dim | 3840 | int | 27 |
| num_layers | 30 | int | 28 |
| n_refiner_layers | 2 | int | 29 |
| num_attention_heads | 30 | int | 30 |
| n_kv_heads | 30 | int | 31 |
| norm_eps | 1e-5 | float | 32 |
| qk_norm | True | bool | 33 |
| cap_feat_dim | 2560 | int | 34 |
| rope_theta | 256.0 | float | 35 |
| axes_dims | (32, 48, 48) | Tuple | 37 |
| axes_lens | (1024, 512, 512) | Tuple | 38 |

## Import Dependencies

Key imports for understanding:
- Line 20-22: Attention layers (UlyssesAttention, USPAttention)
- Line 24-28: Linear layers with parallelism
- Line 37-39: RoPE (Rotary Position Embeddings)

## Method Signatures

```python
# ZImageTransformerBlock.forward
def forward(
    self,
    x: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    adaln_input: Optional[torch.Tensor] = None,
    num_replicated_prefix: int = 0,
    num_replicated_suffix: int = 0,
) -> torch.Tensor:  # Lines 443-450

# FinalLayer.forward
def forward(self, x, c):  # Lines 533

# ZImageTransformer2DModel.unpatchify
def unpatchify(
    self, x: List[torch.Tensor], size: List[Tuple], patch_size, f_patch_size
) -> List[torch.Tensor]:  # Lines 761-762

# ZImageTransformer2DModel.forward
def forward(
    self,
    hidden_states: List[torch.Tensor],
    encoder_hidden_states: List[torch.Tensor],
    timestep,
    guidance=0,
    patch_size=2,
    f_patch_size=1,
    freqs_cis=None,
    **kwargs,
):  # Lines 853-863
```

## Critical Code Sections

### AdaLN Modulation (With modulation=True)
```python
# Lines 453-460
scale_msa_gate, _ = self.adaLN_modulation(adaln_input)
scale_msa, gate_msa, scale_mlp, gate_mlp = scale_msa_gate.unsqueeze(
    1
).chunk(4, dim=2)
gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp
```

### Attention with Residual (modulation=True)
```python
# Lines 464-474
nvtx.range_push(f"Layer{self.layer_id}::attn_pre_norm")
attn_in = self.attention_norm1(x) * scale_msa
nvtx.range_pop()
attn_out = self.attention(
    attn_in,
    freqs_cis=freqs_cis,
    num_replicated_prefix=num_replicated_prefix,
    num_replicated_suffix=num_replicated_suffix,
)
nvtx.range_push(f"Layer{self.layer_id}::attn_post_norm_residual")
x = x + gate_msa * self.attention_norm2(attn_out)
nvtx.range_pop()
```

### Forward Type Casting Bug (Line 877)
```python
x = hidden_states  # Line 868: List[torch.Tensor]
...
t = self.t_embedder(t)  # Line 876: torch.Tensor
adaln_input = t.type_as(x)  # Line 877: ISSUE - x is a list!
```

### Unpatchify Tensor Reshaping (Lines 768-777)
```python
ori_len = (F // pF) * (H // pH) * (W // pW)
# "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
x[i] = (
    x[i][:ori_len]
    .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
    .permute(6, 0, 3, 1, 4, 2, 5)
    .reshape(self.out_channels, F, H, W)
)
```

### Main Loop with Cache-DiT Support (Lines 958-975)
```python
if hasattr(layer, "transformer_blocks"):
    # CachedBlocks wrapper — set on all original blocks inside
    for block in layer.transformer_blocks:
        block.attention.attn.skip_sequence_parallel = (
            use_full_unified_sequence
        )
else:
    # Normal ZImageTransformerBlock
    layer.attention.attn.skip_sequence_parallel = (
        use_full_unified_sequence
    )
unified = layer(
    unified,
    unified_freqs_cis,
    adaln_input,
    num_replicated_suffix=num_replicated_suffix,
)
```

## Execution Order in forward()

1. **Lines 875-878**: Timestep embedding
2. **Lines 880-896**: Patchify & embed + noise refiner
3. **Lines 907-927**: Caption embedding & context refiner  
4. **Lines 929-947**: Unified concatenation setup
5. **Lines 949-976**: Main transformer loop (30 blocks)
6. **Lines 978-982**: Final layer
7. **Lines 984-992**: Unpatchify
8. **Line 995**: Return negated output

## Distributed Parallelism Flags

Set in main forward loop (lines 958-968):
- `skip_sequence_parallel`: Dynamic based on `use_full_unified_sequence`
- `use_full_unified_sequence`: Determined by `get_sp_world_size() > 1 and get_ring_parallel_world_size() > 1`
- `num_replicated_suffix`: Set to `cap_seq_len` when not using full unified sequence

## Notes on Cache-DiT Integration

Lines 952-963 handle Cache-DiT wrapping:
- Checks for `layer.transformer_blocks` attribute
- If present, sets `skip_sequence_parallel` on each wrapped block
- If absent, sets on the ZImageTransformerBlock directly
