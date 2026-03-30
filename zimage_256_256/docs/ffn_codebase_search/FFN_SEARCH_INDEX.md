# SGLang FFN Search Results Index

Generated: 2026-03-27

This document is a quick-access index to FFN-related code found in the SGLang codebase.

---

## File Listing

### Core FFN Implementations

| File Path | Lines | Content | Priority |
|-----------|-------|---------|----------|
| `python/sglang/multimodal_gen/runtime/models/dits/zimage.py` | 112-144 | **FeedForward class** (w13/w2 SwiGLU) | ⭐⭐⭐ |
| `python/sglang/multimodal_gen/configs/models/dits/zimage.py` | 42-89 | **param_names_mapping** (weight loading) | ⭐⭐⭐ |
| `python/sglang/multimodal_gen/runtime/layers/mlp.py` | 26-121 | **MLP, FeedForward** (generic layers) | ⭐⭐ |
| `python/sglang/srt/models/llama.py` | - | **LlamaMLP** (gate_up_proj/down_proj) | ⭐⭐⭐ |

### Model Variants Using gate_up_proj/down_proj

| File Path | Class | Notes |
|-----------|-------|-------|
| `python/sglang/srt/models/xverse.py` | XverseMLP | Xverse model FFN |
| `python/sglang/srt/models/grok.py` | GrokMLP | Grok model FFN |
| `python/sglang/srt/models/nemotron_nas.py` | NemotronNasMLP | Nemotron-NAS model FFN |
| `python/sglang/srt/models/phimoe.py` | PhiMoeMLP | Phi-MoE model FFN |
| `python/sglang/srt/models/qwen3_5_mtp.py` | Qwen3.5MTP | Qwen 3.5 model FFN |
| `python/sglang/srt/models/sarashina2_vision.py` | - | Sarashina2 FFN |
| `python/sglang/srt/models/glm4v_moe.py` | - | GLM4V-MoE FFN |
| `python/sglang/srt/models/gemma3_mm.py` | - | Gemma3 multimodal FFN |
| `python/sglang/srt/models/dots_vlm_vit.py` | - | DotS VLM ViT FFN |
| `python/sglang/srt/models/exaone4.py` | - | Exaone4 FFN |
| `python/sglang/srt/models/internvl.py` | - | InternVL FFN |

### Related Infrastructure

| File Path | Content | Purpose |
|-----------|---------|---------|
| `python/sglang/multimodal_gen/runtime/layers/linear.py` | MergedColumnParallelLinear, RowParallelLinear | Distributed linear layers |
| `python/sglang/multimodal_gen/runtime/layers/activation.py` | SiluAndMul, get_act_fn | Activation functions |
| `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py` | MinimalA2AAttnOp | TurboWan sparse attention |
| `python/sglang/multimodal_gen/configs/models/dits/zimage.py` | ZImageArchConfig, ZImageDitConfig | Z-Image model config |
| `python/sglang/multimodal_gen/configs/pipeline_configs/zimage.py` | ZImagePipelineConfig | Z-Image pipeline config |

### Testing & Examples

| File Path | Type | Content |
|-----------|------|---------|
| `test/registered/amd/test_zimage_turbo.py` | Test | Z-Image Turbo on AMD |
| `test/srt/cpu/test_qkv_proj_with_rope.py` | Test | QKV projection with RoPE |
| `test/srt/cpu/test_shared_expert.py` | Test | Shared expert (MoE) testing |
| `python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion/test/test_zimage_pipeline.py` | Test | Z-Image pipeline test |

---

## Search Query Results

### 1. w13, w2, gate_up_proj, down_proj References

**Search Query**: `w13|w2|gate_up_proj|down_proj`

**Files Found**: ~250 matches across:
- Weight initialization/loading code
- Model definitions
- Configuration mapping
- Distributed training setup

**Key Files**:
- ✅ `zimage.py` - Z-Image model (w13/w2)
- ✅ `llama.py` - Llama model (gate_up_proj/down_proj)
- ✅ `xverse.py` - Xverse model (gate_up_proj/down_proj)
- ✅ `zimage.py` config - Parameter mappings
- ✅ `cpu/test_qkv_proj_with_rope.py` - w2 quantization tests

### 2. SwiGLU and FFN References

**Search Query**: `SwiGLU|FFN`

**Files Found**: ~30 matches

**Key Files**:
- ✅ `mlp.py` - Generic SwiGLU support (line 106)
- ✅ `zimage.py` - Uses SiluAndMul (SwiGLU variant)
- ✅ Various model configs with FFN references
- ✅ Profiling and analysis documents

### 3. Zimage and Turbo References

**Search Query**: `zimage|turbo`

**Files Found**: ~78 matches

**Key Locations**:
- ✅ `python/sglang/multimodal_gen/runtime/models/dits/zimage.py` - Main model
- ✅ `python/sglang/multimodal_gen/configs/models/dits/zimage.py` - Model config
- ✅ `python/sglang/multimodal_gen/runtime/layers/attention/turbo_layer.py` - Sparse attention
- ✅ `test/registered/amd/test_zimage_turbo.py` - Test file
- ✅ Multiple analysis and profiling documents in `zimage_256_256/`

---

## Key Findings Summary

### Finding 1: Two Main FFN Naming Conventions
```
Diffusion Models (Z-Image, FLUX):
  w13 (gate + up) → activation → w2 (down)

LLM Models (Llama, Xverse):
  gate_up_proj (gate + up) → activation → down_proj (down)
```

### Finding 2: SwiGLU is the Standard Activation
- Used in: Z-Image, FLUX, Qwen, Llama, Xverse, etc.
- Implementation: `SiluAndMul()` class
- Formula: `x[:H] * silu(x[H:])`

### Finding 3: Parameter Mapping System
- Allows loading from separate w1/w3 into merged w13
- Handles FP8 scales (block and per-tensor)
- Supports LoRA adapter mapping
- Enables distributed sharding

### Finding 4: Distributed Training Support
- `MergedColumnParallelLinear`: Fused projections
- `RowParallelLinear`: Row-parallel down projection
- Tensor parallelism (TP) ready
- Sequence parallelism (SP) ready

### Finding 5: Quantization Integration
- FP8 block-quantized scales
- FP8 per-tensor scales
- LoRA fine-tuning support
- Nunchaku quantization framework

---

## Quick Access Patterns

### To Find FFN Code for a Specific Model

1. **Z-Image/Diffusion Models**:
   - Model: `python/sglang/multimodal_gen/runtime/models/dits/zimage.py`
   - Config: `python/sglang/multimodal_gen/configs/models/dits/zimage.py`
   - FFN Class: `FeedForward` (w13/w2)

2. **Llama-Based Models**:
   - Model: `python/sglang/srt/models/llama.py`
   - Config: `python/sglang/srt/configs/llama.py` (or auto-generated)
   - FFN Class: `LlamaMLP` (gate_up_proj/down_proj)

3. **Other Models**:
   - Search for `gate_up_proj` or `w13` in model files
   - Check model's config for parameter mappings

### To Understand Weight Loading

1. Start with: `python/sglang/multimodal_gen/configs/models/dits/zimage.py`
2. Look at: `param_names_mapping` dict
3. Find: How w1/w3 maps to w13
4. Check: FP8 scale and LoRA mappings

### To Find Activation Functions

1. Generic SwiGLU: `python/sglang/multimodal_gen/runtime/layers/mlp.py` line 106
2. Model-specific: `SiluAndMul()` class
3. Location: `python/sglang/multimodal_gen/runtime/layers/activation.py`

### To Find Distributed Training Code

1. `MergedColumnParallelLinear`: `python/sglang/multimodal_gen/runtime/layers/linear.py`
2. `RowParallelLinear`: Same file
3. Prefix handling: `add_prefix()` utility function

---

## Documentation Generated

### Summary Document
📄 `FFN_SEARCH_SUMMARY.md` (12 sections, ~500 lines)
- Complete overview of all FFN patterns
- Weight structure analysis
- Model variants catalog
- Quantization support details

### Code Reference Guide
📄 `FFN_CODE_REFERENCE.md` (13 sections, ~600 lines)
- Detailed code examples with annotations
- Three FFN implementation variants
- Weight loading and mapping examples
- Usage patterns and best practices
- Performance optimization tips

### This Index
📄 `FFN_SEARCH_INDEX.md` (this file)
- Quick navigation guide
- File listing with priorities
- Search results summary
- Key findings and access patterns

---

## How to Use These Documents

### For Learning FFN Structure
1. Start with `FFN_SEARCH_SUMMARY.md` (Section 1)
2. Read code examples in `FFN_CODE_REFERENCE.md`
3. Compare Z-Image vs Llama patterns

### For Finding Specific Code
1. Use `FFN_SEARCH_INDEX.md` to locate files
2. Go directly to file and line numbers
3. Reference `FFN_CODE_REFERENCE.md` for explanation

### For Implementing New FFN
1. Read `FFN_CODE_REFERENCE.md` (Patterns section)
2. Choose between w13/w2 or gate_up_proj/down_proj naming
3. Copy structure from similar model
4. Use param_names_mapping for weight loading

### For Debugging Quantization
1. Check `FFN_SEARCH_SUMMARY.md` (Section 5)
2. Review param_names_mapping in config
3. Verify FP8 scale naming conventions
4. Check LoRA adapter mappings

---

## Statistics

### Code Search Coverage
- Total files with FFN references: ~250
- Key implementation files: 4
- Model variants identified: 10+
- Test files: 4+

### Weight Variable Naming
- `w13`/`w2` pattern: Z-Image, FLUX, Qwen, etc.
- `gate_up_proj`/`down_proj` pattern: Llama, Xverse, Grok, etc.
- Alternative names found: w1/w3, gate_proj/up_proj

### Activation Types Supported
- SwiGLU (primary for diffusion)
- GEGLU
- GELU
- ApproximateGELU
- LinearActivation

---

## Related Files in Workspace

- `ANALYSIS_FILES.txt` - Previous analysis documentation
- `FP8_ANALYSIS_SUMMARY.md` - FP8 quantization analysis
- `FP8_DISPATCH_ANALYSIS.md` - FP8 dispatch analysis
- `PROFILING_ANALYSIS_SUMMARY.md` - Performance profiling
- `zimage_256_256/` - Z-Image profiling and analysis folder

---

## Version Info

- **Generated**: 2026-03-27
- **SGLang Repo**: `/data/home/rhyshen/sgl-workspace/sglang`
- **Python Version**: 3.x (torch/diffusers compatible)
- **Scope**: FFN implementations, naming conventions, weight loading, quantization

---

## Search Strategy Used

### Phase 1: Pattern Search
```bash
grep -r "w13|w2|gate_up_proj|down_proj" --include="*.py"
```

### Phase 2: Semantic Search
```bash
grep -r "SwiGLU|FFN" --include="*.py"
```

### Phase 3: Zimage/Turbo Search
```bash
grep -r "zimage|turbo" --include="*.py"
```

### Phase 4: File Globbing
```bash
find . -path "*/models/*.py" -o -path "*/layers/*.py"
```

### Phase 5: Context Extraction
- Examined: FeedForward classes (3 variants)
- Extracted: Parameter mappings
- Analyzed: Weight loading mechanisms
- Documented: Quantization integration

---

## Notes

- All line numbers are approximate and may shift with future commits
- File paths are relative to SGLang root
- Some test files may be deprecated; check git history
- Quantization features require optional dependencies (nunchaku, etc.)
- For production use, always verify with latest main branch

