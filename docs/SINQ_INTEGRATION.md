# SINQ Integration Strategy for SGLang

## Overview

SINQ (Sinkhorn-Normalized Quantization) from `huawei-csl/SINQ` is a **calibration-free** quantization method. Unlike AWQ/GPTQ which require ~20 minutes of data calibration, SINQ calculates scales analytically, enabling instant quantization.

## Value Proposition

| Method | Calibration Time | Requires Dataset | Online Serving |
|--------|-----------------|------------------|----------------|
| AWQ    | ~20 min         | Yes              | No             |
| GPTQ   | ~30 min         | Yes              | No             |
| SINQ   | ~30 sec         | No               | **Yes**        |

## Integration Strategies

### Strategy A: Just-In-Time Quantization (Recommended)

**Problem:** Users want to serve fine-tunes (e.g., `My-Llama-3-Finance`) in 4-bit mode, but no pre-quantized version exists.

**Solution:** Add `--quantization sinq` flag for instant quantization at load time.

```bash
# Current: Requires pre-quantized model
python -m sglang.launch_server --model my-finetune-awq --quantization awq

# Proposed: Instant quantization of any BF16 model
python -m sglang.launch_server --model my-finetune-bf16 --quantization sinq
```

**Implementation:**
1. Load BF16 weights from HuggingFace
2. Run SINQ transformation (~30s for 7B)
3. Serve in INT4 with existing kernels

**Files to modify:**
- `python/sglang/srt/model_executor/model_loader.py` - Add SINQ quantization path
- `python/sglang/srt/server_args.py` - Add "sinq" to quantization options

### Strategy B: Zero-Overhead Kernel Reuse

SINQ's key insight: Column scales can be absorbed into the previous layer's weights, producing standard W4A16 (Weight Int4, Activation FP16) format.

**Benefit:** No new CUDA kernels needed. Reuse existing Marlin/AWQ kernels.

**Implementation steps:**
1. Extract SINQ weight transformation logic (no training code)
2. Verify bit-packing format matches Marlin's expectation
3. Add repacking step in loader if needed

**Key SINQ functions to extract:**
```python
# From SINQ repo - weight transformation
def sinkhorn_normalize(weight, n_iters=10):
    """Doubly stochastic normalization for balanced quantization"""
    ...

def compute_scales(weight):
    """Analytical scale computation (no calibration data)"""
    ...
```

### Strategy C: Validation via Attention Visualization

Use the attention visualization tool to validate quantization quality:

1. Load model in BF16, capture attention patterns
2. Quantize with SINQ to INT4
3. Compare Top-K attention patterns between BF16 and INT4
4. If patterns diverge significantly → quantization too aggressive

**Metric:** Jaccard similarity of Top-K attended positions should be >0.8

## Implementation Phases

### Phase 1: Proof of Concept (1-2 days)
- [ ] Clone SINQ repo, extract transformation logic
- [ ] Test on single Llama-7B layer
- [ ] Verify output matches expected W4A16 format

### Phase 2: Integration (2-3 days)
- [ ] Add `sinq` option to `--quantization` flag
- [ ] Integrate transformation into model loader
- [ ] Verify compatibility with Marlin kernels

### Phase 3: Validation (1 day)
- [ ] Run perplexity benchmarks (WikiText, C4)
- [ ] Run attention pattern comparison
- [ ] Document any model-specific issues

## Code Location

SINQ transformation should be added to:
```
python/sglang/srt/
├── quantization/
│   ├── sinq/
│   │   ├── __init__.py
│   │   ├── transform.py      # Core transformation logic
│   │   └── utils.py          # Scale computation
│   └── ...
└── model_executor/
    └── model_loader.py       # Integration point
```

## References

- SINQ Repository: https://github.com/huawei-csl/SINQ
- Paper: "SINQ: Sinkhorn-Normalized Quantization for LLMs"
- Related: SGLang existing quantization in `python/sglang/srt/layers/quantization/`

## Notes

- Do NOT import SINQ's training code or custom Triton kernels
- Extract ONLY the weight transformation logic
- Target: Enable instant 4-bit serving for any HuggingFace model
