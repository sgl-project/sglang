# GLM-Image Benchmark Position & Next Steps Analysis

## Current Position in Roadmap

### Roadmap Category
**"Improve performance for diffusers backend"** (Issue #16642)

### Current Status
- ✅ **Phase 1: Benchmarking** - Completed
  - SGLang-D vs Diffusers baseline comparison
  - Single GPU vs Multi-GPU comparison
  - Performance metrics collected across 8 configurations

- 🔄 **Phase 2: Analysis** - In Progress
  - Identified potential bottlenecks (Sequence Parallelism missing)
  - Multi-GPU shows 1.19x-1.27x speedup but room for improvement
  - P99 latency inconsistencies in some configurations

### Relationship to Roadmap Items

| Roadmap Item | Relevance to GLM-Image | Status |
|:-------------|:----------------------|:-------|
| **Kernel optimizations** | High - Norm kernels, QKV processing | Not started |
| **Sequence Parallelism** | Critical - Identified as missing | Not started |
| **Layerwise Offload** | Medium - If GLM-Image is multi-DiT | To investigate |
| **Cache-DiT Integration** | High - Can test with GLM-Image | To test |
| **Parallel VAE decoding** | Medium - Could help overall pipeline | Not started |
| **Profiling Suite** | Critical - Need to identify bottlenecks | Next step |

---

## Workflow Reference: "Two Months In" Approach

Based on the successful improvements in "Two Months In", the workflow follows this pattern:

1. **Profiling First** - Identify bottlenecks through systematic profiling
2. **Targeted Optimization** - Focus on specific bottlenecks
3. **Integration** - Leverage existing optimizations (Cache-DiT, etc.)
4. **Validation** - Benchmark before/after improvements

### Key Improvements from "Two Months In"

#### 1. Layerwise Offload
- **Process**: Profiling → Identify bottleneck → Implement solution
- **For GLM-Image**: Need to check if GLM-Image benefits from layerwise offload
- **Action**: Profile model loading/offloading patterns

#### 2. Kernel Improvements
- **Process**: Analyze performance trade-offs → Implement optimized kernels
- **For GLM-Image**: 
  - QKV processing optimization
  - QK Norm fusion
  - RoPE optimization
  - Weight fusion patterns
- **Action**: Profile attention and DiT block operations

#### 3. Cache-DiT Integration
- **Process**: Test with environment variables → Validate performance gain
- **For GLM-Image**: 
  - Test `SGLANG_CACHE_DIT_ENABLED=true`
  - Measure speedup (up to 169% in other models)
- **Action**: Enable and benchmark Cache-DiT for GLM-Image

#### 4. Profiling Suite
- **Process**: Use PyTorch Profiler and Nsight Systems
- **For GLM-Image**: 
  - Full-stage profiling
  - Identify compute vs memory bottlenecks
  - Find synchronization points
- **Action**: Run comprehensive profiling

---

## Recommended Next Steps (Following "Two Months In" Workflow)

### Phase 1: Profiling & Analysis (Immediate)

#### 1.1 Comprehensive Profiling
**Goal**: Identify specific bottlenecks in GLM-Image inference

**Tools**:
- PyTorch Profiler (step-by-step docs available)
- Nsight Systems (full-stage support)

**Focus Areas**:
- Attention operations (QKV processing, RoPE, attention compute)
- DiT block operations (norm, projection, activation)
- Model loading/offloading patterns
- Memory access patterns
- GPU utilization

**Deliverable**: Profiling report identifying top bottlenecks

#### 1.2 Sequence Parallelism Analysis
**Goal**: Understand why SP is missing and what's needed

**Tasks**:
- Review GLM-Image architecture
- Compare with models that have SP support (Wan2.2, Qwen-Image)
- Identify integration points
- Estimate performance gain potential

**Deliverable**: SP integration plan for GLM-Image

### Phase 2: Quick Wins (1-2 weeks)

#### 2.1 Cache-DiT Testing
**Action**: Enable Cache-DiT for GLM-Image

```bash
SGLANG_CACHE_DIT_ENABLED=true \
SGLANG_CACHE_DIT_SCM_PRESET=fast \
sglang serve --model-path zai-org/GLM-Image --backend sglang
```

**Benchmark**: Compare with baseline (expect up to 169% speedup)

#### 2.2 Layerwise Offload Investigation
**Action**: Check if GLM-Image benefits from layerwise offload
- Profile model loading patterns
- Check if it's multi-DiT architecture
- Test with layerwise offload enabled

### Phase 3: Kernel Optimizations (2-4 weeks)

Based on profiling results, prioritize:

#### 3.1 High Priority (if identified in profiling)
- **QKV Processing**: Optimize unpacking without extra contiguous operations
- **QK Norm Fusion**: Fuse Q/K RMSNorm into single inplace kernel
- **RoPE Optimization**: Apply RoPE inplace with FlashInfer
- **Weight Fusion**: Fuse projection + activation patterns

#### 3.2 Medium Priority
- **Timestep Kernel**: Dedicated CUDA kernel for sinusoidal embedding
- **FlashAttention**: Sync with latest upstream version

### Phase 4: Sequence Parallelism Integration (4-8 weeks)

#### 4.1 Implementation
- Integrate SP support for GLM-Image
- Test with different SP configurations
- Optimize communication patterns

#### 4.2 Validation
- Benchmark SP vs non-SP
- Test with different resolutions
- Validate correctness

### Phase 5: Advanced Optimizations (Ongoing)

#### 5.1 Parallel VAE Decoding
- If VAE is a bottleneck, implement parallel decoding

#### 5.2 CUDA Graph
- For small models like GLM-Image, CUDA graph can help
- Test and validate

---

## Specific Recommendations for GLM-Image

### Immediate Actions (This Week)

1. **Run Profiling**
   ```bash
   # Use PyTorch Profiler
   # Use Nsight Systems
   # Profile full inference pipeline
   ```

2. **Test Cache-DiT**
   ```bash
   SGLANG_CACHE_DIT_ENABLED=true \
   SGLANG_CACHE_DIT_SCM_PRESET=fast \
   # Run benchmark and compare
   ```

3. **Analyze P99 Latency Issues**
   - Investigate why P99 is worse in some multi-GPU configurations
   - Check for synchronization issues
   - Profile tail latency

### Short-term (Next 2 Weeks)

1. **Complete Profiling Report**
   - Document all bottlenecks
   - Prioritize optimization targets
   - Estimate potential gains

2. **SP Integration Plan**
   - Technical design document
   - Implementation roadmap
   - Testing strategy

### Medium-term (Next Month)

1. **Implement Top Kernel Optimizations**
   - Based on profiling results
   - Focus on highest-impact items

2. **SP Integration**
   - Implement and test
   - Benchmark improvements

---

## Success Metrics

Following "Two Months In" approach, target improvements:

- **Baseline**: Current 8-13% faster than Diffusers
- **Target**: 2-3x faster (similar to "Two Months In" 2.5x improvement)
- **Key Metrics**:
  - Latency reduction: 50-70%
  - Throughput increase: 2-3x
  - Memory efficiency: Maintain or improve
  - P99 latency: Consistent improvements

---

## Resources

- **Profiling Docs**: `/python/sglang/multimodal_gen/docs/profiling.md`
- **Cookbook**: Diffusion Cookbook (best practices and benchmarking guides)
- **Related PRs**: 
  - Layerwise Offload: #15511, #16150
  - Kernel Improvements: #16382, #12995
  - Cache-DiT: #14234, #15163, #16532
- **Slack Channel**: #diffusion

---

## Notes

- Follow the same systematic approach as "Two Months In"
- Profile first, optimize second
- Leverage existing optimizations (Cache-DiT, etc.)
- Document all findings and improvements
- Benchmark at each stage to validate progress
