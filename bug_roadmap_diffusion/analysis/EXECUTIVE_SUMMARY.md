# GLM-Image Optimization: Executive Summary

## Current Position

**Roadmap Category**: "Improve performance for diffusers backend"  
**Status**: ✅ Benchmarking Complete → 🔄 Profiling & Optimization Phase

## Key Findings from Benchmark

- SGLang-D is **8-13% faster** than Diffusers (baseline established)
- Multi-GPU shows **1.19x-1.27x speedup** but has room for improvement
- **Sequence Parallelism missing** - identified as key optimization target
- P99 latency inconsistencies need investigation

## Next Steps (Following "Two Months In" Workflow)

### Immediate (This Week)
1. **Profiling** - Use PyTorch Profiler & Nsight Systems to identify bottlenecks
2. **Test Cache-DiT** - Enable and benchmark (potential 169% speedup)
3. **Analyze P99 Issues** - Investigate multi-GPU tail latency

### Short-term (2 Weeks)
1. Complete profiling report with prioritized optimization targets
2. Create SP integration plan for GLM-Image

### Medium-term (1 Month)
1. Implement top kernel optimizations (QKV, Norm, RoPE, Weight Fusion)
2. Integrate Sequence Parallelism support

## Target Improvement

- **Current**: 8-13% faster than Diffusers
- **Target**: 2-3x faster (following "Two Months In" 2.5x improvement pattern)

## Workflow Reference

Follow the "Two Months In" approach:
1. **Profile First** → Identify bottlenecks
2. **Targeted Optimization** → Focus on specific issues
3. **Leverage Existing** → Use Cache-DiT, Layerwise Offload, etc.
4. **Validate** → Benchmark at each stage

---

See `GLM_Image_Position_and_Next_Steps.md` for detailed analysis.
