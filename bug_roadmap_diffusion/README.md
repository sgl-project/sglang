# SGLang-Diffusion Roadmap & Related Issues

## Overview

This directory contains the roadmap and related documentation for SGLang-Diffusion development.

## Files

- **SGLang-Diffusion_Two_Months_In.md** - Summary of progress from Nov 2025 to Jan 2026
- **Roadmap_26Q1.md** - Q1 2026 development roadmap (Issue #18286)

## Related Issues

### Issue #18077: GLM-Image Performance Benchmark

**Location**: `/data/users/yandache/workspaces/sglang/repo/sglang-src/bug_18077_analysis/`

**Context**: This issue is part of the roadmap's performance improvement efforts, specifically focused on optimizing GLM-Image model performance in SGLang-Diffusion.

**Relationship to Roadmap**:
- Part of **"Performance Improvements"** section
- Related to **"Improve performance for diffusers backend"** (Issue #16642)
- Initial benchmarking and analysis phase for GLM-Image optimization

**Status**: 
- ✅ Benchmarking completed (SGLang-D vs Diffusers, Single GPU vs Multi-GPU)
- 🔄 Optimization phase (identifying bottlenecks, Sequence Parallelism integration)

**Key Findings**:
- SGLang-D shows 8-13% faster than Diffusers on GLM-Image
- Multi-GPU (2 GPUs) shows 1.19x-1.27x speedup over single GPU
- Room for further optimization, especially Sequence Parallelism support

## Roadmap Categories

### Performance Improvements
- **Lossless**: Kernel optimizations, CUDA graphs, parallel VAE decoding
- **Lossy**: Quantization, sparse attention backends

### Features
- Postprocessing (frame interpolation, upscaling)
- LoRA improvements
- Distillation support

### Platform & Backend
- Diffusers backend improvements
- MacOS support
- Consumer-level GPU optimizations

## Notes

Bug 18077 serves as the initial benchmarking and analysis phase for GLM-Image performance optimization, which aligns with the broader roadmap goals of improving model performance and backend efficiency.
