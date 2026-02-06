# [Roadmap] SGLang-Diffusion (26 Q1) #18286

**Issue**: [#18286](https://github.com/sgl-project/sglang/issues/18286)  
**Status**: 0 / 2 issues completed  
**Opened**: 2 days ago (Feb 2026)

---

## Performance Improvements

### Lossless

- **Cuda graph for small models** (e.g., Qwen-Image) - @zyksir
- **Parallel vae decoding**
- **Kernel optimizations**
  - Norm related kernel develop
  - Rope_apply_head_dim fuse for MOVA
  - Non per-head QK Norm Fuse
- **B200 Kernel Optimization** - @HydraQYH
  - Apply 256B load/store to diffusion kernel in SM>=100 GPU
- **TBD**
  - Better comm kernels for USP

### Lossy

- **Quantization** - @fsygd @RubiaCx
  - Dtype: (E4M3, E5M2), INT8, MXFP8, NVFP4, MXFP4
  - Granularity: Token, Block, Tensor
  - Scale: E8M0 for MXFP8
  - Kernel: SageAttn for B200 and Other Linear
  - Frameworks: Nunchaku
- **More Sparse Attention Backends** (svg and others)

---

## Features

- @yhyang201 @mickqian
  - **Postprocessing**
    - Support frame interpolation
    - Support upscaling

### Misc

- @RubiaCx
  - **Distillation support**
- **LoRA** - @niehen6174
  - Coverage
  - Performance

---

## New Models

- @yhyang201

---

## Refine ComfyUI plugin for SGLang-Diffusion

- @niehen6174

---

## Improve performance for diffusers backend

- @DefTruth
- **GLM-Image Performance Optimization** - See Issue [#18077](https://github.com/sgl-project/sglang/issues/18077)
  - Initial benchmarking completed (SGLang-D vs Diffusers, Single GPU vs Multi-GPU)
  - Focus: Sequence Parallelism integration, kernel optimizations for GLM-Image

---

## Improve LayerwiseOffloadManager

---

## Platform Support: MacOS

---

## Optimizations for consumer-level GPUs

- @ryang-max

---

## CI

- consistency and accuracy tests

---

## Community Contributions

We welcome all forms of community contribution — from bug reports and documentation improvements to kernel optimizations, model integration, and new feature ideas.

You're welcomed to take over any items unassigned listed above by contacting with the POC (via GitHub or Slack)

If you're interested in participating, please check:

- Slack channel: #diffusion
- Cookbook for SGLang-Diffusion
- Documentation on SGLang-Diffusion
- SGLang-Diffusion: Two Months In

Your improvements — no matter small or large — can help make SGLang-Diffusion serving faster, easier, and more versatile.

---

## Sub-issues

- [ ] diffusion, parallelism: VAE Decode Parallel #13191
- [ ] [diffusion] Postprocess: frame interpolation and upscaling #18327

---

## Activity Log

- **mickqian** self-assigned this 2 days ago
- **mickqian** added sub-issues 18 hours ago
- **Ratish1** commented yesterday: "Hey @mickqian , I'd like to contribute to Parallel vae decoding. Thanks"
- **qimcis** commented 19 hours ago: "I'd love to pick up some of the kernel optimization work, can you assign me to the norm related kernel development?"
- **yingluosanqian** commented 12 hours ago: Offered existing kernel work for norm fusion
- **qimcis** commented 4 hours ago: "I'd be happy to take this over, I'll message you on slack for details once I get started!"
