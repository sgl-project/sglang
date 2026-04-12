<div align="center">
<img src="assets/sgl-1-bit-turbo-eagle.png" alt="SGLang 1-Bit Turbo" width="600"></img>
</div>


[![license](https://img.shields.io/github/license/sgl-project/sglang.svg)](https://github.com/carlosfundora/sglang-1-bit-turbo/blob/main/LICENSE)
---

## SGLang 1-Bit Turbo EAGLE — AMD ROCm Inference with Aggressive Quantization and Speculative Decoding

A fork of [SGLang](https://github.com/sgl-project/sglang) that brings high-performance inference to AMD consumer GPUs (RX 6000/7000 series) with integrated support for extreme KV cache compression, speculative decoding, and 1-bit quantized model serving.

---

## Core Features

### 🌀 RotorQuant KV Cache Compression
Geometric-rotation-based KV cache quantization from the [RotorQuant paper](https://github.com/scrya-com/rotorquant) by [Scrya](https://github.com/scrya-com). Data-oblivious (no calibration needed), dramatically faster than TurboQuant with comparable quality.

| Mode | Method | Bits | Rotation | Speed | Quality |
|------|--------|------|----------|-------|---------|
| `--kv-cache-dtype rq3_planar` | PlanarQuant | 3-bit | 2D Givens | ⚡ Fastest | Good |
| `--kv-cache-dtype rq4_planar` | PlanarQuant | 4-bit | 2D Givens | ⚡ Fastest | Better |
| `--kv-cache-dtype rq3_iso` | IsoQuant | 3-bit | 4D Quaternion | Fast | Best at 3-bit |
| `--kv-cache-dtype rq4_iso` | IsoQuant | 4-bit | 4D Quaternion | Fast | Best overall |

- **PlanarQuant**: 64× fewer FMAs than TQ (256 vs 16,384) — ~28% faster decode
- **IsoQuant**: 32× fewer FMAs (512 vs 16,384) — better quality than TQ at same bit-width
- HIP/ROCm compatible via Triton kernels (PlanarQuant) and PyTorch fallback (IsoQuant)

### 🧊 TurboQuant KV Cache Compression
Extreme KV cache quantization modes that dramatically reduce VRAM usage, enabling larger context and bigger models on memory-constrained GPUs.

| Mode | Bits per element | VRAM savings vs FP16 |
|------|-----------------|---------------------|
| `--kv-cache-dtype tq4` | 4-bit | ~75% |
| `--kv-cache-dtype tq3` | 3-bit | ~81% |
| `--kv-cache-dtype tq2` | 2-bit | ~87% |

### 🦅 EAGLE3 Speculative Decoding
Full integration of [EAGLE3](https://github.com/SafeAILab/EAGLE) (Lossless Acceleration of LLM Decoding by Feature Extrapolation) by [Yuhui Li et al. / SafeAILab](https://github.com/SafeAILab/EAGLE) for speculative decoding on ROCm:
- 3-layer feature extraction + 1-layer decoder draft model architecture
- Probabilistic tree sampling via three-tier fallback: HIP C++ → Triton → PyTorch
- `P_EAGLE` parallel variant via `mask_hidden` for concurrent draft generation
- Works with any EAGLE3-compatible draft model weights

### 📦 PrismML 1-Bit GGUF Model Support (Q1_0_G128)
Native serving of [PrismML Bonsai](https://huggingface.co/PrismML) Q1_0_G128 1-bit GGUF models — the most aggressively quantized models available:
- GPU dequantization kernels for 1-bit group-128 quantization
- 572 MB model files for 4B parameter models
- Bridges the GGUF quantization ecosystem with SGLang's runtime
- Tested with Bonsai-1.7B and Bonsai-4B

---

## AMD ROCm Exclusive Features

These features are original to this fork and designed specifically for AMD GPUs on ROCm:

### 👻 PHANTOM — Zero-Copy Ghost-Draft N-Gram Speculation

**P**inned-memory **H**ost-**A**synchronous **N**-gram **T**oken **O**racle with **M**asked Negative Learning.

A novel speculative decoding algorithm that uses background CPU threads to pre-compute n-gram draft tokens via zero-copy pinned memory DMA, eliminating GPU draft overhead entirely.

```
┌─────────────────────────────────────┐
│  GPU: Model Decode + Verify         │ ← Normal forward pass
│  (no draft computation overhead)    │
├─────────────────────────────────────┤
│  CPU Ghost Worker(s)                │ ← Background thread(s)
│  ┌─────────────────────────────┐    │
│  │ C++ NgramCorpus (GIL-free) │    │   Corpus lookup + draft build
│  │ Bloom negative filter      │    │   Skip known-bad bigrams
│  │ Pinned memory ring buffer  │    │   Zero-copy CPU→GPU DMA
│  └─────────────────────────────┘    │
├─────────────────────────────────────┤
│  PHANTOM-X Scaler (optional)        │ ← Hill-climbing throughput optimizer
│  1→N workers, 100-round trials      │   Scales up only when math proves it
│  Context diversity per worker       │   full / half / double / offset windows
└─────────────────────────────────────┘
```

**Key capabilities:**
- **Zero GPU draft cost**: All draft token generation happens on CPU threads
- **GIL-released C++ corpus**: pybind11 `py::call_guard<py::gil_scoped_release>()` enables true parallel corpus operations
- **Pinned memory DMA**: Draft tokens and masks written to pinned host memory, zero-copy visible to GPU
- **Bloom negative filter**: Rejected bigram transitions cached to avoid repeating failed draft sequences
- **PHANTOM-X adaptive scaling**: 1 worker by default; throughput-based hill-climbing adds workers only when ≥2% improvement is measured over 100-round trials, with 200-round cooldown on revert. Maximum set via `--phantom-num-ghosts` (1-8, default 1)
- **Context window diversity**: Multi-worker mode assigns different context windows per worker (full, half, double, offset) for draft diversity

**Benchmarks (Bonsai-4B Q1_0_G128, RX 6700 XT):**
- PHANTOM: **+10.5–10.9 t/s** (stable)
- NGRAM on same model: +3.3–4.4 t/s

### 🛠️ RDNA2 Crash Fixes (1–12)

12 critical fixes for AMD RDNA2 (gfx1030/gfx1031) GPU crashes during Triton kernel execution:

| Crash | Root Cause | Fix |
|-------|-----------|-----|
| 1–4 | RDNA2 validates virtual addresses on ALL wavefront lanes including exec-masked (inactive) — NVIDIA silently ignores, RDNA2 faults with HIP error 700 | `offs_safe = tl.where(mask, offs, 0)` before all pointer arithmetic |
| 5 | `extend_attention.py` OOB on partial tiles | Clamp offs_m/offs_n to `[0, seq_len)` |
| 6 | `decode_attention.py` kv_indices OOB on partial KV tiles | `offs_n_safe` clamp pattern |
| 7 | GGUF MMVQ batch>1 crashes with Q1_0_G128 dequant | Limit PRISM_Q1_TYPES MMVQ to batch=1 |
| 8 | `kv_indices` builder masked-lane OOB | Clamp in `triton_backend.py` index computation |
| 9a | `qo_indptr` mismatch on dynamic batch size | Dynamic rebuild in extend path |
| 9b | `extend_seq_lens` shape mismatch after verify | Sync after verify returns |
| 10 | Fallback decode attention index OOB | Safe indexing in fallback path |
| 11 | `seq_lens_sum` desync: verify updates `seq_lens` but not sum | `batch.seq_lens_sum = batch.seq_lens.sum().item()` |
| 12 | PHANTOM-X `active_k` tracking: K_alloc vs actual K | `_GhostBuffer.swap(bs, k=K)` stores actual K |

### 🔧 RDNA2-Optimized Kernels

Custom Wave32-optimized kernels compiled via `torch.utils.cpp_extension` at startup:
- **RMSNorm** — Wave32 reduction, bypasses aiter CK-based path
- **RoPE** — Rotary position embedding for RDNA2
- **Activation kernels** — SiLU/GELU fused operations
- Enabled via `SGLANG_RDNA2_KERNELS=1`

### 🔗 Speculative Sampling Fallback Chain

Three-tier automatic fallback for probabilistic tree speculative sampling (no flashinfer dependency):

```
┌─────────────────────────────────────────────────┐
│  HIP C++ Kernel (speculative_sampling.hip)      │ ← Fastest, gfx1030
│  Self-contained port, no flashinfer dependency  │
├─────────────────────────────────────────────────┤
│  Triton Kernel (speculative_sampling_triton.py) │ ← Fast, any GPU
│  @triton.jit, device-agnostic                   │
├─────────────────────────────────────────────────┤
│  PyTorch Fallback (speculative_sampling_pytorch)│ ← Universal
│  Pure tensor ops, works on any backend          │
└─────────────────────────────────────────────────┘
```

Detection is automatic — the best available backend is selected at import time.

### 📦 Pre-Built ROCm sgl_kernel

Ships a pre-compiled `sgl_kernel` binary for ROCm gfx1030 (RDNA2):
- `verify_tree_greedy`
- `build_tree_kernel_efficient`
- `tree_speculative_sampling_target_only` (HIP port)
- All standard sglang ops (activation, MoE, rotary, allreduce, etc.)

### 🔄 gfx1031 Compatibility

Full compatibility layer for gfx1031 (RX 6700 XT) via gfx1030 normalization:
- `HSA_OVERRIDE_GFX_VERSION=10.3.0` environment variable
- Architecture allowlist patches where needed
- All 12 RDNA2 crash fixes apply to both gfx1030 and gfx1031

---

## Speculative Decoding Algorithms

| Algorithm | Flag | Description | Needs Draft Model | Status |
|-----------|------|-------------|-------------------|--------|
| `PHANTOM` | `--speculative-algorithm PHANTOM` | Zero-copy ghost-draft n-gram (AMD exclusive) | ❌ | ✅ Verified |
| `NGRAM` | `--speculative-algorithm NGRAM` | Statistical trie-based, zero extra compute | ❌ | ✅ Verified |
| `EAGLE3` | `--speculative-algorithm EAGLE3` | 3-layer feature extraction + 1-layer decoder | ✅ | ✅ Integrated |
| `P_EAGLE` | `--speculative-algorithm P_EAGLE` | Parallel EAGLE3 via mask_hidden | ✅ | ✅ Integrated |
| `STANDALONE` | `--speculative-algorithm STANDALONE` | Independent draft model (no shared weights) | ✅ | ✅ Integrated |

<details>
<summary>Experimental algorithms (not yet verified on ROCm)</summary>

| Algorithm | Flag | Description | Needs Draft Model |
|-----------|------|-------------|-------------------|
| `MEDUSA` | `--speculative-algorithm MEDUSA` | Parallel MLP draft heads + tree verify | Medusa heads |
| `P_CASCADE` | `--speculative-algorithm P_CASCADE` | Adaptive routing: EAGLE→reduced→ngram | ✅ |
| `CHIMERA` | `--speculative-algorithm CHIMERA` | Fused P-EAGLE + Hydra + DyTC + SSD | ✅ |
| `SAGUARO` | `--ssd-enable` (wraps any) | LRU draft caching wrapper | ❌ |

These algorithms are present in the codebase but have not been validated on AMD ROCm hardware.
</details>

---

## Tested Configurations

| GPU | Model | Algorithm | KV Cache | Throughput | Status |
|-----|-------|-----------|----------|------------|--------|
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | NGRAM | fp16 | **27.8 t/s** | ✅ Stable (80/80 stress) |
| RX 6700 XT (12GB) | Bonsai-1.7B (Q1_0_G128 GGUF) | Baseline (no spec) | fp16 | **17.2 t/s** | ✅ Baseline |
| RX 6700 XT (12GB) | Bonsai-4B (Q1_0_G128 GGUF) | Baseline (no spec) | fp16 | **12.4 t/s** | ✅ Baseline |
| RX 6700 XT (12GB) | Bonsai-4B (Q1_0_G128 GGUF) | **PHANTOM** | fp16 | **10.9 t/s** | ✅ Stable |
| RX 6700 XT (12GB) | Bonsai-4B (Q1_0_G128 GGUF) | NGRAM | fp16 | **3.3–4.4 t/s** | ✅ Works |

---

## Quick Start (ROCm)

### Prerequisites
- AMD GPU with ROCm support (tested on RX 6700 XT / gfx1031 and RX 6900 XT / gfx1030)
- ROCm 6.x with PyTorch (`torch` with ROCm backend)
- Python 3.12+

### Install
```bash
git clone https://github.com/carlosfundora/sglang-1-bit-turbo.git
cd sglang-1-bit-turbo

# Install the runtime
pip install -e "python[all]"

# Install the pre-built kernel (ROCm gfx1030)
pip install -e sgl-kernel
```

### Environment Variables
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0    # Required for gfx1030/gfx1031
export PYTORCH_ROCM_ARCH=gfx1030
export SGLANG_RDNA2_KERNELS=1              # RDNA2-optimized Wave32 kernels
export SGLANG_DISABLE_FLASHINFER=1         # Required on ROCm
export SGLANG_EAGLE_SKIP_TARGET_EMBED_SHARE=1  # Required for GGUF + EAGLE3
```

### Serve a 1-Bit Model with PHANTOM Speculation
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 SGLANG_RDNA2_KERNELS=1 SGLANG_DISABLE_FLASHINFER=1 \
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-4B.gguf --load-format gguf \
  --speculative-algorithm PHANTOM \
  --speculative-num-draft-tokens 4 \
  --attention-backend triton --disable-overlap-schedule --disable-cuda-graph \
  --dtype bfloat16 --trust-remote-code --port 30000
```

### Serve with EAGLE3 + RotorQuant KV Cache
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 SGLANG_RDNA2_KERNELS=1 SGLANG_DISABLE_FLASHINFER=1 \
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-4B.gguf --load-format gguf \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /path/to/Bonsai-4B-EAGLE3/ \
  --kv-cache-dtype rq4_planar \
  --attention-backend triton --disable-cuda-graph \
  --dtype bfloat16 --trust-remote-code --port 30000
```

### Serve with NGRAM (Zero Extra Compute)
```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0 SGLANG_RDNA2_KERNELS=1 SGLANG_DISABLE_FLASHINFER=1 \
python -m sglang.launch_server \
  --model-path /path/to/Bonsai-1.7B.gguf --load-format gguf \
  --speculative-algorithm NGRAM \
  --speculative-num-draft-tokens 5 \
  --attention-backend triton --disable-cuda-graph \
  --dtype float16 --trust-remote-code --port 30000
```

---

## CLI Reference

### PHANTOM / PHANTOM-X Flags
```
--speculative-algorithm PHANTOM
--speculative-num-draft-tokens 4       # Draft tokens per round (default: 4)
--phantom-num-ghosts 1                 # Max ghost workers (1-8, default: 1)
                                       # PHANTOM-X scaler handles upscaling automatically
--phantom-num-buffers 2                # Pinned ring buffer depth per worker (default: 2)
```

### EAGLE3 Flags
```
--speculative-algorithm EAGLE3
--speculative-draft-model-path PATH    # Path to EAGLE3 draft model weights
--speculative-eagle-topk K             # Top-k candidates per draft step
--speculative-num-steps N              # Max draft steps per round
--speculative-num-draft-tokens N       # Max total draft tokens
```

### NGRAM Flags
```
--speculative-algorithm NGRAM
--speculative-num-draft-tokens N       # Draft tokens per round
--speculative-ngram-max-trie-depth 4   # Max trie depth
--speculative-ngram-match-type BFS     # BFS or PROB
```

### KV Cache Compression Flags
```
--kv-cache-dtype rq4_planar            # RotorQuant 4-bit PlanarQuant (fastest)
--kv-cache-dtype rq3_planar            # RotorQuant 3-bit PlanarQuant
--kv-cache-dtype rq4_iso               # RotorQuant 4-bit IsoQuant (best quality)
--kv-cache-dtype rq3_iso               # RotorQuant 3-bit IsoQuant
--kv-cache-dtype tq4                   # TurboQuant 4-bit
--kv-cache-dtype tq3                   # TurboQuant 3-bit
--kv-cache-dtype tq2                   # TurboQuant 2-bit (most aggressive)
```

---

## Building sgl_kernel from Source

If you need to rebuild the kernel (e.g., for a different GPU target):

```bash
cd sgl-kernel
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export AMDGPU_TARGET=gfx1030
python setup_rocm.py build_ext --inplace
```

Verify:
```bash
python -c "from sgl_kernel import common_ops; print('tree_speculative_sampling_target_only' in dir(common_ops))"
```

---

## Acknowledgments

This fork builds on the work of:
- [SGLang / LMSYS](https://github.com/sgl-project/sglang) — the upstream inference engine
- [Scrya / RotorQuant](https://github.com/scrya-com/rotorquant) — geometric-rotation KV cache compression (PlanarQuant, IsoQuant) — ICLR 2026
- [TurboQuant](https://arxiv.org/abs/2501.06903) — "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate" (ICLR 2026) — data-oblivious KV cache quantization via random rotation + Lloyd-Max scalar quantization
- [SafeAILab / EAGLE](https://github.com/SafeAILab/EAGLE) — EAGLE speculative decoding (Yuhui Li, Fangyun Wei, Chao Zhang, Hongyang Zhang)
- [FasterDecoding / Medusa](https://github.com/FasterDecoding/Medusa) — multi-head speculative decoding architecture
- [PrismML / Bonsai](https://huggingface.co/PrismML) — 1-bit GGUF model ecosystem
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) — sampling kernels adapted for the HIP port
- [vLLM](https://github.com/vllm-project/vllm) — reference for Triton-based rejection sampling patterns

## License
Apache 2.0 — same as upstream SGLang.
