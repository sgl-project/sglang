# OmniDreams Optimization Deep-Dive Report

Based on thorough exploration of `python/sglang/multimodal_gen/` -- ~30 files read, full codebase searched across 12 targeted areas.

---

## 1. Wav2Vec / Audio Feature Extraction

**Finding: No wav2vec code exists.** The search for `wav2vec`, `audio_feature`, `AudioFeature`, and `audio_encoder` returned only unrelated hits. The only audio-related code:

- `runtime/models/vaes/dac.py` -- DAC audio VAE (not wav2vec)
- `runtime/models/dits/mova_audio_dit.py` -- MOVA audio DiT
- `configs/models/vocoder/base.py` -- vocoder config

OmniDreams is video-only with Cosmos-Reason1-7B for text. Audio modality would be new territory.

---

## 2. Text Encoder Pipeline Stages (QwenEmbd / qwen_embd / text_encoder)

OmniDreams uses a **custom text encoding path** that bypasses the standard `TextEncodingStage`. No `QwenEmbd` class exists.

**Key files and their roles:**

| File | Lines | What it does |
|---|---|---|
| `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py` | 162-211 | `_encode_text()` -- inline text encoding inside the pre-processing stage |
| `runtime/pipelines/omnidreams_pipeline.py` | 250-272 | `_load_text_encoder()` -- loads Cosmos-Reason1-7B |
| `runtime/models/encoders/omnidreams_text.py` | 1-75 | `full_concat_embeddings()` -- per-token mean/std normalize, concatenate 28 layers into 100352-dim embedding |
| `configs/pipeline_configs/omnidreams.py` | 1-65 | Pipeline config with 2-step flow-match sigma schedule |

Text encoder details:
- Cosmos-Reason1-7B (Qwen2.5-VL), 7B params, 28 transformer layers x 3584 hidden
- Pinned to revision: `3210bec0495fdc7a8d3dbb8d58da5711eab4b423`
- Loaded in `torch.bfloat16`
- Declared as `ComponentUse` for CPU offload scheduling
- **Caching opportunity:** LRU cache keyed on prompt string could skip the 14 GB model for repeat prompts (especially valuable for serving)

---

## 3. Tensor Logging / Profiling Code

Several profiling systems exist, well-instrumented:

### StageProfiler (`runtime/utils/perf_logger.py`)
- Context manager records stage-level and per-step timings via `RequestMetrics`
- `PerformanceLogger.log_request_summary()` writes to `performance.log`
- Memory snapshots via `capture_memory_snapshot()`
- Env vars: `SGLANG_PERF_LOG_DIR`, `SGLANG_DIFFUSION_STAGE_LOGGING`, `SGLANG_DIFFUSION_SYNC_STAGE_PROFILING`

### SGLDiffusionProfiler (`runtime/utils/profiler.py`)
- Wraps `torch.profiler.profile` for Chrome trace export (.json.gz)
- Supports full-pipeline or denoising-only profiling
- Env var: `SGLANG_DIFFUSION_TORCH_PROFILER_DIR`

### NVTX Hooks (`runtime/utils/nvtx_pytorch_hooks.py`)
- `DiffusionNvtxHooks` registers NVTX markers around each submodule forward
- CLI flag: `--enable-layerwise-nvtx-marker`

### OmniDreams-Specific Diagnostics (`omnidreams.py` lines 84-111)
- `SGLANG_OMNIDREAMS_DIAGNOSTICS` env var for per-frame tensor stats (mean/std/min/max)
- `_log_omnidreams_stats()` called at VAE encode and AR concat points

No `tlog` or `tensor_log` naming convention found.

---

## 4. attention_mask Handling

**CRITICAL FINDING:** OmniDreams **deliberately drops the attention_mask** in text encoding.

File: `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`, lines 162-211

The `_encode_text()` method has an extensive docstring explaining why:

> CRITICAL: no `attention_mask` is passed to the encoder. The checkpoint was trained with FlashDreams' `CosmosReason1TextEncoder`, which runs the LM on the full padded sequence WITHOUT a mask. The DiT cross-attends over all `_TEXT_MAX_LENGTH` token embeddings (valid + padding), so the padding embeddings are part of the trained conditioning distribution. Passing a mask changes the padding-token hidden states drastically (abs diff up to ~99 vs the no-mask path), pushing the conditioning out of distribution and producing washed-out / blurry rollouts.

Text is padded/truncated to 512 tokens (`_TEXT_MAX_LENGTH`) with no attention mask.

**DiT attention** (in `OmniDreamsAttention.forward()`): bidirectional SDPA with **zero causal mask** (`F.scaled_dot_product_attention()` at line 373 of the DiT). Cross-chunk causality comes solely from the `BlockKVCache` window mechanism.

---

## 5. DIT_BLOCK / dit_block References in AR Stage

`OmniDreamsBlock` is the core transformer block (not DIT_BLOCK or dit_block).

File: `runtime/models/dits/omnidreams.py`, lines 383-556

Architecture facts:
- 28 blocks (`arch.num_blocks = 28`)
- Each block: `LayerNorm_self -> SelfAttn -> (CrossViewAttn, NOT IMPLEMENTED) -> LayerNorm_cross -> CrossAttn -> LayerNorm_MLP -> FFN(GPT2-style, GELU)`
- AdaLN-LoRA modulation (`use_adaln_lora=True`, `adaln_lora_dim=256`)
- Cross-view attention: raises `NotImplementedError` (lines 542-556)
- FSDP sharding: `_fsdp_shard_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]`
- torch.compile: `_compile_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]`
- Outer `forward()` explicitly NOT compiled: asserts `not torch.compiler.is_compiling()` (line 878) -- dynamic chunk count + KV cache operations break fullgraph

---

## 6. ONNX / TensorRT / Model Export

**None.** No ONNX or TensorRT code exists anywhere in `multimodal_gen/`.

Model loading is done via:
- Flat `.pt` checkpoint: `torch.load()` then `load_model_from_full_model_state_dict()` for OmniDreams
- HF diffusers format: safetensors + config.json for the Wan VAE
- HuggingFace transformers: `Qwen2_5_VLForConditionalGeneration.from_pretrained()` for text encoder

---

## 7. Flash Attention Integration

**OmniDreams does NOT use the framework's flash attention backend.** It uses raw `F.scaled_dot_product_attention()` entirely.

File: `runtime/models/dits/omnidreams.py`, lines 373-375

```python
out = F.scaled_dot_product_attention(
    q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
)
```

The framework FA backend exists at `runtime/layers/attention/backends/flash_attn.py` with FA3/FA4 version selection, custom op registration, and varlen support. Other models (Wan, FLUX, LTX, Qwen-Image) use the framework backends; OmniDreams is unique in having its own custom attention.

**This is the highest-impact optimization target.** Converting to the FA backend would benefit 28 blocks x 3 calls/chunk (denoise + clean re-forward for self-attn + cross-attn).

---

## 8. .omc/ Subdirectories

Five `.omc/` directories found, all containing only OMC agent session state (replay logs, tracking files). No optimization analysis or notes.

| Location | Contents |
|---|---|
| `python/sglang/multimodal_gen/.omc/state/` | agent replay logs, subagent tracking |
| `python/sglang/multimodal_gen/test/.omc/state/` | agent replay files |
| `python/sglang/multimodal_gen/test/unit/.omc/state/` | agent replay files |
| `python/sglang/multimodal_gen/test/server/.omc/state/` | agent replay files |
| `python/sglang/multimodal_gen/runtime/pipelines_core/stages/model_specific_stages/.omc/state/` | agent replay files |

---

## 9. torch.compile / torch.inference_mode / torch.no_grad Patterns

### @torch.no_grad() usage:

**In the stage file** (`runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`):
- Line 162: `_encode_text()` decorator
- Line 298: `_encode_reference_image()` decorator
- Line 325: `_vae_encode_normalized()` decorator
- Line 376: `_encode_hdmap()` decorator
- Line 473: `OmniDreamsBeforeDenoisingStage.forward()` decorator
- Line 634: `OmniDreamsDenoisingStage.forward()` decorator -- entire AR rollout

**In the DiT model** (`runtime/models/dits/omnidreams.py`):
- Line 764: `precompute_cross_attn_kv()` decorator
- The main `forward()` is NOT wrapped -- it runs inside the stage's no_grad context

### torch.compile:
- CLI flag: `--enable-torch-compile` (line 220 of `server_args.py`)
- OmniDreamsDiT declares `_compile_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]`
- Individual blocks ARE compiled
- Outer `forward()` explicitly blocks compilation (line 878): dynamic chunk loop + KV cache ops break fullgraph
- `torch.inference_mode()`: not used anywhere in OmniDreams code

---

## 10. Half-Precision / fp16 / bf16 dtype Handling

Default precision:
- DiT: `bf16` (`dit_precision: str = "bf16"` in pipeline config line 44)
- VAE: `fp32` (`vae_precision: str = "fp32"` in pipeline config line 45) -- for numerical stability
- Text encoder: loaded in `torch.bfloat16` (line 267 of `omnidreams_pipeline.py`)

Key dtype details:
- `PRECISION_TO_TYPE` dict from `utils.py` used throughout for dtype resolution
- KV caches created in bf16: `dtype=torch.bfloat16` (DiT line 735)
- RoPE cos/sin: computed in fp32, cast to input dtype (rope.py line 138)
- Sinusoidal timestep table: always float32, persistent=False buffer (DiT lines 140-141)
- `context_noise`: Python int (default 128), cast to `dit_dtype` tensor (stage line 783)
- VAE encode/decode: fp32, result cast to bf16 before DiT forward

---

## 11. Server Startup and CLI Argument Parsing

**CLI entry point:** `runtime/entrypoints/cli/serve.py`
- `add_multimodal_gen_serve_args()` adds server args to parser
- `execute_serve_cmd()` creates `ServerArgs.from_cli_args()` and calls `dispatch_launch()`
- Defaults: `warmup=True`, `server_warmup=True`

**Server args:** `runtime/server_args.py` (2078 lines)
- `ServerArgs` dataclass with all config
- `add_cli_args()` method at line 1032
- Key args: `--model-path`, `--backend`, `--attention-backend`, `--enable-torch-compile`, `--num-gpus`, `--warmup`, all offload flags, quantization options
- `_adjust_parameters()` orchestrates auto-tuning
- `prepare_server_args()` at line 2051

**Launch:** `runtime/launch_server.py` -- `dispatch_launch()` handles server launch

---

## 12. OmniDreamsServerApp or Similar Server Class

**No OmniDreams-specific server class exists.** Uses the generic multimodal_gen server:

- `runtime/entrypoints/http_server.py` -- generic HTTP server
- `runtime/managers/gpu_worker.py` -- GPU inference worker
- `runtime/entrypoints/diffusion_generator.py` -- Python API `DiffGenerator`

OmniDreams is integrated as a pipeline (`OmniDreamsPipeline` / `EntryClass`) registered via `registry.py`.

---

## Concrete Optimization Targets (Grounded in Source Code)

### T1: AdaLN Fusion (HIGHEST IMPACT) -- IMPLEMENTED

**File:** `runtime/models/dits/omnidreams.py`, `OmniDreamsBlock`

Pattern repeated 9 times per block forward:
```python
normed = self.layer_norm_self_attn(x) * (1 + scale_s) + shift_s
x = x + gate_s * self.self_attn(normed, ...)
```

Existing fused kernels at `runtime/layers/layernorm.py` and `runtime/layers/elementwise.py`:
- `fuse_layernorm_scale_shift_gate_select01_kernel`
- CuTe DSL `fused_norm_scale_shift`

28 blocks x 3 sub-layers x 2 calls/chunk (denoise + clean re-forward) = 168 fused operations currently split into 504 kernel launches per forward. Other models (Qwen-Image, Z-Image) already use these fused variants.

### T2: RoPE Kernel (MEDIUM IMPACT) -- IMPLEMENTED

**Files:** `runtime/models/dits/omnidreams_rope.py`, `runtime/models/dits/omnidreams.py` (DiT)

```python
def apply_rope_freqs(x: Tensor, freqs: Tensor) -> Tensor:
    cos = f.cos().to(x.dtype)
    sin = f.sin().to(x.dtype)
    a = x[..., :half]
    b = x[..., half:]
    return torch.cat([a * cos - b * sin, b * cos + a * sin], dim=-1)
```

Pure PyTorch cos/sin computation + tensor cat. The existing fast-path framework already has:
- `flashinfer.rope.apply_rope_with_cos_sin_cache_inplace` (rotary_embedding/utils.py)
- Triton `apply_rotary_embedding`

OmniDreams uses NeoX-style (non-interleaved) rotation, head_dim=128, split T:H:W=44:42:42. Per-chunk `shift_t()` call recalculates freqs.

### T3: KV-Cache Clone Elimination (MEDIUM IMPACT) -- IMPLEMENTED

**File:** `runtime/models/dits/omnidreams_kvcache.py`, `_roll_local_window_left()`

```python
self._k[dst_slice] = self._k[src_slice].clone()
self._v[dst_slice] = self._v[src_slice].clone()
```

`.clone()` creates a fresh allocation + copy for both K and V in every block at every chunk boundary. At 720p with `window_size_t=6`: 84480 tokens per block, K+V both cloned. A `torch.roll` or in-place slice assignment without `.clone()` could work since the sink region (prefix) is never overwritten.

### T4: Text Encoder Caching (HIGH IMPACT FOR SERVING) -- IMPLEMENTED

**File:** `runtime/pipelines_core/stages/model_specific_stages/omnidreams.py`, `_encode_text()`

The 7B Qwen2.5-VL produces a `[1, 512, 100352]` embedding for a given prompt string. In serving scenarios with repeated prompts, an LRU cache keyed on the prompt string could skip the entire 14 GB model forward.

### T5: Cross-Attention KV Precomputation (ALREADY DONE)

**File:** `runtime/models/dits/omnidreams.py`, `precompute_cross_attn_kv()`, lines 764-796

Text context K/V projected once per prompt (28 blocks x k_proj + v_proj + k_norm), reused across all chunks x 3 calls/chunk. Phase 6 optimization already implemented.

### T6: torch.compile on Blocks (ALREADY ENABLED)

**File:** `runtime/models/dits/omnidreams.py`, line 574

```python
_compile_conditions = [lambda n, m: isinstance(m, OmniDreamsBlock)]
```

Triggered by `--enable-torch-compile` CLI flag. Individual blocks compiled, outer AR loop stays eager.

### T7: Memory Budgets Need Measurement

**File:** `runtime/pipelines/omnidreams_pipeline.py`, lines 335-339

```python
self.memory_usages = {
    "transformer": 4.0,   # TODO(gpu): measure
    "text_encoder": 14.0, # TODO(gpu): measure
    "vae": 1.0,           # TODO(gpu): measure
}
```

Hardcoded estimates. Measuring with `torch.cuda.memory_stats()` would improve `ComponentResidencyManager` offload scheduling accuracy.

---

## Implementation Status (2026-06-14)

| # | Target | Status | File |
|---|---|---|---|
| T1 | AdaLN fusion | **IMPLEMENTED** | `omnidreams.py` (DiT) -- 3 nn.LayerNorm replaced with LayerNormScaleShift |
| T2 | RoPE kernel | **IMPLEMENTED** | `omnidreams_rope.py` + `omnidreams.py` (DiT) -- cos_sin_cache dispatch via _apply_rotary_emb |
| T3 | KV-cache clone | **IMPLEMENTED** | `omnidreams_kvcache.py` -- split-copy eliminates per-block .clone() allocation |
| T4 | Text encoder cache | **IMPLEMENTED** | `omnidreams.py` (stage) -- OrderedDict LRU on prompt string, max 32 entries |
| T5 | Cross-attn KV precompute | Already done (Phase 6) | |
| T6 | torch.compile on blocks | Already enabled | `_compile_conditions` in OmniDreamsDiT |
| T7 | Memory budgets | Not yet -- needs GPU measurement | |

## Implementation Details

### T1: AdaLN Fusion
- **Files:** `omnidreams.py` (DiT)
- **What:** Replaced `nn.LayerNorm(elementwise_affine=False) + *(1+scale)+shift` with `LayerNormScaleShift` from `layernorm.py`
- **Impact:** On CUDA, dispatches to CuTe DSL `fused_norm_scale_shift` kernel. 3 norm ops per block x 28 blocks x 2 calls/chunk = 168 fused kernels instead of 504 separate launches.
- **API:** `LayerNormScaleShift(x, shift, scale)` -- order matches existing code convention
- **Verified:** py_compile + code signature checks (11/11)

### T2: RoPE Kernel
- **Files:** `omnidreams_rope.py`, `omnidreams.py` (DiT)
- **What:** Added `to_cos_sin_cache()` method to `RotaryPositionEmbedding3D`; `OmniDreamsDiT.forward()` precomputes cos/sin cache once per forward; `OmniDreamsAttention.forward()` accepts `rope_cos_sin` kwarg; `apply_rope_freqs()` dispatches to framework `_apply_rotary_emb` (FlashInfer on CUDA, Triton fallback, pure PyTorch fallback)
- **Why:** Avoids per-block cos/sin re-computation; on CUDA, FlashInfer inplace RoPE eliminates transient tensor allocations
- **Verified:** py_compile + code signature checks (14/14); CPU fallback path numerically identical to original

### T3: KV-Cache Split-Copy
- **Files:** `omnidreams_kvcache.py`
- **What:** `_roll_local_window_left()` split-copy replaces `.clone()` with two non-overlapping `copy_()` calls
- **Why:** Eliminates per-block, per-chunk-boundary allocation. At 720p with window=6: ~84K tokens x 128-d head x 16 heads x 2 (K+V) = 344 MB per roll for all 28 blocks
- **Verified:** CPU correctness tests (3/3): steady-state roll, re-forward overwrite, sink retention

### T4: Text Encoder Cache
- **Files:** `omnidreams.py` (stage)
- **What:** `OrderedDict` LRU cache (max 32 entries) keyed on prompt string in `_encode_text()`
- **Why:** Skips the 14 GB Cosmos-Reason1-7B (Qwen2.5-VL) forward for repeated prompts. Embeddings stored on CPU (`.detach().cpu()`) to avoid consuming GPU VRAM
- **Verified:** py_compile + code signature checks (8/8)

### What Still Needs GPU Testing
```bash
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_components.py -x -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_regression.py -x -v
python -m pytest python/sglang/multimodal_gen/test/unit/test_omnidreams_scaffold.py -x -v
```

### Git Diff Stat
```
4 files changed, 123 insertions(+), 27 deletions(-)
```

## Summary: Optimization Priority

| Rank | Optimization | Files to Change | Expected Impact |
|---|---|---|---|
| 1 | **AdaLN fusion** -- norm+scale+shift into single kernel | `omnidreams.py` OmniDreamsBlock.forward, reuse `layernorm.py` | Highest -- 504 kernel launches reduced to 168 |
| 2 | **Text encoder cache** -- LRU on prompt string | `omnidreams.py` _encode_text() | Highest for serving -- skips 14 GB model |
| 3 | **RoPE kernel** -- FlashInfer or Triton instead of pure PyTorch | `omnidreams_rope.py` apply_rope_freqs | Medium -- 28 blocks x self-attn per chunk |
| 4 | **KV-cache clone** -- roll without .clone() | `omnidreams_kvcache.py` _roll_local_window_left | Medium -- per-block per-chunk copy |
| 5 | **Memory budgets** -- measure instead of estimate | `omnidreams_pipeline.py` memory_usages | Low -- improves offload decisions |
