# Attention Backend Unit Test Plan

## Problem

### Issue 1 — Coverage gaps
Current tests do not cover arbitrary combinations of:
- Runners (eager, cuda_graph, PCG/BCG, EAGLE and MTP runners with variants)
- Attention backends (many exist; most are untested at the unit level)
- Input cases (various bsz, input len, KV cache size)

### Issue 2 — Test expense
Most tests are e2e (full server + model load + eval). That is slow and unnecessary for
verifying attention backend correctness.

---

## Approach

### Reference implementation
Use the HuggingFace attention implementation for each attention type as the reference.

Rationale:
- There is no universal sglang reference backend — `torch_native` does not support SWA,
  Mamba, DeepSeek-V4-style sparse attention, or linear attention variants.
- HF implementations serve as an independent ground truth per attention type.
- Preparing HF inputs is straightforward: populate the paged KV pool from dense tensors
  before the forward pass, or read them back from the pool afterwards. This is a
  one-time `reconstruct_dense_kv` helper, not a per-test burden.
- Model configs (head counts, head dim, window size, etc.) are hardcoded in the test
  file. No network access to HuggingFace is required. Only the attention math logic
  from transformers (or a small hand-written reference kernel) is used.

### Test target: complete attention layer, not just the backend
Tests must exercise the full attention layer (e.g., `RadixAttention` or the
model-specific class), not just `backend.forward_decode/extend()` directly.

Reason: some attention paths call additional backend methods during the model forward
that are not invoked by calling the backend directly. For example, `forward_mha.py`
calls `get_attn_backend().init_mha_chunk_metadata(forward_batch)` for the chunked-KV
path. Bypassing these calls produces a silently wrong test.

### Mock model runner
Backends are instantiated with a `model_runner` that supplies configuration
(`page_size`, `sliding_window_size`, pool handles, dtype, etc.). Tests construct a
lightweight mock/stub with exactly the fields each backend reads in `__init__` and
`init_forward_metadata`. These config values live on the mock runner, not as parameters
of `make_forward_batch`.

### Random weights, real model configs
Attention layer weights are randomly initialized (no checkpoint download needed).
Model configs (num_heads, num_kv_heads, head_dim, window_size, etc.) are copied from
real HuggingFace model cards and hardcoded in the test file. This is sufficient because
attention correctness does not depend on the specific weight values.

---

## Full list of attention backends

### Registered backends (from `attention_registry.py`)

| Name | Class | Notes |
|---|---|---|
| `flashinfer` | `FlashInferAttnBackend` / `FlashInferMLABackend` | Primary CUDA backend; dispatches to MLA variant for MLA models |
| `triton` | `TritonAttnBackend` | Pure Triton kernels |
| `torch_native` | `TorchNativeAttnBackend` | PyTorch SDPA |
| `flex_attention` | `TorchFlexAttnBackend` | PyTorch flex attention |
| `fa3` | `FlashAttentionBackend` (v3) | SM90+ only |
| `fa4` | `FlashAttentionBackend` (v4) | SM90+ only |
| `flashmla` | `FlashMLABackend` | MLA-only, SM90+ |
| `cutlass_mla` | `CutlassMLABackend` | MLA-only |
| `trtllm_mha` | `TrtllmMHABackend` | TensorRT-LLM MHA |
| `trtllm_mla` | `TrtllmMLABackend` | TensorRT-LLM MLA |
| `tokenspeed_mla` | `TokenSpeedMLABackend` | MLA variant |
| `aiter` | `AiterAttnBackend` | AMD ROCm |
| `wave` | `WaveAttnBackend` | Wave attention |
| `ascend` | — | Ascend NPU |
| `dsa` | `DSABackend` | DeepSeek Sparse Attention |
| `dsv4` | `DeepseekV4Backend` | DeepSeek V4 CUDA (HIP variant: `deepseek_v4_backend_hip_radix`) |
| `dual_chunk_flash_attn` | `DualChunkFlashAttentionBackend` | Long-context chunked |
| `intel_amx` | `IntelAMXBackend` | Intel CPU AMX |
| `intel_xpu` | `IntelXPUBackend` | Intel XPU |

### Wrapper / hybrid backends (not in registry; composed at runtime)

| Name | Class | Notes |
|---|---|---|
| `hybrid_attn` | `HybridAttnBackend` | Configurable prefill+decode split (e.g., FA3 prefill + FlashInfer decode) |
| `tbo` | `TboAttnBackend` | Two-batch overlap backend; wraps a primary backend |
| `hybrid_linear_attn` | `MambaAttnBackendBase` | Wraps any attention backend with Mamba/linear-attention layers |

### Linear / state-space attention backends (under `linear/`)

| Name | Class | Notes |
|---|---|---|
| `kda` | `KDABackend` | KDA linear attention |
| `lightning_attn` | `LightningAttnBackend` | Lightning attention |
| `gdn` | `GDNBackend` | GDN linear attention |

---

## Full list of speculative decoding workers and runners

### Workers (draft model inference)

| Worker | Description |
|---|---|
| `EAGLEWorker` | EAGLE v1 draft worker |
| `EAGLEWorkerV2` | EAGLE v2 draft worker (tree-based, supports `topk > 1`) |
| `StandaloneWorker` / `StandaloneWorkerV2` | Self-speculative (draft = target) |
| `MultiLayerEagleWorker` / `MultiLayerEagleDraftWorker` (v2) | Multi-layer EAGLE |
| `FrozenKVMTPWorker` / `FrozenKVMTPWorkerV2` | Frozen-KV MTP |
| `DFlashWorker` | DFlash speculative worker |
| `NGRAMWorker` | N-gram draft worker |

### CUDA graph runners (for draft phase)

| Runner | Description |
|---|---|
| `EAGLEDraftCudaGraphRunner` | EAGLE v1 decode draft |
| `EAGLEDraftExtendCudaGraphRunner` | EAGLE v1/v2 extend draft |
| `MultiLayerEagleDraftExtendCudaGraphRunner` | Multi-layer EAGLE extend |
| `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner` | Multi-layer, multi-step extend |
| `FrozenKVMTPCudaGraphRunner` | Frozen-KV MTP decode |

---

## Dimensions to enumerate

### Phase 2 matrix (attention backend correctness)

| Dimension | Values |
|---|---|
| **Attention type** | Standard MHA, GQA, SWA, MLA (DeepSeek), DSA, DSV4, linear (KDA/Lightning/GDN), Mamba |
| **Backend** | See full list above; one or more per attention type |
| **Forward mode** | `DECODE`, `EXTEND`, `MIXED`, chunked-KV (`MHA_CHUNKED_KV` path) |
| **Input shape** | small bsz + short seq, large bsz + long seq, bsz=1 decode |

Representative model configs (hardcoded, no download):

| Attention type | Representative model | Key config |
|---|---|---|
| Standard MHA | GPT-2 small | `num_heads=12, head_dim=64` |
| GQA | LLaMA-3-8B | `num_heads=32, num_kv_heads=8, head_dim=128` |
| SWA | Gemma-3 / Mistral | `num_heads=8, num_kv_heads=4, head_dim=256, window_size=4096` |
| MLA | DeepSeek-V2-Lite | `qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512` |
| DSA | DeepSeek-V3 | DSA-specific head config |
| DSV4 | DeepSeek-V4 | DSV4-specific config |
| Linear (KDA) | Kimi model | KDA linear config |
| Linear (Lightning) | Bailing model | Lightning config |
| Mamba | Mamba-2 | SSM config |

### Phase 3 matrix (runner integration)

| Dimension | Values |
|---|---|
| **Runner** | eager, cuda_graph (regular / BCG), piecewise_cuda_graph (PCG) |
| **Backend** | flashinfer, triton, torch_native, flex_attention, fa3, fa4 (SM90+), flashmla, cutlass_mla, trtllm_mha, trtllm_mla, dual_chunk_flash_attn, hybrid_attn, tbo |
| **Forward mode** | DECODE, EXTEND |

Not all (runner, backend) pairs are valid — e.g., PCG is only used with certain MLA
backends; TBO wraps another backend and is tested in composition.

### Phase 4 matrix (speculative decoding attention)

| Dimension | Values |
|---|---|
| **Spec worker** | EAGLEWorker, EAGLEWorkerV2, StandaloneWorker, StandaloneWorkerV2, MultiLayerEagleWorker, MultiLayerEagleDraftWorker, FrozenKVMTPWorker, FrozenKVMTPWorkerV2, DFlashWorker, NGRAMWorker |
| **Execution mode** | eager (no CUDA graph), cuda_graph |
| **Draft runner (cuda_graph only)** | EAGLEDraftCudaGraphRunner, EAGLEDraftExtendCudaGraphRunner, MultiLayerEagleDraftExtendCudaGraphRunner, MultiLayerEagleMultiStepDraftExtendCudaGraphRunner, FrozenKVMTPCudaGraphRunner |
| **topk** | 1 (greedy / chain draft), >1 (tree draft) |
| **num_draft_steps** | 1, >1 (multi-step draft) |
| **Forward mode** | `DRAFT_EXTEND`, `TARGET_VERIFY` |
| **Backend** | flashinfer, triton, fa3, fa4, cutlass_mla, flashmla, trtllm_mha, trtllm_mla, dual_chunk_flash_attn, hybrid_attn |

`topk=1` vs `topk>1` and `num_draft_steps=1` vs `>1` are explicit axes because they
change code paths in `eagle_info.py`/`eagle_info_v2.py` (different tensor layouts,
different acceptance kernel branches).

Eager execution (no CUDA graph) is tested separately from the CUDA graph runners
because it exercises a different code path in both the worker and the attention backend.

---

## Implementation phases

### Phase 1 — Infrastructure (`test/srt/attention/utils.py`)

#### 1a. Mock model runner

```python
def make_mock_model_runner(
    page_size: int,          # backend config, NOT a ForwardBatch param
    sliding_window_size: int | None,
    max_total_num_tokens: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
    **extra_fields,          # backend-specific (e.g., kv_lora_rank for MLA)
) -> SimpleNamespace:
    """
    Lightweight mock of ModelRunner. Allocates real ReqToTokenPool and
    TokenToKVPool. page_size and sliding_window_size live here.
    """
```

#### 1b. Forward batch factory

```python
def make_forward_batch(
    mode: ForwardMode,
    bsz: int,
    seq_lens: list[int],
    model_runner: SimpleNamespace,
) -> ForwardBatch:
    """
    Build a minimal ForwardBatch with random KV data pre-loaded into the pools.
    Does NOT accept page_size or window_size — those come from model_runner.
    """
```

#### 1c. KV reconstruction

```python
def reconstruct_dense_kv(
    forward_batch: ForwardBatch,
    model_runner: SimpleNamespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read K and V back from the paged pool in (bsz, seq_len, num_kv_heads, head_dim)
    dense layout for feeding into HF reference attention.
    """
```

#### 1d. HF reference kernels (hardcoded, no network access)

```python
def hf_mha_reference(q, k, v, causal=True) -> torch.Tensor: ...
def hf_gqa_reference(q, k, v, causal=True) -> torch.Tensor: ...
def hf_swa_reference(q, k, v, window_size: int) -> torch.Tensor: ...
def hf_mla_reference(q, k_compressed, v_compressed, W_kv_up, ...) -> torch.Tensor: ...
# etc. per attention type
```

Each calls `F.scaled_dot_product_attention` with the appropriate mask, or a small
hand-written kernel. No checkpoint downloads.

#### 1e. Assertion helper

```python
def assert_close(ref, out, atol=1e-2, rtol=1e-2): ...
```

---

### Phase 2 — Backend correctness tests (`test/srt/attention/test_backend_correctness.py`)

For each valid (attention_type, backend, mode, input_shape) combination:

1. Build mock model runner with hardcoded model config.
2. Instantiate backend via `Backend.__init__(model_runner)`.
3. Instantiate attention layer (`RadixAttention` for standard types;
   model-specific class for MLA/DSA/DSV4/linear).
4. Build `ForwardBatch` via `make_forward_batch`.
5. Call `backend.init_forward_metadata(forward_batch)`.
6. Call `attention_layer.forward(q, k, v, forward_batch)` — dispatches to correct
   backend method plus any auxiliary calls (e.g., `init_mha_chunk_metadata`).
7. Reconstruct dense K, V from pool via `reconstruct_dense_kv`.
8. Run HF reference on the same Q, dense K, dense V.
9. `assert_close(hf_output, sglang_output)`.

Invalid combinations are `skipIf`-guarded, not silently dropped.

---

### Phase 3 — Runner integration tests (`test/srt/attention/test_runner_integration.py`)

Tests that runner bookkeeping does not corrupt attention outputs.

**Eager**: `init_forward_metadata` → `attention_layer.forward`. Assert vs. HF reference.

**CUDA graph / BCG**:
1. `init_cuda_graph_state` → warmup capture → `on_after_cuda_graph_warmup`.
2. Replay → forward.
3. Assert replayed output matches eager output.

**PCG**: same pattern with piecewise capture/replay API.

---

### Phase 4 — Speculative decoding (`test/srt/attention/test_spec_decoding_attention.py`)

Enumerate (spec_worker, draft_runner, topk, num_draft_steps, forward_mode, backend):

1. Construct `SpecInfo` / `EagleInfo` / `FrozenKVMTPInfo` fixture with synthetic draft
   tree matching the (topk, num_draft_steps) parameters.
2. Build `ForwardBatch` for `DRAFT_EXTEND` or `TARGET_VERIFY`.
3. `backend.init_forward_metadata(batch)` → `attention_layer.forward(...)`.
4. Compare output against HF reference with the same Q/K/V.

Explicitly test:
- `topk=1` (greedy chain) vs `topk=4` (tree draft) — different tensor layouts in
  `eagle_info_v2.py`.
- `num_draft_steps=1` vs `num_draft_steps=4` — multi-step draft extend paths.
- `get_verify_buffers_to_fill_after_draft()` shape correctness.
- `update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)` correctness.

---

## Open questions

1. **Linear / Mamba backends**: Their `forward` signature differs (they use
   `RadixLinearAttention` not `RadixAttention`). Should the Phase 1 helper accept
   a layer type parameter, or should linear attention have a separate helper?

2. **DSA / DSV4**: These use model-specific indexer metadata
   (`get_indexer_metadata`). Should Phase 2 tests use the actual model class
   (DeepSeek forward path) or stub out the indexer?

3. **TBO backend**: It wraps another backend. Should it be tested as a composition
   (primary=flashinfer wrapped in TBO) or deferred?

4. **HIP-only backends** (`aiter`, HIP DSV4): Same framework with `skipIf` guards,
   or separate AMD CI registration block?

---

## CI registration

- Phase 2: `register_cuda_ci()`, fast tier (target < 60 s total).
- Phase 3: `register_cuda_ci()`, medium tier.
- Phase 4: `register_cuda_ci()` under speculative decoding group.
- AMD variants: `register_amd_ci()` where applicable.
