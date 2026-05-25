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
Tests subclass the real `ModelRunner` and override `__init__` to skip server startup,
replacing it with direct field assignment. Using a real subclass (rather than
`SimpleNamespace`) means: `isinstance` checks pass, any new field added to the real
`ModelRunner` that a backend starts reading will surface immediately as `AttributeError`
in tests rather than silently returning `None`.

### Random weights, real model configs
Attention layer weights are randomly initialized (no checkpoint download needed).
Model configs (num_heads, num_kv_heads, head_dim, window_size, etc.) are copied from
real HuggingFace model cards and hardcoded in the test file. This is sufficient because
attention correctness does not depend on the specific weight values.

For model-specific attention classes that have learnable weights (MLA compression
matrices, linear attention gating, etc.), the same random weight tensors are copied
into both the sglang class and the HF reference class before comparison.

### RoPE handling
Two cases depending on the attention class under test:

- **`RadixAttention` (standard MHA / GQA / SWA)**: RoPE is applied by the surrounding
  model class *before* `RadixAttention.forward(q, k, v, forward_batch)` is called.
  Test inputs are random post-RoPE Q, K, V tensors passed directly to both sglang and
  the HF reference. No RoPE computation is needed in either path.

- **Model-specific attention classes (MLA, DSA, DSV4, linear, etc.)**: These classes
  take hidden states as input and apply QKV projection + RoPE internally before calling
  the backend. RoPE is part of the forward path and must run in both sglang and the HF
  reference. The test feeds the same random hidden state to both, shares the same
  random weight tensors (including RoPE frequency buffers), and provides the same
  `positions` tensor to both paths (`forward_batch.positions` for sglang, passed
  explicitly to the HF reference kernel).

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

Note: some backends (e.g., `DFlash`) are exclusively used during spec decode modes and
are not exercised by Phase 2. Phase 4 is therefore the *only* correctness check for
those backends — the HF comparison cannot be dropped.

---

## Implementation phases

### Phase 1 — Infrastructure (`test/srt/attention/utils.py`)

#### 1a. Mock model runner

Subclass the real `ModelRunner`, overriding `__init__` to skip server startup:

```python
class MockModelRunner(ModelRunner):
    def __init__(self, page_size, sliding_window_size, max_total_num_tokens,
                 num_heads, num_kv_heads, head_dim, dtype, device, **extra_fields):
        # Skip ModelRunner.__init__; assign fields directly.
        self.page_size = page_size
        self.sliding_window_size = sliding_window_size
        self.req_to_token_pool = ReqToTokenPool(...)
        self.token_to_kv_pool = TokenToKVPool(...)
        # ... other fields backends read
        for k, v in extra_fields.items():
            setattr(self, k, v)
```

Using a real subclass means any new field a backend starts reading surfaces immediately
as `AttributeError` in tests rather than silently returning `None`.

`page_size` and `sliding_window_size` live here — not on `ForwardBatch`.

#### 1b. Forward batch factories (per mode)

One function per mode, making the required fields explicit:

```python
def make_decode_batch(bsz, seq_lens, model_runner) -> ForwardBatch:
    """Populates: req_pool_indices, out_cache_loc, seq_lens, forward_mode=DECODE."""

def make_extend_batch(bsz, prefix_lens, extend_lens, model_runner) -> ForwardBatch:
    """Populates: req_pool_indices, out_cache_loc, extend_prefix_lens,
    extend_seq_lens, extend_num_tokens, seq_lens, forward_mode=EXTEND."""

def make_mixed_batch(decode_bsz, extend_bsz, ..., model_runner) -> ForwardBatch:
    """Combines decode and extend portions; forward_mode=MIXED."""

def make_forward_batch(mode, bsz, seq_lens, model_runner) -> ForwardBatch:
    """Thin dispatcher to the above."""
```

Chunked-KV (`MHA_CHUNKED_KV`) path: activated when `sum_prefix_length >=
chunked_prefix_cache_threshold` (default 8192). Tests monkeypatch this threshold to a
small value (e.g., 64) on the mock model runner so the path is triggered with small
synthetic sequences without expensive large-batch construction.

#### 1c. KV reconstruction

```python
def reconstruct_dense_kv(
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Read K and V back from the paged pool in (bsz, seq_len, num_kv_heads, head_dim)
    dense layout for feeding into HF reference attention.
    Called after the sglang forward pass so newly written tokens are included.
    """
```

#### 1d. HF reference kernels (hardcoded, no network access)

```python
def hf_mha_reference(q, k, v, causal=True) -> torch.Tensor: ...
def hf_gqa_reference(q, k, v, causal=True) -> torch.Tensor: ...
def hf_swa_reference(q, k, v, window_size: int) -> torch.Tensor: ...
def hf_mla_reference(q, k_compressed, v_compressed, W_kv_up, ...) -> torch.Tensor: ...
def hf_draft_extend_reference(q, k, v) -> torch.Tensor:
    """Standard causal attention — DRAFT_EXTEND is a normal prefill."""
def hf_target_verify_reference(q, k, v, spec_info) -> torch.Tensor:
    """Builds the explicit draft tree attention mask from spec_info, then calls
    F.scaled_dot_product_attention with that mask."""
```

`make_tree_attn_mask(spec_info) -> torch.BoolTensor` is a standalone helper used by
`hf_target_verify_reference`. It constructs the `(total_verify_tokens, total_kv_tokens)`
boolean mask from the draft tree structure in `spec_info`.

Each reference calls `F.scaled_dot_product_attention` with the appropriate mask. No
checkpoint downloads.

#### 1e. Assertion helper

```python
def assert_close(ref, out, atol=1e-2, rtol=1e-2): ...
```

---

### Phase 2 — Backend correctness tests (`test/srt/attention/test_backend_correctness.py`)

Each attention type uses its own model-specific attention class — never a bare backend
call. This ensures all auxiliary calls made during real model forward (e.g.,
`init_mha_chunk_metadata`, indexer metadata setup) are exercised automatically.

| Attention type | SGLang attention class used in test |
|---|---|
| Standard MHA / GQA | `RadixAttention` |
| SWA | `RadixAttention` (with `sliding_window_size` set on mock runner) |
| MLA | DeepSeek MLA attention class from `deepseek_v2.py` |
| DSA | DeepSeek DSA attention class |
| DSV4 | DeepSeek V4 attention class |
| Linear (KDA / Lightning / GDN) | `RadixLinearAttention` with the respective backend |
| Mamba | Mamba mixer class |

For each valid (attention_type, backend, mode, input_shape) combination:

1. Build `MockModelRunner` with hardcoded model config.
2. Instantiate backend via `Backend.__init__(model_runner)`.
3. Instantiate the model-specific attention class with random weights.
   For classes with learnable weights (MLA, linear), copy the same random tensors
   into the HF reference class before comparison.
4. Build `ForwardBatch` via the appropriate per-mode factory.
5. Call `backend.init_forward_metadata(forward_batch)`.
6. Call the attention class's `forward(...)` — this dispatches to the correct backend
   method and makes all auxiliary calls as the real model forward would.
7. Reconstruct dense K, V from pool via `reconstruct_dense_kv` (called after forward).
8. Run the appropriate HF reference on the same Q, dense K, dense V.
9. `assert_close(hf_output, sglang_output)`.

TBO backend: included in the backend list but deferred — placeholder
`skipIf(True, "TBO deferred")` tracks the gap.

Invalid combinations are `skipIf`-guarded, not silently dropped.

---

### Phase 3 — Runner integration tests (`test/srt/attention/test_runner_integration.py`)

Tests that runner bookkeeping does not corrupt attention outputs. Two-step verification:

**Step A — Eager correctness**: `init_forward_metadata` → `attention_layer.forward`.
Assert output matches HF reference (same as Phase 2). This establishes a known-correct
eager baseline.

**Step B — Graph consistency**: run CUDA graph capture + replay; assert replayed output
matches the eager output from Step A. Since Step A verified correctness, matching eager
is sufficient to prove the graph path is also correct.

**CUDA graph / BCG (Step B)**:
1. `init_cuda_graph_state` → warmup capture → `on_after_cuda_graph_warmup`.
2. Replay → forward.
3. Assert replayed output == eager output from Step A.

**PCG**: same two-step pattern with piecewise capture/replay API.

---

### Phase 4 — Speculative decoding (`test/srt/attention/test_spec_decoding_attention.py`)

Some backends (e.g., `DFlash`) are exclusively used during spec decode modes and are
not exercised by Phase 2. Phase 4 is therefore the *only* correctness check for those
backends — HF comparison is mandatory, not optional.

HF reference strategy per forward mode:
- **`DRAFT_EXTEND`**: standard causal attention. Use `hf_draft_extend_reference` directly.
- **`TARGET_VERIFY`**: tree-masked attention. Use `hf_target_verify_reference`, which
  calls `make_tree_attn_mask(spec_info)` to build the explicit tree mask from the
  synthetic `SpecInfo`, then passes it to `F.scaled_dot_product_attention`.

For each (spec_worker, execution_mode, draft_runner, topk, num_draft_steps, forward_mode, backend):

1. Construct `SpecInfo` / `EagleInfo` / `FrozenKVMTPInfo` with synthetic draft tree
   matching (topk, num_draft_steps).
2. Build `ForwardBatch` for the target forward mode.
3. `backend.init_forward_metadata(batch)` → `attention_layer.forward(...)`.
4. Compare output against HF reference (mode-appropriate reference from above).

**Secondary consistency check** (in addition to HF comparison):
For each (spec_worker, backend) pair, run both eager and cuda_graph execution and assert
their outputs match. This validates runner bookkeeping independently of math correctness.

Explicitly test:
- `topk=1` (greedy chain) vs `topk=4` (tree draft).
- `num_draft_steps=1` vs `num_draft_steps=4`.
- `get_verify_buffers_to_fill_after_draft()` shape correctness.
- `update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)` correctness.

---

## Resolved decisions

1. **Linear / Mamba backends**: Use `RadixLinearAttention` (or the Mamba mixer class)
   as the test target, not a bare backend call. The model-specific attention class
   is always the entry point.

2. **DSA / DSV4**: Use the model-specific DeepSeek attention class. The indexer
   metadata is set up naturally by the class's own `forward` method.

3. **TBO backend**: Included in the backend table; implementation deferred. A
   `skipIf(True, "TBO deferred")` placeholder tracks the gap.

4. **HIP-only backends** (`aiter`, HIP DSV4, `wave`, `ascend`): Same test framework
   with `skipIf` guards for CUDA-only runners; registered with `register_amd_ci()`
   (or the appropriate non-CUDA CI registration) alongside the main test class.

5. **RoPE**: Out of scope for `RadixAttention` (inputs are post-RoPE Q/K/V). In scope
   for model-specific attention classes (MLA, DSA, DSV4, etc.) — RoPE runs inside
   their forward; both sglang and HF reference receive the same hidden state, shared
   weights, and same `positions` tensor.

6. **Chunked-KV threshold**: Monkeypatched to a small value (e.g., 64) on the mock
   model runner so the `MHA_CHUNKED_KV` path is triggered with small synthetic
   sequences.

7. **Mock model runner**: Implemented as a `ModelRunner` subclass (not `SimpleNamespace`)
   so `isinstance` checks pass and missing fields surface as `AttributeError`.

8. **Phase 3 verification**: Two-step — Step A compares eager against HF reference,
   Step B compares graph replay against eager. Only Step A establishes correctness;
   Step B establishes consistency.

9. **Phase 4 HF reference**: `DRAFT_EXTEND` uses standard causal HF reference.
   `TARGET_VERIFY` uses `make_tree_attn_mask(spec_info)` to build the explicit tree
   mask, then `F.scaled_dot_product_attention`. HF comparison is mandatory in Phase 4
   because some backends are spec-decode-exclusive and not covered by Phase 2.

---

## CI registration

- Phase 2: `register_cuda_ci()`, fast tier (target < 60 s total).
- Phase 3: `register_cuda_ci()`, medium tier.
- Phase 4: `register_cuda_ci()` under speculative decoding group.
- AMD variants: `register_amd_ci()` where applicable.
