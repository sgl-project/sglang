# Attention Backend Unit Test Plan

## Current progress

Last updated: 2026-05-25

Implemented:
- Shared dense MHA/GQA correctness helpers exist in
  `test/manual/attention/unittest/common/dense_attention.py`.
- Dense attention backend correctness files exist under
  `test/manual/attention/unittest/dense/` for `torch_native`, `triton`, and
  `flashinfer`. Dense/SWA expected paths now use separate HF-style reference
  modules with copied random projection weights instead of calling projection
  helpers on the SGLang actual module.
- SWA attention backend correctness files exist under
  `test/manual/attention/unittest/swa/` for `triton` and `flashinfer`.
- MLA attention backend correctness exists under
  `test/manual/attention/unittest/mla/` for `triton`; its actual path now uses a
  small DeepSeek-shaped absorb-MLA module that explicitly writes latent KV through
  `get_token_to_kv_pool()` before calling `attn_mqa`, while its expected path is a
  separate HF-style PyTorch reference module with copied random weights and no
  `RadixAttention` or backend calls.
- GDN hybrid-linear attention backend correctness exists under
  `test/manual/attention/unittest/gdn/` for full-attention backend `triton` with
  linear-attention kernel backend `triton`; its expected path is now a separate
  reference module plus pure PyTorch gated-delta recurrence, not the Triton/FLA
  GDN kernels.
- Phase 3 dense runner integration is implemented for representative attention
  backends: eager mode for `torch_native`, and CUDA-graph metadata capture/replay
  decode mode for `triton` and `flashinfer`. The CUDA-graph tests now capture
  against a fixed padded decode batch and replay distinct request metadata/input
  tensors, so capture data is not identical to forward data.
- Dense input-config cases now cover page size 1, zero-prefix exact page,
  zero-prefix input lengths below/equal/above a page, prefix-length exact page,
  total-length exact page, total-length crossing a page boundary, ragged
  below/equal/above page-boundary batches, representative page-size-32 crossing,
  decode page-boundary batches, and bsz=1 decode with nonzero prefix.
- Dense attention-config cases cover GQA and MQA separately from input-layout
  coverage.
- The `flashinfer` file uses `head_dim=64` because FlashInfer SM90 prefill kernels
  require value head dim in `{64, 128, 256}`.
- SWA input-config cases cover no-prefix lengths below/equal/above the window for
  `triton` and `flashinfer`; `triton` also covers prefix lengths below/equal/above
  the window. Prefix+SWA for FlashInfer needs a more faithful metadata fixture
  before enabling.
- The harness uses random shared weights, real `ForwardBatch` metadata, and real
  KV/request pools. Reference implementations must be independent HF-style
  PyTorch modules or functions; they may receive copied weights from the actual
  module, but must not call SGLang attention modules, SGLang backend wrappers, or
  SGLang kernel helpers.
- RoPE is intentionally omitted from the current unit-level runner x attention
  tests. These tests feed post-RoPE-equivalent Q/K tensors because rotary math is
  orthogonal to runner/backend metadata compatibility.

In progress:
- No active representative-reference split remains for dense/SWA/MLA/GDN.
- Priority is now to complete Phase 2, Phase 3, and Phase 4 comprehensively for the
  representative backend set before expanding Phase 2 to additional attention
  backends.

Next implementation steps:
- Finish Phase 2 module-level backend correctness for the representative backend
  set: `torch_native`, `triton`, and `flashinfer`, plus method-specific
  representative paths such as Triton GDN and Triton MLA.
- Finish Phase 3 runner/graph integration for the same representative backend set,
  including eager, CUDA graph, BCG, PCG where supported, and realistic capture vs.
  replay metadata.
- Finish Phase 4 speculative metadata coverage for representative valid backends
  before adding more Phase 2 backend files.
- After representative Phase 2/3/4 coverage is stable, expand Phase 2 to additional
  attention backends such as `flashmla`, `cutlass_mla`, `trtllm_mha`,
  `trtllm_mla`, `fa3`, `fa4`, `dual_chunk_flash_attn`, and `flex_attention`.
- Keep `torch_native` SWA out of the matrix until the backend honors
  `RadixAttention.sliding_window_size`.

Latest verification:
- `python -m py_compile test/manual/attention/unittest/common/mla_attention.py test/manual/attention/unittest/mla/test_triton.py`
- `python test/manual/attention/unittest/mla/test_triton.py -v`
- `python -m py_compile test/manual/attention/unittest/common/dense_attention.py test/manual/attention/unittest/dense/test_torch_native.py test/manual/attention/unittest/dense/test_triton.py test/manual/attention/unittest/dense/test_flashinfer.py test/manual/attention/unittest/swa/test_triton.py test/manual/attention/unittest/swa/test_flashinfer.py`
- `python -m unittest discover -s test/manual/attention/unittest/dense -p 'test_*.py' -v`
- `python -m unittest discover -s test/manual/attention/unittest/swa -p 'test_*.py' -v`
- `python -m py_compile test/manual/attention/unittest/common/gdn_attention.py test/manual/attention/unittest/gdn/test_triton.py`
- `python test/manual/attention/unittest/gdn/test_triton.py -v`
- `python -m unittest discover -s test/manual/attention/unittest -p 'test_*.py' -v`

---

## Test File Layout

Tests are organized by attention method first and attention backend second:

```text
test/manual/attention/unittest/
  common/
    dense_attention.py
  dense/
    test_torch_native.py
    test_triton.py
    test_flashinfer.py
  swa/
    test_triton.py
    test_flashinfer.py
  mla/
    test_triton.py
    test_flashinfer.py
    test_flashmla.py
    test_trtllm_mla.py
  gdn/
    test_triton.py
```

Each `test_<attn_backend>.py` file verifies one attention method against one
attention backend across supported runner modes, forward modes, and input configs.
Speculative decoding is not an attention-method folder; it is represented as
`ForwardMode`, runner mode, and synthetic spec metadata cases inside the affected
attention-method folder.

## Problem

### Issue 1 — Coverage gaps

Current tests do not cover representative combinations of:
- Runners: eager, CUDA graph, BCG/PCG, EAGLE and MTP runners with variants.
- Attention backends: many exist, but most are not covered by fast unit-level
  correctness tests.
- Input cases: batch size, prefix length, extend length, page size, ragged
  sequence lengths, sliding-window boundaries, and speculative tree layouts.

### Issue 2 — Test expense

Most coverage today is e2e: full server launch, model load, and eval. That is too
slow and too coarse for validating attention backend math and metadata.

---

## Approach

### Primary test target: attention-module boundary

The primary correctness tests enter through the smallest real attention module that
represents the model family under test. Do not call backend methods directly, and do
not force every case through `RadixAttention`.

Reason:
- For standard MHA/GQA/SWA, the natural boundary can be a small projected
  attention module before calling `RadixAttention`.
- For MLA, DSA, DSV4, linear attention, Mamba, and speculative verify paths, the
  real behavior includes projections, compression, sparse index metadata, state
  updates, tree masks, or other side effects before backend dispatch.
- Backend-only tests miss auxiliary calls such as
  `get_attn_backend().init_mha_chunk_metadata(forward_batch)` in the DeepSeek
  chunked-KV path.

Keep a tiny `RadixAttention` smoke-test suite for the leaf backend contract
(`q/k/v + ForwardBatch -> output`), but keep it outside the main correctness matrix.

### Unified module target adapter

Every attention family implements the same adapter contract:

```python
class AttentionModuleTarget:
    def build_runner(self, case) -> MockModelRunner: ...
    def build_backend(self, runner, case) -> AttentionBackend: ...
    def build_module(self, runner, case) -> torch.nn.Module: ...
    def init_shared_random_weights(self, module, reference, seed) -> None: ...
    def make_inputs(self, batch, case) -> dict[str, torch.Tensor]: ...
    def run_sglang(self, module, inputs, batch) -> torch.Tensor: ...
    def run_reference(self, reference, inputs, batch, dense_kv) -> torch.Tensor: ...
```

This gives one test harness, one capability system, and one input-preparation
pipeline while preserving model-specific behavior.

Recommended targets:

| Attention family | Primary test target |
|---|---|
| Standard MHA | Small GPT/LLaMA-style attention module |
| GQA | Small LLaMA/Qwen-style attention module |
| SWA | Small Mistral/Gemma-style sliding-window attention module |
| MLA | Small DeepSeek-style MLA contract module; direct `DeepseekV2AttentionMLA` tests are optional dispatch/performance-path coverage |
| DSA | DeepSeek sparse attention module |
| DSV4 | DeepSeek V4 attention module |
| Linear KDA/Lightning/GDN | Model-specific linear attention module, falling back to `RadixLinearAttention` only if no smaller model module is exposed |
| Mamba | Mamba/Mamba2 mixer module |
| Spec verify/draft | Module target plus synthetic `SpecInput` metadata |

### Reference implementations

Use an independent reference at the same module boundary whenever practical. No
checkpoint download is required: configs are hardcoded, and weights are random but
copied from the SGLang module into the reference.

Rule: correctness tests must not compare one SGLang/backend implementation against
another SGLang/backend implementation. The expected path may share tensors by
explicit copy, but it must not call `RadixAttention`, `RadixLinearAttention`,
attention backend wrappers, Triton/FlashInfer/FLA kernels, or SGLang helper methods
that encode backend-specific attention behavior.

Reference strategy by family:
- MHA/GQA/SWA: explicit PyTorch reference using dense Q/K/V after the same random
  projections and RoPE as the SGLang module. Use `F.scaled_dot_product_attention`
  with causal or sliding-window masks.
- MLA: explicit DeepSeek MLA math or a minimal HF-compatible attention module with
  copied random projection/compression weights. The actual path should exercise the
  MLA backend-facing contract: `q_nope -> w_kc`, latent KV cache writes through
  `get_token_to_kv_pool()`, `attn_mqa`, and `w_vc`. It does not need to reproduce
  every `DeepseekV2AttentionMLA` forward-method dispatch choice because those
  choices are performance paths that should be mathematically equivalent.
- DSA/DSV4: model-family reference that builds the equivalent sparse or compressed
  attention mask/index result, then compares the final module output.
- Linear KDA/Lightning/GDN and Mamba: compact PyTorch implementations whenever
  feasible. Kernel-to-kernel comparisons are only smoke tests and must be labeled
  as such.
- Speculative verify: explicit causal/tree masks built from synthetic `SpecInput`
  objects, then compared at the same module boundary.

### Real configs, random weights

Model configs (head counts, KV heads, head dim, window size, compression rank, etc.)
are copied from representative HuggingFace configs and hardcoded in the test file.
Attention weights are randomly initialized. For modules with learnable projections,
compression matrices, gates, or RoPE buffers, copy the same random tensors into the
reference before comparison.

### RoPE handling

RoPE is deliberately out of scope for these runner x attention unit tests. Use
post-RoPE-equivalent Q/K tensors or set the model-specific RoPE dimension to zero
when the backend supports that shape. RoPE-specific coverage should live in focused
model/rotary tests, not in every runner/backend compatibility case.

### Forward context

All module-level tests that dispatch through `get_attn_backend()` must publish the
active backend:

```python
with forward_context(ForwardContext(attn_backend=backend)):
    backend.init_forward_metadata(forward_batch)
    output = target.run_sglang(module, inputs, forward_batch)
```

### Capability-first enumeration

Before enumerating parametrized tests, define a capability table/helper:

```python
def supports(case) -> tuple[bool, str]:
    """Return (supported, skip_reason)."""
```

The helper gates invalid combinations by attention family, backend, forward mode,
hardware, graph mode, dtype, page size, and speculative mode. Invalid combinations
are skipped with explicit reasons; they are not silently omitted and not discovered
by crashing deep inside backend initialization.

---

## Attention Backends

### Registered backends from `attention_registry.py`

| Name | Class | Notes |
|---|---|---|
| `flashinfer` | `FlashInferAttnBackend` / `FlashInferMLAAttnBackend` | Primary CUDA backend; dispatches to MLA variant for MLA models |
| `triton` | `TritonAttnBackend` | Pure Triton kernels |
| `torch_native` | `TorchNativeAttnBackend` | PyTorch SDPA |
| `flex_attention` | `TorchFlexAttnBackend` | PyTorch flex attention |
| `fa3` | `FlashAttentionBackend` v3 | Hardware-gated |
| `fa4` | `FlashAttentionBackend` v4 | Hardware-gated |
| `flashmla` | `FlashMLABackend` | MLA-only |
| `cutlass_mla` | `CutlassMLABackend` | MLA-only |
| `trtllm_mha` | `TRTLLMHAAttnBackend` | TensorRT-LLM MHA |
| `trtllm_mla` | `TRTLLMMLABackend` | TensorRT-LLM MLA |
| `tokenspeed_mla` | `TokenspeedMLABackend` | MLA variant |
| `aiter` | `AiterAttnBackend` | AMD ROCm |
| `wave` | `WaveAttnBackend` | Wave attention |
| `ascend` | `AscendAttnBackend` | Ascend NPU |
| `dsa` | `DeepseekSparseAttnBackend` | DeepSeek sparse attention |
| `nsa` | Alias for `dsa` | Deprecated compatibility alias |
| `dsv4` | `DeepseekV4AttnBackend` / HIP radix variant | DeepSeek V4 compressed attention |
| `dual_chunk_flash_attn` | `DualChunkFlashAttentionBackend` | Long-context chunked attention |
| `intel_amx` | `IntelAMXAttnBackend` | Intel CPU AMX |
| `intel_xpu` | `XPUAttentionBackend` | Intel XPU |

### Wrapper / hybrid backends

| Name | Class | Notes |
|---|---|---|
| `hybrid_attn` | `HybridAttnBackend` | Composed when prefill and decode backends differ |
| `tbo` | `TboAttnBackend` | Two-batch overlap wrapper around child backends |
| `hybrid_linear_attn` | `HybridLinearAttnBackend` | Wraps full attention with Mamba/linear-attention layers |

### Linear / state-space backends

These are selected by model config through `attn_backend_wrapper`, not by direct
`--attention-backend` names in the same way as full attention backends.

| Family | Class | Notes |
|---|---|---|
| KDA | `KDAAttnBackend` | Kimi-style linear attention |
| Lightning | `LightningAttentionBackend` | Lightning attention |
| GDN | `GDNAttnBackend` | GDN linear attention |
| Mamba2 | `Mamba2AttnBackend` | Mamba/state-space path |

---

## Speculative Decoding Workers And Runners

### Workers

| Worker | Description |
|---|---|
| `EAGLEWorker` | EAGLE v1 draft worker |
| `EAGLEWorkerV2` | EAGLE v2 draft worker, tree/spec-v2 path |
| `StandaloneWorker` / `StandaloneWorkerV2` | Self-speculative, draft equals target |
| `MultiLayerEagleWorker` / `MultiLayerEagleWorkerV2` | Multi-layer EAGLE |
| `MultiLayerEagleDraftWorker` | Draft worker used by multi-layer EAGLE v2 |
| `FrozenKVMTPWorker` / `FrozenKVMTPWorkerV2` | Frozen-KV MTP |
| `DFlashWorker` | DFlash speculative worker |
| `NGRAMWorker` | N-gram draft worker |

### CUDA graph runners

| Runner | Description |
|---|---|
| `EAGLEDraftCudaGraphRunner` | EAGLE decode draft |
| `EAGLEDraftExtendCudaGraphRunner` | EAGLE draft-extend, including v2 mode |
| `MultiLayerEagleDraftExtendCudaGraphRunner` | Multi-layer EAGLE extend |
| `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner` | Multi-layer, multi-step extend |
| `FrozenKVMTPCudaGraphRunner` | Frozen-KV MTP decode |

---

## Dimensions To Enumerate

### Capability matrix

Represent each test case as:

```python
@dataclass(frozen=True)
class AttentionCase:
    attention_method: str
    config_variant: str
    attention_backend: str
    forward_mode: ForwardMode
    runner_mode: str  # eager, cuda_graph, bcg, pcg, worker-specific graph path
    input_shape: str
    page_size: int
    dtype: torch.dtype
    hardware: str
```

Minimum capability dimensions:
- Backend supports attention family: MHA/GQA/SWA/MLA/DSA/DSV4/linear/Mamba.
- Backend supports attention config variant: MHA head layout, GQA head grouping,
  MQA, SWA, MLA ranks, sparse-index config, linear-attention config, etc.
- Backend supports mode: `DECODE`, `EXTEND`, `MIXED`, `TARGET_VERIFY`,
  `DRAFT_EXTEND`, `DRAFT_EXTEND_V2`.
- Backend supports hardware and dtype.
- Backend supports graph path: eager, CUDA graph, BCG, PCG.
- Model family supports page size and KV pool type.
- Spec worker supports `topk`, `num_draft_steps`, draft runner, and forward mode.

### Phase 2 matrix: module-level backend correctness

Execute the matrix in two passes:
1. **Representative-first pass**: finish comprehensive Phase 2 coverage for
   `torch_native`, `triton`, and `flashinfer`, plus method-specific representative
   paths already in scope such as Triton MLA and Triton GDN.
2. **Backend-expansion pass**: after Phase 2/3/4 are stable for the representative
   set, add more backend files such as `flashmla`, `cutlass_mla`, `trtllm_mha`,
   `trtllm_mla`, `fa3`, `fa4`, `dual_chunk_flash_attn`, and `flex_attention`.

| Dimension | Values |
|---|---|
| **Attention family** | Standard MHA, GQA, SWA, MLA, DSA, DSV4, linear KDA/Lightning/GDN, Mamba |
| **Attention config** | MHA, GQA with `num_heads / num_kv_heads > 1`, MQA with `num_kv_heads=1`, finite-window SWA, MLA rank variants, sparse-index variants |
| **Backend** | Representative-first subset, then capability-gated expansion subset |
| **Forward mode** | `DECODE`, `EXTEND`, `MIXED` |
| **Input shape** | Small ragged batch, exact-page batch, page-boundary batch, long prefix batch, bsz=1 decode |
| **Page size** | 1, representative paged value such as 32 or 64 |

Representative configs, hardcoded with no network access:

| Attention family | Representative model | Key config |
|---|---|---|
| Standard MHA | GPT-2/LLaMA-style tiny config | `num_heads=12, head_dim=64` |
| GQA | LLaMA-3/Qwen-style tiny config | `num_heads=32, num_kv_heads=8, head_dim=128` |
| SWA | Mistral/Gemma-style tiny config | `num_heads=8, num_kv_heads=4, head_dim=256, window_size < seq_len` |
| MLA | DeepSeek-V2-Lite | `qk_nope_head_dim=128, qk_rope_head_dim=64, kv_lora_rank=512` |
| DSA | DeepSeek-V3 sparse config | DSA-specific head and index config |
| DSV4 | DeepSeek-V4 | DSV4 compressed attention config |
| Linear KDA | Kimi-style config | KDA linear config |
| Linear Lightning | Bailing-style config | Lightning config |
| Linear GDN | Hybrid GDN config | GDN config |
| Mamba | Mamba-2 | SSM config |

DeepSeek `MHA_CHUNKED_KV`, absorbed MLA, one-shot, ragged, and paged variants are
model forward-method choices, not `ForwardMode` values. They are intended to be
mathematically equivalent performance paths. Runner x attention compatibility tests do
not need to strictly reproduce every DeepSeek dispatch choice; add focused
dispatch-path tests only when validating those production dispatch decisions.

### Phase 3 matrix: runner and graph integration

| Dimension | Values |
|---|---|
| **Runner mode** | eager baseline, CUDA graph, BCG, PCG |
| **Backend** | Representative-first subset: `torch_native`, `triton`, `flashinfer`; add wrappers such as `hybrid_attn` and TBO after base runner coverage is stable |
| **Forward mode** | `DECODE`, `EXTEND`; selected speculative graph modes in Phase 4 |
| **Attention family** | Small representative subset, not the full Phase 2 matrix |

Two-step verification:
1. Eager module output matches the family reference.
2. Graph replay output matches the eager module output.

Graph consistency does not replace the reference comparison; it validates runner and
metadata bookkeeping after correctness has been established by the eager path.

### Phase 4 matrix: speculative decoding attention

Split Phase 4 into two layers.

#### Layer A: synthetic spec metadata attention tests

These tests construct synthetic `SpecInput` objects and validate attention metadata and
module output directly.

| Dimension | Values |
|---|---|
| **Spec info** | `EagleVerifyInput`, EAGLE v2 verify input, `FrozenKVMTPInfo`, `DFlashVerifyInput`, `NGRAM` verify input |
| **Forward mode** | `TARGET_VERIFY`, `DRAFT_EXTEND`, `DRAFT_EXTEND_V2` |
| **topk** | 1 and 4 |
| **num_draft_steps** | 1 and 4 |
| **Backend** | Representative valid backends first, usually `triton` and `flashinfer`; expand only after synthetic spec metadata and graph replay are stable |
| **Execution mode** | eager and CUDA graph where supported |

Reference strategy:
- `DRAFT_EXTEND`: causal prefill reference.
- `DRAFT_EXTEND_V2`: fixed-shape draft-extend reference that accounts for all
  speculative tokens, not only accepted tokens.
- `TARGET_VERIFY`: explicit tree mask built from `spec_info`, then reference
  attention against dense KV.

Also test:
- `get_verify_buffers_to_fill_after_draft()` shape and dtype.
- `update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)` contents.
- Tree-mask equivalence for `topk=1` chain draft and `topk=4` tree draft.

#### Layer B: worker and draft-runner integration tests

These tests use a small subset of workers/runners to prove they produce compatible
`ForwardBatch` and `SpecInput` metadata. They are not a full Cartesian product.

| Dimension | Values |
|---|---|
| **Spec worker** | `EAGLEWorker`, `EAGLEWorkerV2`, `StandaloneWorker`, `StandaloneWorkerV2`, `MultiLayerEagleWorker`, `MultiLayerEagleWorkerV2`, `FrozenKVMTPWorker`, `FrozenKVMTPWorkerV2`, `DFlashWorker`, `NGRAMWorker` |
| **Draft runner** | `EAGLEDraftCudaGraphRunner`, `EAGLEDraftExtendCudaGraphRunner`, `MultiLayerEagleDraftExtendCudaGraphRunner`, `MultiLayerEagleMultiStepDraftExtendCudaGraphRunner`, `FrozenKVMTPCudaGraphRunner` |
| **Execution mode** | eager worker path, CUDA graph worker path where supported |
| **Backend** | One or two representative valid backends per worker family |

DFlash is a speculative worker/path, not an attention backend. Phase 4 covers
speculative-only attention modes and metadata that Phase 2 cannot exercise.

---

## Implementation Phases

### Phase 1 — Infrastructure (`test/manual/attention/unittest/common/`)

#### 1a. Module target adapters

Create one adapter per attention family:

```python
class MHAAttentionTarget(AttentionModuleTarget): ...
class GQAAttentionTarget(AttentionModuleTarget): ...
class SWAAttentionTarget(AttentionModuleTarget): ...
class DeepSeekMLATarget(AttentionModuleTarget): ...
class DeepSeekDSATarget(AttentionModuleTarget): ...
class DeepSeekV4Target(AttentionModuleTarget): ...
class LinearAttentionTarget(AttentionModuleTarget): ...
class MambaAttentionTarget(AttentionModuleTarget): ...
class SpecVerifyTarget(AttentionModuleTarget): ...
```

Each adapter owns module construction, input preparation, reference execution, and
shape-specific assertions. The outer test harness should not need attention-family
branches except capability filtering.

#### 1b. Mock model runner

Subclass the real `ModelRunner`, overriding `__init__` to skip server startup:

```python
class MockModelRunner(ModelRunner):
    def __init__(
        self,
        *,
        model_config,
        server_args,
        page_size,
        max_batch_size,
        max_context_len,
        token_to_kv_pool,
        dtype,
        device,
        **extra_fields,
    ):
        # Skip ModelRunner.__init__; assign fields directly.
        self.model_config = model_config
        self.server_args = server_args
        self.page_size = page_size
        self.device = device
        self.dtype = dtype
        self.req_to_token_pool = ReqToTokenPool(...)
        self.token_to_kv_pool = token_to_kv_pool
        self.token_to_kv_pool_allocator = ...
        self.sliding_window_size = getattr(model_config, "sliding_window_size", None)
        self.use_mla_backend = model_config.attention_arch == AttentionArch.MLA
        for k, v in extra_fields.items():
            setattr(self, k, v)
```

Use real `ReqToTokenPool` and real KV pools (`MHATokenToKVPool`, `MLATokenToKVPool`,
DSA/DSV4-specific pools) instead of ad hoc tensors. A real subclass means
`isinstance` checks pass and missing fields surface as `AttributeError`.

Unit fixtures run without distributed initialization. For single-rank unit tests,
patch attention tensor-parallel helpers such as `get_attention_tp_size()` to return
`1`, or initialize an equivalent local parallel-state fixture before constructing
backends that size buffers from attention TP metadata.

#### 1c. Forward context helper

```python
@contextmanager
def attention_test_context(backend):
    with forward_context(ForwardContext(attn_backend=backend)):
        yield
```

All module-level tests use this helper before calling `backend.init_forward_metadata`
or module `forward`.

#### 1d. Forward batch factories

One function per mode, making required fields explicit:

```python
def make_decode_batch(bsz, seq_lens, model_runner) -> ForwardBatch:
    """req_pool_indices, out_cache_loc, seq_lens, positions, forward_mode=DECODE."""

def make_extend_batch(bsz, prefix_lens, extend_lens, model_runner) -> ForwardBatch:
    """extend_prefix_lens, extend_seq_lens, extend_num_tokens, positions."""

def make_mixed_batch(decode_bsz, extend_bsz, ..., model_runner) -> ForwardBatch:
    """Combined decode and extend portions; forward_mode=MIXED."""

def make_target_verify_batch(spec_info, model_runner) -> ForwardBatch: ...
def make_draft_extend_batch(spec_info, model_runner, *, v2: bool = False) -> ForwardBatch: ...
def make_forward_batch(case, model_runner) -> ForwardBatch: ...
```

Factories must populate `req_to_token_pool.req_to_token` consistently with
`out_cache_loc`, prefix lengths, page size, and sequence lengths. Add assertions that
the dense reconstruction sees the expected token order.

#### 1e. KV setup and reconstruction

```python
def populate_prefix_kv(
    module,
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
) -> None: ...

def reconstruct_dense_kv(
    module,
    forward_batch: ForwardBatch,
    model_runner: MockModelRunner,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return dense K/V in batch-major sequence order after SGLang forward."""
```

For MLA/DSA/DSV4 pools, reconstruction may return family-specific compressed or
expanded tensors as required by the reference.

#### 1f. Reference helpers

```python
def sdpa_mha_reference(...): ...
def sdpa_gqa_reference(...): ...
def sdpa_swa_reference(...): ...
def deepseek_mla_reference(...): ...
def dsa_reference(...): ...
def dsv4_reference(...): ...
def linear_attention_reference(...): ...
def mamba_reference(...): ...
def spec_target_verify_reference(...): ...
```

`make_tree_attn_mask(spec_info)` remains a standalone helper used by speculative
references.

#### 1g. Assertion helper

```python
def assert_close(case, ref, out):
    """Use dtype/family/backend-specific tolerances and print first mismatch."""
```

---

### Phase 2 — Backend correctness tests (`test/manual/attention/unittest/<attention_method>/test_<attention_backend>.py`)

For each supported `AttentionCase`:

1. Select the `AttentionModuleTarget`.
2. Build `MockModelRunner` with the hardcoded representative config.
3. Instantiate the backend through the same registry path used by `ModelRunner`.
4. Build the module and reference module/kernel.
5. Initialize shared random weights and deterministic input tensors.
6. Build `ForwardBatch` through the mode-specific factory.
7. Populate prefix KV for decode or prefix-extend cases.
8. Enter `attention_test_context(backend)`.
9. Call `backend.init_forward_metadata(forward_batch)`.
10. Call `target.run_sglang(module, inputs, forward_batch)`.
11. Reconstruct dense or family-specific KV after the forward pass.
12. Run `target.run_reference(...)`.
13. Assert output closeness with case-specific tolerances.

Required attention config cases:
- MHA where `num_heads == num_kv_heads`.
- GQA where `num_heads / num_kv_heads > 1`.
- MQA where `num_kv_heads == 1`, if supported by the target/backend.
- SWA with a finite window size.
- MLA, DSA, DSV4, linear, and Mamba family-specific configs.

Required input cases:
- Page size 1.
- Paged KV with representative page sizes such as 32 or 64.
- Sequence length exactly equal to one page.
- Sequence length one token below and one token above a page boundary.
- Prefix length exactly equal to one page.
- Prefix plus extend length exactly equal to one page.
- Prefix plus extend length crossing a page boundary.
- Ragged batch with requests below, exactly at, and above a page boundary.
- Decode with nonzero prefix.
- Extend with zero prefix and with nonzero prefix.
- SWA with `seq_len < window_size`, `seq_len == window_size`, and
  `seq_len > window_size`.

Invalid combinations are `skipIf`-guarded through the capability helper.

Optional dispatch-path cases:
- DeepSeek chunked-KV and other `attn_forward_method` choices can be covered by
  focused tests that intentionally instantiate the production DeepSeek module. These
  are not required for the main runner x attention compatibility matrix because the
  forward methods are performance paths with equivalent math.

Initial implementation slice:
- `common/dense_attention.py` contains a projected attention target that owns
  Q/K/V/O projections and dispatches through `RadixAttention`.
- `dense/test_torch_native.py`, `dense/test_triton.py`, and
  `dense/test_flashinfer.py` cover representative MHA and GQA cases.
- Dense tests exercise page size 1, exact-page extend, page-boundary decode
  (`seq_len = page_size - 1`, `page_size`, `page_size + 1`), zero/nonzero prefix,
  and GQA head grouping as an attention-config case.
- `swa/test_triton.py` covers no-prefix and prefix SWA extend with
  `seq_len < window_size`, `seq_len == window_size`, and
  `seq_len > window_size`.
- `swa/test_flashinfer.py` covers no-prefix SWA extend across the same window
  boundary cases. Prefix+SWA for FlashInfer is intentionally not enabled yet
  because the current synthetic metadata fixture does not faithfully match that
  production path.
- FlashInfer cases use `head_dim=64` to match FlashInfer kernel constraints.
- Future representative-first slices should complete missing attention families,
  runner modes, and speculative modes for `torch_native`, `triton`, and
  `flashinfer` before adding additional Phase 2 backend files.

---

### Phase 3 — Runner integration tests (`test/manual/attention/unittest/test_runner_integration.py`)

Use a smaller representative subset from Phase 2. The goal is runner bookkeeping, not
another full correctness matrix.

For each supported runner/backend/family case:

1. Run the eager module path and compare against the reference.
2. Capture/warm up the graph path:
   - CUDA graph: `init_cuda_graph_state`, capture metadata, warmup,
     `on_after_cuda_graph_warmup`, replay metadata.
   - BCG/PCG: use the corresponding runner APIs and active forward context.
3. Replay the graph path.
4. Assert graph replay output matches the eager output.

This phase should include `hybrid_attn` and TBO composition in a focused way instead
of as part of the full Cartesian product.

---

### Phase 4 — Speculative decoding attention (`test/manual/attention/unittest/test_spec_decoding_attention.py`)

#### 4a. Synthetic spec metadata correctness

For each supported synthetic spec case:

1. Construct synthetic `SpecInput` / EAGLE / Frozen-KV MTP / DFlash / NGRAM metadata
   matching `topk` and `num_draft_steps`.
2. Build `ForwardBatch` for `TARGET_VERIFY`, `DRAFT_EXTEND`, or `DRAFT_EXTEND_V2`.
3. Build module inputs and populate prefix/draft KV as required.
4. Enter `attention_test_context(backend)`.
5. Run `backend.init_forward_metadata(batch)` and module forward.
6. Build the explicit reference mask from `spec_info`.
7. Compare module output against the speculative reference.
8. Validate verify-buffer shape/content helpers.

#### 4b. Worker and draft-runner integration

For each selected worker family:

1. Run the worker path with tiny synthetic requests or minimal local fixtures.
2. Inspect the produced `ForwardBatch` and `SpecInput`.
3. Verify required fields, shapes, cache locations, positions, and mode.
4. For CUDA graph runners, compare graph replay output against eager output for the
   same worker/backend family.

Do not run a full Cartesian product of workers, draft runners, backends, `topk`, and
`num_draft_steps`. Use the capability matrix to select representative cases.

---

## Resolved Decisions

1. **Primary target boundary**: The main matrix uses model-level attention modules via
   `AttentionModuleTarget` adapters. Direct `RadixAttention` coverage is retained only
   as a small leaf-backend smoke suite.

2. **References**: References are family-specific. SDPA is only the reference for
   MHA/GQA/SWA-style dense attention and speculative masks after dense KV
   reconstruction; linear, Mamba, MLA, DSA, and DSV4 need their own references.

3. **Forward context**: All module-level tests publish `ForwardContext(attn_backend)`
   before backend metadata initialization and module forward.

4. **Chunked-KV**: `MHA_CHUNKED_KV` is a DeepSeek attention forward method, not a
   `ForwardMode`. Trigger it through the DeepSeek attention module by patching the
   module threshold or environment to a small value.

5. **Capability matrix**: Every parametrized case is filtered through an explicit
   support helper that returns a skip reason.

6. **Speculative decoding**: Phase 4 is split into synthetic metadata correctness and
   worker/draft-runner integration. It includes `DRAFT_EXTEND_V2`.

7. **DFlash**: DFlash is a speculative worker/path, not an attention backend.

8. **Mock model runner**: Use a `ModelRunner` subclass plus real request-token and KV
   pools so missing backend fields fail loudly and pool layout matches production.

9. **Graph verification**: Eager-vs-reference establishes correctness; graph-vs-eager
   establishes graph metadata and replay consistency.

---

## CI Registration

Use tiers instead of trying to fit the full matrix into one fast job.

- Fast CUDA PR tier: `register_cuda_ci()`, target under 60s. Cover representative
  MHA/GQA/SWA plus one MLA case across `torch_native`, `triton`, and `flashinfer`;
  include decode, extend, ragged lengths, and one paged-KV case.
- Medium CUDA tier: graph integration for selected backends and families, including
  CUDA graph/BCG/PCG where supported.
- Nightly CUDA tier: FA3/FA4/FlashMLA/Cutlass/TRTLLM/dual-chunk/hybrid/TBO and larger
  speculative matrices.
- Speculative tier: synthetic metadata correctness in fast/medium, worker integration
  in medium/nightly depending on cost.
- AMD/NPU/CPU/XPU tiers: register with the appropriate backend-specific CI helpers and
  skip CUDA-only runner modes.
