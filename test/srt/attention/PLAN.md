# Attention Backend Unit Test Plan

## Problem

### Issue 1 — Coverage gaps
Current tests do not cover arbitrary combinations of:
- Runners (eager, cuda_graph, PCG/BCG, EAGLE runner)
- Attention backends (with different page sizes or config variants)
- Input cases (various bsz, input len, KV cache size)

### Issue 2 — Test expense
Most tests are e2e (full server + model load + eval). That is slow and unnecessary for
verifying attention backend correctness.

---

## Approach

### Reference implementation
Use `torch_native` (SDPA) in eager mode as the universal reference instead of HuggingFace.

Rationale:
- No external dependency.
- No need to bridge paged-KV vs. dense-KV layouts.
- Validates all other backends against a known-correct, simple code path.
- HF comparison is reserved for SWA-specific logic where sglang's sliding-window mask
  semantics need external validation.

### Test structure
- No server launch, no model download, no eval harness.
- Random weights and random Q/K/V inputs.
- Synthetic `ForwardBatch` built from a helper, backed by real pool objects.
- Numerical comparison: sglang backend output vs. `torch_native` reference.

---

## Dimensions to enumerate

| Dimension | Values |
|---|---|
| Attention backend | `flashinfer`, `triton`, `torch_native`, `flashattention` (fa3/fa4), `flashinfer_mla`, `cutlass_mla` |
| Runner | eager, cuda_graph (regular / BCG), piecewise_cuda_graph (PCG) |
| Forward mode | `DECODE`, `EXTEND`, `MIXED` (chunked prefill) |
| Head config | Standard MHA, GQA (num_kv_heads < num_heads), SWA (sliding window) |
| Input shape | small bsz + short seq, large bsz + long seq, bsz=1 decode |

Not all combinations are valid. Constraints:
- MLA backends only apply to MLA model configs.
- SWA requires a window_size parameter.
- FA3/FA4 require SM90+.
- Some backends are HIP-only or XPU-only.

These are encoded as skip conditions (`skipIf`), not silently omitted.

Speculative decoding modes (`TARGET_VERIFY`, `DRAFT_EXTEND`) are a separate matrix
covered in Phase 4.

---

## Implementation phases

### Phase 1 — Infrastructure

**File**: `test/srt/attention/utils.py`

Functions to implement:

```python
def make_forward_batch(
    mode: ForwardMode,
    bsz: int,
    seq_lens: list[int],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    window_size: int | None = None,
) -> tuple[ForwardBatch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a minimal ForwardBatch with random KV data loaded into real
    ReqToTokenPool / TokenToKVPool instances. Also returns random Q, K, V tensors
    that match the batch layout.
    """

def run_backend(
    backend: AttentionBackend,
    batch: ForwardBatch,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Call init_forward_metadata + the appropriate forward_decode / forward_extend /
    forward_mixed method. Returns the output tensor.
    """

def assert_close_backends(
    ref_output: torch.Tensor,
    test_output: torch.Tensor,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    """Numerical comparison with configurable tolerance."""
```

Validation: replicate one existing triton decode test using these helpers to confirm
the infrastructure is correct before scaling.

---

### Phase 2 — Kernel-level unit tests (fast, no server)

**File**: `test/srt/attention/test_backend_correctness.py`

For each valid (backend, mode, head_config) combination:
1. Build `ForwardBatch` via `make_forward_batch`.
2. Run `torch_native` → reference output.
3. Run test backend → test output.
4. `assert_close_backends(reference, test)`.
5. Assert output shape and dtype.

Estimated matrix: ~6 backends × 3 modes × 3 head configs ≈ 54 combinations;
~25 valid after exclusions.

Target tolerance: atol=1e-2, rtol=1e-2 for float16; tighter for bfloat16 where
backends agree more closely.

SWA exception: compare sglang triton SWA kernel against
`torch.nn.functional.scaled_dot_product_attention` with an explicit causal+window mask,
not against `torch_native` backend (which may not implement SWA natively).

---

### Phase 3 — Runner integration tests (medium cost)

**File**: `test/srt/attention/test_runner_integration.py`

For each valid (runner, backend) pair:

**Eager path**:
- Call `init_forward_metadata` directly.
- Assert output matches Phase 2 reference.

**CUDA graph path** (regular / BCG):
1. `init_cuda_graph_state(max_bs, max_num_tokens)`
2. Warmup: `init_forward_metadata_capture_cuda_graph(...)` + forward pass.
3. `on_after_cuda_graph_warmup()`
4. Replay: `init_forward_metadata_replay_cuda_graph(...)` + forward pass.
5. Assert replayed output matches eager output.

**PCG path**:
- Same pattern as CUDA graph but using piecewise capture/replay API.

This phase tests that runner bookkeeping (graph capture, metadata buffer fill, replay)
does not corrupt results. The attention math itself is already validated in Phase 2.

---

### Phase 4 — Speculative decoding attention paths

**File**: `test/srt/attention/test_spec_decoding_attention.py`

Separate matrix for `TARGET_VERIFY` and `DRAFT_EXTEND` modes.

For each spec mode × backend combination:
1. Construct `SpecInfo` fixture with synthetic draft tree / draft tokens.
2. Build `ForwardBatch` with spec mode.
3. Compare output against eager `torch_native` reference with same inputs.

Also test:
- `get_verify_buffers_to_fill_after_draft()` returns correct buffer shapes.
- `update_verify_buffers_to_fill_after_draft(spec_info, cuda_graph_bs)` fills buffers
  without error.

---

## Open questions

1. **Page size axis**: FlashInfer requires specific page sizes (16, 32, 64). Add page size
   as a parameter in Phase 2, or fix one representative size per backend?

2. **SWA reference**: Validate sglang SWA against SDPA+explicit mask (simpler) or keep a
   small HF-based check for Gemma/Mistral configs?

3. **MLA backends**: `flashinfer_mla` and `cutlass_mla` use a compressed KV format.
   `make_forward_batch` may need an MLA variant. Include in Phase 2 or defer?

4. **Non-CUDA backends** (XPU, AMX, HIP): same framework with `skipIf` guards, or
   separate registration block?

---

## CI registration

- Phase 2 tests: register with `register_cuda_ci()`, mark fast (< 60 s total).
- Phase 3 tests: register with `register_cuda_ci()`, medium tier.
- Phase 4 tests: register with `register_cuda_ci()` under speculative decoding group.
- AMD/HIP variants: add `register_amd_ci()` where backends support HIP.
