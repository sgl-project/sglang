# Attention Token Capture: Design Decisions & Review Response

This document addresses engineering concerns raised in PR #16398 review.

## Concern 1: PagedAttention Index Translation

**Risk:** Returning physical block offsets instead of logical token indices.

**Implementation:** The `kv_indices` tensor provides the logical-to-physical mapping. The triton kernel uses `kv_indices[kv_start + offset]` to translate chunk-local positions back to logical token indices before returning.

```python
# From decode_attention_with_topk.py:469
kv_pos = kv_indices[kv_start + chunk_start:kv_start + chunk_end]

# Final output returns logical positions
"token_positions": indices[b].cpu().tolist()  # Line 1001
```

**Result:** All returned `token_positions` are logical indices into the original prompt, not physical block offsets.

---

## Concern 2: Bandwidth Problem (40KB per token)

**Risk:** 32 layers × 32 heads × Top-10 × 4 bytes = 40KB per token.

**Implementation:** Three operating modes with GPU-side aggregation:

| Mode | Output Size | Use Case |
|------|-------------|----------|
| `raw` | ~200 bytes/step | Debug/short generations |
| `sketch` | ~500 bytes/layer | Long outputs (86k+) |
| `fingerprint` | 64 bytes/step | Production routing |

### Key Parameters:

```bash
# GPU-aggregated fingerprint (64 bytes vs 200KB)
--attention-fingerprint-mode

# Per-layer sketches (top_hubs, dist_hist, entropy)
--attention-sketch-mode

# Select specific layers (not all 32)
--attention-capture-layers "last"   # Default: only last layer
--attention-capture-layers "auto"   # 4 layers spread across depth
--attention-capture-layers "7,15,23,31"  # Custom selection

# Early exit (stop after manifold stabilizes)
--attention-fingerprint-max-steps 256
```

### Memory Efficiency:

The chunked extraction avoids materializing the full attention matrix:

```
For 1M context with chunk_size=4096, k=10:
  Intermediate: ~6MB during computation (vs 4GB for full matrix)
  Final: 80 bytes per sequence
```

---

## Concern 3: Batching Heterogeneity

**Risk:** Requests with attention capture slowing down requests without it.

**Implementation:** Per-request flags in the scheduler:

```python
# From schedule_batch.py
class ScheduleBatch:
    return_attention_tokens: bool = False  # Per-request flag
    attention_capture_layer_ids: List[int] = []
```

The attention extraction kernel (`extract_top_k_attention`) is called **after** the main forward pass, not integrated into the attention kernel. This means:

1. The performance-critical attention forward remains unchanged
2. Extraction overhead only applies to requests that requested it
3. Requests can share a batch even with different capture settings

**Trade-off:** There is slight overhead from the separate extraction pass, but it's isolated to requesting sequences.

---

## Concern 4: Privacy / System Prompt Leakage

**Risk:** Attention patterns reveal system prompt structure and length.

**Status:** ✅ Implemented

**Implementation:**

```bash
# Server-wide defaults
--attention-mask-system-prompt   # Auto-detect and mask system prompt tokens
--attention-mask-prefix 100      # Manual: mask first 100 tokens

# Per-request override via API
extra_body={
    "attention_mask_prefix": 50  # Mask first 50 tokens for this request
}
```

**How it works:**
1. `apply_attention_privacy_mask()` filters positions < mask_prefix
2. Remaining positions are offset so masked region appears as position 0
3. Scores and logits are filtered in parallel
4. Applied to raw mode, sketch mode, and multi-layer captures

**Example:**
```python
# Original: positions [5, 15, 25] with mask_prefix=10
# Result:   positions [5, 15] (offset: 15-10=5, 25-10=15)
#           position 5 is filtered (< 10)
```

---

## Summary: How Concerns Are Addressed

| Concern | Status | Implementation |
|---------|--------|----------------|
| PagedAttention index translation | ✅ Addressed | `kv_indices` mapping in triton kernel |
| Bandwidth (40KB/token) | ✅ Addressed | fingerprint (64B), sketch (~500B/layer), layer selection |
| Batching heterogeneity | ✅ Addressed | Post-forward extraction, per-request flags |
| Privacy/system prompt | ✅ Addressed | `attention_mask_prefix` parameter + `apply_attention_privacy_mask()` |

---

## Use Case Enablement

The implementation enables all 10 use cases from the review:

| Use Case | Mode | Enabled By |
|----------|------|------------|
| RAG Lie Detector | raw/sketch | `token_positions` + `attention_scores` |
| CoT Auditing | raw | Multi-step position tracking |
| Show Your Work UI | raw | Real-time streaming + position mapping |
| Ghost Context Pruning | fingerprint | Aggregated histograms over sessions |
| Smart Cache Eviction | fingerprint | `hubness` metric per position |
| Model Distillation | raw | Full attention pattern export |
| Prompt Injection Defense | raw | Monitor attention on system tokens |
| Lost in Middle Diagnosis | sketch | `dist_hist` distribution visualization |
| Bias Detection | raw | Position-specific attention analysis |
| Speculative Decoding Debug | raw | Cross-model attention comparison |
