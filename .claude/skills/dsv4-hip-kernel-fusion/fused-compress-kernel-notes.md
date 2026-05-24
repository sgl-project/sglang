# Fused Compressor Attention Kernel for DeepSeek V4 on HIP

## Overview

This document records the implementation of a fused Triton kernel that replaces 5-7 separate kernel launches in the DeepSeek V4 compressor decode path with a single kernel launch. The kernel fuses APE-add, overlap-transform, softmax-pool, RMSNorm, and RoPE.

**Files:**
- `python/sglang/srt/layers/attention/dsv4/fused_compress_kernel.py` — Triton kernel
- `python/sglang/srt/layers/attention/dsv4/compress_hip.py` — Integration into `CompressorHip`
- `python/sglang/srt/environ.py` — `SGLANG_OPT_USE_FUSED_COMPRESS_TRITON` env flag

**Enable:** `export SGLANG_OPT_USE_FUSED_COMPRESS_TRITON=true`

**Benchmark results (GSM8K, 100 questions, DeepSeek-V4-Pro, 8xMI300X):**

| Configuration | Accuracy | Throughput | Latency |
|---|---|---|---|
| Baseline (false) | 0.98 | 424 tok/s | 20.3s |
| Fused kernel (true) | 0.94 | 451 tok/s | 19.0s |

---

## Why Porting from ATOM Failed (v1)

The initial approach directly ported ATOM's `fused_compress_attn` kernel from `atom/model_ops/v4_kernels/fused_compress.py`. This produced **accuracy 0.01** and **throughput 240 tok/s** (both catastrophic). Three fundamental semantic mismatches caused this:

### 1. State Write Ordering

| Aspect | ATOM | SGLang |
|--------|------|--------|
| Order | Read old state FIRST, then write new state | Write current token FIRST, then read K entries from state |
| Rationale | ATOM's `fused_compress_attn` sees the previous fwd's state; `update_compressor_states` runs after | SGLang's `compress_decode_paged` writes the decode token to the state ring at line 325 BEFORE gathering the K entries for softmax-pool at line 347 |
| Impact | The current decode token is read from `kv_score_input` (ragged input) in ATOM, but from the state pool (after write-back) in SGLang |

This caused every compression boundary to pool over stale data missing the current token.

### 2. APE (Absolute Position Embedding) Handling

| Aspect | ATOM | SGLang |
|--------|------|--------|
| State storage | State stores score WITH APE pre-added (`update_compressor_states` fuses `score + ape` before write) | State stores RAW score (no APE) |
| At compress time | State scores already have APE; only ragged input scores need APE added | ALL K scores get APE added at read time (`kv_and_score_to_compress.score.add_(self.ape)`) |
| Impact of porting ATOM logic | Phase 1 (state read) skipped APE addition → wrong softmax weights. `update_compressor_states` added APE to state → double-APE accumulation across steps, cascading precision degradation |

### 3. Overlap Transform Cross-Boundary Dependency

ATOM's overlap transform is embedded inside the Triton kernel using column-offset remapping (`col_off = (k_static >= RATIO) * head_dim`). This works because ATOM's kernel addresses the state ring directly.

SGLang's `overlap_transform` for prefill has a CROSS-BOUNDARY dependency:
```python
new_tensor[:, r:] = tensor[:, :, d:]      # B-side from current boundary
new_tensor[1:, :r] = tensor[:-1, :, :d]   # A-side from PREVIOUS boundary
```

A Triton program seeing only one boundary at a time cannot access the previous boundary's data. This makes the prefill overlap transform impossible to fuse into a per-boundary kernel.

### 4. CUDAGraph Incompatibility

The plan-based approach required CPU->GPU tensor transfers (`numpy -> torch.Tensor.to(device)`) during decode. During CUDAGraph capture, `seq_lens.cpu()` triggers `HIP error: operation not permitted when stream is capturing`. This was fixed with a GPU-resident Triton plan builder, but the semantic issues above remained.

### 5. Performance Regression

The v1 kernel was slower (240 tok/s vs 424 tok/s baseline) because:
- CPU numpy plan builder added overhead per forward pass
- `torch.view_as_real(freqs_cis).flatten(-2).contiguous()` was called every forward (not cached)
- The kernel did redundant work: two-phase K-loop with separate state/input addressing instead of a single loop over pre-gathered data

---

## How the Working v2 Kernel Was Designed

The key insight: **don't replace the entire compressor — fuse only the compute-bound tail.**

```
[State write + Gather]  →  [APE-add + Overlap + Softmax + Norm + RoPE]
  keep as-is (tensor ops)     fuse into 1 Triton kernel (5-7 launches → 1)
```

### What the Kernel Receives

After the existing gather logic (unchanged, lines 321-342 in `compress_decode_paged`), we have a `KVAndScore` tensor `[bs, ratio*coff, 2*coff*head_dim]` already gathered from the state pool. The kernel operates on this tensor directly.

### What the Kernel Does (per batch element)

```
1. Loop over K_POOL entries (ratio * coff):
   a. Compute column offset for overlap transform (in-register reindex)
   b. Load kv[head_dim] and score[head_dim] from correct half
   c. Add APE to score
   d. Online softmax accumulation: m, kv_acc, w_acc
2. Divide: compressed = kv_acc / w_acc  →  [head_dim]
3. RMSNorm: var = sum(x^2)/dim, normed = x * rsqrt(var + eps) * weight
4. Store normed to output
5. Load back rope segment pairs, apply complex mul with freqs, store back
```

### Overlap Transform as In-Register Reindexing

For decode, `overlap_transform_decode` is:
```python
# Input:  [bs, 2*ratio, 2*head_dim]  (kv or score half)
# Output: cat(input[:, :ratio, :head_dim], input[:, ratio:, head_dim:])
#       = [bs, 2*ratio, head_dim]
```

In the kernel, this becomes a column-offset selection per K iteration:
```python
if OVERLAP:
    col_off = tl.where(k >= RATIO, head_dim, 0)
```
No data movement, no cross-batch dependency, no extra memory.

### freqs_cis Handling

`freqs_cis` is `complex64` `[max_seq, rope_dim/2]`. Triton doesn't support complex dtypes. The conversion `torch.view_as_real(freqs_cis).flatten(-2)` produces `float32` `[max_seq, rope_dim]` with interleaved `[cos0, sin0, cos1, sin1, ...]` layout.

This conversion is cached once in `CompressorHip._freqs_cis_real` (set in `_get_freqs_cis_real()`) to avoid per-forward overhead.

---

## Debugging Timeline

### Iteration 1: ATOM port (accuracy 0.01)
- Root cause: all three semantic mismatches above
- Symptom: GSM8K accuracy dropped from 0.98 to 0.01

### Iteration 2: Fixed semantics (accuracy 0.95)
- Rewrote kernel to read from pre-gathered tensor (not state pool directly)
- Added APE to ALL scores (not just ragged input)
- Kept state write + gather logic unchanged
- Symptom: accuracy restored to 0.95, but throughput 291 tok/s (slower than baseline)

### Iteration 3: Removed dead code (accuracy 0.74)
- Removed `-inf` guards from softmax: `tl.exp(-inf - (-inf))` = `exp(NaN)` = NaN
- NaN propagated through layers causing silent precision degradation
- Lesson: online softmax MUST guard against `-inf` operands

### Iteration 4: Restored NaN guards + cleanup (accuracy 0.94, throughput 451 tok/s)
- Restored `tl.where(m_prev == float("-inf"), 0.0, ...)` guards
- Added `num_warps` tuning
- Removed unused kernel parameters
- Final result: accuracy 0.94, throughput 451 tok/s (+6.3% over baseline)

---

## Key Lessons

1. **Never port kernel semantics blindly across frameworks.** ATOM and SGLang have different state management, APE handling, and write ordering. Always trace the exact reference code line-by-line before writing a kernel.

2. **Online softmax requires `-inf` guards.** When `m_prev = -inf` and `score_k = -inf` (both padding), `max(-inf, -inf) = -inf`, then `exp(-inf - (-inf))` is NaN. The guard `tl.where(x == -inf, 0.0, tl.exp(x - m_new))` is essential.

3. **Fuse the compute tail, not the entire pipeline.** The gather/addressing steps are already batched tensor ops. Fusing them into a kernel adds complexity without benefit. The real win is fusing the 5-7 sequential compute kernels (APE, overlap, softmax, norm, RoPE) into one.

4. **Cache complex-to-real conversion.** `torch.view_as_real(freqs_cis).flatten(-2).contiguous()` allocates a new tensor every call. Caching it once eliminates repeated allocation + copy.

5. **CUDAGraph requires fully GPU-resident ops.** Any `.cpu()`, `.numpy()`, or `.item()` call during stream capture will crash with `HIP error: operation not permitted when stream is capturing`.

---

## Iteration 5: Consistency Testing Reveals CSA Overlap Bug

Three consecutive benchmark runs showed inconsistent accuracy (0.84/0.95/0.93) vs baseline (0.98/0.96/0.95). Root cause analysis:

### The CSA Overlap Reshape Bug

For CSA (ratio=4, coff=2, overlap=True), the gathered tensor from state pool has shape `[bs*2, 4, 4*head_dim]` via KVAndScore.view. The kernel reshape `raw.reshape(bs, 8, 4*head_dim)` concatenates the two coff-groups in memory order:

```
Memory: [coff0_row0, coff0_row1, coff0_row2, coff0_row3, coff1_row0, coff1_row1, coff1_row2, coff1_row3]
```

But the reference `overlap_transform_decode` does:
```python
# A-side = input[:, :ratio, :head_dim]     (coff0, first head_dim cols)
# B-side = input[:, ratio:, head_dim:]     (coff1, last head_dim cols)
# Output = cat(A, B) along dim=1
```

This is NOT equivalent to the kernel's `col_off = tl.where(k >= RATIO, head_dim, 0)` applied on the reshape'd rows, because the reshape interleaves different semantic groups.

### HCA (ratio=128) Performance Problem

For HCA (no overlap, K=128), the kernel works correctly but is SLOWER than the unfused path because a 128-iteration loop in a single Triton program is very sequential, while the unfused path uses batched `softmax(dim=1).sum(dim=1)` which parallelizes across the K dimension.

### Resolution

The fused kernel is disabled by default (`use_fused_compress_triton` returns `False`). The kernel code remains for reference and future development.

---

## Current Status

The fused kernel is **disabled**. The code remains in `fused_compress_kernel.py` as a reference implementation. To enable it requires solving:

1. **CSA overlap transform**: Need a kernel that correctly reindexes the A/B halves from the `[bs*coff, ratio, 2*coff*head_dim]` layout without an intermediate reshape. Possible approach: pass the coff-group strides to the kernel and have it directly index `kv_score_ptr[bid*coff + coff_group, k_in_group, col_offset + d]`.

2. **HCA K=128 performance**: The single-program K=128 loop needs either (a) a split-K approach with multiple programs per batch element, or (b) accepting that HCA's large K makes per-element fusion less beneficial than batched ops.

## Future Optimization Opportunities

1. **CSA-specific kernel with coff-aware indexing**: Instead of reshape, pass `coff_stride` and have the kernel directly index `[coff_group * coff_stride + k * row_stride + col_off + d]`. This avoids the reshape entirely and handles A/B halves correctly.

2. **Split-K for HCA**: Launch `(bs, num_k_tiles)` grid where each program handles `K/num_k_tiles` entries, then a second reduction kernel combines the partial sums. This parallelizes the K=128 loop.

3. **Prefill extend path**: The per-seq Python loop is the bigger bottleneck. The overlap transform for prefill has cross-boundary dependencies (`new_tensor[1:, :r] = tensor[:-1, :, :d]`) that require inter-program communication.

4. **RoPE without store-load roundtrip**: Use `tl.reshape` + `tl.split` to keep everything in registers. Non-trivial for non-power-of-2 rope dimensions.
