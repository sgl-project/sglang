# 05 — flash-topk-attention (ftka) kernel library

Audience: an ML engineer porting a minimal-but-performant Double-Sparse-style
path into sglang for DeepSeek-V3.2 / GLM-5.1 (MLA attention).

Scope: the `flash-topk-attention/` sibling repo under
`/sgl-workspace/sglang/development/past_implementations/Twilight/`, and its
(non-)relationship to the Twilight pyimpl runtime at
`/sgl-workspace/sglang/development/past_implementations/Twilight/twilight/`.

Convention: **Observed** = read from the source tree; **Inferred** = inference
from observed code/comments. Every non-trivial claim is anchored to a file
path and (where useful) a line range.

---

## 0. Headline (Observed)

ftka is **not** a runtime dependency of Twilight. The only call site of ftka
in this entire tree is the GEMV microbenchmark
`benchmark/efficiency/bench_gemv.py` (lines 8–10 do
`from ftka.cuda_ops.gemv import ...`,
`from ftka.triton_ops.tokens_moving import discontinuous_move_tokens`,
`from ftka.utils.benchmark import benchmark_forward_with_warmup`).
A `grep -r "ftka\|flash.topk" --include="*.py"` over the whole tree returns
hits only from inside `flash-topk-attention/` itself plus that one bench file.

Twilight's runtime (`twilight/pyimpl/*.py`, `twilight/kernel/triton/*.py`,
`twilight/kernel/cuda/sampling.py`) uses its own Triton kernels —
`bgemv_int8.py`, `channel.py`, `qk_int8_per_block.py` — and `torch.topk` for
top-k selection (`twilight/pyimpl/top_k.py:12`,
`twilight/pyimpl/double_sparse.py:121,148`, `twilight/pyimpl/quest.py:99,201`).
The README acknowledges this status with the dangling link
"[Flash-TopK-Attention (Stay Tuned)]" (Twilight `README.md:25`) and the
description on line 100: "We have organized an implementation of
Flash-TopK-Attention using FlashInfer(CUDA), Triton, and TileLang for the
existing top-k algorithm."

> **The accuracy numbers in Twilight's README table (LongBench / RULER /
> Passkey) were produced WITHOUT ftka.** ftka is a sibling kernel library
> built by the same authors but not wired into Twilight's evaluation harness.

---

## 1. Kernel inventory

| File | Python entry | What it computes | Shapes / dtypes (Observed) | Backend | Test |
|---|---|---|---|---|---|
| `csrc/include/gemv.cuh` (single, lines 137–504) | `cuda_ops.gemv.single_sparse_gemv` | Single-request fp16 `q @ k.T` decode-GEMV over a contiguous k tensor; FlashInfer-style pipelined `cp.async` ping-pong | `q[H,D] · k[S,Hkv,D] → o[H,S]`, fp16 (`tests/test_gemv.py:23–33`) | CUDA / FlashInfer JIT | `tests/test_gemv.py::test_single_sparse_gemv` |
| `csrc/include/gemv.cuh` (batched, lines 246–554) | `cuda_ops.gemv.batched_sparse_gemv` | Batched paged-KV decode-GEMV `q @ k.T`, k pulled via `paged_kv_t` (indptr/indices/last_page_len) | `q[B,H,D]`, `o[B,H,1,S]`, `k_cache[NK,Page,Hkv,D]`, indptr `[B+1]`, indices `[NK]`, last `[B]`, fp16 (`test_gemv.py:218–242`) | CUDA / FlashInfer JIT | `tests/test_gemv.py::bench_sparse_gemv_fp16` |
| `csrc/include/gemv.cuh` + `dequantize.cuh::fast_dequantize_i8s_to_fp16s` | `cuda_ops.gemv.batched_sparse_gemv_int8_k` | Same as `batched_sparse_gemv`, but k\_cache is uint8 and dequantized on-the-fly with per-(page,token,head) `quant_scales` + `quant_zeros` | `k_cache[NK,Page,Hkv,D]` uint8 (signed offset +128, see test line 94 "INTERLEAVED and +128"), `scales/zeros[NK,Page,Hkv]` fp16 (`test_gemv.py:99–113`) | CUDA / FlashInfer JIT | `tests/test_gemv.py::test_sparse_gemv_int8` |
| `csrc/include/gemv.cuh` + `dequantize.cuh::fast_dequantize_i4s_to_fp16s` + `lowbit_dtypes.cuh::uint_4bit` | `cuda_ops.gemv.batched_sparse_gemv_int4_k` | Same as int8 variant, but k\_cache packed uint4 (head\_dim/2 storage), uses `lowbit::uint_4bit` tag dispatch | `k_cache[NK,Page,Hkv,D/2]` int8 storage interpreted as uint4 pairs (`test_gemv.py:395–408`); scales/zeros same as int8 | CUDA / FlashInfer JIT | `tests/test_gemv.py::test_sparse_gemv_int4` |
| `csrc/include/quest_gemv.cuh` | `cuda_ops.gemv.quest_sparse_gemv` | "MaxPossibleSampleWithPagedKVCache": per-page (max,min) → upper-bound page score `Σ_i max(q_i·max_i, q_i·min_i)`. Score is the **Quest** chunk-level upper-bound estimator | `q[B,H,D]`, `o[B,H,S]`, `k_cache,v_cache[NK,Page,Hkv,D]` (where k_cache stores per-page max, v_cache stores per-page min — see line 134 comment) (`test_gemv.py:317–352`) | CUDA / FlashInfer JIT | `tests/test_gemv.py::test_quest_sparse_gemv` |
| `csrc/include/raft_topk.cuh` (vendored from RAPIDS RAFT 25.12) | `cuda_ops.topk.raft_topk` | Batched radix-select top-k via `decode_select_k → radix_topk_one_block_kernel<T,IdxT,BitsPerPass=8,BlockSize=512>` | `in[B,L]`, `in_idx[B,L]`, `out[B,k]`, `out_idx[B,k]`, `buf[B, *]` scratch, dtype = fp16/bf16/fp32, idx int32 (`test_topk.py:25–69`) | CUDA / FlashInfer JIT | `tests/test_topk.py::bench_topk` |
| `ftka/triton_ops/tokens_moving.py:33–108` | `triton_ops.tokens_moving.discontinuous_move_tokens` | Discontinuous gather of `[N,H,D]` tokens from src into dst by `src_indices`/`dest_indices`; one program per token, BLOCK_H = next_pow2(H), BLOCK_D = head_dim | `src[Nsrc,H,D]`, `dst[Ndst,H,D]`, `src_idx[T]`, `dst_idx[T]` (long); 1 warp per token | Triton | (used by `test_gemv.py:285`) |
| `ftka/triton_ops/tokens_moving.py:112–226` | `move_tokens_from_blocked_k_cache` | vLLM-layout `[NB,H,D/x,B,x]` → flat `[N,H,D]` gather (block→token reshape on the fly) | as above plus block_size | Triton | local `if __name__` block, line 429 |
| `ftka/triton_ops/tokens_moving.py:230–321` | `move_tokens_from_blocked_v_cache` | vLLM-layout `[NB,H,D,B]` → flat `[N,H,D]` gather | as above | Triton | local `if __name__` block, line 430 |
| `ftka/triton_ops/gemv.py:13–66` | `triton_ops.gemv.bgemv` | fp16 label-GEMV (Q_label @ K_label.T per (B,H)), exact mirror of DoubleSparse's `bgemv.py` from `andy-yang-1/DoubleSparse` (see header comment line 1) | `Q_Label[B,H,r]`, `K_Label[B*S,H,r]`, `Out[B,H,S]`, fp16, `r=HEAVY_CHANNEL_NUM` | Triton | local `if __name__` block |
| `ftka/tilelang_ops/__init__.py` | (empty) | — | — | TileLang (stub) | — |

CUDA headers in detail:

| Header | Purpose | Provenance (Observed in license headers) |
|---|---|---|
| `csrc/include/gemv.cuh` (558 LoC) | `SingleSparseGEMVKernel`, `BatchedSparseGEMVKernel`, host wrappers | "Modified from … FlashInfer team" (line 2) |
| `csrc/include/quest_gemv.cuh` (311 LoC) | `MaxPossibleSampleWithPagedKVCacheKernel` (Quest page-MaxMin score) | "modified based on … flashinfer/include/flashinfer/decode.cuh … Support for Page-Sparsity Self-Attention by dynamic selection" (lines 17–21) |
| `csrc/include/dequantize.cuh` (202 LoC) | `fast_dequantize_i8s_to_fp16s`, `fast_dequantize_i4s_to_fp16s` (PTX `prmt`/`sub.f16x2` based) | "Learned from … FasterTransformer interleaved_numeric_conversion.h" (lines 1–4) |
| `csrc/include/lowbit_dtypes.cuh` (323 LoC) | `lowbit::uint_4bit` tag, `size_of_type<T>()` for 0.5-byte packed types, `vec_t<uint8_t>` specializations | "Modified from … Atom/kernels/include/flashinfer/quantization.cuh" (lines 1–5) |
| `csrc/include/raft_topk.cuh` (1014 LoC) | `decode_select_k` host entry + `radix_topk_one_block_kernel` + `Counter`/`scan`/`choose_bucket`/`filter_and_histogram`/`last_filter` | "Modified from … rapidsai/raft/branch-25.12/.../matrix/detail/select_radix.cuh" (lines 1–7) |
| `csrc/include/raft/macros.h`, `integer_utils.h`, `pow2_utils.h`, `vectorized.h` (936 LoC total) | Vendored slice of RAFT support utilities (`_RAFT_HOST_DEVICE`, `Pow2<aligned>::roundDown`, `vec_t<T>` vectorized I/O) | "NVIDIA CORPORATION" (line 1) |

The loading mechanism (Observed): `ftka/cuda_ops/module.py:36–55` lazily
calls FlashInfer's `load_cuda_ops("ftka_ops", sources=[FTKA_CSRC_DIR/"ftka_ops.cu"],
extra_include_paths=[FLASHINFER_INCLUDE_DIR, FLASHINFER_CSRC_DIR,
FTKA_INCLUDE_DIR])`. The C++ `PYBIND11_MODULE` block at
`csrc/src/ftka_ops.cu:212–221` registers all six entry points. Dtype dispatch
inside each cu function uses `DISPATCH_PYTORCH_DTYPE_TO_CTYPE` from
`csrc/src/pytorch_extension_utils.h:88–134` (fp16 / bf16 / fp8\_e4m3 /
fp8\_e5m2 cases, gated by `FLASHINFER_ENABLE_BF16`/`FLASHINFER_ENABLE_FP8`).
There is no AOT build; everything is JIT, cached in
`~/.cache/flashinfer` (per ftka `README.md:24`).

---

## 2. The "top-k" gap

**Observed.** Twilight pyimpl uses `torch.topk` (or `.topk`) wherever top-k is
needed:
- `twilight/pyimpl/top_k.py:12` — `_, indices = attn_weights.topk(k=k, dim=-1)`
  then `mask.scatter_(...)`. Returns a boolean mask the same shape as
  `attn_weights`. Used by `oracle_topk_selector`.
- `twilight/pyimpl/double_sparse.py:121,148` — `return top_k(estimated_weights,
  token_budget)`.
- `twilight/pyimpl/quest.py:99,201` — `estimated_weight.topk(k=_chunk_budget,
  dim=-1)` on chunk scores.

**Observed.** ftka exposes `raft_topk` as a drop-in batched top-k. From
`csrc/src/ftka_ops.cu:198–210` and `csrc/include/raft_topk.cuh:984–1010`:

```cpp
template <typename T, typename IdxT>
void decode_select_k(const T* in, const IdxT* in_idx, char* bufs,
                     int batch_size, IdxT len, IdxT k,
                     T* out, IdxT* out_idx, bool greater = true)
```

Algorithm:
- Radix-select (`raft::matrix::detail::select_radix`), `BitsPerPass=8`,
  `BlockSize=512`, one block per row of the `(B,L)` matrix
  (`raft_topk.cuh:986–990`, `radix_topk_one_block_kernel` at line 896).
- 8 bits per pass → ceil(sizeof(T)*8 / 8) = 2 passes for fp16, 4 for fp32.
- `select_min = !greater` is wired through from `decode_select_k(..., true)`
  (line 207–208 of `ftka_ops.cu`: `true` is passed for `greater`, so
  `select_min=false` ⇒ top-k largest).

**Semantic constraints / gotchas for use as a `torch.topk` replacement:**

| Concern | Observation | Reference |
|---|---|---|
| Output sorting | **Unsorted**. `filter_and_histogram_for_one_block` (lines 866–893) uses `atomicAdd(p_out_cnt, 1)` to assign output positions; final ordering is atomic-arrival order, not by magnitude. The header comment makes this explicit: "we can skip some passes in the end at the cost of having an unsorted output" (line 145). | `raft_topk.cuh:145, 887–891` |
| Tie-breaking | Bit-equal values land in the same bucket; resolved by atomic-arrival order (non-deterministic across runs). | implicit from line 887 |
| Supported `T` | fp16, bf16, fp32 via the FlashInfer dispatch macro (`pytorch_extension_utils.h:88–114`). The `lower_bound`/`upper_bound` specializations cover `half` (line 100), `__nv_bfloat16` (line 119 area), and the cub `Traits<T>::TwiddleIn` path covers fp32/int32. | `raft_topk.cuh:100–128, 170–186` |
| `IdxT` | int32 only (hard-coded at the call site, `ftka_ops.cu:204`). For sequence lengths > 2^31 this is fine; for an int32 index over the whole `B*L` flat tensor it would not be — but `raft_topk` indexes within each row only. | `ftka_ops.cu:198–209` |
| `k` bound | No explicit upper bound in code; `k>len` would degenerate. The bench tests `k=512..2048` against `L=2048..32768` (`test_topk.py:73–80`). | — |
| Scratch buffer | Caller-allocated `buf` shaped roughly `[B, calc_buf_len(L) * 2 * (sizeof(T)+sizeof(IdxT))]`. Test allocates an empirical `(B, 8192*2*6//2//48)` fp16 buffer (`test_topk.py:39`). | `raft_topk.cuh:921–922, 199–215` |
| `in_idx` | Caller supplies pre-filled `arange(L)` per row (`test_topk.py:36`); kernel writes out the int32 indices selected. | `test_topk.py:36, ftka_ops.cu:205` |

**Could it drop-in replace `torch.topk` in Twilight pyimpl?** (Inferred)
*For the index-only use case (`_, indices = scores.topk(k)`), yes, after
flattening to `(B, L)`. Three obstacles:*
1. **Unsorted output.** Twilight's downstream consumer is
   `mask.scatter_(dim=-1, index=indices, value=True)` (top_k.py:14), which is
   permutation-invariant — so the unsorted output is **harmless** for the
   pyimpl masked-attention path. For a "gather then dense-attend"
   replacement, unsorted indices would force a per-token gather anyway, so
   still fine.
2. **Scratch allocation.** Caller has to size `buf` correctly. This is
   trivially solved by computing once at warmup. Important caveat: dynamic
   `L` (sequence-length-dependent buf size) means warmup-time allocation
   must be the max `L`, or buf must be re-allocated on resize — which would
   break cuda-graph capture if it crosses a graph boundary.
3. **CUDA-graph friendliness.** `decode_select_k` is a fixed sequence of
   `cudaLaunchKernel` calls with no host sync; the `Counter` struct lives
   on-device (`raft_topk.cuh:901`). This should capture cleanly. The
   pyimpl `torch.topk` path, in contrast, returns a tensor sized by `k` but
   does not synchronize unless the caller `.item()`s; what *does* break cuda
   graphs is the **`scatter_` with a dynamically-sized index tensor** in
   `top_k.py:14`. raft_topk does not fix that — Twilight's pyimpl pipeline
   would still need a fixed-`k` rewrite.

**Test ground-truth** (`tests/test_topk.py:30–67`): the bench compares
`raft_topk` against `torch.topk` only by *sum of selected values* (line 33–
67), not by index equality — explicitly because "the random value leads to
similar tensor … instead, we pay attention to the qk result" (line 13–14).
This is consistent with the unsorted-output / nondeterministic-tie
properties above.

---

## 3. The GEMV variants

All five variants share a layout assumption: **NHD** (sequence-major,
then heads, then head_dim) and a fixed `HEAD_DIM=128` template argument
plumbed from `ftka_ops.cu` (every `BatchedSparseGEMV<128, ...>` call).
"Sparse" in the name means **sparse along the sequence axis** —
the kernels assume the caller has already selected a subset of pages
(via `paged_kv_indices` + `indptr`) and the GEMV runs over precisely
that subset. There is no sparsity along heads or channels.

| Variant | Sparse axis | Q/K shape (Observed) | Quantization scheme | Test (Observed) |
|---|---|---|---|---|
| `single_sparse_gemv` | sequence (single-request, no batch) | `q[H,D]`, `k[S,Hkv,D]`, `o[H,S]` (`ftka_ops.cu:25–45`) | none | `test_gemv.py:13–63` (compares against `q.unsqueeze(1) @ k.transpose(0,1).transpose(1,2)`) |
| `batched_sparse_gemv` | sequence (batched paged KV) | `q[B,H,D]`, `o[B,H,1,S]`, `k_cache[NK,Page,Hkv,D]` + `(indices,indptr,last_page_len)` (`ftka_ops.cu:47–77`) | none (fp16/bf16) | `test_gemv.py:192–301` |
| `batched_sparse_gemv_int8_k` | sequence | as above, k_cache uint8 | per-(page, token, head) `scales[NK,Page,Hkv]` + `zeros[NK,Page,Hkv]`; both fp16 (`test_gemv.py:99–113`) | `test_gemv.py:67–188` |
| `batched_sparse_gemv_int4_k` | sequence | as above, k_cache uint4 packed (D/2 bytes) | same per-(page,token,head) scales/zeros, fp16 (`test_gemv.py:413–427`) | `test_gemv.py:381–502` |
| `quest_sparse_gemv` | sequence + **page-level approximation**: input is per-page (max,min), not raw K | `q[B,H,D]`, `o[B,H,S]`, `k_cache=PER_PAGE_MAX[NK,Page,Hkv,D]`, `v_cache=PER_PAGE_MIN[NK,Page,Hkv,D]` (`ftka_ops.cu:149–183`; quest_gemv.cuh:134–140 comment) | none on the (max,min) values themselves | `test_gemv.py:304–378` |

**Math.**
- `single_sparse_gemv`, `batched_sparse_gemv`: standard `q · k` accumulation,
  `compute_qk` in `gemv.cuh:51–112` does `__hfma` over `vec_size` lanes,
  shuffle-reduces across the `bdx` warp lanes (line 100–104), and stores one
  fp16 score per (head, token) into `o`.
- `int8_k` / `int4_k`: same `compute_qk`, but `k_smem` holds packed
  uint8/uint4; line 65–70 dispatches to `fast_dequantize_i4s_to_fp16s`
  (PTX-level `prmt` interleave + magic-constant subtract for sign recovery,
  `dequantize.cuh:34–80`) or the int8 variant; `scale`/`zero` per-row are
  loaded into smem alongside.
- `quest_sparse_gemv`: instead of `q·k`, computes the Quest upper-bound
  estimator
  `max_possible = Σ_i max(q_i * max_i, q_i * min_i)` (`quest_gemv.cuh:80–
  89`). This is the page-level "MaxPossible" score from Quest §3.2.

**Reference benchmark shapes** (`bench_gemv.py:13–502`, mirrored from
`tests/test_gemv.py`): `batch_size=16, num_heads=32, head_size=128,
page_size=1`, `seq_len ∈ {256, 512, 1024, 2048, 4096, 8192, 16384, 32768}`.
These shapes correspond roughly to Llama-3 8B / 70B-shaped attention with
GQA group-size 1 (note `page_size=1` means token-granular, not page-
granular, "paged" KV).

**Functional equivalence to Twilight pyimpl's `bgemv_int8`** (Observed via
side-by-side):
- `twilight/kernel/triton/bgemv_int8.py:13–50` (`bgemv_int8_kernel`):
  - Q is fp16 `[B,H,r]` (`r = HEAVY_CHANNEL_NUM`, typically 8).
  - K is int8 `[B*S, H, r]`.
  - `K_Scales` is fp16 `[B*S, H]` — **per-(token, head)**, no zero point.
  - Computes `Σ_d Q[d] * K[d]`, then `att_value *= K_Scales`.
  - Output is fp16 `[B, H, S]`.
- ftka `batched_sparse_gemv_int8_k` (`ftka_ops.cu:79–112`, `gemv.cuh:246–
  554`):
  - Q is fp16 `[B,H,D]` with `D=128` (full head_dim, not the "heavy
    channel" subset).
  - K is uint8 `[NK,Page,Hkv,D]` paged + indexed.
  - Has both `scales` and `zeros` (allows asymmetric int8).
  - Computes `Σ_d Q[d] * dequantize(K[d])`, all in fp16.
  - Output is fp16 `[B,H,S]`.

**Conclusion (Inferred):** ftka's `batched_sparse_gemv_int8_k` is **not** a
drop-in replacement for Twilight's `bgemv_int8` — the contracts differ in
the most important place: ftka assumes the *full head-dim 128 channel
loop* (general approximate decode attention over a paged selection), while
DoubleSparse/Twilight's `bgemv_int8` assumes the *r-dimensional
heavy-channel subset* (r ≪ D). They are different operators that happen to
share a shape signature; to use ftka here you would have to (a) pre-gather
the heavy-channel labels into a packed `D=r=128` (likely with r=128
by padding heads' heavy-channel sets to a uniform 128), (b) re-quantize per-
token-per-head, and (c) accept the asymmetric scale+zero overhead. The
benefit is the FlashInfer-grade `cp.async` pipelining and PTX-level
dequant which the Twilight Triton kernel does not match.

The same equivalence/extension story applies to:
- `single_sparse_gemv` ≈ DoubleSparse `bgemv` (fp16, batch=1).
- `batched_sparse_gemv` (fp16) ≈ DoubleSparse `bgemv` extended to batched
  paged KV.
- `batched_sparse_gemv_int4_k` extends int8 to int4 — Twilight has no
  Triton analogue.
- `quest_sparse_gemv` extends the family with a different *operator*
  (MaxMin upper bound) — closest Twilight analogue is the
  `quest_pageinfo_select`/`approx_pageinfo` logic in `pyimpl/quest.py`,
  which does the same math in pure torch (see `pyimpl/quest.py:75–103`).

---

## 4. `discontinuous_move_tokens`

**Observed** (`ftka/triton_ops/tokens_moving.py:33–108`).

Contract:
- Inputs: `src_storage[Nsrc, H, D]`, `dst_storage[Ndst, H, D]`,
  `src_indices[T]`, `dst_indices[T]` (both int64).
- Per-token program (`grid=(T,)`, `num_warps=1`).
- Each program loads token `src_storage[src_indices[t]]` (shape `[H,D]`,
  HEAVY_CHANNEL_NUM padded to `next_power_of_2(H)`) and stores it at
  `dst_storage[dst_indices[t]]`.
- Modeled after `lightllm/.../destindex_copy_kv.py` (per the docstring) and
  `parrot` (per the copyright header line 6).

This is exactly the missing "physical gather of selected K/V rows" kernel
that the runtime would need to fuse top-k selection with a *dense* attention
call over the selected tokens (instead of computing a *masked* attention
over the full KV cache).

**Where it would fit in the Twilight / sglang DS path (Inferred).** Twilight
pyimpl uses two patterns for "apply the mask":
1. **Masked attention.** `pyimpl/double_sparse.py` / `pyimpl/quest.py`:
   `mask.scatter_(...)` then mask the K/V before standard SDPA. This does
   *not* save compute on the GPU — it only saves arithmetic if the mask is
   fused into a sparse-attention kernel (which `pyimpl` does not have; it
   relies on a downstream `flash_attn` variant).
2. **Index gather.** Not used in pyimpl. The DoubleSparse "real" path
   gathers K_Label rows via `get_label_tensor` (`twilight/kernel/triton/
   channel.py:11–53`) but never gathers full-D K rows.

`discontinuous_move_tokens` is the right primitive for a third pattern:
*pre-gather selected K/V into a contiguous buffer of size budget, then call
a dense attention kernel*. This is the pattern that would map cleanly onto
sglang's existing FlashInfer / FlashAttention backends.

**Catch (Observed).** The kernel parallelizes only over `num_tokens`, with
one warp per token. For budget = 1024 and H*D = 32*128 = 4096 fp16 = 8 KiB
per token, this is ~8 MiB of writes per layer per request — bandwidth-bound,
not latency-bound — and the 1-warp-per-token launch will leave the SMs
under-occupied on large batches. The vLLM-layout sibling kernels
`move_tokens_from_blocked_{k,v}_cache_kernel` (lines 112–321) are the same
shape with different stride math.

---

## 5. TileLang & FlashInfer JIT

**TileLang** (Observed). `ftka/tilelang_ops/__init__.py` is **empty** (0
lines, confirmed via `wc -l`). The README in `flash-topk-attention/README.md`
makes no mention of TileLang either. Twilight's parent README (line 100)
claims a "TileLang" backend but this is **aspirational / not implemented**.

(Inferred) TileLang is a Bytedance/Tsinghua DSL (`tile-ai/tilelang`) for
authoring tiled CUDA kernels at a higher level than CUTLASS. The empty
package directory is consistent with the "Stay Tuned" disclaimer in
Twilight `README.md:25`.

**FlashInfer JIT loading mechanism** (Observed):
1. `ftka/cuda_ops/module.py:36–55` — `get_kernel_module()` is the lazy
   singleton; on first call it either imports a pre-built `_kernels` module
   if `flashinfer.jit.has_prebuilt_ops` is True (it almost never is for
   this repo), or invokes `flashinfer.jit.load_cuda_ops("ftka_ops",
   sources=[FTKA_CSRC_DIR/"ftka_ops.cu"], extra_include_paths=[…])`.
2. `flashinfer.jit.load_cuda_ops` internally calls
   `torch.utils.cpp_extension.load`, which `nvcc`-compiles
   `csrc/src/ftka_ops.cu` against the FlashInfer headers
   (`FLASHINFER_INCLUDE_DIR`, `FLASHINFER_CSRC_DIR`) plus the ftka headers
   (`FTKA_INCLUDE_DIR`). The resulting `.so` is cached under
   `~/.cache/flashinfer/ftka_ops/`.
3. `csrc/src/ftka_ops.cu:212–221` registers `single_sparse_gemv`,
   `batched_sparse_gemv`, `batched_sparse_gemv_int8_k`,
   `batched_sparse_gemv_int4_k`, `quest_sparse_gemv`, `raft_topk` via
   `PYBIND11_MODULE`.
4. Dtype dispatch macro `DISPATCH_PYTORCH_DTYPE_TO_CTYPE` in
   `csrc/src/pytorch_extension_utils.h:88–162` selects c-type by the
   pytorch ScalarType of the first argument. Pre-processor switches
   `FLASHINFER_ENABLE_BF16` and `FLASHINFER_ENABLE_FP8` toggle the matrix.

(Inferred) Cold-start compile is on the order of 30–90 seconds with this
many headers (FlashInfer is template-heavy). Warm cache load is ~tens of
ms. For a production sglang path this would need to either move to AOT
(custom `setup.py` extension) or rely on FlashInfer's existing AOT build.

---

## 6. Relationship to Twilight (the headline, restated)

**Plainly:**
- ftka is **not** a runtime dependency of Twilight pyimpl. The only call
  site in the entire repository tree is `benchmark/efficiency/bench_gemv.py`
  (a microbenchmark of GEMV kernels in isolation, not an end-to-end
  inference path).
- Twilight's accuracy table in `README.md` (LongBench / RULER / Passkey
  numbers) was produced with `twilight/kernel/triton/bgemv_int8.py`
  + `torch.topk` + `twilight/kernel/cuda/sampling.py`'s top-p pruner —
  **not** with ftka.
- ftka is a more performant kernel library — same authors (header credits:
  Chaofan Lin, Bytedance, 2025) — that was **prepared but never wired into
  Twilight's inference path**. This is consistent with Twilight README
  line 25's link "[Flash-TopK-Attention (Stay Tuned)]" pointing nowhere.
- The TileLang backend is an unfilled stub
  (`ftka/tilelang_ops/__init__.py` empty), which further suggests ftka
  was paused mid-development.

(Inferred) ftka was intended as the kernel layer for a future "fused
sparse-attention" path that would replace Twilight pyimpl's
masked-attention-via-bool-tensor with a "select → gather → dense-attend"
pipeline. The pieces are there: `batched_sparse_gemv_int8_k` for the
selector, `raft_topk` for the index extraction, `discontinuous_move_tokens`
for the physical gather, plus FlashInfer's regular paged-attention
decode for the dense-attend step. They just are not glued together in
this snapshot of the repo.

---

## 7. Implications for the sglang DS port (DeepSeek-V3.2 / GLM-5.1 MLA)

Recall MLA's twist: K/V are reconstituted on-the-fly from a low-rank latent
`c_kv ∈ R^{D_low}` (DeepSeek-V3.2: `D_low=512`, `D_qk_nope=128`,
`D_qk_rope=64`, `D_v=128`). The "heavy channel" trick from DoubleSparse
needs a careful reinterpretation: is the heavy axis (a) a subset of the
latent dim `c_kv`, (b) a subset of the reconstituted Q's nope+rope dim, or
(c) a subset of the per-head 128-dim?

| ftka kernel | Reusable for sglang DS-on-MLA? | Obstacles |
|---|---|---|
| **`raft_topk`** | **Yes — highest priority.** Solves the "`torch.topk` forces CPU sync + breaks cuda graph capture" pain in any DS-style path. | (1) `IdxT=int32` — fine for per-row top-k. (2) Output unsorted — fine for boolean-mask or gather consumers, *not* fine if the consumer expects top-k *ranked*. (3) `buf` must be sized at warmup; trivial. (4) JIT compile of FlashInfer headers — could replace with sglang's existing `flashinfer` JIT cache, or vendor only `raft_topk.cuh` + the `raft/` slice (~2000 LoC total) into `sgl-kernel/` as a standalone AOT extension. |
| **`batched_sparse_gemv_int8_k` / `_int4_k`** | **Yes, conditional on choosing a quantized label-cache representation.** Relevant if the porting plan keeps label-cache K in int8 or int4 (e.g. 8-channel × int8 = 8 bytes per token per head — vs. 16 bytes for fp16 r=8). | (1) Currently hard-codes `HEAD_DIM=128` template arg; would need a recompile per head_dim, or a small dispatch. (2) MLA's per-head 128 vs latent 512 mismatch: if labels live in *latent* space the kernel would need `D=64` or `D=128` slices (latent is divisible by neither 32 nor heuristic-favored sizes). (3) Assumes paged-KV layout (`paged_kv_t`); sglang's KV pool is already paged but uses a different `req_to_token` indirection. (4) The kernel computes `q·k_full`, not `q[heavy] · k[heavy]` — without further work it is "approximate attention over all channels of selected tokens", not "exact attention over heavy channels of all tokens" which is what DoubleSparse selector wants. The MLA-aware adaptation is non-trivial. |
| **`quest_sparse_gemv` (MaxPossible)** | **Yes if a chunk-level/page-level estimator is on the table.** Quest's page-MaxMin upper bound is the right primitive for sglang's existing 64-or-128-token page granularity. | (1) Per-page (max,min) maintenance is extra writeback at prefill — sglang's KV cache writers would need a hook to update min/max for each page on insert. (2) MLA's latent-K means min/max would be computed in latent space, not in per-head space — algorithm semantics shift. (3) Kernel hard-codes `HEAD_DIM=128`, see above. |
| **`single_sparse_gemv` / `batched_sparse_gemv` (fp16)** | **Useful but redundant** with FlashInfer's existing batch-decode attention kernel (`BatchDecodeWithPagedKVCachePyTorchWrapper`), which sglang already wires up. Worth using only for label-GEMV where head_dim is small/non-standard. | Same `HEAD_DIM=128` and paged-KV layout assumptions. |
| **`discontinuous_move_tokens`** | **Yes — second priority** for a select-then-gather-then-dense-attend pipeline. The cleanest path for sglang. | (1) Per-token 1-warp launch is bandwidth-bound but under-occupies SMs at small budgets; would want a tiled rewrite for budget < 512. (2) Currently moves `[N,H,D]` tensors; MLA's stored K is `[N, 1, D_low]` (single "head" in latent space), so the kernel applies directly only if you copy in latent space. (3) Cuda-graph compatible (one-shot Triton launch, no host sync, no allocator calls). |
| **TileLang variants** | **No** for the near term. Adding a TileLang dep to sgl-kernel is out of scope unless the user explicitly opts in, and the TileLang path here is an empty stub anyway. | — |

**Cuda-graph compatibility note (Observed/Inferred):** none of the ftka
kernels allocate or synchronize internally; all take pre-allocated outputs
and scratch buffers. They are graph-capturable. The graph-incompatibility
in the Twilight pyimpl path comes from
`mask.scatter_(dim=-1, index=indices, value=True)` with an `indices`
tensor whose size depends on `min(token_budget, S)` (`top_k.py:11`) — that
shape dependency on `S` is the actual cuda-graph hazard, and it is fixed
not by `raft_topk` per se but by a fixed-`k` rewrite.

---

## Appendix A — `csrc/include/raft/*` files (vendored slice)

| File | LoC | Purpose | Source |
|---|---|---|---|
| `raft/macros.h` | 139 | `_RAFT_HOST_DEVICE`, `_RAFT_HAS_CUDA`, `_RAFT_KERNEL`, `HDI` (host-device-inline) | RAPIDS RAFT |
| `raft/integer_utils.h` | 237 | `is_a_power_of_two`, integer log2/div helpers | RAPIDS RAFT |
| `raft/pow2_utils.h` | 175 | `Pow2<N>::roundDown/roundUp` static math | RAPIDS RAFT |
| `raft/vectorized.h` | 385 | `TxN_t<T,N>` vectorized loads/stores via PTX `ld.global.v4` | RAPIDS RAFT |

(Inferred) These four are the **minimal** support headers needed to compile
`raft_topk.cuh` in isolation from a full RAPIDS RAFT install — the
`select_radix.cuh` translation unit's only RAFT dependencies. This makes
`raft_topk` straightforward to lift wholesale into sgl-kernel as a
standalone AOT kernel without taking on a RAFT submodule.

---

## Appendix B — verifying file counts (Observed)

```
csrc/include/dequantize.cuh     202
csrc/include/gemv.cuh           558
csrc/include/lowbit_dtypes.cuh  323
csrc/include/quest_gemv.cuh     311
csrc/include/raft_topk.cuh     1014
csrc/include/raft/*.h           936  (4 files)
csrc/src/ftka_ops.cu            221
csrc/src/pytorch_extension_utils.h  (FlashInfer dispatch macros)
ftka/cuda_ops/{__init__, env, module, gemv, topk}.py   ~250 total
ftka/triton_ops/tokens_moving.py  430
ftka/triton_ops/gemv.py          117
ftka/tilelang_ops/__init__.py      0  (stub)
ftka/utils/benchmark.py          293
tests/test_gemv.py               515
tests/test_topk.py                80
```

Total ftka CUDA C++: ~3.5k LoC; Python wrapper: ~250 LoC; Triton: ~550 LoC;
TileLang: 0 LoC.
