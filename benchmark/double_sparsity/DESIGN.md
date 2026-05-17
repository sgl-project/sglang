# Double Sparsity Native Decode — Design + Status

K-channel Double Sparsity sparse-decode for long-context LLM inference in
SGLang. Bypasses the FA3 page-table sparse adaptor with a self-contained
Triton pipeline; both perf and quality gates pass simultaneously on
Llama-3.1-70B at 128K.

---

## 1. Status

### Headline (passes both gates simultaneously)

70B / Llama-3.1-Instruct / TP=8 on 8×H200, ctx=131072, output_len=512,
n_requests=32, max_running_requests=32:

| operating point | TBT(off) | TBT(on) | TBT ratio | NIAH(off, n=10) | NIAH(on, n=10) | NIAH delta |
|---|---:|---:|---:|---:|---:|---:|
| **conc=16 / tb=2048 / retrieval calib / torch selector**  | **27.94 ms** | **22.45 ms** | **0.8035× PASS** | **0.80** | **1.00** | **+0.20 PASS** |
| conc=32 / tb=8192 / wikitext calib  / torch selector  | 34.68 ms | 30.52 ms | 0.8800× PASS | 0.80 | 1.00 | +0.20 PASS |

Gate definitions: **perf** = `TBT(on) ≤ 0.90 × TBT(off)`; **quality** =
`NIAH(on) ≥ NIAH(off) − 0.02`.

Bench JSONs in `repro_session/conc16_move_left/` and
`repro_session/sweep_70b_128k_tbt_win/`.

### What landed

* `python/sglang/srt/layers/attention/triton_ops/double_sparsity_native_decode.py`
  — score → topk → build_selected_physical → sparse-attn (stage2+stage3) Triton pipeline.
* `python/sglang/srt/mem_cache/sparsity/algorithms/double_sparsity.py`
  — `try_native_sparse_decode` algorithm entry, native scratch lifecycle,
  per-step `req_to_token` caching via `forward_begin`.
* `python/sglang/srt/mem_cache/sparsity/algorithms/selector_backends.py`
  — pluggable top-k selectors (`torch` default, `flashinfer_topk_page_table`,
  `sgl_fast_topk_transform` and `jit_fused_selector` reserved).
* `python/sglang/srt/layers/radix_attention.py` — DS dispatch tries native
  path first, falls through to legacy FA3 adaptor.
* `python/sglang/srt/mem_cache/sparsity/core/sparse_coordinator.py` —
  `forward_begin(forward_batch)` delegates to algorithm hook for per-step
  state setup outside the CUDA-graph captured region.
* `python/sglang/srt/model_executor/model_runner.py` +
  `cuda_graph_runner.py` — invoke `forward_begin` from both the eager and
  graph-capture paths so the captured graph reads from an already-populated
  scratch.
* `scripts/double_sparsity/make_retrieval_calib_prompts.py` — synthetic
  NIAH-shaped calibration prompt generator (the unlock for low-`tb` NIAH).
* `benchmark/double_sparsity/bench_decode.py` — adds `--selector-backend`,
  records `selector_backend_tag` in result JSON.
* Tests: 130 in `test/registered/unit/mem_cache/sparsity/` covering the
  native pipeline, the per-step caching hook, selector parity (torch vs
  FlashInfer), runtime-config validation for the selector ceiling.

### What does NOT yet work

* **`selector_backend=flashinfer_topk_page_table` under CUDA-graph
  capture.** Microbench parity passes and shows 1.10–1.30× speedup at
  bs ≥ 16, but the FlashInfer Triton kernel crashes with
  `Triton Error [CUDA]: illegal memory access` in `load_binary` on the
  first captured forward call, even after a per-bs warmup sweep
  completes cleanly outside capture. Documented in
  `selector_backends.py:FLASHINFER_TOPK_MAX` block comment. Use
  `selector_backend=torch` for production.
* **`selector_backend=sgl_fast_topk_transform`** — `sgl_kernel.fast_topk_transform_fused`
  in the installed build doesn't expose `row_to_batch`, so the per-h_kv
  broadcast contract doesn't fit. `make_selector` raises NotImplementedError
  at registry resolution.
* **`selector_backend=jit_fused_selector`** — reserved placeholder.

---

## 2. Design — from first principles

### 2.1 The decode bottleneck at long context

Autoregressive decode generates one token per forward pass. For each
new token, attention reads the full KV cache `[seq_len, num_kv_heads,
head_dim]` per layer. With GQA, `num_q_heads / num_kv_heads` query heads
share each KV head's data, so the dominant cost is the KV bandwidth
read, not the GEMM:

```
70B / TP=8 / 128K ctx, bf16:
  KV cache per rank per layer = 1 KV head × 128 dim × 2 (K+V) × 2 B × 131072 tok
                              = 64 MiB / layer
  × 80 layers                 = 5 GB per decode step per rank
  H200 HBM peak               ≈ 4.8 TB/s
  KV-read floor               ≈ 1.0 ms per decode step
```

Empirically dense TBT at this point is 27.94 ms at conc=16 → KV bandwidth
is the dominant share at the batch size where the queue actually feeds
the GPUs. Reducing the KV read by reading only a *subset* of past tokens
per step is the entire optimization.

### 2.2 Why most tokens don't matter (the attention-pattern observation)

Empirically, attention weights in trained transformers are sparse: a
small subset of past tokens accounts for most of the softmax mass. The
**top-k attention approximation** is to keep only those tokens. The
trade is exact (full-KV) correctness for an O(k/seq_len) bandwidth
reduction.

The catch: identifying which tokens matter normally requires computing
the full attention scores, which costs as much as just doing the
attention. Double Sparsity's contribution is making this identification
itself cheap.

### 2.3 The K-channel observation (Double Sparsity)

For Llama-style attention, `score[t] = q · K[t] / sqrt(d)`. The
high-attention positions are determined by the dot product. The
Double Sparsity insight: across calibration inputs, only a small fixed
set of K-channels actually drives the dot product. The other channels
are noise.

Concretely: pick `S = 32` heavy channels per (layer, KV head) via
offline calibration. At decode time, project `q` and `K[t]` onto only
those S channels — call them `q_label` and `K_label[t]` — and use the
`q_label · K_label[t]` dot product as an *approximate* score. This
approximate score is `head_dim / S = 4×` cheaper than the full dot
product and is empirically accurate enough to identify the right
top-k positions for `tb ≪ seq_len`.

Two facts about the K_label side cache:

* It's a side cache of shape `[num_tokens_in_pool, num_kv_heads_local, S]`,
  ~12% the size of the full KV cache at `S=32, head_dim=128`. We *only*
  read K_label during scoring; we read full-`head_dim` K/V only at the
  top-k positions during sparse attention.
* It's written on every K projection (extend + decode). The current
  decode position's K_label is written *after* attention so scoring
  covers history `[0, seq_len-1)` only — the current position is
  unconditionally retained via the recency window.

### 2.4 GQA broadcast: one selected set per (request, KV head)

With GQA, multiple Q heads share each KV head's data. Computing a
separate top-k per Q head wastes work: the KV read can be amortized
across the group. Double Sparsity scores once per `(bs, kv_head)` and
shares the selected positions across all Q heads in the group. The
Q heads' contributions to the score are merged via a GQA reduction
(`max_abs`, `mean`, or `soq`); `max_abs` is the default and the choice
that matches the original DoubleSparse paper.

### 2.5 Sink + recent windows: structural always-keep

Two structural priors are well-documented in attention research:

* **Attention sinks** (Xiao et al, "Efficient Streaming LLM with
  Attention Sinks") — the first few tokens in every sequence absorb
  excess softmax mass even when they're semantically uninformative.
  Dropping them tanks quality. DS reserves the first `sink_tokens` (=4)
  positions unconditionally.
* **Recency** — the most recent few tokens drive next-token prediction
  directly. The current decode token's K_label is not yet written when
  selection runs, so we explicitly include the last `recent_tokens` (=64)
  history positions as always-keep. `recent_tokens >= 1` ensures the
  current decode token is always in the selected set.

The selected set per `(bs, kv_head)` is therefore
`[top-k physical | sink physical | recent physical]`, total length
`total_selected = top_k + sink_tokens + recent_tokens`.

### 2.6 Calibration shape determines what K_label captures

The heavy-channel choice depends entirely on what the calibration data
looked like. Wikitext (next-token prediction over generic prose) shapes
K_label channels to flag tokens that drive language-model perplexity —
which is not the same as flagging tokens a retrieval question would
need. At narrow token budgets (`tb ≤ 4096`), wikitext-calibrated
K_label cannot reliably surface needle tokens in NIAH-style prompts:

| `tb`  | NIAH (wikitext calib) | NIAH (retrieval calib) |
|---:|---:|---:|
| 512  | 0.00 (n=5)  | (not measured) |
| 2048 | 0.40 (n=10) | **1.00 (n=10)** |
| 8192 | 0.90 (n=10) | (not measured; wikitext already passes) |

Same kernel shape, same hyperparameters, same operating point — only
the K-channel selection differs. The retrieval-shaped corpus
(`scripts/double_sparsity/make_retrieval_calib_prompts.py`) generates
128 prompts each ~20K characters with a `Note: <key> is <value>` needle
embedded at random 30–70% positions, followed by a `Question: What is
<key>?`. Calibrating on this corpus shifts 19% of K-channel picks vs
wikitext at `heavy_channels=32`; the shifted channels are the ones that
activate on retrieval-pattern tokens.

The decision: **production calibration must be retrieval-shaped when
targeting `tb ≤ 4096`**. Wikitext is fine for the conc=32 / tb=8192
operating point. See `.pensieve/short-term/decisions/2026-05-14-ds-v2-retrieval-shaped-calibration-required-for-low-tb.md`.

---

## 3. System design

### 3.1 Where DS sits in SGLang

```
ModelRunner.forward(forward_batch)
├─ HiSparse coordinator: wait_for_pending_backup + num_real_reqs.fill_(bs)
├─ DS  coordinator     : forward_begin(forward_batch)   ← writes _native_req_to_token_indexed
└─ {can_run_graph}
    ├─ True : graph_runner.replay(forward_batch)
    │           [captured graph reads scratch via stable device pointers]
    └─ False: forward_decode(forward_batch) → model.forward(...)
                  [eager dispatch; same scratch read]

Each transformer layer's attention:

  RadixAttention.forward(q, k, v, forward_batch)
  ├─ DS off → _forward_inner (FA3 backend, unchanged)
  └─ DS on
      ├─ decode mode:
      │     DoubleSparsityAlgorithm.try_native_sparse_decode(...)
      │       → writes K/V to KV pool, returns [bs, H_q*D] attn output
      │       → coordinator.attention_end → K_label write for new decode token
      └─ extend mode (or fallback):
            coordinator.attention_begin (FA3 page-table rewrite)
            → _forward_inner (FA3 sparse)
            → coordinator.attention_end (K_label write for extend tokens)
```

DS-off is byte-for-byte unchanged from the dense codepath. The DS branch
is the static `self.ds_enabled` attribute on each `RadixAttention`
module; the branch is Python-static so the captured graph traces only
one of the two paths.

### 3.2 The native sparse-decode pipeline

`triton_ops/double_sparsity_native_decode.py:ds_native_sparse_decode`
runs 4 kernels per layer per step. All operate on preallocated scratch
sized to the worst-case batch (`scratch_max_bs`) and `.narrow(0, 0, bs)`'d
to the current batch.

**(1) Score** — `_ds_native_score_kernel`. Per `(bs, kv_head, BLOCK_T)`
program. Loads `q_label[bs, kv_head, :S]` once into registers; iterates
`t` over `BLOCK_T` positions; gathers `K_label[req_to_token[bs, t], kv_head, :S]`
and computes the bf16 dot product. Masks `t < sink_tokens`,
`t ∈ [seq_len - recent, seq_len)`, and `t >= seq_len - 1` (history-only
invariant) to `-inf` so `torch.topk` cannot select them. Writes
`att_out[bs, kv_head, t]` (shape `[bs, h_kv, max_ctx]`).

**(2) Top-k** — Selector backend. The default `TorchTopKSelector`
calls `torch.topk(att_out, k=top_k, dim=-1, sorted=False)` (decomposes
into 8 `at::native::mbtopk::*` + `scan_by_key::DeviceScanByKey*` CUB
kernels in the captured graph; total ~0.64 s of stream-time at conc=32
in the nsys trace). The `FlashInferTopKPageTableSelector` collapses
this to one launch via `flashinfer.top_k_page_table_transform` —
microbench shows 1.10–1.30× speedup at bs ≥ 16 but doesn't yet survive
graph capture.

**(3) Build selected_physical** — `_ds_native_build_selected_physical_kernel`
(torch selector path) or `_ds_native_append_sink_recent_kernel`
(FlashInfer path, which writes top-k physical directly). One program per
`(bs, kv_head)`. Writes:

  ```
  selected_physical[bs, kv_head, 0 : top_k)                   = top-k physical (mapped via req_to_token)
  selected_physical[bs, kv_head, top_k : top_k + sink)        = sink physical (positions 0..sink_tokens)
  selected_physical[bs, kv_head, top_k + sink : total)        = recent physical (seq_lens - recent .. seq_lens)
  ```

**(4) Sparse attention** — `_ds_native_sparse_attn_stage2_kernel` +
`_ds_native_sparse_attn_stage3_kernel`. Split-K decoder seeded from
PR #22992. Stage 2 per `(bs, q_head, block_seq)` program reads
`selected_physical` directly (no logical→physical round-trip inside the
kernel), gathers `K_buffer[phys] / V_buffer[phys]` at full `head_dim`,
runs online softmax over one tile, writes partial result. Stage 3 reduces
across the split-K blocks.

**Cost model**: selection cost is bounded by `top_k`, not `seq_len`;
sparse-attn cost is bounded by `total_selected`, not `seq_len`. Both are
the headline DS properties. Microbench at conc=16, max_ctx=131072 shows
the sparse-attn time is flat across `seq_len ∈ {32K, 64K, 128K}` at
fixed `total_selected` (≤2% jitter per row).

### 3.3 Per-step `req_to_token` caching via `forward_begin`

`req_to_token[req_pool_indices]` is the same gather across all 80
decoder layers in one forward pass. Doing it per-layer wastes 79
`index_select` launches. The pattern:

1. `DoubleSparsityAlgorithm._allocate_native_scratch` preallocates
   `_native_req_to_token_indexed[scratch_max_bs, max_ctx]` int32.
2. `SparseCoordinator.forward_begin` delegates to
   `algorithm.forward_begin(forward_batch)`.
3. `algorithm.forward_begin` does the gather once per decode step:
   `torch.index_select(req_to_token, 0, req_pool_indices, out=scratch.narrow(0, 0, bs))`.
4. Each layer's `try_native_sparse_decode` reads from the scratch via
   `.narrow(0, 0, bs)` — no `index_select`.
5. `ModelRunner.forward` invokes `ds_coordinator.forward_begin(forward_batch)`
   before the `can_run_graph` branch — so the scratch is fresh before
   either eager dispatch or graph replay.
6. `CUDAGraphRunner.capture_one_batch_size` invokes the same hook
   before `run_once`, so the captured graph traces against an
   already-populated scratch.

This is the same pattern HiSparse's `num_real_reqs.fill_(bs)` uses:
**a host-side write to a stable device tensor outside the captured
region, read inside the captured graph through a captured pointer**.
Microbench shows ~0.4–0.9 ms saved per decode step at bs=1–32.

### 3.4 CUDA-graph capture safety

The native pipeline is captured into SGLang's CUDA graph. Three rules
make this work and one rule is violated by the FlashInfer backend:

* **No host syncs in eligibility gates.** Anything like
  `if seq_lens.min().item() < N: ...` freezes the captured graph to
  one branch at capture time. The native dispatch is unconditional
  (Python-static); `bs > scratch_max_bs` returns None *outside* capture
  before the layer fires.
* **No allocation inside the captured region.** All scratch is
  preallocated at `_allocate_native_scratch` time (init), the FlashInfer
  selector's aux tensors at construction. Per-step refresh uses
  `out=`-arg writes into preallocated buffers.
* **No JIT compile inside capture.** Triton kernels are pre-warmed by
  running once at every captured bs *before* `capture()` is called. The
  FlashInfer Triton kernel currently violates this transitively: its
  internal `load_binary` re-invokes from inside capture even after
  warmup completes outside, because the kernel handle is bound to a
  different stream/context.
* **Stable device pointers are read-OK.** Captured graphs reference the
  preallocated tensor by pointer; mutating the contents from outside
  capture is the entire `forward_begin` pattern.

See `.pensieve/short-term/maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates.md`
for the full enumeration.

### 3.5 Scratch buffer layout

Worst case at 70B / TP=8 / conc=32 / 128K / tb=8192:

```
_native_req_to_token_indexed : [32, 131072]            int32   16 MiB
_native_att_out              : [32, 1, 131072]          fp32   16 MiB
_native_selected_physical    : [32, 1, 8260]            int32  ~1 MiB
_native_mid_out              : [32, 8, ceil(8260/128)*128, 128] fp32 ~13 MiB
_native_mid_o_logexpsum      : [32, 8, ceil(8260/128)]   fp32  ~7 KiB
_native_output               : [32, 8, 128]              bf16  64 KiB
K_label[layer]               : [pool_tokens, 1, 32]      bf16  (12% of KV pool)
```

Total native scratch ≈ 46 MiB per rank, plus K_label which sits next to
the KV pool. All buffers are static-shape across decode steps; replay
needs no allocator activity for the captured graph.

### 3.6 Selector backend interface

```python
class _BaseSelector:
    name: str
    def select(
        self,
        *,
        att_out_approx: torch.Tensor,         # [bs, h_kv, max_ctx] fp32
        req_to_token_indexed: torch.Tensor,   # [bs, max_ctx] int32
        seq_lens: torch.Tensor,               # [bs] int64
        top_k: int,
        sink_tokens: int,
        recent_tokens: int,
        out: torch.Tensor,                    # [bs, h_kv, total] int32, written in-place
    ) -> None: ...
```

All backends write the same layout into `out`. The torch backend uses
`torch.topk` to get logical positions then a fused build kernel that
gathers physical positions and writes sink + recent. The FlashInfer
backend uses `flashinfer.top_k_page_table_transform` (fused topk +
physical lookup) then a smaller sink+recent kernel. New backends only
need to satisfy the `out`-layout contract — sparse attention reads
`out` directly.

`make_selector(backend, *, max_bs, h_kv, device)` resolves the backend
name. Unknown names raise `ValueError`. `sgl_fast_topk_transform` and
`jit_fused_selector` raise `NotImplementedError` at registry time
(not at construction or first call) so config-time validation catches
mistyped names. The FlashInfer backend's `top_k <= 2048` ceiling is
also validated at config time.

---

## 4. Production recipe

```bash
# 1. Generate retrieval-shaped calibration (one-time, ~1 min on 8×H200):
python3 scripts/double_sparsity/make_retrieval_calib_prompts.py \
    --output /workspace/ds_retrieval_calib_prompts.txt \
    --n-prompts 128 --target-chars 20000 --seed 0

python3 scripts/double_sparsity/calibrate.py \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --output /workspace/calib_llama_3_1_70b_retrieval_s32.json \
    --heavy-channels 32 --n-samples 64 --seq-len 4096 \
    --prompts-file /workspace/ds_retrieval_calib_prompts.txt \
    --device-map auto

# 2. Headline operating point (both gates pass at conc=16):
SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
PYTHONPATH=python python3 benchmark/double_sparsity/bench_decode.py \
  --config branch_ds_on \
  --calibration /workspace/calib_llama_3_1_70b_retrieval_s32.json \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --context-len 131072 --output-len 512 \
  --n-requests 32 --concurrency 16 \
  --tp-size 8 --mem-fraction-static 0.85 --max-running-requests 32 \
  --token-budget 2048 --recent-tokens 64 --sink-tokens 4 \
  --min-seq-len 4096 --max-selected-per-request 8192 \
  --block-t 1024 --k-block 64 \
  --selector-backend torch \
  --niah --niah-n-samples 10 \
  --output-json /tmp/branch_ds_on.json

# 3. Compare against an off baseline at the same conc:
python3 benchmark/double_sparsity/compare.py \
  --branch-off /path/to/off.json --branch-on /tmp/branch_ds_on.json
```

For the conc=32 / tb=8192 headline (wikitext calibration is fine here):
swap `--token-budget 8192 --max-selected-per-request 16384 --concurrency 32`
and `--calibration /workspace/calib_llama_3_1_70b_wikitext_s32.json`.

---

## 5. Next steps

### Near-term

1. **FlashInfer selector under graph capture.** Microbench parity passes
   and shows real per-step savings, but the kernel currently crashes
   in `load_binary` from inside the captured region. Two paths: file an
   upstream issue with a minimal reproducer + the warmup ladder we use,
   or swap to a different fused-topk + page-table kernel.
2. **`sgl_fast_topk_transform` adaptor.** Either land an upstream
   `row_to_batch` argument on `fast_topk_transform_fused` or write a
   thin score-row-duplication adaptor on the algorithm side that
   reshapes the per-bs page table into a per-h_kv-row page table. With
   `top_k=2048` this is the configuration `fast_topk_transform_fused`
   is optimized for.
3. **`ds_native_sparse_decode` API cleanup.** The orchestrator takes 19
   keyword params; pack into `NativeScratch` + `NativeKernelConfig`
   dataclasses. Pure refactor, no behavior change.

### Medium-term

4. **Remove the legacy FA3-adaptor path.** Native is canonical at the
   production operating points. The legacy `retrieve_topk` +
   `DSFlashAttentionAdaptor` + stage-1/stage-2/union kernels remain as
   fallback for `bs > scratch_max_bs`, but that case is never hit in
   production. Removing ~2000 lines of legacy code is a separate refactor PR.
5. **Per-request fallback for short prompts.** The current native
   dispatch is unconditional once DS is enabled. A real server admitting
   short prompts (`seq_len < total_selected`) would compute against
   garbage K/V (no crash; the score kernel masks the out-of-history
   tail). Add a Python-static eligibility check via `min_seq_len` at
   admission time, or a per-request metadata signal.
6. **MLA / DeepSeek-style attention.** Current DS targets Llama-style
   GQA. MLA's compressed KV needs a different K_label projection.
7. **Inline `q_label` into the score kernel.** The current path
   pre-computes `q_label = q · channel_indices` in a separate torch op.
   An inlined version was tried (stashed as
   `uncommitted-inline-qlabel-change`) and showed neutral-to-slight
   regression on synthetic; worth a careful re-test now that the
   surrounding code has changed.

### Long-term

8. **`jit_fused_selector`** — single Triton kernel that fuses
   score + topk + physical translation + sink/recent. The selector
   abstraction is in place for this; the registry slot is reserved.
9. **Higher concurrency.** The DS-vs-dense ratio widens with batch
   (dense KV bandwidth scales with `bs × seq_len`; DS stays flat in
   `total_selected`). Conc=64 / 128 untested but the Pareto extrapolates.
10. **Quality probes beyond NIAH.** LongBench, RULER, or end-task
    metrics on real workloads. NIAH is a useful gate, not a quality
    floor.

---

## 6. References

* Andy Yang et al., "DoubleSparse" — https://github.com/andy-yang-1/DoubleSparse
* Xiao et al., "Efficient Streaming LLM with Attention Sinks"
* Leviathan et al., "Fast Inference from Transformers via Speculative Decoding"
* SGLang PR #22992 — prior v1 sparse-decode kernels seeded this work
* `.pensieve/short-term/decisions/2026-05-14-ds-v2-replace-fa3-adaptor-with-native-sparse-decode.md`
  — why the native pivot
* `.pensieve/short-term/decisions/2026-05-14-ds-v2-retrieval-shaped-calibration-required-for-low-tb.md`
  — why retrieval-shaped calibration is required for low-tb NIAH
* `.pensieve/short-term/knowledge/ds-flashinfer-top-k-page-table-boundaries/content.md`
  — FlashInfer kernel boundaries
* `.pensieve/short-term/maxims/cuda-graph-capture-rejects-host-syncs-in-eligibility-gates.md`
  — capture-safety rules
