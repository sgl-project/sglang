# Critical review — loop4 DS scaffolding + pre-cutover loop5 fixes

Scope: the Double Sparsity runtime built in loop4 (never executed on hardware) and
the loop5 fixes made before the model cutover. Driven by hardware testing on 8× H200
this session, which exercised these paths for the first time.

## Overarching finding
The loop4 DS code was written and unit-tested against **synthetic decode-shaped
fixtures** and **never ran end-to-end on hardware**. As a result it carries a family
of latent defects that only appear on a real boot/serve: stale identifier renames,
attributes bound on only one of the two attention layers, reshape widths taken from the
wrong layer, metadata read from `forward_batch` instead of `ForwardContext`, and —
most fundamentally — a selection batch built for the **decode** shape (one query per
request) that breaks on **prefill** (many query tokens per request). Each fix below was
needed before the next defect could even surface.

## Defects found this session

### FIXED + validated
1. **validator.py: stale `is_deepseek_nsa`** (renamed → `is_deepseek_dsa`). ImportError
   at startup. Commit 34b243b07. +regression.
2. **deepseek_v2.py: `self.use_nsa`** in the DS-enablement branch (renamed → `use_dsa`).
   AttributeError at model construction. Commit 34b243b07. +regression.
3. **serve_double_sparsity.sh: no `--mem-fraction-static`** headroom for the DS
   TokenLabelTable → CUDA OOM at boot (0.897) / generation (0.7). Default 0.6. Commit 34b243b07.
4. **deepseek_v2.py: `_ds_channel_selection` bound on CPU** while the KV-write hook
   gathers GPU-resident K_nope → device-mismatch crash. Move to label-table device.
   Commit 34b243b07.
5. **AC-0: `_write_token_labels` referenced `forward_batch` without accepting it** (Round-38
   producer bug). Threaded from all 4 call sites. Commit 4f4c620df. +regressions.
6. **calibrate.py: HF cannot load `deepseek_v32`** + bf16 single-device load can't fit.
   deepseek_v3 config remap + native-FP8 device-sharded load + fail-closed dry-run +
   Triton FP8 fallback. Commits 7cbbce088, c99ed3644. Mask generated + validated (AC-4).
7. **DS decode degeneration (#16)** — two bugs:
   (a) `req_to_token` None on decode → selector used physical-domain + skipped
       `logical_to_physical` (ds_out all -1). Resolve via ForwardContext. Commit 2af5f4e65.
   (b) decode tokens never label-written: `kv_b_proj` only on `attn_mha` (not `attn_mqa`),
       and `head_width` derived from `layer.v_head_dim` (128 vs absorbed 512). Attach to
       `attn_mqa` + derive `head_width` from the projection output. Commit 8375b76a5.
   Validated: coherent decode, `selected_tokens` grows with seq, dense_fallback=0.

### OPEN — the central architectural defect (#18, task #18)
8. **DS selection uses the DECODE batch shape; breaks on PREFILL.**
   `forward_mla.py:268` makes `q_nope_for_ds` shape `[num_tokens, H, dim]` (one query
   per token), but `_select_topk_indices` + the selection kernels take per-REQUEST
   `req_pool_indices`/`seq_lens` and derive the batch from the query rows. For prefill
   (num_tokens ≫ num_reqs) it indexes `req_pool_indices` past its end → garbage pool
   indices ("N bad req_pool_indices in batch") → out-of-range physical slots → CUDA
   illegal access for any prompt with seq_len > top_k=2048. Decode works only because
   num_tokens == num_reqs there. Blocks AC-1.1 + all real-shape runs. Fix: per-extend-token
   prefill selection (expand req_pool_indices/seq_lens per token; each prefill token
   attends to its own prefix), matching DSA's indexer prefill handling. See
   `sparse_path_oob_finding.md`.

### OPEN — secondary
9. **DS `per_request_summary` not surfaced under CUDA graph (#17, task #17).**
   `_publish_ds_request_summary` is skipped during capture/replay (host syncs illegal),
   so `double_sparsity`/`radix_capture` meta only appears in eager mode. AC-0 hardware
   capture probe needs a capture-safe publish path or an eager probe.

## Areas to re-review next (same decode-shape suspicion)
- **selection_kernel.py / cuda_graph.py**: every buffer/grid that uses `bs` — confirm it
  is consistently num_tokens vs num_reqs across prefill and decode (the #18 root). The
  graph-safe scratch (`scratch_scores` etc.) sizing vs the prefill query count.
- **page_table_adapter.logical_to_physical**: reviewed — internally clamps/masks
  correctly; the bug is the upstream batch/req_pool_indices it is handed (#18).
- **_select_topk_indices MHA bypass**: currently only bypasses for `use_mha` (short
  prefill). The fact that long prefill (absorb) runs selection at all is what exposes #18.
- **token_label_write / radix capture under chunked prefill**: not yet exercised with
  chunked prefill (AC-1b regime).

## Process note
The loop4 unit tests pass (253) but mock the decode path with `SimpleNamespace`/
`object.__new__` fixtures that mirror the decode batch shape — so they never caught the
prefill batch mismatch. Future DS tests should include a prefill-shaped batch
(num_tokens > num_reqs) to lock the #18 fix and prevent regression.

## Loops 1-3 (foundational DS modules) — review

Reviewed: `channel_mask.py`, `token_label_table.py`, `selector.py`, `config.py`, the
calibrate channel-importance core, and the MLA noPE extraction in `dsa_backend.py`.

### Verdict: foundations are sound and NOW hardware-validated
Unlike the loop4 integration code, the loops 1-3 data structures held up — and this
session's working end-to-end DS run exercised them for the first time on hardware:
- **channel_mask.py**: `load_channel_mask` validation is robust (3-D shape, label_dim
  match, channel indices in `[0, head_dim)`, content-SHA recompute) — validated on the
  real generated mask. TP head-sharding `slice_per_rank` (contiguous
  `[rank*num_local_heads : +num_local_heads]`, with `h_full == num_local_heads*tp_size`
  check) — exercised correctly at the TP=8 boot (num_local_heads=16 = 128/8).
- **token_label_table.py**: correctly sized by the KV-slot address space
  (`max_tokens = pool.size + page_size`), with an explicit note on why the
  req-row count would under-size and OOB — allocated + written + read this session.
- **config.py**: narrow, strict validation (positive top_k/page_size; rejects the
  legacy top-p ABI) — exercised via the launcher JSON.
- **selector.py / selection core**: the decode-path selection produced correct genuine
  sparsity (0<rate<1) once the loop4 integration bugs were fixed.
- **MLA noPE extraction** (BL-20260527-reshape-before-slice-mla, round 3): the
  reshape-before-slice fix is present and correct; the round-2 `head_width` fix
  (derive from projection output, not `layer.v_head_dim`) hardened it further.

### Caveats / not-yet-exercised in loops 1-3 foundations
- The **FP8 scale-stability fixture** (AC-10 radix gate) and the **radix label-capture
  fixture** are not yet run on hardware (M3-B; needed for AC-10).
- **Multi-request batched decode** (bs>1) selection not yet stress-tested on hardware
  (smoke used bs=1); the per-request batch path should be exercised before AC-11.
- **Chunked prefill** (AC-1b) interaction with the now-dense DS prefill not yet tested.

### Net
The "never ran on hardware" defects were almost entirely in the loop4 runtime
integration (selection batch shape, attention-layer attribute binding, NSA→DSA rename
stragglers, ForwardContext metadata) and the loop5 calibrate load path — NOT in the
loops 1-3 foundational data structures, which are correct and now validated end-to-end.
