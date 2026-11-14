# uTLX attention on gfx1250

This experimental backend uses the uTLX extension to Triton for explicit TDM
loads and shared-memory synchronization. It requires an AMD gfx1250 GPU and a
compatible `utlx_plugin` build with `tdm_async_load` and `workgroup_barrier`.
Importing the SGLang kernel registry does not import the optional plugin.

## Interface

`paged.unified_attention` computes causal or noncausal D64/D128 attention with
GQA ratios 1/2/4/8/16, ragged queries, page sizes 16/64/128, and a left window.
Q and KV may use FP8 e4m3fn with scalar FP32 descales; output is FP16/BF16.
It reads page tables and sequence lengths on the device during graph replay.

The kernel packs query heads sharing one KV head, overlaps page loads with
attention computation, and splits underfilled launches across KV segments.
Split counts scale with the device's CU count. The caller owns valid page
tables, monotonic query offsets and cache writes. Long D64 queries with GQA
ratio 8 load several pages per compute tile through indexed views of the same
shared allocation; this requires `local_reinterpret` support.

## SGLang integration

Select `--attention-backend tlx` to run the AITER backend's unified-attention
calls (extend, decode, target verification and draft extend) on the paged
implementation. The adapter inherits AITER's cache writes, metadata
construction and graph lifecycle. Calls outside the kernel's contract, such as
attention softcapping, delegate to AITER with a one-time warning.

`SGLANG_TLX_PHASES` selects the attention phases that run on uTLX; AITER runs
the rest. The phases are `decode` (decode and speculative verification),
`prefill_window` (sliding-window prefill), `prefill_short` and `prefill_long`
(full-attention prefill below and from 1024 query tokens). The default,
`decode,prefill_window,prefill_short`, keeps long full-attention prefill on
AITER, whose kernel is faster there on gfx1250.

The backend needs `--page-size` 16, 64 or 128 (other values default to 16) and
the AITER runtime. It does not support MLA, the vectorized 5D AITER KV layout
or deterministic inference. It is not a draft backend: speculative decoding
needs an explicit `--speculative-draft-attention-backend`, such as `triton`
for DFlash2. A model monokernel that bypasses the attention backend also
bypasses this integration.

Muse Spark MXFP4 with DFlash2 at TP8 on MI450 matches AITER's GSM8K accuracy
with this backend. Kernel latencies vary by shape; the default phases route
each one to the faster kernel measured at Muse Spark's TP8 shapes.

## Validation

With SGLang and the compatible uTLX extension installed, run on an idle gfx1250
GPU:

```bash
python -m pytest -q \
  test/manual/kernels/attention/test_tlx_paged_attention.py \
  test/manual/kernels/attention/test_tlx_backend.py
```

The tests use independent FP64 references, nonunit quantization scales,
shuffled page tables, cache-tail canaries, graph replays and the real SGLang
forward methods.

These are manual tests because the repository does not currently provide a
gfx1250 GPU runner. They are not claimed as automated GPU CI coverage.
