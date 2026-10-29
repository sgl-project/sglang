# Mooncake SWA sender packing

Opt-in sender-side packing for heterogeneous attention TP. The sender gathers
the destination's head slice into persistent registered scratch, then writes
whole pages directly into the **final SWA KV pool**. Existing full-KV staging
and its decode scatter/watermark protocol remain unchanged.

```bash
export SGLANG_MOONCAKE_SWA_PACKING=1                 # default: false
export SGLANG_MOONCAKE_SWA_PACK_BUFFER_SIZE_MB=64   # default: 64 MiB per worker
unset MC_FORCE_TCP
```

Mooncake native code checks whether `MC_FORCE_TCP` exists: setting it to `0`
can still force TCP. Verify actual RDMA installation in both roles.

## Eligibility and lifetime

The first implementation requires P attention TP < D attention TP, an integer
TP ratio, nonreplicated GQA heads, PP/CP/DCP=1, and ordinary contiguous NHD K/V
storage. Both peers must prove matching logical/physical dtype, head dimensions,
page geometry, tensor/layer order and registered capacities. Signed page indices
are checked before gathering. MLA, draft KV, scale sidecars, unified page
envelopes and unsupported state layouts fall back to the existing direct path.

The optional peer proof is appended at wire field 20; missing/malformed proof
preserves legacy transfer. Fallback happens before gathering/submission. After a
native exception or nonzero sync return, scratch, engine and futures are retained
in permanent quarantine and the room's deferred acknowledgement is poisoned.
A completed Python future is not proof that timed-out native writes stopped.
Unsettled GPU gathers also retain scratch/metadata. Teardown deregisters only
known-quiescent buffers. No retry through direct transfer follows submission.

## Validation environment

Results below were recorded against integration `8854b6998e`, based on SGLang
`1776e2dcf8`, on two AMD Instinct MI355X nodes. Recorded gates passed **51 CPU/GPU
checks plus 3 subtests**, including metadata/wire compatibility, failure ownership,
MoRI regressions and actual GPU/native transfer correctness. This historical
gate count applies to that integration snapshot, rather than the new PR head.

The independent PR branch is extracted onto main `acdad27d74`. It retains the
tested Mooncake transfer algorithm, uses the backend-neutral `kv_pack_layout`
field, and adds only the generic setup proof needed by Mooncake. No MoRI backend
changes are included. The tables describe the original tested revision; the
816B model measurements have **not** been repeated on this newer base. The
extraction passed 15 focused CPU tests (the existing 10 cases plus 5 setup/wire
regressions); GPU/native/model validation on the extracted head remains separate.

- Image: `lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261028`.
- Image ID: `sha256:b51edf074b567263afb19be9108756645ee00d0604374b6e4c5006caccaa17fe`.
- Mooncake `engine.so` SHA256: `13c648be36887ae362b4dbd9851268d2075c152be9dd8214cf7d2fb410fbf3ac`.
  Installed distribution version metadata was unavailable; a separate source
  checkout is not evidence of this binary's revision.
- Model: `amd/Muse-Spark-Open-816B-A42B-Quark-MXFP4`, revision
  `c7bc74fd64371315a091bc15f9723882eead0dc1`.
- Experimental ROCm Muse runtime from PR #44667, revision
  `37197b72708c0bc6f89c5c54a1719df22455613c`, with isolated routing-call and
  optional logits-output-buffer compatibility fixes. These results do not imply
  that unmodified upstream Muse serving works with this image.

## Native one-peer microbenchmark

Logical P4 rank 0 → D8 rank 1; 16 global KV heads, head dimension 64, BF16,
page size 16, 1,024 tokens and four SWA layers. Independent random source and
destination pages; 2 MiB useful payload. Both arms use the same thread-pool
concurrency and **8 native batch calls**; logical entries fall from 8,192 to 512.
The baseline rebuilds equivalent NumPy address plans inside timing. Registration
is outside timing. GPU reads, destination bytes and untouched guards matched.

Sender completion time in ms; 8 warmups and 30 samples per arm/block:

| Block | Direct mean / p50 / p95 | Packed mean / p50 / p95 |
|---|---|---|
| 1 | 4.307 / 4.311 / 4.434 | 0.734 / 0.730 / 0.766 |
| 2 | 4.415 / 4.423 / 4.506 | 0.737 / 0.737 / 0.761 |
| 3 | 4.575 / 4.570 / 4.835 | 0.736 / 0.735 / 0.765 |

Input entries are not hardware WQE counts. This fixture is not model serving.

## Real Muse: full staging disabled

P4 → D8, 1,024 input tokens, BF16 KV/page16, concurrency **1**. Graphs, radix
cache and overlap scheduling disabled; chunked prefill disabled. Direct A →
SWA packed → direct B, fresh prefill processes and one shared decode process.
Each OSL1 arm has 16 warmups and 100 measured requests. All samples, including
outliers, are retained. Times are serial direct-HTTP observations in ms.

| Arm | TTFT mean / p50 / p95 | E2E mean / p50 / p95 |
|---|---|---|
| Direct A | 356.971 / 342.751 / 349.779 | 357.123 / 342.887 / 349.912 |
| SWA packed | 181.879 / 181.640 / 185.372 | 182.000 / 181.736 / 185.494 |
| Direct B | 348.079 / 341.803 / 345.365 | 348.227 / 341.993 / 345.518 |

TTFT p50 decreased about 46.9% against the two direct p50 values' average.
Direct-arm means include retained latency outliers; this is not a significance
test or a throughput result.

Additional OSL64 checks used 8 warmups + 10 measured requests per arm:

| Arm | TTFT mean / p50 / p95 | E2E mean / p50 / p95 | Observed mean ITL: mean / p50 / p95 |
|---|---|---|---|
| Direct A | 340.772 / 341.130 / 344.246 | 4501.032 / 4472.139 / 4644.162 | 66.034 / 65.565 / 68.266 |
| SWA packed | 180.778 / 180.908 / 182.363 | 4493.121 / 4491.240 / 4515.162 | 68.448 / 68.401 / 68.790 |

Earlier first-token arrival did **not** establish faster total generation or
ITL. These 10-request arms do not establish tail probabilities.

## Real Muse: existing full staging enabled

All arms set `SGLANG_DISAGG_STAGING_BUFFER=1` and
`--chunked-prefill-size 1024`; the 1,024-token prompt fits one forward and one
staging grid slot. Do not compare this with the upstream `-1` staging-grid
behavior, which produces one-page slots. Other settings match above; OSL2,
8 warmups + 30 measured requests per arm, direct A → packed → direct B,
with the same decode process throughout.

| Arm | TTFT mean / p50 / p95 | E2E mean / p50 / p95 |
|---|---|---|
| Direct A | 286.882 / 287.429 / 289.765 | 385.824 / 389.635 / 394.622 |
| SWA packed | 129.459 / 129.411 / 130.689 | 224.547 / 224.997 / 227.786 |
| Direct B | 287.379 / 289.136 / 292.027 | 384.063 / 389.849 / 397.773 |

Against the two direct p50 values' average, TTFT decreased **55.1%** and E2E
decreased **42.3%**. Separate diagnostic requests proved successful full staging
on every P rank, completed scatter on every D rank, and SWA packing hits.
Diagnostic hooks were removed; both roles restarted and source hashes verified
identical before formal timing. No diagnostic instrumentation ran during timing.

Across the two configurations, **498 requests** matched the independent greedy
reference: 348 OSL1 + 36 OSL64 + 114 OSL2, including warmups. Each had exactly
1,024 prompt tokens; decode produced 2,880 output tokens in total. The two
path-proof diagnostic requests are excluded from these counts. Both roles'
native RDMA installation was verified and `MC_FORCE_TCP` was absent.

These are low-concurrency, text-only results for one model/workload and an
experimental runtime. They establish neither broad concurrency scaling nor
decode-compute acceleration. Raw artifacts, exact commands, source hashes and
per-request output IDs are retained separately; private node addresses and raw
logs are intentionally not included here.
