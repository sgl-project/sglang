# perf(hisparse): add an MTP Demand Buffer

## Summary

This PR adds an opt-in, request-local HBM Demand Buffer for HiSparse target
verification. FlashMLA consumes logical HiSparse TopK indices directly, probes
the MTP buffer, and reads the authoritative pinned-Host KV row only on a miss.
It is a performance optimization for the MTP lifecycle introduced by #26445:
speculative-decoding semantics and accepted-token ownership remain in the
parent, while this child removes correctness-first multi-step staging from the
verify critical path.

## Stacked PR dependency

This is a stacked change on top of #26445, `feat(hisparse): support
MTP/spec decoding with extra-page draft slots`. The intended review and merge
order is:

1. Update and merge #26445 with the stable page-table lifetime fallback used by
   the refreshed parent below.
2. Rebase this PR onto the merged commit.
3. Review this PR as a performance feature for that native path.

The public #26445 head (`aeefd1dbf`) runs all verify rows through one mutable
4096-row LRU buffer. A later row can therefore overwrite a physical slot still
referenced by an earlier row's returned page table. The local refreshed parent
fixes that lifetime gap by giving historical occurrences stable staging rows;
that parent update must land before this child is rebased for upload.

The local child is mechanically based on that refreshed parent on current
`main`. Its diff does not repeat the parent's generic extra-page allocation,
accepted-token finalization, scheduler handoff, or draft-worker lifecycle; it
adds only the narrow Demand commit/retirement hooks required by the buffer's
request-local state. After the parent is uploaded, the child branch can be
pushed as the second PR without changing this ownership boundary.

#26445 owns the HiSparse speculative-decoding lifecycle: draft/verify slots,
request state, and accepted-token finalization. The refreshed parent also owns
the native stable-materialization fallback. This PR keeps that lifecycle
unchanged and optimizes only how target verify attention obtains historical KV
rows. The diff will be
rebased so code already owned by #26445 is not repeated here.

Reviewers can therefore evaluate the ownership boundary independently:

- **#26445:** MTP slots, request lifecycle, accepted-token ownership, and the
  native materialization fallback.
- **This PR:** Direct, on-demand historical-KV resolution inside FlashMLA and
  its correctness/performance evidence.

The refreshed parent keeps native target-verify TopK in the request-relative
index namespace required by its materialization kernel. This child re-enables
fused TopK-v2 only for Demand, where the kernel explicitly publishes separate
raw-position and physical-Host-row outputs.

The child consumes two explicit contracts from #26445: the one-layer MTP
draft worker keeps its KV device-resident, and only the target worker owns the
HiSparse Host pool; target verification exposes four ordered query rows per
request plus their committed-length boundary. Compatibility code for these
contracts is present only while the parent is being refreshed and is removed
from the final rebased child diff.

## Motivation

The refreshed native path materializes every historical TopK occurrence into
stable request-local staging before launching attention. That prevents a later
verify row from overwriting a physical page table still consumed by an earlier
row, but its eager Host-to-device copies remain on the target-verify critical
path and reserve `num_draft_tokens * top_k` rows per active request.

A focused CUDA regression test isolates the original lifetime failure. With a
four-row hot buffer, step 0 loads logical rows 4-7 and step 1 loads rows 8-11.
On public #26445 head, reading through step 0's returned page table after the
call returns rows 8-11; the later step has overwritten every referenced slot.
The stable parent and the Demand Buffer preserve both rows' mappings.

The MTP Demand Buffer moves source selection into the FlashMLA producer. A
verify row first probes a small request-local HBM buffer and reads the
authoritative HiSparse Host row only on a miss. The row is filled and published
for subsequent consumers while attention is already running. This removes the
separate multi-step staging pass from the target-verify critical path without
changing TopK selection or speculative acceptance.

## Design

- HiSparse remains the owner of the full Host KV pool and request lifecycle.
- The existing MTP extra-page area remains the source for current verify rows.
- Historical rows use a 4096-row, per-request HBM MTP Demand Buffer.
- Each logical Host row has two independently hashed candidate slots in the
  4096-row buffer. A hit probes both slots; a miss claims an empty or lazily
  replaceable candidate. The second hash uses otherwise idle capacity instead
  of turning a first-slot collision into an immediate Host fallback.
- A compact tag stores the row, lazy epoch, and `FILLING` / `READY` state.
- The first producer group that claims a miss fills the slot and publishes it
  with release ordering. A concurrent consumer performs a bounded sequence of
  acquire observations and otherwise falls back to the authoritative Host row;
  it never waits or consumes partially written data.
- Entries are retained across adjacent decode calls and replaced lazily when a
  later access needs one of their candidate slots. No periodic clear or
  prefetch pass is added.
- Accepted MTP rows are committed through #26445's existing HiSparse lifecycle.

This is a direct-demand cache. It does not add a planner, prefetch stream,
materialized union buffer, or benchmark-only execution mode.

## Non-goals

- Replacing HiSparse's authoritative Host KV pool or request ownership.
- Reusing approximate TopK indices across target layers or MTP iterations.
- Changing EAGLE/MTP sampling, verification, or accepted-token finalization.
- Adding an asynchronous prefetch pipeline or a second CUDA stream.
- Generalizing the first PR to unsupported attention backends or KV layouts.

## Change boundary

The child PR contains only:

- the `mtp_demand_buffer` HiSparse configuration and support gate;
- request-local MTP buffer allocation, tags, generation, and release;
- a target-verify adapter that forwards logical TopK and HiSparse source
  metadata to FlashMLA;
- the FlashMLA producer-side source resolution, fill, and publication protocol.

The benchmark-only fixed-acceptance and deterministic-expert controls already
exist in SGLang and are not part of the feature diff.

### Implementation ownership

The implementation follows the ownership boundaries already present in
HiSparse instead of introducing a parallel Demand subsystem:

- `HiSparseCoordinator` owns request-local buffer allocation, lazy epochs,
  current-step overlay binding, accepted-row commit, and request release. These
  operations extend the existing HiSparse request lifecycle and stay next to
  its native MTP path.
- The DSA TopK path accepts one typed output bundle. Its generic kernel only
  gains the ability to emit logical indices and directly transformed Host rows;
  it does not know about Demand tags or replacement policy.
- The Python FlashMLA wrapper accepts one typed HiSparse adapter rather than a
  list of feature-specific tensor arguments. The adapter validates the
  supported layout and dispatches the dedicated operator.
- FlashMLA keeps validation, parameter construction, source resolution, cache
  fill, and publication in dedicated HiSparse headers. Its public sparse-decode
  entry point, common parameter object, and main kernel contain only narrow
  adapter hooks at the points where source rows are selected and consumed.
- Accepted-window Host writeback lives in a dedicated HiSparse MTP kernel
  header; the existing HiSparse JIT translation unit only includes it.

This keeps policy and lifecycle cohesive inside HiSparse while limiting changes
to shared attention and TopK code to the minimum contracts the feature needs.

## Configuration and support matrix

The feature is opt-in through HiSparse configuration:

```bash
--enable-hisparse \
--hisparse-config '{"top_k":2048,"device_buffer_size":4096,"mtp_demand_buffer":true}'
```

Initial support is intentionally fail-closed to the validated path:

- CUDA SM90
- MLA DSA with FP8 KV and `flashmla_kv`
- fused SGLang TopK-v2 (`dsa_topk_backend=sgl-kernel`)
- HiSparse page size 64
- EAGLE/MTP `topk=1`, `num_steps=3`, `num_draft_tokens=4`
- TP8 / DP8 with DP attention, PP1, CP1
- integrated serving, or PD decode with
  `index_share_for_mtp_iteration=false`

Unsupported configurations continue to use #26445's native HiSparse MTP path.
`mtp_demand_buffer` defaults to `false`, so merging this PR does not change the
existing HiSparse MTP behavior unless the feature is explicitly requested.

## Capacity

The Demand Buffer reserves 4096 physical KV rows per active request from the
existing HiSparse device allocator. Each physical row contains one 656-byte FP8
MLA row for every local attention layer. For the 78-layer GLM-5.2 target this is
199.875 MiB per active request and DP rank; B16 uses two active requests per
rank, or 399.75 MiB. Tags consume another
2.4375 MiB per configured request slot across 78 layers and are allocated once
at server startup. The benchmark uses `max_running_requests=16` and a KV token
budget that admits two 128K requests per DP rank. Allocation failure is
reported rather than silently switching execution semantics.

## Correctness

- Byte-exact FlashMLA output/LSE parity against the HBM source for B1 and B8.
- Same-epoch cache-hit test with the corresponding Host rows corrupted, proving
  that resident rows are consumed from HBM.
- Lazy-expiration test across an epoch boundary, including replacement and
  release publication; no `FILLING` tag may remain after the kernel.
- 24-bit generation wrap test proving lazy promotion/replacement does not expose
  stale or partially filled rows.
- Native overflow regression with 640 unique TopK rows against a 512-row hot
  slice, proving all four returned page tables remain byte-correct.
- End-to-end natural-text runs compare speculative acceptance against Base and
  record the input/output hashes, verify count, acceptance, and retractions.
- A fixed 70% acceptance run is used only for deterministic performance A/B.

## Performance

Hardware: 8x NVIDIA H20, GLM-5.2 W4AFP8, TP8/DP8.

Common workload and server settings:

- 128K input, 1K output, identical input-token hashes and seed
- EAGLE/MTP: `num_steps=3`, `topk=1`, `num_draft_tokens=4`
- `index_share_for_mtp_iteration=false`, so the comparison exercises complete
  per-row target TopK rather than the checkpoint's cross-iteration shortcut
- fixed 70% acceptance for deterministic timing; a separate natural-text run
  checks for a material speculative-acceptance regression, while GPU byte-exact
  output/LSE parity is the strict attention-correctness gate
- `max_running_requests=16`; enough KV token budget for two 128K requests per
  DP rank
- B1 has one global request; B8 has one request per DP rank; B16 is balanced
  exactly, with DP ranks 0-7 receiving two requests each
- identical warmup and no co-tenant GPU activity
- identical model, full layer count, KV dtype, attention backends, CUDA Graph
  capture, seed, prompt-token hashes, source commit, and compiled extension
  across variants; the A/B changes only the HiSparse feature configuration

Variant definitions:

- **Base**: MTP enabled; target KV remains fully resident in HBM; HiSparse off.
- **HiSparse + MTP**: refreshed-parent native stable-materialization path;
  `mtp_demand_buffer=false`.
- **HiSparse + MTP + Demand Buffer**: same HiSparse/MTP lifecycle and workload,
  with `mtp_demand_buffer=true` and a 4096-row request-local side reserve instead
  of native's `4 * 2048` stable staging rows.

The three variants are built from the same child-PR commit and load the same
FlashMLA extension. `Base` and `HiSparse + MTP` exercise the unmodified paths
from that build; only the server-side HiSparse configuration differs. This
avoids comparing the Demand Buffer against the public #26445 checkout or a
different kernel binary.

The public #26445 table (`20.69 ms` HBM+MTP versus `26.24 ms` HiSparse+MTP) is
not reused as this PR's native baseline. It measures the older mutable-LRU path
with about 388 input tokens per request on average, whereas this table measures
the stable fallback at 128K input. The algorithm and Host-miss pressure are
therefore both different.

The deterministic performance run uses:

```text
SGLANG_SIMULATE_ACC_LEN=3.1
SGLANG_SIMULATE_ACC_METHOD=match-expected
SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed
SGLANG_SIMULATE_ROUND_ROBIN_EXPERTS=1
```

The first three controls produce a reproducible ~70% accepted-draft workload;
the last removes data-dependent MoE route-shape variance from the A/B. These
controls are used only for timing. The separate natural-text run has none of
them enabled.

Natural-text acceptance validation uses the same deterministic ShareGPT token
corpus (8K input, 256 output, B1) without any simulation environment variable:

| Run | Input hash | Verify steps | Mean accept length | Mean accept rate |
| --- | --- | ---: | ---: | ---: |
| Base / HBM | `776f3f4712f5...` | 79 | 3.2405 | 75.53% |
| HiSparse + MTP (refreshed parent) | `776f3f4712f5...` | 79 | 3.2405 | 75.11% |
| HiSparse + MTP + Demand Buffer | `776f3f4712f5...` | 78 | 3.2821 | 75.64% |

The three runs use the current stacked code and no acceptance simulation. Native
matches HBM's 79 verify steps and 3.2405 mean accepted length; its reported rate
differs by 0.42 percentage point. Demand differs from HBM by one verify step,
1.28% in mean accepted length, and 0.11 percentage point in reported acceptance
rate. None of the runs retracts a request. Output hashes are recorded in the raw
artifacts but are not used as an oracle because this MoE checkpoint is not
output-deterministic across identical replays. GPU byte-exact output/LSE parity
remains the strict attention-correctness gate.

| Workload | Base (MTP, all KV in HBM) TPOT | HiSparse + MTP TPOT (refreshed native) | HiSparse + MTP + Demand Buffer TPOT | Demand vs Base | Demand vs refreshed native |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128K / 1K / B1 | 21.607 ms | 85.807 ms | 23.160 ms | +7.18% | -73.01% |
| 128K / 1K / B8 | 21.591 ms | 86.545 ms | 24.120 ms | +11.72% | -72.13% |
| 128K / 1K / B16 | 149.935 ms | 220.020 ms | 155.589 ms | +3.77% | -29.28% |

Supporting full-model results:

| Workload | Variant | TTFT | Aggregate output throughput | Completion span |
| --- | --- | ---: | ---: | ---: |
| 128K / 1K / B1 | Base | 244.074 s | 3.764 tok/s | 265.660 s |
|  | HiSparse + MTP | 248.650 s | 2.991 tok/s | 334.371 s |
|  | HiSparse + MTP + Demand Buffer | 249.919 s | 3.662 tok/s | 273.056 s |
| 128K / 1K / B8 | Base | 250.972 s | 29.346 tok/s | 272.609 s |
|  | HiSparse + MTP | 258.530 s | 23.177 tok/s | 345.163 s |
|  | HiSparse + MTP + Demand Buffer | 257.879 s | 28.363 tok/s | 282.054 s |
| 128K / 1K / B16 | Base | 378.602 s | 30.253 tok/s | 528.878 s |
|  | HiSparse + MTP | 391.389 s | 26.085 tok/s | 613.375 s |
|  | HiSparse + MTP + Demand Buffer | 386.236 s | 29.501 tok/s | 542.356 s |

The table reports the measured deltas without a pass/fail performance
threshold. TTFT, end-to-end latency, output throughput, actual concurrency,
per-rank placement, CUDA Graph usage, and cache state are captured in the raw
JSON artifacts; the aggregate timing and placement evidence is reported above.
No reduced-layer or synthetic-model result is included in the release table.

For B8 and B16, the raw result must also prove DP placement: B8 routes exactly
one request to each DP rank; B16 routes exactly two to each rank. A run is
discarded if prompt hashes, verify counts, CUDA Graph coverage, rank placement,
or co-tenant GPU state differs between variants.

The table reports mean per-request TPOT. For B16, aggregate output throughput
is reported alongside it because each DP rank receives two 128K requests and
chunked-prefill scheduling can delay one request behind the other. The same
scheduling policy is retained for all variants rather than introducing a
Demand-specific benchmark mode.

## Profiling

Matched Nsight Systems traces for the unchanged HBM and Demand kernels contain
810 FlashMLA split-K calls per variant. HBM averages 22.183 us per call; Demand
averages 83.831 us, including Host fallback, read-through fill, tag publication,
and attention. The 61.648 us delta is therefore the fused Demand cost, not an
additional planner or staging kernel.

The earlier native trace is intentionally excluded from release evidence. It
predates the parent's stable per-occurrence materialization change, which
materially changes both allocation and copy traffic. The final performance table
is rerun against the refreshed parent; no pre-refresh native profiling number is
presented as current.

## Test plan

- [x] Python unit tests for configuration, lifecycle, and fallback
- [x] CUDA B1/B8 exact-parity and epoch tests
- [x] Multi-step page-table lifetime regression (fails on public #26445 head)
- [x] FlashMLA patch dry-run against the pinned upstream revision
- [x] AOT FlashMLA build
- [x] Natural-text end-to-end acceptance comparison
- [x] Fixed-acceptance B1/B8/B16 performance table
- [x] Matched profiling comparison
- [x] Child diff rebased onto the refreshed #26445 head with parent-owned code
      absent
