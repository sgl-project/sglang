# Proposal: remote shared Engram tables

> Discussion draft. This branch contains an experimental adapter for an
> external research runtime; it is not a standalone or production feature.
> The provider interface below is illustrative, not the implemented API.

## Motivation

Independent serving instances may each hold the same large, read-only Engram
table in host memory. A shared registered table could let multiple instances
fetch the rows needed by their own requests while keeping one backing copy. The
goal is to reduce duplicated table memory while keeping serving overhead small.

SGLang already supports GPU and CPU placements, per-node host sharing and
several lookup strategies. This proposal concerns sharing across independent
instances, potentially on different nodes; it should complement those existing
paths rather than replace their defaults. Remote access itself is not novel.

## Candidate integration

One possible integration is an opt-in table provider behind the existing Engram
configuration. The provider owns connection setup, row requests and completion;
the Engram layer retains hashing, TP/CP mapping, dequantization and output shape.
The native table provider remains the default. The following is pseudocode,
included to make the proposed boundary reviewable rather than prescribe an API:

```python
# Illustrative only: not implemented in this branch.
provider = open_remote_table_provider(config, layer_id, row_range)
provider.finish_load()  # establish views before CUDA graph capture
rows, scales = provider.fetch(indices, stream=current_stream())
# fetch must establish completion and visibility before these values are used.
output = dequantize_and_assemble(rows, scales, indices)
```

This needs a concrete decision on asynchronous completion and GPU visibility;
the pseudocode does not specify whether `fetch` waits, returns a completion
handle, or arranges a stream dependency. Configuration, error propagation,
checkpoint identity, table ownership, cancellation, graph lifetime, and
shutdown/drain behavior also need review. An inaccessible remote table must not
silently return incorrect data. Fallback policy needs an explicit design, since
readers that omit local weights cannot automatically fall back to a local table.

### Experimental adapter in this draft

Setting `SGLANG_EXPERIMENTAL_REMOTE_ENGRAM=1` and selecting
`--engram-config '{"placement":"cpu","replication":"none"}'` routes
TP-owned rows through the research adapter. The option is off by default.
The adapter requires the external `shared_model_adapter`, `row264_kernels`,
owner service, and experimental NIXL/GPUNETIO library used in the prior
research runtime. It currently only supports 256-wide Engram layers 1 and 14.
`ENGRAM_SHARED_PREFETCH=1` enables the external reader's experimental CP hash
prefetch; `ENGRAM_SHARED_READ_PLAN=1` uses its prepared multi-QP READ plan.
These options require the matching external runtime and have not been qualified
for all graph, cancellation, or model configurations. It is not a self-contained installation,
nor does this draft provide the owner, NIXL GET implementation or setup script.
The experimental runtime validates checkpoint identity and registered table
metadata. With the experimental runtime, one same-version H20 1P1D smoke ran
the real model with 16K input and produced 64 output tokens matching a prior
reference output exactly. The P used TP8/CP8, bounded replay disabled and the
remote reader; D used DP8, a native per-node host table and DSpark. Mooncake
handled P/D KV transfer. P recorded one KV transfer, and both P/D requests
returned successfully. This is a single functional smoke, not a performance,
concurrency or production qualification. The reader runtime and owner remain
external dependencies.

A subsequent paired C8 test used three rounds of 32 measured requests per arm
with 16K input and 64 output tokens on the same two H20 nodes. The Shared P
used 16 QPs and the experimental READ plan/prefetch; Native P used the current
per-node local CPU table. Both used the same D process, Mooncake KV, model
version, prompt sets, and bounded replay disabled. Median-of-round TTFT p50 was
5159.83 ms Native and 5204.95 ms Shared (+0.87%); output throughput was 94.33
and 93.52 tokens/s (-0.85%). All 96 paired input and output hashes matched,
with zero measured cache hits, retractions or client errors. P was restarted
between arms, Shared ran first, and no per-window NIC counter audit was made;
the sample is not a production or p99 qualification. The 1-QP path had a
5928.08 ms TTFT p50 in one earlier 32-request round; the combined READ plan and
prefetch changes improved that matching-input round by 12.44%, but individual
contributions and multi-run 1-QP variance were not isolated.

## Research evidence and limits

An external research prototype, based on SGLang commit
`9a9ae36865e6c7c24fcbf3ce44a14b5e2945709a` and adapter commit
`c2ba4222787820e5d46069a163b49ded95a94aee`, used GPU-initiated READs from a
registered pinned-host table. It used an experimental NIXL/GPUNETIO path and
DOCA 3.1; it is not an implementation of this proposal on current main. Its
private TP/CP hooks and underlying NIXL interfaces need to be ported and
reviewed. NIXL device-GET discussion: https://github.com/ai-dynamo/nixl/issues/2543
and reference draft: https://github.com/ai-dynamo/nixl/pull/2544. The latter is
not a current-SDK runnable implementation.

In a paired H20 experiment, three independent TP8 prefill groups shared one
remote table while a TP8/DP8 decode group kept its native table. P/D KV
transport used Mooncake in both arms. With 16K input tokens, 64 output tokens,
client concurrency 24, DSPARK block size 5 and decode CUDA graphs, three rounds
of 96 measured requests per arm produced matching outputs for all 288 pairs.
The median of the three round medians was 1632.793 ms native versus 1647.930 ms
shared TTFT (+0.93%), and 834.304 versus 828.517 output tokens/s (-0.69%).
TTFT p99 was +2.68% and TPOT p90 +5.69%; per-round samples are small and these
are not production tail guarantees. All measured client requests completed;
prefill abort-submission counters were nonzero in both arms and their cause
remains unresolved.

The P-side logical table payload was three copies versus one shared copy, a
66.7% reduction. Including the unchanged D table, it was four copies versus two,
a 50% reduction. These are logical payload counts, not measured total RSS. The
owner still consumes pinned host memory and readers consume workspace. The
experiment used synthetic prompts and a debug router; it does not establish
production capacity or 1M-token performance. It does not compare against
GIN/NVSHMEM or the latest SGLang CPU/THP configuration.

## Questions for review

1. Should remote table access be expressed as an Engram provider, or integrated
   differently with the existing placement, replication and lookup options?
2. Where should table ownership, checkpoint verification and failures live when
   independent instances share a table?
3. Which completion, CUDA graph and lifecycle guarantees are required at the
   SGLang/NIXL boundary?
4. What would be the smallest useful functional test and reproducible benchmark
   for an initial implementation?

The intention is to discuss and agree on the design before converting this
draft into a ready-for-review implementation.
