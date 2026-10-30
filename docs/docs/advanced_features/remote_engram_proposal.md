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
research runtime. It currently only supports 256-wide Engram layers 1 and 14;
the old prefetch path is rejected. It is not a self-contained installation,
nor does this draft provide the owner, NIXL GET implementation or setup script.
The experimental runtime validates checkpoint identity and registered table
metadata. This branch has not yet passed a full model or 1P1D run.

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
