# KV load recovery through ordinary prefill

`--kv-load-failure-policy {recompute,repair}` is a SGLang setting, independent
of the external cache provider. The default `recompute` keeps the valid prefix
before the first missing block and uses ordinary suffix prefill. `repair`
retains the healthy suffix privately and schedules only the missing contiguous
span as a normal chunked prefill. Its sampled token is discarded by the existing
middle-chunk path. After that forward completes, the normal checkpoint boundary
adopts the private healthy suffix and resumes from the restored extent.

A provider reports a fully validated loaded range, one missing span, and exclusive
ownership of the allocated destination indices. It must order successful copies
before the consuming model forward and drain failed copies before freeing memory.
The shared engine recovery object retains the obsolete missing-block allocation
and healthy suffix separately from the valid-prefix allocation. Ordinary prefill
allocates fresh slots for the hole. Successful checkpointing frees the obsolete
slots, appends the untouched suffix to the request row, and only then publishes
it in the radix tree. Abort frees all still-private allocations.

The first adapter is LMCache-MP terminal `RetrieveCoverage`, backed by actual CMX
reads in qualification. HiCache can supply the same engine-owned recovery object;
no scheduler policy or replay execution belongs in the LMCache adapter. HiCache
partial-read evidence delivery itself is not implemented by this patch.

Initial scope: one GPU, full-attention-only model, one contiguous missing span,
page-aligned boundaries, synchronous scheduling, and standard chunked prefill.
No recurrent state, sliding windows, speculative decoding, logprob/embedding
requests, sessions, shared destination writes or disaggregated execution. Refuse
unsupported repair configurations or safely use suffix recompute before retaining
private suffix state. Multi-request repair batching is not required: reuse the
existing single middle-chunk request, while normal prefill/decode may share work.
No attention kernel changes are required. Normal compilation and CUDA graphs stay
enabled. Validate correctness before measuring; exclude all startup compilation,
graph capture and workload warmup from timed comparisons.

## Adapter stream ownership

External transfers must fence the stream that executes the consuming forward.
SGLang TP1 without overlap uses the scheduler's current stream; overlap and PP
use the model runner's dedicated forward stream. The LMCache adapter selects
that stream for its producer event and H2D consumer wait. Waiting on the unused
runner stream leaves a healthy large-prefix load racy even though partial-load
recovery happens to drain its GPU completion before changing allocations.
This fix uses normal stream ordering, without adding a CPU wait for healthy hits.
