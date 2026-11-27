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

Attention-only scope: one GPU, full-attention-only model, one contiguous missing span,
page-aligned boundaries, synchronous scheduling, and standard chunked prefill.
No sliding windows, speculative decoding, logprob/embedding
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

## Hybrid models: attention KV only

Hybrid recovery requires complete, independently validated coverage for every
registered attention/state group and a healthy later recurrent checkpoint.
Recovery of missing recurrent state is deliberately omitted. HiCache can use the
same engine recovery object, but its evidence adapter is not implemented here.

For a hole `[h,b)` and healthy checkpoint at `E >= b`, replay begins at a
healthy local joint checkpoint `c <= h`, or initial state if none is local.
The ordinary chunked-prefill scheduler allocates fresh attention slots and uses
the request's private recurrent/conv slots as scratch. Healthy delivered
attention rows `[c,h)` and `[b,E)`, plus the immutable checkpoint at `E`, remain
retained and unpublished. Intermediate replay chunks do not publish radix
checkpoints. After the forward through `b` completes, the engine restores the
checkpoint at `E` into active state, restores original healthy attention rows,
adopts the repaired hole, and frees scratch healthy replay rows. Normal radix
checkpoint publication then resumes at `E`.

Abort frees retained private attention rows; the external-load flow retains
ownership of its checkpoint until commit or normal cleanup. Insufficient
attention scratch capacity selects ordinary recompute before starting repair.
The normal source-node lock pins a successful repair's STORE allocation until
ordinary transfer completion. The LMCache adapter targets only the dirty
attention group, using existing null-group suppression to leave recurrent
storage unchanged. No separate retry, overwrite or persistence policy is added.

One hybrid repair may share a forward with ordinary traffic; it can span
multiple normal prefill chunks. No new attention or recurrent kernels are
required. Only already-local earlier checkpoints are reused: fetching an older
external checkpoint to shorten replay is a future optimization. GPU-cold
recovery can therefore replay from zero to the hole's end.

For compiled GDN decode, the existing linear-attention backend is kept outside
Torch tracing. Tracing its indexed recurrent/conv writes can functionalize
entire state pools and exhaust memory during graph capture. The surrounding
model remains compiled, and the ordinary backend kernels remain inside the
decode CUDA graph. This compiler boundary changes no attention algorithm or
kernel; a CPU regression checks state updates and live hybrid qualification
checks compiled execution and unchanged retained state bytes.

Replay can change prefill chunk and batch shapes. Ownership and checkpoint
preservation do not guarantee bitwise regenerated BF16 KV or identical greedy
outputs across different execution shapes. Numerical/model-quality qualification
is separate; ordinary prefix-cache reuse can also change greedy outputs.
