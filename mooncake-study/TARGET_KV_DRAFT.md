# Target-KV DSpark Serving Contract

This implements the SGLang serving side of design package P8. It consumes an
explicit KV-input checkpoint. A fixed-input numerical gate now exercises the
existing SpecForge backbone through a test-only KV adapter. The retained Qwen3
BF16 fixture passes against the pinned FlexAttention reference with exact layer
and logit equality. A production SpecForge exporter and trained-checkpoint
validation remain integration work. Synthetic test
checkpoints are wiring fixtures, not trained drafts or acceptance benchmarks.

## Checkpoint Identity

`config.json` must contain:

- `architectures: ["DSparkTargetKVDraftModel"]`.
- `input_mode: "target_kv"`.
- `target_kv_contract`, described by
  [the generated schema](training-data-contract/dspark-target-kv.schema.json).
- `enable_confidence_head: false`.
- Normal dense DSpark backbone configuration, including `hidden_size`,
  `num_hidden_layers`, attention geometry, RoPE, `block_size`, `mask_token_id`,
  `markov_rank` and `markov_head_type`.

The nested contract pins the teacher's weights, tokenizer, adapter identity,
resolved implementation fingerprint and shared-head transform. It also pins
selected layer order, K/V geometry, source K stage/norm, RoPE, feature K stage,
encoder dimensions/norm semantics, sequence alignment, training objective,
TV tail policy, confidence policy, compatibility version and validation metadata.
Use `TargetKVDraftContract.decode` and `read_target_kv_draft_contract` in addition
to JSON Schema: shape relationships, finite values and semantic compatibility
are checked in Python.

The runtime computes target identity from local safetensors/config/tokenizer
artifacts and the loaded model's attention implementation. The expected identity
comes from the checkpoint's data contract. A mismatch rejects startup. Storage
chunk size and source page size may differ between capture and serving; they do
not change the logical feature representation. Layer order and numerical codec
must match exactly.

## Parameters and Math

The checkpoint contains these trainable components:

| Names | Meaning |
| --- | --- |
| `kv_encoder.projection.weight` | Bias-free `[draft_hidden, feature_size]` projection |
| `kv_encoder.norm_weight` | Output RMSNorm scale |
| `layers.*`, `norm.weight` | Dense DFlash/DSpark backbone and context projections |
| `markov_head.*` | Vanilla, gated or RNN Markov head selected by the config |

The shared embedding and output head are attached from the bound target after
loading. Do not include target decoder weights, `embed_tokens`, `lm_head`, the
legacy `fc`/`hidden_norm`, or confidence weights. Unknown, missing, duplicate,
partial Q/K/V or gate/up shards, and mixed full/sharded parameters are rejected.
Native fused weights and complete HF-style Q/K/V or gate/up shards are accepted.
An optional leading `model.` is normalized before validation.
Exact parameter names take precedence over shard aliases: a gated Markov head's
`gate_proj` is a complete parameter, not an MLP `gate_up_proj` shard.

For each context row, concatenate selected layers in the declared order; within
each layer, flatten K then V in head/dimension order. The feature width is
`sum(num_kv_heads * (key_head_dim + value_head_dim))`. The input tensors are
detached constants; gradients through the shared encoder implementation reach
its projection and norm parameters.

For `feature_k_stage=pre_rope`, restore post-RoPE K using standard, unscaled RoPE
in FP32 at the actual token positions. Keep target K normalization. Construct
FP32 inverse frequencies as `1 / theta ** (arange(0, rotary_dim, 2) / rotary_dim)`;
casting an entire frequency buffer to BF16 or changing the FP32 operation order
can alter rounding. Split-half, interleaved and partial rotary layouts are
defined by the codec. Inversion cannot recover precision already lost in the
source KV dtype.

Cast features to the projection weight dtype, apply the linear projection,
compute RMS variance and norm-weight multiplication in FP32, then cast back.
This encoder rule is separate from the backbone's Qwen3 RMSNorm semantics:
backbone residual addition rounds to the activation dtype before computing its
FP32 variance, and normalized activations round again before multiplying by the
norm weight. The KV-input model uses the existing HF-cast norm kernel after an
explicit narrow residual addition. Fused Q/K processing, fused context writes
and stacked context projection honor the same before-weight cast; their legacy
default retains the previous semantics. A mixed norm policy falls back to
individual operations. Parameter names and checkpoint weight layout are unchanged.

Draft context projections use the dense DSpark K/V projection. The draft
backbone uses full split-half table RoPE: cosine/sine and each product round to
the activation dtype before addition. The BF16 Q/K and context-write kernels
retain fusion but disable floating-point contraction in this mode to preserve
those boundaries. Stacked and individual projections use the same policy;
mixed layer policies fall back to individual operations. The shared rotary
cache is not mutated. The KV-input MLP uses native SiLU followed by a separate
multiplication, preserving the narrow activation boundary. This introduces an
extra kernel boundary; its latency impact has not been measured.

Attention rounding still requires the full fixed-input gate against the chosen
training implementation. Matching the auxiliary arithmetic does not certify
backbone/logit parity. The current reference uses SpecForge's default Qwen3
modules, not its optional Liger provider. Legacy hidden-input drafts keep their
previous RoPE and activation behavior.
The target shared output head's
scale and softcap are applied in FP32 exactly once, before Markov correction.
No random confidence head is created; static verification is required.

## Sequence and Cache Ownership

For anchor position `a` and `G=prediction_count=block_size`, the context is
target KV `[0:a)`. Draft input has exactly G rows: the anchor followed by G-1
mask tokens. Positions are `[a:a+G)`, labels are `[a+1:a+G+1)`, and Markov
previous tokens are the anchor followed by the previous labels. The target
verification window has G+1 rows, separately from the draft input length.

`TargetKVInjector` exposes:

```python
bind(tokenizer_path=..., prediction_count=..., mask_token_id=...)
inject_target_kv(req, committed_prefix_end=..., source_ranges=...,
                 draft_weight_version=...) -> torch.cuda.Event | None
invalidate_projected_context(req, reason=..., new_weight_version=...)
```

`source_ranges` are contiguous logical ranges containing physical source slots
and actual positions. Gaps, overlaps and stale weight versions are rejected.
Projection gathers at most 1024 rows at once to bound temporary feature storage.
The completion event is recorded after the writes on the caller's CUDA stream;
CPU unit fixtures return `None`. Serving currently produces and consumes these
buffers on the ordered worker stream, before the scheduler can reuse slots.

After prefill, `ensure_context` projects the whole committed prefix on first
use, including target KV obtained through prefix reuse, and only appends new
rows afterward. After verify, `inject_verify` projects the forwarded anchor and
correct drafts. Rejected drafts and the next, unforwarded bonus have no entry in
that appended range. Generation stopping/truncation is still handled by the
existing scheduler result processing after verification.

Each request records its projected end and checkpoint/cache version. Request
retraction clears this state, including same-length re-prefill at different
physical locations. Cache flush advances the injector epoch. A fresh request
reusing a request slot always starts with an empty projection state.

Online weight replacement and memory release/resume are rejected before any
worker mutation. Weight updates return their existing failure reply. Memory
control replies now carry `success=True, message=""` defaults; a failure is
propagated by the tokenizer control layer to HTTP 400. The serving process stays
alive and can continue generating. Deploy a new service instance to switch this
checkpoint variant; changing the target also requires a matching draft contract.

## Runtime Scope

Current capability gates allow TP and require PP=DP=1, dense
unquantized NHD target/draft pools, no LoRA or PD, and standard RoPE. Synchronous
scheduling and normal overlap are supported. Target identity binding currently
supports Qwen3, Qwen2 and Llama text models; real-model evidence currently covers
Qwen3-0.6B at TP1 and TP2 only. Static verify
supports ordinary execution and decode/verify CUDA graphs. Target hidden-state
capture is disabled for both ordinary execution and graph construction.

At TP startup, ranks agree on the actual target identity and geometry, then the
draft contract and checkpoint digest before projection collectives can begin.
Checkpoint validation uses global logical Q/K/V and MLP dimensions; native
parallel loaders shard or replicate them into rank-local parameters. The KV
encoder is replicated. Before each bounded 1024-token projection chunk, the
injector gathers selected target K/V across the target TP group in global-head
order and keeps one canonical copy of replicated heads. The replicated encoder
then writes each rank's local draft context projections. This inference gather
is separate from training capture, whose owner-local snapshot payloads go
directly to the Store without gathering through the capture control group.

Example with an exported checkpoint and local target artifacts:

```bash
SGLANG_RAGGED_VERIFY_MODE=static python -m sglang.launch_server \
  --model-path /models/target \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /models/kv-draft \
  --attention-backend triton \
  --speculative-draft-attention-backend triton
```

The checkpoint's golden-fixture/acceptance metadata is recorded and validated
structurally; startup does not execute or certify a SpecForge validation report.
Production exporter compatibility, trained-model quality/throughput gates,
broader parallel topologies, non-static verification, PD, RDMA and production rollout
remain open. The retained Qwen3 BF16 fixture now passes the complete backbone
and logits gate against the pinned FlexAttention training reference; this is
not a certificate for other models, backends or runtime versions.

### KV-Draft Attention

`TargetKVAttention` explicitly selects a logical-order Triton kernel. It reads
prefix and current-block slot indices directly from the existing backend
metadata, including non-contiguous physical slots. The logical 64-token tiles
can straddle the prefix/block boundary; attention never restarts softmax at
that boundary. Grouped query heads share a KV tile, with FP32 accumulation,
exp2 softmax and probabilities rounded to the input dtype before the value
product. The short-block path uses two warps and one pipeline stage.

This follows the arithmetic observed in the pinned training reference. Its
generated FlexAttention decoding kernel has 32 nominal splits, but its default
mask covers a single enormous sparse block, so only the first split contains
valid KV for the retained fixture. Dividing the actual short context among
32 splits changes the BF16 result. The serving kernel has no split scratch
buffers and does not gather or concatenate context KV.

The wrapper checks NHD shapes, dtype, index layout and block bounds using host
metadata, including during CUDA graph capture. It supports block lengths 1..64,
head dimensions 16..256 and 1..64 query heads per KV head. The backend rejects
custom masks, sliding windows, scaled/quantized KV, sinks and distributed
context attention for this path. Other SGLang attention layers keep their
existing selection. Independent kernel tests cover ragged lengths, empty KV,
strided tensors, tile boundaries and changing graph inputs.

## Online Capture

Add `--training-capture-config /path/to/capture.json` to collect training samples
while serving a DSpark draft. The existing registered Host arena,
Mooncake writer and Catalog producer protocol are reused. The manifest records
`capture_mode=speculative_accepted_target_path`; its tensor contract is unchanged.
The collector supports static, cap-accept and compact verification. The target-KV
v1 checkpoint described above remains static-only; confidence-scheduled modes
use the existing hidden-input draft and confidence head. Simulated acceptance,
pipeline speculation and other speculative algorithms remain rejected by the
capture capability gate.

`TargetVerifyExecutor` calls `CaptureCoordinator.after_verify_forward` immediately
after the target forward, before grammar, penalties, bias or rejection sampling.
An owned ticket preserves compact raw top-128 IDs/scores, full-vocabulary LSE,
the selected requests' original batch rows, input tokens, positions and KV slots.
It does not retain mutable graph-output views.
For compact verification, the hook receives the original per-request
`verify_lens`. Cumulative offsets include unselected requests, while owned
ticket offsets include only selected real rows. Graph padding never becomes
sample data. Acceptance is bounded by each request's actual forwarded length,
so it cannot consume the next request's rows or a padded suffix.

`DSparkWorkerV2` passes the result to `after_verify_accept` before KV slot reuse.
For a prefix ending at `s` and a commit length `L`, the forwarded inputs are the
anchor plus `L-1` correct drafts. Their KV occupies `[s, s+L)`, while raw teacher
rows predict positions `[s+1, s+L+1)`, including the bonus/correction token.
The collector validates the anchor, consecutive positions and correct-draft
tokens against the actual emitted prefix. Rejected suffixes do not enter the
sample, and temporary copies are clipped to the reserved Host capacity.
Budget-trimmed suffixes are excluded by the same commit boundary, even when
their token IDs happen to equal a later committed output.

Scheduler stop/length/grammar processing can shorten that result further.
Finalization commits `Req.output_ids_through_stop` and truncates the owned teacher
and KV ranges before sealing. A final response token already computed by verify
has valid KV; an unforwarded bonus has `kv_valid=0`. No target forward is added
to fill missing KV or regenerate teacher scores. Capture failures invalidate the
sample through the existing writer cleanup path.

With overlap, a verify result can precede CPU processing of prefill or the
previous verify result. The request context preserves an ordered, bounded set of
pending model tokens, including the anchor and next bonus. Later CPU commits must
match those tokens; a mismatch fails the sample. Only CPU-confirmed tokens enter
the sealed snapshot. Stop/grammar/length processing may discard a pending suffix.
A lookahead verify starting beyond Host capacity is skipped while its previous
result can still finish and publish the sample. At the last possible position,
KV is copied without retaining a teacher prediction outside the sample.

`DSparkWorkerV2.needs_cpu_seq_lens` declares that KV projection or capture needs
the host sequence-length mirror. The scheduler combines this requirement with
the attention backends' requirements when constructing `FutureMap`. This keeps
the prefix length current even with a backend such as Triton that otherwise
uses only device lengths. All projection and capture copies remain on the
ordered forward stream; the latest capture event fences writer ownership.

### Batched KV D2H

Optional capture-configuration fields batch short KV ranges in a request-owned
device arena before copying them to the existing registered Host tensors:

```json
{
  "kv_d2h_batch_tokens": 16,
  "max_device_bytes": 16777216
}
```

The default batch size is one and allocates no device staging. A larger batch
requires an explicit positive device budget. The arena capacity is the smaller
of the batch size and `max_sample_tokens`; its allocation covers all selected
K/V components and all `max_inflight_samples` slots, including alignment gaps.
The budget is checked before any Host registration. It bounds staging tensor
bytes, not CUDA allocator reservations, model KV, teacher scores or transient
gathers used by large prefill ranges.

Large contiguous token ranges still go directly to Host. Short ranges gather
into the private arena immediately, so serving KV slots can be reused normally.
A full staging batch queues D2H on the producer stream before reusing the arena.
Sealing flushes the tail on that stream, even when finalization is called from
another stream context. The final completion event also covers earlier capture
streams. The writer waits for this event before constructing Store objects.

Abort/retraction discards the pending, unpublished tail and still waits for
already queued GPU operations. Host and device storage share a slot lease and
are both retained when transfer completion is uncertain. Stored tensor shapes,
checksums, keys and the training contract are unchanged; this optimization does
not add a target forward or retain rejected speculative tokens in a snapshot.

`/server_info` exposes `host_pool.device_allocated_bytes` and
`host_pool.device_limit_bytes`. With metrics enabled, the corresponding series
are `sglang:training_capture_kv_staging_allocated_bytes` and
`sglang:training_capture_kv_staging_limit_bytes`.

## Verification

`python -m sglang.test.dspark_target_kv_parity --checkpoint /models/kv-draft
--target-path /models/target` is an explicit fixed-input gate. Put the pinned
SpecForge source checkout on `PYTHONPATH`. The checkpoint must include the
contract-bound `validation/inputs.safetensors` snapshot tensors. The gate checks
their digest and the target artifact identity before reading the shared embedding
and output-head weights; tied target heads use the embedding. A guard rejects any
target decoder forward. No target prefill or teacher-logit regeneration occurs.

The training reference uses SpecForge's existing backbone and Markov head with
the shared KV encoder as a test adapter. Actual SGLang layers, context projection,
non-contiguous KV pool slots and Triton paged attention produce the serving output.
Different prefix lengths run together. Every decoder layer, normalized hidden
state, transformed shared-head logits and teacher-forced Markov logits is compared
using the checkpoint's declared tolerance. The windows include the first response
label and a partial final block. On success, a test-only cached CE+TV128 loss checks
finite, nonzero encoder/backbone/Markov gradients and frozen shared weights.

The gate writes `validation/parity.json` with pass/fail status, all stage errors,
artifact/source digests, dtype, backends and runtime versions. It also records
CUDA/cuDNN versions and the observed SDPA operators: PyTorch's `sdpa` selection
can choose different kernels with different BF16 rounding. Nonfinite values
fail. A failed rerun invalidates an older passing report; early artifact-loading
failures leave no report. The command exits unsuccessfully on a failed comparison.
This gate is separate from online capture's exact tensor readback test and is not
implicitly selected by installing SpecForge. Neither a successful tiny fixture
nor the separate FP32 diagnostic certifies the real BF16 serving path.

`test/registered/spec/dspark/test_dspark_target_kv_parity.py` covers vanilla,
gated and RNN heads, a real optimizer step followed by export/reload, prefix-only
input ownership, and report failure handling. It requires the pinned SpecForge
checkout and is explicitly disabled in generic CI until that dependency is wired.
The adapter and loss are test references, not a production SpecForge trainer,
collator, Catalog consumer or exporter.

`test/registered/unit/spec/test_dspark_target_kv.py` covers contracts, codec math,
gradient ownership, prefix isolation, range validation, request-slot reuse,
epoch/retraction behavior, bounded projection, strict loading, management
failure replies and graph hidden-mode selection.

`test/registered/storage/test_training_capture_runtime.py` captures actual
Qwen3 samples, reads them from Mooncake after the producer exits, and builds
synthetic KV-input draft checkpoints from their contracts. It checks chunked
prefill, prefix reuse, batches with different verify commit lengths, ordinary
and graph execution, raw projected-KV values, lossless greedy output, and
continued serving after rejected management calls. Observers and forced proposal
weights exist only in `sglang.test`; they do not modify normal serving behavior.
The DSpark servers also publish training snapshots. A test observer saves raw
vocabulary scores and target-pool rows before reuse; final output tokens select
the reference path independently of the collector's acceptance ticket. Readback
checks raw scores, top-128 membership, LSE, masks, positions and KV validity.
The live cases include stochastic sampling, repetition/frequency penalties,
minimum output length, stop tokens, EOS and constrained regex generation.
Both synchronous and overlap scheduling run with ordinary execution and CUDA
graphs. The overlap driver also checks exact Host capacity, a three-request batch
padded to a four-row graph, and streamed cancellation with fenced Catalog failure
and Host-slot recycling. Completed snapshots are re-read after each producer exits.
See [implementation evidence](IMPLEMENTATION_STATUS.md) for exact completed runs
and limits. The runtime Catalog remains a test double, and transport is TCP.
