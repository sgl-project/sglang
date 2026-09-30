# Target-KV DSpark Serving Contract

This implements the SGLang serving side of design package P8. It consumes an
explicit KV-input checkpoint. A SpecForge exporter and complete training versus
serving backbone/logit parity remain separate integration work. Synthetic test
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
Draft context projections use the existing dense DSpark K/V projection,
K-normalization and draft RoPE implementation. The target shared output head's
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

Current capability gates require one target/draft GPU, TP=PP=DP=1, synchronous
scheduling, dense unquantized NHD target/draft pools, no LoRA or PD, and standard
RoPE. Target identity binding currently supports Qwen3, Qwen2 and Llama text
models; real-model evidence currently covers Qwen3-0.6B only. Static verify
supports ordinary execution and decode/verify CUDA graphs. Target hidden-state
capture is disabled for both ordinary execution and graph construction.

Example with an exported checkpoint and local target artifacts:

```bash
SGLANG_RAGGED_VERIFY_MODE=static python -m sglang.launch_server \
  --model-path /models/target \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path /models/kv-draft \
  --disable-overlap-schedule \
  --attention-backend triton \
  --speculative-draft-attention-backend triton
```

The producer's `--training-capture-config` still rejects speculative execution.
Collecting accepted-token teacher/KV data while serving this draft is P9 work.
The checkpoint's golden-fixture/acceptance metadata is recorded and validated
structurally; startup does not execute or certify a SpecForge validation report.
Full training/serving backbone and logits parity, quality/throughput gates,
parallel topology, overlap, PD, RDMA and production rollout remain open.

## Verification

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
See [implementation evidence](IMPLEMENTATION_STATUS.md) for exact completed runs
and limits. The runtime Catalog remains a test double, and transport is TCP.
