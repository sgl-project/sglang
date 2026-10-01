# Implementation Evidence

The active goal is the SGLang and Mooncake portion of
`SGLANG_MOONCAKE_SPECFORGE_DESIGN.zh-CN.md`. This file records verified progress;
it does not redefine the goal as the modules already implemented.

## Baseline

- SGLang source baseline: `b3bffef70aa17733b48af91e4b529e72c913bc6e`.
- Implementation branch: `codex/dspark-maas-capture`, isolated from the original
  worktree and its existing staged changes.
- Mooncake reference source: `76bd234d7ae072edd3aed6ff595f94c85b635c2f`.
- Runtime SDK tested: `mooncake-transfer-engine-cuda13==0.3.11.post1`.
- H100 image: `harbor.local.clusters/bp/lmsysorg/sglang:v0.5.15`.
- Foundation tests used the image's PyTorch 2.11.0+cu130. Actual inference uses
  a dedicated Python 3.12.3 environment with PyTorch 2.13.0, sglang-kernel
  0.4.6.post1, FlashInfer 0.6.17 and Transformers 5.12.1, matching this source's
  requirements. See `experiments/h100-runtime-lock.json` for the base image
  digest and dependency overlay hashes. The system environment remains the
  idle worker's environment; the image's installed SGLang is not tested here.
- Resident allocation: `job-fe1ce1dcdea6-20261001023258`, one H100 80GB on
  initial node `node064`. Experiment submissions pause/resume the idle load.

## Current Evidence

| Area | Implemented | Evidence / Remaining Work |
| --- | --- | --- |
| Wire contract | Typed manifest, raw tensor descriptors, shape/byte/digest/coverage/content validation | Generated fixtures pass the design's JSON Schema; malformed metadata and contents are rejected |
| Raw teacher capture | Unpadded top-128 IDs/values and full-vocabulary LSE before serving processors | Independent online logits observer validates every captured row; serving bias does not leak into teacher scores |
| KV export | Selected layers, arbitrary source slots, NHD BF16/FP16, direct or bounded batched D2H | H100 source-reuse, cross-stream staging and exact online attention-input comparison pass, including chunked prefill, prefix hits and decode |
| Host ownership | Bounded registered arenas, quota rejection, reuse, transfer quarantine | Coordinator admission, renewal, expiry, retract, shutdown and publication tests pass; traffic-scale stress remains open |
| Mooncake adapter | Required hard pin, registered raw buffers, immutable retry verification, exact read length | Real cross-process TCP roundtrip passes; cross-node RDMA is pending |
| Publication | Catalog producer client, manifest-last writer, durable metadata journal, fenced replay | Lost responses, failed puts, stale fences, missing/corrupt objects and identical retries tested; actual Catalog service is SpecForge-owned |
| Partition publication | Owner-local writes and fenced all-owner publication receipts | Two independent writer processes publish logical head shards through real TCP Store; distributed inference admission and scheduler integration remain open |
| Partition ownership | Canonical replicated-head owners, PP-local Host/device staging, local KV export and metadata assembly | Native QKV loader agreement at TP1/2/4/8, exact CUDA source-reuse checks and independent Store writers pass; distributed coordination remains open |
| Global target binding | Rank-local projection/pool inspection and all-rank global identity assembly | TP4/PP3 metadata fixture matches a full target contract; deployed TP/PP model validation remains open |
| Startup identity exchange | Bounded JSON over the existing CPU group, phase failure votes and final digest agreement | Real four-process Gloo TP2/PP2 and TP4/PP1 cases pass, including local failures, peer exit and finite waits; distributed request/resource coordination remains open |
| Runtime collection | Opt-in CLI config, capability gates, request ledger, prefill/decode hooks, invalidation and counters | Six real Qwen3-0.6B requests published through Mooncake; ordinary and CUDA graph replay executions pass |
| Real model identity/parity | Weight/tokenizer artifact digests, actual selected-layer geometry, K norm and RoPE | Captured KV and teacher scores match online tensors exactly; full-vocabulary LSE matches within 1e-5; HF teacher logits pass numerical comparison, but cross-engine KV equivalence is not certified |
| Draft serving | Explicit KV-input architecture, contract, encoder, incremental injector and invalidation | Real Qwen3 target plus synthetic KV draft passes ordinary/batched/graph generation; retained BF16 fixture passes full backbone/logit parity against pinned FlexAttention, with production exporter and trained-model validation still open |
| Draft checkpoint validation | Exact packed/split shapes, supported floating dtypes and finite destination values before parameter writes | Malformed exports fail without changing parameters or projection caches; real GQA/MLP loaders, cross-dtype loads and fixed-input export/reload parity pass |
| Speculative collection | Static DSpark raw verify ticket, commit mapping and terminal truncation | Actual KV-input draft requests publish and read back through Mooncake in ordinary and graph modes; see evidence below |
| Overlap collection | AR lookahead and static DSpark pending-token ledgers, capacity boundary and terminal trimming | Real ordinary/graph requests, prefix reuse, delayed grammar and exact KV/teacher readback pass; see per-mode evidence below |
| AR cache lifecycle | Snapshot ownership across RadixCache eviction and explicit retract/resume | Real 256-token KV pool eviction, physical slot reuse, failed-capture exclusion and subsequent admission pass in synchronous and overlap/graph modes; automatic AR OOM remains open |
| DSpark memory pressure | Draft context reset/rebuild and capture retirement after automatic retraction | Real 512-token KV pool exhaustion passes in all four synchronous/overlap and eager/graph combinations; failed captures are excluded and fresh capture admission recovers |
| PD collection | D-owned complete snapshot with fenced first-teacher handoff and TP cohort publication | Real P1/D1, P1/D2, P2/D1 and P2/D2 pass eager and graph/overlap, exact source parity, batches, prefix reuse and failed/aborted sample exclusion; PD PP/speculation and RDMA remain open |
| Deployment coverage | Partial | TP2/PP1 and TP1/PP2 AR, TP confidence-scheduled DSpark and matching/asymmetric TP AR PD have runtime evidence below; combined topologies, PD speculation, RDMA and workload SLO gates remain open |

Initial test evidence (shared lab state under
`/gpfs/users/fuxuanwei-1/dspark-maas-lab/state`):

- `01790795836878267602-5cedaea4ecc9`: initial 12 tests / 22 subtests passed,
  including H100 asynchronous source-reuse checks.
- `01790796065501019845-c310c3d55efb`: real Mooncake master/SDK cross-process TCP
  read validated 20 objects / 4532 tensor bytes. This first run exercised the
  raw adapter; the integration test was subsequently extended to use the writer.
- `01790796528073868837-29b6c220333a`: 19 tests / 22 subtests passed, including
  manifest-last and crash-recovery failure boundaries.
- `01790796958201743290-885c301fc4ba`: the actual `SnapshotWriter`, journal and
  registered arena passed the real SDK/master cross-process TCP test. Catalog
  calls in this transport test use a test double. The independent reader
  validated all 20 objects / 4532 tensor bytes; the journal was empty after ACK.

Runtime test evidence:

- `01790801677964242601-39295c40360a`: 30 producer unit tests passed in the
  matching runtime environment, including the CUDA ownership test.
- `01790801966846060429-6158123feed6`: 108 existing configuration/runtime-context
  tests and 37 subtests passed after adding the CLI field and namespace.
- `01790802247034636938-f2f34259e7a8`: actual prefill/decode capture passed with
  a test-only independent attention/logits observer. All selected KV and
  top-128 score values are exact; top-128 membership and full-vocabulary LSE
  are checked independently. Producer exit does not prevent the separate
  Store client from reading the three published snapshots.
- `01790802358606301790-d48f81614828`: extended runtime test passed in 68.757s.
  Three additional requests used normal serving with decode CUDA graphs;
  five actual graph replays were counted. Every captured tensor matches the
  corresponding observed ordinary-execution sample exactly. This checks graph
  replay at batch size one, not padded/batched graphs or prefill graphs.
- `01790802703712589471-1d80760ee575`: final producer/configuration and real
  cross-process Store regression passed: 38 tests and 45 subtests. New capture
  modules pass Ruff I/F checks; touched upstream files add no diagnostics
  relative to the baseline (the baseline itself is not lint-clean).

The runtime fixture uses a 160-token prompt, response lengths 1/4/3, layers
0/14/27, 128-token prefill chunks and 64-token storage chunks. It verifies the
159-token prefix hit, partial final chunk, final-token `kv_valid=0`, accepted
token IDs, prompt/response masks, positions and a serving logit bias. The
resident job has no RDMA allocation; transport evidence is TCP only. All
Catalog calls still target a test double, not a production SpecForge Catalog.

An idle Python scheduler initially starved the publication thread between CPU
tensor checks. A conditional idle yield while publication work is pending
resolved the observed 30-second publication timeouts. This is functional
evidence; capture-on/off latency and throughput budgets are not yet measured.

## Cross-Engine Numerical Diagnostic

The online observer reads per-layer attention inputs before subsequent model
work can reuse them; it does not read the KV pool or use the snapshot exporter.
All 2,979,840 selected KV scalar values across the first three samples match
Mooncake readback exactly. It also reads raw logits before sampling.

HF BF16 eager KV does **not** pass the experimental `rtol=0.03, atol=0.125`
cross-engine check. For example, layer 14 V at position 20/head 0/dimension 118
is -27 in online SGLang and -1.953125 in the HF eager reference. The independent
reference-only experiment `01790802067633772909-b44e737cfdca` also obtains
-11.5625 with HF BF16 SDPA and approximately -17.30 with both FP32 backends.
It reproduces sensitivity without involving the exporter or Mooncake.

`experiments/diagnose_qwen3_kv.py` preserves FP32 RoPE frequency buffers while
loading each weight dtype separately. The runtime test retains all HF KV error
statistics; `--assert-hf-kv` re-enables the original failing cross-engine gate
without increasing its tolerances. The required capture invariant is exact
preservation of online tensors. This evidence does not establish equivalence
between arbitrary target implementations or training/serving DSpark parity.

## Target-KV Draft Serving

The new `DSparkTargetKVDraftModel` is selected explicitly by architecture and
`input_mode=target_kv`. The contract validates teacher identity, ordered layer
geometry, standard RoPE, sequence alignment, CE/TV128 objective metadata and
confidence policy. A shared differentiable encoder restores pre-RoPE K in FP32,
concatenates ordered K/V features, projects, and normalizes. Shared-head scaling
and softcap run once in FP32 before Markov correction. Checkpoint loading rejects
missing, duplicate, foreign and partial-shard weights.

Prefill reconstructs cached prefixes when needed; verification appends only
the forwarded anchor and correct drafts. Projection temporaries are limited to
1024 source rows per chunk. Request retraction and cache epochs invalidate
projected state. Online weight replacement and memory release/resume return
failure before mutation, with HTTP 400 propagated without killing the server.
Ordinary execution and graph construction both disable target hidden capture
for this input mode. Legacy hidden-input DSpark remains a separate path.

See [the serving contract](TARGET_KV_DRAFT.md) and
[generated checkpoint schema](training-data-contract/dspark-target-kv.schema.json).

Evidence:

- `01790805151196250369-780fa16868bd`: 14 contract, math, injection, weight-loader
  and management-reply unit tests passed. Subsequent tests also cover bounded
  projection and graph hidden-mode selection.
- `01790804809714917227-81a47be911cd`: first real KV-input checkpoint startup,
  three ordinary speculative requests and per-layer projected-KV checks passed.
- `01790805223032022696-81ad9ecd1ea7`: extended ordinary execution passed,
  including batch commit lengths `[1, 4]` and continued generation after rejected
  management calls. The graph observer then caught an inherited FULL hidden
  capture setting in graph construction. The setting was fixed at its source.
- `01790805422847548067-6f1c53cfcfbe`: complete extended runtime test passed in
  133.330s. It retains the six real Mooncake producer requests and adds five
  KV-draft generation requests per execution mode: ordinary and CUDA graph.
  Both modes exercise prefix hits, 128-token prefill chunks, request-slot reuse,
  rejection, full draft acceptance and a batch with different commit lengths.
  Graph mode records seven actual target-verify replays. Both modes have 15
  observed projection calls, maximum K absolute error 0.0078125 and exact V
  against the independent per-layer mathematical reference. All ten speculative
  responses match the ordinary target's greedy outputs. Weight-update and
  memory-control rejections leave the service usable.
- `01790805713557099271-c9566028d673`: final focused regression passed with
  103 tests and 74 subtests in 30.36s. This includes all 16 new target-KV unit
  tests, existing capture tests, existing dense DSpark projection parity,
  request IPC normalization, graph-runner helpers and legacy draft selection.
  New modules pass Ruff; touched upstream files add no F/I diagnostics relative
  to HEAD. `git diff --check` passes. An earlier regression submission failed
  before collection because one unchanged test file was not staged in the GPU
  workspace; staging the selected regression files resolved it.

These checkpoints copy target backbone layers and use synthetic encoder/Markov
weights. A forced proposal in the test fixture exercises both accept and reject
branches; it is not trained or evidence of useful acceptance/throughput. The
golden-fixture digest is contract metadata, not a certified SpecForge export.
Full backbone/logit parity, training quality, fresh-instance rollback and
production artifact validation remain open. Runtime gates still require static
verify, one GPU, no PD/LoRA and dense unquantized pools. Static speculative
teacher/KV collection is now implemented as described below.

## Static Speculative Collection

The target verify hook captures compact raw teacher tensors before serving-side
mutation. The commit hook validates the anchor, positions and correct-draft
prefix, then copies only the forwarded commit range into owned Host storage.
The request finalizer applies the scheduler's actual terminal token boundary,
including EOS/stop/length truncation, before sealing. This supports both a final
computed token and an unforwarded bonus without an extra target forward.

Evidence:

- `01790806925908285704-3d00701591aa`: 37 capture tests and 48 subtests passed in
  16.89s. New cases cover source mutation, rejected suffix exclusion, mixed
  selected/unselected batch rows, EOS and length truncation, verify windows
  crossing Host capacity, and invalid anchor/position/token-path rejection.
- `01790807115212349474-201ec59ce104`: complete runtime test passed in 134.655s.
  The six AR samples still pass, and ten static DSpark samples now publish and
  read back through the real Store. Each speculative mode records seven verify
  forwards and eight per-request commit copies; graph mode uses seven actual
  replays. Both modes exercise full rejection, full acceptance, length truncation
  and batch commit lengths `[1, 4]`. Selected KV and raw top-128 values match
  online reference snapshots exactly, and CPU full-vocabulary LSE agrees within
  `rtol=1e-6, atol=1e-6`. All ten greedy responses match the target AR baseline.
- `01790807394929085326-c7467397a17d`: extended runtime test passed in 228.638s.
  With tokenizer initialization enabled, both speculative modes additionally
  exercise temperature/top-k/top-p, repetition/frequency penalties, minimum
  output length, stop tokens, EOS and constrained regex generation. Each mode
  publishes nine samples and records 24 verify forwards / 25 commit copies;
  graph mode confirms 24 actual replays. All 18 speculative snapshots pass
  raw teacher, token-path and KV readback checks alongside the six AR samples.
  EOS/stop requests end at two output tokens despite a larger verify window.
- `01790807395062603809-0299f4562e39`: focused regression passed with 110 tests
  and 79 subtests in 19.03s, covering capture, target-KV serving, legacy draft
  selection/projection, graph-runner helpers and request IPC normalization.
  Added modules/helpers pass full Ruff; capture modules and tests pass Ruff
  F/I checks. The original worktree's staged changes remain untouched.

The speculative observer independently reads the target pool and full vocabulary
outputs before their reuse, then reconstructs the final path from returned token
IDs. It does not use acceptance tickets, call the KV exporter or read captured top-k
tensors. Unlike the earlier AR attention-input observer, this speculative KV
reference reads the pool, so it verifies selection/copy/lifetime, not the
attention backend's pool-write correctness. Test observers add synchronous work;
these runs are correctness evidence, not SLO measurements. Catalog is still a
test double and transport is TCP.

## Autoregressive Overlap Collection

The AR producer now accepts the normal overlap scheduler. `TpModelWorker` passes
the actual overlap setting to the coordinator. Forward-stream capture can lead
CPU result processing by one iteration: the next token's KV and prediction may
already have been copied when the previous result finishes the request.
Finalization trims the owned ranges to `output_ids_through_stop`; it retains a
computed final token's KV without adding a target forward. At exact Host capacity,
the lookahead's out-of-sample teacher row is skipped. A regressing output boundary
fails capture instead of publishing stale tokens.

Each context's latest CUDA event fences all its queued copies on the forward
stream. The writer waits on that event before reading a sealed snapshot or
reporting an aborted capture and recycling its Host slot. No ScheduleBatch is
retained by the writer. `/server_info` reports `enable_overlap` and the
`overlap_forwards` counter. Static speculative overlap is described below.

The shared test observer follows actual forward input tokens/positions, since
`Req.output_ids` can lag in this mode. It preserves the first observed KV per
logical position. `RadixCache.cache_unfinished_req` can subsequently replace
duplicate-prefix slots; those canonical values can differ numerically across
batch shapes. Re-reading a later mapping must not overwrite an earlier reference
snapshot. The test records source slots and asserts real remapping occurs while
the stored first observations remain exact, with no relaxed KV tolerance.

Evidence:

- `01790808100024882127-b3690db9c7bf`: 40 capture tests and 47 subtests passed
  in 18.65s. New coordinator cases cover delayed result finalization, lookahead at
  exact capacity and abort with pending results.
- `01790808330614519203-7b105437c401`: focused regression passed with 114 tests
  and 78 subtests in 20.68s. This includes the added CUDA test that queues two
  capture steps on a forward stream, finalizes on the CPU and verifies the writer
  waits for the latest copy event rather than the earlier step's event.
- `01790808330481545652-b43afe6cb9c2`: the initial runtime driver submitted a
  three-request batch without waiting for three spare capture reservations;
  eight of nine expected samples published. The driver now waits for its required
  admission capacity and asserts cumulative admission counts after each request.
- `01790808583474243750-18ea6add90b5`: all nine AR overlap samples published.
  The reference checker then exposed its own later-prefix overwrite at position
  159. The checker now preserves first observations and tracks physical remaps.
- `01790808873573886521-9c2ba12f38dc`: full runtime test passed in 218.712s.
  It validates 42 snapshots: six ordinary AR, 18 overlap AR and 18 static DSpark.
  Each overlap mode publishes nine completed samples, reports 43 capture forwards
  including the aborted request, and confirms six logical positions changed
  physical slots after their first observation. Graph mode executes 34 actual
  replays, including a three-request batch padded to the four-row graph. Exact
  capacity, EOS, delayed grammar sampling and one-step CPU result lag all pass.
  Both streamed aborts reach fenced `FAILED` Catalog state and publish no sample.
  Completed overlap snapshots remain readable after producer exit. The existing
  speculative cases still pass with 24 graph replays in their graph mode.

Changed capture code/tests pass Ruff F/I checks, added helpers pass full Ruff,
and `git diff --check` passes. The original worktree's staged diff hash is
unchanged. These results cover AR overlap on the resident H100 and Qwen3-0.6B;
they do not by themselves certify speculative overlap or multi-rank ownership.

These observers synchronously read raw outputs and pool rows to validate data;
they do not establish overlap performance, latency SLOs or arbitrary-backend
numerical equivalence. The production capture path still uses its asynchronous
copies and background writer. Catalog remains a test double and Store transport
remains TCP.

## Static Speculative Overlap

The KV-input worker and static DSpark capture now support the normal overlap
scheduler. A worker capability forces FutureMap to retain the CPU sequence-length
mirror needed by KV projection/capture, including when Triton itself opts out.
The existing ordered forward stream still governs source-slot lifetime.

Capture records the accepted model path in a bounded pending-token ledger until
CPU result handling confirms it. The first verify can precede the prefill result;
later verifies can precede the previous CPU commit. The observed anchor and
correct-draft prefix must match, and a later CPU token mismatch fails capture.
Final snapshots use only the scheduler's actual terminal boundary. Verify
lookahead wholly beyond Host capacity is ignored without invalidating the previous
pending result; the last in-capacity KV can be copied without an extra teacher row.

Evidence:

- `01790809576034299755-578443ced2d9`: 44 capture tests and 47 subtests passed
  in 20.03s. New cases cover delayed prefill/verify confirmation, full-accept
  lookahead beyond capacity, and disagreement with a later CPU token.
- `01790809969643408146-8478ad9541be`: full runtime test passed in 297.873s.
  All 62 snapshots pass readback: six ordinary AR, 18 overlap AR, 18 synchronous
  DSpark and 20 overlap DSpark. Each speculative overlap mode publishes ten
  completed samples, records 33 capture verify forwards and 35 commit copies;
  graph mode confirms 33 actual replays. Raw top-128 scores and selected KV match
  online reference tensors exactly; CPU full-vocabulary LSE matches within
  `rtol=1e-6, atol=1e-6`. Greedy baselines, mixed batch acceptance, prefix hits,
  chunked prefill, EOS, stop tokens, constrained grammar and exact 256-token
  capacity pass with delayed CPU results.
- `01790810305466767001-49cef3387214`: extended runtime test passed in 299.218s.
  All 64 snapshots pass exact readback. Each speculative overlap mode publishes
  eleven samples, executes 36 capture verify forwards / 42 per-request commit
  copies including its abort, and reaches fenced `FAILED` for that aborted stream
  without publishing it. Graph mode confirms 36 actual replays, including a
  three-request batch padded to four. Both AR and DSpark overlap modes wait for
  reservation recycling with zero quarantined Host slots after cancellation;
  all four aborts pass. Completed speculative snapshots are re-read after each
  producer exits. The original AR and synchronous DSpark cases still pass.
- `01790810305621302391-128481efabad`: focused regression passed with 117 tests
  and 78 subtests in 22.14s, including legacy DSpark, graph helpers, request IPC,
  capture ownership and target-KV contracts. Capture code/tests pass Ruff F/I;
  runtime observers/helpers pass full Ruff. The touched scheduler and runtime
  test retain their pre-existing lint diagnostics without adding any, and
  `git diff --check` passes. The original worktree's staged diff hash remains
  unchanged. The resident worker resumed its idle CUDA load after both jobs.

The observer reconstructs pending token history from actual verify inputs and
model acceptance output, independently of capture tickets. Its synchronous
reads do not establish performance or prove arbitrary asynchronous timing.
The separate CUDA event ownership test remains the copy-lifetime evidence.
Synthetic draft weights, Catalog test doubles, TCP transport and the Qwen3-0.6B
single-H100 scope are unchanged.

## Fixed-Input Training/Serving Gate

The SGLang-side P8 gate now loads actual SpecForge backbone/Markov modules from
commit `e10ea2fa3c248a4f60d636791dd71efb67338c9c1`, with a test-only shared KV
encoder adapter. It compares every decoder layer, final hidden state, shared-head
logits and Markov-corrected logits against actual SGLang context writes and Triton
paged attention. The teacher/fixture digests are checked, target decoder execution
is forbidden, and only saved KV/teacher tensors plus frozen shared weights are
used. The windows cover the first response label and partial final blocks.

This found and fixed a real checkpoint-loading defect: `markov_head.gate_proj`
was interpreted as an MLP shard. Strict loading now resolves an exact parameter
name first. Missing/duplicate/foreign-weight checks remain active.

The gate records every failing stage, nonfinite values, artifact/source hashes,
backends and runtime versions in `validation/parity.json`. A failed rerun cannot
leave an older successful report. It is an explicit command, independent of the
capture runtime test; installing SpecForge does not silently change that test.
Small-model cases cover vanilla/gated/RNN heads, cached CE+TV128 backward, frozen
shared modules, an optimizer step and export/reload, future-input isolation,
and stale-report invalidation. These adapters are not a production SpecForge
exporter, objective implementation or Catalog consumer.

Completed H100 evidence:

- `01790813285864674139-350dc423edd1`: 122 tests and 81 subtests passed in
  24.89s, including the new numerical/gradient cases and existing capture,
  DSpark, graph and IPC regressions.
- `01790813123946597057-bc93902f0b3e`: the complete online runtime test passed
  in 297.940s. All 64 snapshots pass exact readback, including 22 speculative
  overlap samples, padded graphs and four fenced aborts with no quarantined
  Host slots. Catalog is still a test double and transport is TCP.
- `01790813375946155985-82bf010d8a6a`: the real Qwen3 BF16 fixed-input gate
  **failed** at the unchanged `rtol=0.03, atol=0.03`. Layer 1 has 28/9216
  mismatches with maximum absolute error 0.078125; hidden-state maximum error
  is 0.25, and base/corrected logits reach 0.28125. The retained report is
  [target-kv-parity-bf16-failure.json](experiments/target-kv-parity-bf16-failure.json).
  Its checkpoint and snapshot remain at
  `/gpfs/users/fuxuanwei-1/dspark-maas-lab/fixtures/qwen3-target-kv-parity-20261001`.
- Earlier SDPA (`01790812483883726876-4e3d2b551f95`) and flex-attention
  (`01790812484035034606-2117203d933e`) references also failed on the same
  artifact with the original response-only anchor selection. Switching the
  training attention backend alone did not resolve the difference.
- `01790813286002064725-3024f10ef03f`: the committed FP32/native-auxiliary
  diagnostic passed on anchors `[159,160,162]`, with seven valid labels and
  nonzero encoder/backbone/Markov gradients. Maximum errors were 0.004612 in
  layer 1, 0.021037 in final hidden states and 0.018180 in logits. It retains
  paged attention but substitutes native norm/activation/RoPE because the
  production norm kernel does not support FP32. It is not BF16 serving evidence
  and does not write a serving parity certificate.

Source review and controlled norm/residual/RoPE/activation substitutions identify
BF16 rounding differences that explain part of the discrepancy, but do not yet
establish a complete fix. The real BF16 gate remains open. No tolerances were
relaxed and no speculative numerical-kernel change was applied. The synthetic
draft's forced proposals and shallow copied target layers are not trained-model
quality evidence. Commands and scope are in [the experiment guide](experiments/README.md).

Further numerical isolation, still without a BF16 fix:

- `01790813884844574921-b20922f878c7`: the serving outputs are bit-identical
  between one request at a time and the combined three-request batch on this
  fixture. Batch shape on the serving side does not explain the current error.
- `01790814042025040144-658501bff5d3`: same-input QKV, context-K and gate/up
  projections, and the RoPE cosine/sine tables match exactly. HF versus serving
  Q/K norm and RoPE outputs differ at BF16 rounding boundaries. Injecting the
  reference's exact Q/K/V into the real paged-attention backend leaves maximum
  attention-output differences of 0.0078125 in both layers versus SDPA. These
  are diagnostic observations, not a passed backbone/logit gate.

## Adaptive Capture Admission

An optional `adaptive` capture-config object now enables local Host-pressure
feedback. The configured `sample_ratio` is always the upper bound. High occupancy
halves the target ratio at most once per interval; low occupancy restores it in
10%-of-ceiling increments, with a hold region between thresholds. Active and
writing reservations and quarantined slots count as occupied; available leases
do not. A stalled queued/write task or Catalog/writer failure pauses admission
until cooldown. Repeated faults extend that pause while existing lease renewals
and writer work continue. New lease reservation retries respect the cooldown.

The controller changes new-request admission only. It adds no inference-thread
network operation or CUDA synchronization and cannot recycle uncertain buffers.
Configured zero sampling and permanent capture-disable reasons remain effective.
Internal-state metrics expose the target/effective ratio, control reason,
observed occupancy, writer age, cooldown and adjustment counters. Default fixed
sampling remains unchanged. Configuration and the exact policy are documented
in [the producer guide](../python/sglang/srt/training_capture/README.md).

Evidence on the resident H100:

- `01790815223606883581-23f445232546`: 130 tests and 89 subtests passed in
  27.23s. New tests cover interval limits, watermark hysteresis, zero sampling,
  cooldown extension, ratio recovery, Catalog failure/retry, a blocked background
  writer, and preservation of quarantine after an uncertain transport failure.
  Existing capture, DSpark parity, graph and IPC regressions still pass.
- `01790814750138717560-19563d4cd671`: the expanded real runtime test passed
  in 333.124s. Its original 64 snapshots still pass, and the adaptive overlap
  server adds two validated Mooncake snapshots. The first writer is deliberately
  paused before Store I/O. With one spare Host slot still available, a second
  generation completes with identical greedy output but is sampled out by the
  controller. Releasing the writer publishes the original sample; the ratio
  returns to 1.0 and the next request is captured. Final state reports two READY
  samples, one adaptive exclusion, two available reservations, an empty queue
  and zero quarantined slots. This tests a controlled writer stall with real
  serving and Store transport, not an actual RDMA/network outage.

New controller/helper files pass full Ruff; touched capture code/tests pass
Ruff F/I and `git diff --check`. Pre-existing broad-catch/style diagnostics were
left in place. The original worktree's staged diff hash is unchanged.

This implements P10's initial automatic pressure limit and local observability.
TTFT/TPOT-aware feedback, capture-on/off SLO
benchmarks and production rollout remain open. No production SpecForge Catalog,
cross-node RDMA or trained-draft quality claim follows from this controller.

## Capture Prometheus Metrics

Capture now integrates with the existing HTTP `/metrics` endpoint when both
capture and service metrics are enabled. It inherits scheduler model/rank/extra
labels. A separate one-second background sampler exports bounded event counters,
admission ratios/actions, reservation and Host-slot gauges, allocated bytes,
queue depth, occupancy, writer age, cooldown, disabled state and update time.
No per-request Prometheus operation, network call or CUDA synchronization is
added. Fixed sampling also exposes current occupancy and writer age. The worker
continues sampling during a blocked Catalog or writer and retries export errors.

Event labels group detailed capture failures, writer exception types and
unsupported request features into fixed categories. No request IDs, sample keys
or failure text enter labels. Repeated snapshots only increment counter deltas,
and absent reservation states are explicitly zeroed. The existing multiprocess
`mostrecent` gauge convention is retained; freshness and endpoint `up` must be
checked alongside gauges after process failures. Counter semantics and limits
are documented in the producer guide.

The associated Grafana dashboard is provisioned by the existing monitoring
example. It includes admission, events, reservations, Host slots, pending-writer
age, disabled state, freshness, Host capacity and existing TTFT/inter-token
histograms. It does not define alerts or claim an SLO. JSON structure has been
validated; rendering in a running Grafana has not yet been verified.

The new sampling test exposed a short-cooldown observation edge: the cooldown
could expire between pressure polls while a known writer stall remained. The
controller now keeps its effective ratio at zero until a fresh observation clears
that stall. Actual request admission still observes current pressure first.

Completed H100 evidence:

- `01790816480474825947-f39eb4d53d0e`: 145 tests and 89 subtests passed in
  31.73s. Coverage includes capture, admission/metrics, scheduler observability,
  DSpark, graph and request IPC regressions. New cases verify stable counter
  totals across repeated updates, bounded labels under hundreds of distinct
  failure names, inactive-state reset, quarantine reporting, short-cooldown stall
  persistence and successful metrics retry while Catalog admission is blocked.
- `01790816410979032574-67710e94c0a0`: full real runtime test passed in
  333.775s. All 66 snapshots pass the existing capture/readback checks. The
  adaptive server's real HTTP multiprocess endpoint reports zero effective ratio
  while its writer is held, two available reservations and ratio 1.0 after
  recovery, two READY events, one adaptive exclusion, zero writer age and zero
  quarantined slots. Missing-state gauges return to zero. The test still uses a
  Catalog double, TCP and a controlled writer stall before Store I/O.

New controller/metrics/helper tests pass full Ruff; touched capture, worker and
coordinator tests pass Ruff F/I. The scheduler's two pre-existing duplicate-import
diagnostics are unchanged, confirmed against HEAD. `git diff --check` passes.
The dashboard JSON has unique panel IDs and valid data-source/target structure;
Grafana rendering and Prometheus query-engine acceptance have not been tested.
The original worktree's staged
diff hash is unchanged. These tests do not certify service SLOs or production
SpecForge integration.

## KV-Draft Normalization Semantics

The KV-input draft now follows the default SpecForge Qwen3 normalization
boundaries: residual addition rounds in the activation dtype before the FP32
variance calculation, and normalized activations round before norm-weight
multiplication. The existing HF-cast RMSNorm kernel is reused after an explicit
narrow residual addition. The encoder's separately specified FP32-weight norm
is unchanged. Checkpoint parameter names and shapes are unchanged.

The fused Q/K norm+RoPE path, fused context-KV writer and stacked context
projection now honor the before-weight cast. Their new option defaults to the
legacy behavior. Inconsistent policies between Q/K or context layers fall back
to individual operations. The runtime projection observer independently checks
the corrected normalization definition.

Numerical isolation on the retained Qwen3 fixture:

- `01790817129609730345-9664e885c7ab`: before the fix, the SDPA reference
  comparison has 32 mismatches in layer 1, 442 in hidden states and 94,993 in
  corrected logits. Logit maximum absolute error is 0.3125. Replacing auxiliary
  arithmetic with the reference's exact rounding still leaves Triton attention
  differences: corrected-logit maximum error 0.137207, with 31,040 mismatches.
  Recomputing attention using the reference SDPA on the serving path's actual
  paged KV contents makes all decoder layers, hidden states and logits bit-exact
  for these three anchors. This isolates numerical differences on this fixture;
  it does not certify the unmodified serving backend.
- `01790817632178834033-0ba38279548f`: 37 tests and 42 subtests pass in 39.78s.
  New direct-kernel tests pin normalization rounding, untouched V columns,
  strided QKV inputs and masked context writes. Stacked/per-layer parity covers
  both BF16 and FP16 norm modes; mixed layer policies correctly reject fusion.
- `01790817791869330693-1ed205f56281`: the actual BF16 serving gate **still
  fails** at `rtol=0.03, atol=0.03` after the normalization fix. Layer 1 has six
  mismatches, hidden states have 266, and corrected logits have 78,083, with
  maximum logit error 0.21875. The report is retained in
  [target-kv-parity-bf16-norm-failure.json](experiments/target-kv-parity-bf16-norm-failure.json).
  This report uses SDPA, while the earlier committed baseline report used eager;
  the comparable pre-fix SDPA numbers are from the isolation run above.

The reproducible BF16 diagnostic is now in
`experiments/diagnose_target_kv_parity_bf16.py`. It prints production, reference
auxiliary and reference auxiliary-plus-attention comparisons without writing a
serving certificate. It retains real pool writes/reads, blocks target decoder
execution, and does not measure production latency or verify gradients.

Further completed validation:

- `01790817792230398995-aefa32c204fe`: full runtime test passes in 335.755s,
  including all 66 snapshots, ordinary/static-speculative overlap, padded graphs,
  fenced aborts, adaptive admission and real HTTP Prometheus checks. The revised
  projection observer independently validates the new K-normalization boundary.
  Catalog remains a test double and Store transport remains TCP.
- `01790817911067030073-3988949cc5f4`: the expanded regression passes with
  150 tests and 99 subtests in 31.87s, covering capture, observability, IPC, graph
  helpers, legacy DSpark, KV-draft normalization and the small SpecForge fixture.
- `01790817911390127075-2e01bc05c440`: the committed BF16 diagnostic reproduces
  the real gate's remaining production differences and bit-exact agreement only
  after the auxiliary and attention substitutions. The existing failed
  `validation/parity.json` hash is unchanged by the diagnostic.

New model/test/diagnostic code passes full Ruff, and `git diff --check` passes.
Unchanged import-order diagnostics in the existing DFlash/DSpark helpers and
stacked test were verified against HEAD; unrelated formatting was preserved.

P8 remains open: normalization agreement alone does not establish full BF16
backbone/logit agreement. RoPE, activation and attention arithmetic require
further work, with existing tolerances unchanged.

## KV Draft RoPE and Activation Rounding

The KV-input attention class now rounds table cosine/sine and each RoPE product
to the activation dtype, then performs the split-half addition. BF16 fused Q/K
and context-write kernels share this rule, and stacked/individual projections
honor it. Mixed layer RoPE policies reject fusion. The MLP uses the existing
native SiLU followed by multiplication to preserve its narrow activation
boundary. Shared rotary objects and legacy hidden-input policies are unchanged.
The draft requires full split-half table RoPE; the target-KV feature codec's
support for interleaved/partial target RoPE is separate.

The first nonzero-position test run (`01790819159386743848-c81b2d50a39e`)
caught contraction of intermediate products despite explicit BF16 casts.
Disabling FP fusion only for the new rounding mode fixed the mismatch.
`01790819245904207110-1917ebd413f5` then passed 41 tests and 52 subtests in
15.61s. Coverage includes positions through 4095, padded QKV row strides,
untouched V columns, masked multi-layer pool writes, changed inputs/positions
under CUDA graph replay, native SiLU rounding, and both FP16/BF16 stacked paths.

`01790819282208713198-14512f7efda7` reran the unchanged actual BF16 gate with
SDPA reference and `rtol=0.03, atol=0.03`. Both decoder layers now pass. Hidden
states still have 68 failing values; corrected logits have 31,040 failing values
out of 1,367,424, with maximum error 0.13720703125. This improves on the previous
78,083 corrected-logit failures and maximum error 0.21875, but the gate remains
**failed**. The report is preserved in
[target-kv-parity-bf16-rope-failure.json](experiments/target-kv-parity-bf16-rope-failure.json).
These results match the previous diagnostic's reference-auxiliary/Triton path;
attention rounding remains unresolved. SiLU's additional kernel boundary has
not been measured against the P10 latency budget.

Completed verification:

- `01790819282609396282-192f11c163c8`: the complete runtime test passes in
  335.809s, retaining all 66 snapshots, projected-KV checks, graph/overlap/abort
  coverage, adaptive capture and the HTTP metrics check. Store transport is TCP
  and Catalog remains a test double.
- `01790819397443438748-c7f043ce5b0e`: 155 tests and 109 subtests pass in
  33.98s. This adds an actual KV draft comparison of fused context/QK execution,
  stacked context execution, and individual context/QK fallback at the same
  unchanged tolerance, alongside the direct-kernel exact checks.
- `01790819397753800871-9fdec0d87649`: the independent BF16 diagnostic now
  reports identical stage statistics for production and substituted auxiliary
  arithmetic. Replacing attention using the actual serving pool still produces
  bit-exact layers, hidden states and logits on all three anchors. This remains
  diagnostic substitution, not a production serving certificate. The failed
  `validation/parity.json` hash stays
  `f3c75c2dd18cdf8429008eee15f77c3c98970c116bdc7f213a1c0f399e71fac2`.

New/owned model and test modules pass full Ruff; existing helper diagnostics
are unchanged from HEAD. `git diff --check` passes. All experiment jobs have
terminated and the resident H100 queue resumed its idle workload. P8, P9 and
P10 remain open with their original scope.

## Attention Backend and Partition Isolation

The retained BF16 fixture now has an attention-only diagnostic in
`experiments/diagnose_target_kv_attention.py`. It observes the real serving
Q/K/V after pool writes and compares independent math/SDPA implementations,
then substitutes each result into the complete serving backbone. Source KV is
read through the actual request mapping; FlashInfer probes copy it into their
own continuous or page-size-one layouts. These are numerical experiments, not
new production backends or performance measurements.

`01790820269727254473-251a360b53b3` establishes that PyTorch SDPA actually
selects `aten::_scaled_dot_product_cudnn_attention` in this environment
(cuDNN version 92000). The backend name `sdpa` alone is therefore insufficient
provenance. Using the same cuDNN path on the serving Q/K/V gives bit-exact full
backbone results on the three anchors. More accurate mathematical evaluation
does not reproduce that implementation's BF16 rounding:

| Substituted attention versus SDPA reference | Failing corrected logits | Maximum logit error |
| --- | ---: | ---: |
| Production Triton | 31,040 | 0.13720703125 |
| PyTorch cuDNN | 0 | 0 |
| PyTorch Flash | 15,551 | 0.15625 |
| PyTorch math | 27,792 | 0.15234375 |
| FP64 math | 27,548 | 0.15234375 |
| FlashInfer cuDNN, continuous KV | 2,237 | 0.125 |
| FlashInfer cuDNN, page-size-one KV | 38,941 | 0.140625 |

The pinned SpecForge DSpark offline examples use `flex_attention`, so that
training choice was also tested explicitly without overwriting the SDPA gate.
The default Triton path processes prefix and draft-block KV separately; the
existing deterministic/unified path processes one combined index sequence.
`01790821413528641824-f66d1834b514` confirms the following results against the
FlexAttention reference at the unchanged `rtol=0.03, atol=0.03`:

| Triton execution | Failing hidden values | Failing corrected logits | Maximum logit error |
| --- | ---: | ---: | ---: |
| Production split prefix/block | 56 | 24,668 | 0.125 |
| Existing unified path | 1 | 1,762 | 0.125 |
| Unified with experimental log2 softmax/dot accumulation | 1 | 1,652 | 0.09375 |

The log2 change did not pass, so it is retained only as
`experiments/target-kv-log2-probe.patch`, outside the production kernel. The
BF16 diagnostic accepts `--probe-log2` only when that patch is present. The
H100 copy was restored after the experiment. FlexAttention's short-query
implementation also partitions KV into independent reductions; aligning the
remaining partition/reduction behavior still requires work. Neither a different
training backend nor a smaller error count constitutes a passed gate.

The complete numerical records, this experiment's failed SDPA report, fixture/source
digests and attention-dump digest are retained in
[target-kv-attention-comparison.json](experiments/target-kv-attention-comparison.json).
The sampled Q/K/V dump remains in the H100 lab fixtures, outside Git.

Verification of the report changes:

- `01790820480684197104-93daf89dbb67`: five parity regression tests and three
  subtests pass in 14.20s, including failure-report invalidation and actual
  serving fused/stacked/individual path comparisons.
- `01790820481021818222-25739950b90e`: the actual SDPA gate retains its 31,040
  corrected-logit failures while recording CUDA/cuDNN versions and observed
  SDPA operators. Its report hash is
  `700201af267eca222b0e2aa572e72f7b86ce1409a91af95f44f51e277a30e02f`.
- `01790821324783856730-15e95c50251b`: the updated FlexAttention diagnostic
  runs successfully with the unmodified production kernel and reproduces the
  split/unified/reference-substitution comparisons. No serving certificate is
  written by a diagnostic.
- `01790821545474546786-5923c48b4090`: with the production kernel restored,
  `--probe-log2` exits with the expected CLI status 2 and explains the missing
  isolated experimental patch before opening a checkpoint.

Ruff, JSON validation and patch applicability checks pass. The original
worktree's staged-diff digest is unchanged; all experiments have terminated and
the resident H100 resumed its idle workload.

At that point the serving implementation remained at the previous numerical
policy. The subsequent logical-order kernel below resolves the retained fixture's
FlexAttention comparison. P8, P9 and P10 still need their remaining deliverables.

## Logical-Order KV Draft Attention

The generated reference kernels from
`01790822364547542737-f27603573da7` establish `BLOCK_M=16`, `BLOCK_N=64`,
two warps and one pipeline stage for the retained three-query/eight-KV-head
case. Although `SPLIT_KV=32`, the default sparse block size is `2**30`:
the first split contains all valid KV and the other splits are empty. Splitting
the actual context length into 32 pieces did not reproduce the reference.
With one logical KV stream and two warps, diagnostic substitution reached exact
equality (`01790822773838850981-1f3a5479ff01`).

The final production implementation is
`python/sglang/kernels/ops/speculative/dspark/target_kv_attention.py`, selected
only by `TargetKVAttention` through the normal Triton backend. A single kernel
reads existing prefix/block indices directly and preserves the logical 64-token
tiles across their boundary. There are no copied KV tensors, merged-index
allocations, split scratch buffers or reference-attention replacements.
It validates tensor/index metadata and rejects unsupported attention modifiers.
Existing attention paths retain their selection.

`01790823641652995870-6ef7c685763a` runs the actual, unpatched serving gate on
the retained Qwen3-0.6B BF16 fixture. Both layers, final hidden, base logits and
all 1,367,424 corrected logits are **exactly equal** to the pinned SpecForge
FlexAttention reference; `rtol=0.03, atol=0.03` is unchanged. Cached CE+TV128
backpropagation has finite, nonzero encoder/backbone/Markov gradients and frozen
shared weights. Target decoder execution remains forbidden. The resulting
`validation/parity.json` hash is
`994fb020c41eb7db4fb2adc99764ccce64dc81af49cf12e383400c4c142fb0e1`.

`01790823641338941244-db46a88dfb98` passes 159 tests and 121 subtests in
38.08s. The four new kernel tests cover FP16/BF16/FP32, strided tensors,
non-contiguous slots, zero-length rows/KV, non-power-of-two GQA, dimensions
64/80/128/256, widths up to 64, KV length 8193, invalid metadata and CUDA graph
replay with modified lengths/slots/Q/V. Separate and combined slot indices give
bit-exact output across prefix lengths 0/63/64/65/159/160. Existing capture,
injection, parity, normalization, graph-helper and legacy DSpark tests also pass.

The first final runtime run (`01790823725746885359-973f1a4a8ef0`) failed
on post-producer-exit Mooncake readback with SDK `LEASE_EXPIRED (-707)`.
The log explicitly reports `lease_expired_before_data_transfer_completed`.
The runtime test had forced a 100ms read lease; the installed master's
`--helpfull` reports a 5000ms default. This is separate from the objects'
hard pins and Catalog retention. The runtime test now uses the default read
lease. Production read error handling is unchanged: it rejects the result
and retains the destination in quarantine. An explicit `-707` regression,
`01790824208583279715-b47abbbda7e1`, passes 7 tests / 4 subtests in 10.03s.

The final runtime rerun, `01790824208934763806-34c2884f502f`, passes in
333.872s with 66 completed snapshots: six synchronous AR, 18 overlap AR,
18 synchronous DSpark, 22 overlap DSpark and two adaptive-admission samples.
It includes ordinary/graph modes, exact online KV/teacher observations,
producer-exit readback, prefix reuse, batch padding, terminal truncation and
streamed cancellation. The test still uses a Catalog double and local TCP;
it adds no cross-node RDMA or production Catalog evidence.

`01790823797275492601-64120bb55cdf` also verifies the updated diagnostic:
production has zero error while the old two-stage/unified paths retain
24,668/1,762 failing corrected logits on the same FlexAttention inputs. The
diagnostic does not overwrite the production gate report.

`01790823641976918474-3c0d5b61ec3c` measures identical randomly paged BF16
inputs under CUDA graph replay, with 16 query heads, 8 KV heads and dimension
128. These are attention-only times on the resident H100:

| Batch / Prefix / Block | Old two-stage kernel (us) | Logical KV kernel (us) |
| --- | ---: | ---: |
| 1 / 160 / 3 | 14.27 | 9.38 |
| 4 / 160 / 3 | 14.60 | 9.48 |
| 1 / 1024 / 3 | 58.24 | 45.70 |
| 4 / 1024 / 16 | 57.63 | 51.07 |
| 1 / 8192 / 3 | 497.84 | 419.34 |
| 16 / 8192 / 16 | 1035.15 | 513.39 |

This does not establish end-to-end throughput, capture overhead or the P10 SLO.
The checkpoint still has synthetic encoder/Markov weights and copied target
layers. Exactness is demonstrated for this fixture/runtime/reference, not for
SDPA/cuDNN, arbitrary checkpoints or all FlexAttention kernel configurations.
Production exporter compatibility and trained-draft quality remain open.

The serving report, generated-code parameters/source digests, regression result
and raw timing measurements are retained in
[target-kv-flex-parity.json](experiments/target-kv-flex-parity.json).
New modules pass Ruff; the modified legacy files add no diagnostics relative
to HEAD. Formatting and `git diff --check` pass. All experiment jobs have
terminated and the resident H100 has resumed its idle workload. The original
worktree's staged-diff digest is unchanged.

## Scheduler Latency Feedback

Optional `adaptive.latency` now adds scheduler TTFT and committed-token-normalized
output-interval budgets to capture admission. The scheduler supplies observations
after processing results, including when there is no capture ticket. This lets
unsampled requests drive recovery after new capture is paused. Health checks,
aborted results, duplicate callbacks, rejected drafts and stop-truncated tails
do not add successful observations or inflate committed-token counts. Retraction
preserves prior output timing. The disabled configuration adds no latency monitor.

Each configured metric uses a bounded recent-observation buffer and nearest-rank
percentile. A sufficiently populated metric above its budget pauses new capture
using the existing decrease/cooldown policy. Existing work and generation
continue. Recovery requires fresh observations for all enabled metrics below a
separate lower threshold. Stale data cannot clear a breach or increase a reduced
ratio; healthy data cannot bypass cooldown. There is no CUDA synchronization or
client/network latency inference in this observer.

Prometheus now exposes bounded latency states, budgets, quantiles, observation
counts and recovery readiness. Missing observations are NaN, not successful
zero-latency measurements. The existing dashboard adds scheduler-budget and
protection-state panels. JSON parsing, unique panel IDs and grid bounds pass;
Grafana rendering/runtime acceptance is still open.

The final per-file checks pass 42 tests and 40 subtests:

| File | Job | Tests / Subtests |
| --- | --- | --- |
| `test_latency.py` | `01790826289374789054-5ff9e2eddca6` | 7 / 7 |
| `test_metrics.py` | `01790825544842875032-362600dbb0cb` | 4 / 0 |
| `test_coordinator.py` | `01790825545170019788-859329b9f88e` | 21 / 3 |
| `test_config.py` | `01790825545596690587-c88557088484` | 5 / 28 |
| `test_admission.py` | `01790825545933812277-20aac7428458` | 5 / 2 |

`01790825671730805923-8f31456badb2` passes the complete H100 runtime test in
376.703s, with 68 validated snapshots. An additional ordinary overlap server
uses test-only 750ms result-processing delays against 500ms TTFT/TPOT budgets.
Its first admitted request publishes despite the breach; the next request
continues generating without capture. After the delayed observations expire,
admission stays zero until an unsampled healthy request supplies fresh values.
A fourth request is captured after gradual recovery. All four output sequences
match, both snapshots survive producer exit, and HTTP metrics report four TTFT
observations. The final controller state records twelve TPOT intervals and an
effective ratio of one, with no quarantine.
The earlier AR/DSpark ordinary/graph/overlap checks also pass.

The HTTP Catalog double logs a disconnected-client `BrokenPipeError` during
teardown after these assertions. The authoritative worker result is completed
with exit code zero. The test still uses local TCP and a Catalog double.
The retained result, source digests and test scope are in
[`capture-latency-feedback.json`](experiments/capture-latency-feedback.json).

New code passes Ruff; diagnostics in modified legacy files match HEAD.
Formatting and whitespace checks pass. This implements P10's configurable
latency feedback, but it does not establish capture overhead or a service SLO:
capture-off baselines, input-distribution controls, measured rollout thresholds
and runtime dashboard acceptance remain required.

## Capture Serving Baseline And GPU Attribution

`benchmark_training_capture.py` now runs the existing streaming serving client
against normal SGLang execution, with an isolated real TCP Store and HTTP test
Catalog. It brackets capture phases with capture-off servers, reverses the
sampling order on the second round, excludes warmup from counters, and validates
all measured READY tensors after the producer exits. It records the actual
admitted/selected/READY counts alongside client metrics and source hashes.
Production execution has no added observer hooks or synchronous tensor copies.

H100 job `01790827387658540683-770722b0f254` completed ten phases, each with 2,048
requests of 128 input / 32 output tokens at concurrency eight. All 20,480 timed
requests completed, yielding 655,360 output tokens. Across the two rounds, 422
snapshots passed manifest, checksum and tensor-contract validation after producer
exit. Every selected request was admitted and published, with no backpressure or
quarantine. All phases report zero cached tokens. The target is Qwen3-0.6B BF16,
with layers 0/14/27, normal overlap and decode CUDA graphs. Adaptive admission
is disabled so that the configured sampling rates remain comparable.

| Sampling | Snapshots Per Round | Throughput Change, Rounds 0 / 1 | TPOT p95 Change, Rounds 0 / 1 |
| --- | ---: | ---: | ---: |
| 0.1% | 2 | +1.0% / -2.2% | -0.1% / +1.7% |
| 1% | 19 | -1.9% / -3.4% | +1.3% / +1.3% |
| 10% | 190 | -15.6% / -18.0% | +14.9% / +16.4% |

Each change uses that round's before/after baseline mean. Baseline throughput
means are 1,294.74 and 1,315.91 output tokens/s. This is seeded synthetic traffic,
not an SLO acceptance test: only two samples are admitted at 0.1%, both rounds
reuse the same seed, and round zero's baseline TTFT p99 after/before is 1.53.
The separate 32-request smoke run exercises 100% selection under pressure:
22 snapshots publish while ten requests are excluded by backpressure; generation
still completes for all 32 requests.

`profile_training_capture.py` separately profiles 128-to-1 and 1-to-32 workloads
using test-only CPU annotations around existing capture operations. Job
`01790827723119310326-05bdb014f266` completes both off/on pairs with 40 measured
READY samples per enabled workload. Its probes wait for reservations and are
not latency measurements. In the decode workload, capture's KV/teacher kernels
sum to 25.663/17.190ms and its 13,200 D2H operations sum to 34.300ms, transferring
17,587,680 bytes. The many small gathers/copies are a measured optimization
target. Overlap also stages one-ahead rows beyond the final publication prefix;
actual DMA bytes are not interchangeable with stored tensor sizes.

`summarize_capture_trace.py` attributes device events by CUDA launch correlation
and records trace hashes. It excludes same-name GPU annotations from CPU counts.
The regression reproduces double counting before the fix and passes afterward:
job `01790829371490948615-9b53bbfd6e21`, three tests and two subtests in 10.05s.
Missing DMA byte fields remain unknown, and ambiguous scope mappings fail.

The retained configuration, versions, counters, client metrics, readback sizes,
source/trace hashes and regression result are in
[`capture-serving-performance.json`](experiments/capture-serving-performance.json).
The source-backed profiler tables and rejected generic FP8 heuristic are in
[`capture-profile-analysis.md`](experiments/capture-profile-analysis.md).
The benchmark's post-client drain begins after the client process exits; it is
not response-to-READY latency. These local TCP results do not establish RDMA,
production Catalog retention, draft quality, rollout thresholds or a service SLO.

## Native KV Slot Indices

The baseline's extra `direct_copy_kernel_cuda` launches in KV export come from
`slots.to(dtype=torch.long)`, repeated for each selected K/V buffer. The real
request pool uses int32; PyTorch's native `index_select` accepts both int32 and
int64. CUDA correlation inspection confirms the launches are nested inside
`aten::to` / `aten::_to_copy`, rather than additional KV payload transfers.

`SelectedLayerKVExporter` now preserves either integer dtype, moves indices to
the source device once, and records their use on the producer stream. Independent
gather storage, pinned destinations and completion fencing remain in place.
Noninteger indices fail before a Host write; the old implicit cast silently
changed a fractional slot to a different integer slot. CPU and CUDA ownership
tests now exercise strided int32 indices while retaining int64 coverage.

Job `01790830007807467624-c9b51ad70713` passes 34 tests and seven subtests in
27.22s. A separate run of the new fractional-index regression against the old
exporter fails with `ContractError not raised`, confirming the prior truncation.
H100 runtime job `01790830132480654368-175a28a57d4b` passes in 377.459s with
68 snapshots, including AR/DSpark, ordinary/graph/overlap execution, abort,
prefix remapping, writer pressure and latency recovery. The tests continue to
compare captured tensors against actual forward observations and read Store
snapshots after producer exit.

Profiler job `01790830132909131039-373e5c2bdaac` completes both off/on workload
pairs, with 40 measured READY samples per enabled workload. Relative to the
retained baseline, the native index path removes every attributed index-conversion
kernel. The remaining KV kernel is `indexSelectSmallIndex` with int32 indices.

| Profile Workload | KV Kernel Launches, Before / After | KV GPU Work, Before / After | KV D2H Copies / Bytes, Both Runs |
| --- | ---: | ---: | ---: |
| 128-to-1 | 540 / 270 | 0.874 / 0.498ms | 270 / 62,976,000 |
| 1-to-32 | 15,840 / 7,920 | 25.663 / 15.294ms | 7,920 / 16,220,160 |

Decode-workload KV GPU work decreases 40.4%; its instrumented CPU scope time
decreases from 779.516ms to 527.025ms. These are profiler work measurements,
not serving latency. Teacher and position copy counts/bytes are also unchanged.
The 13,200 total capture D2H calls remain an optimization target.

The separate normal-serving rerun, job
`01790830133240918660-801405517ba5`, completes two off/10%-capture/off rounds:
12,288 timed requests, 393,216 generated output tokens and 380 validated
snapshots after producer exit. All selected requests are admitted and READY,
with no quarantine or cached prompt tokens. Its benchmark driver is unchanged.
Source-hash comparison finds only `kv_exporter.py` changed in the capture package.

| Round | Baseline Mean Output Tokens/s | 10% Capture Output Tokens/s | Throughput Loss | TPOT p95 Increase |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,314.20 | 1,092.64 | 16.9% | 14.6% |
| 1 | 1,311.60 | 1,081.31 | 17.6% | 15.3% |

There is no demonstrated end-to-end throughput improvement: the previous
10%-capture values were 1,092.40 and 1,079.01 tokens/s, with losses of 15.6%
and 18.0% against their own baselines. This change removes redundant GPU work;
it does not resolve overall capture overhead or satisfy P10's service SLO.
The complete evidence, source/trace hashes, before/after kernel counts and
benchmark metrics are retained in
[`capture-native-indices.json`](experiments/capture-native-indices.json).
Formatting and whitespace checks pass. Ruff diagnostics in the modified legacy
files match HEAD. All jobs have terminated and the H100 resumes its idle workload.

## Bounded Batched KV D2H

Capture now accepts opt-in `kv_d2h_batch_tokens` and `max_device_bytes` fields.
Each Host slot owns a bounded device arena for the selected K/V components.
Short ranges gather into that arena immediately; full batches and the sealed
tail copy to the existing registered Host tensors. Large prefill ranges retain
direct D2H. The default batch size is one and allocates no device staging.
The all-slot budget is validated before registration, and Host/device storage
share completion fencing, reuse and quarantine. Store objects and the training
tensor contract are unchanged. See [configuration and ownership](TARGET_KV_DRAFT.md#batched-kv-d2h).

Job `01790832852903241755-4e5417e86316` passes 96 tests and 72 subtests in
43.47s, including both direct and staged coordinator lifecycles. CUDA coverage
checks source mutation, full-batch arena reuse, cross-stream ordering and tail
finalization outside the producer stream. A failed completion event quarantines
both Host and device storage. Budget rejection happens before registration.
The final CUDA file, including the stronger retained-storage assertion, passes
all four tests in 10.42s in job `01790833555947065229-813302e83634`.

H100 runtime job `01790833043043329372-f95092d4a096` passes in 377.872s with
68 validated snapshots using 16-token staging. AR/DSpark, ordinary/graph/overlap
execution, abort, prefix remapping, writer pressure and latency recovery pass.
Independent forward observations still match captured KV and teacher values;
Store snapshots remain readable after producer exit. The four-slot fixture
allocates 786,432 device bytes within its 8MiB budget.

Profiler job `01790833043362056483-6e40fef82107` completes the same off/on
workloads as the native-index baseline, with all 40 measured requests READY in
each enabled workload. KV attribution sums `training_capture.kv` and the new
`training_capture.kv_d2h` flush scope.

| Profile Workload | KV D2H Copies, Before / After | KV D2H Work, Before / After | KV Bytes, Both Runs |
| --- | ---: | ---: | ---: |
| 128-to-1 | 270 / 270 | 2.028 / 2.050ms | 62,976,000 |
| 1-to-32 | 7,920 / 720 | 20.694 / 2.213ms | 16,220,160 |

Decode KV D2H calls decrease 90.9%; total capture D2H calls decrease from
13,200 to 6,000, with the same 17,587,680 bytes transferred. KV gather kernels
remain 7,920. The combined instrumented KV CPU scope time decreases from
527.025 to 319.731ms. Teacher and position copy counts and bytes are unchanged.
These are profiler work measurements, not client latency. DMA may include
lookahead teacher rows beyond the committed response; publication still follows
the accepted token and valid-KV boundaries, so DMA and stored sizes can differ.

Normal-serving benchmark job `01790833043677920824-4b7d751c335a` completes two
off/10%-capture/off rounds with 12,288 timed requests and 393,216 output tokens.
All 380 selected requests are admitted, published and read back after producer
exit. Both capture phases have 373,555,200 KV bytes and 6,700,160 auxiliary bytes,
matching the native-index baseline. There are no quarantined slots, Catalog
errors or cached prompt tokens. The sixteen-slot pool allocates 3MiB of staging
tensors within its explicit 16MiB budget.

| Round | Baseline Mean Output Tokens/s | 10% Capture Output Tokens/s | Throughput Loss | TPOT p95 Increase |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1,303.49 | 1,101.60 | 15.5% | 11.7% |
| 1 | 1,313.45 | 1,099.47 | 16.3% | 12.7% |

Capture throughput is 0.8% and 1.7% higher than the previous native-index runs.
Their losses against their own baselines were 16.9% and 17.6%, and TPOT p95
increases were 14.6% and 15.3%. These two synthetic rounds do not establish
statistical significance or satisfy P10's production SLO. Teacher and position
transfers, per-forward gathers and other capture work remain optimization
targets. The default remains direct D2H; batching requires explicit opt-in.

The retained tests, runtime observations, source/log/trace hashes, profiler
attribution and normal-serving results are in
[`capture-batched-kv-d2h.json`](experiments/capture-batched-kv-d2h.json).
Local and GPU source hashes match. New code passes Ruff, modified legacy
diagnostics match HEAD, and formatting/whitespace checks pass. All submitted
jobs have terminated and the resident H100 has resumed its idle workload.

## Real Cache Eviction And Retract/Resume

`training_capture_lifecycle_runtime.py` extends the registered runtime test with
two ordinary AR servers: synchronous eager execution and overlap with decode
CUDA graphs. Each uses a 256-token serving KV pool and the existing 16-token
capture staging. A repeated 160-token prompt first demonstrates a 159-token
cache hit. An unrelated prompt then forces actual RadixCache eviction; a later
repeat has zero cached tokens. HTTP eviction metrics and the observer's physical
KV slot indices confirm both eviction and reuse.

After the first streaming response token, the test calls the public
`/pause_generation` endpoint with `mode=retract`. It waits for the Catalog's
fenced capture failure before calling `/continue_generation`. The final response
must report at least one retraction and complete all 128 requested output tokens.
Its capture must never become READY or be admitted again after resume. A fresh
request then publishes normally. All five successful snapshots in each mode
retain exact raw KV and teacher values, token/position alignment and loss masks
against the online observer, including snapshots whose serving slots were reused.
They are read again after the producer exits.

| Execution | Tokens Evicted By Replacement | Original KV Slots Reused | Resumed Output Tokens | READY Samples | Failed Retracted Capture |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous eager | 164 | 70 | 128 | 5 | 1 |
| Overlap + decode graphs | 166 | 72 | 128 | 5 | 1 |

Job `01790835514030791687-52c6d1f92435` completes the full runtime test in
444.223s with exit code zero. Its 78 validated snapshots include the previous
68 AR/DSpark/admission cases and these ten lifecycle samples. Neither new mode
quarantines a slot. The graph mode observes 22 capture forwards using actual
CUDA graph replay. `/metrics` reports the same staging allocation (786,432 bytes)
and budget (8,388,608 bytes) as `/server_info` in both modes. The registered
estimate is now 480s, and the independent test Store segment is 256MiB to retain
the expanded snapshot set.

This change adds runtime coverage; no production capture behavior changes were
needed. New code passes Ruff, modified legacy diagnostics match HEAD, and
formatting/whitespace checks pass. Source hashes match the GPU checkout. The job
has terminated and the H100 has resumed its idle workload. Results, source/log
hashes and observed counters are retained in
[`capture-cache-lifecycle.json`](experiments/capture-cache-lifecycle.json).

The retraction trigger here is the public pause API. Automatic OOM retraction,
speculative retraction/cache eviction, concurrent reader retention, TP/PP/PD and
cross-node RDMA still need their own acceptance evidence. These correctness
observers synchronize/copy tensors and do not measure serving overhead or SLOs.

## Static DSpark Automatic Retraction

The DSpark runtime fixture now uses a 512-token serving KV pool and a 0.05
scheduling conservativeness setting. Four disjoint 16-token prompts each request
192 output tokens, exceeding that pool. The debug retract flag is explicitly
disabled. Actual pool exhaustion causes the scheduler to retract two requests
in each synchronous/overlap and eager/CUDA graph combination.

All four requests still complete their 192-token responses. HTTP retraction
metrics agree with the per-request counts. Each retracted request retires its
capture exactly once and never publishes or starts a second capture after
resume. The two surviving requests publish normally; a fresh four-token request
then confirms that capture capacity has recovered. All four reservations are
available afterward, with zero quarantined slots.

Test-only observations around `TargetKVInjector.ensure_context` verify that
each resumed request has no previous draft projection and reconstructs through
its current target prefix. The two reconstructed prefixes contain 125 and 144
tokens in every tested mode. The existing independent encoder/projection/RoPE
observer also checks the actual draft KV written during reconstruction.
Successful samples retain exact token paths, raw selected KV and top-128
values, independently checked top-128 IDs/LSE, positions and response masks.
All twelve new samples remain readable from Mooncake after producer exit.

| Execution | Completed Pressure Requests | Output Tokens | Automatic Retractions | Rebuilt Draft Contexts | READY Including Fresh Request |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synchronous eager | 4 | 768 | 2 | 2 | 3 |
| Synchronous graphs | 4 | 768 | 2 | 2 | 3 |
| Overlap eager | 4 | 768 | 2 | 2 | 3 |
| Overlap graphs | 4 | 768 | 2 | 2 | 3 |

Job `01790836944459482077-87511189724e` passes the full registered runtime
test in 461.341s. Its 90 snapshots comprise the previous 78 cases plus these
twelve pressure/recovery samples. The existing AR eviction/retract cases also
pass again. The 480s registered estimate remains appropriate. No production
code changes were needed for this coverage.

Source hashes match the GPU checkout, Ruff and formatting/whitespace checks
pass, and the resident H100 has resumed its idle workload. Retained counters,
actual OOM log messages and source/log hashes are in
[`capture-dspark-memory-pressure.json`](experiments/capture-dspark-memory-pressure.json).
This is a TP1/TCP correctness test with a synthetic draft and forced outputs;
it does not establish trained-model quality, serving SLOs, non-static verify
correctness or cross-node behavior.

## Draft Checkpoint Tensor Validation

The KV-input loader now validates every exported tensor before invoking the
mutating parent loader. Packed parameters must match their full shape, while
split Q/K/V and gate/up tensors must match the configured projection sizes.
Only dense FP32/FP16/BF16 tensors are accepted. A min/max reduction detects
NaN/Inf and checks representability after conversion to the destination dtype;
only the two extrema are converted, avoiding a full converted weight or
per-element boolean allocation.

This closes observed defects in checkpoint admission. With the old production
code, job `01790838342711980754-a0a5d065e9ff` fails ten new regression
subcases: the generic parallel loaders silently truncate extra rows from
packed/split exports, accept integer/complex/nonfinite values, and allow FP32
values to overflow the FP16 destination. A malformed late tensor reaches the
mutating loader before failing. The repaired path rejects these inputs before
any parameter or projection-cache change.

The tests use actual QKV and merged-MLP loaders with GQA geometry, checking
split-to-packed layout against independently specified row order. Existing
missing/duplicate/foreign-weight checks and Markov gate handling remain covered.
Job `01790838420199480428-2bb5c92e8be4` passes 25 tests and 42 subtests in
14.11s, including all three Markov heads, cached CE+TV128 gradients, an optimizer
step and fresh serving export/reload. After adding successful cross-dtype
coverage, the final unit run `01790838598515310920-4aa144bbb44b` passes
20 tests and 42 subtests in 11.13s.

Job `01790838509115541407-3e7fbe13b6b3` also loads the retained Qwen3
snapshot/draft artifact through the new validation. BF16 fixed-input comparison
against pinned SpecForge FlexAttention passes with zero differences at both
backbone layers, hidden output, base logits and Markov-corrected logits.
No teacher decoder is run. Source and log hashes, the negative baseline and
the complete retained-fixture report are recorded in
[`target-kv-checkpoint-validation.json`](experiments/target-kv-checkpoint-validation.json).

Only `DSparkTargetKVDraftModel` production behavior changes. The existing
TP1/unquantized topology gates remain in force. This validates tensor admission,
not transactional rollback after a hardware copy failure, trained draft quality
or release readiness. SpecForge's production KV-input exporter is still missing;
its current specialized SGLang exporter supports EAGLE3. The full HTTP capture
runtime was not repeated for this loader-only change. Ruff/format/whitespace
checks pass, GPU source hashes match, and all jobs have terminated.

## Owner-Local Store Publication

`SnapshotWriter.write_partition` now writes exactly one owner's registered
Host tensors under a complete immutable manifest. It validates logical coverage,
local object identity/checksums/finite values, and aux semantics on the aux owner.
It returns `OwnerWriteReceipt` only after Store completion and Catalog WRITTEN
acknowledgement. The receipt binds capture ID, fencing token, owner and the
entire manifest digest.

`publish_partitions` requires exactly one matching receipt per expected owner.
Missing, duplicate, stale, foreign-owner or changed-manifest receipts are rejected
before publication work. The aux coordinator registers the manifest, persists
the existing metadata journal, seals against Catalog WRITTEN descriptors, and
publishes the manifest last. The ordinary recovery path works without original
owner buffers. No new Catalog endpoint or tensor wire format is introduced.

Job `01790839497863279291-3e45cdeddf54` passes 102 capture tests and
80 subtests in 43.25s. After the final owner guard and fixture namespace changes,
job `01790839663116165235-a41cdbec2704` passes the affected writer tests
and both real Store integration cases: 15 tests and ten subtests in 54.36s.
Failures cover incomplete/foreign local payloads, invalid response masks,
receipt mismatches, lost WRITTEN responses and lost seal responses.

The new Store case starts two independent owner processes with disjoint
registered tensor sets. The logical TP2 fixture assigns aux to tp1, reverses
owner/descriptor order and splits each selected layer's two KV heads. Both
writers exit before the coordinator publishes. A separate 64MiB storage segment
retains all 32 tensor objects (4,532 bytes), and a fresh reader process verifies
the published sample. The test reader now reconstructs by logical token/head
ranges; exact comparison against the original full-head tensors passes.
The previous single-owner registered-arena transport test passes too.

This is a real owner-local TCP publication interface, not TP2 model inference.
The coordinator that gathers rank-local descriptors, synchronized admission,
canonical ownership of replicated KV heads, rank-local Host pools, scheduler
integration and TP/PP numerical/runtime validation are still required. Serving
capability gates remain unchanged. Receipts are trusted producer ACKs, not
Catalog retention guarantees; seal must independently check exact WRITTEN
descriptors. The production Catalog and cross-node RDMA remain outside this run.

Sources match the GPU checkout. New code passes Ruff, existing diagnostics match
HEAD, and format/whitespace checks pass. All jobs terminated and the H100 resumed
its idle workload. Commands, observations and source/log hashes are retained in
[`capture-partition-publication.json`](experiments/capture-partition-publication.json).

## Canonical KV Ownership and Local Snapshot Preparation

`plan_capture_layout` maps global selected-layer geometry onto explicit ordinary
dense TP/PP stages. It uses the native QKV loader's head placement, selects the
first rank of each replicated-head group as canonical owner, and assigns aux
tensors to a designated rank on the final PP stage. Inactive ranks remain
explicit partitions rather than falling back to full-model allocation.

`HostBufferPool` now accepts a partition and reserves only that rank's selected
KV layers and local heads. Only the aux owner reserves tokens, teacher values,
masks and manifest capacity. KV staging budgets remain independent per rank;
an aux-only owner needs no device staging. `SelectedLayerKVExporter.from_pool`
reads only owned PP layers and validates rank-local head shapes. No KV tensor
gather is introduced.

After local D2H completion, `prepare_snapshot_partition` produces registered
views and descriptors bound to common snapshot metadata. `assemble_snapshot`
validates the complete owner set, matching metadata, consistent KV validity and
canonical head ranges before constructing one manifest. Swapping two owners'
head labels is rejected even when global coverage remains complete. The
existing single-owner `build_snapshot` uses the same preparation/assembly path.

Job `01790840979211010701-7b614e301c6b` passes 111 tests and 104 subtests
in 89.09s, including both real Mooncake Store tests. Ownership is independently
checked against loaded K weights from `QKVParallelLinear`, at TP1/2/4/8 for
layers with eight and two KV heads. A TP4/PP3 fixture checks heterogeneous heads,
inactive replicas and an aux-only final stage. Its KV-only rank allocates exactly
256 Host bytes and 96 device-staging bytes; lowering either budget by one byte
fails before registration. CUDA batching also preserves exact local KV after
immediate source-slot reuse, including the final partial batch.

The two independent Store writers now use a fixture prepared through these
production APIs. Their 32 objects / 4,532 tensor bytes reconstruct exactly by
logical ranges, and a fresh reader succeeds after both writers exit. Descriptors
are still assembled in the test coordinator; this is not distributed model
execution or cross-rank metadata transport.

Since ordinary `build_snapshot` now shares this implementation, job
`01790841385378621132-77011d7c17ca` reruns the complete existing single-H100
serving test and passes in 462.301s, with 90 READY snapshots. AR and static
DSpark, eager/graph replay, overlap, adaptive and latency protection, RadixCache
eviction and retract/resume remain covered. All four DSpark pressure modes
complete 768 output tokens across four requests despite two automatic
retractions per mode. Successful samples remain readable after producer exit.

Serving TP/PP/DP gates remain closed. Global identity binding from local model
instances, distributed admission/failure agreement, descriptor exchange, global
teacher top-128/LSE and scheduler integration remain open. Inactive ranks will
still need to participate in collective/control ordering. No CP/sparse KV,
cross-node RDMA, production Catalog or performance SLO is certified by these
tests.

Source and log hashes, commands and observations are retained in
[`capture-topology-ownership.json`](experiments/capture-topology-ownership.json).
GPU sources match the local checkout. New code passes Ruff; the exporter's
existing `UP035` diagnostic is unchanged. Format and whitespace checks pass,
the original worktree index is unchanged, both jobs have terminated, and the
resident H100 has resumed its idle workload.

## Global Target Contract from Local Ranks

The previous identity binder treated `attention.num_kv_heads` as global and
accessed every selected layer on one model instance. That was valid under the
single-rank serving gate, but could not supply the global geometry required by
distributed capture. `bind_rank_target_contract` now reads native QKV projection
metadata and only the selected layers local to the actual PP stage. It checks
TP placement, projection geometry and actual K/V source shapes/dtypes. Stages
without selected layers still validate their first local projection. The existing
artifact hashing, tokenizer checks, resolved configuration and output-transform
fingerprint are preserved.

`RankTargetContract` is strict serializable metadata. `assemble_target_contract`
requires one record from every TP/PP rank, including replicated-head ranks with
no payload ownership. It checks common teacher identity, global selected-layer
order, storage parameters, complete PP intervals, equal replicated geometry and
RoPE/norm semantics. Every observed logical head range must match the native
canonical placement. It returns the global teacher, global KV spec and layout
for rank-local pools/exporters. Single-rank capture and the DSpark target-KV
injector now use the same binding/assembly path.

The unit fixture uses real `QKVParallelLinear` and `RotaryEmbedding` instances,
mock model/PP/pool containers, and temporary synthetic artifact files. TP4/PP3
includes eight-head and two-head selected layers, inactive KV replicas and an
aux-only final stage. JSON roundtrip and reversed record order preserve exactly
the teacher/KV contract obtained from its complete single-rank model fixture.
Negative cases reject missing inactive ranks, duplicate/foreign ranks, changed
artifacts/output transforms, codec disagreement, wrong source shapes, wrong TP
placement and PP gaps or inconsistent bounds.

The initial broad run `01790842667772369764-94d14d07fdc8` passed 129 existing
tests and 146 subtests, but the six new tests failed during RoPE fixture setup:
the runtime execution namespace was unpublished. A scoped test-only execution
config patch fixes the fixture; production logic did not change. Focused job
`01790842813066987707-1db3ea37132c` then passes all six tests and 31 subtests
in 10.55s. The fixture cleanup was subsequently made Python 3.10-compatible.
Final job `01790842897169950408-7e882fad4b34` passes the same six tests and
31 subtests in 10.28s with that cleanup.

Because both capture and DSpark binding changed, full runtime job
`01790842896845885942-174286d18b31` reruns real Qwen3 prefill/decode and
passes in 464.049s, publishing 90 snapshots through Mooncake. Ordinary/graph
and overlap capture, all four static DSpark memory-pressure modes, adaptive
admission, scheduler latency protection and cache lifecycle scenarios pass.
This exercises actual single-rank model/pool inspection and the draft injector,
with successful Store readback after producer exit.

This is startup metadata binding, not TP/PP model execution. Cross-rank record
exchange, consistent startup failure handling, distributed admission, teacher
top-128/LSE and scheduler integration remain required. The serving TP/PP/DP gates
remain closed, and live target weight replacement remains a separate lifecycle
requirement. Local artifact hashes assume immutable artifacts matching the loaded
model; they do not hash live device parameter contents.

Commands, the initial fixture failure, final results and source/log hashes are
retained in [`capture-global-identity.json`](experiments/capture-global-identity.json).
GPU and local source hashes match. Ruff, formatting, whitespace and host Python
3.10 syntax checks pass. The original worktree index remains unchanged; all four
jobs are terminal and the H100 has resumed its idle workload.

## Collective Startup Identity Agreement

`coordinate_target_startup` now exchanges rank identity through the serving
replica's existing Gloo CPU group, using fixed control headers and bounded JSON
byte tensors. The protocol checks PP-major/TP-minor source rank identity and
declared topology before calling the global contract assembler. It sends no
pickled objects, model tensor contents or exception text.

All ranks vote after local binding, after payload allocation and after global
validation. The final vote includes a SHA-256 digest of teacher/KV/layout; no
rank returns a contract on partial validation success or differing output.
Records are capped at 1MiB and padded receive capacity at 64MiB per rank. Control
buffers are allocated before binding so a failed payload allocation can still
participate in the failure vote. Collective waits default to 120 seconds each;
transport failure requires group/worker teardown rather than retry.

`TpModelWorker.init_training_capture` supplies the existing world CPU group and
actual rank coordinates. `CaptureCoordinator.create` loads configuration inside
the voted callback and creates Store/Catalog resources only after target identity
agreement. Disabled capture still returns before collective work. The existing
TP/PP/DP server gates stay closed and the request coordinator explicitly rejects
distributed capture after identity agreement until its resource/admission and
request lifecycle are connected.

Focused job `01790844323918830249-675c182b4d55` passes three tests and
25 subtests in 38.30s. Four real Gloo processes assemble TP2/PP2 and TP4/PP1
metadata. Eleven injected failures cover local artifact binding, record and
aggregate byte limits, receive allocation, malformed JSON, conflicting teacher
identity, cyclic rank-origin relabeling, topology disagreement, one validator
failing, different final digests and protocol-version disagreement. Each
recoverable failure is followed by a successful exchange on the same group,
proving the collective sequence remains aligned.

An inactive payload rank then exits before entering startup; all three peers
fail transport within 15 seconds. In a separate four-process case, a rank delays
binding for five seconds while collective waits are 0.5 seconds. The other three
return transport failures within four seconds and the delayed rank fails within
ten seconds; no partial contract returns. A focused coordinator test also
requires a configuration-file failure to occur inside the startup callback and
verifies that Store connection has not started.

Full runtime job `01790844400351228227-0a62d5cf56e8` passes in 461.685s
and publishes 90 snapshots through the real local TCP Store. This exercises the
new protocol on the actual serving world CPU group before capture initialization,
including AR/static DSpark, graph/overlap, memory pressure, admission/latency
controls and cache lifecycle checks. The broader capture and DSpark unit run
`01790844400690933229-88827fe808dd` passes 138 tests and 202 subtests in
73.72s, including the multi-process Gloo failure scenarios.

This is a real multi-process metadata protocol, not distributed model inference.
It assumes every group member enters the same invocation sequence. Earlier
model/group initialization failure, an indefinitely blocked local callback or
failure to allocate the small control buffers still requires the serving
supervisor. Distributed Store/resource initialization, common request policy,
admission/failure agreement, per-sample descriptor exchange and global teacher
scores remain open; no RDMA or production Catalog claim is made.

Commands, protocol bounds, fault cases and source/log hashes are retained in
[`capture-startup-agreement.json`](experiments/capture-startup-agreement.json).
Local/GPU source hashes match. New files pass Ruff, legacy coordinator/worker
diagnostics match HEAD, and formatting, whitespace and Python 3.10 syntax checks
pass. The original worktree index is unchanged; all three jobs are terminal and
the resident GPU has resumed its idle workload.

## Collective Resource Readiness

Capture startup now has a passive resource phase after target identity binding.
`CaptureConfig.startup_policy` agrees on request, lease, Catalog and shared Store
settings while allowing rank-local journal paths, byte budgets and transport
configuration. `coordinate_resource_startup` votes policy construction failures,
compares policy digests before allocation, then votes resource readiness and
confirms receipt of that vote. Startup wire version 2 adds an explicit phase ID
to the fixed control header; identity agreement also has a confirmation step.

`CaptureResources.prepare` connects one Store client per active owner and builds
the partition's registered Host pool and selected-layer exporter. Only the aux
owner locks the publication journal. Aux-only ranks do not access source KV;
inactive ranks allocate no Store client or Host pool but participate in every
collective. No Catalog requests or Store writes occur during preparation.

Serving prepares writer, lease and metrics threads behind an activation event.
They begin recovery/admission only after resource confirmation. Failed thread
startup stops any waiting threads. Failed rank preparation cleans its partial
resources, while the protocol closes the other ranks' successfully prepared
resources and votes cleanup failures when transport still works. Store close
must complete before registered storage can lose its references. Failed close
retains the adapter, registered buffers and owning resources until a successful
explicit close or process teardown. SDK setup exceptions now close the partially
initialized client as well as nonzero setup statuses.

The first fault run exposed a late-participant bug: three ranks timed out waiting
for resource readiness, but the fourth completed that old Gloo all-gather and
returned ready. The added acknowledgement prevents this path. Corrected focused
job `01790846332571489732-242c1c36d202` passes six tests and 37 subtests in
54.32s. The initial failing log is retained as evidence of the reproduced fault.

Broader job `01790846523111396436-6453ef509248` passes 146 tests and
216 subtests in 150.98s. It covers capture, DSpark KV-input helpers and the real
TCP Store tests. Four Gloo processes exercise TP4 replicated heads and a TP2/PP2
layout whose last stage owns only aux. Faults include common policy disagreement,
policy construction failure on an inactive rank, ambiguous second-buffer
registration, insufficient local Host budget, an occupied journal, inactive-rank
preparation failure, uncertain transport close and a delayed rank.

The real Store case first rejects rank 2's insufficient budget and closes the
other prepared clients. It then recreates each client using the same configured
local address and journal, performs the readiness protocol, and writes from
three independently registered arenas. After all four producer processes exit,
an independent 64MiB provider/reader verifies all three 64-byte payloads. No
Catalog capture or publication was created. This validates resource lifecycle
and registered transport; the existing separate owner-publication tests still
cover snapshot manifests and receipts.

Full serving job `01790846523443531563-0d819d0e7ceb` passes in 460.686s
and publishes 90 snapshots through the real TCP Store. This covers the new
startup path in AR/static DSpark, graph/overlap, memory pressure, capture
admission/latency and cache lifecycle scenarios. The SDK setup regression was
also run against the pre-change adapter from `8fc2ed67b`: job
`01790846569607181735-6e059025cf6b` fails exactly because the client remains
open after an `OSError` from setup. The corrected adapter passes the same test
in the broader run.

Final focused job `01790846803151233405-013c9cb15ce9` passes eight tests and
38 subtests in 71.36s, including phase mismatch and delayed identity validation.
The latter requires all ranks, including the late validator, to fail instead of
returning a partial contract. These two boundary cases were added after the
broader run; production sources are unchanged. Commands, failures, terminal
results and source/log hashes are retained in
[`capture-resource-startup.json`](experiments/capture-resource-startup.json).

Local/GPU source hashes match. New/changed production helpers and new tests pass
Ruff; legacy diagnostics exactly match the baseline. Formatting, whitespace and
host Python 3.10 syntax checks pass. The original worktree index is unchanged.
All six jobs are terminal and the resident H100 has resumed its idle workload.

This does not enable TP/PP model serving. The single-rank serving coordinator
uses the new resource protocol on its existing world CPU group; distributed
request admission, failure agreement, descriptor exchange, teacher scores and
scheduler integration remain open. Startup confirmation is not a durable
transaction or protection against a process dying after confirmation. Blocking
callbacks/close calls and worker initialization still require the supervisor;
a transport-failed process group must be torn down. Real Store verification uses
TCP on one host, synthetic local KV and a Catalog test double, not RDMA or
distributed target inference.

## Partitioned Request Contexts

Request capture now accepts the same canonical partition as its Host pool and
exporter. A KV-only owner maintains its own committed token ledger and exports
only its local heads, without allocating or accessing aux tensors. An aux-only
owner records the computed prefix from its target forward and captures teacher,
position, token and mask payloads. Ownership violations, wrong local head counts,
commits ahead of the known prefix and foreign Host slots are rejected.

The request context preserves existing D2H event/staging ownership and handles
incremental decode or accepted verify prefixes. Trimming excludes lookahead KV
and teacher rows from the final sequence; sealing flushes a non-full staging
tail. `prepare_partition` waits for local copy completion and returns owner-local
descriptors/views. Partitioned contexts cannot call the full-snapshot API.
Cancellation now also invalidates a sealed context before preparation. The
writer rechecks the state after copy waits and descriptor construction, so a
cancellation during those operations cannot return a new preparation. This does
not revoke already prepared descriptors or authorize deletion of Store objects;
the coordinator must still discard pending work and enforce Catalog fencing.

`PreparedSnapshotPartition` now carries `token_ids_sha256`. Every owner must
attest the same committed sequence and it must match the aux token descriptor.
KV-only callers of `prepare_snapshot_partition` supply their token ledger;
an aux caller may derive it from the completed buffer. A request context always
supplies its ledger, so mutation of the aux token buffer after commit is rejected
before descriptor publication. Equal sequence lengths and complete head coverage
cannot hide owners recording different token sequences. The public manifest
format is unchanged; the internal preparation interface requires the new field.

The shared partition fixture now builds its snapshots through real request
contexts instead of directly filling every Host payload. The real Store tests
therefore exercise context-produced partitions, immutable owner-local writes,
manifest-last publication and independent readback after owner process exit.
These are synthetic generation traces, not distributed model inference.

Focused job `01790848359214920602-5137e5a99376` passes 22 tests and
44 subtests in 10.77s. The initial broader job
`01790848472723190114-741413eb67ab` passes 153 tests and 219 subtests in
167.85s. After the cancellation race fix, broader job
`01790849253038327309-27b60e9cf0aa` passes 154 tests and 221 subtests in
168.00s. CPU traces use TP4/PP2 canonical owners and an aux-only last stage,
non-contiguous source slots, chunked prefill, decode and accepted verify suffixes.
They overwrite source slots immediately, validate exact reconstructed tensors,
reject equal-length divergent replies and reject a mutated aux token payload.
The CUDA partition test now runs through the request context, including source
reuse, bounded staging, finalization outside the forward stream and copy wait.

Pre-fix job `01790848556130153660-3f04d25f410e` loads unmodified baseline
context/assembly code and reproduces acceptance of cancelled sealed data and
inconsistent token digests. Job `01790849252707648740-3733fc91063b` reproduces the
copy-wait race against the context before its final state guard: both full and
partitioned preparation return data after an event-controlled cancellation.
These are expected failures; the final broader run includes their passing tests.

The initial full runtime job `01790848473067409649-d7a6f870aaf3` passes in
463.088s with 90 READY samples. The post-race-fix run
`01790849253418715081-7baf3ab8ddf6` reaches 90 READY samples but fails its final
Catalog error assertion. Terminating the preceding serving process interrupts a
background HTTP POST; the test Catalog incorrectly treats the incomplete body
as malformed JSON and poisons the following lifecycle scenario. The fixture now
discards EOF/reset before body completion without invoking a Catalog operation;
fully received malformed JSON still reports an error. This changes only the
test HTTP service, not production Catalog semantics.

Job `01790850082603955905-f5635c5b2911` reproduces the fixture failure against
the unmodified baseline using socket half-close with zero/partial body bytes.
The corrected fixture and real Store regression job
`01790850082937982590-e6dd4fdf539f` passes 5 tests and 2 subtests in 71.45s.
Full runtime rerun `01790850083292972978-08c808f42f47` passes in 461.962s with
90 READY samples, including both lifecycle scenarios. Final fixture-source job
`01790850209530978134-285b91f3b36a` passes 2 tests and 2 subtests in 11.26s.

Source/log hashes, terminal results, failure reproductions and coverage limits
are recorded in `experiments/capture-partition-context.json`. Ruff passes for
eight changed/new files; the Catalog fixture retains only its two baseline
diagnostics (BLE001 and C408). All nine Python files pass formatting and Python
3.10 compilation. Local/GPU source hashes match, `git diff --check` passes, and
the original root's staged index remains unchanged. All eleven jobs are terminal;
the resident worker has resumed its idle workload.

This is the owner-local state required by distributed admission. It does not
wire cross-rank request decisions, accepted-token transport, abort propagation,
descriptor exchange or TP/PP scheduler hooks. The existing distributed serving
gates remain closed. A prepared token digest detects inconsistent trusted
producer ledgers; it is not proof that arbitrary KV bytes came from the stated
tokens. Runtime source mapping and target execution still require end-to-end
verification on the intended multi-GPU topology.

The next admission integration must respect PP's ordering:
`scheduler_pp_mixin.py` receives the upstream proxy before launching a local
forward, while the upstream stage sends that proxy only after its own forward.
An all-PP collective inside `before_forward` would therefore deadlock. Prepare
multi-owner lease/Host-slot cohorts in a dedicated background control group,
then carry the chosen cohort identity along the existing request propagation
path. Keep control collectives off the inference group's communication sequence;
capture backpressure or a missing cohort must not block model execution.

## All-Owner Cohort Reservations

`CaptureCohortAllocator` connects the agreed layout and passive rank-local
resources to one collective Host-slot reservation and Catalog lease. Every rank
participates, including inactive partitions; only active owners reserve slots,
and only the aux owner calls Catalog begin. A missing slot rolls back all local
acquisitions before returning collective backpressure. A ready cohort carries
one agreed lease/fence, the owner-local slot and conservative local monotonic
deadlines. The Catalog reservation uses the full owner set and summed registered
Host arena bytes.

The control protocol uses a dedicated Gloo group, bounded CPU buffers, round and
phase IDs, policy/lease digests, validation and a final readiness acknowledgement.
Callback/encoding/allocation failures are voted before peers proceed. A known
lease is failed by its aux owner during rollback; an uncertain begin or a changed
response identity relies on Catalog expiry and never authorizes a failure call
against potentially unrelated credentials. No payload transfer has started, so
these rollback paths can release their own unused slots. Transport/protocol
failure poisons the allocator and requires control-group teardown.

Initial four-process job `01790851545598842826-1d444be9275a` passes one test and
15 subtests in 26.03s. The next job
`01790851800639073626-da11aed13d05` reproduces a readiness gap: a rank whose local
clock advances past renewal after validation still returns a cohort. The fix
checks lease freshness again in the final readiness vote; the regression orders
that clock advance deterministically without probabilistic timing.

Combined job `01790851973503224161-6d575d5d37e0` passes 12 tests and 56 subtests
in 149.75s, covering the new allocator, identity/resource startup and real Store
transport/publication. Cohort cases include TP4 replicated heads, an aux-only PP
stage, an inactive first-stage leader, backpressure/retry on the same allocator,
Catalog outage/lost response/changed identity, oversized lease encoding, control
copy failure, expiry, rank-local decode failure, divergent fences, failed cleanup,
round mismatch and a late inactive validator. The reservation runs on a background
thread concurrently with a separate inference-group all-reduce.

The real Store resource test now obtains all three registered slots from the
cohort before writing rank-specific 64-byte payloads. An independent provider
reads them exactly after all owners exit, and the aux owner retires the unused
test capture. These raw transport payloads are not a published training sample;
the separate owner-write/manifest publication test continues to cover publication.

Final job `01790852181547402156-c2f238cbbe3b` passes 4 tests and 18 subtests in
87.18s. It constructs the allocator under a meta-device context and verifies
actual Gloo communication through explicit CPU control buffers. Both new files
pass Ruff; the Store test retains only its baseline SIM115/PLW1510 diagnostics.
All three Python files pass formatting and Python 3.10 compilation, local/GPU
hashes match, and the original staged index is unchanged. Four jobs are terminal
and the worker has resumed its idle load. Commands, source/log hashes, failure
reproduction and limitations are in `experiments/capture-cohort-reservations.json`.

This is the background reservation interface, not distributed serving admission.
Request ticket propagation, a ready-cohort queue, lease renewal, failure agreement
after request binding, accepted-token delivery and descriptor/receipt exchange
still need coordinator/scheduler integration. Inference never calls this new
protocol yet, and TP/PP gates remain closed. The tests use one host, Gloo, a
Catalog test double and TCP Store; they do not prove multi-GPU inference or RDMA.

## Background Cohort Lifecycle

`CaptureCohortService` now provides the bounded ready registry and background
control loop above the allocator. Rank zero claims a request-fingerprinted
ticket; every request actor binds its local cohort without HTTP or collectives.
Unbound cancellation uses the ticket, while bound actors retain their handles
through local CUDA/Store completion. A second registry exchange proves every
rank installed a cohort before it becomes claimable. The allocator's version-2
control sequence adds lease renewal, confirmed retirement and registry exchange
with policy/ledger/shape agreement and final acknowledgement.

Failure and buffer ownership are separate. Peer cancellation, expiry, request
timeout and renewal failure invalidate captures, but a bound actor must still
finish. Retirement waits for prior invalid votes from every rank, preventing a
late bind after a state snapshot from losing its slot. Successful publication
requires all active owners and all bound actors to finish; inactive ranks with
handles participate in this lifetime too. Only aux can report publication, and
an ambiguous publish must be resolved through the journal before that report.
Control failure retains live handles, and uncertain transfers quarantine their
slots and retain resources. Bounded shutdown cannot authorize teardown while
those owners or transfers remain outstanding.

First job `01790853924671212898-cc8ab04f43fc` passes one test and 15 subtests in
23.32s. Combined job `01790854048491678654-246bd83bd37b` passes 13 tests and
71 subtests in 162.30s, covering service/allocator, startup/resources and real
Store transport/publication. The real Store resource test now starts background
services, propagates a ticket, binds all four actors, writes from three registered
slots, and requests shutdown while actors still own their slots. All actors
explicitly finish before resource teardown; the independent provider reads the
exact bytes after the producer processes exit.

Expanded job `01790854390471440607-37089fe1e056` passes all three Store tests but
fails a new lifecycle assertion: its multiple-request setup requested stop on
every rank while expecting another propagation round. Drained captures can
correctly retire in that round. The fixture now issues stop on one rank to test
the intended propagation boundary. This was a test expectation error, not a
production-code fix. Final job `01790854537590748661-5316cf63f0fd` passes one
test and all 19 lifecycle subtests in 24.06s.

The final cases additionally cover unbound cancellation, inactive actor drain,
multiple concurrent cohorts with slot replenishment, and a local shutdown after
an empty control snapshot under backpressure. A blocked aux heartbeat runs
concurrently with foreground handle operations and a separate inference-group
all-reduce. Control-frame construction, shape, ledger and flag faults stop all
ranks without releasing live buffers. Changed renewal fences, stale tickets,
request identity mismatch, expiry and ambiguous Catalog failure are covered.
The service-only publication outcome cases use synthetic actor acknowledgements;
they do not claim a published training sample. The separate real Store writer
suite continues to verify manifest-last publication.

All four modified/added Python files format and compile on host Python 3.10.
Allocator, service and lifecycle tests pass Ruff; the Store test retains only
its pre-existing SIM115/PLW1510 diagnostics. Local/GPU source hashes match.
Commands, terminal results, log hashes and scope limits are recorded in
`experiments/capture-cohort-lifecycle.json`.

This completes the reusable cohort lifecycle, not its serving integration.
The runtime still needs stable request digest construction and ticket transport,
coordinator ownership of these handles, accepted-token delivery, descriptor and
receipt exchange, global teacher scores and PP scheduling hooks. Distributed
serving gates remain closed. Validation uses one host, four Gloo processes, a
Catalog test double and TCP Mooncake; it does not prove multi-GPU inference,
cross-node RDMA, production Catalog retention or trained draft quality.

## Scheduler Request Ticket Routing

`CaptureRequestRouter` now connects cohort selection to the scheduler's existing
request communication boundaries. The first PP stage's TP/CP ingress rank
selects once after input blocking and before TP broadcast. A bounded, versioned
ticket is appended to the tokenized request's array IPC schema and travels with
the ordinary TP broadcast and PP request messages. Batch generation requests
are handled item by item; control messages do not consume cohorts. The ingress
overwrites supplied capture metadata, excludes unsupported/private requests,
and skips sampling or capacity pressure without waiting for Catalog.

The ticket binds a fresh request incarnation to the original prompt, complete
sampling contract, token types, reasoning mode and cache salt. Downstream ranks
validate this identity before attaching a local route. Selected requests use a
private copy of sampling parameters, so scheduler clipping cannot change the
ingress object forwarded to another PP stage. Local bind additionally records
the effective execution digest after normalization, ready for subsequent owner
descriptor agreement. Queue rejection/eviction, timeout, grammar rejection and
queued/running/chunked abort invalidate the corresponding route without releasing
a bound actor's buffers. Failure after bind but before returning ownership to the
actor explicitly drains the unused handle; no transfer has started at that point.

The request tests exercise actual message-pack/pickle serialization, receiver
ordering and scheduler generation/abort/queue methods. Four real Gloo processes
use TP2/PP2 request broadcasts and point-to-point sends with an inactive ingress
stage, active second-stage owners and a separate background cohort group. They
verify one selection, common tickets and effective hashes after per-stage
clipping, queued cancellation, and delayed release of bound actors. This is a
request-transport test with a Catalog double and synthetic Host pools, not a
distributed model forward or published training sample.

Initial job `01790855641040094762-ab69d90c9b68` did not collect tests because the
GPU checkout lacked several existing manager test files. After staging them,
job `01790855723592041926-5a72b5421acc` passed 110 tests and 38 subtests but exposed
missing runtime-config/TP-rank setup in the new fixture. Scoped test configuration
fixed that setup; job `01790856032411578955-789f006a233e` then passed all 14 request
tests and 16 subtests in 25.66s, including real Gloo request transport.

Single-H100 runtime job `01790856128671804572-0e5227e0df60` passed in 461.594s,
generating and validating 90 READY snapshots across the existing AR/DSpark
scenarios, graph/overlap modes, prefix reuse and retraction/abort coverage. It
checks compatibility of the shared scheduler/IPC edits with existing single-rank
capture; it does not activate the distributed router or prove multi-GPU serving.
The runtime ran before the router-only unused-handle cleanup and the final
equivalent `bytes | None` annotation edits.

Job `01790856494956635925-6855ab8fdbce` deterministically reproduces a post-bind
bookkeeping failure that cancelled the cohort without returning the unused actor
handle. The fix finishes that known handle with `transfer_complete=True` before
returning admission failure. Final job `01790856653591344549-cb05939a7875` passes
56 tests and 34 subtests in 25.61s, covering the fixed router and final IPC types.
Actual running-request cancellation still only invalidates the handle and never
claims its transfers completed.

The new router and test pass Ruff. Existing-file diagnostic counts match HEAD;
all seven changed/added Python files compile on Python 3.10. Source hashes match
the GPU checkout, and unrelated formatting differences in the existing scheduler
and batch files are preserved. All six jobs are terminal and the resident worker
has resumed its idle load.

The current single-rank coordinator leaves `request_router=None`; distributed
factory construction still rejects unsupported topology. The scheduler hooks
are available, but coordinator activation, partitioned contexts, execution-digest
agreement, accepted-token delivery, owner descriptor/receipt exchange and global
teacher rows still need integration. Existing TP/PP serving gates remain closed.
Validation results and the final source/log hashes are recorded in
`experiments/capture-request-routing.json`.

## Cohort Descriptor And Receipt Agreement

Control protocol version 3 now exchanges completed owner descriptors and write
receipts through the dedicated background Gloo group. Writer actors submit
immutable metadata bytes and poll local results; model forward does not enter
these collectives. Every rank, including inactive partitions, agrees on ingress
identity and the effective execution digest. Canonical owner placement, the
reserved teacher/KV contract, committed token ledgers, metadata hashes and full
logical coverage are validated before any agreed manifest is exposed. The aux
owner supplies one timestamp so every rank assembles identical manifest bytes.

The allocator votes metadata policy, lengths, allocation success, validation
and final lease freshness. The padded CPU send/receive arena is bounded to
64 MiB, individual offers/results to the configured manifest capacity plus
4096 bytes, and the final manifest to its reserved buffer. Local encoding,
allocation and parser failures cannot advance a peer into a different phase.
Validation failure invalidates the capture while leaving live transfer buffers
owned by their actors. Transport failure retains the existing poisoned-control
shutdown behavior.

Each active writer submits a receipt bound to capture ID, fence, canonical owner
and the entire agreed manifest SHA-256. KV actors may finish after their own
writes; the service retains their frozen receipt until aux receives the complete
set. Inactive actors finish after manifest agreement. Aux publishes through the
existing journaled `SnapshotWriter`, then reports publication. Successful finish
for handles using this protocol rejects missing manifest/local-write/global-
receipt stages. Catalog WRITTEN validation and retention remain independent of
these trusted producer acknowledgements.

Job `01790858228655961823-7f7f7ccb5430` stopped at collection because the new
msgspec offer declared a default field before required fields. Reordering the
field fixed the import. Job `01790858279987384724-38b6654dcc79` then passed three
tests and 59 subtests in 53.51s. Its 22 new four-process cases cover delayed
descriptors/receipts, inactive ingress, execution mismatch on an inactive rank,
token/metadata/contract/owner/fence mismatch, malformed/oversized payloads,
encoder/allocation/parser faults, inconsistent result hashes, expiry and
cancellation during exchange, invalid receipts and local submission failure.
Every failed capture retains its actor-owned slot until explicit completion.

Job `01790858457183417432-3cfd60ece81c` passed 35 tests and 26 subtests across
partition contexts, snapshot writers, request routing and the existing real
Store suite. Its new Store test completed publication and exact tensor comparison,
then failed a fixture assertion that confused Catalog AVAILABLE with manifest
READY. After correcting that assertion, final job
`01790858613107821195-07c37f4ddb39` passed the full new test in 36.55s. Four real
producer processes agree a manifest, three canonical owners write registered
buffers, and aux publishes after all receipts. Once every producer exits, an
independent reader retrieves and validates all 32 tensor objects (4532 bytes).
The parent reader also checks every reconstructed tensor exactly against the
global synthetic fixture, including selected KV, tokens, mask, top-128 IDs/raw
logits and LSE.

All six Python files format and compile on host Python 3.10. Five pass Ruff;
the existing Store test retains only its two baseline diagnostics. Local/GPU
source hashes match. All four jobs are terminal and the resident idle load has
resumed. Commands, source/log hashes and terminal results are recorded in
`experiments/capture-cohort-exchange.json`.

This completes the descriptor/receipt control and Store publication path for
completed synthetic partitions. It does not activate distributed serving:
coordinator ownership, partitioned runtime contexts, accepted-token propagation,
global teacher rows and TP/PP model scheduling still require integration and
runtime validation. The tests use one host, four Gloo processes, TCP Mooncake
and a Catalog double; they do not establish multi-GPU inference, RDMA,
production Catalog retention, training quality or performance SLO compliance.

## Completed Cohort Writer Actors

`CohortSnapshotWriter` now takes ownership of sealed or aborted partitioned
request contexts and advances their Store publication on one background thread
per rank. It waits for D2H before describing payloads, submits owner metadata,
polls the agreed manifest, writes only local objects, exchanges receipts and
publishes from aux. Inactive ranks own only their metadata/handle completion.
Submissions freeze metadata, reject duplicate/foreign ownership and remain
bounded by the reserved cohort count. Failed submission leaves the caller
responsible for its context; successful submission forbids further inference-
thread mutation.

The actor visits every queued request instead of waiting synchronously for the
first request's peers. This prevents different PP/TP completion orders from
forming a circular metadata wait. Startup readiness includes worker-thread
initialization and aux journal replay. Copy uncertainty quarantines the original
slot, and closing the writer cannot acknowledge an in-flight copy.

Aux publication exceptions retain the exact manifest, complete receipt set and
original lease for idempotent reconciliation. `recover_partitions()` validates
every stored tensor and retries using a new registered manifest buffer. Recovery
also handles a failed directory sync after journal unlink, when a missing journal
does not establish that publication failed. Recovery buffer allocation counts
retained quarantined buffers against the receive budget. Cancellation, expiry
and shutdown do not override an unresolved publication outcome.

The cohort service now accepts `published` with `transfer_complete=False`:
publication recovery can succeed while the original source still requires
quarantine. It never reports that committed sample failed, and normal service
close still refuses to tear down quarantined storage. `stored` still requires
complete transfers. The deterministic regression fails on the previous service
in job `01790859744928408454-fb2b1518cb3f` with `invalid local capture completion`
(one expected failure, 10.39s); the old service hash is recorded in the evidence.

Job `01790859862859096712-ee71b92aa398` passed 65 tests and 57 subtests in 113.07s:
new writer lifecycle, snapshot publication/recovery, the existing single-rank
coordinator, cohort lifecycle/metadata exchange and a real Store actor workflow.
After tightening worker initialization readiness and adding two startup cases,
job `01790860084879682234-d85878604758` passed all nine writer tests and the Store
actor test in 52.79s. Writer tests cover copy-gated shutdown, duplicate submit,
metadata mutation after submission, uncertain D2H, lost publish response with
concurrent cancellation, journal cleanup failure, ambiguous manifest transport,
published/quarantined ownership and startup failure/recovery readiness.

The real TCP Mooncake case uses four Gloo processes and two bounded cohorts.
Ranks 0/1 submit sample A first while ranks 2/3 submit B first; a barrier proves
both initial submissions are waiting for metadata before second submissions
arrive. Both samples then publish, all writer actors finish, and both Host slots
per active owner return after service drain. Following producer exit, independent
reader processes validate 32 objects / 4532 bytes per sample. Parent readers also
compare every reconstructed KV/aux tensor exactly to the deterministic fixture.
The contexts use the real collection/seal/partition APIs with CPU fixture tensors;
this is not a distributed target-model or CUDA-forward test.

All six changed/added Python files format and compile on host Python 3.10. Five
pass Ruff; the Store test retains its two baseline diagnostics. The final Store
fixture explicitly binds a synchronous polling lambda's loop variable for Ruff;
that equivalent test-only binding was made after the last run. Final production
sources match the tested GPU files. All three jobs are terminal; the resident
idle load resumed. Full commands, final and tested hashes and limits are in
`experiments/capture-cohort-writer.json`.

Distributed serving remains gated. The writer actor is available for coordinator
ownership, but request admission/finalization, partitioned forward collection,
accepted-token propagation and global teacher scores still require runtime
integration and TP/PP validation. Persistent unresolved fences/journal failures
need Catalog reconciliation; this actor retains ownership rather than guessing
a failed publication. No production Catalog, cross-node RDMA, trained quality or
performance SLO result is established by this change.

## Cohort Request Collection

`CohortCaptureCoordinator` connects the request router, partitioned collection
contexts and completed-context writer actor. A routed request takes ownership
of its bound cohort before any forward copies. Each rank tracks the committed
request history; only KV owners export their local selected-layer/head slices,
and only aux captures teacher scores and auxiliary tensors. Inactive ranks keep
a token/KV-progress ledger without allocating a Host slot. Successful finalization
detaches the request and passes its immutable metadata/context to a bounded
background handoff queue. Failure still fences copies before returning ownership.

The shared AR/static-verify collector now honors local payload ownership.
Non-last PP workers call the collector after forward, before source KV can be
reused. PP result reconstruction does not preserve the process-local forward
ticket, so the cohort coordinator reads the scheduler's committed request history
after normal result processing. The last-stage model contract requires the
existing TP vocabulary gather before sampling; capture adds no model-forward
collective and stores raw pre-sampler top-128 rows plus their vocabulary IDs.
All ranks use the common startup-policy fingerprint in provenance, excluding
rank-local paths, addresses and buffer budgets.

The cohort control protocol is version 4. Admission readiness is now voted by
every rank. An inactive ingress cannot select a request while aux is still
recovering its publication journal. Pausing admission does not release existing
bound handles. Coordinator close first detaches collecting requests and drains
the handoff thread, then closes writer and cohort service before resources. An
incomplete close retains the coordinator and registered buffers; the caller
continues to own the dedicated control group.

Job `01790861976063787084-787874e36bee` passed the real TCP Mooncake integration
in 44.41s. Four Gloo processes run five collector scenarios: TP2/PP2, replicated
KV heads with an aux-only and an inactive rank, inactive PP ingress with delayed
aux startup, dense static verify commits, and cancellation on one rank. Each
scenario routes two actual `Req` objects through msgpack and the request router;
ranks complete them in opposite order. The worker callback is the real
`TpModelWorker.forward_batch_generation`; model forwards are synthetic CPU
fixtures. A simulated sampler overwrites logits after capture, and request KV
buffers are zeroed immediately after finalization.

Nine complete samples publish; the cancelled sample fails. After every producer
exits, the parent compares all reconstructed KV and auxiliary tensors exactly
to the fixture. A separate reader also validates a sample's 32 objects and
4532 tensor bytes. This establishes collection, ownership and publication with
real Store transport, not distributed target-model inference.

Baseline job `01790861357939372557-108b29b2efad` passed 69 tests and 80 subtests
in 76.70s. Two new coordinator lifecycle tests pass independently in job
`01790862297292945734-6d1f99f3704f` (11.43s), covering failure after bind and
copy-gated rejection of writer handoff. Five partition-context cases pass
independently in job `01790862297626550169-f85c64a064dc` (10.16s), including
the inactive rank's committed-prefix checks without payload allocation.

The standalone identity suite passed seven tests and 31 subtests in job
`01790862443347339497-93a1489bba02` (10.27s). Real SGLang serving on the resident
H100 also passed in job `01790862375707774981-4ef3a154c620` (463.159s), producing
90 READY samples across the existing AR/static-DSpark, overlap, CUDA Graph,
prefix reuse, cancellation/retraction, memory-pressure and adaptive/latency
capture scenarios. This is a regression of the shared collector and worker
changes; it still uses the existing single-rank serving coordinator.

All 12 changed/added Python files format and compile on host Python 3.10. Ruff
diagnostics match the pre-existing TP worker, single-rank coordinator and Store
test baselines; the other nine files pass. Final files match the tested GPU
checkout. Commands, log hashes, source hashes and validation limits are recorded
in `experiments/capture-cohort-coordinator.json`.

The distributed serving factory and CLI gate remain unchanged. The next
integration must construct and own a dedicated control group, coordinate
startup/activation failures, and validate real TP/PP scheduling. The four-rank
fixtures above do not establish CUDA TP/PP inference, multi-node RDMA,
production Catalog retention/reconciliation, trained draft quality or SLO
compliance.

## Distributed Factory And Activation

`CaptureCoordinator.create()` now selects the cohort-backed coordinator for
TP/PP topologies after rank identity agreement. Common policy and passive
resource preparation still use the existing startup group. A new owned Gloo
group carries only background capture control. Group creation precedes
rank-local Store/pool allocation so a local preparation failure cannot strand
peers in a different construction phase. The initial factory requires the
complete PP-major worker group; DP/subgroup serving remains unsupported.

Startup protocol version 3 adds two activation votes. Writer, cohort service
and handoff threads start behind an activation event, and only a successful
all-rank acknowledgement releases their work. A rank failing any thread start
rolls back without issuing capture Catalog/Store work. Close handles an allocated
but unstarted thread, drains actors, closes transport, and finally destroys an
owned control group. Failed transport shutdown retains both coordinator and
control group for a later close attempt.

The first two factory test jobs failed during repeated group construction:
`01790863815775220186-b893a4c1ec76` (110.22s) and
`01790864036903616254-42eca573d939` (110.57s). The latter stack dump identifies
`new_group` as the remaining rank's blocking operation. In the installed
PyTorch build, local-synchronization group names hash rank membership and the
current count of live groups; destroy/recreate reused the rendezvous namespace.
The factory now uses globally ordered group creation for the full worker group.
The test additionally requires a fresh group name for every construction.

Job `01790864189951423422-cdb3904d13e8` passes the four-process factory test in
23.52s. It covers TP2/PP2, replicated heads, inactive ingress, policy disagreement,
rank-local resource failure, each of the three actor thread-start failures,
then another successful startup. Every created control group is destroyed only
after its actors exit, every resource closes, and the HTTP Catalog observes
reservations only for successful cases. Model binding and CUDA exporter identity
are simulated in this CPU lifecycle test; it does not run model forwards.

The real Store integration adds a CUDA factory scenario. Four processes share
the resident H100 and construct real `MHATokenToKVPool` GPU buffers, pinned Host
arenas and Mooncake clients through the factory. Only target artifact binding
is substituted with synthetic rank contracts; model forward results remain
deterministic fixtures. The actual worker callbacks copy GPU KV and raw GPU
teacher rows, source buffers/logits are overwritten, and the factory owns the
entire control-group shutdown. Job `01790864190248524997-3335ada5b2c9` passes
in 46.65s: eleven samples across six scenarios publish, including the two CUDA
factory samples, while one cancelled sample fails. Every reconstructed tensor
matches after producer exit; an additional reader validates 32 objects / 4532
bytes for one sample.

The standalone coordinator lifecycle suite passes three cases in job
`01790864317023823470-25206fb42e44` (12.03s), including retention of an owned
control group across failed transport close. This extends factory construction
and startup/rollback implementation; it is not multi-GPU inference or an RDMA
result. The CLI TP/PP capability gate remains in place pending real model
scheduling and numerical validation. Production Catalog, training quality and
performance acceptance remain open.

Regression job `01790864317342206946-dbda1e5cc113` passed 60 tests and 62
subtests in 122.38s, with one failing startup version-mismatch case: its injected
version was hardcoded to 3, which is now valid. The test now injects the current
version plus one. The complete startup file then passed independently in job
`01790864571182368511-9423ae3b2ce8` (four tests, 25 subtests, 54.64s). No
production change followed the successful factory/CUDA integration runs.

All 12 changed/added Python files format and compile. Ten pass Ruff; the
single-rank coordinator and Store test retain their nine and two baseline
diagnostics respectively. Final files match the GPU checkout. Complete job
outcomes, commands, source/log hashes and limits are in
`experiments/capture-cohort-startup.json`.

## Real TP And PP Serving Capture

Ordinary AR capture now accepts TP/PP configurations through the CLI and uses
the distributed factory implemented above. DP/context parallelism and
distributed speculation remain rejected. The existing single-rank static
DSpark path remains enabled. Runtime binding still validates actual model
artifacts, rank-local projection/pool geometry, layer ownership and global
teacher scores before any capture actor activates.

`test_training_capture_distributed.py` runs Qwen3-0.6B with real TP2/PP1 and
TP1/PP2 model workers on two distinct H100 80GB devices. The temporary Northjob
allocation is `job-a3c85e8db374-20261001223747` on node199. It uses the shared
capture venv and a separate TCP Mooncake segment; the Catalog is the existing
HTTP test double. No target identity, forward, CUDA KV pool or capture factory
is mocked. The image's old FlashInfer 0.6.12 cubin/JIT-cache packages were
removed from the temporary container to match the existing 0.6.17 environment.

For each topology, three requests exercise 160-token chunked prefill, prefix
reuse, one-token completion and biased multi-token decode. Independent attention
hooks observe all selected layers (0, 14, 27) on their actual TP/PP ranks.
After the producer exits, an independent Store client reconstructs every global
head/token range and compares K/V and raw top-128 scores exactly. Global vocab
IDs, response alignment, masks and logsumexp are also verified. The initial
eager-only run passed both tests in 83.815s, publishing six samples.

Each topology also restarts with decode CUDA Graphs. TP enables normal overlap;
PP uses its non-overlap pipeline scheduler. A separate test-only observer runs
after ModelRunner.forward, reads KV directly from the actual source pool using
forward output slots, and records logits before sampling. It does not call the
capture exporter or read the captured Host buffers. The complete observed replay
run passes both tests in 133.880s, publishing eight samples: six eager and two
graph samples. Every rank executes at least two graph forwards; TP overlap
executes three. Reconstructed tensors again match the corresponding online
observations exactly after all serving workers exit.

The first replay comparison used a different eager run as its oracle and failed:
PP's BF16 KV differed beyond that comparison's tolerance, and TP overlap had
already materialized the final output token's KV. The final test checks the
actual run, including its observed `kv_valid`; it does not force the last bit to
zero or loosen the numerical equality check. An earlier test fixture incorrectly
expected plaintext trace IDs and was corrected to assert the existing SHA-256
contract. These failures did not require changing the collection data plane.

The configuration suite passes seven tests in job
`01790865985013380000-d40c4af38659`, including the boundary between distributed AR
and distributed speculative execution. Test observer I/O synchronizes device
work, so these runs establish numerical and lifecycle correctness, not latency
or performance acceptance. Combined TP2/PP2, real replicated-head topologies,
distributed cancellation/backpressure, multi-node/RDMA and other models still
need runtime validation. Production Catalog and draft training quality remain
open.

The final distributed file, including cleanup of the Mooncake launcher's child
binary, passes in 134.717s. Process inspection confirms no model or Store child
remained, and the temporary two-H100 job was deleted. The complete single-H100
runtime regression passes in 464.148s in job
`01790866847066766265-eb3d9ce77e8b`; the resident worker has resumed its idle
task. All five changed/added Python files compile and format. Four pass Ruff;
the configuration test retains its two baseline C408 diagnostics. Source hashes
match the tested checkout. Commands, all failed/successful attempts, log hashes
and scope limits are recorded in `experiments/capture-real-tp-pp.json`.

## TP Target-KV Draft And Static Speculative Capture

Static DSpark now supports TP with PP=DP=1. The target-KV draft binds its
selected layers against real per-rank target geometry and agrees on target and
draft artifact identities before any projection collective. Every 1024-token
projection chunk gathers selected K/V in global-head order through the serving
TP group, removes replicated logical heads and runs the replicated encoder.
Each rank writes only its local draft context projections. Snapshot publication
continues to use owner-local registered buffers and Store writes; it does not
move KV payloads through the capture control group.

Strict checkpoint validation previously compared global exports against local
TP parameter shapes. It now uses global logical Q/K/V dimensions (before KV
replication), global merged MLP dimensions and the row-parallel input dimension.
The native parallel weight loader still owns slicing and replication. A concrete
TP2 regression with one replicated KV head failed on both ranks with packed and
split checkpoints before the fix. The full target-KV file passes 22 tests and
48 subtests, including canonical global-head ordering and request-slot lifetime.
The startup file passes four tests and 32 subtests, including coordinated draft
policy failures and disagreements. Resource startup passes four tests and
13 subtests; configuration passes seven tests and 32 subtests.

The temporary Northjob `job-3a8707e23152-20261001233124` provides two H100 80GB
devices on node082. The runtime test uses Qwen3-0.6B and a synthetic two-layer KV
draft, real CUDA pools, real TP collectives and an independent TCP Mooncake
segment. Its HTTP Catalog remains a test double. Each worker independently
records its raw teacher rows and source KV before sampler changes or slot reuse;
the test reconstructs heads without reading captured Host buffers.

The complete DSpark case passes in 116.199s. After three AR baseline requests,
eager and graph/overlap DSpark each publish five samples: one-token completion,
unbiased rejection, forced full acceptance and a two-request mixed-acceptance
batch. The 160-token prompt exercises chunked prefill and prefix reuse. Generated
tokens match the AR baseline. After both serving ranks exit, independent Store
readback matches all selected K/V and raw top-128 scores exactly; global vocab
IDs, masks, positions, accepted-path alignment, KV validity and full-vocabulary
logsumexp are checked. Both ranks' draft context projections also match the
independent test reconstruction. Graph/overlap records 18 verify graph forwards
and 23 overlap forwards. The first attempt failed a fixture assertion that
expected two management replies; the API emits one reply per DP group's TP
leader. This was corrected without changing the data-plane checks.

These observers synchronize GPU work, so the result is numerical and lifecycle
evidence, not latency/SLO acceptance. Real replicated-head inference, more than
two TP ranks, combined TP/PP, pipeline speculation, non-static verify, PD/RDMA,
production Catalog integration and trained draft quality remain open. The full
validation record is `experiments/capture-tp-dspark.json`.

The complete distributed file passes all three tests in 257.098s, retaining the
AR TP2 and PP2 checks alongside the new TP2 DSpark case. Final target-KV and
startup files pass independently in 12.09s and 56.49s. The complete single-H100
runtime regression passes in 468.033s in job
`01790869200834994945-4f27d5be0d90`, including cache lifecycle, graph/overlap,
pressure, adaptive sampling and latency protection. No serving/Store process
remained on the temporary node; the two-H100 allocation was deleted and its Pod
is confirmed absent. All 12 changed Python files compile and match the tested
GPU checkout. Ten files pass Ruff; the worker and configuration test retain
their six and two baseline diagnostics. Eleven files format cleanly; the worker
retains its pre-existing class blank line. The original checkout's staged-index
digest is unchanged.

## Confidence-Scheduled DSpark Capture

The capture hook now accepts the original per-request verify lengths. Owned
tickets flatten selected real rows and retain each request's row offset and
length. Cumulative source offsets include unselected requests; graph padding is
excluded. Acceptance cannot exceed a request's forwarded range, and only the
committed prefix supplies KV and raw teacher rows. No snapshot schema or
Mooncake SDK change is required.

This enables collection while serving the existing hidden-input DSpark draft
with its confidence scheduler in `cap-accept` and `compact` modes. It does not
change the target-KV v1 checkpoint's static-only, confidence-disabled contract.
The synthetic confidence draft uses two Qwen3 layers, fixed confidence scores
and a deterministic proposal head to exercise rejection, budget truncation and
unequal batch lengths. Its hidden inputs use layers 0, 14 and 26; captured target
KV still comes from layers 0, 14 and 27.

The independent observer copies source KV and full raw logits before sampling,
then records the actual acceptance commit length. Token equality alone cannot
establish KV validity: a budget-trimmed candidate can match the eventual output
without ever being committed. Store readback is restricted to the committed
path and still requires exact KV/top-128 values, vocab IDs, masks and positions;
full-vocabulary logsumexp keeps its existing floating-point tolerance.

The single-H100 focused run passes all four mode/scheduling combinations in
164.277s (job `01790872384795104248-79c1768047c9`), publishing twelve samples.
Cap-accept uses Triton attention; compact uses FA3, because Triton does not
support ragged target-verify CUDA Graphs. Compact graph/overlap records nine
verify graph forwards and five graph-folded acceptances, including padded graph
layouts. The teacher and KV comparisons happen after each producer exits.

The final coordinator file passes 46 tests and six subtests in 44.62s; its
derived properties cover unselected requests, source-buffer mutation, graph
padding and refusal to consume another request's rows. Configuration passes
seven tests and 32 subtests in 9.75s. Earlier runtime attempts exposed fixture
assumptions about hidden-layer selection, management API responses, commit
boundaries, eager padding and backend graph support. They are retained in the
validation record; numerical comparisons were not relaxed.

Temporary Northjob `job-0cbb9a69991d-20261002002521` allocates two H100 80GB
devices on node199. The same four non-static combinations publish twelve TP2
samples and pass exact independent readback after both ranks exit. Compact
graph/overlap records nine verify graph forwards on the auxiliary owner and ten
folded acceptances across the two ranks. Actual acceptance logs contain unequal
per-request verify lengths `[3, 2]` and budget-trimmed suffixes. Both ranks'
observations participate in global-head reconstruction. The complete distributed
file passes all four tests in 503.439s, retaining AR TP2, AR PP2 and static
target-KV DSpark TP2 coverage.

The complete single-H100 file passes both tests in 628.904s (job
`01790872599883226014-8dba045cfb82`), retaining cache lifecycle, memory pressure,
retraction, static speculation, adaptive sampling and latency protection checks.
The Store suite initially reached the 180-second task limit after five tests;
with a 360-second limit, all six passed in 187.06s. Process inspection then found
an orphaned master child from the SDK launcher. The fixture now kills the whole
launcher process tree; the complete Store file passes again in 188.37s with no
live Store/test process left on the temporary node.

The temporary two-H100 job was deleted and its Pod is confirmed absent. The
resident worker has resumed its idle task with no serving/Store process left.
All twelve changed Python files compile, format and match the GPU checkout.
Seven pass Ruff; the five remaining files retain 29 diagnostics also present in
HEAD. The original checkout's staged-index digest is unchanged. Commands,
outcomes, source/log hashes and scope limits are recorded in
`experiments/capture-ragged-dspark.json`. Production Catalog/consumer integration,
trained checkpoint quality, broader topology validation, PD/RDMA and P10 SLO
acceptance remain open.

## PD First-Teacher Handoff

The P/D path now selects one publication owner: D. Before publishing destination
addresses, D reserves the existing bounded Host slot and Catalog capture lease.
An optional eleventh Mooncake metadata frame binds the room, prompt, target/KV
contract, sampling parameters, sample/generation and fencing token. P captures
one owned raw top-128/LSE row before sampling. It opens neither a Store client
nor a full prompt Host capture arena. The final KV chunk carries an immutable
handoff, sent over the existing locked control socket before the success
notification; the registered KV/aux buffer layout is unchanged.

After the normal KV/metadata completion gate, D validates the handoff and first
output token, exports its complete canonical prompt prefix, and continues the
ordinary AR ledger. A one-token reply also includes the first teacher row.
Missing/invalid handoffs fail capture while generation continues. Conflicting
duplicates poison the handoff, and cleared-room messages cannot retain state.
Abort/retract and lease invalidation reuse the existing D cleanup path. Fake
warmup/health-check transfers are explicitly excluded; actual server startup
found this required guard, and a unit regression now protects it.

This milestone initially permitted Mooncake AR with TP=PP=DP=1 and no optimistic
prefill. The TP extension below supersedes that TP restriction. PD speculation,
pipeline parallel, cross-node RDMA, and PD cache/rebootstrap stress remain open.
The design's section 13.3.1 records the wire and ownership decision,
compatibility behavior and configuration requirements.

The complete `test_training_capture_pd.py` passes both tests in 134.189s on the
resident H100 (job `01790876096551927410-528c095406d8`). Each mode starts separate
P and D processes and publishes five samples: one-token response, chunked
prefill, reused prompt and a two-request batch. A test-only online observer
retains full raw logits and source KV. The independent Store reader verifies
KV and top-128 values exactly, full-vocabulary LSE within 1e-6, token IDs,
positions, loss masks and KV validity. No reference target forward is rerun.
The overlap case records 18 actual CUDA graph forwards. Missing handoff, stale
fence and stream cancellation each exclude a sample; generation completes for
both handoff faults. Both modes report five READY, two handoff failures, one
abort failure and no quarantined Host slots.

Seven PD unit tests, eight configuration tests, 46 coordinator tests, the real
Gloo cohort-startup test, 22 existing wire tests and six decode cleanup tests
also pass as separate files. All fourteen changed Python files compile and
pass Black, isort and the repository's Ruff F401/F821/UP037 check. New modules
also pass the broader Ruff rules with import ordering delegated to isort.
The final test-only cleanup comment was reformatted after execution. A subsequent
P-side custom-logit-processor exclusion passed the full seven-test PD unit file;
the supported AR path is unchanged. The synchronized GPU checkout matches all final
source hashes. Registered-test validation passes. Process inspection finds no
live serving or Store process; the resident idle workload resumed. The original
checkout's staged-index digest is unchanged. Detailed results and log/source
hashes are in `experiments/capture-pd-handoff.json`.

The Catalog remains an HTTP test double and transport is TCP. These results do
not certify production retention, a production SpecForge consumer, distributed
PD, RDMA, trained draft quality, or P10 performance acceptance.

## TP PD Cohort Publication

PD AR now reuses the distributed decode capture coordinator. All D ranks bind
the existing ingress ticket to one cohort lease before sending destination
metadata. The shared PD import validates the first teacher and committed token;
each KV owner exports its canonical prompt head range and only the aux owner
stores teacher/positions. Aux-only and inactive ranks advance the same ledger.
The existing all-owner descriptor/receipt agreement gates manifest publication,
so one rank's failed handoff excludes the complete sample.

P/D TP sizes may differ because the wire binds global teacher/KV identity, not
the local ownership layout. P requires matching contexts from all non-dummy
destinations. P1/D2 imports one handoff into each D rank; P2/D1 deduplicates
identical teacher payloads and rejects conflicting payloads. No Mooncake SDK or
registered serving KV-buffer layout change is needed. PP=DP=1, AR, Mooncake and
no optimistic prefill remain required.

The final runtime files all pass on real Qwen3-0.6B P/D processes and TCP Store:

| Test File | Topology | Tests | Seconds |
| --- | --- | --- | --- |
| `test_training_capture_pd.py` | P1/D1 | 2 | 133.291 |
| `test_training_capture_pd_tp_expand.py` | P1/D2 | 2 | 136.482 |
| `test_training_capture_pd_tp_reduce.py` | P2/D1 | 2 | 139.462 |
| `test_training_capture_pd_tp.py` | P2/D2 | 2 | 136.128 |

Every file covers eager and actual graph/overlap execution. The eight cases
publish 40 snapshots whose KV, raw top-128 scores/IDs, LSE, token IDs, positions,
loss masks and validity match independent online source observations. The
24 selected missing-handoff, stale-P-rank-0 and stream-abort requests publish
no snapshot, and handoff faults do not stop serving. Both ends use radix cache;
the reused 153-token prompt verifies complete prefix reconstruction. A test-only
ready-queue barrier makes the tagged pair form an actual decode batch.

Eight PD unit tests include TP4 replicated heads with separate KV/aux owners
and an inactive rank. Eight configuration tests, 46 coordinator tests, three
cohort-coordinator tests and one four-process startup rollback test pass as
separate files. The final total is 74 tests. Eleven changed Python files pass
Black, isort, repository Ruff checks and compilation; new helpers/tests also
pass broader Ruff rules with import ordering delegated to isort. Registered
test validation passes, and the GPU checkout matches the final source hashes.

Two initial fixture failures are retained in the evidence: distributed Catalog
failures use `cohort_failed`, and a batched HTTP call does not guarantee the
two transfers become runnable together. The corrected tests also require
admission, cancellation, drained capture work and zero quarantined Host slots.
All final runtime files were executed after those test changes.

The temporary two-H100 job `job-ad7074c7546a-20261002020118` on node199 was
deleted after verifying no live serving/Store process remained; its pod is
NotFound. The resident single-H100 worker has no queued/active experiment and
has resumed idle load. The original checkout's staged-index digest is unchanged.
Full commands, results, runtime counters, failed attempts and source/log hashes
are recorded in `experiments/capture-pd-tp.json` and its referenced logs.

This extends TCP correctness coverage. It does not certify PD PP/speculation,
DP, RDMA, deployed TP4 replicated heads, production Catalog retention, trained
draft quality or workload performance acceptance.

## PP PD Teacher Circulation And Transfer Readiness

AR Mooncake PD capture now supports pipeline stages. Only the last P stage
captures the raw first-teacher row. It encodes a bounded immutable handoff into
the existing sampled-output ring, where each P stage validates the capture
context and batch alignment before attaching it to its final KV transfer.
Encoding releases the owned GPU teacher; recomputing the final prefill replaces
the cached message. D reuses cohort ownership: each stage exports its selected
global layers, the last stage's aux owner stores teacher metadata, and all-owner
receipts gate publication. Serving KV/aux buffer registration is unchanged.

This also fixes a PP transfer-readiness race exposed by the first real request.
The old PP consensus checked receiver Success without checking metadata.
PP0 consumed the request while PP1's bootstrap-room metadata was still zero;
PP1 deferred the import and could never form another intersection after PP0
removed the request. PP consensus now applies the existing metadata gate before
the attention TP/CP reductions. No stage consumes a successful transfer until
all stages have its metadata. Failed and fake transfers retain their behavior.
Three failed diagnostic attempts and the final passing runs are retained in
`experiments/capture-pd-pp.json` with log hashes.

The following separate files pass against real Qwen3-0.6B P/D workers and an
independent Mooncake TCP Store reader:

| Test File | P / D Topology | Tests | Seconds |
| --- | --- | --- | --- |
| `test_training_capture_pd_pp.py` | PP2 / PP2, TP1 | 2 | 144.554 |
| `test_training_capture_pd_pp_reduce.py` | PP2 / PP1, TP1 | 2 | 136.074 |
| `test_training_capture_pd.py` | PP1 / PP1, TP1 | 2 | 134.186 |
| `test_training_capture_pd_tp.py` | TP2 / TP2, PP1 | 2 | 169.943 |

The eight eager/graph cases publish 40 complete snapshots and exclude 24
selected missing-handoff, stale-handoff and cancelled requests. Source KV,
raw top-128 scores/IDs, full-vocabulary LSE, token IDs, positions, loss mask and
validity are compared with online observations, without rerunning the target.
Coverage includes a one-token response, chunked prefill, prefix reuse and an
actual two-request decode batch. PP disables overlap as required by serving;
the single-stage D graph cases exercise overlap. Runtime assertions require
D rank 0 to report zero Host quarantines and drained work after cancellation.

Thirty unit tests pass across five separate files: 11 PD capture tests,
three PP metadata-readiness tests, six decode cleanup tests, eight configuration
tests and two PP/CP rank-offset tests. They cover delayed metadata, reduction
ordering, failed/fake transfers, bounded PP teacher messages, malformed/misaligned
rows, foreign contexts, first-token mismatches and recomputed teacher ownership.

The complete `test_training_capture_distributed.py` regression file also passes
four tests in 432.993 seconds: ordinary TP/PP, target-KV DSpark, and cap-accept
and compact ragged DSpark under eager and graph/overlap execution. The final
total is 42 passing tests. All 12 changed Python files pass Black, isort,
repository Ruff checks and compilation; registered-test validation passes.
New helpers/tests also pass broader Ruff with import ordering handled by isort.
The GPU checkout matches the final source hashes.

The temporary two-H100 job `job-d1c8e138f33a-20261002023845` was deleted after
verifying no live serving/Store processes remained; its pod is NotFound.
The resident H100 has no active or queued experiment and has resumed idle load.
The original checkout's staged-index digest is unchanged. Commands, counters,
source/log hashes and cleanup observations are in `experiments/capture-pd-pp.json`.

The existing Mooncake transport permits matching P/D PP sizes or D PP=1.
P PP1 to D PP2 is still unsupported by that transport. The runtime evidence
does not certify combined TP2/PP2, PD speculation, DP/CP, cross-node RDMA,
production Catalog retention, trained draft quality or performance acceptance.

## Next Implementation

1. Broaden real-request coverage to prefill graphs, automatic AR OOM retraction,
   speculative cache eviction, target weight replacement and
   saturated backpressure.
2. Extend P8's passing retained BF16 fixture to production-exported and trained
   checkpoints, complete exporter compatibility and artifact/quality validation.
3. Extend P9's real TP2/PP1 and TP1/PP2 Qwen3 capture validation to combined
   TP2/PP2, replicated heads, distributed cancellation/backpressure and additional
   model identities. Complete pipeline speculative collection,
   speculative PD and cross-node RDMA. AR PD now transfers the
   first teacher row and publishes from D across matching/asymmetric TP groups
   and matching/reduced PP groups supported by the Mooncake transport.
   Ordinary AR uses the distributed serving
   path, and static and confidence-scheduled DSpark capture support TP. The
   target-KV v1 draft remains static by checkpoint contract; the remaining
   capability gates do not constitute implementation of those paths.
4. Reduce P10's measured capture overhead, extend capture-on/off benchmarks to
   representative workloads and SLO thresholds, and complete dashboard runtime
   acceptance and rollout/rollback checks. Per-model numerical/runtime validation and
   runtime identity coverage also need expansion beyond the tested combination.
5. Integrate with the SpecForge-owned production Catalog and consumer when
   available. Test doubles do not prove retention, consumer checkpoint replay,
   training loss correctness or actual draft-model training quality.
