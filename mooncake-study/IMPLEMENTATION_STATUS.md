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
| KV export | Selected layers, arbitrary source slots, NHD BF16/FP16, independent D2H copies | H100 source-reuse test and exact online attention-input comparison pass, including chunked prefill, prefix hits and decode |
| Host ownership | Bounded registered arenas, quota rejection, reuse, transfer quarantine | Coordinator admission, renewal, expiry, retract, shutdown and publication tests pass; traffic-scale stress remains open |
| Mooncake adapter | Required hard pin, registered raw buffers, immutable retry verification, exact read length | Real cross-process TCP roundtrip passes; cross-node RDMA is pending |
| Publication | Catalog producer client, manifest-last writer, durable metadata journal, fenced replay | Lost responses, failed puts, stale fences, missing/corrupt objects and identical retries tested; actual Catalog service is SpecForge-owned |
| Runtime collection | Opt-in CLI config, capability gates, request ledger, prefill/decode hooks, invalidation and counters | Six real Qwen3-0.6B requests published through Mooncake; ordinary and CUDA graph replay executions pass |
| Real model identity/parity | Weight/tokenizer artifact digests, actual selected-layer geometry, K norm and RoPE | Captured KV and teacher scores match online tensors exactly; full-vocabulary LSE matches within 1e-5; HF teacher logits pass numerical comparison, but cross-engine KV equivalence is not certified |
| Draft serving | Explicit KV-input architecture, contract, encoder, incremental injector and invalidation | Real Qwen3 target plus synthetic KV draft passes ordinary/batched/graph generation and per-layer projected-KV checks; full SpecForge backbone/logit parity remains open |
| Speculative collection | Static DSpark raw verify ticket, commit mapping and terminal truncation | Actual KV-input draft requests publish and read back through Mooncake in ordinary and graph modes; see evidence below |
| Overlap collection | AR lookahead ledger, capacity boundary and terminal trimming | Real ordinary/graph requests, padded batches, prefix remapping, delayed grammar and abort checks pass; speculative overlap remains open |
| Deployment coverage | Partial | TP/PP, speculative overlap, non-static speculative verify, PD, RDMA and workload SLO gates remain open |

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
verify, one GPU, no overlap/PD/LoRA and dense unquantized pools. Static speculative
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
`overlap_forwards` counter. Speculative overlap is still rejected separately.

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
they do not certify speculative overlap or multi-rank ownership.

These observers synchronously read raw outputs and pool rows to validate data;
they do not establish overlap performance, latency SLOs or arbitrary-backend
numerical equivalence. The production capture path still uses its asynchronous
copies and background writer. Catalog remains a test double and Store transport
remains TCP.

## Next Implementation

1. Broaden real-request coverage to prefill graphs, real retraction, cache eviction,
   target weight replacement, speculative cancellation and saturated backpressure.
2. Complete P8's fixed-input backbone/logit parity against the training side,
   exporter compatibility and artifact/quality validation. Broaden the runtime
   lifecycle tests to cache eviction, cancellation and real request retraction.
3. Complete P9's topology work: TP/PP, speculative overlap, non-static speculative layouts,
   PD transfer and cross-node RDMA. Existing capability gates do
   not constitute implementation of these paths.
4. Complete P10's adaptive capture limits, metrics, capture-on/off SLO benchmarks
   and rollout/rollback checks. Per-model numerical/runtime validation and
   runtime identity coverage also need expansion beyond the tested combination.
5. Integrate with the SpecForge-owned production Catalog and consumer when
   available. Test doubles do not prove retention, consumer checkpoint replay,
   training loss correctness or actual draft-model training quality.
