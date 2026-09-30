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
- Runtime: Python 3.12.3, PyTorch 2.11.0+cu130, CUDA 13.0,
  Transformers 5.12.1. Tests select this checkout using `PYTHONPATH=.../python`;
  the image's installed SGLang 0.5.15 is not the code under test.
- Resident allocation: `job-fe1ce1dcdea6-20261001023258`, one H100 80GB on
  initial node `node064`. Experiment submissions pause/resume the idle load.

## Current Evidence

| Area | Implemented | Evidence / Remaining Work |
| --- | --- | --- |
| Wire contract | Typed manifest, raw tensor descriptors, shape/byte/digest/coverage/content validation | Generated fixtures pass the design's JSON Schema; malformed metadata and contents are rejected |
| Raw teacher primitive | Unpadded top-128 IDs/values and full-vocabulary LSE, independent output storage | CPU reference and H100 source-mutation checks pass; runtime hook is pending |
| KV export primitive | Selected layers, arbitrary source slots, NHD BF16/FP16, independent D2H copies | H100 source-reuse check passes; request/prefix/retract integration is pending |
| Host ownership | Bounded registered arenas, quota rejection, reuse, transfer quarantine | Ownership/backpressure tests pass; runtime admission/renewal is pending |
| Mooncake adapter | Required hard pin, registered raw buffers, immutable retry verification, exact read length | Real cross-process TCP roundtrip passes; cross-node RDMA is pending |
| Publication | Catalog producer client, manifest-last writer, durable metadata journal, fenced replay | Lost seal/publish responses, failed tensor puts, stale fence and identical retries tested; actual Catalog service is SpecForge-owned |
| Runtime collection | Not yet connected | Add config/capability gates, request position ledger, prefill/decode hooks, abort/retract/shutdown handling, metrics |
| Real model parity | Not yet run | Bind immutable target/tokenizer/codec identity and compare captured KV/logits against a reference |
| Draft serving | Not yet implemented | KV input checkpoint contract, injector, invalidation and training/serving parity |
| Deployment coverage | Not yet implemented | TP/PP, overlap, speculative/PD, RDMA and workload SLO gates remain open |

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

These checks do not prove a MaaS request is being captured, full training
correctness, retention correctness of an external Catalog, or RDMA performance.

## Next Implementation

1. Add an immutable request position ledger and a bounded background coordinator
   that obtains/renews Catalog leases without blocking inference.
2. Wire raw capture before serving processors and KV gather before source reuse;
   propagate immutable capture tickets through result processing.
3. Bind explicit model/tokenizer/layout identities, reject unsupported runtime
   paths, and invalidate capture across weight updates.
4. Exercise actual prefill/decode, prefix hits, chunk boundaries, single-token
   responses, cancellation and backpressure on the resident H100.
5. Complete the remaining SGLang serving and deployment work packages from the
   design. Keep unsupported paths explicit until their corresponding gates pass.
