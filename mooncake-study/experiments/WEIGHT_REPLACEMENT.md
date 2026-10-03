# Capture Across Weight Replacement

A training sample must bind to one target identity. Ordinary AR may update its
serving weights in place, but a capture spanning that update must never become
READY. The current implementation disables collection before disk, distributed,
tensor or IPC loading and fails active attempts with `target_weights_update`.
It does so even when loading fails: a failure is not proof that every parameter
is unchanged. This document validates that existing behavior; it introduces no
production loader or scheduling change.

## Operational Contract

1. Keep model and tokenizer artifacts immutable for each producer lifetime.
   Pin `expected_weights_revision` and `expected_tokenizer_revision` to the
   capture artifact digests when deploying a known version.
2. After an ordinary AR weight update, inspect `training_capture.disabled_reason`
   in `/server_info`. Generation may continue, but collection stays disabled.
   Capture `resume` does not rebind identity, revalidate weights or restore
   eligibility. Changing the serving `weight_version` label is insufficient.
3. Already sealed and detached snapshots can finish asynchronous copying and
   publication under their original identity. Failed active attempts never
   publish. Existing READY objects remain independent of the producer process
   when a separate Mooncake data node owns storage.
4. Start a new producer using the new immutable artifacts and a compatible
   capture configuration and journal directory. Its artifact binding must pass
   before admission resumes. Target-KV DSpark requires a compatible target/draft
   deployment; its live replacement route is rejected before mutation.

This protects the training dataset. It does not make an in-place-paused serving
request on-policy across a weight change. Such a request can retain old KV while
using new weights; rollout policy determines whether to drain, abort or retract
it. The test deliberately retains that request to prove its capture is excluded.

## Validation

```bash
python -m unittest discover -s test/registered/unit/training_capture -p 'test_weight_updates.py' -v
python test/registered/unit/training_capture/test_coordinator.py -v
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_weights.py -v -f
```

The CUDA test uses one GPU and real SGLang HTTP, Mooncake master and SDK TCP
operations. Its HTTP Catalog is a test double. It covers synchronous eager
execution and overlap scheduling with decode CUDA graphs.

The fixture derives version B from immutable local Qwen3-0.6B weights by negating
the layer-zero V projection, preserving model and tokenizer configuration. It
does not modify the source model directory. Each case:

1. Publishes one complete sample from A.
2. Begins another request and holds it after one committed token using the real
   scheduler's in-place pause method in a test-only entrypoint. A separate client
   thread owns the generation request; HTTP pause sets tokenizer update state.
3. Calls the real disk update endpoint with B. Before/after checksums of the
   loaded V projection must match the intended negation. The active Catalog
   lease must fail, and its owned Host slot must retire without quarantine.
4. Resumes and completes the interrupted request. Two additional requests,
   including one after capture `resume`, must not produce more samples.
5. Stops the old producer, launches B with B's expected artifact digest, and
   publishes a fresh sample. After B also exits, a newly connected Store reader
   validates both manifests and all payloads, including exact online KV and raw
   top-128 logits, vocabulary IDs, masks, positions and full-vocabulary LSE.
   Weight revisions and fingerprints must differ; tokenizer identity must match.
   Both captured V and teacher score tensors must differ between A and B.

The unit tests exercise pre-mutation ordering for all four routes on success,
loader failure and partial-load exceptions, and no-mutation rejection for
target-KV DSpark. Coordinator tests hold an already detached sample at the copy
completion fence and require it to finish publishing under its original identity
after invalidation, without permitting a new capture.

## Recorded Run

On 2026-10-03, the two updater unit methods, 62 coordinator methods and both
GPU methods passed. The GPU run takes 179.390 seconds and validates four
post-exit snapshots: 56 objects and 976,096 tensor bytes. It uses the existing
H100 allocation, which returns to its idle task with an empty experiment queue.

[The report](weight-update.json) records the commands, immutable source snapshots,
source audit, model digests, runtime state, object counts and environment. The
53-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/weight-update-20261003`, with manifest
SHA-256 `bc6698b0128945432a1d16e502cf6b58a2a91d559754df969bca39ebab120a4d`.
All 5,028 final Python sources match the tested snapshot. The archive also
retains the four failed test-driver iterations and their corrections; none is
represented as an observed production bug.

## Scope

The disk replacement runtime check is single-rank Qwen3 AR. Mocked loader tests
do not validate distributed parameter transport, tensor serialization, CUDA IPC
weight exchange or an actual DSpark replacement deployment. TP/PP and P/D weight
replacement, production Catalog retention, checkpoint replay, trained draft
quality and serving SLO acceptance remain separate requirements.
