# Weight Isolation Across P/D

P and D bind the same capture contract, including the target weight and tokenizer
digests. D owns the capture lease and complete snapshot. The first raw teacher
row comes from P through a fenced handoff; the transferred prefix KV and D's
later decode rows must belong to that same contract.

## Behavior

| Change | P behavior | D behavior |
| --- | --- | --- |
| Live P-only weight load | Persistent capture invalidation; omits pending teacher handoffs, including already materialized/encoded rows | Fails the selected attempt with `pd_handoff_failed` |
| Live D-only weight load | Can finish its existing valid handoff | Fails its active attempt with `target_weights_update` and rejects further capture admission |
| Capture resume after either load | Does not clear the local identity fault | Does not rebind model identity |
| Fresh P/B and D/A | Rejects D's mismatched contract before preparing a teacher row | Fails the incomplete handoff; never publishes a snapshot |
| Fresh P/B and D/B | Produces a matching handoff | Can publish a complete snapshot under B's identity |

These capture failures preserve inference progress. MaaS deployment and routing
still own target-version consistency for serving: an in-place-paused request can
retain old KV across a weight update. Collection deliberately excludes that
request. Updating P alone does not remotely disable D's capture coordinator;
subsequent selected attempts fail until matching, freshly bound producers are
available. Resource limits and failure handling continue to apply.

## Runtime Check

```bash
python test/registered/unit/training_capture/test_pd_capture.py -v
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd_weights.py -v -f
```

The runtime uses separate P/D processes colocated on one H100, real Mooncake KV
transfer and Store TCP, and an HTTP test Catalog. It covers both update roles in
synchronous eager and overlap decode-graph execution. Each case:

1. Captures a baseline using A/A, then holds a 200-token prompt after its first
   prefill chunk using the existing test-only scheduling gate.
2. Pauses the selected endpoint through HTTP and loads B from disk while D has
   an active capture. B differs by a negated layer-zero V projection. Checksums
   of the actual loaded parameter must match that mutation.
3. Resumes generation and releases the chunk gate. The request must complete;
   its exact capture lease must fail with no registered or written objects and
   no READY entry. A P-only update must omit its handoff; a D-only update permits
   P's old handoff but must exclude it from capture.
4. Tries capture resume and another request, requiring persistent exclusion.
5. Restarts both endpoints as P/B and D/A, requiring another exact failed lease
   and an observed P-side contract rejection while both coordinators are enabled.
6. Replaces D/A with D/B and requires collection to recover. After all producer
   processes exit, creates a new Store reader and validates both A/A and B/B
   snapshots against their online KV and raw teacher tensors, plus schema,
   hashes, positions, masks, top-128 IDs/values and full-vocabulary LSE.

The unit tests cover P invalidation before teacher computation, after copying,
and after encoding, plus D invalidation followed by a late handoff. They also
verify that resume cannot restore admission under the old identity.

## Recorded Run

On 2026-10-03, all four GPU cases passed in 857.569 seconds. They exclude ten
exact capture leases and validate eight complete snapshots after producer exit:
112 tensor objects and 1,952,192 payload bytes. The 17 P/D unit methods also
pass, including the new version invalidation cases. No production bug or code
change was required by these checks.

[The machine-readable report](pd-weight-update.json) records each case, source
versions, commands, model revisions and environment. Its archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/pd-weight-update-20261003` with 29
artifacts and manifest SHA-256
`f29119fc9a06b36de64400c0424874f4d3cdf2283ebff1a8b91f93e5f1b76b30`.
All 5,031 Python files match their tested source: the runtime used v1; v2 only
binds a unit-test closure default explicitly for lint, and reruns all 17 methods.
The resident GPU worker has returned to its idle task with an empty queue.

## Scope

This validates the existing capture policy and adds no production scheduling or
weight-loader behavior. The runtime uses AR, TP1/PP1 endpoints and disk updates.
It does not validate distributed/tensor/IPC weight transport, TP/PP replacement,
cross-node weight rollout, target-KV DSpark deployment, production Catalog
retention, consumer checkpoint replay or serving SLOs.
