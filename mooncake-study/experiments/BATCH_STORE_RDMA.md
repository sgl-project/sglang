# Native Batch Reads And Writes Over RDMA

This follow-up validates the native batching added to `MooncakeSnapshotStore`
against a storage segment on another physical node. It uses the existing
[remote RDMA lane](RDMA.md), Qwen3-0.6B online capture and the current frozen
source. P and D share the producer GPU and use TCP for their KV handoff; only
the snapshot Store payload path is under RDMA validation here.

## Transport Probe

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 MC_TE_METRIC=1 \
  MC_TE_METRIC_INTERVAL_SECONDS=1 \
  python -m sglang.test.training_capture_rdma probe \
  --setup /path/to/client.json --batch
```

The probe registers one 4 MiB pinned Host arena with three independent object
views of 1, 2 and 1 MiB. The first write creates one object; the second verifies
that object and writes only the other two; the final immutable retry reads back
all three without issuing another write. A wrapper counts calls to the real
SDK and checks mandatory hard pinning. Two native write calls and two native
verification reads must occur.

The writer client then closes and the whole arena is zeroed. A new client
receives all objects using `get_tensors`. A separate batch mixes two existing
objects with one missing key: the successful destinations must be unregistered,
and only the missing 256-byte receive buffer remains quarantined. A wrong digest
must fail without retaining another completed buffer. `verify_tensors` must
still work using fresh destinations. Normal lease-protected removal cleans up
the three probe keys, and successful client close clears the quarantined memory.

## Online Snapshots

The registered `test_training_capture_rdma.py` runs AR eager/graph and static
target-KV DSpark eager/graph. Its existing independent online observer checks
selected-layer KV and raw top-128 IDs/logits, full-vocabulary LSE, tokens,
response masks, positions and committed boundaries. Cancelled requests and
missing/stale teacher handoffs remain excluded from publication.

Each scenario stops its serving processes before a separate interpreter reads
all published samples. This reader now uses one native payload batch per
snapshot, verifies complete manifest coverage and content, and requires no
leftover receive registrations or quarantine. A final independent invocation
can repeat the read after the entire serving suite exits:

```bash
env MC_STORE_MEMCPY=0 MC_GID_INDEX=3 \
  python -m sglang.test.training_capture_rdma read \
  --setup /path/to/client.json --publications /path/to/publications.json \
  --output /path/to/readback.json --batch
```

The shared filesystem contains only configuration, manifest references and test
evidence for this readback. Snapshot tensor payloads are retrieved from Store.
All writer/read clients have `global_segment_size=0`; only the remote Store
mounts a 256 MiB segment. The explicit `rdma` protocol, `mlx5_00`, GID index 3
and `MC_STORE_MEMCPY=0` prevent this test from passing through local-copy reads.

## Resource Constraints

The retained resident H100 uses an isolated network namespace and has no RDMA
allocation. A temporary zero-GPU two-node request was attempted first. The
platform added `cpu_node=true`, and those nodes expose no RDMA resource. The
current Kubernetes account cannot patch Jobs, so that request was deleted.
The supported temporary request uses two host-network nodes with one H100 and
one RDMA resource each. The Store node does not run a model; its GPU allocation
is a platform constraint of this experiment, not a Mooncake requirement.

Only these temporary containers' inherited FlashInfer 0.6.12 cubin/JIT-cache
packages are removed, leaving the locked 0.6.17 runtime. The resident worker and
its idle load remain independent. Experiment commands, process exits, node/GPU
identities and cleanup are retained in the evidence archive.

This is correctness evidence for the batch transport path. It does not prove
saturated RDMA throughput, GPUDirect, cross-node P/D for this run, production
Catalog retention, a trained draft's quality or serving SLOs. The Catalog is a
test double and the serving draft fixture is untrained.

## Verified Results

The 2026-10-03 run uses producer node208 and Store node199, with explicit
`mlx5_00` RDMA and a 256 MiB remote segment. The probe passes all assertions:
two native write batches, two immutable-retry verification batches, four
independent reader batches and exactly 256 quarantined bytes on the partial
failure. All three probe keys are removed without force. Native TransferEngine
logs report positive transfer throughput; this is not a bandwidth benchmark.

All four real Qwen3-0.6B serving cases pass in 653.466 seconds. AR eager/graph
publish five samples each, and target-KV DSpark eager/graph publish six each.
Together with the draft seed, cumulative post-exit reads cover 5, 10, 17 and 23
snapshots. The final fresh reader validates 23 snapshots, 418 tensor objects and
16,960,108 payload bytes, with exactly 23 native payload read calls. Every case
reports zero quarantined capture Host slots; independent readers retain no
receive registrations or quarantine after successful reads.

The producer, Store and Master exit successfully before cleanup. Both temporary
nodes have no GPU compute processes before deletion, and both temporary job
pod sets are confirmed absent. The resident worker and its idle task remain
live with no queued experiment.

The [evidence JSON](batch-store-rdma.json) records commands, SDK/runtime
versions, per-case counters, source hashes and cleanup. All 5,031 Python files
match the tested frozen source. The 36-artifact archive is
`/gpfs/user/fuxuanwei/mooncake-lab-archive/batch-rdma-20261003`; its artifact
manifest SHA-256 is
`a54754fc3881dd6da33564e1bca2a34bc37ce123ea7cd66c90c8f2ce1a02d844`.
