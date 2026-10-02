# Distributed P/D Capture Control

This lane extends the [single-rank P/D control matrix](PD_CAPTURE_CONTROL.md)
to matching TP2/PP1 and TP1/PP2 prefill/decode groups. P and D are independent
HTTP services sharing two H100s. Each topology runs ordinary AR and static
target-KV DSpark in eager and decode CUDA graph modes. TP2 graph execution
enables overlap; PP2 remains synchronous, matching its supported serving path.
Prefill graphs are disabled in this lane.

## Rank-Level Evidence

The test-only server records observations under separate PP/TP rank directories.
A filesystem nonce accompanies an ordinary `/server_info` request; each rank's
actual scheduler handler writes its current capture state with that nonce. The
test waits for every expected rank and rejects stale observations. This adds no
production route or distributed synchronization.

Resume clears the manual pause before the background readiness vote restores
available tickets. Before a request that must be captured, the test waits for
each D rank's available count and local service readiness. Paused requests still
run immediately. An initial run omitted this wait and correctly generated a
response without a capture; this was a test admission race, not partial
publication. The test-only readiness observation does not alter `/server_info`.

After each real `POST /control_training_capture` request, the test verifies the
pause flag, absence of a disable reason, and cumulative action counts on every
rank of the controlled role. Thus a successful ingress reply alone cannot pass.
The HTTP response itself retains its existing asynchronous semantics and is not
an all-rank drain or Store cleanup acknowledgement.

Each partial-prefill gate must be reached by every P rank before a control is
sent. Gate release uses a test-only numeric key in `/set_internal_state`, handled
only by the observer entrypoint, so the release follows the ordinary TP/PP
request-message sequence. Independently polling a removed file on each PP rank
can resume different pipeline iterations; an initial test did this and timed
out. The release acknowledgement is checked on every rank before removing the
gate file. This key is not a production scheduler setting.

After release, observations check each rank's teacher-handoff payload and
capture epoch. D-only abort must discard a valid late payload; P abort must
suppress old payloads even after resume. Both-endpoint pause drains an already
admitted sample. P-only pause fails capture while ordinary generation continues.
Aborting during decode also leaves the full generation response intact.

## Publication Checks

Every successful cell expects eight publications. All D ranks must drain their
local active/queued/writing states without quarantine, admission backpressure or
writer errors. Only the auxiliary owner reports `ready` publications; the test
sums this count across ranks and checks the Catalog independently. Distributed
failures retain the existing Catalog reason `cohort_failed`, while local
coordinator counters record the concrete failure source.

Every D rank records the capture ID assigned to each request at the original
admission hook. Failure checks require that exact ID to fail with no publication.
Distributed abort can additionally fail spare unbound tickets; any such extra
ID must be absent from all request-admission observations, have no registered or
written objects, and never have published. An initial test assumed one Catalog
failure per abort and was corrected to distinguish requests from spare tickets.
Final drain also waits until every rank has retired invalid cohort handles, then
reconciles all new Catalog failures against these request and spare-ticket IDs.

After both producer process trees exit, a new Mooncake client reads every
published sample. Manifest topology must match the requested TP/PP layout.
Exact canonical KV values across all selected layers/heads, token IDs, masks,
positions and raw top-128 IDs/values are checked against independent online
observers; LSE uses the existing `rtol=atol=1e-6` comparison. Missing rank data or
an incomplete manifest cannot pass these checks.

## Results

The final frozen `sglang-pd-distributed-control-v4` source passes **eight methods
in 788.538 seconds** on two H100s. Each cell completes 15 generation requests,
validates eight complete snapshots after producer exit, fails the five intended
request captures, and cleans up nine unbound tickets without payload writes.
All rank-local controls and graph/overlap checks pass, with no disabled capture,
quarantine, admission backpressure, writer-stage errors or Catalog errors.

| Topology On Both Roles | Decode | Execution | Methods | Post-Exit Snapshots |
| --- | --- | --- | --- | --- |
| TP1/PP2 | Static target-KV DSpark | Eager and synchronous graph | 2 | 16 |
| TP2/PP1 | Static target-KV DSpark | Eager and graph + overlap | 2 | 16 |
| TP1/PP2 | AR | Eager and synchronous graph | 2 | 16 |
| TP2/PP1 | AR | Eager and graph + overlap | 2 | 16 |

The final single-GPU regression passes **four methods in 449.052 seconds** with
32 additional post-exit snapshots. Together the final runs cover **12 methods,
180 generation requests and 96 snapshots**. Synthetic draft-contract seed
samples, earlier regression passes and failed setup attempts are excluded from
these totals. The serving implementation remains `aede6f2be`; this change adds
test infrastructure and validation, with no production module changes.

- Distributed final job: `01790961918886501970-e5c3fa26eb7b`.
- Single-rank final job: `01790961919223101261-07aec6da21d5`.

The evidence index retains all seven terminal jobs: these two final successes,
two earlier single-rank successes, and three setup failures described above.
All three Python files pass Black and full Ruff. Changed Python files and the
listed runtime dependencies match the frozen source; all `.py` files under
`python/` also match. Documentation was updated after the runs.

The temporary allocation `job-f63122ef7493-20261003005634` ran on node199. Before
deletion the worker was idle, the GPU process list was empty, and no serving or
Mooncake Master process remained. Northjob deletion succeeded and the Pod was
confirmed absent. The original resident H100 remains allocated with idle load
resumed. Commands, per-mode observations, source/log hashes and cleanup evidence
are in [the evidence index](distributed-pd-capture-control.json).

## Reproduction

With the locked capture environment, two visible CUDA devices, a local
Qwen3-0.6B model and `mooncake_master` available:

```bash
export PYTHONPATH="$PWD/python"
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd_distributed_control.py -f -v
```

The shared-helper single-GPU regression runs separately:

```bash
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
  python test/registered/storage/test_training_capture_pd_control.py -f -v
```

## Limits

These are actual Mooncake TCP P/D and Store operations with a test HTTP Catalog.
The Store data segment belongs to the test process and outlives the producers.
The DSpark checkpoint is synthetic and tests execution/ownership, not model
quality. Source observers copy tensors and write files, so elapsed test time
does not measure serving performance. This lane does not establish mixed TP/PP,
asymmetric P/D control, cross-node/RDMA control, confidence-scheduled DSpark,
production Catalog retention, deployment SLOs or a global drain barrier.
