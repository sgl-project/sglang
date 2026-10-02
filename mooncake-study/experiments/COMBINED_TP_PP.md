# Combined Tensor And Pipeline Parallel Capture

The four-GPU fixtures exercise TP2/PP2 against a real BF16 Qwen3-0.6B target,
synthetic target-KV DSpark checkpoints and Mooncake TCP. The Catalog remains an
HTTP test double. Run complete files with matching SGLang dependencies, the
Mooncake SDK and `mooncake_master` on `PATH`:

```bash
PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_tp_pp.py -v -f

PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_pd_dspark_tp_pp_pressure.py -v -f

PYTHONPATH=python \
TRAINING_CAPTURE_TEST_MODEL=/path/to/Qwen3-0.6B \
python test/registered/storage/test_training_capture_dspark_tp_pp.py -v -f
```

## Execution And Ownership

Each serving instance has four workers, ordered PP-major. TP splits logical KV
heads within a stage; PP divides target layers. The selected layers 0/14/27
exercise both stages, and both tensor ranks own selected KV shards. The final
stage's TP0 owns auxiliary tensors and teacher rows. The existing cohort writer
collects owner-local receipts before publishing one complete manifest.

P/D uses matching TP2/PP2 on each side. Its independent P and D instances share
the four available devices in this correctness fixture. AR and static DSpark
each exercise eager and graph execution; DSpark runs with the draft on D alone
and with drafts on both sides. The test includes single-token responses,
chunked prefill, cached prefixes, rejected draft proposals, real multi-request
batches, missing/stale teacher handoffs and live cancellation.

The target-KV injector gathers the selected logical heads across each owning TP
group, then broadcasts the assembled layer through each PP group. It uses each
rank's local physical slot mapping. Snapshot export remains local to the owner;
these inference collectives do not turn every worker into a snapshot owner.

The colocated fixture reuses the existing pipeline suite with TP2. It checks
greedy output against AR at the same topology, prefix reuse, a mixed batch,
grammar, sampling penalties, stop-token trimming, cancellation and natural
pool exhaustion. Online source and projection observations must contain every
`(tp_rank, pp_rank)` pair. Published manifests must report the actual TP2/PP2
topology, and the independent reader reconstructs full logical KV heads.

## Memory Pressure

Both colocated and P/D cases generate four 208-token paths against a 512-token
pool. Each request fits by itself. Metrics are checked separately for every
tensor/pipeline rank; summing tensor-rank counters would incorrectly count the
same logical request multiple times. Draft context reconstruction must be
observed for each retired request on all four ranks.

Colocated retraction follows its existing recompute path. P/D retraction uses
the request CPU backup/restore path. The P/D observer independently compares
every local target layer K/V before and after restore, then validates the
original capture ID's failed Catalog state. Local retraction and an earlier
peer failure are valid cohort orderings; other failure reasons are rejected.
Retired captures cannot publish, and fresh requests must still be admitted.

Every successful training sample is checked against its online source KV and
raw logits, including top-128 IDs/values, full-vocabulary LSE, token IDs, loss
masks, positions and terminal KV validity. The Store segment outlives both
serving process trees, and all successful samples are reread after they exit.

## Verified Combined Results

All three complete four-H100 files pass against the same production baseline
`c8a9e67e18a5c8c6cbe45e80a5cf208aa88f7305`. This change adds test coverage and
rank-aware pressure assertions; it does not change production serving code.

| Suite | Tests | Seconds | Post-exit snapshots |
| --- | ---: | ---: | ---: |
| Matching TP2/PP2 AR and DSpark P/D | 6 | 642.042 | 34 |
| Matching TP2/PP2 DSpark P/D pressure | 2 | 241.360 | 6 |
| Colocated TP2/PP2 DSpark including pressure | 2 | 271.365 | 20 |
| Combined total | 10 | | 60 |

The ordinary P/D cases exclude 18 missing/stale/aborted captures. The P/D
pressure cases check four retired capture leases and 16 rank-local exact
all-target-layer restores. Both pressure fixtures rebuild each retired draft
context on all four ranks. Their pressure sample counts are included in the
table; seed snapshots used to build synthetic drafts are excluded.

The affected default paths also pass complete regressions: PP1 P/D pressure
(four tests, 337.398 seconds, 12 post-exit snapshots), single-GPU runtime (two
tests, 639.843 seconds), PP2 P/D pressure (two tests, 222.319 seconds, six
snapshots) and colocated PP2 (two tests, 172.748 seconds, 20 snapshots).
The total is 20 tests and 98 strict post-exit snapshots. The monolithic runtime
fixture does not prove the same full-process-tree exit invariant, so its READY
log entries are not included in that snapshot total.

Relevant imported helper/test hashes match the final sources in all frozen
runtime directories. Per-case observations, source/log hashes and resource
cleanup are retained in [the evidence JSON](combined-tp-pp.json). The temporary
four-H100 job was deleted after all tests and its pod is absent; the resident
H100 has no active/queued experiment and has resumed its 60% idle workload.

## Scope

These are synchronous PP, static target-KV drafts and dense unquantized device
KV pools. Prefill CUDA graphs are disabled. Test observers copy and synchronize
tensors, so results are correctness evidence rather than performance numbers.
Synthetic checkpoints and biased pressure replies do not establish trained-draft
quality. Replicated target KV heads, additional model families, distributed
prefill graphs, asynchronous PP, asymmetric combined P/D topologies, cross-node
combined RDMA, production Catalog retention and serving SLOs remain separate
acceptance scopes.
