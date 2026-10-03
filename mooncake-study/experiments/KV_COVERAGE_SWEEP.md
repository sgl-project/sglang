# KV Rectangle Coverage Sweep

The manifest validator must prove exact coverage of every selected layer's K/V
rectangle: all heads for every computed token, allowing only the final token
to lack KV. Token chunks and TP head partitions can have different boundaries.

Previously, each token slab scanned every descriptor again before sorting its
head ranges. With a fixed number of head shards, the repeated Python scans grew
quadratically with the number of token chunks. A 32K-prompt metadata fixture
with 2,064 objects took 45.627 ms per full manifest validation; the repeated
coverage scan dominated its CPU profile.

The validator now sorts start/end events by token position and maintains only
the active, sorted head intervals. At a shared endpoint, removals precede
insertions. Each insertion rejects an overlap with either neighbor; because
the caller already validates head bounds, disjoint intervals cover the slab
exactly when their widths sum to the required head count. Initial/internal
token gaps and incomplete head coverage still fail. Wire formats, hashes,
semantic payload checks and publication boundaries are unchanged.

For `m` rectangles and at most `a` active head intervals, sorting costs
`O(m log m)` and the sorted Python list's insertions/removals can cost `O(m a)`.
Memory is `O(m + a)`. This removes repeated full-history scans for the usual
fixed TP partition count; it is not a claim of logarithmic list updates or a
better worst case for arbitrarily growing head partition counts.

## Correctness Checks

The new CPU suite uses an independent per-cell grid oracle:

- All multisets of up to four rectangles in a two-token/two-head grid, including
  duplicates, are checked with three sequence endpoints: 2,145 comparisons.
- Five hundred recursively split random tilings check both full/final-missing
  coverage, removed rectangles, duplicate rectangles and unrelated random
  rectangles: 3,500 comparisons.
- All 120 permutations of a changing head partition exercise simultaneous
  start/end events. Equal-area overlaps and internal token holes must fail.
- A shuffled complete manifest passes normal validation and JSON roundtrip;
  removing a KV chunk while updating total bytes still fails both boundaries.

Shared protocol, topology, partition context, writer and coordinator regressions
exercise malformed metadata/payloads, partial publication, recovery, cancellation
and speculative accepted-token ownership. Actual TCP Store tests cover independent
readers and distributed owner receipts.

## Reproduction

Use the retained H100 worker and pinned capture environment with frozen source,
`PYTHONPATH` pointing to that source's `python` directory, and `OMP_NUM_THREADS=1`.
Run the same driver for baseline and candidate; their production Python sources
differ only in `training_capture/protocol.py`.

```bash
python test/registered/unit/training_capture/test_coverage.py -v
python test/registered/unit/training_capture/test_protocol.py -v
python test/registered/storage/test_training_snapshot_mooncake.py -v -f

python mooncake-study/experiments/benchmark_training_capture.py \
  --model-path /gpfs/models/huggingface.co/Qwen/Qwen3-0___6B \
  --source-revision SOURCE_REVISION --output-dir NEW_OUTPUT_DIRECTORY \
  --num-prompts 8 --input-len 32768 --output-len 32 --concurrency 1 \
  --capture-slots 2 --host-mib 1024 --manifest-mib 4 --segment-mib 8192 \
  --ratios 1.0 --repeats 1
```

`--manifest-mib` explicitly sizes the reserved per-slot manifest buffer within
the total Host budget; it defaults to the existing 1 MiB. Long sequences with
many selected-layer chunks need a larger metadata arena. Both corrected serving
runs use 4 MiB. The initial two runs were intentionally interrupted before
acceptance after noticing their insufficient 1 MiB configuration; their logs
and terminal worker results are retained, not treated as performance evidence.

The microprobe validates metadata only, with two selected layers and 64-token
chunks. The real serving pair uses three selected layers, raw teacher capture,
overlap and decode graphs, with an off/on/off bracket for each source. Every
measured READY sample must be read and semantically validated after producer exit.
This tests local TCP plus the HTTP test Catalog. A single eight-request phase
does not certify production performance, retention or training quality.

## Verified Results

All 123 test methods pass: 116 protocol/topology/context/writer/coordinator
methods in 56.559 seconds and seven actual TCP Store methods in 194.320 seconds.
The latter includes independent readers, cohort receipts and recovery paths.

Thirty unprofiled repetitions per fixture, after three warmup calls, give these
median times for complete metadata validation:

| Prompt Tokens | Objects | Before ms | After ms | Reduction |
| --- | ---: | ---: | ---: | ---: |
| 512 | 48 | 0.346 | 0.338 | 2.4% |
| 4,096 | 272 | 2.265 | 1.775 | 21.7% |
| 16,384 | 1,040 | 14.746 | 6.986 | 52.6% |
| 32,768 | 2,064 | 45.627 | 13.984 | 69.4% |

Both corrected off/on/off brackets finish 24 measured requests. The two enabled
phases each admit and publish all eight requests, with zero stage errors,
quarantine or disabled capture. Post-exit validation covers **16 snapshots,
49,376 tensor objects and 6,456,617,984 payload bytes** across both sources.
Each manifest is approximately 1,451,161 bytes, exceeding the initial 1 MiB
reservation. Both full comparisons use a 4 MiB reservation and the same
883,099,136-byte total Host arena, within the 1 GiB configured budget.

| Real 32K Capture Metric | Before | After |
| --- | ---: | ---: |
| Snapshot construction ms/sample | 539.833 | 455.239 |
| Complete content validation ms/sample | 615.298 | 559.288 |
| Requests/s | 0.10545 | 0.10641 |
| Throughput / own off-bracket mean | 95.02% | 95.58% |

Content validation includes hashing and semantic checks, so its 9.1% reduction
is smaller than the metadata-only microprobe. Construction falls 15.7%. The
approximately 0.9% request-throughput difference in this one small pair does
not establish an end-to-end speedup. Full-vocabulary teacher capture and GPU
export work are unchanged. This workload is not comparable to the earlier
512-prompt/128-response sustained experiment as a performance regression test.

[The evidence JSON](kv-coverage-sweep.json) records all eight terminal jobs,
including the two intentionally interrupted initial runs. All 5,047 Python
files match the final frozen source. Baseline and candidate serving trees differ
only in the production protocol file. The unit-tested production and test
sources are identical; the final driver only adds the explicit metadata budget.
The new test and benchmark driver pass full Ruff. The protocol retains its
baseline `UP035`; removal of the old adjacent-pair scan also removes `RUF007`.
The first archive audit incorrectly required identical lint findings; the
corrected check permits removed findings and rejects new ones, with both audit
scripts retained. No tested source or assertion changed during that correction.

The archive contains 76 artifacts at
`/gpfs/user/fuxuanwei/mooncake-lab-archive/coverage-sweep-20261003`. Its manifest
SHA-256 is `1e41d1bec94eecd34329506045433717bf00fe3542d16421464edae338c54f5c`.
Live-process inspection confirms the model/Store/test processes have exited and
the resident idle load has resumed. No additional GPU was allocated.
