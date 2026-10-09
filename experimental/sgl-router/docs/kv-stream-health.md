# KV event stream health

PublisherReset clears the emitting worker URL/DP rank from every cache tier,
discards its held pre-reset batches and splice proof, and sets its cursor to -1.
Other ranks are preserved. New publisher batches must start from zero; a
first batch above zero reports a gap.
This prevents pre-restart owners from surviving an explicit reset sentinel.

A forward gap after an applied cursor emits a WARN with worker identity,
previous/current sequence, and the number of missing sequence values.
WARNs are sampled per worker (first occurrence, then every 64); counters
include every observation. The batch still applies. The router does not infer
which transport lost events,
clear the tree, or replay missing batches automatically.

The following metrics have worker_url and dp_rank labels:

- sgl_router_kv_event_decode_failures_total{reason}: rejected KV messages; reasons
  are msgpack, block_hashes_limit, frame_count, and sequence_frame. Load-topic
  errors are excluded. A failure also marks the local stream untrusted.
- sgl_router_kv_event_sequence_gaps_total: observed forward discontinuities.
- sgl_router_kv_event_missing_sequences_total: sequence values inside those gaps.
- sgl_router_kv_event_skipped_batches_total: duplicate/older batches discarded
  by the existing cursor filter. A rollback without a reset sentinel is also
  counted here and sets untrusted=1 with a sampled WARN when seq < cursor.
  Equality alone does not mark the stream untrusted. No implicit reset is
  performed: fresh batches remain filtered until they exceed the old cursor
  or an explicit reset is received.
- sgl_router_kv_event_publisher_resets_total: reset sentinels handled.
- sgl_router_kv_event_stream_untrusted: 1 after a gap, rollback, or decode failure; returns to 0 only after
  an explicit publisher reset or AllBlocksCleared for that rank. Normal
  subsequent traffic does not repair the missing history.

Series appear after a decode failure, gap, skipped batch, or reset, and disappear when the
worker rank is removed. Counters persist across publisher resets. These are
local observations, not peer-synchronized state. The gauge is diagnostic:
it does not gate routing or snapshot export. Zero/absent does not prove cache
coverage; a cold subscriber may have missed all earlier history. A snapshot
bootstrap has its own continuity checks and counters. Batches covered by the
snapshot watermark are expected overlap and excluded from skipped/rollback
statistics. Deferred splice validation handles its first forward gap before
normal accounting, discarding the snapshot via the existing bootstrap path;
subsequent genuine live gaps remain observable. A successful proof does not
clear unrelated prior stream anomalies.

Adaptive Metrics must retain worker_url, dp_rank, and reason for these metrics (and
router pod identity to distinguish local observations). Confirm the rules
before deployment; aggregation would hide which stream is affected. Zero gap
counts alone cannot exclude missing history before subscription or failures
that do not produce a visible forward discontinuity.

Do not use raw store-minus-remove event counts as node growth: duplicate
stores and unmatched removals are counted too. Tree occupancy counts carrier
nodes, which can include the same hash at different tree positions. Compare
unique hashes, carrier nodes, and actual event effects from the same capture
before attributing growth to the engine.

Fixing reset cleanup does not establish the cause of an incident that had
no reset. Enabling an engine replay endpoint alone does not make this router
consume it. Recovery across a forward gap requires a separately validated
replay/resynchronization protocol.


## Informational token fields

The decoder consumes `BlockStored.token_ids` without retaining its contents.
Both array and tagged-map events support unigram and bigram arrays of any
length that fit in the received payload; the public token vector stays empty.
Malformed/truncated MessagePack still fails. The 65,536 block-hash cap remains
in force for both stores and removals. Routing continues to use block hashes.

An oversized informational token field previously rejected an entire batch,
including sibling removal events. This fix prevents that loss but cannot
recover deletions already missed. To rebuild without importing stale peer
snapshots, temporarily disable peer bootstrap (`--kv-peer-selector`) while
replacing all router replicas; restore it only once all replicas contain
rebuilt state. Cold-start recovery duration depends on actual cache turnover;
validate occupancy and engine hit rates rather than assuming a fixed deadline.
No production rollout or cache flush is performed by this code change.

`examples/kv_event_replay.rs` accepts a token-stripped JSONL capture with
`seq` and `batch` fields. It synthesizes bigram token arrays from each store's
block count and block size, then decodes and applies the batches to a fresh
HashTree and a unique-hash shadow. Captures and production identities are not
checked into the repository.
