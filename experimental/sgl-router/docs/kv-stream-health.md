# KV event stream health

PublisherReset clears the emitting worker URL/DP rank from every cache tier,
discards its held pre-reset batches and splice proof, and removes its cursor.
Other ranks are preserved. New publisher batches can then start from zero.
This prevents pre-restart owners from surviving an explicit reset sentinel.

A forward gap after an applied cursor emits a WARN with worker identity,
previous/current sequence, and the number of missing sequence values. The
batch still applies. The router does not infer which transport lost events,
clear the tree, or replay missing batches automatically.

The following metrics have worker_url and dp_rank labels:

- sgl_router_kv_event_sequence_gaps_total: observed forward discontinuities.
- sgl_router_kv_event_missing_sequences_total: sequence values inside those gaps.
- sgl_router_kv_event_skipped_batches_total: duplicate/older batches discarded
  by the existing cursor filter. A rollback without a reset sentinel is also
  counted here; the router cannot distinguish restart from delayed delivery.
- sgl_router_kv_event_publisher_resets_total: reset sentinels handled.
- sgl_router_kv_event_stream_untrusted: 1 after a gap; returns to 0 only after
  an explicit publisher reset or AllBlocksCleared for that rank. Normal
  subsequent traffic does not repair the missing history.

Series appear after a gap, skipped batch, or reset, and disappear when the
worker rank is removed. Counters persist across publisher resets. These are
local observations, not peer-synchronized state. The gauge is diagnostic:
it does not gate routing or snapshot export. Zero/absent does not prove cache
coverage; a cold subscriber may have missed all earlier history. A snapshot
bootstrap has its own continuity checks and counters.

Do not use raw store-minus-remove event counts as node growth: duplicate
stores and unmatched removals are counted too. Tree occupancy counts carrier
nodes, which can include the same hash at different tree positions. Compare
unique hashes, carrier nodes, and actual event effects from the same capture
before attributing growth to the engine.

Fixing reset cleanup does not establish the cause of an incident that had
no reset. Enabling an engine replay endpoint alone does not make this router
consume it. Recovery across a forward gap requires a separately validated
replay/resynchronization protocol.
