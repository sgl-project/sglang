//! This replica as a snapshot source: the export, its cache and the cursors-only body.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::Bytes;
use flate2::write::GzEncoder;
use flate2::Compression;
use tracing::warn;

use super::KvEventIndex;
use crate::state::kv_events::bootstrap::{PeerSnapshot, WireWorker, SNAPSHOT_FORMAT};
#[cfg(test)]
use crate::state::kv_events::tree::Tiers;
use crate::state::kv_events::tree::{KvWorkerId, SnapshotNode};

/// A built snapshot with both of its wire encodings.
///
/// The encodings are cached, not the struct: serialising and compressing a
/// fleet-sized tree costs about as much as the walk that produced it, so a
/// boot herd served from this cache pays each once per build, not per request.
pub(super) struct CachedSnapshot {
    /// Taken before the cursors are read, so it never overstates what the
    /// snapshot covers.
    exported_at: Instant,
    body: SnapshotBody,
}

/// One snapshot encoded for the wire: the JSON document and its gzip.
///
/// Both empty means the encode failed; see [`encode_snapshot`].
#[derive(Debug, Clone, Default)]
pub struct SnapshotBody {
    /// The JSON document, served when the caller does not accept gzip.
    pub identity: Bytes,
    /// `identity`, gzip-compressed.
    pub gzip: Bytes,
}

impl SnapshotBody {
    /// Whether this is the encode-failure sentinel.
    pub fn is_empty(&self) -> bool {
        self.identity.is_empty()
    }
}

/// Whether an export sampled at `exported_at` meets a `max_age` a caller
/// stated when it arrived at `requested_at`.
///
/// Measured from arrival, not from whenever the caller gets the cache lock:
/// its requirement is a fixed point in time (when its ranks began holding),
/// so an export sampled after it arrived always meets it, however long a build
/// kept it queued.
fn meets_max_age(exported_at: Instant, requested_at: Instant, max_age: Duration) -> bool {
    exported_at >= requested_at || requested_at.duration_since(exported_at) < max_age
}

/// Serialise a snapshot to JSON.
///
/// `PeerSnapshot` is a plain `Serialize` struct with no non-string map keys,
/// so this cannot actually fail; an empty body on the impossible branch keeps
/// the caller total, and the route turns it into a non-success status rather
/// than a 200 a consumer would fail to decode.
fn encode_json(snap: &PeerSnapshot) -> Bytes {
    match serde_json::to_vec(snap) {
        Ok(v) => Bytes::from(v),
        Err(e) => {
            warn!(error = %e, "kv-bootstrap: snapshot serialisation failed");
            Bytes::new()
        }
    }
}

/// Encode a snapshot as JSON and as its gzip, or the empty sentinel if either
/// step fails.
///
/// Gzip runs at the fastest level: the compressibility is in the per-node JSON
/// scaffolding, so the lowest level already shrinks the body several-fold, and
/// this runs on a replica that is also serving traffic.
fn encode_snapshot(snap: &PeerSnapshot) -> SnapshotBody {
    let identity = encode_json(snap);
    if identity.is_empty() {
        return SnapshotBody::default();
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::fast());
    match encoder.write_all(&identity).and_then(|()| encoder.finish()) {
        Ok(gzip) => SnapshotBody {
            identity,
            gzip: Bytes::from(gzip),
        },
        Err(e) => {
            warn!(error = %e, "kv-bootstrap: snapshot compression failed");
            SnapshotBody::default()
        }
    }
}

impl KvEventIndex {
    /// This replica as a bootstrap source. `None` in metadata-only mode,
    /// where no subscription is opened and the tree is always empty, so there
    /// is nothing to serve. Symmetric with [`Self::metrics_source`].
    pub fn snapshot_source(self: &Arc<Self>) -> Option<Arc<Self>> {
        self.maintain_tree.then(|| Arc::clone(self))
    }

    /// This replica's snapshot, encoded and ready to serve.
    ///
    /// Single-flighted behind an async lock so a boot herd shares one build,
    /// and a cached entry is reused while it meets `max_age` per
    /// [`meets_max_age`]. The build runs detached and owns the lock, so a
    /// caller that gives up mid-build does not cancel it.
    pub async fn peer_snapshot_body(self: &Arc<Self>, max_age: Duration) -> SnapshotBody {
        let requested_at = Instant::now();
        let mut cache = Arc::clone(&self.snapshot_cache).lock_owned().await;
        if let Some(c) = cache.as_ref() {
            if meets_max_age(c.exported_at, requested_at, max_age) {
                return c.body.clone();
            }
        }
        // Release the stale entry before building, so a fleet-sized body is
        // not held alongside its replacement.
        *cache = None;
        let this = Arc::clone(self);
        let build = tokio::spawn(async move {
            let (body, sampled_at) = this.build_peer_snapshot().await;
            if let Some(exported_at) = sampled_at {
                *cache = Some(CachedSnapshot {
                    exported_at,
                    body: body.clone(),
                });
            }
            body
        });
        match build.await {
            Ok(body) => body,
            Err(e) => {
                // The build task panicked or the runtime is shutting down;
                // nothing was cached, so a later request retries.
                warn!(error = %e, "kv-bootstrap: snapshot build task failed");
                SnapshotBody::default()
            }
        }
    }

    /// Walk, filter and encode one peer snapshot for [`Self::peer_snapshot_body`].
    ///
    /// Returns the body and, when it is worth caching, the instant its
    /// contents were sampled. An empty body means the encode failed; it is
    /// never cacheable, so the route's non-success answer is not pinned for a
    /// whole TTL.
    async fn build_peer_snapshot(&self) -> (SnapshotBody, Option<Instant>) {
        // See `CachedSnapshot::exported_at`.
        let exported_at = Instant::now();

        // Read the cursors BEFORE walking the tree, never after.
        //
        // The order is load-bearing: `export_snapshot` releases the tree
        // lock in chunks, so it is not one instant, and the pump mutates the
        // tree before advancing the cursor. Reading cursors last can therefore
        // report a sequence whose effects the walk only partially captured —
        // and the consumer would then filter its own copy of that batch as
        // already-reflected, losing a `BlockRemoved` permanently. Reading them
        // first makes the watermark lag the tree instead: the consumer replays
        // deltas the snapshot already has, and insert/remove are idempotent,
        // so it converges.
        let cursor_by_worker: Vec<(KvWorkerId, i64)> = self
            .cursors
            .lock()
            .iter()
            .map(|(w, seq)| (w.clone(), *seq))
            .collect();
        let tree = Arc::clone(&self.tree);
        let walked = tokio::task::spawn_blocking(move || tree.export_snapshot()).await;
        let (worker_table, nodes) = match walked {
            Ok(v) => v,
            Err(e) => {
                // The walk panicked or the runtime is shutting down. Answer
                // with a not-ready snapshot rather than a partial tree, and do
                // not cache it, so the next request makes a real attempt.
                // Encoding inline is fine: this body is a handful of bytes.
                warn!(error = %e, "kv-bootstrap: snapshot walk failed; reporting not-ready to peers");
                return (encode_snapshot(&self.not_ready_snapshot()), None);
            }
        };
        let index_of: HashMap<&KvWorkerId, u32> = worker_table
            .iter()
            .enumerate()
            .map(|(i, w)| (w, i as u32))
            .collect();
        // Re-check each cursor against the live map before reporting it. A
        // cursor only ever advances, so one that went backwards or vanished
        // during the walk means that rank's publisher reset, or the rank was
        // removed and re-added: the walk may hold blocks from the NEW stream
        // while the pre-walk cursor still names a position in the old one, and
        // a consumer seeded with it would filter the new stream's batches as
        // already applied. Dropping the cursor reads, to the consumer, as "the
        // producer never saw this rank", which runs it cold instead.
        let cursors: Vec<(u32, i64)> = {
            let live = self.cursors.lock();
            cursor_by_worker
                .iter()
                .filter(|(w, seq)| live.get(w).is_some_and(|now| now >= seq))
                .filter_map(|(w, seq)| index_of.get(w).map(|&i| (i, *seq)))
                .collect()
        };
        // Live ranks the walk found no node for — see `PeerSnapshot::empty_ranks`.
        // Read after the walk, so a rank whose first node lands mid-walk can be
        // reported empty. That errs only toward a consumer settling that rank
        // cold, which is safe, never toward it grafting state that is not here.
        let empty_ranks: Vec<WireWorker> = self
            .live_workers
            .lock()
            .iter()
            .filter(|w| !index_of.contains_key(w))
            .map(WireWorker::from)
            .collect();
        let has_nodes = !nodes.is_empty();
        let workers = worker_table.iter().map(WireWorker::from).collect();
        let mut snap = self.wire_snapshot(has_nodes, workers, cursors, nodes);
        snap.empty_ranks = empty_ranks;
        // Encode on the blocking pool for the same reason the walk goes there:
        // serialising and compressing a fleet-sized tree is CPU-bound with no
        // await point, and this runs on the runtime that is also proxying
        // requests.
        match tokio::task::spawn_blocking(move || encode_snapshot(&snap)).await {
            Ok(body) => {
                let cacheable = (!body.is_empty()).then_some(exported_at);
                (body, cacheable)
            }
            Err(e) => {
                // Runtime shutting down or the encode panicked. Answer this
                // caller without caching, so a later request retries.
                warn!(error = %e, "kv-bootstrap: snapshot encode failed");
                (SnapshotBody::default(), None)
            }
        }
    }

    /// This replica's cursor table alone, encoded as JSON, with no tree.
    ///
    /// Reads the cursor map only: no tree walk, no cache. Every observed rank
    /// is reported, including ones that no longer carry nodes; the full export
    /// lists only carriers. `nodes` is always empty and `producer_ready`
    /// reflects the tree.
    pub fn peer_cursors_body(&self) -> Bytes {
        let (workers, cursors) = {
            let guard = self.cursors.lock();
            let mut workers: Vec<WireWorker> = Vec::with_capacity(guard.len());
            let mut cursors: Vec<(u32, i64)> = Vec::with_capacity(guard.len());
            for (i, (w, seq)) in guard.iter().enumerate() {
                workers.push(WireWorker::from(w));
                cursors.push((i as u32, *seq));
            }
            (workers, cursors)
        };
        let snap = self.wire_snapshot(self.tree.node_count() > 0, workers, cursors, Vec::new());
        // Encoded inline, unlike the tree export: this body is one entry per
        // rank, so a blocking-pool hop would cost more than the serialise.
        encode_json(&snap)
    }

    /// A [`PeerSnapshot`] stamped with this replica's hash config.
    ///
    /// `has_nodes` is whether the tree being described holds any nodes; see
    /// [`PeerSnapshot::producer_ready`].
    fn wire_snapshot(
        &self,
        has_nodes: bool,
        workers: Vec<WireWorker>,
        cursors: Vec<(u32, i64)>,
        nodes: Vec<SnapshotNode>,
    ) -> PeerSnapshot {
        let hash_config = self.block_size_oracle.hash_config();
        let (block_size, is_bigram) = hash_config.unwrap_or((0, false));
        PeerSnapshot {
            format: SNAPSHOT_FORMAT,
            block_size,
            is_bigram,
            producer_ready: hash_config.is_some() && self.bootstrap.settled() && has_nodes,
            workers,
            cursors,
            nodes,
            empty_ranks: Vec::new(),
        }
    }

    /// (test-only) Apply one stored block and its cursor directly, bypassing the pump.
    #[cfg(test)]
    pub(crate) fn seed_stored_block_for_test(
        &self,
        worker: &KvWorkerId,
        seq: i64,
        block_hash: i64,
    ) {
        self.tree
            .insert_tiered(worker, None, &[block_hash], Tiers::DEVICE);
        self.cursors.lock().insert(worker.clone(), seq);
    }

    /// (test-only) Record a cursor for a rank that carries no blocks.
    #[cfg(test)]
    pub(crate) fn seed_cursor_only_for_test(&self, worker: &KvWorkerId, seq: i64) {
        self.cursors.lock().insert(worker.clone(), seq);
    }

    /// A snapshot that declares itself not ready, for the paths that must
    /// answer without a tree.
    fn not_ready_snapshot(&self) -> PeerSnapshot {
        self.wire_snapshot(false, Vec::new(), Vec::new(), Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{batch, worker_id};
    use super::*;
    use crate::state::kv_events::block_size_oracle::BlockSizeOracle;
    use crate::state::kv_events::wire::KvCacheEvent;

    /// A request's `max_age` is anchored at its arrival. An export sampled
    /// after the request arrived meets any `max_age`, including zero — that is
    /// what lets a waiter share the build it queued behind.
    #[test]
    fn max_age_is_measured_from_the_request_arrival() {
        let requested_at = Instant::now();
        let later = requested_at + Duration::from_secs(5);
        assert!(meets_max_age(later, requested_at, Duration::ZERO));
        assert!(meets_max_age(requested_at, requested_at, Duration::ZERO));

        let earlier = requested_at - Duration::from_millis(500);
        assert!(meets_max_age(earlier, requested_at, Duration::from_secs(1)));
        assert!(!meets_max_age(
            earlier,
            requested_at,
            Duration::from_millis(500)
        ));
        assert!(!meets_max_age(earlier, requested_at, Duration::ZERO));
    }

    /// A caller that gives up mid-build must not discard the build: the task
    /// finishes, fills the cache, and the next request is served from it.
    #[tokio::test]
    async fn an_abandoned_snapshot_request_still_fills_the_cache() {
        use futures::FutureExt;

        let index = KvEventIndex::new();
        // One poll takes the uncontended lock and hands the build to its task;
        // on the single-threaded test runtime that task cannot have run yet,
        // so the request is dropped with the build still pending.
        let abandoned = index.peer_snapshot_body(Duration::ZERO).now_or_never();
        assert!(
            abandoned.is_none(),
            "the request must be dropped mid-build, or this proves nothing",
        );
        let cache = index.snapshot_cache.lock().await;
        assert!(
            cache.as_ref().is_some_and(|c| !c.body.is_empty()),
            "the detached build must still cache its body",
        );
        drop(cache);
        index.shutdown().await;
    }

    /// The producer reuses its cached export only for a caller whose freshness
    /// requirement it meets; an older export would predate the caller's
    /// subscription and gap on the watermark.
    #[tokio::test]
    async fn producer_reuses_its_export_only_when_it_meets_the_callers_max_age() {
        let index = KvEventIndex::new();
        let exported_at = || async {
            index
                .snapshot_cache
                .lock()
                .await
                .as_ref()
                .expect("an entry was cached")
                .exported_at
        };

        index.peer_snapshot_body(Duration::from_secs(60)).await;
        let first = exported_at().await;

        // A caller that can live with a minute-old tree gets the same one back.
        index.peer_snapshot_body(Duration::from_secs(60)).await;
        assert_eq!(first, exported_at().await, "a met requirement reuses");

        // A caller that began holding after that export cannot use it.
        tokio::time::sleep(Duration::from_millis(5)).await;
        index.peer_snapshot_body(Duration::ZERO).await;
        assert!(
            exported_at().await > first,
            "an unmet requirement must rebuild, not serve the stale export",
        );
    }

    /// The stamp has to be taken BEFORE the contents are sampled. Stamping
    /// completion would claim coverage the document does not have — exactly the
    /// hole the parameter closes — and the walk plus encode of a real tree is
    /// long enough for that to matter.
    #[tokio::test]
    async fn producer_stamps_the_export_before_it_samples() {
        let index = KvEventIndex::new();
        let before = Instant::now();
        index.peer_snapshot_body(Duration::ZERO).await;
        let after = Instant::now();
        let exported_at = index
            .snapshot_cache
            .lock()
            .await
            .as_ref()
            .expect("an entry was cached")
            .exported_at;
        assert!(
            exported_at >= before && exported_at <= after,
            "the stamp must sit inside the build, at its start",
        );
    }

    /// The evidence a consumer's per-rank settle stands on: every LIVE rank the
    /// export holds no node for is named — one never applied (what a rank the
    /// producer is still bootstrapping looks like) and one whose applied stream
    /// left nothing standing — and a rank that does carry nodes is not.
    #[tokio::test]
    async fn snapshot_names_the_live_ranks_it_holds_nothing_for() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(64).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let (warm, pending, emptied) = (
            worker_id("http://w1:30000", 0),
            worker_id("http://w2:30000", 0),
            worker_id("http://w3:30000", 0),
        );
        for w in [&warm, &pending, &emptied] {
            index.live_workers.lock().insert(w.clone());
        }
        index.seed_stored_block_for_test(&warm, 3, 111);
        index.seed_cursor_only_for_test(&emptied, 2);

        let body = index.peer_snapshot_body(Duration::ZERO).await;
        let snap: PeerSnapshot = serde_json::from_slice(&body.identity).expect("valid JSON");
        assert!(
            snap.wire_cursor_for(&warm.url, 0).is_some(),
            "the warm rank is covered"
        );
        assert!(!snap.holds_nothing_for(&warm.url, 0));
        assert!(snap.holds_nothing_for(&pending.url, 0));
        assert!(snap.holds_nothing_for(&emptied.url, 0));
        assert_eq!(snap.empty_ranks.len(), 2);
    }

    /// This body answers one integer per rank, so it must carry the cursors
    /// and NOT the tree. Asserting `nodes` is empty is also
    /// what keeps the body ungraftable — `from_wire` rejects an empty node list.
    #[tokio::test]
    async fn cursors_only_body_carries_cursors_and_no_nodes() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        index.seed_stored_block_for_test(&worker_id("http://w1:30000", 0), 42, 111);

        let body = index.peer_cursors_body();
        let snap: PeerSnapshot = serde_json::from_slice(&body).expect("valid JSON");

        assert!(snap.nodes.is_empty(), "the tree must not be exported");
        assert_eq!(
            snap.wire_cursor_for("http://w1:30000", 0),
            Some(42),
            "the per-rank cursor must be answerable from this body",
        );
        assert_eq!(snap.block_size, 256);
        assert!(!snap.is_bigram);
        assert!(
            snap.producer_ready,
            "a settled replica holding nodes is a valid witness",
        );
    }

    /// A replica with an empty tree must not claim to be worth believing, for the
    /// same reason `snapshot_entry` checks it: a replica whose own bootstrap timed
    /// out is settled while holding nothing.
    #[tokio::test]
    async fn cursors_only_body_reports_not_ready_with_an_empty_tree() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));

        let snap: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert!(!snap.producer_ready, "an empty tree is not a source");
    }

    /// The cost claim, asserted structurally rather than by timing: serving cursors
    /// must leave the snapshot cache untouched, because populating it is the
    /// expensive walk this path exists to avoid.
    #[tokio::test]
    async fn cursors_only_body_never_populates_the_snapshot_cache() {
        let index = KvEventIndex::new();
        index.peer_cursors_body();
        assert!(
            index.snapshot_cache.lock().await.is_none(),
            "cursors-only must not walk or cache the tree",
        );
    }

    /// The two producer answers are intentionally NOT set-equal. A rank that
    /// observed a publisher and then lost the blocks (cleared here) keeps its
    /// cursor in the cursors-only body — that observation is still a valid
    /// witness — while the full export drops it, because its cursor table only
    /// covers ranks carrying tree nodes. The full side uses a SECOND rank that
    /// still carries a block, so the export is a real one (not the not-ready
    /// body an empty tree would fetch) and the assertion genuinely pins the
    /// carrier filtering. See [`KvEventIndex::peer_cursors_body`].
    #[tokio::test]
    async fn the_cursors_only_table_keeps_witnesses_the_full_export_loses() {
        let oracle = BlockSizeOracle::new();
        oracle.try_set(256).expect("first set establishes");
        oracle.set_bigram(false);
        let index =
            KvEventIndex::new_with_http_and_oracle(reqwest::Client::new(), Arc::clone(&oracle));
        let cleared = worker_id("http://w1:30000", 0);
        let holding = worker_id("http://w2:30000", 0);
        index.seed_stored_block_for_test(&holding, 7, 999);
        index.seed_stored_block_for_test(&cleared, 42, 111);
        // Observed-then-cleared: the cursor survives where the blocks do not.
        super::super::apply_batch(
            &index.tree,
            &index.cursors,
            &index.tally,
            &cleared,
            43,
            &batch(vec![KvCacheEvent::AllBlocksCleared]),
        );

        let thin: PeerSnapshot =
            serde_json::from_slice(&index.peer_cursors_body()).expect("valid JSON");
        assert_eq!(
            thin.wire_cursor_for("http://w1:30000", 0),
            Some(43),
            "a cleared rank is still a witness to its publisher's stream",
        );

        let full: PeerSnapshot =
            serde_json::from_slice(&index.peer_snapshot_body(Duration::ZERO).await.identity)
                .expect("valid JSON");
        assert!(
            !full.nodes.is_empty(),
            "the held block keeps this a real export, not the not-ready body",
        );
        assert_eq!(
            full.wire_cursor_for("http://w1:30000", 0),
            None,
            "carriers only: the cleared rank's cursor is dropped from the export",
        );
        assert_eq!(
            full.wire_cursor_for("http://w2:30000", 0),
            Some(7),
            "the rank still carrying blocks keeps its cursor in the export",
        );
    }
}
