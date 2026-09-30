// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end peer bootstrap: a booting replica must end up with the *same*
//! cache-aware view as the warm replica it copied from.
//!
//! These tests run the real transport — an axum server serving
//! `/internal/kv_snapshot`, a reqwest client fetching it, JSON over a loopback
//! socket — because the interesting failure modes live in the wire format and
//! the vetting step, not in the tree algebra (that is covered by the unit tests
//! on `HashTree::export_snapshot` / `restore_snapshot`). Where a test needs a
//! real tree, the body served is the production producer's own export
//! (`KvEventIndex::peer_snapshot_body`), so a change to how the producer builds
//! it cannot slip past a test-local copy.
//!
//! The equivalence assertion is deliberately behavioural: for a large query
//! set, `prefix_depths` (what the local routing policies score on) and
//! `match_prefix` (the deepest matched node's carriers and their tiers) must
//! agree on both replicas. Comparing node counts alone would pass while routing
//! diverged.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::Query;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::{routing::get, Router};
use sgl_router::state::kv_events::bootstrap::{
    fetch_cursors, fetch_snapshot, FetchAnswer, PeerSnapshot, VetError, VettedSnapshot,
    CURSORS_ONLY_PARAM, SNAPSHOT_FORMAT, SNAPSHOT_PATH,
};
use sgl_router::state::kv_events::index::SnapshotBody;
use sgl_router::state::kv_events::{
    HashTree, KvEventIndex, KvWorkerId, SnapshotNode, Tiers, WireWorker,
};
use tokio::net::TcpListener;

/// Page size both replicas establish; the vetting step compares against it.
const BLOCK_SIZE: u32 = 64;

/// Fetch a snapshot that must exist: the body, or a panic naming the status.
async fn fetch_tree(base_url: &str) -> PeerSnapshot {
    match fetch_snapshot(&reqwest::Client::new(), base_url, None)
        .await
        .expect("fetch succeeds")
    {
        FetchAnswer::Body(snap) => snap,
        FetchAnswer::NoBody(status) => panic!("peer serves a snapshot, got HTTP {status}"),
    }
}

/// Probe `base_url` for cursors the way the splice probe does; the peer must
/// answer.
async fn fetch_witness(base_url: &str) -> PeerSnapshot {
    fetch_cursors(&reqwest::Client::new(), base_url)
        .await
        .expect("transport ok")
        .expect("peer answered")
}

/// Serve `app` on an ephemeral loopback port and return its base URL.
async fn spawn_app(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (format!("http://{addr}"), handle)
}

/// Serve `full` at the snapshot path, or `cursors_only` to a request that asks
/// for cursors when that shape is given — so one server can stand in for a
/// peer that honours the parameter (`Some`) or one that ignores it and answers
/// in full regardless (`None`). Returns the base URL.
///
/// Bodies are encoded once and served as-is, the way the producer serves its
/// pre-encoded export.
async fn serve_snapshot(
    full: PeerSnapshot,
    cursors_only: Option<PeerSnapshot>,
) -> (String, tokio::task::JoinHandle<()>) {
    let encode = |s: &PeerSnapshot| Bytes::from(serde_json::to_vec(s).expect("snapshot encodes"));
    let full = encode(&full);
    let thin = cursors_only.as_ref().map(encode);
    spawn_app(Router::new().route(
        SNAPSHOT_PATH,
        get(
            move |Query(params): Query<HashMap<String, String>>| async move {
                let asks_thin = params.get(CURSORS_ONLY_PARAM).is_some_and(|v| v == "true");
                let body = match thin {
                    Some(t) if asks_thin => t,
                    _ => full,
                };
                ([(header::CONTENT_TYPE, "application/json")], body)
            },
        ),
    ))
    .await
}

/// Serve a producer's pre-built gzip under `Content-Encoding: gzip`, and only
/// to a request whose `Accept-Encoding` is exactly `gzip`; anything else gets
/// `406`. An identity fallback would let a consumer that never asked for gzip
/// pass, and a wider header would mean the snapshot client advertises other
/// codings. The route's own negotiation is asserted against `build_router` in
/// `server::routes::cache`'s tests.
async fn serve_gzip_only(body: SnapshotBody) -> (String, tokio::task::JoinHandle<()>) {
    spawn_app(Router::new().route(
        SNAPSHOT_PATH,
        get(move |headers: HeaderMap| {
            let gzip = body.gzip.clone();
            async move {
                let asks_gzip = headers
                    .get(header::ACCEPT_ENCODING)
                    .is_some_and(|v| v.as_bytes() == b"gzip");
                if !asks_gzip {
                    return StatusCode::NOT_ACCEPTABLE.into_response();
                }
                (
                    [
                        (header::CONTENT_TYPE, "application/json"),
                        (header::CONTENT_ENCODING, "gzip"),
                    ],
                    gzip,
                )
                    .into_response()
            }
        }),
    ))
    .await
}

fn worker(url: &str, dp_rank: u32) -> KvWorkerId {
    KvWorkerId::new(url.to_string(), dp_rank)
}

/// Two ranks of one engine plus a second engine, so a carrier that lost its
/// `dp_rank` on the wire would collide with a sibling and be caught.
fn three_workers() -> Vec<KvWorkerId> {
    vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ]
}

/// A tree shaped like a real one: shared prefixes across workers, divergent
/// tails, single-block chains, a hash occupying two positions, many sibling
/// roots, and carriers on mixed storage tiers — a host-only
/// holder and a device+host holder — so a snapshot that dropped or
/// mis-paired the tier table changes the tiers `match_prefix` reports and is
/// caught.
fn fill_warm(tree: &HashTree, workers: &[KvWorkerId]) {
    let (a, b, c) = (&workers[0], &workers[1], &workers[2]);
    tree.insert(a, None, &[1, 2, 3, 4]);
    tree.insert_tiered(a, None, &[1, 2], Tiers::HOST);
    tree.insert_tiered(b, None, &[1, 2, 3, 4], Tiers::HOST);
    tree.insert(b, None, &[1, 2, 5, 6]);
    tree.insert(c, None, &[7]);
    tree.insert(a, None, &[2, 3, 9]);
    for r in 0..64i64 {
        tree.insert(b, None, &[r * 4096 + 11, r * 4096 + 12]);
        tree.insert(c, None, &[r * 4096 + 11, r * 4096 + 99]);
    }
}

/// A warm replica: a real `KvEventIndex` with an established hash config and
/// its tree filled by `fill`.
///
/// The tree is written directly rather than through the pump, which only a
/// live ZMQ subscription feeds, so this index's cursor map stays empty; see
/// [`with_cursors`].
fn warm_index(fill: impl FnOnce(&HashTree)) -> Arc<KvEventIndex> {
    let index = KvEventIndex::new();
    index.block_size_oracle().try_set(BLOCK_SIZE).unwrap();
    index.block_size_oracle().set_bigram(false);
    fill(index.tree().as_ref());
    index
}

/// What `index` serves a bootstrapping sibling: its production export,
/// decoded. `Duration::ZERO` so an export cached by an earlier call is never
/// reused.
async fn export_of(index: &Arc<KvEventIndex>) -> PeerSnapshot {
    decode_identity(&index.peer_snapshot_body(Duration::ZERO).await)
}

fn decode_identity(body: &SnapshotBody) -> PeerSnapshot {
    serde_json::from_slice(&body.identity).expect("the producer's export decodes")
}

/// `snap` with `cursors` in its cursor table — the one field a test outside
/// `kv_events` cannot make the producer emit, since only the pump writes the
/// cursor map. Filtered to the body's worker table, as the producer's export
/// filters it.
fn with_cursors(mut snap: PeerSnapshot, cursors: &[(KvWorkerId, i64)]) -> PeerSnapshot {
    snap.cursors = cursors
        .iter()
        .filter_map(|(w, seq)| {
            let wire = WireWorker::from(w);
            snap.workers
                .iter()
                .position(|t| *t == wire)
                .map(|i| (i as u32, *seq))
        })
        .collect();
    snap
}

/// `workers` as the live set vetting checks a snapshot against.
fn live_set(workers: &[KvWorkerId]) -> HashSet<KvWorkerId> {
    workers.iter().cloned().collect()
}

/// A distinct cursor for every worker.
fn cursors_for(workers: &[KvWorkerId]) -> Vec<(KvWorkerId, i64)> {
    workers.iter().cloned().zip(41i64..).collect()
}

/// Bootstrap a fresh tree from a fetched body the way the consumer does: vet
/// against the live set, narrow to the ranks being bootstrapped (the pump's
/// `retain_workers`), and graft.
///
/// Every rank must carry a cursor. The sweep skips a snapshot that covers no
/// pending rank, and the pump resolves a cursor-less rank `Uncovered` and
/// clears what the graft gave it — so a view proven equal without cursors is
/// one production would never keep.
fn bootstrap_from(fetched: PeerSnapshot, pending: &[KvWorkerId]) -> HashTree {
    let live = live_set(pending);
    let mut vetted =
        VettedSnapshot::from_wire(fetched, &live, Some(BLOCK_SIZE)).expect("vets clean");
    for rank in pending {
        assert!(
            vetted.cursor_for(rank).is_some(),
            "{rank:?} has no cursor, so the pump would run it cold",
        );
    }
    vetted.retain_workers(&live);
    let tree = HashTree::new();
    vetted.graft_into(&tree).expect("restore succeeds");
    tree
}

fn probe_queries() -> Vec<Vec<i64>> {
    let mut q = vec![
        vec![1],
        vec![1, 2],
        vec![1, 2, 3],
        vec![1, 2, 3, 4],
        vec![1, 2, 5],
        vec![1, 2, 5, 6],
        vec![1, 2, 3, 4, 12345],
        vec![7],
        vec![2],
        vec![2, 3],
        vec![2, 3, 9],
        vec![404],
        vec![],
    ];
    for r in 0..64i64 {
        q.push(vec![r * 4096 + 11]);
        q.push(vec![r * 4096 + 11, r * 4096 + 12]);
        q.push(vec![r * 4096 + 11, r * 4096 + 99]);
    }
    q
}

/// Assert two trees are indistinguishable through the routing interface.
fn assert_same_view(want: &HashTree, got: &HashTree, ctx: &str) {
    assert_eq!(
        got.node_count(),
        want.node_count(),
        "{ctx}: node count diverged",
    );
    for q in probe_queries() {
        assert_eq!(
            got.prefix_depths(None, &q),
            want.prefix_depths(None, &q),
            "{ctx}: prefix_depths diverged for {q:?}",
        );
        let (w, g) = (want.match_prefix(None, &q), got.match_prefix(None, &q));
        assert_eq!(
            (g.matched_blocks, &g.tiers),
            (w.matched_blocks, &w.tiers),
            "{ctx}: match_prefix diverged for {q:?}",
        );
    }
}

/// The snapshot body is compressed in production, so the consumer must end up
/// with the same routing view over a gzipped transport as over an identity one.
/// [`serve_snapshot`] serves identity, which keeps a peer that does not
/// compress covered.
///
/// The gzip served is the producer's own, so the cursor table — which only the
/// pump can make the producer emit — is added after the decode.
#[tokio::test]
async fn new_replica_view_matches_warm_replica_over_gzipped_http() {
    let workers = three_workers();
    let warm = warm_index(|t| fill_warm(t, &workers));
    let body = warm.peer_snapshot_body(Duration::ZERO).await;
    let identity = decode_identity(&body);
    let (base_url, _server) = serve_gzip_only(body).await;

    let fetched = fetch_tree(&base_url).await;
    assert_eq!(
        fetched, identity,
        "the gzipped body must decode to the producer's identity export",
    );
    let new_tree = bootstrap_from(with_cursors(fetched, &cursors_for(&workers)), &workers);
    assert_same_view(&warm.tree(), &new_tree, "gzipped bootstrap");
}

/// Guard on the blast radius of route compression.
///
/// Each of reqwest's crate-wide decompression features (`gzip`, `brotli`,
/// `deflate`, `zstd`) makes `Accepts::default()` advertise its coding, so every
/// client in the process — including the proxy client carrying SSE — would
/// start advertising it and auto-decoding responses. The snapshot fetch
/// therefore asks for gzip on its own request instead. None of the features is
/// enabled, so a default client sends no `Accept-Encoding` at all; adding any
/// of them to Cargo.toml makes this fail.
#[tokio::test]
async fn a_default_client_does_not_advertise_gzip() {
    let (base_url, _server) = spawn_app(Router::new().route(
        "/echo-accept-encoding",
        get(|headers: HeaderMap| async move {
            headers
                .get(header::ACCEPT_ENCODING)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("<absent>")
                .to_string()
        }),
    ))
    .await;

    let seen = reqwest::Client::new()
        .get(format!("{base_url}/echo-accept-encoding"))
        .send()
        .await
        .expect("request succeeds")
        .text()
        .await
        .expect("body reads");
    assert_eq!(
        seen, "<absent>",
        "a default reqwest client must not advertise any content coding — each of \
         reqwest's crate-wide decompression features would flip every client in the \
         process, including the SSE proxy path",
    );
}

/// The headline guarantee: after bootstrapping over real HTTP, the new
/// replica's routing view is identical to the old replica's.
#[tokio::test]
async fn new_replica_view_matches_warm_replica_over_http() {
    let workers = three_workers();
    let warm = warm_index(|t| fill_warm(t, &workers));
    let snap = with_cursors(export_of(&warm).await, &cursors_for(&workers));
    let (base_url, _server) = serve_snapshot(snap, None).await;

    let new_tree = bootstrap_from(fetch_tree(&base_url).await, &workers);

    assert_same_view(&warm.tree(), &new_tree, "after peer bootstrap");
}

/// Cursors must survive the round trip, since they are what lets the new
/// replica filter the deltas the snapshot already reflects.
#[tokio::test]
async fn cursors_survive_the_round_trip() {
    let workers = three_workers();
    let warm = warm_index(|t| fill_warm(t, &workers));
    let cursors = cursors_for(&workers);
    let (base_url, _server) =
        serve_snapshot(with_cursors(export_of(&warm).await, &cursors), None).await;

    let fetched = fetch_tree(&base_url).await;
    let vetted = VettedSnapshot::from_wire(fetched, &live_set(&workers), Some(BLOCK_SIZE)).unwrap();

    for (w, seq) in &cursors {
        assert_eq!(
            vetted.cursor_for(w),
            Some(*seq),
            "cursor for {w:?} must survive",
        );
    }
}

/// The vetting bridge is structural, not conventional: from OUTSIDE the
/// kv_events module — which is what this integration-test crate is — a wire
/// snapshot can reach the tree only by going through `from_wire` and then
/// `graft_into`.
///
/// This test is deliberately about the API surface rather than a runtime
/// behaviour: it is the compile-time property that keeps a future caller from
/// assembling nodes itself and skipping the format / block-size / parent-bounds
/// checks. If `restore_snapshot` or the `VettedSnapshot` fields are ever made
/// public, the equivalent bypass compiles and this comment is the record of why
/// it should not.
#[tokio::test]
async fn grafting_requires_going_through_vetting() {
    let workers = three_workers();
    let warm = warm_index(|t| fill_warm(t, &workers));
    let (base_url, _server) = serve_snapshot(export_of(&warm).await, None).await;

    let fetched = fetch_tree(&base_url).await;
    let live = live_set(&workers);

    // A block size that disagrees with the local one is refused here, before
    // any tree mutation is even reachable: there is no second path to try.
    let mismatched = VettedSnapshot::from_wire(fetched.clone(), &live, Some(32));
    assert!(
        mismatched.is_err(),
        "a block-size mismatch must be refused by the only available bridge",
    );

    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(BLOCK_SIZE)).expect("vets clean");
    let new_tree = HashTree::new();
    assert!(
        vetted.graft_into(&new_tree).unwrap() > 0,
        "the vetted snapshot is the capability to graft",
    );
    assert_same_view(&warm.tree(), &new_tree, "after grafting through the bridge");
}

/// The rolling-update case. A replica that failed its own bootstrap is
/// "settled" with an empty tree; it must not be accepted as a source, or two
/// new replicas would bootstrap from each other and inherit nothing. The
/// producer has to say so itself, and vetting must refuse it regardless.
#[tokio::test]
async fn cold_sibling_is_rejected_as_a_source() {
    let cold = warm_index(|_| {});
    let (base_url, _server) = serve_snapshot(export_of(&cold).await, None).await;

    let fetched = fetch_tree(&base_url).await;
    assert!(
        !fetched.producer_ready,
        "a replica with an empty tree must not advertise itself as a source",
    );
    let err = VettedSnapshot::from_wire(fetched, &HashSet::new(), Some(BLOCK_SIZE))
        .expect_err("an empty snapshot must be refused");
    assert_eq!(err, VetError::ProducerCold);
}

/// A peer that does not serve the endpoint at all (older router image) reads as
/// "no snapshot", not as an error — so a mixed-version fleet degrades to cold
/// boots rather than failing.
#[tokio::test]
async fn older_peer_without_the_endpoint_reads_as_no_snapshot() {
    let (base_url, _server) =
        spawn_app(Router::new().route("/healthz", get(|| async { "ok" }))).await;

    let http = reqwest::Client::new();
    let got = fetch_snapshot(&http, &base_url, None)
        .await
        .expect("a 404 is not a transport error");
    assert!(
        matches!(got, FetchAnswer::NoBody(StatusCode::NOT_FOUND)),
        "404 must read as 'peer has no snapshot', with the status attached",
    );
}

/// A peer naming workers this replica has never discovered must not be able to
/// introduce them, and the surviving view must be exactly what a warm replica
/// that never knew them would hold.
#[tokio::test]
async fn unknown_workers_are_dropped_but_known_view_is_preserved() {
    let known = three_workers();
    let rogue = worker("http://attacker:30000", 0);

    // The warm replica also holds state for a worker the new replica has never
    // seen (e.g. one removed from discovery just before the new pod started):
    // a chain of its own, and a co-carrier on a known, mixed-tier chain that it
    // also extends by one block.
    let warm = warm_index(|t| {
        fill_warm(t, &known);
        t.insert(&rogue, None, &[91, 92, 93]);
        t.insert(&rogue, None, &[1, 2, 3, 4, 77]);
    });

    let (base_url, _server) = serve_snapshot(export_of(&warm).await, None).await;
    let fetched = fetch_tree(&base_url).await;

    let vetted = VettedSnapshot::from_wire(fetched, &live_set(&known), Some(BLOCK_SIZE)).unwrap();
    assert_eq!(vetted.dropped_workers(), 1, "the unknown worker is dropped");
    assert!(
        !vetted.has_worker(&rogue),
        "a worker absent from the local live set must never enter the tree",
    );

    let new_tree = HashTree::new();
    vetted.graft_into(&new_tree).unwrap();

    // What only the dropped worker carried is pruned, not grafted carrier-less:
    // a carrier-less tail below [1, 2, 3, 4] would turn that known hit into a
    // match with no owner.
    assert_eq!(
        new_tree.match_prefix(None, &[91, 92, 93]).matched_blocks,
        0,
        "a chain only the dropped worker carried must not be grafted",
    );
    let tail = new_tree.match_prefix(None, &[1, 2, 3, 4, 77]);
    assert_eq!(
        (tail.matched_blocks, tail.workers()),
        (4, HashSet::from([known[0].clone(), known[1].clone()])),
        "the match must stop at the deepest node a known worker carries",
    );

    // Dropping the co-carrier must leave the known carriers' tiers paired with
    // the right workers.
    let reference = HashTree::new();
    fill_warm(&reference, &known);
    assert_same_view(&reference, &new_tree, "after dropping an unknown worker");
}

/// A peer running a different page size produces incomparable block hashes.
/// Accepting it would silently destroy routing quality, so it must be refused.
#[tokio::test]
async fn mismatched_block_size_is_refused() {
    let workers = three_workers();
    let warm = warm_index(|t| fill_warm(t, &workers));
    let mut snap = export_of(&warm).await;
    snap.block_size = 32;
    let (base_url, _server) = serve_snapshot(snap, None).await;

    let fetched = fetch_tree(&base_url).await;
    let err = VettedSnapshot::from_wire(fetched, &live_set(&workers), Some(BLOCK_SIZE))
        .expect_err("a block-size mismatch must be refused");
    assert_eq!(
        err,
        VetError::BlockSizeMismatch {
            peer: 32,
            local: BLOCK_SIZE,
        }
    );
}

/// A one-rank body reporting `seq` for `http://w1:30000`, with or without a
/// one-node tree.
fn one_rank_body(seq: i64, with_tree: bool) -> PeerSnapshot {
    PeerSnapshot {
        format: SNAPSHOT_FORMAT,
        block_size: BLOCK_SIZE,
        is_bigram: false,
        producer_ready: true,
        workers: vec![WireWorker {
            url: "http://w1:30000".into(),
            dp_rank: 0,
        }],
        cursors: vec![(0, seq)],
        nodes: if with_tree {
            vec![SnapshotNode {
                parent: None,
                block_hash: 111,
                workers: vec![0],
                tiers: vec![],
            }]
        } else {
            Vec::new()
        },
        empty_ranks: vec![],
    }
}

/// The cheap path must answer the probe's question over the real transport.
#[tokio::test]
async fn fetch_cursors_reads_a_cursor_without_any_nodes() {
    // The full arm shares the cursor but carries a tree. Serving the same body
    // for both shapes would make the empty-nodes assert below blind to which
    // arm answered — i.e. to whether the parameter ever reached the peer.
    let (base, _server) =
        serve_snapshot(one_rank_body(99, true), Some(one_rank_body(99, false))).await;

    let got = fetch_witness(&base).await;
    assert!(
        got.nodes.is_empty(),
        "the answer must come from the cursors-only arm",
    );
    assert_eq!(got.wire_cursor_for("http://w1:30000", 0), Some(99));
}

/// A peer running an older image ignores the parameter and answers in full.
/// That must still yield the cursor — the mixed-version fleet keeps its witness
/// and merely pays the old transfer cost.
#[tokio::test]
async fn fetch_cursors_still_works_against_a_peer_that_ignores_the_parameter() {
    // `None` = this peer has no cursors-only behaviour at all.
    let (base, _server) = serve_snapshot(one_rank_body(7, true), None).await;

    let got = fetch_witness(&base).await;
    assert_eq!(
        got.wire_cursor_for("http://w1:30000", 0),
        Some(7),
        "a full body from an old peer must still answer the probe",
    );
    assert!(
        !got.nodes.is_empty(),
        "the old peer really did send its tree"
    );
}

/// The probe's verdict must not depend on which body shape the witness chose
/// to send. Both bodies come from one real export — the thin one is that
/// export with its tree stripped, the same cursor table a peer honouring
/// `cursors_only` would send — so shape is the only difference the consumer
/// sees. That the producer's own cursors-only branch reports the same cursor
/// is pinned against the real route in `server::routes::cache`'s tests.
#[tokio::test]
async fn a_cursors_only_witness_answers_the_same_as_a_full_one() {
    let w = worker("http://w1:30000", 0);
    let warm = warm_index(|t| t.insert(&w, None, &[111]));
    let full = with_cursors(export_of(&warm).await, &[(w, 500)]);
    let thin = PeerSnapshot {
        nodes: Vec::new(),
        ..full.clone()
    };

    // One server for a peer that honours the parameter...
    let (honours, _h1) = serve_snapshot(full.clone(), Some(thin)).await;
    // ...and one for a peer that ignores it.
    let (ignores, _h2) = serve_snapshot(full, None).await;

    let from_thin = fetch_witness(&honours).await;
    let from_full = fetch_witness(&ignores).await;

    assert!(
        from_thin.nodes.is_empty(),
        "the honouring peer really did answer with cursors alone",
    );
    assert!(
        !from_full.nodes.is_empty(),
        "the ignoring peer really did send its tree",
    );
    assert_eq!(
        from_thin.wire_cursor_for("http://w1:30000", 0),
        Some(500),
        "the cursors-only body must carry the cursor",
    );
    assert_eq!(
        from_thin.wire_cursor_for("http://w1:30000", 0),
        from_full.wire_cursor_for("http://w1:30000", 0),
        "the witness's answer must not depend on the body shape",
    );
}
