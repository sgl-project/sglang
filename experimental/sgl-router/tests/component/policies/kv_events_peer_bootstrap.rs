// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end peer bootstrap: a booting replica must end up with the *same*
//! cache-aware view as the warm replica it copied from.
//!
//! These tests run the real transport — an axum server serving
//! `/internal/kv_snapshot`, a reqwest client fetching it, JSON over a loopback
//! socket — because the interesting failure modes live in the wire format and
//! the vetting step, not in the tree algebra (that is covered by the unit tests
//! on `HashTree::export_snapshot` / `restore_snapshot`).
//!
//! The equivalence assertion is deliberately behavioural: for a large query
//! set, `match_prefix` must return the same matched length and the same carrier
//! set on both replicas. That is the property routing actually depends on;
//! comparing node counts alone would pass while routing diverged.

use std::collections::HashSet;
use std::sync::Arc;

use axum::{routing::get, Json, Router};
use sgl_router::policies::kv_events::bootstrap::{
    fetch_cursors, fetch_snapshot, FetchAnswer, PeerSnapshot, VetError, VettedSnapshot,
    CURSORS_ONLY_PARAM, SNAPSHOT_PATH,
};
use sgl_router::policies::kv_events::{HashTree, KvWorkerId, SnapshotNode, Tiers, WireWorker};
use tokio::net::TcpListener;
use tower_http::compression::{CompressionLayer, CompressionLevel};

/// Fetch a snapshot that must exist: the body, or a panic naming the status.
async fn fetch_tree(http: &reqwest::Client, base_url: &str) -> PeerSnapshot {
    match fetch_snapshot(http, base_url, None)
        .await
        .expect("fetch succeeds")
    {
        FetchAnswer::Body(snap) => snap,
        FetchAnswer::NoBody(status) => panic!("peer serves a snapshot, got HTTP {status}"),
    }
}

/// Serve a fixed snapshot at the real path and return the base URL.
///
/// Mirrors production shape (GET, JSON body, 200) without needing a whole
/// router process, so the test stays fast while still crossing a socket.
async fn serve_snapshot(snap: PeerSnapshot) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = Arc::new(snap);
    let app = Router::new().route(
        SNAPSHOT_PATH,
        get(move || {
            let body = Arc::clone(&body);
            async move { Json((*body).clone()) }
        }),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

/// The snapshot path wrapped in a compression layer configured to MATCH
/// `server::app`'s snapshot route — a copy, not the production wiring, so this
/// proves only that the consumer decodes a compressed body over a real socket.
/// That the production route is actually wired to a layer is asserted separately,
/// against `build_router`, in `server::routes::cache`'s tests; without that this
/// helper would keep passing if the real layer were dropped.
async fn serve_snapshot_gzip(snap: PeerSnapshot) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let body = Arc::new(snap);
    let app = Router::new().route(
        SNAPSHOT_PATH,
        get(move || {
            let body = Arc::clone(&body);
            async move { Json((*body).clone()) }
        })
        .layer(
            CompressionLayer::new()
                .gzip(true)
                .no_br()
                .no_deflate()
                .no_zstd()
                .quality(CompressionLevel::Fastest),
        ),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

fn worker(url: &str, dp_rank: u32) -> KvWorkerId {
    KvWorkerId::new(url.to_string(), dp_rank)
}

/// Serve a body chosen by whether the request asked for cursors only, so one
/// server can stand in for both a patched and an unpatched peer.
async fn serve_by_shape(
    full: PeerSnapshot,
    cursors_only: Option<PeerSnapshot>,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let full = Arc::new(full);
    let thin = cursors_only.map(Arc::new);
    let app = Router::new().route(
        SNAPSHOT_PATH,
        get(move |uri: axum::http::Uri| {
            let full = Arc::clone(&full);
            let thin = thin.clone();
            async move {
                let asked_thin = uri
                    .query()
                    .is_some_and(|q| q.contains(&format!("{CURSORS_ONLY_PARAM}=true")));
                match (asked_thin, thin) {
                    // A patched peer honours the parameter.
                    (true, Some(t)) => Json((*t).clone()),
                    // An unpatched peer ignores it and answers in full.
                    _ => Json((*full).clone()),
                }
            }
        }),
    );
    let handle = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{addr}"), handle)
}

/// A tree shaped like a real one: shared prefixes across workers, divergent
/// tails, single-block chains, a hash occupying two positions, chains spread
/// across many shards, and carriers on mixed storage tiers — a host-only
/// holder and a device+host holder — so a snapshot that dropped or
/// mis-paired the tier table changes `device_workers` and is caught.
fn warm_tree(workers: &[KvWorkerId]) -> HashTree {
    let tree = HashTree::new();
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
        let w = want.match_prefix(None, &q);
        let g = got.match_prefix(None, &q);
        // Accessor before the field move: `device_workers()` borrows the
        // result, `workers` consumes it.
        assert_eq!(
            (g.matched_blocks, g.device_workers(), g.workers),
            (w.matched_blocks, w.device_workers(), w.workers),
            "{ctx}: match_prefix diverged for {q:?}",
        );
    }
}

fn snapshot_of(tree: &HashTree, cursors: &[(KvWorkerId, i64)]) -> PeerSnapshot {
    let (worker_table, nodes) = tree.export_snapshot();
    let cursor_wire = cursors
        .iter()
        .filter_map(|(w, seq)| {
            worker_table
                .iter()
                .position(|t| t == w)
                .map(|i| (i as u32, *seq))
        })
        .collect();
    PeerSnapshot {
        format: sgl_router::policies::kv_events::bootstrap::SNAPSHOT_FORMAT,
        block_size: 64,
        is_bigram: false,
        producer_ready: !nodes.is_empty(),
        workers: worker_table
            .iter()
            .map(|w| WireWorker {
                url: w.url.clone(),
                dp_rank: w.dp_rank,
            })
            .collect(),
        cursors: cursor_wire,
        nodes,
    }
}

/// The snapshot body is compressed in production, so the consumer must end up
/// with the same routing view over a gzipped transport as over an identity one.
/// The sibling tests all serve identity, which is what keeps a mixed-version
/// fleet (a peer predating route compression) covered.
#[tokio::test]
async fn new_replica_view_matches_warm_replica_over_gzipped_http() {
    let workers = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let old_tree = warm_tree(&workers);
    let (base_url, _server) = serve_snapshot_gzip(snapshot_of(&old_tree, &[])).await;

    // Prove the body really was compressed, so this test cannot silently
    // degrade into another identity-transport test if the layer is dropped.
    let raw = reqwest::Client::new()
        .get(format!("{base_url}{SNAPSHOT_PATH}"))
        .header(reqwest::header::ACCEPT_ENCODING, "gzip")
        .send()
        .await
        .expect("raw fetch succeeds");
    assert_eq!(
        raw.headers()
            .get(reqwest::header::CONTENT_ENCODING)
            .and_then(|v| v.to_str().ok()),
        Some("gzip"),
        "producer must compress the snapshot route when gzip is acceptable",
    );

    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    let live: HashSet<KvWorkerId> = workers.iter().cloned().collect();
    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(64)).expect("vets clean");
    let new_tree = HashTree::new();
    vetted.graft_into(&new_tree).expect("restore succeeds");
    assert_same_view(&old_tree, &new_tree, "gzipped bootstrap");
}

/// Guard on the blast radius of the compression change.
///
/// Enabling reqwest's crate-wide `gzip` feature would make `Accepts::default()`
/// set `gzip: true`, so every client in the process — including the proxy client
/// carrying SSE — would start advertising gzip and auto-decoding responses. The
/// snapshot fetch therefore asks for gzip on its own request instead. If someone
/// adds the feature to Cargo.toml, a default client starts advertising it and
/// this fails.
#[tokio::test]
async fn a_default_client_does_not_advertise_gzip() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().route(
        "/echo-accept-encoding",
        get(|headers: axum::http::HeaderMap| async move {
            headers
                .get(reqwest::header::ACCEPT_ENCODING)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("<absent>")
                .to_string()
        }),
    );
    let _server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let seen = reqwest::Client::new()
        .get(format!("http://{addr}/echo-accept-encoding"))
        .send()
        .await
        .expect("request succeeds")
        .text()
        .await
        .expect("body reads");
    assert!(
        !seen.contains("gzip"),
        "a default reqwest client must not advertise gzip — reqwest's crate-wide \
         `gzip` feature would flip every client in the process, including the SSE \
         proxy path. Got Accept-Encoding: {seen}",
    );
}

/// The headline guarantee: after bootstrapping over real HTTP, the new
/// replica's routing view is identical to the old replica's.
#[tokio::test]
async fn new_replica_view_matches_warm_replica_over_http() {
    let workers = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let old_tree = warm_tree(&workers);
    let (base_url, _server) = serve_snapshot(snapshot_of(&old_tree, &[])).await;

    // New replica: fetch, vet against its own live worker set, graft.
    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    let live: HashSet<KvWorkerId> = workers.iter().cloned().collect();
    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(64)).expect("vets clean");

    let new_tree = HashTree::new();
    vetted.graft_into(&new_tree).expect("restore succeeds");

    assert_same_view(&old_tree, &new_tree, "after peer bootstrap");
}

/// Cursors must survive the round trip, since they are what lets the new
/// replica filter the deltas the snapshot already reflects.
#[tokio::test]
async fn cursors_survive_the_round_trip() {
    let workers = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let old_tree = warm_tree(&workers);
    let cursors: Vec<(KvWorkerId, i64)> = vec![
        (workers[0].clone(), 41),
        (workers[1].clone(), 42),
        (workers[2].clone(), 43),
    ];
    let (base_url, _server) = serve_snapshot(snapshot_of(&old_tree, &cursors)).await;

    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    let live: HashSet<KvWorkerId> = workers.iter().cloned().collect();
    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(64)).unwrap();

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
/// public again, the equivalent bypass compiles and this comment is the record
/// of why it should not.
#[tokio::test]
async fn grafting_requires_going_through_vetting() {
    let workers = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let old_tree = warm_tree(&workers);
    let (base_url, _server) = serve_snapshot(snapshot_of(&old_tree, &[])).await;

    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    let live: HashSet<KvWorkerId> = workers.iter().cloned().collect();

    // A hostile-or-buggy peer's block size is refused here, before any tree
    // mutation is even reachable: there is no second path to try.
    let mismatched = VettedSnapshot::from_wire(fetched.clone(), &live, Some(32));
    assert!(
        mismatched.is_err(),
        "a block-size mismatch must be refused by the only available bridge",
    );

    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(64)).expect("vets clean");
    let new_tree = HashTree::new();
    assert!(
        vetted.graft_into(&new_tree).unwrap() > 0,
        "the vetted snapshot is the capability to graft",
    );
    assert_same_view(&old_tree, &new_tree, "after grafting through the bridge");
}

/// The rolling-update case. A replica that failed its own bootstrap is
/// "settled" with an empty tree; it must not be accepted as a snapshot source,
/// or two new replicas would bootstrap from each other and inherit nothing.
#[tokio::test]
async fn cold_sibling_is_rejected_as_a_source() {
    let empty = HashTree::new();
    let (base_url, _server) = serve_snapshot(snapshot_of(&empty, &[])).await;

    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    assert!(
        !fetched.producer_ready,
        "a replica with an empty tree must not advertise itself as a source",
    );
    let err = VettedSnapshot::from_wire(fetched, &HashSet::new(), Some(64))
        .expect_err("an empty snapshot must be refused");
    assert_eq!(err, VetError::ProducerCold);
}

/// A peer that does not serve the endpoint at all (older router image) reads as
/// "no snapshot", not as an error — so a mixed-version fleet degrades to cold
/// boots rather than failing.
#[tokio::test]
async fn older_peer_without_the_endpoint_reads_as_no_snapshot() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = Router::new().route("/healthz", get(|| async { "ok" }));
    let _server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let http = reqwest::Client::new();
    let got = fetch_snapshot(&http, &format!("http://{addr}"), None)
        .await
        .expect("a 404 is not a transport error");
    assert!(
        matches!(got, FetchAnswer::NoBody(reqwest::StatusCode::NOT_FOUND)),
        "404 must read as 'peer has no snapshot', with the status attached",
    );
}

/// A peer naming workers this replica has never discovered must not be able to
/// introduce them, and the surviving view must still be correct for the workers
/// both replicas know.
#[tokio::test]
async fn unknown_workers_are_dropped_but_known_view_is_preserved() {
    let known = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let rogue = worker("http://attacker:30000", 0);

    let old_tree = warm_tree(&known);
    // The warm replica also holds state for a worker the new replica has never
    // seen (e.g. one removed from discovery just before the new pod started).
    old_tree.insert(&rogue, None, &[91, 92, 93]);

    let (base_url, _server) = serve_snapshot(snapshot_of(&old_tree, &[])).await;
    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;

    let live: HashSet<KvWorkerId> = known.iter().cloned().collect();
    let vetted = VettedSnapshot::from_wire(fetched, &live, Some(64)).unwrap();
    assert_eq!(vetted.dropped_workers(), 1, "the unknown worker is dropped");
    assert!(
        !vetted.has_worker(&rogue),
        "a worker absent from the local live set must never enter the tree",
    );

    let new_tree = HashTree::new();
    vetted.graft_into(&new_tree).unwrap();

    // The rogue worker's chain is structurally present but carrier-less, so it
    // can never be selected.
    assert!(
        new_tree
            .match_prefix(None, &[91, 92, 93])
            .workers
            .is_empty(),
        "dropped worker must hold nothing",
    );
    // Everything the two replicas agree on still matches identically.
    for q in probe_queries() {
        let w = old_tree.match_prefix(None, &q);
        let g = new_tree.match_prefix(None, &q);
        assert_eq!(
            (g.matched_blocks, g.workers),
            (w.matched_blocks, w.workers),
            "known-worker view diverged for {q:?}",
        );
    }
}

/// A peer running a different page size produces incomparable block hashes.
/// Accepting it would silently destroy routing quality, so it must be refused.
#[tokio::test]
async fn mismatched_block_size_is_refused() {
    let workers = vec![
        worker("http://w1:30000", 0),
        worker("http://w2:30000", 0),
        worker("http://w2:30000", 1),
    ];
    let old_tree = warm_tree(&workers);
    let mut snap = snapshot_of(&old_tree, &[]);
    snap.block_size = 32;
    let (base_url, _server) = serve_snapshot(snap).await;

    let http = reqwest::Client::new();
    let fetched = fetch_tree(&http, &base_url).await;
    let live: HashSet<KvWorkerId> = workers.iter().cloned().collect();
    let err = VettedSnapshot::from_wire(fetched, &live, Some(64))
        .expect_err("a block-size mismatch must be refused");
    assert_eq!(
        err,
        VetError::BlockSizeMismatch {
            peer: 32,
            local: 64
        }
    );
}

/// The cheap path must answer the probe's question over the real transport.
#[tokio::test]
async fn fetch_cursors_reads_a_cursor_without_any_nodes() {
    let thin = PeerSnapshot {
        format: 1,
        block_size: 256,
        is_bigram: false,
        producer_ready: true,
        workers: vec![WireWorker {
            url: "http://w1:30000".into(),
            dp_rank: 0,
        }],
        cursors: vec![(0, 99)],
        nodes: Vec::new(),
    };
    // The full arm shares the cursor but carries a tree. Serving the same body
    // for both shapes would make the empty-nodes assert below blind to which
    // arm answered — i.e. to whether the parameter ever reached the peer.
    let full = PeerSnapshot {
        nodes: vec![SnapshotNode {
            parent: None,
            block_hash: 111,
            workers: vec![0],
            tiers: vec![],
        }],
        ..thin.clone()
    };
    let (base, handle) = serve_by_shape(full, Some(thin)).await;

    let got = fetch_cursors(&reqwest::Client::new(), &base)
        .await
        .expect("transport ok")
        .expect("peer answered");
    assert!(
        got.nodes.is_empty(),
        "the answer must come from the cursors-only arm",
    );
    assert_eq!(got.wire_cursor_for("http://w1:30000", 0), Some(99));
    handle.abort();
}

/// A peer running an older image ignores the parameter and answers in full.
/// That must still yield the cursor — the mixed-version fleet keeps its witness
/// and merely pays the old transfer cost.
#[tokio::test]
async fn fetch_cursors_still_works_against_a_peer_that_ignores_the_parameter() {
    let full = PeerSnapshot {
        format: 1,
        block_size: 256,
        is_bigram: false,
        producer_ready: true,
        workers: vec![WireWorker {
            url: "http://w1:30000".into(),
            dp_rank: 0,
        }],
        cursors: vec![(0, 7)],
        nodes: vec![SnapshotNode {
            parent: None,
            block_hash: 111,
            workers: vec![0],
            tiers: vec![],
        }],
    };
    // `None` = this peer has no cursors-only behaviour at all.
    let (base, handle) = serve_by_shape(full, None).await;

    let got = fetch_cursors(&reqwest::Client::new(), &base)
        .await
        .expect("transport ok")
        .expect("peer answered");
    assert_eq!(
        got.wire_cursor_for("http://w1:30000", 0),
        Some(7),
        "a full body from an old peer must still answer the probe",
    );
    assert!(
        !got.nodes.is_empty(),
        "the old peer really did send its tree"
    );
    handle.abort();
}

/// The cursor `fetch_cursors` returns must not depend on the witness's body
/// shape — the probe's verdict derives from that cursor alone. Both bodies
/// come from ONE source of truth here: the thin one is the same export with
/// the tree stripped, which for this fixture coincides with what the
/// producer's cursors-only branch emits (the one observed rank still carries
/// a node; the real branch builds its table live and ALSO reports observed
/// ranks the full export filters out). Two independently hand-built bodies
/// would prove only that `wire_cursor_for` ignores nodes, not that the
/// shapes agree in practice.
#[tokio::test]
async fn a_cursors_only_witness_answers_the_same_as_a_full_one() {
    let tree = HashTree::new();
    let w = worker("http://w1:30000", 0);
    tree.insert(&w, None, &[111]);
    let full = snapshot_of(&tree, &[(w, 500)]);
    let thin = PeerSnapshot {
        nodes: Vec::new(),
        ..full.clone()
    };

    // One server standing in for a patched peer: honour the parameter...
    let (patched, h1) = serve_by_shape(full.clone(), Some(thin)).await;
    // ...and one for an unpatched peer: ignore it.
    let (unpatched, h2) = serve_by_shape(full, None).await;

    let client = reqwest::Client::new();
    let from_thin = fetch_cursors(&client, &patched).await.unwrap().unwrap();
    let from_full = fetch_cursors(&client, &unpatched).await.unwrap().unwrap();

    assert!(
        from_thin.nodes.is_empty(),
        "the patched peer really did answer with cursors alone",
    );
    assert!(
        !from_full.nodes.is_empty(),
        "the unpatched peer really did send its tree",
    );
    assert_eq!(
        from_thin.wire_cursor_for("http://w1:30000", 0),
        from_full.wire_cursor_for("http://w1:30000", 0),
        "the witness's answer must not depend on the body shape",
    );
    h1.abort();
    h2.abort();
}
