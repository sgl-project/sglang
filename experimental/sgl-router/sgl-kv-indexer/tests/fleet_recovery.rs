// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Local fleet integration: 2 Workers, 1 Router client, 2 Indexers, failover,
//! restart recovery, scale-out, and scale-in eligibility.

use std::net::SocketAddr;
use std::time::Duration;

use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ConfigureExpectedWorkersRequest, ExpectedWorker, ReplaceExternalKvSnapshotRequest, TierHashes,
    TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, IndexerStatusReport,
    KvIndexerService, PrefixIndex, PrefixIndexConfig, PrefixOutcome,
};

fn free_addr() -> SocketAddr {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
}

fn start_indexer(addr: SocketAddr) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        server_builder()
            .add_service(KvIndexerService::new(InMemoryKvIndexerBackend::new()).into_server())
            .serve(addr)
            .await
            .unwrap();
    })
}

async fn configure_replica(endpoint: &str, suffix: &str) {
    let mut client = loop {
        match KvIndexerClient::connect(endpoint.to_owned()).await {
            Ok(client) => break client,
            Err(_) => tokio::time::sleep(Duration::from_millis(10)).await,
        }
    };
    client
        .configure_expected_workers(ConfigureExpectedWorkersRequest {
            workers: vec![
                ExpectedWorker {
                    worker_id: "w1".into(),
                    worker_address: format!("http://w1-{suffix}"),
                    cache_spec: None,
                },
                ExpectedWorker {
                    worker_id: "w2".into(),
                    worker_address: format!("http://w2-{suffix}"),
                    cache_spec: None,
                },
            ],
        })
        .await
        .unwrap();
    for (worker, hashes) in [("w1", vec![1, 2]), ("w2", vec![1])] {
        client
            .replace_external_kv_snapshot(ReplaceExternalKvSnapshotRequest {
                worker_id: worker.into(),
                worker_address: format!("http://{worker}-{suffix}"),
                worker_epoch: format!("epoch-{suffix}"),
                applied_seq: 0,
                hashes_by_tier: vec![TierHashes {
                    tier: TierType::TierHbm as i32,
                    hashes,
                    component_masks: Vec::new(),
                    block_sizes: Vec::new(),
                }],
                cache_spec: None,
            })
            .await
            .unwrap();
    }
}

fn report(id: &str, endpoint: &str, load: f64, ready: bool) -> IndexerStatusReport {
    IndexerStatusReport {
        indexer_id: id.into(),
        endpoint: endpoint.into(),
        ready,
        normalized_load: load,
        ready_workers: if ready { 2 } else { 0 },
        total_workers: 2,
    }
}

fn selected_worker(outcome: PrefixOutcome) -> String {
    match outcome {
        PrefixOutcome::Matched { matches, .. } => matches[0].address.clone(),
        PrefixOutcome::Empty => panic!("expected a prefix match"),
    }
}

#[tokio::test]
async fn two_indexer_fleet_scales_fails_over_and_recovers_after_restart() {
    let addr1 = free_addr();
    let addr2 = free_addr();
    let endpoint1 = format!("http://{addr1}");
    let endpoint2 = format!("http://{addr2}");
    let server1 = start_indexer(addr1);
    let mut server2 = start_indexer(addr2);
    configure_replica(&endpoint1, "i1").await;
    configure_replica(&endpoint2, "i2").await;

    let router = GrpcPrefixIndex::new(PrefixIndexConfig {
        endpoint: endpoint1.clone(),
        query_deadline: Duration::from_millis(100),
        max_inflight: 8,
    })
    .unwrap();
    let registry = router.status_registry();

    // Start with one reported replica; this is the fleet's scale-out baseline.
    registry
        .record(report("i1", &endpoint1, 0.4, true))
        .unwrap();
    assert_eq!(
        selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i1"
    );

    // A lower-load second replica reports READY and immediately receives work.
    registry
        .record(report("i2", &endpoint2, 0.1, true))
        .unwrap();
    assert_eq!(
        selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i2"
    );

    // Failure of the selected replica falls through to the next READY member.
    server2.abort();
    registry
        .record(report("i2", "http://127.0.0.1:1", 0.0, true))
        .unwrap();
    assert_eq!(
        selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i1"
    );

    // Restart loses in-memory state; reconfiguration + snapshot replacement
    // restores it before the status report makes the replica eligible again.
    tokio::time::sleep(Duration::from_millis(30)).await;
    let restarted_addr = free_addr();
    let restarted_endpoint = format!("http://{restarted_addr}");
    server2 = start_indexer(restarted_addr);
    configure_replica(&restarted_endpoint, "i2-restarted").await;
    registry
        .record(report("i2", &restarted_endpoint, 0.0, true))
        .unwrap();
    assert_eq!(
        selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i2-restarted"
    );

    // Scale-in makes i2 ineligible without affecting i1.
    registry
        .record(report("i2", &restarted_endpoint, 0.0, false))
        .unwrap();
    registry
        .record(report("i1", &endpoint1, 0.4, true))
        .unwrap();
    assert_eq!(
        selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i1"
    );

    server1.abort();
    server2.abort();
}
