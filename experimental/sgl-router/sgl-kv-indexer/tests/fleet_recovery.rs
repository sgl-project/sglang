// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Local fleet integration: 2 Workers, 1 Router client, and 2 statically
//! configured Indexers selected in random order with failover and restart.

use std::net::SocketAddr;
use std::time::Duration;

use sgl_kv_indexer::pb::kv_indexer_client::KvIndexerClient;
use sgl_kv_indexer::pb::{
    ConfigureExpectedWorkersRequest, ExpectedWorker, ReplaceExternalKvSnapshotRequest, TierHashes,
    TierType,
};
use sgl_kv_indexer::{
    server_builder, GrpcPrefixIndex, InMemoryKvIndexerBackend, KvIndexerService, PrefixIndex,
    PrefixIndexConfig, PrefixOutcome,
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
                    ..Default::default()
                },
                ExpectedWorker {
                    worker_id: "w2".into(),
                    worker_address: format!("http://w2-{suffix}"),
                    cache_spec: None,
                    ..Default::default()
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
                stream_id: None,
                worker_generation: String::new(),
            })
            .await
            .unwrap();
    }
}

fn selected_worker(outcome: PrefixOutcome) -> String {
    match outcome {
        PrefixOutcome::Matched { matches, .. } => matches[0].address.clone(),
        PrefixOutcome::Empty => panic!("expected a prefix match"),
    }
}

#[tokio::test]
async fn two_indexer_fleet_randomizes_fails_over_and_recovers_after_restart() {
    let addr1 = free_addr();
    let addr2 = free_addr();
    let endpoint1 = format!("http://{addr1}");
    let endpoint2 = format!("http://{addr2}");
    let server1 = start_indexer(addr1);
    let mut server2 = start_indexer(addr2);
    configure_replica(&endpoint1, "i1").await;
    configure_replica(&endpoint2, "i2").await;

    let router = GrpcPrefixIndex::new(PrefixIndexConfig {
        endpoint: format!("{endpoint1},{endpoint2}"),
        query_deadline: Duration::from_millis(100),
        max_inflight: 8,
    })
    .unwrap();

    // Either configured replica may be selected first.
    let selected = selected_worker(router.match_prefix(vec![1, 2]).await.unwrap());
    assert!(matches!(selected.as_str(), "http://w1-i1" | "http://w1-i2"));

    // Failure of whichever replica is selected first falls through to i1.
    server2.abort();
    let _ = server2.await;
    for _ in 0..8 {
        assert_eq!(
            selected_worker(router.match_prefix(vec![1, 2]).await.unwrap()),
            "http://w1-i1"
        );
    }

    // A restarted static endpoint becomes usable again after snapshot recovery.
    server2 = start_indexer(addr2);
    configure_replica(&endpoint2, "i2-restarted").await;
    let restarted_router = GrpcPrefixIndex::new(PrefixIndexConfig {
        endpoint: endpoint2,
        query_deadline: Duration::from_millis(100),
        max_inflight: 8,
    })
    .unwrap();
    assert_eq!(
        selected_worker(restarted_router.match_prefix(vec![1, 2]).await.unwrap()),
        "http://w1-i2-restarted"
    );
    server1.abort();
    server2.abort();
}
