//! `grpc.health.v1` on the node's first listener. Its `inference` service is
//! SERVING only while every Rust listener on the node passes `HealthCheck`, so
//! one probe of the base port covers each DP rank the node serves.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use sglang_api_types::api::v1 as api;
use sglang_api_types::api::v1::sglang_service_client::SglangServiceClient;
use tonic::transport::{Channel, Endpoint};
use tonic_health::ServingStatus;
use tonic_health::pb::health_server::{Health, HealthServer};
use tonic_health::server::HealthReporter;

use crate::api_server::core::HealthStatus;
use crate::api_server::handlers::CoreHandle;
use crate::message::config::DeploymentInfo;

const INFERENCE_SERVICE: &str = "inference";
pub(super) const POLL_INTERVAL: Duration = Duration::from_secs(5);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Room past a peer's own probe window for the call itself; arbitrary.
const PEER_DEADLINE_MARGIN: Duration = Duration::from_secs(5);

/// Another Rust listener on this node.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct NodePeer {
    dp_rank: u32,
    addr: SocketAddr,
}

/// The node's other listeners when the gRPC listener bound at `local_addr` is
/// the node's first (port offset 0); `None` on every other listener.
pub(crate) fn node_peers(
    deployment_info: &DeploymentInfo,
    local_addr: SocketAddr,
) -> Option<Vec<NodePeer>> {
    let listeners = &deployment_info.node_listeners;
    let first = listeners
        .iter()
        .find(|listener| listener.port_offset == 0)?;
    if first.grpc_port != Some(local_addr.port()) {
        return None;
    }
    // The node's listeners bind one `--host`; an unspecified one answers on loopback.
    let ip = match local_addr.ip() {
        ip if !ip.is_unspecified() => ip,
        IpAddr::V4(_) => Ipv4Addr::LOCALHOST.into(),
        IpAddr::V6(_) => Ipv6Addr::LOCALHOST.into(),
    };
    Some(
        listeners
            .iter()
            .filter(|listener| listener.port_offset != 0)
            .filter_map(|listener| {
                Some(NodePeer {
                    dp_rank: listener.dp_rank,
                    addr: SocketAddr::new(ip, listener.grpc_port?),
                })
            })
            .collect(),
    )
}

/// The node's health service. `inference` starts NOT_SERVING, follows a poll
/// of every rank each `interval`, and stays NOT_SERVING once a drain begins.
pub(super) async fn inference_service(
    core: CoreHandle,
    peers: Vec<NodePeer>,
    timeout: Duration,
    interval: Duration,
) -> HealthServer<impl Health> {
    let (reporter, service) = tonic_health::server::health_reporter();
    reporter
        .set_service_status(INFERENCE_SERVICE, ServingStatus::NotServing)
        .await;
    // Cancelled with the api runtime on shutdown.
    tokio::spawn(report(core, peers, reporter, timeout, interval));
    service
}

async fn report(
    core: CoreHandle,
    peers: Vec<NodePeer>,
    reporter: HealthReporter,
    timeout: Duration,
    interval: Duration,
) {
    let peers: Vec<_> = peers
        .into_iter()
        .map(|peer| (peer.dp_rank, connect(peer.addr)))
        .collect();
    let poll = async {
        let mut serving = false;
        loop {
            let failures = failures(&core, &peers, timeout).await;
            if failures.is_empty() != serving {
                serving = failures.is_empty();
                let status = if serving {
                    tracing::info!("every DP rank on this node is healthy; inference SERVING");
                    ServingStatus::Serving
                } else {
                    tracing::warn!(?failures, "inference NOT_SERVING");
                    ServingStatus::NotServing
                };
                reporter.set_service_status(INFERENCE_SERVICE, status).await;
            }
            tokio::time::sleep(interval).await;
        }
    };
    tokio::select! {
        _ = poll => {}
        () = core.drain_began() => {}
    }
    reporter
        .set_service_status(INFERENCE_SERVICE, ServingStatus::NotServing)
        .await;
}

fn connect(addr: SocketAddr) -> SglangServiceClient<Channel> {
    let channel = Endpoint::from_shared(format!("http://{addr}"))
        .expect("a socket address forms a valid URI")
        .connect_timeout(CONNECT_TIMEOUT)
        .connect_lazy();
    SglangServiceClient::new(channel)
}

/// One reason per unhealthy rank on the node; empty when every rank is healthy.
async fn failures(
    core: &CoreHandle,
    peers: &[(u32, SglangServiceClient<Channel>)],
    timeout: Duration,
) -> Vec<String> {
    let own = async {
        match core.probe_health(timeout).await {
            Ok(HealthStatus::Healthy) => None,
            Ok(status) => Some(format!("DP rank {}: {status:?}", core.dp_rank())),
            Err(error) => Some(format!("DP rank {}: {error}", core.dp_rank())),
        }
    };
    let peers = futures::future::join_all(peers.iter().map(|(dp_rank, client)| {
        let mut client = client.clone();
        async move {
            let call = client.health_check(api::HealthCheckRequest {});
            match tokio::time::timeout(timeout + PEER_DEADLINE_MARGIN, call).await {
                Ok(Ok(response)) if response.get_ref().healthy => None,
                Ok(Ok(_)) => Some(format!("DP rank {dp_rank}: not healthy")),
                Ok(Err(status)) => Some(format!("DP rank {dp_rank}: {}", status.message())),
                Err(_) => Some(format!("DP rank {dp_rank}: no answer")),
            }
        }
    }));
    let (own, peers) = tokio::join!(own, peers);
    own.into_iter().chain(peers.into_iter().flatten()).collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use sglang_api_types::api::v1::sglang_service_server::SglangServiceServer;
    use tokio_stream::wrappers::TcpListenerStream;
    use tonic::transport::Server;
    use tonic_health::pb::HealthCheckRequest;
    use tonic_health::pb::health_check_response::ServingStatus as Status;
    use tonic_health::pb::health_client::HealthClient;

    use super::*;
    use crate::api_server::grpc::service::GrpcService;
    use crate::api_server::handlers::{CoreConfig, CoreMetadata};
    use crate::message::config::{NodeRustListener, ServerArgs};
    use crate::tokenizer_manager::from_scheduler::ActivityCounter;
    use crate::tokenizer_manager::wiring::{AbortSource, TmEvent};

    /// One DP rank whose heartbeat moves, as a busy scheduler's does, until
    /// `heartbeat` is aborted.
    struct Rank {
        core: CoreHandle,
        heartbeat: tokio::task::JoinHandle<()>,
        _intake: flume::Receiver<TmEvent>,
        _abort: flume::Receiver<AbortSource>,
    }

    fn rank(startup_ready: bool) -> Rank {
        let (intake_tx, intake) = flume::unbounded();
        let (abort_tx, abort) = flume::unbounded();
        let activity = ActivityCounter::default();
        let core = CoreHandle::new(
            intake_tx,
            abort_tx,
            CoreConfig {
                response_capacity: 8,
                response_activity: activity.clone(),
                startup_ready,
                is_disaggregation: false,
                mm_limits: Default::default(),
                multimodal: false,
                peer_endpoints: Vec::new(),
                metadata: CoreMetadata::from(&ServerArgs::default()),
                metrics: None,
                request_log: None,
                generate_validator: None,
                abort_hook: None,
            },
        );
        let heartbeat = tokio::spawn(async move {
            loop {
                activity.fetch_add(1, Ordering::Relaxed);
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        });
        Rank {
            core,
            heartbeat,
            _intake: intake,
            _abort: abort,
        }
    }

    async fn bind() -> (SocketAddr, TcpListenerStream) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        (
            listener.local_addr().unwrap(),
            TcpListenerStream::new(listener),
        )
    }

    async fn health_client(
        core: CoreHandle,
        peers: Vec<NodePeer>,
        interval: Duration,
    ) -> HealthClient<Channel> {
        let service = inference_service(core, peers, Duration::from_secs(1), interval).await;
        let (addr, incoming) = bind().await;
        tokio::spawn(
            Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming),
        );
        let channel = Endpoint::from_shared(format!("http://{addr}"))
            .unwrap()
            .connect_lazy();
        HealthClient::new(channel)
    }

    async fn status(client: &mut HealthClient<Channel>) -> Status {
        let request = HealthCheckRequest {
            service: INFERENCE_SERVICE.into(),
        };
        client.check(request).await.unwrap().into_inner().status()
    }

    async fn wait_for(client: &mut HealthClient<Channel>, want: Status) {
        for _ in 0..500 {
            if status(client).await == want {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("inference never became {want:?}");
    }

    /// `inference` is SERVING only while this rank and every peer pass `HealthCheck`.
    #[tokio::test]
    async fn inference_serves_only_while_every_rank_on_the_node_is_healthy() {
        let own = rank(false);
        let peer = rank(true);
        let (peer_addr, incoming) = bind().await;
        let peer_service =
            GrpcService::for_test(peer.core.clone(), None, false, Duration::from_secs(5));
        tokio::spawn(
            Server::builder()
                .add_service(SglangServiceServer::new(peer_service))
                .serve_with_incoming(incoming),
        );
        let peers = vec![NodePeer {
            dp_rank: 1,
            addr: peer_addr,
        }];
        let mut client = health_client(own.core.clone(), peers, Duration::from_millis(10)).await;

        // This rank is still warming up.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(status(&mut client).await, Status::NotServing);
        own.core.mark_ready();
        wait_for(&mut client, Status::Serving).await;
        // The peer's scheduler stops producing output.
        peer.heartbeat.abort();
        wait_for(&mut client, Status::NotServing).await;
    }

    /// A drain turns `inference` off at once rather than at the next poll.
    #[tokio::test]
    async fn drain_turns_inference_off_without_waiting_for_a_poll() {
        let own = rank(true);
        let mut client =
            health_client(own.core.clone(), Vec::new(), Duration::from_secs(3600)).await;
        wait_for(&mut client, Status::Serving).await;
        own.core.begin_drain(false);
        wait_for(&mut client, Status::NotServing).await;
    }

    /// Only the node's first listener polls, reaching the others on loopback
    /// when the host is unspecified.
    #[test]
    fn only_the_first_listener_polls_the_others() {
        let listener = |port_offset: u16, dp_rank| NodeRustListener {
            port_offset,
            dp_rank,
            http_port: 30000 + port_offset,
            grpc_port: Some(31000 + port_offset),
        };
        let deployment_info = DeploymentInfo {
            node_listeners: vec![listener(0, 4), listener(1, 5)],
            ..Default::default()
        };
        assert_eq!(
            node_peers(&deployment_info, "0.0.0.0:31000".parse().unwrap()),
            Some(vec![NodePeer {
                dp_rank: 5,
                addr: "127.0.0.1:31001".parse().unwrap(),
            }])
        );
        assert_eq!(
            node_peers(&deployment_info, "0.0.0.0:31001".parse().unwrap()),
            None
        );
    }
}
