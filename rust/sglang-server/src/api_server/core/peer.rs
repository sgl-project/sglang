//! HTTP [`ControlTarget`] for another DP rank's listener; relays the message's own wire bytes.

use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use futures::future::BoxFuture;
use http::StatusCode;

use super::CoreError;
use super::fanout::{ControlTarget, RankId, RankReply, RequestBuilder};
use crate::message::config::DeploymentInfo;

/// The route a listener serves forwarded control requests on.
pub(crate) const FORWARD_PEER_PATH: &str = "/internal/peer";

/// Peers only need a connection to come up; the scheduler-side wait (a
/// flush's own `timeout_s`, a weight check) has no bound on the local path
/// either, so a response is waited for as long as it takes.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

/// One DP rank's listener, reached over HTTP.
pub(crate) struct Peer {
    ranks: [RankId; 1],
    url: String,
    client: reqwest::Client,
}

impl Peer {
    pub(crate) fn new(client: reqwest::Client, dp_rank: RankId, base_url: &str) -> Self {
        Self {
            ranks: [dp_rank],
            url: format!("{}{FORWARD_PEER_PATH}", base_url.trim_end_matches('/')),
            client,
        }
    }

    async fn forward(&self, build: &RequestBuilder<'_>) -> Result<Bytes, CoreError> {
        let header = build()
            .encode()
            .map_err(|error| CoreError::Internal(error.to_string()))?;
        let response = self
            .client
            .post(&self.url)
            .header(http::header::CONTENT_TYPE, "application/msgpack")
            .body(header)
            .send()
            .await
            .map_err(transport_error)?;
        let status = response.status();
        let body = response.bytes().await.map_err(transport_error)?;
        if status.is_success() {
            return Ok(body);
        }
        let message = String::from_utf8_lossy(&body).into_owned();
        Err(match status {
            // The exact variants the handlers match for their own 503/504.
            StatusCode::SERVICE_UNAVAILABLE => CoreError::Unavailable,
            StatusCode::GATEWAY_TIMEOUT => CoreError::DeadlineExceeded(message),
            status => CoreError::from_runtime_rejection(message, status.as_u16()),
        })
    }
}

fn transport_error(error: reqwest::Error) -> CoreError {
    if error.is_timeout() {
        CoreError::DeadlineExceeded(error.to_string())
    } else if error.is_connect() {
        CoreError::Unavailable
    } else {
        CoreError::Internal(error.to_string())
    }
}

impl ControlTarget for Peer {
    fn ranks(&self) -> &[RankId] {
        &self.ranks
    }

    fn control<'a>(&'a self, build: &'a RequestBuilder<'a>) -> BoxFuture<'a, Vec<RankReply>> {
        Box::pin(async move {
            vec![RankReply {
                rank: self.ranks[0],
                outcome: self.forward(build).await,
            }]
        })
    }
}

/// The target endpoints for every DP rank other than `own`.
pub(crate) fn peer_endpoints(
    deployment_info: &DeploymentInfo,
    own: RankId,
) -> Result<Vec<Arc<dyn ControlTarget>>, String> {
    let endpoints: Vec<_> = deployment_info
        .rust_server_endpoints
        .iter()
        .filter(|endpoint| endpoint.dp_rank != own)
        .collect();
    if endpoints.is_empty() {
        return Ok(Vec::new());
    }
    let client = reqwest::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|error| format!("control peer client: {error}"))?;
    Ok(endpoints
        .into_iter()
        .map(|endpoint| {
            Arc::new(Peer::new(
                client.clone(),
                endpoint.dp_rank,
                &endpoint.endpoint,
            )) as Arc<dyn ControlTarget>
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use std::future::IntoFuture;
    use std::sync::Mutex;

    use axum::{Router, routing::post};

    use super::*;
    use crate::api_server::core::CoreErrorKind;
    use crate::api_server::core::fanout::fan_out;
    use crate::message::config::RustServerEndpoint;
    use crate::message::ids::Rid;
    use crate::message::io_struct::{ControlRequest, FlushCacheReqInput};
    use crate::message::types::ForwardedControlReq;

    fn build() -> ControlRequest {
        ControlRequest::FlushCacheReqInput(FlushCacheReqInput::new(
            Rid::new().to_string(),
            Some(1.0),
        ))
    }

    /// A listener standing in for a peer: records each forwarded header and
    /// answers with `reply`.
    async fn peer_listener(reply: (StatusCode, Bytes)) -> (String, Arc<Mutex<Vec<Bytes>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let recorded = seen.clone();
        let app = Router::new().route(
            FORWARD_PEER_PATH,
            post(move |body: Bytes| {
                let recorded = recorded.clone();
                let reply = reply.clone();
                async move {
                    recorded.lock().unwrap().push(body);
                    reply
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(axum::serve(listener, app).into_future());
        (url, seen)
    }

    /// The peer receives the message's own wire bytes, rid included, and its
    /// reply body comes back as that rank's payload.
    #[tokio::test]
    async fn forwards_the_encoded_message_and_returns_the_reply_body() {
        let (url, seen) = peer_listener((StatusCode::OK, Bytes::from_static(b"reply-bytes"))).await;
        let peer = Peer::new(reqwest::Client::new(), 3, &url);
        let replies = peer.control(&build).await;
        assert_eq!(replies.len(), 1);
        assert_eq!(replies[0].rank, 3);
        assert_eq!(replies[0].outcome.as_ref().unwrap(), "reply-bytes");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        let forwarded = ForwardedControlReq::parse(seen[0].clone()).unwrap();
        assert_eq!(forwarded.rid.len(), 32, "a freshly minted uuid rid");
    }

    /// Peer-side statuses keep the categories the handlers match on, and an
    /// unreachable peer is `Unavailable` rather than a transport detail.
    #[tokio::test]
    async fn maps_peer_statuses_and_transport_failures_to_core_errors() {
        for (status, expected) in [
            (StatusCode::SERVICE_UNAVAILABLE, CoreErrorKind::Unavailable),
            (StatusCode::GATEWAY_TIMEOUT, CoreErrorKind::DeadlineExceeded),
            (StatusCode::BAD_REQUEST, CoreErrorKind::InvalidArgument),
            (StatusCode::INTERNAL_SERVER_ERROR, CoreErrorKind::Internal),
        ] {
            let (url, _) = peer_listener((status, Bytes::from_static(b"why"))).await;
            let peer = Peer::new(reqwest::Client::new(), 1, &url);
            let error = peer.control(&build).await.remove(0).outcome.unwrap_err();
            assert_eq!(error.kind(), expected, "{status}");
            if status == StatusCode::BAD_REQUEST {
                assert_eq!(error.to_string(), "why");
                assert_eq!(error.http_status(), status);
            }
        }

        let probe = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let unreachable = format!("http://{}", probe.local_addr().unwrap());
        drop(probe);
        let peer = Peer::new(reqwest::Client::new(), 1, &unreachable);
        let error = peer.control(&build).await.remove(0).outcome.unwrap_err();
        assert!(matches!(error, CoreError::Unavailable), "{error}");
    }

    /// The peer list is every listener in the table but this rank's own,
    /// and the fan-out over it lands each peer's reply on its rank.
    #[tokio::test]
    async fn control_peers_skip_the_local_rank_and_fan_out_by_rank() {
        use crate::api_server::core::fanout::testing::status_bytes;
        #[derive(serde::Deserialize)]
        struct Status {
            message: String,
        }
        let (url_a, _) = peer_listener((StatusCode::OK, status_bytes(true, "from 0"))).await;
        let (url_b, _) = peer_listener((StatusCode::OK, status_bytes(true, "from 2"))).await;
        let info = DeploymentInfo {
            rust_server_endpoints: vec![
                RustServerEndpoint {
                    dp_rank: 2,
                    endpoint: url_b,
                },
                RustServerEndpoint {
                    dp_rank: 1,
                    endpoint: "http://127.0.0.1:1".into(),
                },
                RustServerEndpoint {
                    dp_rank: 0,
                    endpoint: url_a,
                },
            ],
            ..Default::default()
        };
        assert!(
            peer_endpoints(&DeploymentInfo::default(), 0)
                .unwrap()
                .is_empty()
        );
        let peers = peer_endpoints(&info, 1).unwrap();
        assert_eq!(peers.len(), 2);
        let replies = fan_out(
            1,
            async { Ok(status_bytes(true, "local")) },
            &peers,
            &build,
            None,
        )
        .await;
        let messages: Vec<String> = replies
            .decode::<Status>("status")
            .unwrap()
            .into_iter()
            .map(|status| status.message)
            .collect();
        assert_eq!(messages, ["from 0", "local", "from 2"]);
    }
}
