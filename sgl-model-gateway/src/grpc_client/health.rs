use std::time::Duration;

use tonic::{transport::Channel, Request};
use tracing::debug;

#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("grpc.health.v1");
}

/// gRPC health client (grpc.health.v1.Health).
#[derive(Clone)]
pub struct HealthClient {
    client: proto::health_client::HealthClient<Channel>,
}

impl HealthClient {
    pub async fn connect(endpoint: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let http_endpoint = if let Some(addr) = endpoint.strip_prefix("grpc://") {
            format!("http://{}", addr)
        } else {
            endpoint.to_string()
        };

        let channel = Channel::from_shared(http_endpoint)?
            .http2_keep_alive_interval(Duration::from_secs(30))
            .keep_alive_timeout(Duration::from_secs(10))
            .keep_alive_while_idle(true)
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_adaptive_window(true)
            .initial_stream_window_size(Some(16 * 1024 * 1024))
            .initial_connection_window_size(Some(32 * 1024 * 1024))
            .connect()
            .await?;

        Ok(Self {
            client: proto::health_client::HealthClient::new(channel),
        })
    }

    pub async fn check(
        &self,
        service: &str,
    ) -> Result<proto::HealthCheckResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending gRPC health check request");
        let mut client = self.client.clone();
        let request = Request::new(proto::HealthCheckRequest {
            service: service.to_string(),
        });
        let response = client.check(request).await?;
        Ok(response.into_inner())
    }
}
