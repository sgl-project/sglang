use std::time::Duration;

use tonic::{transport::Channel, Request};
use tracing::debug;

use crate::observability::otel_trace::inject_trace_context_grpc;

#[allow(clippy::all)]
pub mod proto {
    #![allow(clippy::all, unused_qualifications)]
    tonic::include_proto!("sglang.grpc.encoder");
}

/// gRPC client for SGLang encoder.
#[derive(Clone)]
pub struct SglangEncoderClient {
    client: proto::sglang_encoder_client::SglangEncoderClient<Channel>,
}

impl SglangEncoderClient {
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
            client: proto::sglang_encoder_client::SglangEncoderClient::new(channel),
        })
    }

    pub async fn encode(
        &self,
        req: proto::EncodeRequest,
    ) -> Result<proto::EncodeResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending gRPC encode request to encoder");
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        inject_trace_context_grpc(request.metadata_mut());

        let response = client.encode(request).await?;
        debug!("Encode response received");
        Ok(response.into_inner())
    }

    pub async fn send(
        &self,
        req: proto::SendRequest,
    ) -> Result<proto::SendResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending gRPC send request to encoder");
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        inject_trace_context_grpc(request.metadata_mut());

        let response = client.send(request).await?;
        debug!("Send response received");
        Ok(response.into_inner())
    }

    pub async fn scheduler_receive_url(
        &self,
        req: proto::SchedulerReceiveUrlRequest,
    ) -> Result<proto::SchedulerReceiveUrlResponse, Box<dyn std::error::Error + Send + Sync>> {
        debug!("Sending gRPC scheduler_receive_url request to encoder");
        let mut client = self.client.clone();
        let mut request = Request::new(req);

        inject_trace_context_grpc(request.metadata_mut());

        let response = client.scheduler_receive_url(request).await?;
        debug!("SchedulerReceiveUrl response received");
        Ok(response.into_inner())
    }
}
