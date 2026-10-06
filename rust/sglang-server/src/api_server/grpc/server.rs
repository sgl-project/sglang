//! Tonic listener lifecycle for the native Rust frontend.
//!
//! The pre-bound socket, shared API runtime, and shutdown shape build on Rain
//! Jiang's multi-protocol prototype in `sgl-project/sglang#36923`. The listener
//! mounts the `api.v1` adapter on the transport-neutral
//! [`crate::frontend::FrontendHandle`] contract.

use sglang_api_types::api::v1::sglang_service_server::SglangServiceServer;
use tokio_stream::wrappers::TcpListenerStream;

use super::GrpcService;

/// Tonic's 4 MiB default is too small for large token or multimodal request
/// bodies; keep the limit the Python-backed gRPC server established.
const DEFAULT_GRPC_MAX_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

fn max_message_size() -> usize {
    match std::env::var("SGLANG_TONIC_PAYLOAD") {
        Ok(raw) => match raw.parse::<usize>() {
            Ok(bytes) if bytes > 0 => {
                tracing::info!(
                    bytes,
                    "using SGLANG_TONIC_PAYLOAD override for gRPC max message size"
                );
                bytes
            }
            _ => {
                tracing::warn!(
                    value = %raw,
                    default = DEFAULT_GRPC_MAX_MESSAGE_SIZE,
                    "ignoring invalid SGLANG_TONIC_PAYLOAD; using default"
                );
                DEFAULT_GRPC_MAX_MESSAGE_SIZE
            }
        },
        Err(_) => DEFAULT_GRPC_MAX_MESSAGE_SIZE,
    }
}

/// Serve the already-constructed adapter on a pre-bound listener until the
/// runtime-wide shutdown signal fires.
///
/// Selecting away from `serve_with_incoming` stops accepting without waiting
/// indefinitely for unfinished streams. Once both transport futures return,
/// the owner drops their shared API runtime, which cancels Tonic's active
/// connection tasks within the frontend's bounded thread-join deadline.
pub(crate) async fn serve(
    listener: std::net::TcpListener,
    service: GrpcService,
    shutdown: flume::Receiver<()>,
) {
    let addr = listener.local_addr().ok();
    let listener = match tokio::net::TcpListener::from_std(listener) {
        Ok(listener) => listener,
        Err(error) => {
            tracing::error!(%error, "failed to adopt pre-bound gRPC listener");
            return;
        }
    };
    let incoming = TcpListenerStream::new(listener);
    let max_message_size = max_message_size();
    let service = SglangServiceServer::new(service)
        .max_decoding_message_size(max_message_size)
        .max_encoding_message_size(max_message_size);
    let serve = tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_incoming(incoming);

    if let Some(addr) = addr {
        tracing::info!(%addr, "gRPC server listening");
    }
    tokio::select! {
        result = serve => {
            if let Err(error) = result {
                tracing::error!(%error, "gRPC serve exited");
            }
        }
        _ = shutdown.recv_async() => {
            tracing::info!("shutdown: stopping gRPC accepts; API runtime teardown cancels in-flight RPCs");
        }
    }
}
