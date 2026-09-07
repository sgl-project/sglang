// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::{env, io};

use sgl_kv_indexer::{
    server_builder_with_max_concurrent_streams, shutdown_signal, stamp_arrival,
    InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService,
    DEFAULT_PREFIX_QUERY_MAX_INFLIGHT, MAX_CONCURRENT_STREAMS,
};
use tonic::service::interceptor::InterceptedService;
use tracing::info;

const PREFIX_QUERY_MAX_INFLIGHT_ENV: &str = "KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT";
const MAX_CONCURRENT_STREAMS_ENV: &str = "KV_INDEXER_MAX_CONCURRENT_STREAMS";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let addr = std::env::var("KV_INDEXER_LISTEN_ADDR")
        .unwrap_or_else(|_| "[::1]:50051".to_string())
        .parse::<SocketAddr>()?;
    let prefix_query_max_inflight = prefix_query_max_inflight_from_env()?;
    let max_concurrent_streams = max_concurrent_streams_from_env()?;

    let backend: Arc<dyn KvIndexerBackend> = Arc::new(InMemoryKvIndexerBackend::new());
    // The interceptor timestamps each request before its own task is queued,
    // which is what lets the query path shed work whose deadline expired.
    let service = InterceptedService::new(
        KvIndexerService::with_prefix_query_max_inflight(backend, prefix_query_max_inflight)
            .into_server(),
        stamp_arrival,
    );

    info!(
        %addr,
        prefix_query_max_inflight,
        max_concurrent_streams,
        "starting single-server in-memory SGLang KV Indexer"
    );
    server_builder_with_max_concurrent_streams(max_concurrent_streams)
        .add_service(service)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

fn prefix_query_max_inflight_from_env() -> io::Result<usize> {
    match env::var(PREFIX_QUERY_MAX_INFLIGHT_ENV) {
        Ok(raw) => parse_prefix_query_max_inflight(&raw),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PREFIX_QUERY_MAX_INFLIGHT),
        Err(env::VarError::NotUnicode(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be valid UTF-8"),
        )),
    }
}

fn parse_prefix_query_max_inflight(raw: &str) -> io::Result<usize> {
    let value = raw.parse::<usize>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be a positive integer, got {raw:?}"),
        )
    })?;
    if value == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be greater than zero"),
        ));
    }
    Ok(value)
}

fn max_concurrent_streams_from_env() -> io::Result<u32> {
    match env::var(MAX_CONCURRENT_STREAMS_ENV) {
        Ok(raw) => parse_max_concurrent_streams(&raw),
        Err(env::VarError::NotPresent) => Ok(MAX_CONCURRENT_STREAMS),
        Err(env::VarError::NotUnicode(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{MAX_CONCURRENT_STREAMS_ENV} must be valid UTF-8"),
        )),
    }
}

fn parse_max_concurrent_streams(raw: &str) -> io::Result<u32> {
    let value = raw.parse::<u32>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{MAX_CONCURRENT_STREAMS_ENV} must be a positive integer, got {raw:?}"),
        )
    })?;
    if value == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{MAX_CONCURRENT_STREAMS_ENV} must be greater than zero"),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_positive_prefix_query_limit() {
        assert_eq!(parse_prefix_query_max_inflight("64").unwrap(), 64);
    }

    #[test]
    fn rejects_invalid_prefix_query_limit() {
        assert!(parse_prefix_query_max_inflight("0").is_err());
        assert!(parse_prefix_query_max_inflight("many").is_err());
    }

    #[test]
    fn parses_positive_stream_limit() {
        assert_eq!(parse_max_concurrent_streams("512").unwrap(), 512);
    }

    #[test]
    fn rejects_invalid_stream_limit() {
        assert!(parse_max_concurrent_streams("0").is_err());
        assert!(parse_max_concurrent_streams("many").is_err());
        assert!(parse_max_concurrent_streams("4294967296").is_err());
    }
}
