// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::{env, io};

use sgl_kv_indexer::{
    server_builder_with_max_concurrent_streams, shutdown_signal, stamp_arrival,
    InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService, ValkeyConfig,
    ValkeyKvIndexerBackend, DEFAULT_PREFIX_QUERY_MAX_INFLIGHT, MAX_CONCURRENT_STREAMS,
    VALKEY_DEFAULT_KEY_PREFIX,
};
use tonic::service::interceptor::InterceptedService;
use tracing::info;

const PREFIX_QUERY_MAX_INFLIGHT_ENV: &str = "KV_INDEXER_PREFIX_QUERY_MAX_INFLIGHT";
const MAX_CONCURRENT_STREAMS_ENV: &str = "KV_INDEXER_MAX_CONCURRENT_STREAMS";
/// `memory` (default) keeps the index in this process; `valkey` shares it
/// through the keyspace named by [`VALKEY_URL_ENV`].
const BACKEND_ENV: &str = "KV_INDEXER_BACKEND";
const VALKEY_URL_ENV: &str = "KV_INDEXER_VALKEY_URL";
const VALKEY_KEY_PREFIX_ENV: &str = "KV_INDEXER_VALKEY_KEY_PREFIX";
const VALKEY_CLUSTER_ENV: &str = "KV_INDEXER_VALKEY_CLUSTER";

#[derive(Debug, Clone, PartialEq, Eq)]
enum BackendChoice {
    Memory,
    Valkey(ValkeyConfig),
}

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

    let choice = backend_choice_from_env()?;
    let (backend, backend_name): (Arc<dyn KvIndexerBackend>, &str) = match &choice {
        BackendChoice::Memory => (Arc::new(InMemoryKvIndexerBackend::new()), "memory"),
        BackendChoice::Valkey(config) => {
            let backend = ValkeyKvIndexerBackend::connect(config.clone())
                .await
                .map_err(|status| {
                    io::Error::new(
                        io::ErrorKind::ConnectionRefused,
                        status.message().to_string(),
                    )
                })?;
            (Arc::new(backend), "valkey")
        }
    };
    // The interceptor timestamps each request before its own task is queued,
    // which is what lets the query path shed work whose deadline expired.
    let service = InterceptedService::new(
        KvIndexerService::with_prefix_query_max_inflight(backend, prefix_query_max_inflight)
            .into_server(),
        stamp_arrival,
    );

    match &choice {
        BackendChoice::Memory => info!(
            %addr,
            prefix_query_max_inflight,
            max_concurrent_streams,
            backend = backend_name,
            "starting single-server in-memory SGLang KV Indexer"
        ),
        BackendChoice::Valkey(config) => info!(
            %addr,
            prefix_query_max_inflight,
            max_concurrent_streams,
            backend = backend_name,
            key_prefix = %config.key_prefix,
            cluster = config.cluster,
            "starting SGLang KV Indexer over a shared Valkey keyspace"
        ),
    }
    server_builder_with_max_concurrent_streams(max_concurrent_streams)
        .add_service(service)
        .serve_with_shutdown(addr, shutdown_signal())
        .await?;

    Ok(())
}

fn backend_choice_from_env() -> io::Result<BackendChoice> {
    let kind = env_string(BACKEND_ENV)?.unwrap_or_else(|| "memory".to_string());
    match kind.as_str() {
        "memory" => Ok(BackendChoice::Memory),
        "valkey" => {
            let url = env_string(VALKEY_URL_ENV)?.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{BACKEND_ENV}=valkey requires {VALKEY_URL_ENV}"),
                )
            })?;
            let key_prefix = env_string(VALKEY_KEY_PREFIX_ENV)?
                .unwrap_or_else(|| VALKEY_DEFAULT_KEY_PREFIX.to_string());
            let cluster = match env_string(VALKEY_CLUSTER_ENV)?.as_deref() {
                None | Some("0") | Some("false") | Some("no") => false,
                Some("1") | Some("true") | Some("yes") => true,
                Some(other) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("{VALKEY_CLUSTER_ENV} must be 0/1, got {other:?}"),
                    ))
                }
            };
            Ok(BackendChoice::Valkey(
                ValkeyConfig::new(url)
                    .with_key_prefix(key_prefix)
                    .with_cluster(cluster),
            ))
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{BACKEND_ENV} must be \"memory\" or \"valkey\", got {other:?}"),
        )),
    }
}

fn env_string(name: &str) -> io::Result<Option<String>> {
    match env::var(name) {
        Ok(raw) => Ok(Some(raw)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{name} must be valid UTF-8"),
        )),
    }
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
