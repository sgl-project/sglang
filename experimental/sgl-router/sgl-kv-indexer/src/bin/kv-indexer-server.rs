// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use std::{env, io};

use sgl_kv_indexer::StreamStart;
use sgl_kv_indexer::{
    server_builder_with_max_concurrent_streams, shutdown_signal, stamp_arrival,
    InMemoryKvIndexerBackend, KvIndexerBackend, KvIndexerService, LivenessWatcher, StreamConsumer,
    StreamConsumerConfig, ValkeyConfig, ValkeyKvIndexerBackend, DEFAULT_PREFIX_QUERY_MAX_INFLIGHT,
    DEFAULT_SWEEP_INTERVAL, MAX_CONCURRENT_STREAMS, VALKEY_DEFAULT_KEY_PREFIX,
};
use tokio::sync::watch;
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
const VALKEY_REQUEST_TIMEOUT_ENV: &str = "KV_INDEXER_VALKEY_REQUEST_TIMEOUT_MS";
const VALKEY_CONNECT_TIMEOUT_ENV: &str = "KV_INDEXER_VALKEY_CONNECT_TIMEOUT_MS";
/// `grpc` (default) applies batches the bridges send over gRPC; `stream` also
/// consumes the Valkey event stream the bridges may publish to.
const EVENT_SOURCE_ENV: &str = "KV_INDEXER_EVENT_SOURCE";
const CONSUMER_NAME_ENV: &str = "KV_INDEXER_CONSUMER_NAME";
/// Overrides the consumer group; with `KV_INDEXER_STREAM_START=beginning` a
/// fresh group name replays the whole retained window into an empty keyspace.
const CONSUMER_GROUP_ENV: &str = "KV_INDEXER_CONSUMER_GROUP";
const STREAM_START_ENV: &str = "KV_INDEXER_STREAM_START";
/// `1` (default with the valkey backend) clears workers whose heartbeat expired.
const LIVENESS_ENV: &str = "KV_INDEXER_LIVENESS";
const LIVENESS_SWEEP_ENV: &str = "KV_INDEXER_LIVENESS_SWEEP_MS";

#[derive(Debug, Clone, PartialEq, Eq)]
enum BackendChoice {
    Memory,
    Valkey(ValkeyConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventSource {
    Grpc,
    Stream,
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

    let valkey = valkey_config_from_env()?;
    let choice = backend_choice_from_env(valkey.as_ref())?;
    let event_source = event_source_from_env(valkey.as_ref())?;
    let liveness = liveness_from_env(&choice)?;

    let mut valkey_backend: Option<ValkeyKvIndexerBackend> = None;
    let (backend, backend_name): (Arc<dyn KvIndexerBackend>, &str) = match &choice {
        BackendChoice::Memory => (Arc::new(InMemoryKvIndexerBackend::new()), "memory"),
        BackendChoice::Valkey(config) => {
            let connected = ValkeyKvIndexerBackend::connect(config.clone())
                .await
                .map_err(unavailable)?;
            valkey_backend = Some(connected.clone());
            (Arc::new(connected), "valkey")
        }
    };

    // One shutdown signal fans out to the gRPC server and every background task.
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    let mut tasks = Vec::new();
    if event_source == EventSource::Stream {
        let valkey = valkey.clone().ok_or_else(|| {
            config_error(format!(
                "{EVENT_SOURCE_ENV}=stream requires {VALKEY_URL_ENV}"
            ))
        })?;
        let consumer_name = consumer_name_from_env()?;
        let mut consumer_config = match choice {
            BackendChoice::Valkey(_) => StreamConsumerConfig::shared(consumer_name),
            BackendChoice::Memory => StreamConsumerConfig::private(consumer_name),
        };
        if let Some(group) = env_string(CONSUMER_GROUP_ENV)?.filter(|g| !g.is_empty()) {
            consumer_config.group = group;
            // Only the auto-generated private group belongs to one process; a
            // named one is shared with whoever else uses that name.
            consumer_config.destroy_group_on_exit = false;
            if choice == BackendChoice::Memory {
                tracing::warn!(
                    group = %consumer_config.group,
                    "an in-memory index in a named group acknowledges entries no other \
                     member of that group will see; give it a group of its own"
                );
            }
        }
        match env_string(STREAM_START_ENV)?.as_deref() {
            None => {}
            Some("tail") => consumer_config.start = StreamStart::Tail,
            Some("beginning") => consumer_config.start = StreamStart::Beginning,
            Some(other) => {
                return Err(config_error(format!(
                    "{STREAM_START_ENV} must be \"tail\" or \"beginning\", got {other:?}"
                ))
                .into())
            }
        }
        let consumer = StreamConsumer::connect(&valkey, consumer_config, Arc::clone(&backend))
            .await
            .map_err(unavailable)?;
        let stop = until(shutdown_rx.clone());
        tasks.push(tokio::spawn(async move {
            if let Err(status) = consumer.run(stop).await {
                tracing::error!(%status, "event stream consumer stopped");
            }
        }));
    }
    if let (Some(sweep), Some(backend), Some(valkey)) = (liveness, valkey_backend, valkey.clone()) {
        let watcher = LivenessWatcher::new(backend, valkey, sweep);
        let stop = until(shutdown_rx.clone());
        tasks.push(tokio::spawn(watcher.run(stop)));
    }

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
        backend = backend_name,
        event_source = ?event_source,
        liveness = liveness.is_some(),
        key_prefix = valkey.as_ref().map(|v| v.key_prefix.as_str()).unwrap_or(""),
        "starting SGLang KV Indexer"
    );
    if event_source == EventSource::Stream {
        // Stream applies hold a lease, a gRPC call does not, so mixing both sinks
        // for one fleet reintroduces the interleaving the lease prevents.
        tracing::warn!(
            "consuming the event stream while still serving ApplyExternalKvBatch: \
             point every bridge of a fleet at one sink, not both"
        );
    }
    server_builder_with_max_concurrent_streams(max_concurrent_streams)
        .add_service(service)
        .serve_with_shutdown(addr, until(shutdown_rx))
        .await?;
    for task in tasks {
        let _ = task.await;
    }

    Ok(())
}

/// Resolves once the shutdown flag is set (or its sender is gone).
async fn until(mut shutdown: watch::Receiver<bool>) {
    while !*shutdown.borrow() {
        if shutdown.changed().await.is_err() {
            return;
        }
    }
}

fn config_error(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

fn unavailable(status: tonic::Status) -> io::Error {
    io::Error::new(
        io::ErrorKind::ConnectionRefused,
        status.message().to_string(),
    )
}

fn valkey_config_from_env() -> io::Result<Option<ValkeyConfig>> {
    let Some(url) = env_string(VALKEY_URL_ENV)? else {
        return Ok(None);
    };
    let key_prefix =
        env_string(VALKEY_KEY_PREFIX_ENV)?.unwrap_or_else(|| VALKEY_DEFAULT_KEY_PREFIX.to_string());
    let cluster = match env_string(VALKEY_CLUSTER_ENV)?.as_deref() {
        None | Some("0") | Some("false") | Some("no") => false,
        Some("1") | Some("true") | Some("yes") => true,
        Some(other) => {
            return Err(config_error(format!(
                "{VALKEY_CLUSTER_ENV} must be 0/1, got {other:?}"
            )))
        }
    };
    let mut config = ValkeyConfig::new(url)
        .with_key_prefix(key_prefix)
        .with_cluster(cluster);
    if let Some(ms) = env_millis(VALKEY_REQUEST_TIMEOUT_ENV)? {
        config = config.with_request_timeout(ms);
    }
    if let Some(ms) = env_millis(VALKEY_CONNECT_TIMEOUT_ENV)? {
        config = config.with_connect_timeout(ms);
    }
    Ok(Some(config))
}

fn backend_choice_from_env(valkey: Option<&ValkeyConfig>) -> io::Result<BackendChoice> {
    let kind = env_string(BACKEND_ENV)?.unwrap_or_else(|| "memory".to_string());
    match kind.as_str() {
        "memory" => Ok(BackendChoice::Memory),
        "valkey" => valkey
            .cloned()
            .map(BackendChoice::Valkey)
            .ok_or_else(|| config_error(format!("{BACKEND_ENV}=valkey requires {VALKEY_URL_ENV}"))),
        other => Err(config_error(format!(
            "{BACKEND_ENV} must be \"memory\" or \"valkey\", got {other:?}"
        ))),
    }
}

fn event_source_from_env(valkey: Option<&ValkeyConfig>) -> io::Result<EventSource> {
    match env_string(EVENT_SOURCE_ENV)?.as_deref() {
        None | Some("grpc") => Ok(EventSource::Grpc),
        Some("stream") if valkey.is_some() => Ok(EventSource::Stream),
        Some("stream") => Err(config_error(format!(
            "{EVENT_SOURCE_ENV}=stream requires {VALKEY_URL_ENV}"
        ))),
        Some(other) => Err(config_error(format!(
            "{EVENT_SOURCE_ENV} must be \"grpc\" or \"stream\", got {other:?}"
        ))),
    }
}

/// `Some(sweep interval)` when liveness is on. Defaults on for the valkey
/// backend; the memory backend has no keyspace to watch.
fn liveness_from_env(choice: &BackendChoice) -> io::Result<Option<Duration>> {
    let enabled = match env_string(LIVENESS_ENV)?.as_deref() {
        None => matches!(choice, BackendChoice::Valkey(_)),
        Some("0") | Some("false") | Some("no") => false,
        Some("1") | Some("true") | Some("yes") => true,
        Some(other) => {
            return Err(config_error(format!(
                "{LIVENESS_ENV} must be 0/1, got {other:?}"
            )))
        }
    };
    if enabled && !matches!(choice, BackendChoice::Valkey(_)) {
        return Err(config_error(format!(
            "{LIVENESS_ENV}=1 requires {BACKEND_ENV}=valkey"
        )));
    }
    if !enabled {
        return Ok(None);
    }
    Ok(Some(
        env_millis(LIVENESS_SWEEP_ENV)?.unwrap_or(DEFAULT_SWEEP_INTERVAL),
    ))
}

fn consumer_name_from_env() -> io::Result<String> {
    if let Some(name) = env_string(CONSUMER_NAME_ENV)? {
        if !name.is_empty() {
            return Ok(name);
        }
    }
    let host = std::fs::read_to_string("/etc/hostname")
        .ok()
        .map(|h| h.trim().to_string())
        .filter(|h| !h.is_empty())
        .unwrap_or_else(|| "indexer".to_string());
    Ok(format!("{host}:{}", std::process::id()))
}

/// A positive millisecond count, or `None` when the variable is unset.
fn env_millis(name: &str) -> io::Result<Option<Duration>> {
    let Some(raw) = env_string(name)? else {
        return Ok(None);
    };
    match raw.parse::<u64>() {
        Ok(ms) if ms > 0 => Ok(Some(Duration::from_millis(ms))),
        _ => Err(config_error(format!(
            "{name} must be a positive integer of milliseconds, got {raw:?}"
        ))),
    }
}

fn env_string(name: &str) -> io::Result<Option<String>> {
    match env::var(name) {
        Ok(raw) => Ok(Some(raw)),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(env::VarError::NotUnicode(_)) => {
            Err(config_error(format!("{name} must be valid UTF-8")))
        }
    }
}

fn prefix_query_max_inflight_from_env() -> io::Result<usize> {
    match env::var(PREFIX_QUERY_MAX_INFLIGHT_ENV) {
        Ok(raw) => parse_prefix_query_max_inflight(&raw),
        Err(env::VarError::NotPresent) => Ok(DEFAULT_PREFIX_QUERY_MAX_INFLIGHT),
        Err(env::VarError::NotUnicode(_)) => Err(config_error(format!(
            "{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be valid UTF-8"
        ))),
    }
}

fn parse_prefix_query_max_inflight(raw: &str) -> io::Result<usize> {
    let value = raw.parse::<usize>().map_err(|_| {
        config_error(format!(
            "{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be a positive integer, got {raw:?}"
        ))
    })?;
    if value == 0 {
        return Err(config_error(format!(
            "{PREFIX_QUERY_MAX_INFLIGHT_ENV} must be greater than zero"
        )));
    }
    Ok(value)
}

fn max_concurrent_streams_from_env() -> io::Result<u32> {
    match env::var(MAX_CONCURRENT_STREAMS_ENV) {
        Ok(raw) => parse_max_concurrent_streams(&raw),
        Err(env::VarError::NotPresent) => Ok(MAX_CONCURRENT_STREAMS),
        Err(env::VarError::NotUnicode(_)) => Err(config_error(format!(
            "{MAX_CONCURRENT_STREAMS_ENV} must be valid UTF-8"
        ))),
    }
}

fn parse_max_concurrent_streams(raw: &str) -> io::Result<u32> {
    let value = raw.parse::<u32>().map_err(|_| {
        config_error(format!(
            "{MAX_CONCURRENT_STREAMS_ENV} must be a positive integer, got {raw:?}"
        ))
    })?;
    if value == 0 {
        return Err(config_error(format!(
            "{MAX_CONCURRENT_STREAMS_ENV} must be greater than zero"
        )));
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
