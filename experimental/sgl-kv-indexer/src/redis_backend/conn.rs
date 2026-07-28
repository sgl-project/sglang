// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Topology-agnostic connection seam. The backend logic issues commands and
//! script invocations through `RedisConn`, so the same apply/match code runs on
//! a single Redis/Dragonfly instance or a Redis Cluster. Connections are cheaply
//! cloneable and multiplexed, so per-hash operations are issued concurrently
//! (one connection, many in-flight commands) rather than serialized.
//!
//! Connections are established lazily and cached: an eager constructor forces
//! the first connect (and surfaces its error) for the "Redis is required"
//! startup path, while a deferred constructor lets the server come up before
//! Redis is reachable and connect on first use. Every attempt is bounded by a
//! connection timeout / limited retries so an unreachable Redis can never wedge
//! startup or a request indefinitely.

use std::sync::Mutex;
use std::time::Duration;

use redis::{
    Cmd, ConnectionAddr, ConnectionInfo, ErrorKind, IntoConnectionInfo, RedisError, RedisResult,
    Value,
};

use super::scripts::RedisScript;

/// Per-attempt connect timeout and bounded retry policy. Without this the
/// redis `ConnectionManager` default has no connection timeout, so a single
/// attempt against an unreachable endpoint can block indefinitely.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
const RESPONSE_TIMEOUT: Duration = Duration::from_secs(5);
const CONNECT_RETRIES: usize = 2;
const CLUSTER_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);

fn manager_config() -> redis::aio::ConnectionManagerConfig {
    redis::aio::ConnectionManagerConfig::new()
        .set_connection_timeout(CONNECT_TIMEOUT)
        .set_response_timeout(RESPONSE_TIMEOUT)
        .set_number_of_retries(CONNECT_RETRIES)
        .set_factor(100)
        .set_max_delay(1000)
}

fn response_timeout_error() -> RedisError {
    RedisError::from((ErrorKind::IoError, "redis response timed out"))
}

fn eval_command(script: &RedisScript, keys: &[String], args: &[String]) -> Cmd {
    let mut command = redis::cmd("EVAL");
    command.arg(script.code).arg(keys.len());
    for key in keys {
        command.arg(key);
    }
    for arg in args {
        command.arg(arg);
    }
    command
}

fn redirected_info(base: &ConnectionInfo, address: &str) -> RedisResult<ConnectionInfo> {
    let (host, port) = if let Some(rest) = address.strip_prefix('[') {
        let (host, port) = rest.rsplit_once("]:").ok_or_else(|| {
            RedisError::from((ErrorKind::InvalidClientConfig, "invalid redirect address"))
        })?;
        (host.to_string(), port)
    } else {
        let (host, port) = address.rsplit_once(':').ok_or_else(|| {
            RedisError::from((ErrorKind::InvalidClientConfig, "invalid redirect address"))
        })?;
        (host.to_string(), port)
    };
    let port = port.parse::<u16>().map_err(|_| {
        RedisError::from((
            ErrorKind::InvalidClientConfig,
            "invalid redirect address port",
        ))
    })?;
    let addr = match &base.addr {
        ConnectionAddr::Tcp(_, _) => ConnectionAddr::Tcp(host, port),
        ConnectionAddr::TcpTls {
            insecure,
            tls_params,
            ..
        } => ConnectionAddr::TcpTls {
            host,
            port,
            insecure: *insecure,
            tls_params: tls_params.clone(),
        },
        ConnectionAddr::Unix(_) => {
            return Err(RedisError::from((
                ErrorKind::InvalidClientConfig,
                "Redis Cluster does not support Unix redirect addresses",
            )));
        }
    };
    Ok(ConnectionInfo {
        addr,
        redis: base.redis.clone(),
    })
}

#[tonic::async_trait]
pub(crate) trait RedisConn: Send + Sync + 'static {
    /// Runs a single command, routed by its key on Cluster.
    async fn query(&self, cmd: Cmd) -> RedisResult<Value>;

    /// Runs a Lua script (EVALSHA with automatic NOSCRIPT fallback), routed by
    /// `keys[0]` on Cluster. All `keys` must share a hash tag.
    async fn invoke(
        &self,
        script: &RedisScript,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value>;
}

/// Single Redis/Dragonfly instance via an auto-reconnecting multiplexed manager,
/// established lazily so the server can start before Redis is reachable.
pub(crate) struct SingleConn {
    url: String,
    conn: Mutex<Option<redis::aio::ConnectionManager>>,
}

impl SingleConn {
    /// Eager: force the first connect now, surfacing any error (used when Redis
    /// is a required dependency and we want a fast, loud startup failure).
    pub(crate) async fn connect(url: &str) -> RedisResult<Self> {
        let this = Self::deferred(url);
        this.manager().await?;
        Ok(this)
    }

    /// Deferred: hold only the target; connect on first use. Never fails here,
    /// so the server can start degraded and self-heal once Redis is up.
    pub(crate) fn deferred(url: &str) -> Self {
        Self {
            url: url.to_string(),
            conn: Mutex::new(None),
        }
    }

    fn cached(&self) -> Option<redis::aio::ConnectionManager> {
        self.conn.lock().unwrap().clone()
    }

    /// Returns the cached manager or builds one. The (async) build happens
    /// without holding the lock; a lost race just drops the redundant manager.
    async fn manager(&self) -> RedisResult<redis::aio::ConnectionManager> {
        if let Some(c) = self.cached() {
            return Ok(c);
        }
        let client = redis::Client::open(self.url.as_str())?;
        let built =
            redis::aio::ConnectionManager::new_with_config(client, manager_config()).await?;
        let mut guard = self.conn.lock().unwrap();
        if let Some(existing) = guard.clone() {
            return Ok(existing);
        }
        *guard = Some(built.clone());
        Ok(built)
    }
}

#[tonic::async_trait]
impl RedisConn for SingleConn {
    async fn query(&self, cmd: Cmd) -> RedisResult<Value> {
        let mut c = self.manager().await?;
        cmd.query_async(&mut c).await
    }

    async fn invoke(
        &self,
        script: &RedisScript,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value> {
        let mut c = self.manager().await?;
        let mut inv = script.prepare_invoke();
        for k in &keys {
            inv.key(k.as_str());
        }
        for a in &args {
            inv.arg(a.as_str());
        }
        let v: Value = inv.invoke_async(&mut c).await?;
        Ok(v)
    }
}

/// Redis Cluster via the async cluster connection. MOVED/ASK redirects and slot
/// map refresh are handled by the client; each command/script is routed by key.
/// Established lazily and bounded by a connect timeout, mirroring `SingleConn`.
pub(crate) struct ClusterConn {
    nodes: Vec<String>,
    conn: Mutex<Option<redis::cluster_async::ClusterConnection>>,
}

impl ClusterConn {
    pub(crate) async fn connect(nodes: Vec<String>) -> RedisResult<Self> {
        let this = Self::deferred(nodes);
        this.connection().await?;
        Ok(this)
    }

    pub(crate) fn deferred(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            conn: Mutex::new(None),
        }
    }

    fn cached(&self) -> Option<redis::cluster_async::ClusterConnection> {
        self.conn.lock().unwrap().clone()
    }

    fn invalidate(&self) {
        *self.conn.lock().unwrap() = None;
    }

    async fn connection(&self) -> RedisResult<redis::cluster_async::ClusterConnection> {
        if let Some(c) = self.cached() {
            return Ok(c);
        }
        let client = redis::cluster::ClusterClient::new(self.nodes.clone())?;
        let built = match tokio::time::timeout(
            CLUSTER_CONNECT_TIMEOUT,
            client.get_async_connection(),
        )
        .await
        {
            Ok(res) => res?,
            Err(_) => {
                return Err(RedisError::from((
                    ErrorKind::IoError,
                    "redis cluster connect timed out",
                )))
            }
        };
        let mut guard = self.conn.lock().unwrap();
        if let Some(existing) = guard.clone() {
            return Ok(existing);
        }
        *guard = Some(built.clone());
        Ok(built)
    }

    async fn invoke_ask(
        &self,
        address: &str,
        script: &RedisScript,
        keys: &[String],
        args: &[String],
    ) -> RedisResult<Value> {
        let base = self.nodes[0].as_str().into_connection_info()?;
        let target = redirected_info(&base, address)?;
        let client = redis::Client::open(target)?;
        let mut connection =
            tokio::time::timeout(CONNECT_TIMEOUT, client.get_multiplexed_async_connection())
                .await
                .map_err(|_| {
                    RedisError::from((ErrorKind::IoError, "Redis ASK connect timed out"))
                })??;
        // ASKING only applies to the immediately following command. A pipeline
        // keeps it adjacent to EVAL on this dedicated target connection.
        let mut pipeline = redis::pipe();
        pipeline.cmd("ASKING").ignore();
        pipeline.add_command(eval_command(script, keys, args));
        tokio::time::timeout(
            RESPONSE_TIMEOUT,
            pipeline.query_async::<Value>(&mut connection),
        )
        .await
        .map_err(|_| response_timeout_error())?
    }
}

#[tonic::async_trait]
impl RedisConn for ClusterConn {
    async fn query(&self, cmd: Cmd) -> RedisResult<Value> {
        let mut c = self.connection().await?;
        match tokio::time::timeout(RESPONSE_TIMEOUT, cmd.query_async(&mut c)).await {
            Ok(result) => result,
            Err(_) => {
                self.invalidate();
                Err(response_timeout_error())
            }
        }
    }

    async fn invoke(
        &self,
        script: &RedisScript,
        keys: Vec<String>,
        args: Vec<String>,
    ) -> RedisResult<Value> {
        match self.query(eval_command(script, &keys, &args)).await {
            Err(error) if error.kind() == ErrorKind::Ask => {
                let Some((address, _)) = error.redirect_node() else {
                    return Err(error);
                };
                self.invoke_ask(address, script, &keys, &args).await
            }
            result => result,
        }
    }
}
