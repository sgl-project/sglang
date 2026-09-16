// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Worker liveness over Valkey keys.
//!
//! The bridge beside each worker writes `<prefix>alive:<worker>` with a TTL and
//! refreshes it at a third of that TTL, but only while the worker's serving port
//! accepts a TCP connection: the key expiring means the worker is gone (or the
//! bridge is, which is the deployer's trade-off; size the TTL above a bridge
//! restart). A permanent `<prefix>hb:<worker>` marker records that the worker
//! ever heartbeated, so a legacy bridge without heartbeats is never declared
//! dead.
//!
//! Indexer servers watch expiries two ways. Keyspace notifications
//! (`__keyevent@*__:expired`) clear a worker within a second of expiry; they
//! need `notify-keyspace-events` to include `Ex`, which the watcher enables with
//! `CONFIG SET` when it is allowed to. A periodic sweep over the worker set is
//! the backstop for missed notifications, managed services without `CONFIG`,
//! and cluster nodes whose notifications did not reach the subscribed node.
//!
//! Clearing goes through the normal apply path (`CLEAR_ALL_AT_TIER` at every
//! tier), so hit counts, pruning and parity semantics hold. The placements of a
//! returning worker are rebuilt from its own events; the phantom prefixes a
//! restart used to leave behind are gone.

use std::future::Future;
use std::net::ToSocketAddrs;
use std::time::Duration;

use redis::cluster::ClusterClientBuilder;
use redis::{AsyncConnectionConfig, ProtocolVersion, PushInfo, PushKind, Value};
use tokio::sync::mpsc;
use tonic::Status;
use tracing::{debug, info, warn};

use crate::valkey_backend::{
    cluster_nodes, connect_conn, valkey_status, Conn, ValkeyConfig, ValkeyKvIndexerBackend,
};

/// Default heartbeat TTL. Above a routine bridge restart, below the time a
/// provider tolerates routing to a dead worker's phantom prefixes.
pub const DEFAULT_HEARTBEAT_TTL: Duration = Duration::from_secs(30);
/// Default interval of the backstop sweep.
pub const DEFAULT_SWEEP_INTERVAL: Duration = Duration::from_secs(30);
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);
const RESUBSCRIBE_MIN: Duration = Duration::from_millis(500);
const RESUBSCRIBE_MAX: Duration = Duration::from_secs(10);

pub fn alive_key(prefix: &str, worker: &str) -> String {
    format!("{prefix}alive:{worker}")
}

pub fn marker_key(prefix: &str, worker: &str) -> String {
    format!("{prefix}hb:{worker}")
}

/// The worker id an expired key names, if it is one of ours.
fn worker_of_expired_key(prefix: &str, key: &str) -> Option<String> {
    key.strip_prefix(prefix)?
        .strip_prefix("alive:")
        .map(str::to_string)
}

/// `host:port` of an `http://host:port` worker address, for the TCP probe.
fn probe_target(worker_address: &str) -> Option<String> {
    let rest = worker_address
        .strip_prefix("http://")
        .or_else(|| worker_address.strip_prefix("https://"))
        .unwrap_or(worker_address);
    let host_port = rest.split('/').next()?;
    host_port.contains(':').then(|| host_port.to_string())
}

/// Bridge side: keeps `alive:<worker>` fresh while the worker answers.
pub struct Heartbeat {
    conn: Conn,
    alive: String,
    marker: String,
    ttl: Duration,
    probe: Option<String>,
}

impl Heartbeat {
    /// `worker_address` is probed with a TCP connect before every beat; an empty
    /// address disables the probe and the beat only tracks the bridge itself.
    pub async fn connect(
        config: &ValkeyConfig,
        worker_id: &str,
        worker_address: &str,
        ttl: Duration,
    ) -> Result<Self, Status> {
        Ok(Self {
            conn: connect_conn(config).await?,
            alive: alive_key(&config.key_prefix, worker_id),
            marker: marker_key(&config.key_prefix, worker_id),
            ttl,
            probe: probe_target(worker_address),
        })
    }

    pub async fn run(mut self, shutdown: impl Future<Output = ()>) {
        tokio::pin!(shutdown);
        let interval = (self.ttl / 3).max(Duration::from_millis(100));
        loop {
            match self.beat().await {
                Ok(true) => debug!(key = %self.alive, "heartbeat"),
                Ok(false) => warn!(key = %self.alive, "worker probe failed; skipping heartbeat"),
                Err(status) => warn!(%status, "heartbeat write failed"),
            }
            tokio::select! {
                () = tokio::time::sleep(interval) => {}
                () = &mut shutdown => break,
            }
        }
        // A clean stop lets the key expire on its own: the worker may still be
        // serving, and a restarting bridge picks the beat back up inside the TTL.
    }

    async fn beat(&mut self) -> Result<bool, Status> {
        if let Some(target) = &self.probe {
            if !probe_tcp(target).await {
                return Ok(false);
            }
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let mut pipe = redis::pipe();
        pipe.cmd("SET").arg(&self.marker).arg(1).ignore();
        pipe.cmd("SET")
            .arg(&self.alive)
            .arg(now_ms)
            .arg("PX")
            .arg(self.ttl.as_millis() as u64)
            .ignore();
        self.conn.exec(&pipe).await?;
        Ok(true)
    }
}

async fn probe_tcp(target: &str) -> bool {
    let Ok(mut addrs) = target.to_socket_addrs() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    matches!(
        tokio::time::timeout(PROBE_TIMEOUT, tokio::net::TcpStream::connect(addr)).await,
        Ok(Ok(_))
    )
}

/// Indexer side: clears workers whose heartbeat expired.
pub struct LivenessWatcher {
    backend: ValkeyKvIndexerBackend,
    valkey: ValkeyConfig,
    sweep_interval: Duration,
}

impl LivenessWatcher {
    pub fn new(
        backend: ValkeyKvIndexerBackend,
        valkey: ValkeyConfig,
        sweep_interval: Duration,
    ) -> Self {
        Self {
            backend,
            valkey,
            sweep_interval,
        }
    }

    /// Runs the notification subscriber and the sweep until `shutdown` resolves.
    pub async fn run(self, shutdown: impl Future<Output = ()>) {
        let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
        let notifications = tokio::spawn(Self::watch_expiries(
            self.backend.clone(),
            self.valkey.clone(),
            stop_rx,
        ));
        tokio::pin!(shutdown);
        loop {
            match self.sweep().await {
                Ok(cleared) if cleared > 0 => info!(cleared, "liveness sweep cleared dead workers"),
                Ok(_) => {}
                Err(status) => warn!(%status, "liveness sweep failed"),
            }
            tokio::select! {
                () = tokio::time::sleep(self.sweep_interval) => {}
                () = &mut shutdown => break,
            }
        }
        let _ = stop_tx.send(true);
        let _ = notifications.await;
    }

    /// Marked workers with no `alive` key. Returns how many were cleared.
    pub async fn sweep(&self) -> Result<usize, Status> {
        let workers = self.backend.worker_ids().await?;
        if workers.is_empty() {
            return Ok(0);
        }
        let prefix = self.backend.key_prefix();
        let mut pipe = redis::pipe();
        for worker in &workers {
            pipe.cmd("EXISTS").arg(marker_key(prefix, worker));
            pipe.cmd("EXISTS").arg(alive_key(prefix, worker));
        }
        let flags: Vec<i64> = self.backend.conn().run(&pipe).await?;
        let mut cleared = 0;
        for (worker, pair) in workers.iter().zip(flags.chunks(2)) {
            let (marked, alive) = (pair.first() == Some(&1), pair.get(1) == Some(&1));
            if marked && !alive && self.backend.clear_worker(worker).await? {
                info!(worker, "heartbeat expired; placements cleared");
                cleared += 1;
            }
        }
        Ok(cleared)
    }

    async fn watch_expiries(
        backend: ValkeyKvIndexerBackend,
        valkey: ValkeyConfig,
        mut stop: tokio::sync::watch::Receiver<bool>,
    ) {
        let mut delay = RESUBSCRIBE_MIN;
        loop {
            if *stop.borrow() {
                return;
            }
            let outcome = tokio::select! {
                outcome = Self::subscribe_and_clear(&backend, &valkey) => outcome,
                _ = stop.changed() => return,
            };
            match outcome {
                Ok(()) => delay = RESUBSCRIBE_MIN,
                Err(status) => {
                    warn!(%status, retry_in = ?delay, "expiry notifications unavailable; sweep remains active")
                }
            }
            tokio::select! {
                () = tokio::time::sleep(delay) => {}
                _ = stop.changed() => return,
            }
            delay = (delay * 2).min(RESUBSCRIBE_MAX);
        }
    }

    /// Subscribes to expiry notifications and clears workers as they arrive.
    /// Returns when the connection drops.
    async fn subscribe_and_clear(
        backend: &ValkeyKvIndexerBackend,
        valkey: &ValkeyConfig,
    ) -> Result<(), Status> {
        ensure_expiry_notifications(&mut backend.conn()).await;
        let (tx, mut rx) = mpsc::unbounded_channel::<PushInfo>();
        // The connection must outlive the subscription; dropping it ends it.
        let _connection = subscribe_expired(valkey, tx).await?;
        info!("subscribed to key expiry notifications");
        let prefix = backend.key_prefix().to_string();
        while let Some(push) = rx.recv().await {
            match push.kind {
                PushKind::PMessage => {
                    // [pattern, channel, payload]; the payload is the expired key.
                    let Some(Value::BulkString(key)) = push.data.get(2) else {
                        continue;
                    };
                    let Some(worker) =
                        worker_of_expired_key(&prefix, &String::from_utf8_lossy(key))
                    else {
                        continue;
                    };
                    match backend.clear_worker(&worker).await {
                        Ok(true) => info!(worker, "heartbeat expired; placements cleared"),
                        Ok(false) => debug!(worker, "expired heartbeat for unknown worker"),
                        Err(status) => warn!(worker, %status, "failed to clear expired worker"),
                    }
                }
                PushKind::Disconnection => {
                    return Err(Status::unavailable("expiry subscription disconnected"));
                }
                _ => {}
            }
        }
        Err(Status::unavailable("expiry subscription closed"))
    }
}

/// Held only to keep the subscription open; dropping it unsubscribes.
#[allow(dead_code)]
enum SubscribedConnection {
    Standalone(redis::aio::MultiplexedConnection),
    Cluster(redis::cluster_async::ClusterConnection),
}

/// A RESP3 connection subscribed to `__keyevent@*__:expired`, pushing into `tx`.
async fn subscribe_expired(
    valkey: &ValkeyConfig,
    tx: mpsc::UnboundedSender<PushInfo>,
) -> Result<SubscribedConnection, Status> {
    const PATTERN: &str = "__keyevent@*__:expired";
    if valkey.cluster {
        let client = ClusterClientBuilder::new(cluster_nodes(&valkey.url))
            .use_protocol(ProtocolVersion::RESP3)
            .connection_timeout(valkey.connect_timeout)
            .push_sender(tx)
            .build()
            .map_err(valkey_status)?;
        let mut conn = client.get_async_connection().await.map_err(valkey_status)?;
        conn.psubscribe(PATTERN).await.map_err(valkey_status)?;
        Ok(SubscribedConnection::Cluster(conn))
    } else {
        // Push notifications need RESP3; the URL query is how redis-rs selects it.
        let separator = if valkey.url.contains('?') { '&' } else { '?' };
        let url = format!("{}{separator}protocol=resp3", valkey.url);
        let client = redis::Client::open(url.as_str()).map_err(valkey_status)?;
        let config = AsyncConnectionConfig::new()
            .set_connection_timeout(Some(valkey.connect_timeout))
            .set_push_sender(tx);
        let mut conn = client
            .get_multiplexed_async_connection_with_config(&config)
            .await
            .map_err(valkey_status)?;
        conn.psubscribe(PATTERN).await.map_err(valkey_status)?;
        Ok(SubscribedConnection::Standalone(conn))
    }
}

/// Adds `E` and `x` to `notify-keyspace-events` when the server lets us. A
/// managed service that refuses `CONFIG` keeps its own setting; the sweep still
/// covers liveness there.
async fn ensure_expiry_notifications(conn: &mut Conn) {
    let mut pipe = redis::pipe();
    pipe.cmd("CONFIG").arg("GET").arg("notify-keyspace-events");
    let current: Vec<Vec<String>> = match conn.run(&pipe).await {
        Ok(current) => current,
        Err(status) => {
            warn!(%status, "CONFIG GET refused; relying on the liveness sweep unless notify-keyspace-events already includes Ex");
            return;
        }
    };
    let current = current.into_iter().flatten().nth(1).unwrap_or_default();
    let has_expired = current.contains('A') || current.contains('x');
    if current.contains('E') && has_expired {
        return;
    }
    let mut wanted = current.clone();
    if !wanted.contains('E') {
        wanted.push('E');
    }
    if !has_expired {
        wanted.push('x');
    }
    let mut pipe = redis::pipe();
    pipe.cmd("CONFIG")
        .arg("SET")
        .arg("notify-keyspace-events")
        .arg(&wanted);
    match conn.run::<Vec<String>>(&pipe).await {
        Ok(_) => info!(from = %current, to = %wanted, "enabled key expiry notifications"),
        Err(status) => warn!(%status, "CONFIG SET refused; relying on the liveness sweep"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expired_key_maps_to_worker() {
        assert_eq!(
            worker_of_expired_key("{p}:", "{p}:alive:worker-0"),
            Some("worker-0".to_string())
        );
        assert_eq!(worker_of_expired_key("{p}:", "{p}:hb:worker-0"), None);
        assert_eq!(worker_of_expired_key("{p}:", "{q}:alive:worker-0"), None);
        // Worker ids may themselves contain colons.
        assert_eq!(
            worker_of_expired_key("{p}:", "{p}:alive:http://10.0.0.1:30000"),
            Some("http://10.0.0.1:30000".to_string())
        );
    }

    #[test]
    fn probe_target_from_worker_address() {
        assert_eq!(
            probe_target("http://127.0.0.1:30001"),
            Some("127.0.0.1:30001".to_string())
        );
        assert_eq!(
            probe_target("http://host:30001/v1"),
            Some("host:30001".to_string())
        );
        assert_eq!(probe_target(""), None);
        assert_eq!(probe_target("http://noport"), None);
    }
}
