//! Mesh cluster metrics for Prometheus
//!
//! Implements all metrics required by issue #10839:
//! - Convergence latency
//! - Traffic metrics (batches, bytes)
//! - Snapshot metrics
//! - Peer health metrics
//! - State integrity metrics
//! - Rate-limit/LB drift metrics

use std::time::{Duration, Instant};

use metrics::{counter, describe_counter, describe_gauge, describe_histogram, gauge, histogram};

/// Initialize mesh metrics descriptions
pub fn init_mesh_metrics() {
    // Convergence latency
    describe_histogram!(
        "router_mesh_convergence_ms",
        "Time for state to converge across mesh in milliseconds"
    );

    // Traffic metrics
    describe_counter!(
        "router_mesh_batches_total",
        "Total number of state update batches sent/received"
    );
    describe_counter!("router_mesh_bytes_total", "Total bytes transmitted in mesh");

    // Snapshot metrics
    describe_counter!(
        "router_mesh_snapshot_trigger_total",
        "Total number of snapshot triggers"
    );
    describe_histogram!(
        "router_mesh_snapshot_duration_seconds",
        "Time to generate and send snapshot"
    );
    describe_counter!(
        "router_mesh_snapshot_bytes_total",
        "Total bytes in snapshots"
    );

    // Peer health metrics
    describe_gauge!(
        "router_mesh_peer_connections",
        "Number of active peer connections"
    );
    describe_counter!(
        "router_mesh_peer_reconnects_total",
        "Total number of peer reconnections"
    );
    describe_counter!("router_mesh_peer_ack_total", "Total number of ACK messages");
    describe_counter!(
        "router_mesh_peer_nack_total",
        "Total number of NACK messages"
    );

    // State integrity metrics
    describe_gauge!(
        "router_mesh_store_cardinality",
        "Number of entries in each store"
    );
    describe_gauge!(
        "router_mesh_store_hash",
        "Hash of store state for integrity checking"
    );

    // Rate-limit and LB drift metrics
    describe_gauge!(
        "router_rl_drift_ratio",
        "Rate-limit drift ratio (actual vs expected)"
    );
    describe_gauge!(
        "router_lb_drift_ratio",
        "Load balance drift ratio (actual vs expected)"
    );
}

/// Record convergence latency
pub fn record_convergence_latency(duration: Duration) {
    histogram!("router_mesh_convergence_ms",
        "quantile" => "p50"
    )
    .record(duration.as_millis() as f64);
}

/// Record batch transmission
pub fn record_batch_sent(peer: &str, batch_size: usize) {
    counter!("router_mesh_batches_total",
        "direction" => "sent",
        "peer" => peer.to_string()
    )
    .increment(1);
    counter!("router_mesh_bytes_total",
        "direction" => "sent",
        "peer" => peer.to_string()
    )
    .increment(batch_size as u64);
}

/// Record batch reception
pub fn record_batch_received(peer: &str, batch_size: usize) {
    counter!("router_mesh_batches_total",
        "direction" => "received",
        "peer" => peer.to_string()
    )
    .increment(1);
    counter!("router_mesh_bytes_total",
        "direction" => "received",
        "peer" => peer.to_string()
    )
    .increment(batch_size as u64);
}

/// Record snapshot trigger
pub fn record_snapshot_trigger(store: &str, reason: &str) {
    counter!("router_mesh_snapshot_trigger_total",
        "store" => store.to_string(),
        "reason" => reason.to_string()
    )
    .increment(1);
}

/// Record snapshot generation duration
pub fn record_snapshot_duration(store: &str, duration: Duration) {
    histogram!("router_mesh_snapshot_duration_seconds",
        "store" => store.to_string()
    )
    .record(duration.as_secs_f64());
}

/// Record snapshot bytes
pub fn record_snapshot_bytes(store: &str, direction: &str, bytes: usize) {
    counter!("router_mesh_snapshot_bytes_total",
        "store" => store.to_string(),
        "direction" => direction.to_string()
    )
    .increment(bytes as u64);
}

/// Update peer connection status
pub fn update_peer_connections(peer: &str, connected: bool) {
    gauge!("router_mesh_peer_connections",
        "peer" => peer.to_string()
    )
    .set(if connected { 1.0 } else { 0.0 });
}

/// Record peer reconnection
pub fn record_peer_reconnect(peer: &str) {
    counter!("router_mesh_peer_reconnects_total",
        "peer" => peer.to_string()
    )
    .increment(1);
}

/// Record ACK
pub fn record_ack(peer: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    counter!("router_mesh_peer_ack_total",
        "peer" => peer.to_string(),
        "status" => status.to_string()
    )
    .increment(1);
}

/// Record NACK
pub fn record_nack(peer: &str) {
    counter!("router_mesh_peer_nack_total",
        "peer" => peer.to_string()
    )
    .increment(1);
}

/// Update store cardinality
pub fn update_store_cardinality(store: &str, count: usize) {
    gauge!("router_mesh_store_cardinality",
        "store" => store.to_string()
    )
    .set(count as f64);
}

/// Update store hash (for integrity checking)
pub fn update_store_hash(store: &str, hash: u64) {
    gauge!("router_mesh_store_hash",
        "store" => store.to_string()
    )
    .set(hash as f64);
}

/// Update rate-limit drift ratio
pub fn update_rl_drift_ratio(key: &str, ratio: f64) {
    gauge!("router_rl_drift_ratio",
        "key" => key.to_string()
    )
    .set(ratio);
}

/// Update load balance drift ratio
pub fn update_lb_drift_ratio(model: &str, ratio: f64) {
    gauge!("router_lb_drift_ratio",
        "model" => model.to_string()
    )
    .set(ratio);
}

/// Helper struct for tracking convergence time
pub struct ConvergenceTracker {
    start_time: Instant,
}

impl ConvergenceTracker {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn record_convergence(&self) {
        let duration = self.start_time.elapsed();
        record_convergence_latency(duration);
    }
}

impl Default for ConvergenceTracker {
    fn default() -> Self {
        Self::new()
    }
}
