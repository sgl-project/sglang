//! Node state machine for cold start
//!
//! Manages node lifecycle: NotReady -> Joining -> SnapshotPull -> Converging -> Ready

use std::{
    collections::HashSet,
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use tracing::{debug, info, warn};

use super::stores::StateStores;

/// Node readiness state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeReadiness {
    /// Node is not ready (initial state)
    NotReady,
    /// Node is joining the cluster
    Joining,
    /// Node is pulling snapshot from peers
    SnapshotPull,
    /// Node is converging (applying state updates)
    Converging,
    /// Node is ready to serve traffic
    Ready,
}

impl NodeReadiness {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeReadiness::NotReady => "not_ready",
            NodeReadiness::Joining => "joining",
            NodeReadiness::SnapshotPull => "snapshot_pull",
            NodeReadiness::Converging => "converging",
            NodeReadiness::Ready => "ready",
        }
    }
}

/// Convergence detection configuration
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Time window for convergence detection (seconds)
    pub convergence_window: Duration,
    /// Minimum number of state updates without changes to consider converged
    pub min_stable_updates: usize,
    /// Timeout for snapshot pull (seconds)
    pub snapshot_timeout: Duration,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            convergence_window: Duration::from_secs(10),
            min_stable_updates: 5,
            snapshot_timeout: Duration::from_secs(60),
        }
    }
}

/// Convergence tracker
#[derive(Debug)]
struct ConvergenceTracker {
    last_update_time: Option<Instant>,
    stable_update_count: usize,
    last_state_hash: Option<u64>,
}

impl ConvergenceTracker {
    fn new() -> Self {
        Self {
            last_update_time: None,
            stable_update_count: 0,
            last_state_hash: None,
        }
    }

    fn record_update(&mut self, state_hash: u64, config: &ConvergenceConfig) -> bool {
        let now = Instant::now();

        if let Some(last_hash) = self.last_state_hash {
            if last_hash == state_hash {
                // State unchanged
                self.stable_update_count += 1;
            } else {
                // State changed, reset counter
                self.stable_update_count = 0;
            }
        } else {
            // First update
            self.stable_update_count = 0;
        }

        self.last_state_hash = Some(state_hash);
        self.last_update_time = Some(now);

        // Check if we've been stable long enough
        if let Some(last_time) = self.last_update_time {
            let elapsed = now.duration_since(last_time);
            if elapsed >= config.convergence_window
                && self.stable_update_count >= config.min_stable_updates
            {
                return true;
            }
        }

        false
    }

    fn reset(&mut self) {
        self.last_update_time = None;
        self.stable_update_count = 0;
        self.last_state_hash = None;
    }
}

/// Node state machine for managing cold start
#[derive(Debug)]
pub struct NodeStateMachine {
    readiness: Arc<RwLock<NodeReadiness>>,
    config: ConvergenceConfig,
    convergence_tracker: Arc<RwLock<ConvergenceTracker>>,
    snapshot_start_time: Arc<RwLock<Option<Instant>>>,
    stores: Arc<StateStores>,
}

impl NodeStateMachine {
    pub fn new(stores: Arc<StateStores>, config: ConvergenceConfig) -> Self {
        Self {
            readiness: Arc::new(RwLock::new(NodeReadiness::NotReady)),
            config,
            convergence_tracker: Arc::new(RwLock::new(ConvergenceTracker::new())),
            snapshot_start_time: Arc::new(RwLock::new(None)),
            stores,
        }
    }

    /// Get current readiness state
    pub fn readiness(&self) -> NodeReadiness {
        *self.readiness.read()
    }

    /// Transition to joining state
    pub fn start_joining(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == NodeReadiness::NotReady {
            *readiness = NodeReadiness::Joining;
            info!("Node state: NotReady -> Joining");
        }
    }

    /// Transition to snapshot pull state
    pub fn start_snapshot_pull(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == NodeReadiness::Joining {
            *readiness = NodeReadiness::SnapshotPull;
            *self.snapshot_start_time.write() = Some(Instant::now());
            info!("Node state: Joining -> SnapshotPull");
        }
    }

    /// Check if snapshot pull has timed out
    pub fn is_snapshot_timeout(&self) -> bool {
        if let Some(start_time) = *self.snapshot_start_time.read() {
            start_time.elapsed() > self.config.snapshot_timeout
        } else {
            false
        }
    }

    /// Transition to converging state
    pub fn start_converging(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == NodeReadiness::SnapshotPull {
            *readiness = NodeReadiness::Converging;
            *self.snapshot_start_time.write() = None;
            self.convergence_tracker.write().reset();
            info!("Node state: SnapshotPull -> Converging");
        }
    }

    /// Record a state update and check for convergence
    pub fn record_state_update(&self) -> bool {
        if self.readiness() != NodeReadiness::Converging {
            return false;
        }

        // Calculate a simple hash of store states
        let state_hash = self.calculate_state_hash();
        let mut tracker = self.convergence_tracker.write();
        let converged = tracker.record_update(state_hash, &self.config);

        if converged {
            self.transition_to_ready();
            return true;
        }

        false
    }

    /// Transition to ready state
    pub fn transition_to_ready(&self) {
        let mut readiness = self.readiness.write();
        if *readiness == NodeReadiness::Converging {
            *readiness = NodeReadiness::Ready;
            info!("Node state: Converging -> Ready");
        }
    }

    /// Check if node is ready
    pub fn is_ready(&self) -> bool {
        self.readiness() == NodeReadiness::Ready
    }

    /// Check if stores are empty (need snapshot)
    pub fn needs_snapshot(&self) -> bool {
        self.stores.membership.is_empty()
            || self.stores.worker.is_empty()
            || self.stores.policy.is_empty()
    }

    /// Calculate a simple hash of current state (for convergence detection)
    fn calculate_state_hash(&self) -> u64 {
        use std::{
            collections::hash_map::DefaultHasher,
            hash::{Hash, Hasher},
        };

        let mut hasher = DefaultHasher::new();
        self.stores.membership.len().hash(&mut hasher);
        self.stores.worker.len().hash(&mut hasher);
        self.stores.policy.len().hash(&mut hasher);
        self.stores.app.len().hash(&mut hasher);
        hasher.finish()
    }

    /// Reset state machine (for testing or recovery)
    pub fn reset(&self) {
        *self.readiness.write() = NodeReadiness::NotReady;
        self.convergence_tracker.write().reset();
        *self.snapshot_start_time.write() = None;
    }
}

impl Default for NodeStateMachine {
    fn default() -> Self {
        Self::new(
            Arc::new(StateStores::default()),
            ConvergenceConfig::default(),
        )
    }
}
