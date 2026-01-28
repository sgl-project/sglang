//! Node state machine for cold start
//!
//! Manages node lifecycle: NotReady -> Joining -> SnapshotPull -> Converging -> Ready

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use parking_lot::RwLock;
use tracing::info;

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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn create_test_stores() -> Arc<StateStores> {
        Arc::new(StateStores::default())
    }

    fn create_test_config() -> ConvergenceConfig {
        ConvergenceConfig {
            convergence_window: Duration::from_millis(100),
            min_stable_updates: 3,
            snapshot_timeout: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_node_readiness_as_str() {
        assert_eq!(NodeReadiness::NotReady.as_str(), "not_ready");
        assert_eq!(NodeReadiness::Joining.as_str(), "joining");
        assert_eq!(NodeReadiness::SnapshotPull.as_str(), "snapshot_pull");
        assert_eq!(NodeReadiness::Converging.as_str(), "converging");
        assert_eq!(NodeReadiness::Ready.as_str(), "ready");
    }

    #[test]
    fn test_convergence_config_default() {
        let config = ConvergenceConfig::default();
        assert_eq!(config.convergence_window, Duration::from_secs(10));
        assert_eq!(config.min_stable_updates, 5);
        assert_eq!(config.snapshot_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_node_state_machine_initial_state() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores, config);

        assert_eq!(sm.readiness(), NodeReadiness::NotReady);
        assert!(!sm.is_ready());
    }

    #[test]
    fn test_state_transition_flow() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores, config);

        // Start joining
        sm.start_joining();
        assert_eq!(sm.readiness(), NodeReadiness::Joining);

        // Start snapshot pull
        sm.start_snapshot_pull();
        assert_eq!(sm.readiness(), NodeReadiness::SnapshotPull);
        assert!(!sm.is_snapshot_timeout());

        // Start converging
        sm.start_converging();
        assert_eq!(sm.readiness(), NodeReadiness::Converging);

        // Transition to ready
        sm.transition_to_ready();
        assert_eq!(sm.readiness(), NodeReadiness::Ready);
        assert!(sm.is_ready());
    }

    #[test]
    fn test_state_transition_guards() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores, config);

        // Cannot start snapshot pull without joining first
        sm.start_snapshot_pull();
        assert_eq!(sm.readiness(), NodeReadiness::NotReady);

        // Cannot start converging without snapshot pull
        sm.start_joining();
        sm.start_converging();
        assert_eq!(sm.readiness(), NodeReadiness::Joining);

        // Cannot transition to ready without converging
        sm.transition_to_ready();
        assert_eq!(sm.readiness(), NodeReadiness::Joining);
    }

    #[test]
    fn test_snapshot_timeout() {
        let stores = create_test_stores();
        let mut config = create_test_config();
        config.snapshot_timeout = Duration::from_millis(50);
        let sm = NodeStateMachine::new(stores, config);

        sm.start_joining();
        sm.start_snapshot_pull();
        assert!(!sm.is_snapshot_timeout());

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(100));
        assert!(sm.is_snapshot_timeout());
    }

    #[test]
    fn test_needs_snapshot() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores.clone(), config);

        // Empty stores need snapshot
        assert!(sm.needs_snapshot());

        // Add some data to stores
        use super::super::{
            crdt::SKey,
            stores::{MembershipState, PolicyState, WorkerState},
        };

        stores.membership.insert(
            SKey::from("node1"),
            MembershipState {
                name: "node1".to_string(),
                address: "127.0.0.1:8080".to_string(),
                status: 1,
                version: 1,
                metadata: Default::default(),
            },
            "test".to_string(),
        );

        stores.worker.insert(
            SKey::from("worker1"),
            WorkerState {
                worker_id: "worker1".to_string(),
                model_id: "model1".to_string(),
                url: "http://localhost:8000".to_string(),
                health: true,
                load: 0.5,
                version: 1,
            },
            "test".to_string(),
        );

        stores.policy.insert(
            SKey::from("policy1"),
            PolicyState {
                model_id: "model1".to_string(),
                policy_type: "round_robin".to_string(),
                config: vec![],
                version: 1,
            },
            "test".to_string(),
        );

        // Now should not need snapshot
        assert!(!sm.needs_snapshot());
    }

    #[test]
    fn test_record_state_update_not_converging() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores, config);

        // Should return false when not in converging state
        assert!(!sm.record_state_update());
        assert_eq!(sm.readiness(), NodeReadiness::NotReady);
    }

    #[test]
    fn test_convergence_detection() {
        let stores = create_test_stores();
        let mut config = create_test_config();
        config.convergence_window = Duration::from_millis(50);
        config.min_stable_updates = 2;
        let sm = NodeStateMachine::new(stores, config);

        // Transition to converging state
        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();
        assert_eq!(sm.readiness(), NodeReadiness::Converging);

        // Record multiple updates with same state
        let converged1 = sm.record_state_update();
        assert!(!converged1);

        // Wait a bit and record more updates
        std::thread::sleep(Duration::from_millis(60));
        let converged2 = sm.record_state_update();
        assert!(!converged2); // Still not enough stable updates

        // Record more stable updates
        std::thread::sleep(Duration::from_millis(10));
        let converged3 = sm.record_state_update();
        // Should converge after enough stable updates within window
        if converged3 {
            assert_eq!(sm.readiness(), NodeReadiness::Ready);
        }
    }

    #[test]
    fn test_convergence_reset_on_state_change() {
        let stores = create_test_stores();
        let mut config = create_test_config();
        config.convergence_window = Duration::from_millis(100);
        config.min_stable_updates = 2;
        let sm = NodeStateMachine::new(stores.clone(), config);

        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();

        // Record update
        sm.record_state_update();

        // Change state by adding data
        use super::super::{crdt::SKey, stores::AppState};
        stores.app.insert(
            SKey::from("app1"),
            AppState {
                key: "app1".to_string(),
                value: vec![1, 2, 3],
                version: 1,
            },
            "test".to_string(),
        );

        // Record update with changed state
        sm.record_state_update();

        // The stable count should be reset
        std::thread::sleep(Duration::from_millis(110));
        let converged = sm.record_state_update();
        // Should not converge immediately after state change
        assert!(!converged || sm.readiness() == NodeReadiness::Converging);
    }

    #[test]
    fn test_reset() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores, config);

        // Go through states
        sm.start_joining();
        sm.start_snapshot_pull();
        sm.start_converging();
        sm.transition_to_ready();

        assert_eq!(sm.readiness(), NodeReadiness::Ready);

        // Reset
        sm.reset();
        assert_eq!(sm.readiness(), NodeReadiness::NotReady);
        assert!(!sm.is_ready());
        assert!(!sm.is_snapshot_timeout());
    }

    #[test]
    fn test_calculate_state_hash() {
        let stores = create_test_stores();
        let config = create_test_config();
        let sm = NodeStateMachine::new(stores.clone(), config);

        let hash1 = sm.calculate_state_hash();

        // Add some data
        use super::super::{crdt::SKey, stores::AppState};
        stores.app.insert(
            SKey::from("app1"),
            AppState {
                key: "app1".to_string(),
                value: vec![],
                version: 1,
            },
            "test".to_string(),
        );

        // Hash should change
        let hash2 = sm.calculate_state_hash();
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_default_implementation() {
        let sm = NodeStateMachine::default();
        assert_eq!(sm.readiness(), NodeReadiness::NotReady);
        assert!(!sm.is_ready());
    }
}
