//! The sibling-replica set a snapshot may be pulled from.

use parking_lot::Mutex;
use rand::seq::SliceRandom;
use tracing::{debug, info};

/// Sibling replicas a snapshot may be pulled from. An empty set is conclusive
/// only after a sync and only if siblings were never seen: before the first
/// delivery it means "not told yet", and after siblings were seen it is a
/// transient dip (slice repack, rolling update). All state is behind one lock
/// so a reader never pairs fields from different writes.
#[derive(Debug, Default)]
pub struct PeerRegistry {
    state: Mutex<PeerState>,
}

#[derive(Debug, Default)]
struct PeerState {
    peers: Vec<String>,
    /// Whether a peer set has been delivered at least once, even an empty one.
    synced: bool,
    /// Whether a non-empty peer set has ever been delivered.
    ever_had_peers: bool,
}

impl PeerRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the peer set wholesale, as a relist or a slice event produces.
    pub fn replace(&self, peers: Vec<String>) {
        let changed = {
            let mut state = self.state.lock();
            state.synced = true;
            state.ever_had_peers |= !peers.is_empty();
            let changed = (state.peers != peers).then(|| peers.clone());
            state.peers = peers;
            changed
        };
        if let Some(peers) = changed {
            // Count at info; the full URL list only at debug — on a large
            // fleet every rolling update re-lists every sibling, and one log
            // line per change carrying every URL gets loud.
            info!(count = peers.len(), "kv-bootstrap: peer set updated");
            debug!(peers = ?peers, "kv-bootstrap: peer set contents");
        }
    }

    pub fn len(&self) -> usize {
        self.state.lock().peers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.state.lock().peers.is_empty()
    }

    /// Whether peer discovery has reported at least once. An empty peer set is
    /// only conclusive once this is true.
    pub fn synced(&self) -> bool {
        self.state.lock().synced
    }

    /// Peer count and `synced`, read together.
    pub fn len_and_synced(&self) -> (usize, bool) {
        let state = self.state.lock();
        (state.peers.len(), state.synced)
    }

    /// True when discovery has confirmed this replica has no siblings, so
    /// there is no point waiting for one.
    pub fn known_to_have_no_peers(&self) -> bool {
        let state = self.state.lock();
        state.synced && state.peers.is_empty() && !state.ever_had_peers
    }

    /// Candidate peers in shuffled order, so simultaneous boots spread their
    /// snapshot fetches instead of stampeding whichever peer sorts first.
    pub fn candidates(&self) -> Vec<String> {
        let mut peers = self.state.lock().peers.clone();
        peers.shuffle(&mut rand::thread_rng());
        peers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty peer set is only conclusive if siblings were never seen.
    #[test]
    fn transient_empty_peer_set_is_not_conclusive() {
        let r = PeerRegistry::new();
        assert!(!r.known_to_have_no_peers(), "unsynced is never conclusive");

        r.replace(vec![]);
        assert!(
            r.known_to_have_no_peers(),
            "synced + never any peers ⇒ genuinely alone",
        );

        r.replace(vec!["http://sibling:8090".into()]);
        r.replace(vec![]);
        assert!(
            !r.known_to_have_no_peers(),
            "once siblings have been seen, an empty set must be read as transient",
        );
    }

    #[test]
    fn peer_registry_shuffles_without_losing_entries() {
        let r = PeerRegistry::new();
        assert!(r.is_empty());
        let peers: Vec<String> = (0..16).map(|i| format!("http://r{i}")).collect();
        r.replace(peers.clone());
        assert_eq!(r.len(), 16);

        let mut got = r.candidates();
        assert_eq!(got.len(), 16);
        got.sort();
        let mut want = peers;
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn a_synced_empty_set_with_no_history_means_no_siblings() {
        let reg = PeerRegistry::new();
        reg.replace(vec![]);
        assert!(reg.synced());
        assert!(reg.known_to_have_no_peers());
    }
}
