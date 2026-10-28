//! Remote-agent bookkeeping: a faithful mirror of `NixlKVManager`'s peer
//! tables (`decode_kv_args_table`, `_peer_reload_times`, the prep-handle
//! lifecycle) from nixl/conn.py:2030-2088, generic over the agent
//! operations so the NIXL shim and tests share one implementation.
//!
//! What stays in Python (for now): everything the prep hooks touch —
//! `_prepare_payload_xfer` / `_prepare_state_xfer` build descriptor plans
//! against live NIXL handles; they arrive with the backend port as the
//! `prepare` callback's innards. This module owns the DECISIONS: when a
//! peer is (re)loaded, rate-limited, invalidated, or dropped.

use std::collections::HashMap;
use std::time::Duration;

use crate::{Error, Result};

/// Python `_PEER_RELOAD_MIN_INTERVAL_S` (nixl/conn.py:76).
pub const PEER_RELOAD_MIN_INTERVAL: Duration = Duration::from_secs(1);

/// The NIXL agent calls the bookkeeping drives. Method names and semantics
/// mirror the Python nixl_agent API exactly.
pub trait PeerOps {
    /// `agent.add_remote_agent(metadata)` — loadRemoteMD. May lazily connect.
    fn add_remote_agent(&mut self, metadata: &[u8]) -> Result<()>;
    /// `agent.check_remote_metadata(name)` — true when the agent's metadata
    /// is still valid.
    fn check_remote_metadata(&mut self, name: &str) -> Result<bool>;
    /// `agent.remove_remote_agent(name)` — invalidateRemoteMD.
    fn remove_remote_agent(&mut self, name: &str) -> Result<()>;
}

/// One registered peer (a `KVArgsRegisterInfo` row plus reload state).
pub struct PeerEntry {
    /// The registration metadata — the reload source NIXL itself never
    /// restores (nixl/conn.py:2050-2051 comment).
    pub agent_metadata: Vec<u8>,
    /// Prep handles built (payload/state plans). Dropped on reload, matching
    /// the three `pop`s + `kv_xfer_segments = None` at nixl/conn.py:2068-2071.
    pub prepared: bool,
    /// Monotonic seconds of the last reload attempt (rate limit).
    last_reload: Option<f64>,
}

/// Outcome of [`PeerTable::add_remote_peer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddOutcome {
    /// New peer: stored, metadata loaded, prepare hook ran (prefill only).
    Added,
    /// nixl/conn.py:2032-2034 — duplicate registrations are logged and
    /// ignored; the agent sees NOTHING.
    AlreadyRegistered,
}

/// The peer table. `now` parameters are monotonic seconds (Python
/// `time.monotonic()`), injected so tests are deterministic.
#[derive(Default)]
pub struct PeerTable {
    peers: HashMap<String, PeerEntry>,
}

impl PeerTable {
    /// nixl/conn.py:2030-2047 `_add_remote_peer`.
    ///
    /// Exact Python ordering: dedupe → store → `add_remote_agent` →
    /// (prefill only) prepare. If the prepare hook fails the error
    /// propagates with the entry STORED and the agent ADDED — Python leaves
    /// the tables exactly so, and the reload path is what recovers.
    pub fn add_remote_peer<A: PeerOps>(
        &mut self,
        agent: &mut A,
        name: &str,
        agent_metadata: Vec<u8>,
        is_prefill: bool,
        prepare: impl FnOnce(&mut A, &str) -> Result<()>,
    ) -> Result<AddOutcome> {
        if self.peers.contains_key(name) {
            return Ok(AddOutcome::AlreadyRegistered);
        }
        self.peers.insert(
            name.to_string(),
            PeerEntry {
                agent_metadata,
                prepared: false,
                last_reload: None,
            },
        );
        agent.add_remote_agent(&self.peers[name].agent_metadata)?;
        if is_prefill {
            prepare(agent, name)?;
            self.peers.get_mut(name).unwrap().prepared = true;
        }
        Ok(AddOutcome::Added)
    }

    /// nixl/conn.py:2049-2088 `_reload_invalidated_peers`.
    ///
    /// For every agent named in `room_agents` that is registered here:
    /// healthy → skip; invalidated → rate-limited reload (drop prep state,
    /// re-add metadata, re-run prepare). A failed reload removes the remote
    /// and leaves the peer invalid so a later failure retries. Per-agent
    /// errors are contained (Python logs and continues) and collected in
    /// the return value, keyed by agent name.
    pub fn reload_invalidated<A: PeerOps>(
        &mut self,
        agent: &mut A,
        room_agents: &[String],
        now: f64,
        mut prepare: impl FnMut(&mut A, &str) -> Result<()>,
    ) -> Vec<(String, Error)> {
        let mut failures = Vec::new();
        for name in room_agents {
            let Some(entry) = self.peers.get_mut(name) else {
                // nixl/conn.py:2053-2055 — unknown rooms' agents are skipped.
                continue;
            };
            let result = reload_one(agent, name, entry, now, &mut prepare);
            if let Err(e) = result {
                failures.push((name.clone(), e));
            }
        }
        failures
    }

    /// Test/inspection accessors.
    pub fn contains(&self, name: &str) -> bool {
        self.peers.contains_key(name)
    }

    pub fn is_prepared(&self, name: &str) -> bool {
        self.peers.get(name).is_some_and(|e| e.prepared)
    }

    pub fn last_reload(&self, name: &str) -> Option<f64> {
        self.peers.get(name).and_then(|e| e.last_reload)
    }
}

fn reload_one<A: PeerOps>(
    agent: &mut A,
    name: &str,
    entry: &mut PeerEntry,
    now: f64,
    prepare: &mut impl FnMut(&mut A, &str) -> Result<()>,
) -> Result<()> {
    // Python's outer per-agent try: ANY error (including a raising
    // check_remote_metadata) is logged and the agent is skipped.
    if agent.check_remote_metadata(name)? {
        return Ok(());
    }
    // Inside `_peer_reload_lock`: re-check, then rate-limit. The lock itself
    // is the caller's concern (single-threaded here by construction).
    if agent.check_remote_metadata(name)? {
        return Ok(());
    }
    if let Some(last) = entry.last_reload
        && now - last < PEER_RELOAD_MIN_INTERVAL.as_secs_f64()
    {
        return Ok(());
    }
    entry.last_reload = Some(now);
    entry.prepared = false;
    agent.add_remote_agent(&entry.agent_metadata)?;
    if let Err(e) = prepare(agent, name) {
        // nixl/conn.py:2077-2080 — leave the peer invalid so a later
        // failure retries.
        let _ = agent.remove_remote_agent(name);
        return Err(e);
    }
    entry.prepared = true;
    Ok(())
}
