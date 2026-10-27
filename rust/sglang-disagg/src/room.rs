//! Room state machines (`common/conn.py` `CommonKVManager` + `nixl/conn.py`
//! prefill bootstrap thread).
//!
//! A bootstrap room moves Failed=0 < Bootstrapping=1 < WaitingForInput=2 <
//! Transferring=3 < Success=4 (KVPoll). `update_status` is the single choke
//! point on both sides; its Python semantics — including the two places the
//! naive "monotone max" reading is wrong — are reproduced verbatim:
//!
//! - An incoming Failed sets Failed UNCONDITIONALLY, even over Success while
//!   the room is still alive (common/conn.py:665-672). A room reaches Success
//!   and then fails late (abort, lost peer) before clear() runs.
//! - A current Failed is TERMINAL: max() alone would promote it back since
//!   Failed sorts lowest (common/conn.py:673-676).
//!
//! "Failed after Success is impossible" therefore holds only because clear()
//! pops the room; a late Failed arriving after clear() is dropped by the
//! create-only-on-opening rule.
//!
//! Out of scope here (later phases): the PP-consensus `request_status`
//! adapter (PPConsensusStore — collective-based, stays Python-side until the
//! manager port), the deferred-bootstrap side effects (`_notify_bootstrap`
//! sends on Failed), and the sender-side Transferring/Success transitions
//! (backend-coupled).

use std::collections::{BTreeMap, BTreeSet};

use crate::{KVPoll, Room};

/// `CommonKVManager.DEFAULT_PREFILL_FAILURE_REASON` (common/conn.py:312-314).
pub const DEFAULT_PREFILL_FAILURE_REASON: &str =
    "Failed to get kvcache from prefill instance, it might be dead";

// --------------------------------------------------------------------------- //
// Status table — the shared update_status core (common/conn.py:650-677)
// --------------------------------------------------------------------------- //

/// request_status + failure records. `BTreeMap` for rank-replicated
/// iteration order (INV-1).
#[derive(Clone, Debug, Default)]
pub struct StatusTable {
    pub status: BTreeMap<Room, KVPoll>,
    pub failures: BTreeMap<Room, String>,
}

impl StatusTable {
    /// `update_status` (common/conn.py:650-677).
    ///
    /// - Room unknown: create ONLY on Bootstrapping | WaitingForInput — anything
    ///   else would resurrect a concluded room and pollute a later request
    ///   reusing the same bootstrap_room. (Upstream also pops deferred-ack and
    ///   staging state on create; those tables arrive with the backend phase —
    ///   the room-open path there MUST perform the same pops.)
    /// - Incoming Failed: set Failed unconditionally (overrides Success while
    ///   the room is alive). (Upstream also notifies deferred bootstrap; that
    ///   send is a manager-level side effect, returned by callers that need
    ///   it, not done here.)
    /// - Current Failed: terminal, no-op.
    /// - Else: max(current, status).
    pub fn update_status(&mut self, room: Room, status: KVPoll) {
        let current = self.status.get(&room).copied();
        match current {
            None => {
                if matches!(status, KVPoll::Bootstrapping | KVPoll::WaitingForInput) {
                    self.status.insert(room, status);
                }
            }
            Some(_) if status == KVPoll::Failed => {
                self.status.insert(room, KVPoll::Failed);
            }
            Some(KVPoll::Failed) => {}
            Some(c) => {
                self.status.insert(room, c.max(status));
            }
        }
    }

    /// Room exists in the table (`bootstrap_room in request_status`).
    pub fn contains(&self, room: Room) -> bool {
        self.status.contains_key(&room)
    }

    /// `check_status`: None when absent (Python raises KeyError; every
    /// caller guards with `in` first).
    pub fn check_status(&self, room: Room) -> Option<KVPoll> {
        self.status.get(&room).copied()
    }

    /// `record_failure` (common/conn.py:715-717): unconditional overwrite.
    pub fn record_failure(&mut self, room: Room, reason: impl Into<String>) {
        self.failures.insert(room, reason.into());
    }

    /// The ABORT-message room transition (common/conn.py:710-712): set
    /// Failed only when the room exists and has not reached Success; NO
    /// failure record (the aborting peer owns the root cause). Returns
    /// whether the transition fired.
    pub fn apply_abort_message(&mut self, room: Room) -> bool {
        let active = matches!(self.check_status(room), Some(s) if s != KVPoll::Success);
        if active {
            self.update_status(room, KVPoll::Failed);
        }
        active
    }

    /// The receiver `_clear` status pop (common/conn.py:2932). NOTE:
    /// failure records are NOT popped here — Python only pops them in
    /// `failure_exception` (common/conn.py:2945); a recorded root cause
    /// survives room cleanup so the error path can still read it.
    pub fn clear(&mut self, room: Room) {
        self.status.remove(&room);
    }

    /// `failure_records.pop(room, None)` (`failure_exception`,
    /// common/conn.py:2945) — returns the recorded root cause.
    pub fn pop_failure(&mut self, room: Room) -> Option<String> {
        self.failures.remove(&room)
    }
}

// --------------------------------------------------------------------------- //
// Decode side (common/conn.py CommonKVReceiver)
// --------------------------------------------------------------------------- //

/// What `apply_prefill_status` did — the caller's ordered side effects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApplyOutcome {
    /// Late/unknown room, non-terminal status, below quorum, or missing
    /// expected count (Python logs and returns; a recorded rank STAYS
    /// recorded).
    Ignored,
    /// WaitingForInput applied under defer_decode_allocation.
    DeferredWaiting,
    /// Failure recorded and room set Failed.
    Failed,
    /// The prefill-response quorum was reached. Python runs the staging
    /// completion hook (`submit_last_scatter_async`) BEFORE publishing
    /// Success so no poller can observe Success early; the caller MUST run
    /// its staging hook (if the room is a staging room) and then call
    /// [`DecodeRooms::publish_success`].
    SuccessQuorum { room: Room },
}

/// Decode-side room bookkeeping.
#[derive(Clone, Debug, Default)]
pub struct DecodeRooms {
    pub table: StatusTable,
    /// room -> set of prefill ranks that reported Success
    /// (`prefill_response_tracker`).
    pub response_tracker: BTreeMap<Room, BTreeSet<u32>>,
    /// room -> required prefill-response count
    /// (`required_prefill_response_num_table`).
    pub required_responses: BTreeMap<Room, u32>,
    /// Config flag. When true, a WaitingForInput status bypasses the
    /// tracker path in apply_prefill_status (deferred allocation requires
    /// one authoritative prefill rank and must not mix into the
    /// transfer-success counter).
    pub defer_decode_allocation: bool,
}

impl DecodeRooms {
    /// Receiver-side open for a request: seed the tracker and create the
    /// room at Bootstrapping.
    pub fn open(&mut self, room: Room, required: u32) {
        self.response_tracker.entry(room).or_default();
        self.required_responses.insert(room, required);
        self.table.update_status(room, KVPoll::Bootstrapping);
    }

    /// `apply_prefill_status` (common/conn.py:918-970).
    pub fn apply_prefill_status(
        &mut self,
        room: Room,
        status: KVPoll,
        prefill_rank: u32,
        reason: Option<&str>,
    ) -> ApplyOutcome {
        if !self.table.contains(room) {
            // The room concluded and was cleared. Recording a failure now
            // would leave an entry that a later request reusing this
            // bootstrap_room would pick up as its own root cause.
            return ApplyOutcome::Ignored;
        }
        match status {
            KVPoll::WaitingForInput if self.defer_decode_allocation => {
                self.table.update_status(room, status);
                ApplyOutcome::DeferredWaiting
            }
            KVPoll::Success => {
                let completed = match self.response_tracker.get_mut(&room) {
                    Some(c) => c,
                    None => return ApplyOutcome::Ignored,
                };
                completed.insert(prefill_rank);
                let expected = match self.required_responses.get(&room) {
                    Some(e) => *e,
                    // Python logs a warning and returns; the rank STAYS added.
                    None => return ApplyOutcome::Ignored,
                };
                if completed.len() < expected as usize {
                    return ApplyOutcome::Ignored;
                }
                ApplyOutcome::SuccessQuorum { room }
            }
            KVPoll::Failed => {
                self.table
                    .record_failure(room, reason.unwrap_or(DEFAULT_PREFILL_FAILURE_REASON));
                self.table.update_status(room, KVPoll::Failed);
                ApplyOutcome::Failed
            }
            _ => ApplyOutcome::Ignored,
        }
    }

    /// Publish the Success transition after the caller ran its staging
    /// completion hook (common/conn.py:963-967 ordering).
    pub fn publish_success(&mut self, room: Room) {
        self.table.update_status(room, KVPoll::Success);
    }

    /// The receiver's `abort()` (common/conn.py:2952-2959), minus the
    /// per-request `conclude_state` guard: record the abort root cause and
    /// set Failed. Unconditional like Python — on an unknown room
    /// update_status is a no-op (Failed is not an opening status) but the
    /// failure record lands, exactly as Python leaves it for
    /// failure_exception/clear to pop.
    pub fn abort(&mut self, room: Room) {
        self.table.record_failure(room, "Aborted by AbortReq.");
        self.table.update_status(room, KVPoll::Failed);
    }

    /// The receiver `_clear` table pops (common/conn.py:2932-2938): status,
    /// required-response count, response tracker. Failure records survive
    /// (see [`StatusTable::clear`]).
    pub fn clear(&mut self, room: Room) {
        self.table.clear(room);
        self.response_tracker.remove(&room);
        self.required_responses.remove(&room);
    }
}

// --------------------------------------------------------------------------- //
// Deferred abort ACK tracking (common/conn.py:973-1019)
// --------------------------------------------------------------------------- //

/// Deferred KV release: room -> current generation and the prefill ranks
/// that ACKed. The generation rejects an old request's late ACK after room
/// reuse.
#[derive(Clone, Debug, Default)]
pub struct DeferredAbortTracker {
    /// `_deferred_abort_generation`.
    generation: i64,
    /// `_deferred_abort_ack_tracker`.
    rooms: BTreeMap<Room, DeferredAbortAckState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DeferredAbortAckState {
    generation: i64,
    prefill_ranks: BTreeSet<u32>,
}

impl DeferredAbortTracker {
    /// `register_deferred_abort_room`: arm a new generation for a held room
    /// and return its wire id.
    pub fn register_room(&mut self, room: Room) -> i64 {
        self.generation += 1;
        self.rooms.insert(
            room,
            DeferredAbortAckState {
                generation: self.generation,
                prefill_ranks: BTreeSet::new(),
            },
        );
        self.generation
    }

    /// `note_abort_ack`: record an ACK only for the generation currently
    /// holding the room.
    pub fn note_ack(&mut self, room: Room, prefill_rank: u32, generation: i64) {
        if let Some(state) = self.rooms.get_mut(&room)
            && state.generation == generation
        {
            state.prefill_ranks.insert(prefill_rank);
        }
    }

    /// `is_abort_release_safe`: true once every prefill rank that could
    /// still write these pages has acked.
    pub fn is_release_safe(&self, room: Room, required_acks: usize) -> bool {
        let acks = self.rooms.get(&room).map_or(0, |s| s.prefill_ranks.len());
        acks >= required_acks
    }

    /// `clear_deferred_abort_state`.
    pub fn clear(&mut self, room: Room) {
        self.rooms.remove(&room);
    }
}

// --------------------------------------------------------------------------- //
// Prefill side (nixl/conn.py:3392-3413 bootstrap thread)
// --------------------------------------------------------------------------- //

/// The fields of one agent's TransferInfo the prefill room logic reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TransferInfoEntry {
    pub required_dst_info_num: i64,
    pub decode_prefix_len: Option<i64>,
    pub is_dummy: bool,
}

/// Prefill-side room bookkeeping.
#[derive(Clone, Debug, Default)]
pub struct PrefillRooms {
    pub table: StatusTable,
    /// room -> agent_name -> TransferInfo (transfer_infos).
    pub transfer_infos: BTreeMap<Room, BTreeMap<String, TransferInfoEntry>>,
    /// room -> decode prefix captured at completion
    /// (`req_to_decode_prefix_len`).
    pub decode_prefix_lens: BTreeMap<Room, i64>,
}

impl PrefillRooms {
    /// The bootstrap thread's add-transfer-info path (nixl/conn.py:3392-3413):
    /// store `transfer_infos[room][agent_name]`; `required` is THIS message's
    /// required_dst_info_num; when the room's info count reaches it, the
    /// decode prefix is the first non-None decode_prefix_len across the
    /// room's infos (else 0), the prefix is recorded, and the room advances
    /// to WaitingForInput.
    ///
    /// (Upstream also calls `resolve_kv_replica_factor` here — metrics-only,
    /// not consensus; it arrives with the manager port.)
    ///
    /// Order note: Python scans dict insertion order; we scan BTreeMap
    /// (lexicographic agent-name) order (INV-1). The picked values agree in
    /// practice because every decode rank reports the same prefix for a room;
    /// the order is pinned so all ranks decide identically.
    ///
    /// Returns Some(decode_prefix_len) on the completion transition, else None.
    pub fn add_transfer_info(
        &mut self,
        room: Room,
        agent_name: &str,
        info: TransferInfoEntry,
    ) -> Option<i64> {
        let infos = self.transfer_infos.entry(room).or_default();
        infos.insert(agent_name.to_string(), info);
        let required = info.required_dst_info_num;
        if infos.len() != required as usize {
            return None;
        }
        let prefix = infos
            .values()
            .find_map(|i| i.decode_prefix_len)
            .unwrap_or(0);
        self.decode_prefix_lens.insert(room, prefix);
        self.table.update_status(room, KVPoll::WaitingForInput);
        Some(prefix)
    }

    pub fn clear(&mut self, room: Room) {
        self.table.clear(room);
        self.transfer_infos.remove(&room);
        self.decode_prefix_lens.remove(&room);
    }
}
