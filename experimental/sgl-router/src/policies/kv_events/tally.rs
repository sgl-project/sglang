// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Tally of the KV-cache events the pump applied, by event kind and by the
//! storage `medium` tag each carried. Rendered as
//! `sgl_router_kv_events_total` and `sgl_router_kv_event_blocks_total`.
//!
//! WHY this exists: an engine running a hierarchical cache publishes a
//! host-tier store for every block it backs up and a device-tier removal when
//! the device copy goes. Whether those tagged events reach the router, and at
//! what volume, was not observable anywhere — the tree consumed them and
//! nothing counted them — so a router discarding the tag looked identical to
//! an engine never sending it. Counting by medium makes the tier stream a
//! time series: on such a fleet `block_stored / CPU_PINNED` runs at about the
//! `block_removed / GPU` rate, and a `CPU_PINNED` row pinned at zero with
//! hicache enabled points at the publisher or the subscription, not the tree.
//!
//! Every cell is rendered, zeros included: the zero IS the finding.
//!
//! Label cardinality is fixed: the medium label is folded to the values the
//! tree can rank plus `untagged` (no `medium` field) and `unknown` (a string
//! this build does not recognise), so a misbehaving publisher cannot mint
//! series.

use std::sync::atomic::{AtomicU64, Ordering};

use super::tree::Tiers;

/// Event kinds, in the order [`EventTally`] stores them.
pub const EVENT_KINDS: [&str; 3] = ["block_stored", "block_removed", "all_blocks_cleared"];

/// Medium labels, in the order [`EventTally`] stores them: the wire strings
/// the tree ranks, read off [`Tiers::WIRE_MEDIA`] so the two can never
/// disagree, then the two folds.
pub const MEDIUM_LABELS: [&str; 6] = [
    Tiers::WIRE_MEDIA[0].0,
    Tiers::WIRE_MEDIA[1].0,
    Tiers::WIRE_MEDIA[2].0,
    Tiers::WIRE_MEDIA[3].0,
    "untagged",
    "unknown",
];
const UNTAGGED: usize = 4;
const UNKNOWN: usize = 5;
const _: () = assert!(
    Tiers::WIRE_MEDIA.len() == UNTAGGED,
    "MEDIUM_LABELS lists every WIRE_MEDIA entry before the folds",
);

/// Which event a tally entry describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EventKind {
    BlockStored = 0,
    BlockRemoved = 1,
    AllBlocksCleared = 2,
}

/// One rendered cell of the tally.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TallyRow {
    pub event: &'static str,
    pub medium: &'static str,
    /// Events applied.
    pub events: u64,
    /// Block hashes those events carried (0 for `all_blocks_cleared`).
    pub blocks: u64,
}

/// Lock-free counters, written by the single pump task and read on scrape.
#[derive(Debug, Default)]
pub struct EventTally {
    events: [[AtomicU64; MEDIUM_LABELS.len()]; EVENT_KINDS.len()],
    blocks: [[AtomicU64; MEDIUM_LABELS.len()]; EVENT_KINDS.len()],
}

impl EventTally {
    pub fn new() -> Self {
        Self::default()
    }

    fn medium_slot(medium: Option<&str>) -> usize {
        match medium {
            None => UNTAGGED,
            Some(m) => MEDIUM_LABELS[..UNTAGGED]
                .iter()
                .position(|known| *known == m)
                .unwrap_or(UNKNOWN),
        }
    }

    /// Book one applied event carrying `blocks` block hashes.
    pub fn record(&self, event: EventKind, medium: Option<&str>, blocks: usize) {
        let (e, m) = (event as usize, Self::medium_slot(medium));
        self.events[e][m].fetch_add(1, Ordering::Relaxed);
        self.blocks[e][m].fetch_add(blocks as u64, Ordering::Relaxed);
    }

    /// Every cell in (event, medium) order, zeros included.
    pub fn snapshot(&self) -> Vec<TallyRow> {
        let mut rows = Vec::with_capacity(EVENT_KINDS.len() * MEDIUM_LABELS.len());
        for (e, event) in EVENT_KINDS.iter().enumerate() {
            for (m, medium) in MEDIUM_LABELS.iter().enumerate() {
                rows.push(TallyRow {
                    event,
                    medium,
                    events: self.events[e][m].load(Ordering::Relaxed),
                    blocks: self.blocks[e][m].load(Ordering::Relaxed),
                });
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell<'a>(rows: &'a [TallyRow], event: &str, medium: &str) -> &'a TallyRow {
        rows.iter()
            .find(|r| r.event == event && r.medium == medium)
            .expect("every (event, medium) cell is rendered")
    }

    #[test]
    fn records_by_kind_and_medium_and_folds_the_rest() {
        let t = EventTally::new();
        t.record(EventKind::BlockStored, Some("GPU"), 3);
        t.record(EventKind::BlockStored, Some("CPU_PINNED"), 3);
        t.record(EventKind::BlockRemoved, Some("GPU"), 1);
        t.record(EventKind::BlockRemoved, None, 2);
        t.record(EventKind::BlockStored, Some("NVLINK_PEER"), 5);
        t.record(EventKind::AllBlocksCleared, None, 0);

        let rows = t.snapshot();
        assert_eq!(rows.len(), EVENT_KINDS.len() * MEDIUM_LABELS.len());
        assert_eq!(cell(&rows, "block_stored", "GPU").blocks, 3);
        assert_eq!(cell(&rows, "block_stored", "CPU_PINNED").events, 1);
        assert_eq!(cell(&rows, "block_removed", "GPU").blocks, 1);
        assert_eq!(cell(&rows, "block_removed", "untagged").blocks, 2);
        assert_eq!(cell(&rows, "block_stored", "unknown").blocks, 5);
        assert_eq!(cell(&rows, "all_blocks_cleared", "untagged").events, 1);
        // Zero cells are present, not omitted.
        assert_eq!(cell(&rows, "block_removed", "CPU_PINNED").events, 0);
    }
}
