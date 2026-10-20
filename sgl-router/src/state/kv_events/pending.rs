// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefixes the router just routed that no KV event confirms yet.
//!
//! A burst sharing a cold prefix (parallel sampling, RL rollouts, agent
//! fan-out) finds no cached owner anywhere and would scatter, prefilling the
//! same prefix on every worker. Routing records the request's block hashes
//! against its worker for a short TTL, and lookups take the per-worker max of
//! this and the event-driven tree. Entries leave by expiry, by the record cap,
//! or when the worker's cache is cleared.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

/// Bound on remembered (block, worker) records; arbitrary.
const MAX_RECORDS: usize = 1 << 20;

/// Off (TTL 0) until [`PendingPrefixes::enable`].
#[derive(Debug)]
pub struct PendingPrefixes {
    ttl_ms: AtomicU64,
    hits: AtomicU64,
    /// Expiries are milliseconds since this, so a refresh is one atomic store.
    epoch: Instant,
    state: RwLock<State>,
}

impl Default for PendingPrefixes {
    fn default() -> Self {
        Self {
            ttl_ms: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            epoch: Instant::now(),
            state: RwLock::default(),
        }
    }
}

#[derive(Debug)]
struct Owner {
    url: Arc<str>,
    expiry_ms: AtomicU64,
}

#[derive(Debug, Default)]
struct State {
    /// Workers routed through each block. Block hashes chain, so holding a
    /// block implies holding its whole prefix.
    by_hash: FxHashMap<i64, Vec<Owner>>,
    /// One entry per (block, worker) record, holding the expiry it had when
    /// queued. A refresh moves the expiry but not the entry; trimming requeues
    /// it then, so the queue never holds duplicates.
    fifo: VecDeque<(u64, i64, Arc<str>)>,
}

impl State {
    fn owner(&self, hash: i64, url: &str) -> Option<&Owner> {
        self.by_hash.get(&hash)?.iter().find(|o| &*o.url == url)
    }

    /// Drop expired records, then the oldest until under the cap.
    fn trim(&mut self, now_ms: u64) {
        while let Some(&(deadline, ..)) = self.fifo.front() {
            let over = self.fifo.len() > MAX_RECORDS;
            if deadline > now_ms && !over {
                break;
            }
            let Some((_, hash, url)) = self.fifo.pop_front() else {
                break;
            };
            let Some(owners) = self.by_hash.get_mut(&hash) else {
                continue;
            };
            let Some(i) = owners.iter().position(|o| o.url == url) else {
                continue;
            };
            let expiry = owners[i].expiry_ms.load(Ordering::Relaxed);
            if expiry > now_ms && !over {
                // Refreshed since it was queued.
                self.fifo.push_back((expiry, hash, url));
                continue;
            }
            owners.swap_remove(i);
            if owners.is_empty() {
                self.by_hash.remove(&hash);
            }
        }
    }
}

impl PendingPrefixes {
    pub fn enable(&self, ttl: Duration) {
        let ms = u64::try_from(ttl.as_millis()).unwrap_or(u64::MAX);
        self.ttl_ms.store(ms, Ordering::Relaxed);
    }

    pub fn is_enabled(&self) -> bool {
        self.ttl_ms.load(Ordering::Relaxed) > 0
    }

    fn millis(&self, at: Instant) -> u64 {
        u64::try_from(at.saturating_duration_since(self.epoch).as_millis()).unwrap_or(u64::MAX)
    }

    /// Lookups where a pending prefix matched deeper than every confirmed one.
    pub fn hits(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    pub(crate) fn record_hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    /// Credit `url` with the prefix `hashes` until the TTL runs out.
    pub fn record(&self, url: &str, hashes: &[i64]) {
        self.record_at(url, hashes, Instant::now());
    }

    fn record_at(&self, url: &str, hashes: &[i64], now: Instant) {
        let ttl_ms = self.ttl_ms.load(Ordering::Relaxed);
        if ttl_ms == 0 {
            return;
        }
        let now_ms = self.millis(now);
        let expiry = now_ms.saturating_add(ttl_ms);
        // Re-routing a known prompt only refreshes, which the read lock allows.
        let known = {
            let state = self.state.read();
            hashes
                .iter()
                .take_while(|&&hash| match state.owner(hash, url) {
                    Some(owner) => {
                        owner.expiry_ms.fetch_max(expiry, Ordering::Relaxed);
                        true
                    }
                    None => false,
                })
                .count()
        };
        if known == hashes.len() {
            return;
        }
        let url: Arc<str> = url.into();
        let mut state = self.state.write();
        // Queue the tail first, so the cap trims a prompt from its end and
        // what survives is still a walkable prefix.
        for &hash in hashes[known..].iter().rev() {
            let owners = state.by_hash.entry(hash).or_default();
            match owners.iter().find(|o| o.url == url) {
                Some(owner) => {
                    owner.expiry_ms.fetch_max(expiry, Ordering::Relaxed);
                }
                None => {
                    owners.push(Owner {
                        url: Arc::clone(&url),
                        expiry_ms: AtomicU64::new(expiry),
                    });
                    state.fifo.push_back((expiry, hash, Arc::clone(&url)));
                }
            }
        }
        state.trim(now_ms);
    }

    /// Drop everything credited to `url`, whose cache is gone.
    pub fn forget_worker(&self, url: &str) {
        let mut state = self.state.write();
        if state.fifo.is_empty() {
            return;
        }
        state.by_hash.retain(|_, owners| {
            owners.retain(|o| &*o.url != url);
            !owners.is_empty()
        });
        state.fifo.retain(|(_, _, owner)| &**owner != url);
    }

    /// Contiguous pending depth of `hashes` per worker URL.
    pub fn depths(&self, hashes: &[i64]) -> Vec<(Arc<str>, usize)> {
        self.depths_at(hashes, Instant::now())
    }

    fn depths_at(&self, hashes: &[i64], now: Instant) -> Vec<(Arc<str>, usize)> {
        if !self.is_enabled() {
            return Vec::new();
        }
        let now_ms = self.millis(now);
        let state = self.state.read();
        let mut walking: Vec<(Arc<str>, usize)> = Vec::new();
        let mut done = Vec::new();
        for (i, hash) in hashes.iter().enumerate() {
            let owners = state.by_hash.get(hash).map_or(&[][..], Vec::as_slice);
            let live = |o: &Owner| o.expiry_ms.load(Ordering::Relaxed) > now_ms;
            if i == 0 {
                walking = owners
                    .iter()
                    .filter(|o| live(o))
                    .map(|o| (Arc::clone(&o.url), 1))
                    .collect();
            } else {
                let (held, stopped): (Vec<_>, Vec<_>) = walking
                    .into_iter()
                    .partition(|(url, _)| owners.iter().any(|o| o.url == *url && live(o)));
                done.extend(stopped);
                walking = held.into_iter().map(|(url, d)| (url, d + 1)).collect();
            }
            if walking.is_empty() {
                break;
            }
        }
        done.extend(walking);
        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted(mut depths: Vec<(Arc<str>, usize)>) -> Vec<(String, usize)> {
        depths.sort();
        depths
            .into_iter()
            .map(|(u, d)| (u.to_string(), d))
            .collect()
    }

    #[test]
    fn off_until_enabled() {
        let pending = PendingPrefixes::default();
        pending.record("a", &[1, 2]);
        assert!(pending.depths(&[1, 2]).is_empty());
    }

    #[test]
    fn depth_is_the_contiguous_prefix_per_worker() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.record("a", &[1, 2, 3]);
        pending.record("b", &[1, 9]);
        assert_eq!(
            sorted(pending.depths(&[1, 2, 3, 4])),
            [("a".into(), 3), ("b".into(), 1)]
        );
        assert!(pending.depths(&[7, 1]).is_empty());
    }

    #[test]
    fn entries_expire_and_rerouting_refreshes() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_millis(50));
        let t0 = pending.epoch;
        let at = |ms| t0 + Duration::from_millis(ms);
        pending.record_at("a", &[1], at(0));
        pending.record_at("a", &[1], at(30));
        assert_eq!(sorted(pending.depths_at(&[1], at(60))), [("a".into(), 1)]);
        assert!(pending.depths_at(&[1], at(90)).is_empty());
        pending.record_at("b", &[2], at(90));
        assert!(!pending.state.read().by_hash.contains_key(&1));
    }

    #[test]
    fn refreshes_do_not_queue_duplicates() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_millis(50));
        let t0 = pending.epoch;
        let at = |ms| t0 + Duration::from_millis(ms);
        for ms in 0..100 {
            pending.record_at("a", &[1, 2, 3], at(ms));
        }
        // Each refreshed record is requeued once its first expiry passes.
        pending.record_at("b", &[9], at(100));
        let state = pending.state.read();
        assert_eq!(state.fifo.len(), 4);
        drop(state);
        assert_eq!(
            sorted(pending.depths_at(&[1, 2, 3], at(140))),
            [("a".into(), 3)]
        );
    }

    #[test]
    fn huge_ttl_saturates_instead_of_overflowing() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::MAX);
        pending.record("a", &[1]);
        assert_eq!(sorted(pending.depths(&[1])), [("a".into(), 1)]);
    }

    #[test]
    fn forget_worker_drops_only_that_worker() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.record("a", &[1, 2]);
        pending.record("b", &[1]);
        pending.forget_worker("a");
        assert_eq!(sorted(pending.depths(&[1, 2])), [("b".into(), 1)]);
        assert_eq!(pending.state.read().fifo.len(), 1);
    }
}
