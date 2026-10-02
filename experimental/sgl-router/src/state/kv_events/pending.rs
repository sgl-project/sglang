// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Prefixes the router just routed that no KV event confirms yet.
//!
//! A burst sharing a cold prefix (parallel sampling, RL rollouts, agent
//! fan-out) finds no cached owner anywhere and would scatter, prefilling the
//! same prefix on every worker. Routing records the request's block hashes
//! against its worker for a short TTL, and lookups take the per-worker max of
//! this and the event-driven tree. Entries leave only by expiry.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

/// Bound on remembered (block, expiry) records; arbitrary.
const MAX_RECORDS: usize = 1 << 20;

/// Off (TTL 0) until [`PendingPrefixes::enable`].
#[derive(Debug, Default)]
pub struct PendingPrefixes {
    ttl_ms: AtomicU64,
    hits: AtomicU64,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    /// Workers routed through each block, with their expiry. Block hashes
    /// chain, so holding a block implies holding its whole prefix.
    by_hash: FxHashMap<i64, Vec<(Arc<str>, Instant)>>,
    /// Records in insertion order, which is expiry order under one TTL.
    fifo: VecDeque<(Instant, i64)>,
}

impl State {
    /// Drop owners of `hash` whose entry expires at or before `deadline`.
    fn forget(&mut self, hash: i64, deadline: Instant) {
        if let Some(owners) = self.by_hash.get_mut(&hash) {
            owners.retain(|(_, expiry)| *expiry > deadline);
            if owners.is_empty() {
                self.by_hash.remove(&hash);
            }
        }
    }
}

impl PendingPrefixes {
    pub fn enable(&self, ttl: Duration) {
        self.ttl_ms.store(ttl.as_millis() as u64, Ordering::Relaxed);
    }

    fn ttl(&self) -> Option<Duration> {
        let ms = self.ttl_ms.load(Ordering::Relaxed);
        (ms > 0).then(|| Duration::from_millis(ms))
    }

    /// Lookups where a pending prefix beat every confirmed one.
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
        let Some(ttl) = self.ttl() else {
            return;
        };
        let expiry = now + ttl;
        let url: Arc<str> = url.into();
        let mut state = self.state.lock();
        while let Some(&(deadline, hash)) = state.fifo.front() {
            if deadline > now && state.fifo.len() < MAX_RECORDS {
                break;
            }
            state.fifo.pop_front();
            state.forget(hash, deadline);
        }
        for &hash in hashes {
            let owners = state.by_hash.entry(hash).or_default();
            match owners.iter_mut().find(|(owner, _)| *owner == url) {
                Some(owner) => owner.1 = expiry,
                None => owners.push((Arc::clone(&url), expiry)),
            }
            state.fifo.push_back((expiry, hash));
        }
    }

    /// Contiguous pending depth of `hashes` per worker URL.
    pub fn depths(&self, hashes: &[i64]) -> Vec<(Arc<str>, usize)> {
        self.depths_at(hashes, Instant::now())
    }

    fn depths_at(&self, hashes: &[i64], now: Instant) -> Vec<(Arc<str>, usize)> {
        if self.ttl().is_none() {
            return Vec::new();
        }
        let state = self.state.lock();
        let mut walking: Vec<(Arc<str>, usize)> = Vec::new();
        let mut done = Vec::new();
        for (i, hash) in hashes.iter().enumerate() {
            let owners = state.by_hash.get(hash).map_or(&[][..], Vec::as_slice);
            let live = |url: &Arc<str>| owners.iter().any(|(o, e)| o == url && *e > now);
            if i == 0 {
                walking = owners
                    .iter()
                    .filter(|(_, expiry)| *expiry > now)
                    .map(|(url, _)| (Arc::clone(url), 1))
                    .collect();
            } else {
                let (held, stopped): (Vec<_>, Vec<_>) =
                    walking.into_iter().partition(|(url, _)| live(url));
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
        let t0 = Instant::now();
        let at = |ms| t0 + Duration::from_millis(ms);
        pending.record_at("a", &[1], at(0));
        pending.record_at("a", &[1], at(30));
        assert_eq!(sorted(pending.depths_at(&[1], at(60))), [("a".into(), 1)]);
        assert!(pending.depths_at(&[1], at(90)).is_empty());
        pending.record_at("b", &[2], at(90));
        assert!(!pending.state.lock().by_hash.contains_key(&1));
    }
}
