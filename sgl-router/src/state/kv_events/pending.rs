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
    /// Max requests one stake may attract before it stops counting; 0 = no cap.
    max_followers: AtomicU64,
    /// Requests routed to a worker because of its pending stake.
    followed: AtomicU64,
    /// Lookups the cap left without any pending candidate.
    capped: AtomicU64,
    /// Expiries are milliseconds since this, so a refresh is one atomic store.
    epoch: Instant,
    state: RwLock<State>,
}

impl Default for PendingPrefixes {
    fn default() -> Self {
        Self {
            ttl_ms: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            max_followers: AtomicU64::new(0),
            followed: AtomicU64::new(0),
            capped: AtomicU64::new(0),
            epoch: Instant::now(),
            state: RwLock::default(),
        }
    }
}

#[derive(Debug)]
struct Owner {
    url: Arc<str>,
    expiry_ms: AtomicU64,
    /// Requests this stake has already attracted (see `max_followers`).
    followers: AtomicU64,
}

impl Owner {
    /// Extend the record to `expiry`. A record that had already expired is a
    /// new stake, not a continuation of the old one, so its followers start
    /// over; the cap counts requests one stake attracted within its TTL.
    fn refresh(&self, now_ms: u64, expiry: u64) {
        if self.expiry_ms.load(Ordering::Relaxed) <= now_ms {
            self.followers.store(0, Ordering::Relaxed);
        }
        self.expiry_ms.fetch_max(expiry, Ordering::Relaxed);
    }
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

    /// Cap on requests one stake may attract; 0 disables the cap.
    pub fn set_max_followers(&self, max: u64) {
        self.max_followers.store(max, Ordering::Relaxed);
    }

    pub fn max_followers(&self) -> u64 {
        self.max_followers.load(Ordering::Relaxed)
    }

    pub fn followed(&self) -> u64 {
        self.followed.load(Ordering::Relaxed)
    }

    pub fn capped(&self) -> u64 {
        self.capped.load(Ordering::Relaxed)
    }

    /// A request was routed to `url` because of its pending stake on
    /// `hashes`: charge every live owner record of that stake one follower.
    /// Once a record reaches the cap, [`Self::depths`] stops reporting the
    /// stake: later requests are routed as if it did not exist, by their
    /// confirmed matches (looked up first, always) or else by the load
    /// policy, instead of piling onto one worker.
    pub fn record_follow(&self, url: &str, hashes: &[i64]) {
        self.record_follow_at(url, hashes, Instant::now());
    }

    fn record_follow_at(&self, url: &str, hashes: &[i64], now: Instant) {
        if !self.is_enabled() || hashes.is_empty() {
            return;
        }
        // Charge the DEEPEST live record of the stake the request matched:
        // the block that is specific to this prompt, and the one
        // [`Self::depths`] reported. The leading blocks (a shared system
        // prompt, say) are owned by every stake on the worker, and charging
        // them would cap every conversation on the worker at once.
        let now_ms = self.millis(now);
        let state = self.state.read();
        let tail = hashes
            .iter()
            .map(|hash| {
                state
                    .owner(*hash, url)
                    .filter(|o| o.expiry_ms.load(Ordering::Relaxed) > now_ms)
            })
            .take_while(Option::is_some)
            .last()
            .flatten();
        if let Some(owner) = tail {
            owner.followers.fetch_add(1, Ordering::Relaxed);
            self.followed.fetch_add(1, Ordering::Relaxed);
        }
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
                        owner.refresh(now_ms, expiry);
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
                Some(owner) => owner.refresh(now_ms, expiry),
                None => {
                    owners.push(Owner {
                        url: Arc::clone(&url),
                        expiry_ms: AtomicU64::new(expiry),
                        followers: AtomicU64::new(0),
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
        let max_followers = self.max_followers();
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
        if max_followers == 0 {
            return done;
        }
        // A stake at its follower cap is invisible. The cap lives on the
        // deepest block the prompt matched (see `record_follow`), so a capped
        // burst prefix hides only itself, not the worker's other stakes.
        let before = done.len();
        done.retain(|(url, depth)| {
            state
                .owner(hashes[depth - 1], url)
                .is_none_or(|o| o.followers.load(Ordering::Relaxed) < max_followers)
        });
        // Counted only when the cap left the lookup with no pending candidate:
        // that is the request the cap diverted to the confirmed tree or the
        // load policy. A capped stake alongside a live one changes nothing.
        if before > 0 && done.is_empty() {
            self.capped.fetch_add(1, Ordering::Relaxed);
        }
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
    fn follower_cap_hides_a_stake_and_counts_it() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.set_max_followers(2);
        pending.record("a", &[1, 2, 3]);
        assert_eq!(sorted(pending.depths(&[1, 2, 3])), [("a".into(), 3)]);
        pending.record_follow("a", &[1, 2, 3]);
        assert_eq!(pending.followed(), 1);
        assert_eq!(sorted(pending.depths(&[1, 2, 3])), [("a".into(), 3)]);
        pending.record_follow("a", &[1, 2, 3, 4]);
        assert_eq!(pending.followed(), 2);
        // At the cap: the stake is invisible, and the lookup that lost its
        // only candidate is counted.
        assert!(pending.depths(&[1, 2, 3]).is_empty());
        assert_eq!(pending.capped(), 1);
        // Another worker's stake on the same prefix is unaffected, and a
        // lookup it still serves is not counted as capped.
        pending.record("b", &[1, 2]);
        assert_eq!(sorted(pending.depths(&[1, 2, 3])), [("b".into(), 2)]);
        assert_eq!(pending.capped(), 1);
    }

    #[test]
    fn cap_is_charged_on_the_deepest_block_not_the_shared_prefix() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.set_max_followers(2);
        // Two conversations on "a" share blocks 1,2 (a system prompt).
        pending.record("a", &[1, 2, 3]);
        pending.record("a", &[1, 2, 9]);
        // A burst on the first conversation's prefix reaches the cap...
        pending.record_follow("a", &[1, 2, 3]);
        pending.record_follow("a", &[1, 2, 3, 4]);
        assert!(pending.depths(&[1, 2, 3]).is_empty());
        // ...but the other conversation on the same worker is untouched,
        // and so is a prompt that only shares the system prompt.
        assert_eq!(sorted(pending.depths(&[1, 2, 9])), [("a".into(), 3)]);
        assert_eq!(sorted(pending.depths(&[1, 2, 7])), [("a".into(), 2)]);
        // A longer turn of the capped conversation matches past the capped
        // block once its own tail is staked, so the conversation itself
        // keeps following its stake.
        pending.record("a", &[1, 2, 3, 4]);
        assert_eq!(sorted(pending.depths(&[1, 2, 3, 4])), [("a".into(), 4)]);
    }

    #[test]
    fn no_cap_by_default_and_follow_of_unknown_stake_is_free() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_secs(60));
        pending.record("a", &[1, 2]);
        for _ in 0..100 {
            pending.record_follow("a", &[1, 2]);
        }
        assert_eq!(sorted(pending.depths(&[1, 2])), [("a".into(), 2)]);
        pending.record_follow("zzz", &[1, 2]);
        assert_eq!(pending.followed(), 100);
        assert_eq!(pending.capped(), 0);
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
    fn a_follow_charges_the_live_tail_and_a_revived_stake_starts_fresh() {
        let pending = PendingPrefixes::default();
        pending.enable(Duration::from_millis(50));
        pending.set_max_followers(1);
        let t0 = pending.epoch;
        let at = |ms| t0 + Duration::from_millis(ms);
        // Blocks 1..3 staked at 0; a shorter turn refreshes only 1..2 at 40.
        pending.record_at("a", &[1, 2, 3], at(0));
        pending.record_at("a", &[1, 2], at(40));
        // At 60 the lookup sees depth 2 (block 3 expired, not yet trimmed),
        // so the follow is charged on block 2, the record the lookup reported.
        assert_eq!(
            sorted(pending.depths_at(&[1, 2, 3], at(60))),
            [("a".into(), 2)]
        );
        pending.record_follow_at("a", &[1, 2, 3], at(60));
        assert!(pending.depths_at(&[1, 2, 3], at(60)).is_empty(), "capped");
        assert_eq!(
            pending
                .state
                .read()
                .owner(3, "a")
                .unwrap()
                .followers
                .load(Ordering::Relaxed),
            0
        );
        // Once the whole stake has expired, re-staking it is a new stake: the
        // cap starts over.
        assert!(pending.depths_at(&[1, 2], at(100)).is_empty());
        pending.record_at("a", &[1, 2, 3], at(100));
        assert_eq!(
            sorted(pending.depths_at(&[1, 2, 3], at(100))),
            [("a".into(), 3)]
        );
        pending.record_follow_at("a", &[1, 2], at(100));
        assert!(
            pending.depths_at(&[1, 2], at(100)).is_empty(),
            "capped on block 2"
        );
        assert_eq!(
            sorted(pending.depths_at(&[1, 2, 3], at(100))),
            [("a".into(), 3)]
        );
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
