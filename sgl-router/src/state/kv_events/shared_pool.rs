// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! A HiCache L3 pool that every worker reads from, tracked from the pool's
//! own event stream: Mooncake Store's master publishes object `stored` /
//! `removed` / `cleared` events (Mooncake `docs/.../publisher-design.md`),
//! and SGLang's Mooncake HiCache backend names each page's objects after the
//! page hash (`mooncake_store.py`: `[<extra_backend_tag>_][<model>_]<page
//! hash hex>_<rank suffix>_{k|v}`).
//!
//! A page in the pool can be loaded by any worker attached to it, not only
//! by the one that wrote it, so the routing path credits the pool's prefix
//! depth to every member worker (`depth`, `is_member`). The tree's own
//! per-worker holdings (device, host, and the engine's tier tags) are left
//! as they are.
//!
//! The master's stream has no replay. A transport-sequence gap, or a
//! regression (a master restart), drops every held page: the pool then
//! under-credits until the pages are stored again, and never routes on a
//! page it may have lost.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use serde::de::{self, IgnoredAny, SeqAccess, Visitor};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use zeromq::{Socket, SocketRecv, SubSocket, ZmqMessage};

/// How the router attaches to one shared pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedPoolConfig {
    /// ZMQ endpoint the pool's master publishes its KV events on, e.g.
    /// `tcp://mooncake-master:19997` (`--kv_events_bind_endpoint`).
    pub endpoint: String,
    /// Only object keys that start with `<key_prefix>_` count: the SGLang
    /// key prefix of the served model (`<extra_backend_tag>_<model>`, the
    /// model path with `/` replaced by `-`). `None` takes every key.
    pub key_prefix: Option<String>,
    /// Only objects of this Mooncake tenant count; `None` takes every tenant.
    pub tenant: Option<String>,
    /// Component objects (rank suffix x K/V) a page needs before it counts
    /// as held, e.g. 2 for one rank's K and V. 1 counts a page once any of
    /// its objects is stored.
    pub min_components: u32,
    /// Worker URLs attached to the pool; `None` attaches every worker.
    pub members: Option<Vec<String>>,
}

/// One decoded pool event.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolEvent {
    Stored {
        object_key: String,
        medium: Option<String>,
        tenant: Option<String>,
    },
    Removed {
        object_key: String,
        medium: Option<String>,
        tenant: Option<String>,
    },
    Cleared {
        tenant: Option<String>,
    },
}

#[derive(Debug, Default)]
struct PoolState {
    config: Option<SharedPoolConfig>,
    members: Option<HashSet<String>>,
    /// Workers the KV index follows: the members when none are configured.
    known: HashSet<String>,
    /// Page hash -> its component objects held, as `<suffix>@<medium>`.
    pages: HashMap<i64, HashSet<Box<str>>>,
}

/// The shared pool's view, owned by [`super::HashTree`]. Off until
/// [`SharedPool::enable`].
#[derive(Default)]
pub struct SharedPool {
    state: RwLock<PoolState>,
    resets: AtomicU64,
    applied: AtomicU64,
}

impl fmt::Debug for SharedPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SharedPool")
            .field("enabled", &self.is_enabled())
            .field("pages", &self.pages())
            .finish()
    }
}

impl SharedPool {
    /// Attach the pool. Pages already held are dropped.
    pub fn enable(&self, config: SharedPoolConfig) {
        let mut state = self.write();
        state.members = config
            .members
            .as_ref()
            .map(|urls| urls.iter().cloned().collect());
        state.config = Some(config);
        state.pages.clear();
    }

    pub fn is_enabled(&self) -> bool {
        self.read().config.is_some()
    }

    /// Whether the worker at `url` reads from the pool.
    pub fn is_member(&self, url: &str) -> bool {
        let state = self.read();
        state.config.is_some() && state.members.as_ref().is_none_or(|m| m.contains(url))
    }

    /// Note a worker the KV index follows (`KvEventIndex::add_worker`).
    pub fn track_worker(&self, url: &str) {
        self.write().known.insert(url.to_string());
    }

    pub fn forget_worker(&self, url: &str) {
        self.write().known.remove(url);
    }

    /// The URLs a pool hit is credited to: the configured members, or else
    /// every worker the KV index follows. Empty when the pool is off.
    pub fn member_urls(&self) -> Vec<String> {
        let state = self.read();
        if state.config.is_none() {
            return Vec::new();
        }
        match &state.members {
            Some(members) => members.iter().cloned().collect(),
            None => state.known.iter().cloned().collect(),
        }
    }

    /// Leading blocks of `hashes` (a request's page-hash chain) the pool
    /// holds; 0 when the pool is off.
    pub fn depth(&self, hashes: &[i64]) -> usize {
        let state = self.read();
        let Some(config) = &state.config else {
            return 0;
        };
        let need = config.min_components.max(1) as usize;
        hashes
            .iter()
            .take_while(|hash| state.pages.get(hash).is_some_and(|c| c.len() >= need))
            .count()
    }

    /// Pages that count as held.
    pub fn pages(&self) -> usize {
        let state = self.read();
        let need = state
            .config
            .as_ref()
            .map_or(1, |c| c.min_components.max(1) as usize);
        state.pages.values().filter(|c| c.len() >= need).count()
    }

    /// Times the view was dropped on a stream gap or restart.
    pub fn resets(&self) -> u64 {
        self.resets.load(Ordering::Relaxed)
    }

    /// Events applied since the pool was enabled.
    pub fn applied(&self) -> u64 {
        self.applied.load(Ordering::Relaxed)
    }

    /// Drop every held page: the stream lost events, so any of them may be gone.
    pub fn reset(&self) {
        self.write().pages.clear();
        self.resets.fetch_add(1, Ordering::Relaxed);
    }

    pub fn apply(&self, events: &[PoolEvent]) {
        let mut state = self.write();
        let Some(config) = state.config.clone() else {
            return;
        };
        for event in events {
            match event {
                PoolEvent::Stored {
                    object_key,
                    medium,
                    tenant,
                } => {
                    if !tenant_matches(&config, tenant.as_deref()) {
                        continue;
                    }
                    if let Some((hash, component)) = page_of_key(object_key, &config, medium) {
                        state.pages.entry(hash).or_default().insert(component);
                    }
                }
                PoolEvent::Removed {
                    object_key,
                    medium,
                    tenant,
                } => {
                    if !tenant_matches(&config, tenant.as_deref()) {
                        continue;
                    }
                    if let Some((hash, component)) = page_of_key(object_key, &config, medium) {
                        if let Some(components) = state.pages.get_mut(&hash) {
                            components.remove(&component);
                            if components.is_empty() {
                                state.pages.remove(&hash);
                            }
                        }
                    }
                }
                // A tenant clear names no objects; with a tenant filter only
                // that tenant's clear drops the view, otherwise any does.
                PoolEvent::Cleared { tenant } => {
                    if tenant_matches(&config, tenant.as_deref()) {
                        state.pages.clear();
                    }
                }
            }
        }
        self.applied
            .fetch_add(events.len() as u64, Ordering::Relaxed);
    }

    fn read(&self) -> std::sync::RwLockReadGuard<'_, PoolState> {
        self.state.read().unwrap_or_else(|e| e.into_inner())
    }

    fn write(&self) -> std::sync::RwLockWriteGuard<'_, PoolState> {
        self.state.write().unwrap_or_else(|e| e.into_inner())
    }
}

fn tenant_matches(config: &SharedPoolConfig, tenant: Option<&str>) -> bool {
    config
        .tenant
        .as_deref()
        .is_none_or(|want| tenant == Some(want))
}

/// The router block hash (top 64 bits, big-endian, as [`super::sha256_to_i64`])
/// of a SGLang HiCache Mooncake object key and the component it names, or
/// `None` for a key outside the configured prefix or with no page hash.
fn page_of_key(
    key: &str,
    config: &SharedPoolConfig,
    medium: &Option<String>,
) -> Option<(i64, Box<str>)> {
    let rest = match &config.key_prefix {
        Some(prefix) => key.strip_prefix(prefix.as_str())?.strip_prefix('_')?,
        None => key,
    };
    let (hash, suffix) = split_page_hash(rest)?;
    let component = format!("{suffix}@{}", medium.as_deref().unwrap_or(""));
    Some((hash, component.into_boxed_str()))
}

/// The first `_`-separated token of 64 hex digits (a SHA-256 page hash) in
/// `key`, as the router's block hash, and what follows it.
pub fn split_page_hash(key: &str) -> Option<(i64, &str)> {
    let mut offset = 0;
    for token in key.split('_') {
        let end = offset + token.len();
        if token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit()) {
            let top = u64::from_str_radix(&token[..16], 16).ok()?;
            let suffix = key.get(end..).unwrap_or("").trim_start_matches('_');
            return Some((top as i64, suffix));
        }
        offset = end + 1;
    }
    None
}

#[derive(Deserialize)]
struct RawEvent {
    #[serde(default)]
    event_type: Option<String>,
    /// The legacy alias (`BlockStored` / `BlockRemoved` / `AllBlocksCleared`).
    #[serde(default, rename = "type")]
    legacy_type: Option<String>,
    #[serde(default)]
    object_key: Option<String>,
    #[serde(default)]
    medium: Option<String>,
    #[serde(default)]
    tenant_id: Option<String>,
}

/// `[timestamp_ms, [event maps], dp_rank]`: the event maps, the rest skipped.
struct RawBatch(Vec<RawEvent>);

impl<'de> Deserialize<'de> for RawBatch {
    fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct BatchVisitor;
        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = RawBatch;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("[timestamp, [events], dp_rank]")
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RawBatch, A::Error> {
                seq.next_element::<IgnoredAny>()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let events = seq
                    .next_element::<Vec<RawEvent>>()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(RawBatch(events))
            }
        }
        deserializer.deserialize_seq(BatchVisitor)
    }
}

/// Decode one master payload. Events of an unknown type, and `stored` /
/// `removed` events without an object key, are skipped.
pub fn decode_pool_batch(payload: &[u8]) -> Result<Vec<PoolEvent>, rmp_serde::decode::Error> {
    let RawBatch(raw) = rmp_serde::from_slice(payload)?;
    Ok(raw
        .into_iter()
        .filter_map(|e| {
            let kind = e.event_type.as_deref().or(e.legacy_type.as_deref())?;
            match kind {
                "stored" | "BlockStored" => Some(PoolEvent::Stored {
                    object_key: e.object_key?,
                    medium: e.medium,
                    tenant: e.tenant_id,
                }),
                "removed" | "BlockRemoved" => Some(PoolEvent::Removed {
                    object_key: e.object_key?,
                    medium: e.medium,
                    tenant: e.tenant_id,
                }),
                "cleared" | "AllBlocksCleared" => Some(PoolEvent::Cleared {
                    tenant: e.tenant_id,
                }),
                _ => None,
            }
        })
        .collect())
}

const CONNECT_BACKOFF_BASE: Duration = Duration::from_millis(200);
const CONNECT_BACKOFF_CAP: Duration = Duration::from_secs(10);
const RECV_ERROR_CEILING: u32 = 32;

/// Follow the pool's event stream into `pool` until `cancel`. Messages are
/// `[topic, seq (u64 big-endian), payload]`.
pub async fn run_pool_subscriber(
    endpoint: String,
    pool: Arc<super::HashTree>,
    cancel: CancellationToken,
) {
    info!(%endpoint, "kv shared pool: following the pool's KV events");
    let Some(mut sub) = connect(&endpoint, &cancel).await else {
        return;
    };
    let mut last_seq: Option<u64> = None;
    let mut errors_in_a_row = 0u32;
    loop {
        let msg = tokio::select! {
            biased;
            _ = cancel.cancelled() => return,
            res = sub.recv() => res,
        };
        match msg {
            Ok(msg) => {
                errors_in_a_row = 0;
                handle_message(pool.shared_pool(), &msg, &mut last_seq);
            }
            Err(error) => {
                errors_in_a_row += 1;
                if errors_in_a_row < RECV_ERROR_CEILING {
                    warn!(%endpoint, %error, "kv shared pool: recv error; continuing");
                    tokio::task::yield_now().await;
                    continue;
                }
                warn!(%endpoint, %error, "kv shared pool: socket keeps failing; reconnecting");
                pool.shared_pool().reset();
                last_seq = None;
                let Some(fresh) = connect(&endpoint, &cancel).await else {
                    return;
                };
                sub = fresh;
                errors_in_a_row = 0;
            }
        }
    }
}

/// Apply one message, dropping the view first when the sequence skips or
/// goes back.
pub fn handle_message(pool: &SharedPool, msg: &ZmqMessage, last_seq: &mut Option<u64>) {
    let frames: Vec<&[u8]> = msg.iter().map(|f| f.as_ref()).collect();
    let (seq, payload) = match frames.as_slice() {
        [_topic, seq, payload] if seq.len() == 8 => (
            u64::from_be_bytes((*seq).try_into().expect("8 bytes")),
            *payload,
        ),
        _ => {
            debug!(
                frames = frames.len(),
                "kv shared pool: unexpected message shape; skipped"
            );
            return;
        }
    };
    match *last_seq {
        Some(last) if seq != last.wrapping_add(1) => {
            warn!(
                last,
                seq, "kv shared pool: event stream skipped or restarted; dropping the pool view"
            );
            pool.reset();
        }
        _ => {}
    }
    *last_seq = Some(seq);
    match decode_pool_batch(payload) {
        Ok(events) => pool.apply(&events),
        Err(error) => {
            warn!(%error, "kv shared pool: undecodable batch; dropping the pool view");
            pool.reset();
        }
    }
}

async fn connect(endpoint: &str, cancel: &CancellationToken) -> Option<SubSocket> {
    let mut delay = CONNECT_BACKOFF_BASE;
    loop {
        let mut sub = SubSocket::new();
        let connected = tokio::select! {
            _ = cancel.cancelled() => return None,
            res = sub.connect(endpoint) => res,
        };
        match connected {
            Ok(()) => match sub.subscribe("").await {
                Ok(()) => return Some(sub),
                Err(error) => {
                    warn!(%endpoint, %error, "kv shared pool: subscribe failed; retrying")
                }
            },
            Err(error) => warn!(%endpoint, %error, "kv shared pool: connect failed; retrying"),
        }
        tokio::select! {
            _ = cancel.cancelled() => return None,
            _ = tokio::time::sleep(delay) => {}
        }
        delay = (delay * 2).min(CONNECT_BACKOFF_CAP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    const H1: &str = "00000000000000010000000000000000000000000000000000000000000000aa";
    const H2: &str = "00000000000000020000000000000000000000000000000000000000000000bb";
    const H3: &str = "00000000000000030000000000000000000000000000000000000000000000cc";

    fn config(prefix: Option<&str>, min_components: u32) -> SharedPoolConfig {
        SharedPoolConfig {
            endpoint: "tcp://127.0.0.1:1".into(),
            key_prefix: prefix.map(Into::into),
            tenant: None,
            min_components,
            members: None,
        }
    }

    fn stored(key: String) -> PoolEvent {
        PoolEvent::Stored {
            object_key: key,
            medium: Some("cpu".into()),
            tenant: None,
        }
    }

    fn removed(key: String) -> PoolEvent {
        PoolEvent::Removed {
            object_key: key,
            medium: Some("cpu".into()),
            tenant: None,
        }
    }

    #[test]
    fn page_hash_is_the_top_64_bits_of_the_hex_token() {
        let key = format!("tag_Qwen-Qwen3-8B_{H1}_0_k");
        let (hash, suffix) = split_page_hash(&key).unwrap();
        assert_eq!(hash, 1);
        assert_eq!(suffix, "0_k");
        let neg = "f".repeat(64);
        assert_eq!(split_page_hash(&format!("{neg}_1_v")).unwrap().0, -1);
        assert!(split_page_hash("Qwen-Qwen3-8B_deadbeef_0_k").is_none());
        // A model name token is never 64 hex digits, so the page hash is found after it.
        let model_with_underscores = format!("my_model_v2_{H2}_3_1_v");
        assert_eq!(split_page_hash(&model_with_underscores), Some((2, "3_1_v")));
    }

    #[test]
    fn depth_counts_the_leading_held_pages() {
        let pool = SharedPool::default();
        pool.enable(config(Some("m"), 1));
        pool.apply(&[stored(format!("m_{H1}_0_k")), stored(format!("m_{H3}_0_k"))]);
        assert_eq!(
            pool.depth(&[1, 2, 3]),
            1,
            "page 2 is missing, so page 3 is unreachable"
        );
        pool.apply(&[stored(format!("m_{H2}_0_k"))]);
        assert_eq!(pool.depth(&[1, 2, 3]), 3);
        assert_eq!(pool.depth(&[2, 3]), 2);
        assert_eq!(pool.pages(), 3);
    }

    #[test]
    fn other_prefixes_and_tenants_do_not_count() {
        let pool = SharedPool::default();
        pool.enable(SharedPoolConfig {
            tenant: Some("t1".into()),
            ..config(Some("tag_Qwen-Qwen3-8B"), 1)
        });
        pool.apply(&[
            stored(format!("tag_Other-Model_{H1}_0_k")),
            PoolEvent::Stored {
                object_key: format!("tag_Qwen-Qwen3-8B_{H1}_0_k"),
                medium: None,
                tenant: Some("t2".into()),
            },
        ]);
        assert_eq!(pool.depth(&[1]), 0);
        pool.apply(&[PoolEvent::Stored {
            object_key: format!("tag_Qwen-Qwen3-8B_{H1}_0_k"),
            medium: None,
            tenant: Some("t1".into()),
        }]);
        assert_eq!(pool.depth(&[1]), 1);
    }

    #[test]
    fn a_page_needs_its_components_and_leaves_with_the_last() {
        let pool = SharedPool::default();
        pool.enable(config(None, 2));
        pool.apply(&[stored(format!("{H1}_0_k"))]);
        assert_eq!(pool.depth(&[1]), 0, "K alone of a K/V page");
        pool.apply(&[stored(format!("{H1}_0_v"))]);
        assert_eq!(pool.depth(&[1]), 1);
        // A duplicate store and a removal on another medium change nothing.
        pool.apply(&[
            stored(format!("{H1}_0_v")),
            PoolEvent::Removed {
                object_key: format!("{H1}_0_v"),
                medium: Some("disk".into()),
                tenant: None,
            },
        ]);
        assert_eq!(pool.depth(&[1]), 1);
        pool.apply(&[removed(format!("{H1}_0_v"))]);
        assert_eq!(pool.depth(&[1]), 0);
        pool.apply(&[removed(format!("{H1}_0_k")), removed(format!("{H1}_0_k"))]);
        assert_eq!(pool.pages(), 0);
    }

    #[test]
    fn clear_reset_and_membership() {
        let pool = SharedPool::default();
        assert!(!pool.is_member("http://a"), "off until enabled");
        assert_eq!(pool.depth(&[1]), 0);
        pool.enable(SharedPoolConfig {
            members: Some(vec!["http://a".into()]),
            ..config(None, 1)
        });
        assert!(pool.is_member("http://a"));
        assert!(!pool.is_member("http://b"));
        pool.track_worker("http://b");
        assert_eq!(
            pool.member_urls(),
            vec!["http://a".to_string()],
            "configured members win"
        );
        pool.apply(&[stored(format!("{H1}_0_k"))]);
        pool.apply(&[PoolEvent::Cleared { tenant: None }]);
        assert_eq!(pool.depth(&[1]), 0);
        pool.apply(&[stored(format!("{H1}_0_k"))]);
        pool.reset();
        assert_eq!((pool.depth(&[1]), pool.resets()), (0, 1));
    }

    #[test]
    fn unconfigured_members_are_the_followed_workers() {
        let pool = SharedPool::default();
        pool.track_worker("http://a");
        assert!(pool.member_urls().is_empty(), "off until enabled");
        pool.enable(config(None, 1));
        pool.track_worker("http://b");
        let mut urls = pool.member_urls();
        urls.sort();
        assert_eq!(urls, vec!["http://a".to_string(), "http://b".to_string()]);
        pool.forget_worker("http://a");
        assert_eq!(pool.member_urls(), vec!["http://b".to_string()]);
    }

    #[derive(Serialize)]
    struct WireEvent<'a> {
        event_type: &'a str,
        #[serde(rename = "type")]
        legacy: &'a str,
        #[serde(skip_serializing_if = "Option::is_none")]
        object_key: Option<&'a str>,
        medium: Option<&'a str>,
        tenant_id: &'a str,
        seq_hashes: Vec<u64>,
        token_ids: Option<Vec<u32>>,
    }

    fn payload(events: &[WireEvent<'_>]) -> Vec<u8> {
        rmp_serde::to_vec_named(&(1_700_000_000_000u64, events, 0u32)).unwrap()
    }

    #[test]
    fn decodes_the_master_batch() {
        let key = format!("m_{H1}_0_k");
        let bytes = payload(&[
            WireEvent {
                event_type: "stored",
                legacy: "BlockStored",
                object_key: Some(&key),
                medium: Some("cpu"),
                tenant_id: "default",
                seq_hashes: vec![],
                token_ids: None,
            },
            WireEvent {
                event_type: "removed",
                legacy: "BlockRemoved",
                object_key: Some(&key),
                medium: Some("disk"),
                tenant_id: "default",
                seq_hashes: vec![],
                token_ids: None,
            },
            WireEvent {
                event_type: "cleared",
                legacy: "AllBlocksCleared",
                object_key: None,
                medium: None,
                tenant_id: "default",
                seq_hashes: vec![],
                token_ids: None,
            },
            WireEvent {
                event_type: "something_new",
                legacy: "",
                object_key: Some(&key),
                medium: None,
                tenant_id: "default",
                seq_hashes: vec![],
                token_ids: None,
            },
        ]);
        let events = decode_pool_batch(&bytes).unwrap();
        assert_eq!(
            events,
            vec![
                PoolEvent::Stored {
                    object_key: key.clone(),
                    medium: Some("cpu".into()),
                    tenant: Some("default".into()),
                },
                PoolEvent::Removed {
                    object_key: key,
                    medium: Some("disk".into()),
                    tenant: Some("default".into()),
                },
                PoolEvent::Cleared {
                    tenant: Some("default".into()),
                },
            ]
        );
        // A two-element batch (no dp_rank trailer) decodes too.
        let short = rmp_serde::to_vec_named(&(0u64, Vec::<WireEvent<'_>>::new())).unwrap();
        assert!(decode_pool_batch(&short).unwrap().is_empty());
        assert!(decode_pool_batch(b"\x01").is_err());
    }

    fn message(seq: u64, events: &[WireEvent<'_>]) -> ZmqMessage {
        let mut msg = ZmqMessage::from(Vec::<u8>::new());
        msg.push_back(seq.to_be_bytes().to_vec().into());
        msg.push_back(payload(events).into());
        msg
    }

    #[test]
    fn a_sequence_gap_drops_the_view() {
        let pool = SharedPool::default();
        pool.enable(config(None, 1));
        let k1 = format!("{H1}_0_k");
        let k2 = format!("{H2}_0_k");
        let ev = |key: &'static str| WireEvent {
            event_type: "stored",
            legacy: "BlockStored",
            object_key: Some(key),
            medium: Some("cpu"),
            tenant_id: "default",
            seq_hashes: vec![],
            token_ids: None,
        };
        let (k1, k2): (&'static str, &'static str) = (k1.leak(), k2.leak());
        let mut last = None;
        handle_message(&pool, &message(7, &[ev(k1)]), &mut last);
        handle_message(&pool, &message(8, &[ev(k2)]), &mut last);
        assert_eq!(pool.depth(&[1, 2]), 2);
        handle_message(&pool, &message(10, &[]), &mut last);
        assert_eq!(
            (pool.depth(&[1, 2]), pool.resets()),
            (0, 1),
            "seq 9 was lost"
        );
        handle_message(&pool, &message(11, &[ev(k1)]), &mut last);
        assert_eq!(pool.depth(&[1, 2]), 1);
        handle_message(&pool, &message(3, &[]), &mut last);
        assert_eq!(pool.resets(), 2, "a regressed seq is a master restart");
    }

    /// `[seq, payload]` hex of the stream a Mooncake 0.3.14 `mooncake_master`
    /// (`--enable_kv_events=true --kv_events_backend_id=mooncake-l3`)
    /// published for: put of `tag_Qwen-Qwen3-8B_<ab x 32>_0_k` and `..._0_v`,
    /// then remove of `..._0_v`.
    const MASTER_STREAM: &[(&str, &str)] = &[
        ("0000000000000001", "93cf000001a1d3bd291491de0015a86576656e745f696401a974696d657374616d70cf000001a1d3bd2914aa6576656e745f74797065a673746f726564a474797065ab426c6f636b53746f726564aa6d6f64656c5f6e616d65c0aa626c6f636b5f73697a65c0af6164646974696f6e616c5f73616c74c0a96c6f72615f6e616d65c0a974656e616e745f6964a764656661756c74aa6261636b656e645f6964ab6d6f6f6e63616b652d6c33a66d656469756da3637075a764705f72616e6b00a867726f75705f6964c0aa6f626a6563745f6b6579d9567461675f5177656e2d5177656e332d38425f616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261625f305f6baa7365715f68617368657390ac626c6f636b5f68617368657390aa73657373696f6e5f6964c0ae626173655f626c6f636b5f696478c0ab706172656e745f68617368c0a9746f6b656e5f696473c0b1706172656e745f626c6f636b5f68617368c000"),
        ("0000000000000002", "93cf000001a1d3bd291491de0015a86576656e745f696402a974696d657374616d70cf000001a1d3bd2914aa6576656e745f74797065a673746f726564a474797065ab426c6f636b53746f726564aa6d6f64656c5f6e616d65c0aa626c6f636b5f73697a65c0af6164646974696f6e616c5f73616c74c0a96c6f72615f6e616d65c0a974656e616e745f6964a764656661756c74aa6261636b656e645f6964ab6d6f6f6e63616b652d6c33a66d656469756da3637075a764705f72616e6b00a867726f75705f6964c0aa6f626a6563745f6b6579d9567461675f5177656e2d5177656e332d38425f616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261625f305f76aa7365715f68617368657390ac626c6f636b5f68617368657390aa73657373696f6e5f6964c0ae626173655f626c6f636b5f696478c0ab706172656e745f68617368c0a9746f6b656e5f696473c0b1706172656e745f626c6f636b5f68617368c000"),
        ("0000000000000003", "93cf000001a1d3bd2cfd91de0011a86576656e745f696403a974696d657374616d70cf000001a1d3bd2cfdaa6576656e745f74797065a772656d6f766564a474797065ac426c6f636b52656d6f766564aa6d6f64656c5f6e616d65c0aa626c6f636b5f73697a65c0af6164646974696f6e616c5f73616c74c0a96c6f72615f6e616d65c0a974656e616e745f6964a764656661756c74aa6261636b656e645f6964ab6d6f6f6e63616b652d6c33a66d656469756da3637075a764705f72616e6b00a867726f75705f6964c0aa6f626a6563745f6b6579d9567461675f5177656e2d5177656e332d38425f616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261626162616261625f305f76aa7365715f68617368657390ac626c6f636b5f68617368657390ae626173655f626c6f636b5f696478c000"),
    ];

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    #[test]
    fn follows_a_real_master_stream() {
        let pool = SharedPool::default();
        pool.enable(config(Some("tag_Qwen-Qwen3-8B"), 2));
        let hash = i64::from_be_bytes([0xab; 8]);
        let mut last = None;
        let mut depths = Vec::new();
        for (seq, payload) in MASTER_STREAM {
            let mut msg = ZmqMessage::from(Vec::<u8>::new());
            msg.push_back(unhex(seq).into());
            msg.push_back(unhex(payload).into());
            handle_message(&pool, &msg, &mut last);
            depths.push(pool.depth(&[hash]));
        }
        assert_eq!(
            depths,
            vec![0, 1, 0],
            "K, then K and V (held), then V removed"
        );
        assert_eq!((pool.resets(), pool.applied()), (0, 3));
    }

    #[tokio::test]
    async fn follows_a_publisher_over_zmq() {
        use zeromq::{PubSocket, SocketSend};
        let mut publisher = PubSocket::new();
        let endpoint = publisher
            .bind("tcp://127.0.0.1:0")
            .await
            .unwrap()
            .to_string();
        let tree = Arc::new(super::super::HashTree::new());
        tree.shared_pool().enable(config(None, 1));
        let cancel = CancellationToken::new();
        let task = tokio::spawn(run_pool_subscriber(
            endpoint,
            Arc::clone(&tree),
            cancel.clone(),
        ));
        let key: &'static str = format!("{H1}_0_k").leak();
        let ev = WireEvent {
            event_type: "stored",
            legacy: "BlockStored",
            object_key: Some(key),
            medium: Some("cpu"),
            tenant_id: "default",
            seq_hashes: vec![],
            token_ids: None,
        };
        // PUB drops messages until the subscription lands; resend until seen.
        let mut seq = 0u64;
        for _ in 0..100 {
            seq += 1;
            publisher
                .send(message(seq, std::slice::from_ref(&ev)))
                .await
                .unwrap();
            if tree.shared_pool().depth(&[1]) == 1 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(tree.shared_pool().depth(&[1]), 1);
        cancel.cancel();
        task.await.unwrap();
    }
}
