use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    response::Response,
};
use bytes::Bytes;
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::watch;

pub const RESPONSE_CACHE_HEADER: HeaderName = HeaderName::from_static("x-response-cache");

pub type ResponseCacheKey = [u8; 32];

#[derive(Clone, Debug)]
pub struct CachedResponse {
    status: StatusCode,
    headers: HeaderMap,
    body: Bytes,
}

impl CachedResponse {
    pub fn new(status: StatusCode, headers: HeaderMap, body: Bytes) -> Self {
        Self {
            status,
            headers,
            body,
        }
    }

    pub fn into_response(&self, cache_status: &'static str) -> Response {
        let mut response = Response::new(Body::from(self.body.clone()));
        *response.status_mut() = self.status;
        *response.headers_mut() = self.headers.clone();
        response.headers_mut().insert(
            RESPONSE_CACHE_HEADER,
            HeaderValue::from_static(cache_status),
        );
        response
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }
}

#[derive(Clone, Debug)]
struct ReadyEntry {
    response: Arc<CachedResponse>,
    stored_at: Instant,
    last_access: u64,
}

#[derive(Clone, Debug)]
enum CacheEntry {
    Ready(ReadyEntry),
    InFlight(watch::Receiver<Option<Arc<CachedResponse>>>),
}

#[derive(Debug, Default)]
struct CacheState {
    entries: HashMap<ResponseCacheKey, CacheEntry>,
    ready_count: usize,
    inflight_count: usize,
    access_clock: u64,
    epoch: u64,
}

#[derive(Debug)]
pub enum CacheLookup {
    Hit(Arc<CachedResponse>),
    Owner {
        sender: watch::Sender<Option<Arc<CachedResponse>>>,
        epoch: u64,
    },
    Wait(watch::Receiver<Option<Arc<CachedResponse>>>),
    Bypass,
}

#[derive(Debug)]
pub struct ResponseCache {
    max_entries: usize,
    ttl: Duration,
    max_response_bytes: usize,
    state: Mutex<CacheState>,
}

impl ResponseCache {
    pub fn new(max_entries: usize, ttl: Duration, max_response_bytes: usize) -> Self {
        assert!(max_entries > 0, "response cache capacity must be positive");
        assert!(!ttl.is_zero(), "response cache TTL must be positive");
        assert!(
            max_response_bytes > 0,
            "response cache response-size limit must be positive"
        );
        Self {
            max_entries,
            ttl,
            max_response_bytes,
            state: Mutex::new(CacheState::default()),
        }
    }

    pub fn max_response_bytes(&self) -> usize {
        self.max_response_bytes
    }

    pub fn begin(&self, key: ResponseCacheKey) -> CacheLookup {
        let mut state = self.state.lock();

        match state.entries.get(&key).cloned() {
            Some(CacheEntry::Ready(entry)) if entry.stored_at.elapsed() <= self.ttl => {
                state.access_clock = state.access_clock.wrapping_add(1);
                let access = state.access_clock;
                if let Some(CacheEntry::Ready(stored)) = state.entries.get_mut(&key) {
                    stored.last_access = access;
                }
                return CacheLookup::Hit(entry.response);
            }
            Some(CacheEntry::Ready(_)) => {
                state.entries.remove(&key);
                state.ready_count -= 1;
            }
            Some(CacheEntry::InFlight(receiver)) if receiver.has_changed().is_ok() => {
                return CacheLookup::Wait(receiver);
            }
            Some(CacheEntry::InFlight(_)) => {
                // The owner was cancelled before completing the request. Let this
                // caller become the new owner instead of leaving a dead entry.
                state.entries.remove(&key);
                state.inflight_count -= 1;
            }
            None => {}
        }

        if state.inflight_count >= self.max_entries {
            // Reap cancelled owners across keys only when the independent
            // in-flight budget is saturated, keeping the common miss path O(1).
            let cancelled: Vec<_> = state
                .entries
                .iter_mut()
                .filter_map(|(candidate, entry)| match entry {
                    CacheEntry::InFlight(receiver) if receiver.has_changed().is_err() => {
                        Some(*candidate)
                    }
                    _ => None,
                })
                .collect();
            for cancelled_key in cancelled {
                state.entries.remove(&cancelled_key);
                state.inflight_count -= 1;
            }
            if state.inflight_count >= self.max_entries {
                return CacheLookup::Bypass;
            }
        }

        let (sender, receiver) = watch::channel(None);
        state.entries.insert(key, CacheEntry::InFlight(receiver));
        state.inflight_count += 1;
        CacheLookup::Owner {
            sender,
            epoch: state.epoch,
        }
    }

    pub async fn wait(
        &self,
        mut receiver: watch::Receiver<Option<Arc<CachedResponse>>>,
    ) -> Option<Arc<CachedResponse>> {
        tokio::time::timeout(self.ttl, async move {
            loop {
                if let Some(response) = receiver.borrow().clone() {
                    return Some(response);
                }
                if receiver.changed().await.is_err() {
                    return None;
                }
            }
        })
        .await
        .ok()
        .flatten()
    }

    pub fn complete(
        &self,
        key: ResponseCacheKey,
        epoch: u64,
        sender: watch::Sender<Option<Arc<CachedResponse>>>,
        response: Arc<CachedResponse>,
        persist: bool,
    ) {
        let mut state = self.state.lock();
        if epoch != state.epoch {
            return;
        }
        if matches!(state.entries.remove(&key), Some(CacheEntry::InFlight(_))) {
            state.inflight_count -= 1;
        }
        if persist {
            if state.ready_count >= self.max_entries {
                let evicted = state
                    .entries
                    .iter()
                    .filter_map(|(candidate, entry)| match entry {
                        CacheEntry::Ready(ready) => Some((*candidate, ready.last_access)),
                        CacheEntry::InFlight(_) => None,
                    })
                    .min_by_key(|(_, last_access)| *last_access)
                    .map(|(candidate, _)| candidate);
                if let Some(evicted) = evicted {
                    state.entries.remove(&evicted);
                    state.ready_count -= 1;
                }
            }
            state.access_clock = state.access_clock.wrapping_add(1);
            let last_access = state.access_clock;
            state.entries.insert(
                key,
                CacheEntry::Ready(ReadyEntry {
                    response: response.clone(),
                    stored_at: Instant::now(),
                    last_access,
                }),
            );
            state.ready_count += 1;
            drop(state);
            let _ = sender.send(Some(response));
        } else {
            // Dropping the owner wakes waiters with None. They bypass the cache
            // independently instead of inheriting one transient error.
        }
    }

    pub fn abandon(
        &self,
        key: ResponseCacheKey,
        epoch: u64,
        sender: watch::Sender<Option<Arc<CachedResponse>>>,
    ) {
        let mut state = self.state.lock();
        if epoch == state.epoch
            && matches!(state.entries.remove(&key), Some(CacheEntry::InFlight(_)))
        {
            state.inflight_count -= 1;
        }
        drop(state);
        drop(sender);
    }

    pub fn clear(&self) {
        let mut state = self.state.lock();
        state.epoch = state.epoch.wrapping_add(1);
        state.entries.clear();
        state.ready_count = 0;
        state.inflight_count = 0;
    }

    #[cfg(test)]
    fn ready_len(&self) -> usize {
        self.state.lock().ready_count
    }

    #[cfg(test)]
    fn contains_ready(&self, key: ResponseCacheKey) -> bool {
        matches!(
            self.state.lock().entries.get(&key),
            Some(CacheEntry::Ready(_))
        )
    }
}

pub fn response_cache_key(
    namespace: &str,
    method: &Method,
    path: &str,
    headers: &HeaderMap,
    body: &[u8],
) -> Option<ResponseCacheKey> {
    if method != Method::POST
        || !matches!(
            path,
            "/generate" | "/v1/completions" | "/v1/chat/completions"
        )
    {
        return None;
    }

    let payload: Value = serde_json::from_slice(body).ok()?;
    let object = payload.as_object()?;
    if object.get("stream").and_then(Value::as_bool) == Some(true)
        || object.contains_key("session_params")
        || object.contains_key("session_id")
    {
        return None;
    }

    let temperature = if path == "/generate" {
        object
            .get("sampling_params")
            .and_then(Value::as_object)
            .and_then(|params| params.get("temperature"))
    } else {
        object.get("temperature")
    };
    if temperature.and_then(Value::as_f64) != Some(0.0) {
        return None;
    }

    if namespace.is_empty()
        || headers.get("authorization").is_none()
        || headers
            .get("x-response-cache-scope")
            .is_none_or(|value| value.is_empty())
    {
        return None;
    }

    let mut hasher = blake3::Hasher::new();
    hasher.update(namespace.as_bytes());
    hasher.update(&[0]);
    hasher.update(method.as_str().as_bytes());
    hasher.update(&[0]);
    hasher.update(path.as_bytes());
    hasher.update(&[0]);
    for header_name in [
        "authorization",
        "x-response-cache-scope",
        "x-smg-routing-key",
        "x-smg-target-worker",
    ] {
        hasher.update(header_name.as_bytes());
        hasher.update(&[0]);
        if let Some(value) = headers.get(header_name) {
            hasher.update(value.as_bytes());
        }
        hasher.update(&[0]);
    }
    hasher.update(body);
    Some(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn response(body: &'static [u8]) -> Arc<CachedResponse> {
        Arc::new(CachedResponse::new(
            StatusCode::OK,
            HeaderMap::new(),
            Bytes::from_static(body),
        ))
    }

    fn key(byte: u8) -> ResponseCacheKey {
        [byte; 32]
    }

    #[test]
    fn key_accepts_only_non_streaming_explicit_greedy_requests() {
        let mut headers = HeaderMap::new();
        headers.insert("authorization", HeaderValue::from_static("Bearer test"));
        headers.insert(
            "x-response-cache-scope",
            HeaderValue::from_static("test-scope"),
        );
        let greedy = br#"{"text":"hello","sampling_params":{"temperature":0}}"#;
        let stochastic = br#"{"text":"hello","sampling_params":{"temperature":0.7}}"#;
        let streaming = br#"{"text":"hello","stream":true,"sampling_params":{"temperature":0}}"#;
        assert!(
            response_cache_key("revision", &Method::POST, "/generate", &headers, greedy).is_some()
        );
        assert!(
            response_cache_key("revision", &Method::POST, "/generate", &headers, stochastic)
                .is_none()
        );
        assert!(
            response_cache_key("revision", &Method::POST, "/generate", &headers, streaming)
                .is_none()
        );
        assert!(
            response_cache_key("revision", &Method::GET, "/generate", &headers, greedy).is_none()
        );

        let no_identity = HeaderMap::new();
        assert!(
            response_cache_key("revision", &Method::POST, "/generate", &no_identity, greedy)
                .is_none()
        );
        assert_ne!(
            response_cache_key("revision-a", &Method::POST, "/generate", &headers, greedy),
            response_cache_key("revision-b", &Method::POST, "/generate", &headers, greedy)
        );
    }

    #[test]
    fn key_isolates_authorization_and_explicit_scope() {
        let body = br#"{"model":"m","messages":[],"temperature":0}"#;
        let mut first = HeaderMap::new();
        first.insert("authorization", HeaderValue::from_static("Bearer first"));
        first.insert(
            "x-response-cache-scope",
            HeaderValue::from_static("tenant-a"),
        );
        let mut second = HeaderMap::new();
        second.insert("authorization", HeaderValue::from_static("Bearer second"));
        second.insert(
            "x-response-cache-scope",
            HeaderValue::from_static("tenant-a"),
        );
        assert_ne!(
            response_cache_key(
                "revision",
                &Method::POST,
                "/v1/chat/completions",
                &first,
                body
            ),
            response_cache_key(
                "revision",
                &Method::POST,
                "/v1/chat/completions",
                &second,
                body
            )
        );
        second.insert("authorization", HeaderValue::from_static("Bearer first"));
        second.insert(
            "x-response-cache-scope",
            HeaderValue::from_static("tenant-b"),
        );
        assert_ne!(
            response_cache_key(
                "revision",
                &Method::POST,
                "/v1/chat/completions",
                &first,
                body
            ),
            response_cache_key(
                "revision",
                &Method::POST,
                "/v1/chat/completions",
                &second,
                body
            )
        );
    }

    #[tokio::test]
    async fn coalesces_inflight_requests() {
        let cache = ResponseCache::new(2, Duration::from_secs(60), 1024);
        let (owner, epoch) = match cache.begin(key(1)) {
            CacheLookup::Owner { sender, epoch } => (sender, epoch),
            other => panic!("expected owner, got {other:?}"),
        };
        let waiter = match cache.begin(key(1)) {
            CacheLookup::Wait(receiver) => receiver,
            other => panic!("expected waiter, got {other:?}"),
        };
        let expected = response(b"answer");
        cache.complete(key(1), epoch, owner, expected.clone(), true);
        assert_eq!(
            cache.wait(waiter).await.unwrap().body.as_ref(),
            expected.body.as_ref()
        );
        assert!(matches!(cache.begin(key(1)), CacheLookup::Hit(_)));
    }

    #[test]
    fn evicts_least_recently_used_ready_entry() {
        let cache = ResponseCache::new(2, Duration::from_secs(60), 1024);
        for byte in [1, 2] {
            let CacheLookup::Owner {
                sender: owner,
                epoch,
            } = cache.begin(key(byte))
            else {
                panic!("expected owner");
            };
            cache.complete(key(byte), epoch, owner, response(b"value"), true);
        }
        assert!(matches!(cache.begin(key(1)), CacheLookup::Hit(_)));
        let CacheLookup::Owner {
            sender: owner,
            epoch,
        } = cache.begin(key(3))
        else {
            panic!("expected owner");
        };
        cache.complete(key(3), epoch, owner, response(b"value"), true);
        assert_eq!(cache.ready_len(), 2);
        assert!(cache.contains_ready(key(1)));
        assert!(!cache.contains_ready(key(2)));
        assert!(cache.contains_ready(key(3)));
    }

    #[tokio::test]
    async fn expired_entry_is_not_returned() {
        let cache = ResponseCache::new(1, Duration::from_millis(1), 1024);
        let CacheLookup::Owner {
            sender: owner,
            epoch,
        } = cache.begin(key(1))
        else {
            panic!("expected owner");
        };
        cache.complete(key(1), epoch, owner, response(b"value"), true);
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(matches!(cache.begin(key(1)), CacheLookup::Owner { .. }));
    }

    #[test]
    fn bounds_unique_inflight_requests() {
        let cache = ResponseCache::new(1, Duration::from_secs(60), 1024);
        let CacheLookup::Owner { sender, .. } = cache.begin(key(1)) else {
            panic!("expected owner");
        };
        assert!(matches!(cache.begin(key(2)), CacheLookup::Bypass));
        drop(sender);
        assert!(matches!(cache.begin(key(2)), CacheLookup::Owner { .. }));
    }

    #[tokio::test]
    async fn waiters_bypass_after_non_cacheable_completion() {
        let cache = ResponseCache::new(1, Duration::from_secs(60), 1024);
        let CacheLookup::Owner {
            sender: owner,
            epoch,
        } = cache.begin(key(1))
        else {
            panic!("expected owner");
        };
        let CacheLookup::Wait(waiter) = cache.begin(key(1)) else {
            panic!("expected waiter");
        };
        cache.complete(key(1), epoch, owner, response(b"error"), false);
        assert!(cache.wait(waiter).await.is_none());
    }

    #[tokio::test]
    async fn waiters_have_a_deadline() {
        let cache = ResponseCache::new(1, Duration::from_millis(1), 1024);
        let CacheLookup::Owner { sender: owner, .. } = cache.begin(key(1)) else {
            panic!("expected owner");
        };
        let CacheLookup::Wait(waiter) = cache.begin(key(1)) else {
            panic!("expected waiter");
        };
        assert!(cache.wait(waiter).await.is_none());
        drop(owner);
    }

    #[test]
    fn clear_invalidates_ready_and_inflight_entries() {
        let cache = ResponseCache::new(2, Duration::from_secs(60), 1024);
        let CacheLookup::Owner {
            sender: ready_owner,
            epoch: ready_epoch,
        } = cache.begin(key(1))
        else {
            panic!("expected owner");
        };
        let CacheLookup::Owner {
            sender: stale_owner,
            epoch: stale_epoch,
        } = cache.begin(key(2))
        else {
            panic!("expected owner");
        };
        cache.complete(key(1), ready_epoch, ready_owner, response(b"ready"), true);
        cache.clear();
        assert_eq!(cache.ready_len(), 0);
        assert!(matches!(cache.begin(key(1)), CacheLookup::Owner { .. }));
        assert!(matches!(cache.begin(key(2)), CacheLookup::Owner { .. }));

        cache.complete(key(2), stale_epoch, stale_owner, response(b"stale"), true);
        assert!(!matches!(cache.begin(key(2)), CacheLookup::Hit(_)));
    }
}
