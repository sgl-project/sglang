// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Worker affinity for multimodal conversations.
//!
//! # Why cache-aware routing alone cannot handle images
//!
//! The engine turns each image into a run of synthetic token ids: it hashes the
//! PREPROCESSED pixel tensor and derives a `pad_value` from that hash
//! (`_compute_pad_value` in `python/sglang/srt/managers/schedule_batch.py`),
//! then substitutes that id for the image's placeholder run. Image identity
//! therefore lives inside the token ids, and the engine's radix cache reuses an
//! image's KV across turns exactly as it does for text.
//!
//! The router cannot reproduce those ids. It never fetches or decodes images, and
//! even if it did, matching `pad_value` would require running the model's vision
//! preprocessing bit-exactly to hash the identical tensor. So the router's block
//! hashes diverge at the first image block and at every block after it.
//!
//! The consequence is worse than "the image doesn't match": for a conversation
//! whose image arrives in turn 1, everything after the image is unmatchable, and
//! the matchable remainder — the system prompt — is identical on every worker.
//! Overlap is a tie, the policy falls through to min-load, and turn 2 lands on a
//! different worker than turn 1. The engine would have hit its cache; the router
//! sent the request somewhere that could not.
//!
//! # What this does instead
//!
//! Pin a conversation to the worker that already served it, keyed on something
//! the router CAN compute from the request: the first image's reference (its URL
//! or inline payload). That key is stable across turns — clients resend the full
//! history — so turn 2 returns to turn 1's worker and the ENGINE's cache hits.
//!
//! This is affinity, not prefix matching. It does not make the router's hashes
//! match the engine's, and it makes no attempt to; it removes the mis-routing
//! that prevents the engine's own (working) image cache from being used.
//!
//! # Relationship to [`super::sticky`]
//!
//! Both map a key to a pinned worker with idle eviction, but they are not the
//! same feature and are deliberately not shared: sticky pins a CLIENT-declared
//! session header and is the whole routing policy, while this pins a
//! ROUTER-derived content key and is a fallback inside cache-aware routing for
//! the one input class that routing cannot hash. They can be active at once.

use std::borrow::Cow;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use dashmap::DashMap;
use sha2::{Digest, Sha256};

use crate::policies::active_load::{spawn_sweeper, Clock, JanitorHandle, SystemTimeClock};
use crate::server::metrics::{MetricsRegistry, MmAffinityOutcome};
use crate::tokenizer::pyjson::{compact_json, deep_sort};
use crate::workers::Worker;

/// Content part types that denote an image across the request shapes the router
/// sees (OpenAI `image_url`, the Responses-style `input_image`, and the bare
/// `image` the Kimi processor accepts).
const IMAGE_PART_TYPES: [&str; 3] = ["image_url", "image", "input_image"];

/// Media part types affinity does NOT key on. Listed solely so a request
/// carrying them is distinguishable from a text-only one: affinity keys off the
/// first IMAGE, so these yield [`AffinityKey::Unkeyed`] — the label whose whole
/// purpose is to say "a shape class is routing cold every turn". Without this
/// list an audio-only conversation would be indistinguishable from plain text
/// and the regression would be invisible, which is the thing the label exists
/// to prevent.
const NON_IMAGE_MEDIA_PART_TYPES: [&str; 5] =
    ["input_audio", "audio", "audio_url", "video", "video_url"];

/// What a request's media content yields for affinity purposes.
///
/// Three states, not `Option`, because the caller must distinguish "no media, so
/// affinity does not apply" from "media present but unkeyable, which is a
/// regression worth a metric". Collapsing them is exactly the bug this replaced:
/// the outcome was inferred from "some `content` is an array", which is true of
/// the `[{"type":"text",…}]` shape most SDKs send, so ordinary text traffic
/// counted as unkeyed multimodal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AffinityKey {
    /// No media part at all — route by prefix overlap, record nothing.
    NoMedia,
    /// A media part is present but no stable reference could be derived from it
    /// (a non-image medium, or an image part carrying no payload). Gets no
    /// affinity and no usable prefix hash, so it routes cold every turn.
    Unkeyed,
    /// The conversation's key.
    Key(String),
}

/// Derive the affinity key for a chat request. See [`AffinityKey`] for the
/// three outcomes.
///
/// The key is the first image's reference, hashed. First rather than all:
/// a later turn may ADD an image, and a key over every image would change when
/// it did — breaking affinity at exactly the moment the conversation got more
/// expensive to recompute. The first image is present in every turn from the one
/// that introduced it onward, so the key is stable for the conversation's life.
///
/// Hashed rather than used raw because an inline `data:` payload is megabytes;
/// the map holds one bounded-size key per conversation instead.
///
/// Two conversations that open with the SAME image share a key, and so share a
/// worker. That is intentional: they also share the engine-side prefix that the
/// image dominates, so co-locating them is the same bet cache-aware routing
/// makes for a shared text prefix.
pub fn affinity_key(model: &str, value: &serde_json::Value) -> AffinityKey {
    let reference = match first_image_reference(value) {
        MediaScan::None => return AffinityKey::NoMedia,
        MediaScan::Unkeyable => return AffinityKey::Unkeyed,
        MediaScan::Reference(r) => r,
    };
    let mut hasher = Sha256::new();
    // Model-qualified so one router serving two models cannot collide on an
    // image used by both.
    hasher.update(model.as_bytes());
    hasher.update([0u8]);
    hasher.update(reference.as_ref().as_bytes());
    AffinityKey::Key(format!("{:x}", hasher.finalize())[..32].to_string())
}

/// Outcome of scanning `messages` for a keyable media reference.
enum MediaScan<'a> {
    /// No media part was seen.
    None,
    /// At least one media part was seen, none of them keyable.
    Unkeyable,
    /// The first image's reference.
    Reference(Cow<'a, str>),
}

/// The first image reference in `messages`, in wire order.
///
/// Prefers a recognized reference field (`url`, `file_id`, or a bare string
/// payload). Falls back to a stable serialization of the payload, so a shape
/// nobody anticipated still produces a STABLE key rather than none — affinity
/// never interprets this value, it only needs it to repeat.
///
/// Reports [`MediaScan::Unkeyable`] rather than `None` when media was present
/// but nothing keyable came out of it, so the caller can tell a cold-routing
/// regression from ordinary text traffic.
///
/// `Cow` because the common case borrows the reference straight out of the
/// request; only the fallback allocates.
fn first_image_reference(value: &serde_json::Value) -> MediaScan<'_> {
    let Some(messages) = value.get("messages").and_then(|m| m.as_array()) else {
        return MediaScan::None;
    };
    let mut saw_media = false;
    for message in messages {
        let Some(parts) = message.get("content").and_then(|c| c.as_array()) else {
            continue;
        };
        for part in parts {
            let Some(part_type) = part.get("type").and_then(|t| t.as_str()) else {
                continue;
            };
            if NON_IMAGE_MEDIA_PART_TYPES.contains(&part_type) {
                saw_media = true;
                continue;
            }
            if !IMAGE_PART_TYPES.contains(&part_type) {
                continue;
            }
            saw_media = true;
            let payload = part.get(part_type).or_else(|| part.get("image_url"));
            // A recognized reference field first...
            let reference = match payload {
                Some(serde_json::Value::String(s)) => Some(s.as_str()),
                Some(serde_json::Value::Object(o)) => o
                    .get("url")
                    .or_else(|| o.get("file_id"))
                    .and_then(|u| u.as_str()),
                _ => None,
            }
            .or_else(|| part.get("url").and_then(|u| u.as_str()))
            .or_else(|| part.get("file_id").and_then(|u| u.as_str()));
            if let Some(r) = reference.filter(|r| !r.is_empty()) {
                return MediaScan::Reference(Cow::Borrowed(r));
            }
            // ...then a stable serialization of the PAYLOAD. Affinity only needs
            // a key that repeats across turns; it never interprets the value.
            // Without this, an image shape nobody anticipated silently gets no
            // affinity at all, which is strictly worse than keying on bytes we
            // do not understand.
            //
            // `deep_sort` first: `serde_json`'s `preserve_order` is on, so
            // without it two turns of ONE conversation would key differently the
            // moment a client re-serialized the same payload with different key
            // order — which is precisely what a client that rebuilds the request
            // per turn from its own object model does. Sorting makes the key
            // depend on the payload's content rather than on its wire byte
            // order. Note the key covers only the payload, not the whole part:
            // two parts differing only in a sibling field (`detail`) collide,
            // which is harmless here — they carry the same image.
            if let Some(p) = payload.filter(|p| !p.is_null()) {
                return MediaScan::Reference(Cow::Owned(compact_json(&deep_sort(p))));
            }
        }
    }
    if saw_media {
        MediaScan::Unkeyable
    } else {
        MediaScan::None
    }
}

/// Result of a pin lookup.
pub enum PinLookup {
    /// The pinned worker is on offer — route there.
    Hit(Arc<Worker>),
    /// A pin exists but its worker is not on offer: at its in-flight cap, or
    /// departed. Route elsewhere for this turn but LEAVE THE PIN, so a
    /// momentarily-saturated conversation can come home.
    Unavailable,
    /// No pin for this key — the caller selects normally and records one.
    Miss,
}

/// How long a pin may stay CONTINUOUSLY unavailable before it is dropped.
///
/// Saturation is transient — a worker at its in-flight cap frees a slot within
/// milliseconds to seconds. Departure is permanent. From [`MultimodalAffinity::pinned`]'s
/// vantage point the two are indistinguishable (admission offers only workers
/// under their cap, so a saturated worker disappears from the candidate list
/// exactly like a dead one), which is why the pin is kept on `Unavailable` at
/// all. But kept UNBOUNDEDLY, a departed worker's pin survives until the pin
/// goes idle — 30 minutes by default — and every turn of an active image
/// conversation routes cold for that whole window. A rolling restart does this
/// to every image conversation at once.
///
/// Bounding it preserves come-home for real saturation while making a departure
/// cost one re-pin. Deliberately a DURATION and not a count of unavailable
/// lookups: `Policy::select` runs several times per request on the retry path,
/// so any count would be a function of retry behavior rather than of how long
/// the worker has actually been gone.
const UNAVAILABLE_GRACE: Duration = Duration::from_secs(60);

/// One key → worker pin, with the last reference time for idle eviction.
#[derive(Debug)]
struct Pin {
    worker_url: String,
    last_seen: Instant,
    /// When this pin's worker was first found missing from the offered workers,
    /// cleared on every hit. `Some` means "continuously unavailable since"; see
    /// [`UNAVAILABLE_GRACE`].
    unavailable_since: Option<Instant>,
}

#[derive(Debug)]
struct State {
    pins: DashMap<String, Pin>,
    clock: Arc<dyn Clock>,
    idle: Duration,
    metrics: OnceLock<Arc<MetricsRegistry>>,
}

impl State {
    fn sweep_expired(&self) -> usize {
        let now = self.clock.now();
        let mut removed = 0;
        self.pins.retain(|_key, p| {
            let keep = now.saturating_duration_since(p.last_seen) <= self.idle;
            if !keep {
                removed += 1;
            }
            keep
        });
        removed
    }

    fn record(&self, outcome: MmAffinityOutcome) {
        if let Some(m) = self.metrics.get() {
            m.record_mm_affinity(outcome);
        }
    }
}

/// Bounded key → worker pin map for multimodal conversations.
pub struct MultimodalAffinity {
    state: Arc<State>,
    _janitor: Option<JanitorHandle>,
}

impl std::fmt::Debug for MultimodalAffinity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultimodalAffinity")
            .field("pins", &self.state.pins.len())
            .finish()
    }
}

impl MultimodalAffinity {
    /// Pins idle longer than `idle` are swept on `eviction_interval`.
    ///
    /// `idle == 0` DISABLES affinity: no pin is ever recorded or returned. It is
    /// an explicit guard rather than "everything expires instantly", because the
    /// latter depends on sweep timing — between sweeps a zero-idle map would
    /// still serve pins, which is not what disabling means.
    ///
    /// The sweeper needs a Tokio runtime. Production builds this policy inside
    /// `#[tokio::main]` (via the policy factory), so the sweeper always spawns;
    /// only sync test helpers hit the other branch, where eviction is off and
    /// the map is bounded by the test instead.
    pub fn new(idle: Duration, eviction_interval: Duration) -> Self {
        let state = Arc::new(State {
            pins: DashMap::new(),
            clock: Arc::new(SystemTimeClock),
            idle,
            metrics: OnceLock::new(),
        });
        let _janitor = if idle.is_zero() {
            tracing::info!("multimodal affinity disabled (idle window is 0)");
            None
        } else if tokio::runtime::Handle::try_current().is_ok() {
            let swept = Arc::clone(&state);
            Some(spawn_sweeper(
                move || swept.sweep_expired(),
                eviction_interval,
                "mm-affinity-eviction",
            ))
        } else {
            tracing::debug!(
                "MultimodalAffinity constructed outside a Tokio runtime; idle eviction is disabled"
            );
            None
        };
        Self { state, _janitor }
    }

    /// Affinity is off; `pinned` never returns and `record` never stores.
    fn disabled(&self) -> bool {
        self.state.idle.is_zero()
    }

    /// [`UNAVAILABLE_GRACE`], capped at the idle window: a router configured
    /// with a short idle must not grant a longer grace than the window in which
    /// the pin would have expired anyway.
    fn unavailable_grace(&self) -> Duration {
        UNAVAILABLE_GRACE.min(self.state.idle)
    }

    pub fn attach_metrics(&self, metrics: Arc<MetricsRegistry>) {
        let _ = self.state.metrics.set(metrics);
    }

    /// Look up `key`'s pinned worker among the workers on offer.
    ///
    /// Refreshes the pin's idle timer on a hit, so an active conversation is
    /// never swept out from under itself.
    ///
    /// [`PinLookup::Unavailable`] is distinct from [`PinLookup::Miss`] because
    /// the caller must treat them differently, and the difference is easy to
    /// get wrong. Admission offers the policy only workers under their
    /// in-flight cap, so a pinned worker that is merely SATURATED disappears
    /// from `workers` exactly like one that has died. Overwriting the pin in
    /// that case permanently migrates a busy conversation off the worker
    /// holding its KV — degrading affinity precisely under the load where
    /// cache reuse pays most.
    pub fn pinned(&self, key: &str, workers: &[Arc<Worker>]) -> PinLookup {
        if self.disabled() {
            return PinLookup::Miss;
        }
        let Some(mut entry) = self.state.pins.get_mut(key) else {
            return PinLookup::Miss;
        };
        let Some(worker) = workers.iter().find(|w| w.url == entry.worker_url).cloned() else {
            // Deliberately does NOT refresh `last_seen`. A pin whose worker is
            // saturated comes home on a later turn; one whose worker really is
            // gone goes idle and gets swept, after which the conversation
            // re-pins. Refreshing here would make a departed worker's pin
            // immortal for as long as the conversation stayed active.
            let now = self.state.clock.now();
            let since = *entry.unavailable_since.get_or_insert(now);
            if now.saturating_duration_since(since) > self.unavailable_grace() {
                // Long past any plausible saturation: treat the worker as gone
                // and drop the pin so THIS turn re-pins, instead of routing cold
                // every turn until the pin goes idle.
                let stale_url = entry.worker_url.clone();
                drop(entry);
                self.state.pins.remove(key);
                tracing::debug!(
                    worker = %stale_url,
                    grace_secs = self.unavailable_grace().as_secs(),
                    "mm-affinity: pinned worker unavailable beyond the grace window; \
                     dropping the pin so the conversation re-pins",
                );
                return PinLookup::Miss;
            }
            drop(entry);
            self.state.record(MmAffinityOutcome::Unavailable);
            return PinLookup::Unavailable;
        };
        entry.last_seen = self.state.clock.now();
        entry.unavailable_since = None;
        drop(entry);
        self.state.record(MmAffinityOutcome::Hit);
        PinLookup::Hit(worker)
    }

    /// Pin `key` to `worker_url`, overwriting any previous pin.
    ///
    /// Called when a multimodal selection did not come from a pin hit and the
    /// key had no usable pin, so the pin tracks the worker the policy chose —
    /// including when the load-imbalance path overrode affinity.
    ///
    /// NOT a guarantee that the pin equals where every request physically went:
    /// admission can hand a parked request straight to a worker as a slot frees
    /// (`release_slot`), bypassing `Policy::select` entirely. Those turns are
    /// served off-pin and leave it unchanged, which is harmless — the pin still
    /// names a worker holding the conversation's prefix.
    pub fn record(&self, key: &str, worker_url: &str) {
        if self.disabled() {
            return;
        }
        self.state.pins.insert(
            key.to_string(),
            Pin {
                worker_url: worker_url.to_string(),
                last_seen: self.state.clock.now(),
                unavailable_since: None,
            },
        );
        self.state.record(MmAffinityOutcome::Assigned);
    }

    #[cfg(test)]
    fn with_clock(idle: Duration, clock: Arc<dyn Clock>) -> Self {
        Self {
            state: Arc::new(State {
                pins: DashMap::new(),
                clock,
                idle,
                metrics: OnceLock::new(),
            }),
            _janitor: None,
        }
    }

    #[cfg(test)]
    fn sweep_expired(&self) -> usize {
        self.state.sweep_expired()
    }

    #[cfg(test)]
    fn pin_count(&self) -> usize {
        self.state.pins.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policies::active_load::MockClock;
    use serde_json::json;

    /// The keyed outcome only — for the tests that care about key VALUES rather
    /// than which of the two keyless states was reported.
    fn key(model: &str, value: &serde_json::Value) -> Option<String> {
        match affinity_key(model, value) {
            AffinityKey::Key(k) => Some(k),
            AffinityKey::NoMedia | AffinityKey::Unkeyed => None,
        }
    }

    fn worker(url: &str) -> Arc<Worker> {
        use crate::discovery::{ModelId, WorkerId, WorkerMode, WorkerSpec};
        Arc::new(Worker::new(WorkerSpec {
            id: WorkerId(url.into()),
            url: url.into(),
            mode: WorkerMode::Plain,
            model_ids: vec![ModelId("m".into())],
            bootstrap_port: None,
        }))
    }

    /// The key must be identical across turns of one conversation — that is the
    /// entire mechanism. Turn 2 appends messages; the first image does not move.
    #[test]
    fn key_is_stable_as_the_conversation_grows() {
        let turn1 = json!({"messages":[
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x/a.png"}},
                {"type":"text","text":"what is this?"}]}]});
        let turn2 = json!({"messages":[
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x/a.png"}},
                {"type":"text","text":"what is this?"}]},
            {"role":"assistant","content":"a cat"},
            {"role":"user","content":"and its colour?"}]});
        let k1 = key("m", &turn1).expect("turn 1 has an image");
        let k2 = key("m", &turn2).expect("turn 2 still carries the image");
        assert_eq!(k1, k2);

        // A LATER-added image must not change the key, or affinity would break
        // exactly when the conversation became most expensive to recompute.
        let turn3 = json!({"messages":[
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x/a.png"}}]},
            {"role":"user","content":[
                {"type":"image_url","image_url":{"url":"http://x/SECOND.png"}}]}]});
        assert_eq!(key("m", &turn3).unwrap(), k1);
    }

    /// Different images, and the same image on different models, must not share
    /// a pin.
    #[test]
    fn key_separates_images_and_models() {
        let a = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"http://x/a.png"}}]}]});
        let b = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"http://x/b.png"}}]}]});
        assert_ne!(key("m", &a), key("m", &b));
        assert_ne!(key("m1", &a), key("m2", &a));
    }

    /// Every image part shape the router actually receives.
    #[test]
    fn key_reads_the_wire_shapes() {
        for content in [
            json!([{"type":"image_url","image_url":{"url":"REF"}}]),
            json!([{"type":"image_url","image_url":"REF"}]),
            json!([{"type":"image","image":"REF"}]),
            json!([{"type":"input_image","image_url":"REF"}]),
            json!([{"type":"image_url","url":"REF"}]),
            json!([{"type":"input_image","image_url":{"file_id":"REF"}}]),
        ] {
            let v = json!({"messages":[{"role":"user","content":content}]});
            assert_eq!(
                key("m", &v),
                key(
                    "m",
                    &json!({"messages":[{"role":"user","content":
                    [{"type":"image","image":"REF"}]}]})
                ),
                "all shapes carrying REF must produce one key: {content}"
            );
        }
    }

    /// An image shape with no recognized reference field must STILL key, off a
    /// stable serialization of the part — no affinity at all is strictly worse
    /// than keying on bytes we do not interpret.
    #[test]
    fn unrecognized_image_shape_still_keys_stably() {
        let v = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"b64_json":"AAAA","detail":"high"}}]}]});
        let k = key("m", &v).expect("an unrecognized image shape must still key");
        assert_eq!(key("m", &v).unwrap(), k, "and stably");

        let other = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"b64_json":"BBBB","detail":"high"}}]}]});
        assert_ne!(
            key("m", &other).unwrap(),
            k,
            "different payloads must not collide"
        );
    }

    /// Text-only requests get no key, so they keep using cache-aware matching —
    /// affinity must not capture traffic that prefix hashing handles better.
    ///
    /// `NoMedia`, specifically, not merely "not a key". The array-content shape
    /// is the one that matters: `[{"type":"text",…}]` is what most SDKs send for
    /// ordinary text, and reporting it as `Unkeyed` made the metric count the
    /// majority of normal traffic as a multimodal regression.
    #[test]
    fn text_only_requests_report_no_media() {
        for value in [
            json!({"messages":[{"role":"user","content":"hi"}]}),
            json!({"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}),
            json!({"messages":[{"role":"user","content":[
                {"type":"text","text":"hi"},
                {"type":"text","text":"there"}]}]}),
            json!({"prompt":"hi"}),
        ] {
            assert_eq!(
                affinity_key("m", &value),
                AffinityKey::NoMedia,
                "text-only request must report NoMedia: {value}"
            );
        }
    }

    /// Media the key derivation cannot key reports `Unkeyed` — the label exists
    /// to make exactly this class visible, since it routes cold every turn.
    #[test]
    fn unkeyable_media_reports_unkeyed() {
        for value in [
            // A non-image medium: affinity keys off images only.
            json!({"messages":[{"role":"user","content":[
                {"type":"input_audio","input_audio":{"data":"AAAA"}}]}]}),
            json!({"messages":[{"role":"user","content":[{"type":"video","video":"x"}]}]}),
            // An image part carrying no payload at all. Note the contrast with
            // `{"url":""}`, which DOES key: an empty reference falls through to
            // the payload fallback, and `{"url":""}` is a stable payload. Only a
            // missing or null payload leaves nothing to key on.
            json!({"messages":[{"role":"user","content":[{"type":"image_url"}]}]}),
            json!({"messages":[{"role":"user","content":[
                {"type":"image_url","image_url":null}]}]}),
        ] {
            assert_eq!(
                affinity_key("m", &value),
                AffinityKey::Unkeyed,
                "unkeyable media must report Unkeyed: {value}"
            );
        }
    }

    /// An image alongside unkeyable media still keys — the scan returns the
    /// first IMAGE reference, and `saw_media` must not short-circuit it.
    #[test]
    fn media_before_an_image_does_not_suppress_the_key() {
        let audio_first = json!({"messages":[{"role":"user","content":[
            {"type":"input_audio","input_audio":{"data":"AAAA"}},
            {"type":"image_url","image_url":{"url":"http://x/a.png"}}]}]});
        let image_only = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"http://x/a.png"}}]}]});
        assert_eq!(
            key("m", &audio_first).expect("the image is still keyable"),
            key("m", &image_only).unwrap(),
            "an audio part before the image must not change the key"
        );
    }

    /// The fallback key must depend on the payload's CONTENT, not its wire byte
    /// order: `preserve_order` is on, so a client that rebuilds the request from
    /// its own object model can legitimately emit the same payload with keys in
    /// a different order — and must land on the same worker.
    #[test]
    fn fallback_key_ignores_payload_key_order() {
        let a = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"b64_json":"AAAA","detail":"high"}}]}]});
        let b = json!({"messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"detail":"high","b64_json":"AAAA"}}]}]});
        assert_eq!(
            key("m", &a).expect("keys off the payload"),
            key("m", &b).unwrap(),
            "reordered payload keys must not break the pin"
        );
    }

    /// `idle == 0` must disable affinity outright, not merely expire pins on
    /// the next sweep — between sweeps that would still serve them.
    #[test]
    fn zero_idle_disables_affinity() {
        let a = MultimodalAffinity::with_clock(
            Duration::ZERO,
            Arc::new(MockClock::new(Instant::now())),
        );
        let workers = vec![worker("http://w0")];
        a.record("k", "http://w0");
        assert_eq!(a.pin_count(), 0, "nothing is recorded when disabled");
        assert!(matches!(a.pinned("k", &workers), PinLookup::Miss));
    }

    #[test]
    fn pin_hit_returns_the_same_worker() {
        let a = MultimodalAffinity::with_clock(
            Duration::from_secs(60),
            Arc::new(MockClock::new(Instant::now())),
        );
        let w0 = worker("http://w0");
        let w1 = worker("http://w1");
        let workers = vec![Arc::clone(&w0), Arc::clone(&w1)];

        assert!(
            matches!(a.pinned("k", &workers), PinLookup::Miss),
            "no pin yet"
        );
        a.record("k", "http://w1");
        let PinLookup::Hit(w) = a.pinned("k", &workers) else {
            panic!("expected a pin hit");
        };
        assert_eq!(w.url, "http://w1");
    }

    /// A pinned worker that is not on offer must report Unavailable, NOT Miss.
    ///
    /// The caller keys the keep-or-overwrite decision on that difference:
    /// admission hides workers that are merely at their in-flight cap, and
    /// treating that as "no pin" would permanently migrate a busy conversation
    /// off the worker holding its KV.
    #[test]
    fn pin_to_unavailable_worker_reports_unavailable_not_miss() {
        let a = MultimodalAffinity::with_clock(
            Duration::from_secs(60),
            Arc::new(MockClock::new(Instant::now())),
        );
        a.record("k", "http://not-offered");
        assert!(matches!(
            a.pinned("k", &[worker("http://w0")]),
            PinLookup::Unavailable
        ));
        // The pin survives, so a later turn can come home.
        assert!(matches!(
            a.pinned("k", &[worker("http://not-offered")]),
            PinLookup::Hit(_)
        ));
    }

    /// A pin whose worker stays unavailable past the grace window is DROPPED,
    /// so the conversation re-pins instead of routing cold every turn until the
    /// pin goes idle. This is what bounds the cost of a rolling restart, where
    /// the pinned worker is gone for good rather than briefly saturated.
    #[test]
    fn pin_unavailable_past_the_grace_window_is_dropped() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        // Idle far exceeds the grace window, so the grace is what fires here —
        // not idle eviction.
        let a = MultimodalAffinity::with_clock(Duration::from_secs(1800), clock.clone());
        a.record("k", "http://departed");
        assert_eq!(a.pin_count(), 1);

        // The grace clock starts at the FIRST lookup that finds the worker
        // missing, not at `record` — so this call is what arms it.
        assert!(matches!(
            a.pinned("k", &[worker("http://w0")]),
            PinLookup::Unavailable
        ));
        // Still inside the window: this could be saturation, and the
        // conversation must be able to come home.
        clock.advance(UNAVAILABLE_GRACE / 2);
        assert!(matches!(
            a.pinned("k", &[worker("http://w0")]),
            PinLookup::Unavailable
        ));
        assert_eq!(a.pin_count(), 1, "pin must survive inside the grace window");

        // Past it, the pin goes and the caller is told to select normally.
        clock.advance(UNAVAILABLE_GRACE);
        assert!(matches!(
            a.pinned("k", &[worker("http://w0")]),
            PinLookup::Miss
        ));
        assert_eq!(
            a.pin_count(),
            0,
            "pin must be dropped past the grace window"
        );
    }

    /// The grace window measures CONTINUOUS unavailability: a worker that comes
    /// back resets it, so intermittent saturation spread over a long
    /// conversation never looks like a departure.
    #[test]
    fn a_hit_resets_the_unavailable_grace() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let a = MultimodalAffinity::with_clock(Duration::from_secs(1800), clock.clone());
        let offered = vec![worker("http://w0")];
        a.record("k", "http://w0");

        for _ in 0..3 {
            // Saturated for most of the window...
            clock.advance(UNAVAILABLE_GRACE - Duration::from_secs(1));
            assert!(matches!(
                a.pinned("k", &[worker("http://other")]),
                PinLookup::Unavailable
            ));
            // ...then comes home, which must clear the clock.
            assert!(matches!(a.pinned("k", &offered), PinLookup::Hit(_)));
        }
        assert_eq!(
            a.pin_count(),
            1,
            "repeated recoveries must never accumulate into a drop"
        );
    }

    /// Idle pins are swept, and a HIT refreshes the timer so an active
    /// conversation is never evicted mid-flight.
    #[test]
    fn idle_pins_are_swept_but_active_ones_survive() {
        let clock = Arc::new(MockClock::new(Instant::now()));
        let a = MultimodalAffinity::with_clock(Duration::from_secs(60), clock.clone());
        let workers = vec![worker("http://w0")];

        a.record("idle", "http://w0");
        a.record("active", "http://w0");

        clock.advance(Duration::from_secs(40));
        assert!(
            matches!(a.pinned("active", &workers), PinLookup::Hit(_)),
            "refreshes last_seen"
        );

        clock.advance(Duration::from_secs(40)); // idle: 80s, active: 40s
        assert_eq!(a.sweep_expired(), 1);
        assert_eq!(a.pin_count(), 1);
        assert!(matches!(a.pinned("active", &workers), PinLookup::Hit(_)));
    }
}
