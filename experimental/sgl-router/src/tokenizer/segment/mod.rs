// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Segment tokenize-cache (opt-in). A chat prompt is split at its chat-template
//! marker tokens (turn boundaries) and each segment's token ids are cached, so a
//! multi-turn conversation only re-tokenizes the newest turn. `encode_cached` is
//! **byte-identical** to encoding the whole rendered prompt in one shot.
//!
//! # Programming model
//!
//! The cache has two pluggable axes, both trait objects:
//!   * [`SegmentEncoder`] — a model's chat encoder (render + tokenize + marker
//!     candidates). Implement it to support a new model template / encoder.
//!   * [`SegmentStore`] — the storage backend (in-process today; shm / tiered
//!     later). Implement it to add a backend.
//!
//! [`SegmentCache`] owns an `Arc<dyn SegmentEncoder>` and an
//! `Arc<dyn SegmentStore>`; neither the cache logic nor the other axis changes
//! when you add one. Usage: the caller renders the prompt via the encoder, then
//! hands the rendered string to [`SegmentCache::encode_cached`].
//! [`SegmentCache::new`] uses the default [`InProcStore`];
//! [`SegmentCache::with_store`] injects a custom backend.
//!
//! Correctness rests on three things, none involving hashes:
//!   1. the encoder's tokenizer is aligned to the engine (operator's resolved
//!      tokenizer.json) — orthogonal to this layer;
//!   2. markers are AddedTokens = HuggingFace hard split boundaries, so
//!      splitting there is byte-exact;
//!   3. a startup self-check asserts `concat(encode(segment_i)) == encode(whole)`
//!      and disables segmentation (falls back to whole-prompt encode) on any
//!      mismatch.

mod cache;
mod markers;

use markers::{derive_markers, split_at_markers, Segmenter};
use std::sync::Arc;

pub use crate::config::SegmentCacheConfig;
pub use cache::{InProcStore, SegmentStore};

/// A model's chat-encoding contract, and the only thing [`SegmentCache`] depends
/// on. Implement this to add a new model template / encoder — the cache is
/// untouched. Must be thread-safe: one instance is shared across all requests.
pub trait SegmentEncoder: Send + Sync {
    /// Render a chat `messages` array into the prompt string the engine would
    /// tokenize. `None` if this encoder can't render that shape (the marker
    /// probe then skips it). Only called at [`SegmentCache::new`] (marker
    /// derivation); the request path renders once at ingress and passes the
    /// string to [`SegmentCache::encode_cached`].
    fn render(&self, messages: &serde_json::Value) -> Option<String>;

    /// Tokenize `text` to ids with `add_special_tokens = false` (specials come
    /// from the rendered text, not the tokenizer's auto-insertion).
    fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>>;

    /// Special-token strings that may appear as segment markers (turn / tool /
    /// vision framing). Typically the tokenizer's `added_tokens`; the probe
    /// filters these to the ones the encoder actually emits.
    fn marker_candidates(&self) -> &[String];
}

/// Per-model segment cache: derived markers + the [`SegmentStore`] + the safety
/// gate, plus the [`SegmentEncoder`] it was built from (used for encode on a
/// miss). Both the store and the encoder are pluggable via trait objects.
pub struct SegmentCache {
    /// Turn-boundary markers, longest-first (from [`derive_markers`]).
    markers: Vec<String>,
    /// Gate: false when there are no markers or the self-check failed. When
    /// false, `encode_cached` degrades to a plain whole-prompt encode.
    segmentable: bool,
    /// Below this many bytes, skip the cache and encode whole: for short prompts
    /// the per-segment bookkeeping outweighs re-tokenizing.
    min_bytes: usize,
    store: Arc<dyn SegmentStore>,
    encoder: Arc<dyn SegmentEncoder>,
}

impl std::fmt::Debug for SegmentCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SegmentCache")
            .field("markers", &self.markers.len())
            .field("segmentable", &self.segmentable)
            .field("min_bytes", &self.min_bytes)
            .finish_non_exhaustive()
    }
}

impl SegmentCache {
    /// Build for a model with the default in-process store ([`InProcStore`],
    /// sized by `cfg.capacity`). Convenience over [`Self::with_store`].
    pub fn new(model_id: &str, encoder: Arc<dyn SegmentEncoder>, cfg: &SegmentCacheConfig) -> Self {
        let store: Arc<dyn SegmentStore> = Arc::new(InProcStore::new(cfg.capacity));
        Self::with_store(model_id, encoder, cfg.min_bytes, store)
    }

    /// Build with an injected [`SegmentStore`] (e.g. a shared/tiered backend, or
    /// a test double). Derives markers by probe-rendering `encoder` against its
    /// marker candidates, then runs the startup self-check with `encoder.encode`.
    pub fn with_store(
        model_id: &str,
        encoder: Arc<dyn SegmentEncoder>,
        min_bytes: usize,
        store: Arc<dyn SegmentStore>,
    ) -> Self {
        let Segmenter {
            markers,
            segmentable,
        } = derive_markers(|m| encoder.render(m), encoder.marker_candidates()).unwrap_or_else(|e| {
            tracing::warn!(model = %model_id, error = %e,
                "segment-cache marker derivation failed; segmentation disabled (whole-prompt encode)");
            Segmenter {
                markers: Vec::new(),
                segmentable: false,
            }
        });
        let mut me = Self {
            markers,
            segmentable,
            min_bytes,
            store,
            encoder,
        };
        me.self_check(model_id);
        me
    }

    /// Whether segmentation is live (markers found AND self-check passed).
    pub fn is_segmentable(&self) -> bool {
        self.segmentable
    }

    /// Startup safety gate — the universal correctness oracle. Synthesize a
    /// prompt that exercises every boundary shape around each marker, then assert
    /// `concat(encode(segment_i)) == encode(whole)`. Template-independent: it
    /// validates BPE-safety of *these markers* with *this tokenizer* directly, so
    /// it catches ANY tokenizer where per-segment encoding diverges — whitespace-
    /// absorbing tokens, Metaspace/SentencePiece prefix-space, context-dependent
    /// normalizers, etc. A proven mismatch disables segmentation (stays correct,
    /// falls back to whole-prompt encode); an `encode` error leaves it enabled
    /// (can't disprove the hard-boundary invariant).
    ///
    /// The probe deliberately puts each marker in four adjacencies —
    /// `content <m>` (space before → lstrip), `<m> content` (space after →
    /// rstrip), `content<m>content` (glued), and `<m0><m1>` (adjacent markers) —
    /// so a boundary that isn't byte-exact reliably shows up.
    fn self_check(&mut self, model_id: &str) {
        if !self.segmentable {
            return;
        }
        let fillers = [
            "hello world",
            "the quick brown fox jumps",
            "你好,世界",
            "def f(x): return x + 1",
        ];
        let mut probe = String::from("preamble text ");
        for (i, m) in self.markers.iter().enumerate() {
            let f = fillers[i % fillers.len()];
            // "<f> <m> <f><m><f> ": space-before, space-after, and glued-both-sides.
            probe.push_str(f);
            probe.push(' ');
            probe.push_str(m);
            probe.push(' ');
            probe.push_str(f);
            probe.push_str(m);
            probe.push_str(f);
            probe.push(' ');
        }
        // Exercise two adjacent markers (no content between) if we have them.
        if self.markers.len() >= 2 {
            probe.push_str(&self.markers[0]);
            probe.push_str(&self.markers[1]);
            probe.push_str("tail");
        }

        let whole = match self.encoder.encode(&probe) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(model = %model_id, error = %e,
                    "segment-cache self-check: probe encode failed; leaving segmentation enabled (relying on added-token hard-boundary invariant)");
                return;
            }
        };
        let mut seg = Vec::with_capacity(whole.len());
        for s in split_at_markers(&probe, &self.markers) {
            match self.encoder.encode(s) {
                Ok(v) => seg.extend(v),
                Err(e) => {
                    tracing::debug!(model = %model_id, error = %e,
                        "segment-cache self-check: segment encode failed; leaving segmentation enabled");
                    return;
                }
            }
        }
        if whole != seg {
            tracing::warn!(model = %model_id, markers = self.markers.len(),
                "segment-cache self-check FAILED (segmented != whole); disabling segmentation for this model (whole-prompt encode)");
            self.segmentable = false;
        } else {
            tracing::info!(model = %model_id, markers = self.markers.len(),
                "segment-cache enabled (self-check passed)");
        }
    }

    /// Encode a rendered prompt with segment caching. **Byte-identical** to
    /// `encoder.encode(rendered)`; only faster when segments repeat across
    /// requests. The caller renders `rendered` via the same encoder at ingress.
    pub fn encode_cached(&self, rendered: &str) -> anyhow::Result<Vec<u32>> {
        if !self.segmentable || rendered.len() < self.min_bytes {
            return self.encoder.encode(rendered);
        }
        let mut out: Vec<u32> = Vec::with_capacity(rendered.len() / 3 + 1);
        for seg in split_at_markers(rendered, &self.markers) {
            if let Some(ids) = self.store.get(seg) {
                out.extend_from_slice(&ids);
            } else {
                let ids = self.encoder.encode(seg)?;
                self.store.insert(seg, Arc::from(ids.as_slice()));
                out.extend_from_slice(&ids);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Generic test [`SegmentEncoder`] backed by two closures + a marker set.
    struct TestEncoder<R, E> {
        render_fn: R,
        encode_fn: E,
        markers: Vec<String>,
    }

    impl<R, E> SegmentEncoder for TestEncoder<R, E>
    where
        R: Fn(&Value) -> Option<String> + Send + Sync,
        E: Fn(&str) -> anyhow::Result<Vec<u32>> + Send + Sync,
    {
        fn render(&self, m: &Value) -> Option<String> {
            (self.render_fn)(m)
        }
        fn encode(&self, t: &str) -> anyhow::Result<Vec<u32>> {
            (self.encode_fn)(t)
        }
        fn marker_candidates(&self) -> &[String] {
            &self.markers
        }
    }

    fn enc_arc<R, E>(render_fn: R, encode_fn: E, markers: Vec<String>) -> Arc<dyn SegmentEncoder>
    where
        R: Fn(&Value) -> Option<String> + Send + Sync + 'static,
        E: Fn(&str) -> anyhow::Result<Vec<u32>> + Send + Sync + 'static,
    {
        Arc::new(TestEncoder {
            render_fn,
            encode_fn,
            markers,
        })
    }

    /// Stand-in Qwen-style chat encoder for tests: wraps each message in
    /// `<|im_start|>` / `<|im_end|>` framing so both are derived as markers.
    fn qwen_render(msgs: &Value) -> Option<String> {
        let mut s = String::new();
        for m in msgs.as_array()? {
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let c = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
            s.push_str(&format!("<|im_start|>{role}\n{c}<|im_end|>\n"));
        }
        Some(s)
    }

    /// A deterministic byte-level fake encoder: one id per UTF-8 byte. Trivially
    /// split-safe (`concat(enc(parts)) == enc(whole)` always), so it exercises the
    /// cache plumbing / segmentation independent of a real BPE.
    fn byte_encode(s: &str) -> anyhow::Result<Vec<u32>> {
        Ok(s.bytes().map(|b| b as u32).collect())
    }

    fn cfg() -> SegmentCacheConfig {
        SegmentCacheConfig {
            capacity: 1024,
            min_bytes: 0, // exercise the cache path even for tiny prompts in tests
        }
    }

    fn added() -> Vec<String> {
        vec!["<|im_start|>".to_string(), "<|im_end|>".to_string()]
    }

    fn qwen_cache() -> SegmentCache {
        SegmentCache::new("m", enc_arc(qwen_render, byte_encode, added()), &cfg())
    }

    #[test]
    fn cached_equals_whole_encode() {
        let sc = qwen_cache();
        assert!(sc.is_segmentable());
        let rendered =
            "<|im_start|>user\nhello there<|im_end|>\n<|im_start|>assistant\nhi<|im_end|>\n";
        let got = sc.encode_cached(rendered).unwrap();
        assert_eq!(got, byte_encode(rendered).unwrap());
    }

    #[test]
    fn second_call_hits_cache_and_matches() {
        let sc = qwen_cache();
        let r1 = "<|im_start|>user\nturn one<|im_end|>\n";
        let r2 = "<|im_start|>user\nturn one<|im_end|>\n<|im_start|>user\nturn two<|im_end|>\n";
        assert_eq!(sc.encode_cached(r1).unwrap(), byte_encode(r1).unwrap());
        // r2 reuses r1's segments from cache; result still byte-identical.
        assert_eq!(sc.encode_cached(r2).unwrap(), byte_encode(r2).unwrap());
    }

    #[test]
    fn empty_and_marker_only_and_no_content() {
        let sc = qwen_cache();
        for r in [
            "",
            "<|im_start|>",
            "<|im_start|><|im_end|>",
            "no markers here",
        ] {
            assert_eq!(
                sc.encode_cached(r).unwrap(),
                byte_encode(r).unwrap(),
                "mismatch for {r:?}"
            );
        }
    }

    #[test]
    fn self_check_disables_on_bad_encoder() {
        // An encoder whose whole-encode differs from concat-of-parts (drops the
        // first byte of every call) must trip the self-check → segmentation off.
        fn lossy(s: &str) -> anyhow::Result<Vec<u32>> {
            Ok(s.bytes().skip(1).map(|b| b as u32).collect())
        }
        let sc = SegmentCache::new("m", enc_arc(qwen_render, lossy, added()), &cfg());
        assert!(
            !sc.is_segmentable(),
            "lossy encoder must disable segmentation"
        );
        // encode_cached then just delegates to the (lossy) whole encode.
        let r = "<|im_start|>user\nhi<|im_end|>\n";
        assert_eq!(sc.encode_cached(r).unwrap(), lossy(r).unwrap());
    }

    #[test]
    fn min_bytes_bypasses_cache() {
        let big_min = SegmentCacheConfig {
            capacity: 16,
            min_bytes: 10_000,
        };
        let sc = SegmentCache::new("m", enc_arc(qwen_render, byte_encode, added()), &big_min);
        let r = "<|im_start|>user\nhi<|im_end|>\n";
        assert_eq!(sc.encode_cached(r).unwrap(), byte_encode(r).unwrap());
        // nothing cached because we bypassed the cache path.
        assert_eq!(sc.store.entry_count(), 0);
    }

    /// End-to-end differential against the REAL dynamo tokenizer (checked-in tiny
    /// byte-level BPE fixture): `encode_cached == encode(whole)` for an actual BPE
    /// with real marker splitting (cold + warm + prefix reuse). The fixture's one
    /// special added token, `<|endoftext|>`, is the turn marker.
    #[test]
    fn real_tokenizer_cached_equals_whole() {
        use crate::tokenizer::adapter;
        let tok = adapter::load("tests/fixtures/tiny_tokenizer.json").expect("load tiny fixture");
        let tok_enc = Arc::clone(&tok);
        let render = |msgs: &Value| {
            let mut s = String::new();
            for m in msgs.as_array().unwrap() {
                s.push_str("<|endoftext|>");
                s.push_str(m.get("content").and_then(|c| c.as_str()).unwrap_or(""));
            }
            Some(s)
        };
        let sc = SegmentCache::new(
            "tiny",
            enc_arc(
                render,
                move |s: &str| adapter::encode(&tok_enc, s),
                vec!["<|endoftext|>".to_string()],
            ),
            &cfg(),
        );
        assert!(
            sc.is_segmentable(),
            "marker <|endoftext|> should be derived and self-check should pass"
        );

        let rendered =
            "<|endoftext|>hello there friend<|endoftext|>how are you<|endoftext|>goodbye now";
        let whole = adapter::encode(&tok, rendered).unwrap();
        assert_eq!(sc.encode_cached(rendered).unwrap(), whole); // cold
        assert_eq!(sc.encode_cached(rendered).unwrap(), whole); // warm

        let extended = format!("{rendered}<|endoftext|>one more turn here");
        assert_eq!(
            sc.encode_cached(&extended).unwrap(),
            adapter::encode(&tok, &extended).unwrap()
        );
    }

    #[test]
    fn concurrent_encode_matches_reference() {
        use std::thread;
        let sc = Arc::new(qwen_cache());
        let inputs: Vec<String> = (0..32)
            .map(|i| {
                format!(
                    "<|im_start|>user\nmsg {i}<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n"
                )
            })
            .collect();
        let expected: Vec<Vec<u32>> = inputs.iter().map(|s| byte_encode(s).unwrap()).collect();
        let mut handles = Vec::new();
        for (i, s) in inputs.into_iter().enumerate() {
            let sc = Arc::clone(&sc);
            handles.push(thread::spawn(move || (i, sc.encode_cached(&s).unwrap())));
        }
        for h in handles {
            let (i, got) = h.join().unwrap();
            assert_eq!(got, expected[i], "concurrent mismatch at {i}");
        }
    }
}
