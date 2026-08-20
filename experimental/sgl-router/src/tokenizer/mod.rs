// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

pub mod adapter;
pub mod chat_template;
pub mod dsv4;
pub mod kimi_k3;
pub mod kimi_vocab;
pub(crate) mod pyjson;

use anyhow::Result;
use chat_template::ChatTemplate;
use dashmap::DashMap;
use dynamo_tokenizers::Tokenizer;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// One piece of a rendered prompt, together with how it must be tokenized.
///
/// Most encoders produce a single string in which every `<|marker|>` was put
/// there by the encoder itself, so promoting all of them to special ids is
/// correct. The Kimi-K3 encoder does not: it interleaves structural markers with
/// client-supplied text, and the engine encodes the two differently — markers
/// with specials recognized, client text with specials DISABLED, so a literal
/// `<|open|>` in a prompt can never become a control token. Preserving that
/// split is what makes the router's ids match the engine's for such prompts.
/// Construct via [`Segment::marker`] / [`Segment::client_text`]. Note what the
/// private fields do and do not buy: they stop callers OUTSIDE this module from
/// marking client-supplied text as a marker, but `kimi_k3` is a CHILD module and
/// so can write the struct literal directly — its tests already read the private
/// field. The load-bearing guard is [`Segment::marker`]'s `&'static str`, which
/// a runtime `String` cannot reach even from inside this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    text: String,
    allow_special: bool,
}

impl Segment {
    /// Structure the ENCODER emitted. Tokenized with special tokens recognized.
    ///
    /// `&'static str` is the guard, not a convenience: client text is always a
    /// runtime `String`, so it cannot be passed here even by mistake.
    pub fn marker(text: &'static str) -> Self {
        Segment {
            text: text.to_string(),
            allow_special: true,
        }
    }

    /// Text that came from the CLIENT, or that the encoder formatted from client
    /// data. Tokenized with special tokens disabled.
    pub fn client_text(text: impl Into<String>) -> Self {
        Segment {
            text: text.into(),
            allow_special: false,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    /// Whether this segment tokenizes with special tokens recognized. Read only
    /// by [`kimi_vocab::KimiVocab::encode_segments`] — the single interpreter.
    pub fn allows_special(&self) -> bool {
        self.allow_special
    }

    /// Rebuild a segment from a parity fixture, which stores the raw flag.
    ///
    /// Test-only, and deliberately the one way to set the flag from data: the
    /// fixtures are generated from the Python reference, so this is the only
    /// place the flag legitimately comes from outside the encoder.
    #[cfg(test)]
    pub(crate) fn from_fixture(text: String, allow_special: bool) -> Self {
        Segment {
            text,
            allow_special,
        }
    }
}

/// How to render a chat request, for every built-in encoder.
///
/// Bundled rather than per-encoder because the ingress resolves options once,
/// before it knows (or cares) which encoder the model uses; each encoder reads
/// only its own half.
///
/// The unused half is cheap but NOT free: resolving it is a handful of JSON
/// lookups, plus a clone of `response_format` when the request carries one. Both
/// halves must therefore stay side-effect free — an earlier version logged the
/// K3 defaults during resolution, which made a DeepSeek-only router announce
/// Kimi settings on its first request. Per-encoder side effects belong at
/// encoder-attach time (see `resolve_chat_encoder`), not here.
///
/// If a third encoder lands, prefer passing the request down and letting each
/// encoder resolve its own options over growing this struct.
#[derive(Clone, Debug, Default)]
pub struct ChatRenderOptsDsv4Parts {
    /// See [`dsv4::RequestParts::task`]. Owned rather than borrowed: resolved
    /// options outlive the request body they came from (the cache-sim tee
    /// carries them past the handler that parsed it).
    pub task: Option<String>,
    /// See [`dsv4::RequestParts::continue_final_message`].
    pub continue_final_message: bool,
}

#[derive(Clone, Debug)]
pub struct ChatRenderOpts {
    pub dsv4: dsv4::RenderOpts,
    /// Request-level dsv4 steering (`task`, `continue_final_message`) —
    /// mirrored by the built-in dsv4 encoder, ignored by Jinja and by K3.
    pub dsv4_parts: ChatRenderOptsDsv4Parts,
    pub kimi_k3: kimi_k3::RenderOpts,
}

// Deliberately NO `Default` impl: the only sensible body would be `chat()`, and
// a derived `..Default::default()` in a struct update is exactly how a request
// would silently acquire reference defaults instead of the router's resolved
// ones. Callers must choose `chat()` or `resolve()` explicitly.

impl ChatRenderOpts {
    /// Every encoder's REFERENCE default: DeepSeek-V4 in non-thinking mode with
    /// no effort preamble, Kimi-K3 at `apply_chat_template`'s own defaults.
    ///
    /// Deliberately env-independent, and therefore NOT what a request should be
    /// rendered with. Both encoders' `resolve_render_opts` fall back to the
    /// router's env defaults (`SGLANG_ROUTER_DSV4_*` / `SGLANG_ROUTER_K3_*`) for
    /// anything the request omits, so on a router where those are set this
    /// disagrees with [`ChatRenderOpts::resolve`] on an empty request — by
    /// design. Use it for fixtures and for probes that sweep modes explicitly
    /// (see [`extension_concat_safe`]); use `resolve` for anything derived from
    /// a real request.
    pub fn chat() -> Self {
        ChatRenderOpts {
            dsv4: dsv4::RenderOpts::chat(),
            dsv4_parts: ChatRenderOptsDsv4Parts::default(),
            kimi_k3: kimi_k3::RenderOpts::default(),
        }
    }

    /// Borrow the request-level dsv4 steering in the shape [`dsv4`] renders
    /// with.
    fn dsv4_parts(&self) -> dsv4::RequestParts<'_> {
        dsv4::RequestParts {
            task: self.dsv4_parts.task.as_deref(),
            continue_final_message: self.dsv4_parts.continue_final_message,
        }
    }

    /// Resolve both encoders' options from one request body.
    pub fn resolve(request: &serde_json::Value) -> Self {
        ChatRenderOpts {
            dsv4: dsv4::resolve_render_opts(request),
            // The bool is coerced the way pydantic would (`openai_bool`), not
            // `as_bool` — an engine-true `"true"`/`1` must not render as
            // router-false while the engine does the surgery.
            dsv4_parts: ChatRenderOptsDsv4Parts {
                task: request
                    .get("task")
                    .and_then(|v| v.as_str())
                    .map(str::to_owned),
                continue_final_message: request.get("continue_final_message").and_then(openai_bool)
                    == Some(true),
            },
            kimi_k3: kimi_k3::resolve_render_opts(request),
        }
    }
}

/// How to turn a chat request's `messages` into the prompt the engine tokenizes
/// and caches. Cache-aware routing renders this before hashing so its query
/// tokens match the engine's stored blocks.
pub enum ChatEncoder {
    /// HuggingFace Jinja chat template from `tokenizer_config.json` (most
    /// models). Boxed: it holds a minijinja `Environment`, far larger than the
    /// other variants.
    Jinja(Box<ChatTemplate>),
    /// DeepSeek-V4 ships no template; the engine encodes in code. See [`dsv4`].
    DeepSeekV4,
    /// Kimi-K3 ships no chat template; the prompt is built in code and tokenized
    /// segment by segment (see [`Segment`]) rather than as one string, so this
    /// variant holds a backend that implements `Encoder::encode_segments`.
    /// [`kimi_vocab::KimiVocab`] is that backend — the HF one does not implement
    /// segmented encode at all. The `Arc` is the SAME instance the registry serves
    /// from `get()`, so this costs no second copy of the vocabulary.
    KimiK3(Arc<kimi_vocab::KimiVocab>),
}

impl ChatEncoder {
    /// Render `messages` (+ the request's top-level `tools`, or `None`) into the
    /// engine-equivalent prompt text. Returns the text plus, when
    /// `continue_final_message` extracted a trailing assistant turn, its content
    /// for the caller to encode and append after the prompt ids (the engine's
    /// `_append_assistant_prefix_to_prompt_ids`).
    ///
    /// Only for the encoders whose output IS one uniformly-tokenized string;
    /// [`ChatEncoder::KimiK3`] has no such form and is handled by
    /// [`TokenizerRegistry::encode_chat`] directly.
    ///
    /// The DeepSeek-V4 encoder renders `tools` (see [`dsv4::render_messages`]) so
    /// cache-aware routing matches the engine's cached blocks for tool traffic,
    /// and it threads the request-level `parts` (`task`, `continue_final_message`;
    /// see [`dsv4::render_request`]). The Jinja path does not yet thread
    /// `tools`/`parts` (a tools-carrying request there still routes on the
    /// no-tools rendering); adding per-model Jinja tool rendering is future work.
    fn render(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        opts: &ChatRenderOpts,
    ) -> Result<(String, Option<String>)> {
        match self {
            // The Jinja path does not thread thinking-mode/tools/parts yet
            // (future work); it ignores `opts` and renders the model's default
            // template.
            ChatEncoder::Jinja(t) => t.render(messages).map(|s| (s, None)),
            ChatEncoder::DeepSeekV4 => {
                dsv4::render_request(messages, tools, opts.dsv4, opts.dsv4_parts())
                    .map_err(anyhow::Error::from)
            }
            // `encode_chat` routes K3 to the segment path before reaching here.
            // An error rather than a panic: a mis-wired caller should degrade
            // this model to raw-text routing, not take the router down.
            ChatEncoder::KimiK3(_) => Err(anyhow::anyhow!(
                "Kimi-K3 renders segments, not a string; encode_chat must take the segment path"
            )),
        }
    }

    /// Render WITHOUT request-level surgery (`task` attach, trailing-assistant
    /// handling). The surgery belongs to ingress REQUEST rendering — where a
    /// trailing assistant turn is a client continuation; a mid-conversation
    /// assistant turn must never see it — so the reply-suffix derivation and
    /// the concat self-check (which model the NEXT round's history, where the
    /// reply sits mid-conversation) render through this plain path.
    fn render_plain(
        &self,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        opts: &ChatRenderOpts,
    ) -> Result<String> {
        match self {
            ChatEncoder::Jinja(t) => t.render(messages),
            ChatEncoder::DeepSeekV4 => Ok(dsv4::render_messages(messages, tools, opts.dsv4)),
            // As in `render`: K3 has no single-string form, so the segment path
            // in `encode_chat_plain` handles it before reaching here.
            ChatEncoder::KimiK3(_) => Err(anyhow::anyhow!(
                "Kimi-K3 renders segments, not a string; encode_chat_plain must take the segment path"
            )),
        }
    }

    /// The forwarding-safety contract this encoder's ids satisfy:
    /// which `input_ids_safe_to_forward`-family predicate may gate them.
    /// `request_tokens_for` stamps it onto the ids at production time, so the
    /// provenance travels with the tokens instead of being re-derived
    /// downstream from another registry lookup.
    fn forward_parity(&self) -> ForwardParity {
        match self {
            // The Jinja encoder needs the conservative predicate (no tool /
            // thinking / task rendering — see `input_ids_safe_to_forward`).
            ChatEncoder::Jinja(_) => ForwardParity::Conservative,
            // The dsv4 encoder mirrors the engine's full dsv4 request
            // handling (`input_ids_safe_to_forward_dsv4`).
            ChatEncoder::DeepSeekV4 => ForwardParity::Dsv4Full,
            // K3 has no forwarding predicate of its own yet, and
            // `input_ids_safe_to_forward_dsv4` encodes dsv4-specific
            // reasoning about what the engine mirrors — it must not be
            // borrowed for another model. Conservative until a K3 predicate
            // exists: K3 ids still drive cache-aware routing, they are just
            // not forwarded as `input_ids`.
            ChatEncoder::KimiK3(_) => ForwardParity::Conservative,
        }
    }
}

/// Which forwarding predicate a chat encoder's ids may pass through (see
/// [`ChatEncoder::forward_parity`]). Exhaustive by construction: a new
/// encoder variant forces a decision here, not in scattered call sites.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ForwardParity {
    /// `input_ids_safe_to_forward` (tools, thinking overrides, multimodal,
    /// trailing assistant, … all withheld).
    Conservative,
    /// `input_ids_safe_to_forward_dsv4` (only genuinely unmirrored engine
    /// internals withheld).
    Dsv4Full,
}

/// Parse a JSON value the way pydantic v2 (lax mode) coerces an OpenAI
/// request's boolean field (e.g. `continue_final_message: bool = False`): a
/// bool as-is; `1`/`0` (numeric) and the strings `true/yes/on/y/t/1` → true,
/// `false/no/off/n/f/0` → false, case-insensitive. Anything else is `None` —
/// pydantic would 422 the request, and a caller deciding correctness-sensitive
/// behavior on such a value must treat it as unknown rather than silently
/// defaulting it to `false`.
pub fn openai_bool(v: &serde_json::Value) -> Option<bool> {
    match v {
        serde_json::Value::Bool(b) => Some(*b),
        serde_json::Value::Number(n) => match n.as_f64() {
            Some(1.0) => Some(true),
            Some(0.0) => Some(false),
            _ => None,
        },
        serde_json::Value::String(s) => match s.to_ascii_lowercase().as_str() {
            "true" | "yes" | "on" | "y" | "t" | "1" => Some(true),
            "false" | "no" | "off" | "n" | "f" | "0" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// A model's chat encoder plus its fallback-logging state.
struct ChatEncoderEntry {
    encoder: ChatEncoder,
    fallback_warned: AtomicBool,
    /// Lazily-computed verdict of the incremental-extension self-check (see
    /// [`extension_concat_safe`]): whether `render(msgs + [reply])` provably
    /// tokenizes to `encode(render(msgs)) ++ encode(turn suffix)` for this
    /// encoder+tokenizer pair. Computed once, on the first
    /// [`TokenizerRegistry::encode_chat_extension`] call for the model.
    extension_safe: std::sync::OnceLock<bool>,
}

impl ChatEncoderEntry {
    fn new(encoder: ChatEncoder) -> Self {
        Self {
            encoder,
            fallback_warned: AtomicBool::new(false),
            extension_safe: std::sync::OnceLock::new(),
        }
    }

    /// Log a per-request fallback to raw prompt-text hashing. "Enabled but
    /// failing every request" must be distinguishable from "healthy" at the
    /// default (info) log level — otherwise cache-aware overlap silently
    /// degrades to 0 with no signal — so the first failure for a model logs at
    /// warn; subsequent ones at debug to avoid a per-request log flood.
    fn log_fallback(&self, model_id: &str, cause: &str) {
        if !self.fallback_warned.swap(true, Ordering::Relaxed) {
            tracing::warn!(model = %model_id, %cause,
                "chat-encoder failed; falling back to raw prompt-text hashing \
                 (cache-aware overlap degrades for this model; further failures log at debug)");
        } else {
            tracing::debug!(model = %model_id, %cause,
                "chat-encoder failed; falling back to raw prompt-text hashing");
        }
    }
}

/// N independent `Tokenizer` instances for one model, selected round-robin.
///
/// WHY: `dynamo_tokenizers::Tokenizer` is `Arc<dyn traits::Tokenizer>`
/// internally; cloning it only clones the pointer, so every caller sharing
/// one instance also shares its underlying `BPE` model's word-merge cache —
/// a single `std::sync::RwLock<AHashMap<..>>` behind
/// `tokenizers::utils::cache::Cache`. Under concurrent encode() calls from
/// many tokio worker threads, that one lock becomes the bottleneck.
///
/// This is a second, complementary fix to the `dynamo-tokenizers` version
/// bump documented on that dependency in Cargo.toml (which cut this same
/// `Cache::get` frame's share of total process CPU from a measured 87% to a
/// measured 20%, on identical code/traffic — see that comment for the only
/// numbers in this repo that are independently checkable against a
/// committed artifact). A later live capture during the same investigation,
/// taken after that bump had already landed, broke the REMAINING `Cache::get`
/// time down further: ~25 of its points were specifically `RwLock::try_read`'s
/// CAS-retry loop plus `read_unlock`, not actual cache lookups — i.e. lock
/// overhead, not useful work, on that one shared lock. That capture wasn't
/// saved as a repo artifact, so treat "~25" as this investigation's own
/// finding, not a number a future reader can re-derive from anything
/// committed. Loading N instances from the same file gives each its own
/// independent cache/lock, so concurrent callers spread across N locks
/// instead of contending one. Every instance is loaded from the same source
/// with the same opts via [`adapter::load_with_opts`], so which shard a
/// call lands on can never change the tokenization output — only which lock
/// it contends.
struct TokenizerShards {
    /// Always non-empty — the type's only constructors (`load` and `shared`)
    /// both guarantee at least one element, so `next()`'s `%
    /// self.shards.len()` can never divide by zero. Never resized after
    /// construction: no method here takes `&mut self`.
    shards: Vec<Arc<Tokenizer>>,
    /// Round-robin cursor, incremented per selection. Wraps via `%
    /// shards.len()`; overflow of the counter itself is harmless (wrapping
    /// add) since only its value modulo `shards.len()` is ever read.
    /// `&self` suffices in `next()` despite mutating this — interior
    /// mutability via `AtomicUsize`, no `&mut` needed (and `&mut self` would
    /// defeat the point: `TokenizerRegistry::get` is called concurrently
    /// from every tokio worker thread, which is exactly the concurrency this
    /// type exists to spread out, not serialize on).
    next: AtomicUsize,
}

impl TokenizerShards {
    /// Load `n` independent tokenizer instances from `source`. `n` is
    /// clamped to at least 1 so a misconfigured 0 can't build an empty,
    /// unusable registry entry.
    ///
    /// Sequential by design — do not parallelize this loop. For an HF
    /// repo-id `source`, the first `adapter::load_with_opts` call downloads
    /// and populates `hf-hub`'s on-disk cache; every subsequent call in this
    /// same sequential loop then hits that cache with no network at all
    /// (`hf_hub::api::sync::ApiRepo::get` checks the cache before ever
    /// calling out). Running these N calls concurrently instead would
    /// reintroduce a real race: `ApiRepo::get`'s cache check happens before
    /// its own download lock is taken, so N concurrent first-callers could
    /// each decide "not cached" and each kick off their own download
    /// attempt, serialized only by `hf-hub`'s file lock rather than
    /// deduplicated — turning "1 download + (N-1) free cache reads" back
    /// into up to N downloads racing each other.
    fn load(source: &str, n: usize, opts: adapter::TokenizerLoadOpts) -> Result<Self> {
        // Settle the L1 half of the opts FIRST: a requested-but-inert cache
        // (tokenizer declares no safely-splittable specials) is downgraded
        // to budget 0 there, so it must not cost the shard spread below.
        let opts = adapter::finalize_load_opts(source, opts)?;
        // The L1 prefix cache lives INSIDE each CachedTokenizer instance, so
        // N shards would mean N independent caches: N× the byte budget and a
        // 1/N hit rate (a conversation's turn-2 request only hits if it lands
        // on the same shard as turn 1 — round-robin guarantees it usually
        // won't). One shared instance wins for the hit-dominated multi-turn
        // traffic the cache targets: a hit touches the BPE merge cache only
        // for the (short) fresh suffix, and with the fast backend misses
        // avoid that lock too. KNOWN TRADE-OFF: with the HF backend,
        // miss-heavy traffic (all-fresh conversations, boundary-less
        // prompts) funnels full encodes through the single instance's
        // merge-cache RwLock that sharding existed to spread — prefer
        // --tokenizer-backend fast alongside the cache.
        let n = if opts.l1_cache_bytes > 0 {
            if n > 1 {
                tracing::info!(
                    requested_shards = n,
                    "tokenizer L1 cache enabled; using a single shared tokenizer instance \
                     (per-shard caches would split the hit rate and multiply the byte budget)"
                );
            }
            1
        } else {
            n.max(1)
        };
        let shards = (0..n)
            .map(|_| adapter::load_with_opts(source, opts))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            shards,
            next: AtomicUsize::new(0),
        })
    }

    /// Pick the next shard round-robin. `Relaxed` is sufficient for this
    /// counter specifically because `shards` itself needs no ordering to
    /// piggyback on: it's fully built in `load` before a `TokenizerShards`
    /// is ever shared (`load_from_config` only inserts it into the registry
    /// `DashMap` afterward), so the `DashMap` insert/lookup that publishes
    /// this struct to other threads already provides the only cross-thread
    /// visibility this type needs — `shards.len()` and its backing
    /// allocation are immutable for the struct's entire life, never
    /// requiring synchronization of their own. `Relaxed` on `fetch_add` only
    /// has to pick a valid index into that fixed, already-visible `Vec`, and
    /// every shard is behaviorally identical (see the type-level doc
    /// comment), so any interleaving of concurrent `fetch_add`s yields a
    /// valid, if not perfectly even, distribution.
    fn next(&self) -> Arc<Tokenizer> {
        let i = self.next.fetch_add(1, Ordering::Relaxed) % self.shards.len();
        Arc::clone(&self.shards[i])
    }

    /// Wrap one already-loaded instance as a single-shard set, for a backend that
    /// gains nothing from sharding. Sharding exists to spread the HF backend's
    /// shared merge-cache lock; a backend without one gets no throughput from N
    /// instances and pays for the vocabulary N times.
    fn shared(t: Arc<Tokenizer>) -> Self {
        Self {
            shards: vec![t],
            next: AtomicUsize::new(0),
        }
    }
}

#[derive(Default)]
pub struct TokenizerRegistry {
    inner: DashMap<String, TokenizerShards>,
    /// Per-model chat encoder, present only when the model's prompt format is
    /// known (a `tokenizer_config.json` chat template, or a built-in encoder
    /// like DeepSeek-V4's). Cache-aware routing uses it to tokenize chat
    /// requests the way the engine does; models without one fall back to raw
    /// prompt-text tokenization.
    encoders: DashMap<String, Arc<ChatEncoderEntry>>,
}

impl std::fmt::Debug for TokenizerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizerRegistry")
            .field("models", &self.ids())
            .finish()
    }
}

impl TokenizerRegistry {
    pub fn load_from_config(cfg: &crate::config::Config) -> Result<Self> {
        let me = TokenizerRegistry::default();
        let m = &cfg.model;
        let opts = adapter::TokenizerLoadOpts {
            backend: m.tokenizer_backend,
            l1_cache_bytes: m.tokenizer_l1_cache_mb.saturating_mul(1024 * 1024),
        };

        // A Kimi-K3 model is loaded explicitly rather than through
        // `TokenizerShards::load`, because it needs the SEGMENTED encode path and
        // `Encoder::encode_segments` is implemented by the Baseten backend only —
        // the HF backend that `Tokenizer::from_file` picks for a `.json` inherits
        // the erroring trait default. The one instance built here backs both the
        // registry and the chat encoder; sharding it would duplicate a ~164k-entry
        // vocabulary per shard.
        let k3_tokenizer = match adapter::resolve_artifact(&m.tokenizer_path)? {
            // A raw tiktoken directory carries no `tokenizer.json`, so the K3
            // segment encoder cannot be built from it — but the vocabulary itself
            // still loads through upstream's tiktoken backend, which is what
            // Kimi-K2 and Moonlight (tiktoken-only repos, no `tokenizer.json`)
            // need. They get a working tokenizer and raw-prompt routing; only K3
            // chat encoding is unavailable, and only K3 is told where to look.
            adapter::TokenizerArtifact::TikTokenDir(dir) => {
                let tk = adapter::load_tiktoken(&dir)?;
                tracing::info!(model = %m.id, dir = %dir.display(),
                    "loaded tiktoken vocabulary (this repo ships no tokenizer.json); \
                     tokenizer sharding is not applied");
                if is_kimi_k3(&m.id) {
                    tracing::warn!(model = %m.id,
                        "model id looks like Kimi-K3 but its tokenizer path has no \
                         tokenizer.json, which the K3 segment encoder requires; chat traffic \
                         routes by raw prompt text. Point --tokenizer-path at \
                         `baseten/kimi-k3-tokenizer` to enable K3 encoding");
                }
                me.inner
                    .insert(m.id.clone(), TokenizerShards::shared(Arc::new(tk)));
                None
            }
            adapter::TokenizerArtifact::HfJson(path) if is_kimi_k3(&m.id) => {
                let source = path.to_str().unwrap_or(&m.tokenizer_path);
                // A vocabulary the Baseten backend cannot read is NOT fatal. It
                // costs K3 segment encoding — so chat traffic routes by raw text —
                // but the standard loader can still serve `/v1/tokenize` and
                // raw-prompt routing for this model, which is strictly better than
                // refusing to start. Same degrade-with-signal shape as the fastokens
                // fallback in `adapter::load_with_opts`.
                match kimi_vocab::KimiVocab::from_file(&path) {
                    Ok(bt) => {
                        let bt = Arc::new(bt);
                        tracing::info!(model = %m.id, path = %source,
                            vocab_size = bt.vocab_size(),
                            "loaded Kimi-K3 vocabulary (Baseten BPE, reference-faithful input \
                             chunking); tokenizer sharding is not applied");
                        // These knobs are properties of the HuggingFace/fast
                        // backends. Ignoring them silently would leave an operator
                        // reading a dashboard that says `backend="fast"` while none
                        // of it applies, so say so once at startup instead.
                        if m.tokenizer_backend != adapter::TokenizerBackend::default()
                            || m.tokenizer_l1_cache_mb > 0
                            || m.tokenizer_shards > 1
                        {
                            tracing::warn!(model = %m.id,
                                backend = ?m.tokenizer_backend,
                                l1_cache_mb = m.tokenizer_l1_cache_mb,
                                shards = m.tokenizer_shards,
                                "--tokenizer-backend / --tokenizer-l1-cache-mb / \
                                 --tokenizer-shards do not apply to a Kimi-K3 model and are \
                                 ignored");
                        }
                        me.inner.insert(
                            m.id.clone(),
                            TokenizerShards::shared(Arc::new(Tokenizer::from(Arc::clone(&bt)))),
                        );
                        Some(bt)
                    }
                    Err(e) => {
                        tracing::warn!(model = %m.id, path = %source, error = %format!("{e:#}"),
                            "Kimi-K3 vocabulary could not be loaded through the Baseten backend; \
                             falling back to the standard tokenizer, so chat traffic routes by \
                             raw prompt text. Point --tokenizer-path at \
                             `baseten/kimi-k3-tokenizer` to enable K3 encoding");
                        let shards = TokenizerShards::load(source, m.tokenizer_shards, opts)?;
                        me.inner.insert(m.id.clone(), shards);
                        None
                    }
                }
            }
            // Load from the RESOLVED path, not the raw config string:
            // `resolve_artifact` already turned a model DIRECTORY into the
            // `tokenizer.json` inside it, and re-resolving would hand the
            // directory itself to the HF loader, which cannot open it.
            adapter::TokenizerArtifact::HfJson(path) => {
                let source = path.to_str().unwrap_or(&m.tokenizer_path);
                let shards = TokenizerShards::load(source, m.tokenizer_shards, opts)?;
                me.inner.insert(m.id.clone(), shards);
                None
            }
        };
        // Resolve the chat encoder, best-effort: a Jinja template from
        // tokenizer_config.json, else a built-in encoder for a recognized model
        // (DeepSeek-V4), else none (chat traffic routes via raw text). Every
        // path logs its outcome — whether chat-aware routing is live for this
        // model is the single most useful signal for diagnosing "cache-aware
        // routing degraded to overlap=0 on chat traffic", so it must never be
        // silent.
        if let Some(encoder) = me.resolve_chat_encoder(&m.id, &m.tokenizer_path, k3_tokenizer) {
            me.encoders
                .insert(m.id.clone(), Arc::new(ChatEncoderEntry::new(encoder)));
        }
        Ok(me)
    }

    /// Pick the chat encoder for a model, logging the outcome on every branch.
    ///
    /// `k3` is the Baseten-backed vocabulary loaded for a Kimi-K3 model; it is
    /// what makes the K3 encoder available.
    fn resolve_chat_encoder(
        &self,
        model_id: &str,
        tokenizer_path: &str,
        k3: Option<Arc<kimi_vocab::KimiVocab>>,
    ) -> Option<ChatEncoder> {
        // Checked before the Jinja template: K3 ships a `tokenizer_config.json`
        // with no `chat_template`, so the template branch would fall through
        // anyway — but ordering it first keeps "we loaded a K3 vocabulary for
        // this model" and "we use the matching encoder" in one place.
        if is_kimi_k3(model_id) {
            // Every outcome sets the resolved state, not just the failures: an
            // exported "active" is what makes its absence alertable. A warn alone
            // is invisible after log rotation, and this degradation is permanent
            // for the process — see `adapter::k3_encoder_state`.
            let Some(tk) = k3 else {
                // The load already warned with the underlying cause; do not invent
                // a second, different explanation for the same state here.
                tracing::warn!(model = %model_id,
                    "Kimi-K3 encoding is unavailable for this model (see the preceding \
                     tokenizer warning for why); chat traffic routes by raw prompt text");
                adapter::note_k3_encoder(adapter::K3EncoderState::VocabUnavailable);
                return None;
            };
            // Loading is NOT sufficient: without the structural markers the
            // encoder would emit a prompt containing no control tokens at all,
            // then mark it engine-equivalent and forward it. Raw-text routing is a
            // large downgrade; forwarding a structurally empty prompt is a wrong
            // answer. Refuse — and say WHICH marker failed and how, because an
            // unregistered marker and an unusable vocabulary need different fixes.
            if let Err(why) = kimi_k3::markers_resolve(tk.as_ref()) {
                tracing::error!(model = %model_id, error = %format!("{why:#}"),
                    markers = ?kimi_k3::CONTROL_MARKERS,
                    "Kimi-K3 vocabulary does not resolve the XTML control markers to single \
                     ids; K3 encoding is DISABLED and chat traffic routes by raw prompt text. \
                     Point --tokenizer-path at `baseten/kimi-k3-tokenizer`");
                adapter::note_k3_encoder(adapter::K3EncoderState::MarkersUnresolved);
                return None;
            }
            tracing::info!(model = %model_id,
                "Kimi-K3 routing enabled; chat requests route via the built-in K3 XTML \
                 encoder over the Baseten-backed vocabulary");
            kimi_k3::log_defaults();
            adapter::note_k3_encoder(adapter::K3EncoderState::Active);
            return Some(ChatEncoder::KimiK3(tk));
        }
        match adapter::load_tokenizer_config(tokenizer_path) {
            Ok(Some(cfg_json)) => match ChatTemplate::from_tokenizer_config(&cfg_json) {
                Ok(Some(tmpl)) => {
                    tracing::info!(model = %model_id,
                        "chat-template routing enabled; chat requests route by templated tokens");
                    return Some(ChatEncoder::Jinja(Box::new(tmpl)));
                }
                Ok(None) => {} // no template — fall through to built-in detection
                Err(e) => tracing::warn!(model = %model_id, error = %e,
                    "failed to compile chat template; falling back to built-in detection"),
            },
            Ok(None) => {}
            Err(e) => tracing::warn!(model = %model_id, error = %e,
                "failed to load tokenizer_config.json; falling back to built-in detection"),
        }
        if is_deepseek_v4(model_id) {
            tracing::info!(model = %model_id,
                "DeepSeek-V4 routing enabled; chat requests route via the built-in V4 encoder");
            return Some(ChatEncoder::DeepSeekV4);
        }
        tracing::info!(model = %model_id,
            "no chat template or built-in encoder; chat traffic routes via raw prompt text");
        None
    }

    /// Return one of this model's tokenizer shards, round-robin. Every call
    /// may return a *different* `Arc<Tokenizer>` instance than the previous
    /// one for the same model — callers must not rely on pointer identity
    /// across calls (see [`TokenizerShards`] for why: spreading callers
    /// across N independently-locked instances is the whole point).
    pub fn get(&self, model_id: &str) -> Option<Arc<Tokenizer>> {
        self.inner.get(model_id).map(|r| r.next())
    }

    /// Whether this model has a chat encoder (and thus the chat-aware
    /// tokenization path is available for it).
    pub fn has_chat_encoder(&self, model_id: &str) -> bool {
        self.encoders.contains_key(model_id)
    }

    /// This model's chat encoder's forwarding parity ([`ForwardParity`]),
    /// `Conservative` when the model has no encoder at all (whose ids are
    /// never engine-equivalent anyway).
    pub fn forward_parity(&self, model_id: &str) -> ForwardParity {
        self.encoders
            .get(model_id)
            .map(|e| e.encoder.forward_parity())
            .unwrap_or(ForwardParity::Conservative)
    }

    /// Render `messages` through the model's chat encoder and tokenize the
    /// result, by one of two paths depending on the encoder:
    ///
    /// - **String** (Jinja, DeepSeek-V4): render to text, then tokenize it the
    ///   same way the engine does (`add_special_tokens = false`, so the
    ///   encoder's literal `bos_token`/role markers carry the specials).
    /// - **Segments** (Kimi-K3): render to [`Segment`]s and tokenize each with
    ///   specials recognized or disabled per segment. There is no intermediate
    ///   string, which is the whole point — a literal `<|open|>` in client text
    ///   must not become a control token.
    ///
    /// `opts.dsv4_parts` carries the request-level dsv4 steering (`task`,
    /// `continue_final_message`; ignored on the Jinja and K3 paths).
    ///
    /// Returns `None` — caller falls back to raw routing — when the model has no
    /// encoder, no tokenizer, or rendering/encoding fails or yields no tokens.
    pub fn encode_chat(
        &self,
        model_id: &str,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        opts: &ChatRenderOpts,
    ) -> Option<Vec<u32>> {
        // Clone the Arc and drop the DashMap guard before the CPU-bound
        // render+encode (mirrors `get`), so no shard read-lock is held across it.
        let entry = Arc::clone(&*self.encoders.get(model_id)?);
        // Listed explicitly, not `_`: a future segment-based encoder must break
        // THIS match too, not silently fall into the string path where it would
        // fail every request behind one warn.
        match &entry.encoder {
            // Segment-wise: each piece carries whether special tokens are
            // recognized in it, and only the K3 tokenizer can honor that
            // distinction — so this path never goes through `adapter::encode`.
            // K3 does no request-level surgery, so there is no assistant prefix
            // to append after the prompt ids either.
            ChatEncoder::KimiK3(tk) => {
                let segments = kimi_k3::render_segments(messages, tools, &opts.kimi_k3)
                    .inspect_err(|e| entry.log_fallback(model_id, &format!("render failed: {e:#}")))
                    .ok()?;
                return match tk.encode_segments(&segments) {
                    Ok(ids) if !ids.is_empty() => Some(ids),
                    Ok(_) => {
                        entry.log_fallback(model_id, "rendered prompt tokenized to zero tokens");
                        None
                    }
                    Err(e) => {
                        entry.log_fallback(model_id, &format!("tokenize failed: {e:#}"));
                        None
                    }
                };
            }
            ChatEncoder::Jinja(_) | ChatEncoder::DeepSeekV4 => {}
        }

        let tokenizer = self.get(model_id)?;
        let (rendered, assistant_prefix) = entry
            .encoder
            .render(messages, tools, opts)
            .inspect_err(|e| {
                // A dsv4 RenderErr is a REQUEST error the engine would also
                // reject (invalid task / task without user), not encoder
                // breakage — it must not consume the model's one-shot
                // broken-encoder WARN latch (the engine-facing story continues
                // at the request level: the engine rejects it identically).
                if e.downcast_ref::<dsv4::RenderErr>().is_some() {
                    tracing::debug!(model = %model_id, error = %e,
                        "dsv4 request-level render error (engine-invalid request)");
                } else {
                    // `{e:#}` prints the full anyhow chain, so the underlying
                    // minijinja cause (e.g. a `raise_exception` message) is
                    // visible, not just the "render chat template" context.
                    entry.log_fallback(model_id, &format!("render failed: {e:#}"))
                }
            })
            .ok()?;
        let mut ids = match adapter::encode(&tokenizer, &rendered) {
            Ok(ids) if !ids.is_empty() => ids,
            Ok(_) => {
                entry.log_fallback(model_id, "rendered prompt tokenized to zero tokens");
                return None;
            }
            Err(e) => {
                entry.log_fallback(model_id, &format!("tokenize failed: {e:#}"));
                return None;
            }
        };
        // Mirror `_append_assistant_prefix_to_prompt_ids`: the engine encodes
        // the extracted prefix (stripping a leading BOS its tokenizer may
        // have added) and appends it after the generation prompt. The router's
        // encode never adds specials — and the pinned single-user-turn engine
        // vector (`[0, 128803, …]`, a lone BOS from the literal text) proves
        // the V4 tokenizer adds none either, making the strip a no-op — so
        // this plain encode is exactly what the engine appends.
        if let Some(prefix) = assistant_prefix.filter(|p| !p.is_empty()) {
            match adapter::encode(&tokenizer, &prefix) {
                Ok(pids) if !pids.is_empty() => ids.extend(pids),
                Ok(_) => {
                    entry.log_fallback(model_id, "assistant prefix tokenized to zero tokens");
                    return None;
                }
                Err(e) => {
                    entry.log_fallback(model_id, &format!("prefix tokenize failed: {e:#}"));
                    return None;
                }
            }
        }
        Some(ids)
    }

    /// Render WITHOUT request-level surgery and tokenize — the encoding of a
    /// conversation whose trailing assistant turn is HISTORY (the cache-sim
    /// extension's full-re-encode fallback reconstructing `messages + [reply]`
    /// for NEXT-round matching), not an ingress request where that turn is a
    /// client continuation. `encode_chat`'s surgery (role rewrite / prefix
    /// extraction) would corrupt exactly that trailing turn; this path keeps
    /// it an ordinary closed turn. Same fallback semantics as `encode_chat`.
    pub fn encode_chat_plain(
        &self,
        model_id: &str,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
        opts: &ChatRenderOpts,
    ) -> Option<Vec<u32>> {
        let entry = Arc::clone(&*self.encoders.get(model_id)?);
        // Same split as `encode_chat`: K3 tokenizes segments, everything else
        // renders to one string. K3 has no request-level surgery to skip, so
        // "plain" and request rendering coincide for it.
        let encoded = match &entry.encoder {
            ChatEncoder::KimiK3(tk) => {
                let segments = kimi_k3::render_segments(messages, tools, &opts.kimi_k3)
                    .inspect_err(|e| {
                        entry.log_fallback(model_id, &format!("plain render failed: {e:#}"))
                    })
                    .ok()?;
                tk.encode_segments(&segments)
            }
            ChatEncoder::Jinja(_) | ChatEncoder::DeepSeekV4 => {
                let tokenizer = self.get(model_id)?;
                let rendered = entry
                    .encoder
                    .render_plain(messages, tools, opts)
                    .inspect_err(|e| {
                        entry.log_fallback(model_id, &format!("plain render failed: {e:#}"))
                    })
                    .ok()?;
                adapter::encode(&tokenizer, &rendered)
            }
        };
        match encoded {
            Ok(ids) if !ids.is_empty() => Some(ids),
            Ok(_) => {
                entry.log_fallback(model_id, "plain render tokenized to zero tokens");
                None
            }
            Err(e) => {
                entry.log_fallback(model_id, &format!("tokenize failed: {e:#}"));
                None
            }
        }
    }

    /// Incremental chat-extension encode: the ids of `render(messages + [reply])`
    /// computed as `prompt_ids ++ encode(reply's rendered turn suffix)` —
    /// O(reply) work instead of O(whole conversation). The cache-sim response
    /// tee calls this once per completed response, so at production rates the
    /// full-re-encode alternative would roughly double the router's
    /// tokenization CPU; this path makes the steady-state cost proportional to
    /// the generated output only.
    ///
    /// `prompt_ids` MUST be this request's ingress [`Self::encode_chat`] output
    /// (engine-equivalent ids) — never the raw-prompt fallback tokenization,
    /// which renders differently and would produce a garbage concatenation.
    ///
    /// Correctness rests on two per-model properties: the encoder renders a
    /// trailing assistant turn concatenatively, and the tokenizer never
    /// BPE-merges across the prompt/turn boundary (DSV4's prompt render ends
    /// in `<think>`/`</think>`, added tokens merges can't cross). Both are
    /// proven together by a one-time end-to-end self-check
    /// ([`extension_concat_safe`], cached on the encoder entry; it probes
    /// chat, thinking, and effort-preamble modes, with and without tools).
    /// Returns `None` — caller falls back to a full re-encode — when the model
    /// has no encoder, the self-check failed, or the suffix render/encode
    /// fails.
    ///
    /// `opts` MUST be the same [`ChatRenderOpts`] the ingress resolved for this
    /// request ([`ChatRenderOpts::resolve`]): DSV4 thinking mode changes whether
    /// the reply's `reasoning_content` renders into the turn suffix.
    pub fn encode_chat_extension(
        &self,
        model_id: &str,
        prompt_ids: &[u32],
        reply: &serde_json::Value,
        opts: &ChatRenderOpts,
    ) -> Option<Vec<u32>> {
        let entry = Arc::clone(&*self.encoders.get(model_id)?);
        let tokenizer = self.get(model_id)?;
        let safe = *entry.extension_safe.get_or_init(|| {
            let ok = extension_concat_safe(&entry.encoder, &tokenizer);
            if !ok {
                tracing::warn!(model = %model_id,
                    "chat encoder failed the incremental-extension self-check; \
                     cache-sim response extensions fall back to a full \
                     conversation re-encode (correct, but costs O(context) \
                     tokenize CPU per response instead of O(output))");
            }
            ok
        });
        if !safe {
            return None;
        }
        let suffix = assistant_turn_suffix(&entry.encoder, reply, opts)?;
        let suffix_ids = adapter::encode(&tokenizer, &suffix).ok()?;
        if suffix_ids.is_empty() {
            return None;
        }
        let mut ids = Vec::with_capacity(prompt_ids.len() + suffix_ids.len());
        ids.extend_from_slice(prompt_ids);
        ids.extend(suffix_ids);
        Some(ids)
    }

    pub fn ids(&self) -> Vec<String> {
        self.inner.iter().map(|kv| kv.key().clone()).collect()
    }

    /// Attach a chat encoder to an already-loaded model. Lets policy tests in
    /// other modules exercise the chat-aware routing path without a co-located
    /// fixture.
    #[cfg(test)]
    pub(crate) fn attach_chat_encoder_for_test(&self, model_id: &str, encoder: ChatEncoder) {
        self.encoders.insert(
            model_id.to_string(),
            Arc::new(ChatEncoderEntry::new(encoder)),
        );
    }

    /// Convenience: attach a Jinja chat encoder built from an inline
    /// `tokenizer_config.json` value.
    #[cfg(test)]
    pub(crate) fn attach_chat_template_for_test(
        &self,
        model_id: &str,
        tokenizer_config: &serde_json::Value,
    ) {
        let template = ChatTemplate::from_tokenizer_config(tokenizer_config)
            .expect("valid test chat template")
            .expect("test tokenizer_config has a chat_template");
        self.attach_chat_encoder_for_test(model_id, ChatEncoder::Jinja(Box::new(template)));
    }
}

/// The reply's rendered turn suffix: the text `render` appends for a trailing
/// assistant message under `opts`. Derived generically — render the reply as a
/// lone one-message conversation and strip the constant empty-conversation
/// prefix (BOS, effort preamble, empty system turn, …) — so every
/// [`ChatEncoder`] variant gets it without a per-variant seam. For DSV4 chat
/// mode this yields exactly `content [+ DSML tool calls] + EOS`; thinking mode
/// prepends `reasoning_content + </think>` (a trailing turn is always at/after
/// the last user index, so `drop_thinking` never strips it — same as in the
/// full conversation). Tools are irrelevant here: they render into the
/// system-turn prefix, which the strip removes. An encoder for which the
/// derivation is wrong (e.g. a Jinja template that appends a generation
/// prompt, or errors on an empty conversation) fails
/// [`extension_concat_safe`] and never takes the incremental path in
/// production.
fn assistant_turn_suffix(
    encoder: &ChatEncoder,
    reply: &serde_json::Value,
    opts: &ChatRenderOpts,
) -> Option<String> {
    // Plain render: the reply models a mid-conversation assistant turn in the
    // NEXT round's history — request-level surgery must not run here (it
    // would rewrite the trailing turn's role, producing a suffix no next
    // round ever contains).
    let empty = encoder
        .render_plain(&serde_json::Value::Array(Vec::new()), None, opts)
        .ok()?;
    let solo = encoder
        .render_plain(&serde_json::Value::Array(vec![reply.clone()]), None, opts)
        .ok()?;
    solo.strip_prefix(&empty).map(str::to_owned)
}

/// One-time per-model probe backing [`TokenizerRegistry::encode_chat_extension`]:
/// verify on boundary-stressing conversations that
/// `encode(render(msgs + [reply]))` equals
/// `encode(render(msgs)) ++ encode(assistant_turn_suffix(reply))` — the exact
/// transformation the incremental path applies, checked end-to-end, so it
/// subsumes both of its assumptions at once (concatenative turn rendering AND
/// no BPE merge across the prompt/turn boundary). Probes cross every render
/// mode (chat / thinking / thinking+max-effort, since thinking changes the
/// boundary token from `</think>` to `<think>` and what a turn renders) and
/// tools presence (tools flip DSV4's `drop_thinking`, and the suffix render
/// never sees the request's tools) against merge-prone reply openings
/// (letter / space / newline / punctuation), a tool-call turn, a
/// reasoning-carrying turn, and a multi-turn history. Probabilistic in
/// principle (BPE merges are local, so a boundary that survives these
/// openings has no merge rule crossing it in practice) and exact for DSV4,
/// whose prompt render ends in the added tokens `<think>`/`</think>` that
/// merges cannot cross.
fn extension_concat_safe(encoder: &ChatEncoder, tokenizer: &Tokenizer) -> bool {
    // Probe under the profile this router actually renders with: `high` emits
    // no preamble under `preview` but the max-preview preamble under
    // `official`, so a verdict reached under the wrong profile would bless a
    // render state production never produces.
    let profile = dsv4::active_effort_profile();
    // Only the DSV4 half varies across probes: these are DSV4's render modes,
    // the Jinja path ignores `opts` entirely, and K3 fails the probe on its
    // first `render` call regardless of them.
    let dsv4_variant = |dsv4| ChatRenderOpts {
        dsv4,
        ..ChatRenderOpts::chat()
    };
    let opt_variants = [
        dsv4_variant(dsv4::RenderOpts::chat()),
        dsv4_variant(dsv4::RenderOpts {
            thinking: true,
            reasoning_effort: dsv4::ReasoningEffort::None,
            reasoning_effort_profile: profile,
        }),
        // High renders identically to None today, but it is a distinct engine
        // state — probe it so an engine build that gives `high` its own
        // rendering can't be silently blessed by a verdict that never saw it.
        dsv4_variant(dsv4::RenderOpts {
            thinking: true,
            reasoning_effort: dsv4::ReasoningEffort::High,
            reasoning_effort_profile: profile,
        }),
        dsv4_variant(dsv4::RenderOpts {
            thinking: true,
            reasoning_effort: dsv4::ReasoningEffort::Max,
            reasoning_effort_profile: profile,
        }),
    ];
    let tools_probe = serde_json::json!([{
        "type": "function",
        "function": {"name": "probe", "parameters": {"type": "object"}},
    }]);
    let tool_variants = [None, Some(&tools_probe)];
    let bases = [
        serde_json::json!([{"role": "user", "content": "Hello there"}]),
        serde_json::json!([
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer",
             "reasoning_content": "prior reasoning"},
            {"role": "user", "content": "second question"},
        ]),
    ];
    let replies = [
        serde_json::json!({"role": "assistant", "content": "ok then"}),
        serde_json::json!({"role": "assistant", "content": " space first"}),
        serde_json::json!({"role": "assistant", "content": "\nnewline first"}),
        serde_json::json!({"role": "assistant", "content": ", punctuation"}),
        serde_json::json!({"role": "assistant", "content": "answer",
                           "reasoning_content": "let me think about this"}),
        serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
            "id": "probe_call_1", "type": "function",
            "function": {"name": "probe", "arguments": "{\"a\": 1}"},
        }]}),
    ];
    for opts in &opt_variants {
        for tools in tool_variants {
            for base in &bases {
                let Ok(prompt_text) = encoder.render_plain(base, tools, opts) else {
                    return false;
                };
                let Ok(prompt_ids) = adapter::encode(tokenizer, &prompt_text) else {
                    return false;
                };
                for reply in &replies {
                    let mut msgs = base.as_array().expect("probe base is an array").clone();
                    msgs.push(reply.clone());
                    let Ok(full_text) =
                        encoder.render_plain(&serde_json::Value::Array(msgs), tools, opts)
                    else {
                        return false;
                    };
                    let Ok(full_ids) = adapter::encode(tokenizer, &full_text) else {
                        return false;
                    };
                    let Some(suffix) = assistant_turn_suffix(encoder, reply, opts) else {
                        return false;
                    };
                    let Ok(suffix_ids) = adapter::encode(tokenizer, &suffix) else {
                        return false;
                    };
                    let concat: Vec<u32> = prompt_ids
                        .iter()
                        .chain(suffix_ids.iter())
                        .copied()
                        .collect();
                    if full_ids != concat {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Whether `model_id` denotes a DeepSeek-V4 model, which the engine encodes via
/// the built-in [`dsv4`] encoder rather than a Jinja template. Heuristic on the
/// served model id (the router has no model architecture from `/server_info`);
/// scoped to "deepseek" + "v4" so it doesn't claim V3-family models, whose
/// encoding differs.
fn is_deepseek_v4(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    id.contains("deepseek") && id.contains("v4")
}

/// Whether `model_id` denotes a Kimi-K3 model, which ships no Jinja template and
/// is encoded by [`kimi_k3`].
///
/// Heuristic on the served model id, like [`is_deepseek_v4`]. Deliberately
/// narrow: K2/K2.5 use a completely different (non-XTML) prompt format, so
/// matching bare "kimi" would render them wrong. `k3` is matched with its
/// separator so a version string like `k30` can't be mistaken for it.
///
/// A DOTTED minor version (`kimi-k3.5`) is likewise not matched. K2 → K2.5 is
/// precisely a bump across which the prompt format changed, so a new minor
/// version has to be opted into here explicitly. The cost of the false negative
/// is raw-text routing (a degradation); the cost of the false positive would be
/// rendering a different format's prompt and calling it engine-equivalent (a
/// wrong answer). This function prefers the former, as the encoder-attach path
/// does for unresolved markers.
fn is_kimi_k3(model_id: &str) -> bool {
    let id = model_id.to_ascii_lowercase();
    if !id.contains("kimi") {
        return false;
    }
    ["k3", "k-3", "k_3"].iter().any(|marker| {
        id.match_indices(marker).any(|(i, m)| {
            let after = id[i + m.len()..].chars().next();
            after.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '.')
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::config::PolicyKind;

    fn cfg() -> crate::config::Config {
        crate::config::Config {
            server: crate::config::ServerConfig {
                host: "0".into(),
                port: 0,
                ..Default::default()
            },
            observability: Default::default(),
            model: crate::config::ModelConfig {
                id: "tiny".into(),
                tokenizer_path: "tests/fixtures/tiny_tokenizer.json".into(),
                tokenizer_shards: 1,
                tokenizer_backend: Default::default(),
                tokenizer_l1_cache_mb: 0,
                policy: PolicyKind::RoundRobin,
                circuit_breaker: None,
                cache_aware: None,
                sticky: None,
                max_output_tokens: None,
                sampling_overrides: Default::default(),
                forward_input_ids: true,
            },
            discovery: crate::config::DiscoveryBackend::StaticUrls(
                crate::config::StaticUrlsDiscoveryConfig {
                    urls: vec!["http://placeholder:0".into()],
                },
            ),
            proxy: crate::config::ProxyConfig::default(),
            active_load: crate::config::ActiveLoadConfig::default(),
            admission: crate::config::AdmissionConfig::default(),
            retry: crate::config::RetryConfig::default(),
        }
    }

    /// K2/K2.5 must NOT match: they share the "kimi" name but use a completely
    /// different prompt format, so claiming them would render every request
    /// wrong rather than merely missing an optimization.
    #[test]
    fn kimi_k3_detection_is_narrow() {
        for id in [
            "moonshotai/Kimi-K3",
            "kimi-k3",
            "Kimi_K3_Instruct",
            "org/kimi-k3-fp8",
        ] {
            assert!(is_kimi_k3(id), "{id} should be detected as Kimi-K3");
        }
        for id in [
            "moonshotai/Kimi-K2-Instruct",
            "Kimi-K2.5",
            "kimi-linear",
            "deepseek-v4-flash",
            // A longer version token that merely starts with k3.
            "kimi-k30",
            // A dotted MINOR version. K2 → K2.5 is exactly the bump across
            // which the prompt format changed, so K3.x must be opted in
            // explicitly rather than inheriting K3's encoder.
            "Kimi-K3.5",
            "moonshotai/kimi-k3.5-instruct",
        ] {
            assert!(!is_kimi_k3(id), "{id} must not be detected as Kimi-K3");
        }
    }

    /// End-to-end registry wiring for a Kimi-K3 model: the Baseten-backed
    /// vocabulary loads, `get()` serves it, and the chat encoder attaches and
    /// produces ids. Uses the committed synthetic vocabulary.
    #[test]
    fn kimi_k3_model_loads_baseten_vocab_and_attaches_encoder() {
        let mut c = cfg();
        c.model.id = "moonshotai/Kimi-K3".into();
        c.model.tokenizer_path = "src/tokenizer/testdata/kimi_k3_tiny_vocab".into();
        let r = TokenizerRegistry::load_from_config(&c).expect("load K3 registry");

        assert!(
            r.get("moonshotai/Kimi-K3").is_some(),
            "tokenizer must be served"
        );
        assert!(
            r.has_chat_encoder("moonshotai/Kimi-K3"),
            "K3 must get the built-in XTML encoder"
        );

        // The encoder must hold the SAME vocabulary instance the registry
        // serves, not a second copy — the doc on ChatEncoder::KimiK3 asserts it
        // and nothing else checks it.
        {
            let entry = r
                .encoders
                .get("moonshotai/Kimi-K3")
                .expect("K3 encoder attached");
            let ChatEncoder::KimiK3(encoder_tk) = &entry.encoder else {
                panic!("expected the KimiK3 encoder variant");
            };
            // A direct pointer comparison isn't available: the registry holds
            // `Arc<Tokenizer>`, and `Tokenizer` wraps its own `Arc<dyn ..>`, so
            // the two smart pointers have different types. Strong count is the
            // usable proxy — it is >= 2 only because the shard and the encoder
            // share ONE `Arc<KimiVocab>`. Rebuilding the vocabulary for
            // the encoder (a second load) would leave this at 1 and fail.
            assert!(
                Arc::strong_count(encoder_tk) >= 2,
                "the encoder's vocabulary must be shared with the registry shard, \
                 not a second copy (strong_count = {})",
                Arc::strong_count(encoder_tk)
            );
        }

        let messages = serde_json::json!([{"role": "user", "content": "hello world"}]);
        let ids = r
            .encode_chat(
                "moonshotai/Kimi-K3",
                &messages,
                None,
                &ChatRenderOpts::chat(),
            )
            .expect("K3 chat encoding");
        assert!(!ids.is_empty());

        // The registry's own tokenizer is the SAME vocabulary the encoder used,
        // so its raw encode of the equivalent text shares the chat ids' tail.
        let raw = adapter::encode(&r.get("moonshotai/Kimi-K3").unwrap(), "hello world")
            .expect("raw encode");
        assert!(
            ids.windows(raw.len()).any(|w| w == raw),
            "the user's text must appear verbatim inside the rendered prompt's ids"
        );
    }

    /// A model whose id looks like K3 but whose tokenizer path holds a plain
    /// `tokenizer.json` gets NO encoder — better to route by raw text than to
    /// render K3 XTML against a vocabulary that isn't K3's.
    #[test]
    fn kimi_k3_falls_back_to_the_standard_loader_when_the_vocab_is_unreadable() {
        let mut c = cfg();
        // `tiny_tokenizer.json`'s `model` object has no `type`, which the Baseten
        // backend rejects — so this drives the load-failure arm specifically.
        c.model.id = "kimi-k3-mislabelled".into();
        let r = TokenizerRegistry::load_from_config(&c).expect("must NOT refuse to start");

        // The whole point of degrading instead of bailing: the model is still
        // served. Asserting only `!has_chat_encoder` would pass even if the
        // fallback dropped the model entirely.
        let tk = r
            .get("kimi-k3-mislabelled")
            .expect("the fallback must still register a usable tokenizer");
        assert!(
            !adapter::encode(&tk, "hello")
                .expect("fallback tokenizer encodes")
                .is_empty(),
            "the fallback tokenizer must actually tokenize"
        );
        assert!(
            !r.has_chat_encoder("kimi-k3-mislabelled"),
            "K3 segment encoding is unavailable, so no encoder attaches"
        );
        // Deliberately NOT asserting `adapter::k3_encoder_state()` here.
        // It is a process-global with last-write-wins (same contract as
        // `BACKEND_STATE`/`L1_STATE`), and the sibling tests that load a K3
        // registry set it to other values on other threads — so asserting a
        // specific value here is a race, not a check. The gauge's label set is
        // covered by the /metrics render test instead.
    }

    /// A K3 model pointed at a tiktoken-only directory: the vocabulary still
    /// loads (upstream's tiktoken backend reads it), but segment encoding needs a
    /// `tokenizer.json`, so the encoder declines rather than the router refusing
    /// to start.
    #[test]
    fn kimi_k3_with_only_a_tiktoken_dir_serves_without_the_encoder() {
        let dir = std::env::temp_dir().join("sgl_router_k3_tiktoken_only");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/testdata");
        for f in ["tiktoken.model", "tokenizer_config.json"] {
            std::fs::copy(src.join("kimi_k3_tiny_vocab").join(f), dir.join(f)).unwrap();
        }
        // `from_file_auto` needs a `model_type` to pick the BPE pattern.
        std::fs::write(dir.join("config.json"), br#"{"model_type": "kimi_k3"}"#).unwrap();

        let mut c = cfg();
        c.model.id = "moonshotai/Kimi-K3".into();
        c.model.tokenizer_path = dir.to_str().unwrap().into();
        let r = TokenizerRegistry::load_from_config(&c).expect("must NOT refuse to start");
        assert!(
            r.get("moonshotai/Kimi-K3").is_some(),
            "the tiktoken vocabulary still serves /v1/tokenize and raw-prompt routing"
        );
        assert!(
            !r.has_chat_encoder("moonshotai/Kimi-K3"),
            "segment encoding needs a tokenizer.json"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A NON-K3 tiktoken-only model (Kimi-K2, Moonlight) must load exactly as it
    /// did before Kimi-K3 support existed. Regression guard: making the tiktoken
    /// arm fatal broke every one of these.
    #[test]
    fn non_k3_tiktoken_only_model_still_loads() {
        let dir = std::env::temp_dir().join("sgl_router_k2_tiktoken_only");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/testdata");
        for f in ["tiktoken.model", "tokenizer_config.json"] {
            std::fs::copy(src.join("kimi_k3_tiny_vocab").join(f), dir.join(f)).unwrap();
        }
        std::fs::write(dir.join("config.json"), br#"{"model_type": "kimi_k2"}"#).unwrap();

        let mut c = cfg();
        c.model.id = "moonshotai/Kimi-K2-Instruct".into();
        c.model.tokenizer_path = dir.to_str().unwrap().into();
        let r = TokenizerRegistry::load_from_config(&c).expect("a K2 model must still start");
        let tk = r.get("moonshotai/Kimi-K2-Instruct").expect("served");
        assert!(
            !adapter::encode(&tk, "hello").expect("encodes").is_empty(),
            "a tiktoken-only vocabulary must tokenize"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn loads_from_config() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        assert!(r.get("tiny").is_some());
        assert!(r.get("missing").is_none());
    }

    /// With `tokenizer_shards = 1` (the default `cfg()` fixture), round-robin
    /// selection over a single-element shard vec always returns the same
    /// instance, so this preserves the pre-sharding "one shared Arc per
    /// model" behavior for models that don't opt into multiple shards.
    #[test]
    fn shared_arc_per_model_with_one_shard() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let a = r.get("tiny").unwrap();
        let b = r.get("tiny").unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "with a single shard, registry should return the same Arc every call"
        );
    }

    /// The wire from `--tokenizer-l1-cache-mb` to the feature: a nonzero
    /// MiB budget flowing through `load_from_config` must actually enable
    /// the cache — observable as the shard collapse (4 configured shards →
    /// one shared instance). If the MiB→bytes conversion regressed to 0,
    /// four distinct Arcs would come back and this fails.
    #[test]
    fn load_from_config_wires_l1_cache_budget() {
        let mut c = cfg();
        c.model.tokenizer_shards = 4;
        c.model.tokenizer_l1_cache_mb = 1;
        // The BPE fixture's <|endoftext|> special keeps the cache genuinely
        // active (finalize_load_opts would zero the budget on a
        // specials-less tokenizer and the shard collapse wouldn't happen).
        c.model.tokenizer_path = "tests/fixtures/tiny_bpe_tokenizer.json".into();
        let r = TokenizerRegistry::load_from_config(&c).unwrap();
        let a = r.get("tiny").unwrap();
        let b = r.get("tiny").unwrap();
        assert!(
            Arc::ptr_eq(&a, &b),
            "an active L1 cache must collapse to one shared tokenizer instance"
        );
    }

    /// With `tokenizer_shards = N > 1`, `get` round-robins across N distinct
    /// `Arc<Tokenizer>` instances rather than always returning the same one.
    #[test]
    fn get_round_robins_across_shards() {
        let mut c = cfg();
        c.model.tokenizer_shards = 4;
        let r = TokenizerRegistry::load_from_config(&c).unwrap();

        let picks: Vec<Arc<Tokenizer>> = (0..8).map(|_| r.get("tiny").unwrap()).collect();

        // Exactly 4 distinct underlying instances, cycling with period 4.
        let distinct: std::collections::HashSet<usize> =
            picks.iter().map(|a| Arc::as_ptr(a) as usize).collect();
        assert_eq!(distinct.len(), 4, "expected exactly 4 distinct shards");
        for i in 0..4 {
            assert!(
                Arc::ptr_eq(&picks[i], &picks[i + 4]),
                "selection should cycle with period == shard count"
            );
        }
    }

    #[test]
    fn decode_complete_preserves_round_trip() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();
        let ids = adapter::encode(&t, "hello world").unwrap();
        assert!(!ids.is_empty());
        let text = adapter::decode_complete(&t, &ids, true).unwrap();
        // tiny BPE fixture is byte-level and lossless for ASCII.
        assert_eq!(text, "hello world");
    }

    /// Forces `decode_complete` through its `DecodeResult::Partial` branch.
    ///
    /// The fixture is a no-merge byte-level BPE. The 4-byte UTF-8 emoji
    /// `😀` (`\xF0\x9F\x98\x80`) encodes to its raw byte token ids:
    /// `[240, 159, 152, 128]`. Decoding only a prefix yields leading bytes
    /// that the HF adapter passes through `String::from_utf8_lossy`,
    /// producing a trailing U+FFFD. dynamo's `DecodeResult::from_decoded`
    /// then classifies that as `Partial`.
    ///
    /// Pinning the literal token ids keeps the test deterministic: if the
    /// fixture shape or upstream byte-level handling ever shifts, this fails
    /// loudly rather than silently dropping back into `Complete` and losing
    /// coverage.
    #[test]
    fn decode_complete_returns_string_on_partial_utf8() {
        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();

        // Sanity-check that the fixture still tokenises `😀` the way we
        // expect; if upstream changes this we want a loud failure here.
        let full = adapter::encode(&t, "😀").unwrap();
        assert_eq!(
            full,
            vec![240, 159, 152, 128],
            "fixture tokenisation drift: '😀' no longer encodes to [240, 159, 152, 128]"
        );

        // Feed only the first three bytes of a 4-byte UTF-8 codepoint,
        // which is incomplete.
        let s = adapter::decode_complete(&t, &full[..3], false).unwrap();

        // We pin the exact output: the lossy decoder folds the 3 leading
        // bytes into a single U+FFFD. Anything else (empty string, Err, or
        // the original bytes) would be a regression.
        assert_eq!(s, "\u{FFFD}");
    }

    /// Concurrent encode against one shared `Arc<Tokenizer>`. Pins that the
    /// registry's `Arc<Tokenizer>` is `Send + Sync` and that
    /// `dynamo_tokenizers::Tokenizer::encode` can be called concurrently
    /// without interior mutability hazards. A regression that wraps
    /// `Tokenizer` in `RefCell` / `!Sync` data would fail to compile;
    /// a regression that introduces non-thread-safe internal caches
    /// would surface as one of the tasks returning wrong ids (caught by
    /// the per-task assertion against the sequentially-computed
    /// reference).
    ///
    /// Uses a multi-thread runtime + `JoinSet` so the 10 tasks really do
    /// run in parallel on distinct worker threads — a single-thread
    /// runtime wouldn't exercise the `Sync` contract.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tokenizer_supports_concurrent_encode() {
        use tokio::task::JoinSet;

        let r = TokenizerRegistry::load_from_config(&cfg()).unwrap();
        let t = r.get("tiny").unwrap();

        // Build the reference sequentially — what each task should return.
        let inputs: Vec<String> = (0..10).map(|i| format!("hello {i}")).collect();
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| adapter::encode(&t, s).unwrap())
            .collect();

        let mut set = JoinSet::new();
        for (i, text) in inputs.into_iter().enumerate() {
            let shared = Arc::clone(&t);
            set.spawn(async move {
                let ids = adapter::encode(&shared, &text).expect("concurrent encode must not fail");
                (i, ids)
            });
        }

        let mut got: Vec<Option<Vec<u32>>> = vec![None; expected.len()];
        while let Some(joined) = set.join_next().await {
            let (i, ids) = joined.expect("task panicked");
            got[i] = Some(ids);
        }

        for (i, ids) in got.into_iter().enumerate() {
            let ids = ids.unwrap_or_else(|| panic!("task {i} did not record a result"));
            assert_eq!(
                ids, expected[i],
                "concurrent encode produced wrong tokens for task {i}; \
                 sign of a non-thread-safe internal cache regression"
            );
        }
    }

    #[test]
    fn missing_file_errors() {
        let mut c = cfg();
        c.model.tokenizer_path = "/nonexistent.json".into();
        let err = TokenizerRegistry::load_from_config(&c).unwrap_err();
        assert!(err.to_string().to_lowercase().contains("tokenizer"));
    }

    #[test]
    fn load_tokenizer_config_reads_sibling() {
        let dir = tempfile::tempdir().unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, "{}").unwrap();
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"chat_template":"X","bos_token":"<s>"}"#,
        )
        .unwrap();
        let cfg = adapter::load_tokenizer_config(tok.to_str().unwrap())
            .unwrap()
            .expect("sibling tokenizer_config.json is loaded");
        assert_eq!(cfg["chat_template"], "X");
    }

    /// A model DIRECTORY is a valid `--tokenizer-path`, and it also satisfies
    /// `looks_like_path` — so the config probe must read the file inside it,
    /// not `.parent()`'s. Getting this wrong disables chat-template routing
    /// while logging exactly what a template-less model logs, which is why the
    /// decoy config one level up is part of the fixture: the parent-probing
    /// version of this code passes a test that only asserts "not the right
    /// one is absent".
    #[test]
    fn load_tokenizer_config_reads_inside_a_directory_path() {
        let parent = tempfile::tempdir().unwrap();
        std::fs::write(
            parent.path().join("tokenizer_config.json"),
            r#"{"chat_template":"DECOY"}"#,
        )
        .unwrap();
        let model_dir = parent.path().join("model");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("tokenizer.json"), "{}").unwrap();
        std::fs::write(
            model_dir.join("tokenizer_config.json"),
            r#"{"chat_template":"REAL"}"#,
        )
        .unwrap();
        let cfg = adapter::load_tokenizer_config(model_dir.to_str().unwrap())
            .unwrap()
            .expect("the directory's own tokenizer_config.json is loaded");
        assert_eq!(cfg["chat_template"], "REAL");
    }

    #[test]
    fn load_tokenizer_config_absent_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, "{}").unwrap();
        assert!(adapter::load_tokenizer_config(tok.to_str().unwrap())
            .unwrap()
            .is_none());
    }

    /// `openai_bool` mirrors pydantic v2 lax bool coercion: bools, 0/1
    /// numeric, and the string token sets (case-insensitive); everything else
    /// is unknown, never silently defaulted.
    #[test]
    fn openai_bool_matches_pydantic_lax_coercion() {
        use serde_json::json;
        for v in [
            json!(true),
            json!("true"),
            json!("TRUE"),
            json!("yes"),
            json!("on"),
            json!("y"),
            json!("t"),
            json!("1"),
            json!(1),
            json!(1.0),
        ] {
            assert_eq!(openai_bool(&v), Some(true), "must be true: {v}");
        }
        for v in [
            json!(false),
            json!("false"),
            json!("FALSE"),
            json!("no"),
            json!("off"),
            json!("n"),
            json!("f"),
            json!("0"),
            json!(0),
            json!(0.0),
        ] {
            assert_eq!(openai_bool(&v), Some(false), "must be false: {v}");
        }
        for v in [
            json!(null),
            json!("maybe"),
            json!(" 1"),
            json!(""),
            json!(2),
            json!(0.5),
            json!(-1),
        ] {
            assert_eq!(openai_bool(&v), None, "must be unknown: {v}");
        }
    }

    /// `encode_chat` renders the template then tokenizes the result — and that
    /// token sequence differs from tokenizing the raw message content (the very
    /// reason raw-content hashing missed the engine's chat-templated blocks).
    #[test]
    fn encode_chat_renders_then_tokenizes() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        let cfg = serde_json::json!({
            "chat_template": "{{ bos_token }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}",
            "bos_token": "<s>",
        });
        reg.attach_chat_template_for_test("tiny", &cfg);
        assert!(reg.has_chat_encoder("tiny"));

        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        let chat_ids = reg
            .encode_chat("tiny", &messages, None, &ChatRenderOpts::chat())
            .expect("encode_chat");
        assert!(!chat_ids.is_empty());

        let tok = reg.get("tiny").unwrap();
        let raw_ids = adapter::encode(&tok, "hi").unwrap();
        assert_ne!(
            chat_ids, raw_ids,
            "chat-templated tokens must differ from raw-content tokens"
        );

        // encode_chat is exactly tokenize(render(messages)).
        let (rendered, _) = reg
            .encoders
            .get("tiny")
            .unwrap()
            .encoder
            .render(&messages, None, &ChatRenderOpts::chat())
            .unwrap();
        assert_eq!(chat_ids, adapter::encode(&tok, &rendered).unwrap());
    }

    /// `continue_final_message`: encode_chat's ids are exactly
    /// `encode(render of messages-minus-trailing-assistant)` followed by
    /// `encode(extracted prefix)` — the engine's prompt shape for a client
    /// continuation (prefix appended AFTER the generation prompt, never
    /// rendered into the assistant turn).
    #[test]
    fn encode_chat_appends_extracted_prefix_after_generation_prompt() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        reg.attach_chat_encoder_for_test("tiny", ChatEncoder::DeepSeekV4);

        let messages = serde_json::json!([
            {"role":"user","content":"count: 2 + 2 ="},
            {"role":"assistant","content":"4, and 3 + 3 ="}
        ]);
        let opts = ChatRenderOpts {
            dsv4_parts: ChatRenderOptsDsv4Parts {
                task: None,
                continue_final_message: true,
            },
            ..ChatRenderOpts::chat()
        };
        let got = reg
            .encode_chat("tiny", &messages, None, &opts)
            .expect("encode_chat");

        let tok = reg.get("tiny").unwrap();
        let mut want = adapter::encode(
            &tok,
            "<｜begin▁of▁sentence｜><｜User｜>count: 2 + 2 =<｜Assistant｜></think>",
        )
        .unwrap();
        want.extend(adapter::encode(&tok, "4, and 3 + 3 =").unwrap());
        assert_eq!(got, want);

        // Control: without the flag the SAME messages rewrite the trailing
        // assistant to a user turn (no prefix, different ids).
        let without = reg
            .encode_chat("tiny", &messages, None, &ChatRenderOpts::chat())
            .expect("encode_chat");
        assert_ne!(got, without);
    }

    /// The incremental extension is byte-identical to a full re-encode of
    /// `messages + [reply]` on the DSV4 encoder — the production case the
    /// O(output) path exists for — across chat and thinking modes, with and
    /// without tools. The fixture tokenizer is merge-free (byte-level, zero
    /// merges), so the self-check provably passes and the incremental path
    /// must engage (a `None` here means the self-check or suffix derivation
    /// regressed).
    #[test]
    fn encode_chat_extension_matches_full_reencode_for_dsv4() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "dsv4".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        reg.attach_chat_encoder_for_test("dsv4", ChatEncoder::DeepSeekV4);

        let messages = serde_json::json!([
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "and 3+3?"},
        ]);
        let tools = serde_json::json!([{
            "type": "function",
            "function": {"name": "add", "parameters": {"type": "object"}},
        }]);
        let opt_variants = [
            ChatRenderOpts::chat(),
            ChatRenderOpts {
                dsv4: dsv4::RenderOpts {
                    thinking: true,
                    reasoning_effort: dsv4::ReasoningEffort::Max,
                    reasoning_effort_profile: dsv4::active_effort_profile(),
                },
                ..ChatRenderOpts::chat()
            },
        ];
        let replies = [
            serde_json::json!({"role": "assistant", "content": "6, obviously"}),
            serde_json::json!({"role": "assistant", "content": "6",
                               "reasoning_content": "3+3 is 6"}),
            serde_json::json!({"role": "assistant", "content": "", "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "add", "arguments": "{\"a\":3,\"b\":3}"},
            }]}),
        ];
        for opts in &opt_variants {
            for tools in [None, Some(&tools)] {
                let prompt_ids = reg
                    .encode_chat("dsv4", &messages, tools, opts)
                    .expect("encode_chat");
                for reply in &replies {
                    let inc = reg
                        .encode_chat_extension("dsv4", &prompt_ids, reply, opts)
                        .expect("dsv4 + merge-free fixture must take the incremental path");
                    let mut msgs = messages.as_array().unwrap().clone();
                    msgs.push(reply.clone());
                    // The comparison target is the PLAIN reconstruction —
                    // the reply is next-round history, not an ingress
                    // continuation (request surgery must not run on it).
                    let full = reg
                        .encode_chat_plain("dsv4", &serde_json::Value::Array(msgs), tools, opts)
                        .expect("full re-encode");
                    assert_eq!(
                        inc, full,
                        "incremental extension must be byte-identical to a full \
                         re-encode for {reply} (thinking={})",
                        opts.dsv4.thinking,
                    );
                }
            }
        }
    }

    /// A model without a chat encoder can't extend incrementally — `None`
    /// sends the caller down the full-re-encode fallback.
    #[test]
    fn encode_chat_extension_none_without_encoder() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        let reply = serde_json::json!({"role": "assistant", "content": "x"});
        assert!(reg
            .encode_chat_extension("tiny", &[1, 2], &reply, &ChatRenderOpts::chat())
            .is_none());
    }

    /// An encoder that can't render the self-check probes (here: every render
    /// raises) fails `extension_concat_safe`, so the extension refuses the
    /// incremental path rather than concatenating garbage.
    #[test]
    fn encode_chat_extension_none_when_self_check_fails() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        reg.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('nope') }}",
                "bos_token": "<s>",
            }),
        );
        let reply = serde_json::json!({"role": "assistant", "content": "x"});
        assert!(
            reg.encode_chat_extension("tiny", &[1, 2], &reply, &ChatRenderOpts::chat())
                .is_none(),
            "a failing self-check must force the full-re-encode fallback"
        );
    }

    #[test]
    fn encode_chat_none_without_template() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        assert!(!reg.has_chat_encoder("tiny"));
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        assert!(reg
            .encode_chat("tiny", &messages, None, &ChatRenderOpts::chat())
            .is_none());
    }

    /// A template that fails to render (here, one that calls `raise_exception`)
    /// makes `encode_chat` return `None`, so the policy falls back to the raw
    /// prompt-text path rather than failing the request.
    #[test]
    fn encode_chat_none_on_render_failure() {
        let reg = TokenizerRegistry::default();
        reg.inner.insert(
            "tiny".into(),
            TokenizerShards::shared(adapter::load("tests/fixtures/tiny_tokenizer.json").unwrap()),
        );
        reg.attach_chat_template_for_test(
            "tiny",
            &serde_json::json!({
                "chat_template": "{{ raise_exception('nope') }}",
                "bos_token": "<s>",
            }),
        );
        assert!(reg.has_chat_encoder("tiny"));
        let messages = serde_json::json!([{"role":"user","content":"hi"}]);
        assert!(
            reg.encode_chat("tiny", &messages, None, &ChatRenderOpts::chat())
                .is_none(),
            "a failing render must yield None so routing falls back to raw text"
        );
    }

    #[test]
    fn is_deepseek_v4_matches_v4_only() {
        assert!(is_deepseek_v4("deepseek-ai/DeepSeek-V4-Flash"));
        assert!(is_deepseek_v4("DeepSeek-V4-Pro"));
        // Not V4-family models.
        assert!(!is_deepseek_v4("deepseek-ai/DeepSeek-V3.2"));
        assert!(!is_deepseek_v4("Qwen/Qwen3-0.6B"));
        assert!(!is_deepseek_v4("tiny"));
    }

    /// Find a real, non-trivial `tokenizer.json` in the local HuggingFace
    /// cache, if any is present. Returns `None` (the test that calls this
    /// skips itself) rather than failing on machines/CI runners with no HF
    /// cache populated — this test's value is in exercising a large,
    /// real-world BPE vocab/merge table, not in requiring network access or a
    /// specific model to be cached.
    fn find_cached_real_tokenizer_json() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let hub = std::path::Path::new(&home).join(".cache/huggingface/hub");
        for entry in std::fs::read_dir(&hub).ok()?.flatten() {
            let snapshots = entry.path().join("snapshots");
            for snap in std::fs::read_dir(&snapshots).ok()?.flatten() {
                let candidate = snap.path().join("tokenizer.json");
                if candidate.is_file() {
                    return Some(candidate);
                }
            }
        }
        None
    }

    /// Sharding must never change tokenization output: N independently
    /// loaded instances of the same `tokenizer.json` are (by construction)
    /// identical apart from their private, output-invisible merge caches, but
    /// this pins that empirically rather than by argument alone — a
    /// regression here (e.g. a shared mutable default inside the BPE model,
    /// or a loader that isn't actually deterministic) would silently corrupt
    /// downstream cache-affinity hashing, which requires byte-for-byte
    /// identical token ids across shards.
    ///
    /// Prefers a real, large tokenizer.json from the local HF cache when one
    /// is present (extra real-world coverage in local dev), but the fallback
    /// is `tests/fixtures/tiny_bpe_tokenizer.json`, NOT the plain
    /// `tiny_tokenizer.json` fixture other tests in this file use —
    /// `tiny_tokenizer.json` has an empty `merges` array (pure byte-level
    /// vocab, no BPE merging at all), which would make this test check
    /// nothing about merge-cache divergence in exactly the CI environment
    /// (`ubuntu-latest`, no HF cache — see `pr-test-sgl-router.yml`) this
    /// test exists to protect: every run there would silently fall back and
    /// silently pass regardless of whether sharding actually preserves
    /// output. `tiny_bpe_tokenizer.json` is `tiny_tokenizer.json` plus four
    /// real merges (t+h, th+e, i+n, in+g), so `assert_merge_actually_fired`
    /// below can confirm real BPE merging ran, not just byte-level passthrough,
    /// on ANY machine.
    #[test]
    fn sharded_instances_produce_identical_output() {
        let source = find_cached_real_tokenizer_json()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "tests/fixtures/tiny_bpe_tokenizer.json".into());
        eprintln!("sharded_instances_produce_identical_output: using {source}");

        const N: usize = 8;
        let shards = TokenizerShards::load(&source, N, adapter::TokenizerLoadOpts::default())
            .expect("load N independent instances");
        assert_eq!(shards.shards.len(), N);

        let inputs: &[&str] = &[
            "Hello, how are you today?",
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)\n",
            "The quick brown fox jumps over the lazy dog near the riverbank while the sun sets slowly behind the distant mountains, casting long shadows across the valley below.",
            "你好，世界！这是一个测试。今天天气怎么样？",
            "SGLang is a fast serving framework for large language models and vision language models.",
            "aaaaaaaaaa bbbbbbbbbb aaaaaaaaaa bbbbbbbbbb aaaaaaaaaa",
            "",
            "🚀🔥💯 emoji stress test with mixed ASCII and 日本語 text!!!",
        ];

        let mut any_merge_fired = false;
        for text in inputs {
            let outputs: Vec<Vec<u32>> = shards
                .shards
                .iter()
                .map(|t| adapter::encode(t, text).expect("encode must succeed"))
                .collect();
            // A merge fired if the token count is less than the raw byte
            // count — every unmerged byte is its own token in this
            // byte-level vocab, so any reduction below `text.len()` (bytes,
            // not chars) proves at least one BPE merge actually ran.
            if !outputs[0].is_empty() && outputs[0].len() < text.len() {
                any_merge_fired = true;
            }
            for (i, ids_i) in outputs.iter().enumerate() {
                for (j, ids_j) in outputs.iter().enumerate() {
                    assert_eq!(
                        ids_i, ids_j,
                        "shard {i} and shard {j} produced different token ids for {text:?}: \
                         sharding must never change tokenization output"
                    );
                }
            }
        }
        assert!(
            any_merge_fired,
            "none of the test inputs triggered a single BPE merge against {source} — this test \
             would then be checking N identical byte-level passthroughs, not real shard-to-shard \
             merge-cache divergence, silently validating nothing. Use a tokenizer.json with a \
             non-empty merges table (see tests/fixtures/tiny_bpe_tokenizer.json)."
        );
    }

    /// A growing multi-turn-shaped conversation over the tiny BPE fixture,
    /// using its one special token (`<|endoftext|>`, id 256) as the turn
    /// boundary — the shape the L1 prefix cache exists for: element `k` is
    /// element `k-1` plus one boundary and one new "turn".
    fn growing_conversation(turns: usize) -> Vec<String> {
        let mut out = Vec::with_capacity(turns);
        let mut text = String::new();
        for t in 0..turns {
            text.push_str(&format!("the thing in turn {t} is interesting "));
            text.push_str("<|endoftext|>");
            out.push(text.clone());
        }
        out
    }

    /// The L1-cached tokenizer must be OUTPUT-INVISIBLE: byte-identical ids
    /// vs a plain (uncached) load, on cold encodes, warm re-encodes, AND the
    /// growing-conversation pattern where extend-on-hit builds ever-deeper
    /// cached prefixes. These ids feed cache-affinity hashing and are
    /// forwarded to the engine as `input_ids`, so any divergence is silent
    /// output corruption — equality here is the gate for enabling the cache.
    #[test]
    fn l1_cached_encode_matches_plain_encode() {
        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let cached = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Hf,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();

        for text in growing_conversation(8) {
            let want = adapter::encode(&plain, &text).unwrap();
            let cold = adapter::encode(&cached, &text).unwrap();
            assert_eq!(cold, want, "cold cached encode diverged for {text:?}");
            // Second encode of the same text re-runs the lookup; from the
            // 2-turn element on it hits at the deepest INTERIOR boundary (a
            // trailing special's end-of-text boundary is never cached, so
            // the 1-turn element stays a miss and hits are partial).
            let warm = adapter::encode(&cached, &text).unwrap();
            assert_eq!(warm, want, "warm cached encode diverged for {text:?}");
        }
        // Also texts with NO special-token boundary (cache can't help) and
        // suffixes AFTER a cached boundary that differ from what was cached.
        for text in [
            "no boundaries here at all",
            "the thing in turn 0 is interesting <|endoftext|>but a different suffix",
            "",
        ] {
            let want = adapter::encode(&plain, text).unwrap();
            let got = adapter::encode(&cached, text).unwrap();
            assert_eq!(got, want, "cached encode diverged for {text:?}");
        }
    }

    /// The fastokens backend must be encode-equivalent to the HF backend.
    /// The first assertion pins that fastokens genuinely LOADS the fixture:
    /// `load_with_opts` silently falls back to HF on a fastokens load
    /// failure, which would turn the equivalence assertions below into
    /// HF-vs-HF theater (this happened — the fixture originally lacked the
    /// `"type": "BPE"` model tag fastokens requires, and the test passed
    /// while exercising zero fastokens code).
    #[test]
    fn fast_backend_encode_matches_hf() {
        assert!(
            dynamo_tokenizers::FastTokenizer::from_file("tests/fixtures/tiny_bpe_tokenizer.json")
                .is_ok(),
            "fixture no longer loads under fastokens — the equivalence assertions below are \
             vacuous (load_with_opts silently falls back to HF). Fix the fixture or fastokens \
             pin before trusting this test."
        );
        let hf = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let fast = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 0,
            },
        )
        .unwrap();
        let corpus = [
            "hello world",
            "the thing <|endoftext|> another thing",
            "你好，世界！ mixed 🚀 content",
            "",
            "aaaaaaaaaa the the the ing ing ing",
        ];
        for text in corpus {
            let want = adapter::encode(&hf, text).unwrap();
            let got = adapter::encode(&fast, text).unwrap();
            assert_eq!(
                got, want,
                "fast-backend encode diverged from HF for {text:?} — fastokens is NOT \
                 engine-equivalent for this tokenizer; do not deploy --tokenizer-backend fast"
            );
        }
    }

    /// The PRODUCTION target config — fast backend + L1 cache STACKED — must
    /// match the plain HF oracle. The two single-lever tests don't compose
    /// automatically (disjoint corpora, and the cache wraps a different
    /// inner), so the deploy config gets its own always-run equivalence.
    #[test]
    fn fast_l1_stacked_encode_matches_plain_encode() {
        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let stacked = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();
        for text in growing_conversation(8) {
            let want = adapter::encode(&plain, &text).unwrap();
            assert_eq!(
                adapter::encode(&stacked, &text).unwrap(),
                want,
                "cold fast+L1 encode diverged for {text:?}"
            );
            assert_eq!(
                adapter::encode(&stacked, &text).unwrap(),
                want,
                "warm fast+L1 encode diverged for {text:?}"
            );
        }
    }

    /// Concurrent encodes through ONE shared L1-cached tokenizer — the
    /// production topology (the cache collapses shards to a single
    /// instance). Unlike the plain-path sibling test, every call here also
    /// MUTATES cache state (insert / extend-on-hit / eviction bookkeeping),
    /// so this is the only place a logic race in the cache's concurrent
    /// bookkeeping would surface as wrong ids. Growing-conversation inputs
    /// make extend-on-hit actually race.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cached_tokenizer_supports_concurrent_encode() {
        use tokio::task::JoinSet;

        let plain = adapter::load("tests/fixtures/tiny_bpe_tokenizer.json").unwrap();
        let cached = adapter::load_with_opts(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 8 * 1024 * 1024,
            },
        )
        .unwrap();

        // Sequential reference from the UNCACHED oracle; 4 interleaved
        // copies of each growing-conversation element so concurrent tasks
        // contend on the same prefixes.
        let mut inputs: Vec<String> = Vec::new();
        for text in growing_conversation(6) {
            for _ in 0..4 {
                inputs.push(text.clone());
            }
        }
        let expected: Vec<Vec<u32>> = inputs
            .iter()
            .map(|s| adapter::encode(&plain, s).unwrap())
            .collect();

        let mut set = JoinSet::new();
        for (i, text) in inputs.into_iter().enumerate() {
            let shared = Arc::clone(&cached);
            set.spawn(async move {
                let ids = adapter::encode(&shared, &text)
                    .expect("concurrent cached encode must not fail");
                (i, ids)
            });
        }
        let mut got: Vec<Option<Vec<u32>>> = vec![None; expected.len()];
        while let Some(joined) = set.join_next().await {
            let (i, ids) = joined.expect("task panicked");
            got[i] = Some(ids);
        }
        for (i, ids) in got.into_iter().enumerate() {
            let ids = ids.unwrap_or_else(|| panic!("task {i} did not record a result"));
            assert_eq!(
                ids, expected[i],
                "concurrent cached encode produced wrong tokens for task {i} — \
                 sign of a race in the shared L1 cache's insert/extend path"
            );
        }
    }

    /// An enabled L1 cache forces a single tokenizer instance — per-shard
    /// caches would split the hit rate across shards and multiply the byte
    /// budget (see `TokenizerShards::load`).
    #[test]
    fn l1_cache_forces_single_shard() {
        let shards = TokenizerShards::load(
            "tests/fixtures/tiny_bpe_tokenizer.json",
            8,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Hf,
                l1_cache_bytes: 1024 * 1024,
            },
        )
        .unwrap();
        assert_eq!(
            shards.shards.len(),
            1,
            "L1 cache enabled must collapse tokenizer shards to one shared instance"
        );
    }

    /// Real-model equivalence + timing for the DSV4 workload this prototype
    /// targets. Ignored by default: needs a real DeepSeek-V4 tokenizer.json,
    /// pointed at via env. Run with:
    ///
    ///   DSV4_TOKENIZER_JSON=/path/to/tokenizer.json \
    ///     cargo test --release -p sgl-router dsv4_real -- --ignored --nocapture
    ///
    /// Asserts (the deploy gates):
    ///   1. fastokens encode ids == HF encode ids on multi-turn DSV4-rendered
    ///      prompts (ids are forwarded to the engine — divergence is silent
    ///      wrong output);
    ///   2. L1-cached encode ids == plain ids at every turn of a growing
    ///      conversation;
    ///   3. the pinned engine-`/tokenize` vector from `dsv4.rs` holds through
    ///      every backend/wrapper combination.
    ///
    /// Prints cold/warm timings for HF, fast, and fast+L1 so the run doubles
    /// as the prototype benchmark.
    #[test]
    #[ignore = "needs DSV4_TOKENIZER_JSON pointing at a real DeepSeek-V4 tokenizer.json"]
    fn dsv4_real_tokenizer_equivalence_and_timing() {
        let Ok(path) = std::env::var("DSV4_TOKENIZER_JSON") else {
            panic!("set DSV4_TOKENIZER_JSON to a DeepSeek-V4 tokenizer.json path");
        };
        let hf = adapter::load(&path).unwrap();
        let fast = adapter::load_with_opts(
            &path,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 0,
            },
        )
        .unwrap();
        let fast_l1 = adapter::load_with_opts(
            &path,
            adapter::TokenizerLoadOpts {
                backend: adapter::TokenizerBackend::Fast,
                l1_cache_bytes: 512 * 1024 * 1024,
            },
        )
        .unwrap();

        // Gate 3: the pinned engine-/tokenize vector (dsv4.rs module doc).
        let pinned = serde_json::json!([{"role": "user", "content": "ABCD"}]);
        let rendered = dsv4::render_messages(&pinned, None, dsv4::RenderOpts::chat());
        for (name, tok) in [("hf", &hf), ("fast", &fast), ("fast+l1", &fast_l1)] {
            assert_eq!(
                adapter::encode(tok, &rendered).unwrap(),
                vec![0, 128803, 51453, 128804, 128822],
                "pinned DSV4 vector diverged through {name}"
            );
        }

        // Build a deterministic multi-turn conversation totalling ~70k tokens
        // when rendered — the shape and scale of the live workload.
        let mut messages: Vec<serde_json::Value> = Vec::new();
        let words = [
            "alpha", "beta", "gamma", "delta", "system", "router", "tensor", "kernel", "deploy",
            "metric", "latency", "bucket", "engine", "token", "prefix", "cache", "the", "of",
            "and", "request",
        ];
        let mut w = 0usize;
        for turn in 0..40 {
            let mut content = format!("turn {turn}: ");
            for _ in 0..1500 {
                content.push_str(words[w % words.len()]);
                content.push(' ');
                w = w.wrapping_mul(31).wrapping_add(17);
            }
            let role = if turn % 2 == 0 { "user" } else { "assistant" };
            messages.push(serde_json::json!({"role": role, "content": content}));
        }

        let time_encode = |tok: &Arc<Tokenizer>, text: &str| -> (Vec<u32>, f64) {
            let t0 = std::time::Instant::now();
            let ids = adapter::encode(tok, text).unwrap();
            (ids, t0.elapsed().as_secs_f64() * 1000.0)
        };

        // Gates 1 + 2 across a GROWING conversation (each iteration appends
        // one turn — the production shape), timing every layer per turn.
        eprintln!("turn | tokens | hf_ms | fast_ms | fast_l1_ms");
        for upto in (2..=messages.len()).step_by(8) {
            let msgs = serde_json::Value::Array(messages[..upto].to_vec());
            let text = dsv4::render_messages(&msgs, None, dsv4::RenderOpts::chat());
            let (want, hf_ms) = time_encode(&hf, &text);
            let (got_fast, fast_ms) = time_encode(&fast, &text);
            let (got_l1, l1_ms) = time_encode(&fast_l1, &text);
            assert_eq!(got_fast, want, "fastokens diverged at {upto} turns");
            assert_eq!(got_l1, want, "fast+L1 diverged at {upto} turns");
            eprintln!(
                "{upto:4} | {:6} | {hf_ms:7.1} | {fast_ms:7.1} | {l1_ms:7.1}",
                want.len()
            );
        }
        let (hits, misses, cached_tok, encoded_tok) = adapter::l1_cache_counters();
        eprintln!(
            "l1 counters: hits={hits} misses={misses} cached_tokens={cached_tok} encoded_tokens={encoded_tok}"
        );
    }
}
