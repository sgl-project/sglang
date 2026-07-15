// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result};
use dynamo_tokenizers::{traits::DecodeResult, CachedTokenizer, FastTokenizer, Tokenizer};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

/// Which encode implementation backs a loaded tokenizer.
///
/// Accepted on the CLI (`--tokenizer-backend`) as `hf` / `fast`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum TokenizerBackend {
    /// HuggingFace `tokenizers` via dynamo's `Tokenizer::from_file`. The
    /// pre-existing behavior and the default.
    #[default]
    #[value(name = "hf")]
    Hf,
    /// `fastokens` hybrid encode (`dynamo_tokenizers::FastTokenizer`):
    /// fastokens for `encode`, HuggingFace for `decode`. Falls back to
    /// [`TokenizerBackend::Hf`] with a warning when fastokens cannot load
    /// the tokenizer file, so an exotic `tokenizer.json` degrades to the
    /// known-good path instead of failing startup. The fallback is visible
    /// after startup via `sgl_router_tokenizer_backend` (see
    /// [`tokenizer_runtime_states`]), not just the startup log line.
    #[value(name = "fast")]
    Fast,
}

/// How [`load_with_opts`] should build a tokenizer instance.
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenizerLoadOpts {
    pub backend: TokenizerBackend,
    /// Byte budget for the L1 special-token-boundary prefix cache
    /// (`dynamo_tokenizers::CachedTokenizer`). 0 disables the wrapper —
    /// every encode goes straight to the backend, the pre-existing
    /// behavior.
    pub l1_cache_bytes: usize,
}

/// Process-wide L1 prefix-cache counters, fed by the `CachedTokenizer`
/// observers wired in [`load_with_opts`]: `L1_HITS`/`L1_MISSES` by the
/// hit/miss (lookup-level) observer, `L1_CACHED_TOKENS`/
/// `L1_UNCACHED_TOKENS` by the token-level observer. Rendered as
/// `sgl_router_tokenizer_l1_*` counters. Process-global (not per-model)
/// because the router serves one model per process today; revisit if that
/// changes.
static L1_CACHED_TOKENS: AtomicU64 = AtomicU64::new(0);
static L1_UNCACHED_TOKENS: AtomicU64 = AtomicU64::new(0);
static L1_HITS: AtomicU64 = AtomicU64::new(0);
static L1_MISSES: AtomicU64 = AtomicU64::new(0);

/// Which backend the last [`load_with_opts`] actually built (as opposed to
/// what was requested) and whether the L1 cache ended up active. A
/// fastokens load failure silently degrades performance to the HF baseline
/// — a startup warn alone is invisible after log rotation, so the resolved
/// state is exported on `/metrics` (`sgl_router_tokenizer_backend` /
/// `sgl_router_tokenizer_l1_state`) where a dashboard can alert on
/// "requested fast, running hf". Last-write-wins across loads: production
/// loads one model per process, so the last load IS the model's state.
static BACKEND_STATE: AtomicU8 = AtomicU8::new(BACKEND_HF);
static L1_STATE: AtomicU8 = AtomicU8::new(L1_OFF);

const BACKEND_HF: u8 = 0;
const BACKEND_FAST: u8 = 1;
const BACKEND_FAST_FALLBACK_HF: u8 = 2;
const L1_OFF: u8 = 0;
const L1_ACTIVE: u8 = 1;
const L1_DISABLED_NO_SPECIALS: u8 = 2;

/// Snapshot of the process-wide L1 prefix-cache counters, in the order
/// `(hits, misses, cached_tokens, uncached_tokens)`. Hit/miss count cache
/// LOOKUPS (one per encode while L1 is active — boundary-less prompts count
/// as misses); the token counts split each successful encode's output into
/// prefix tokens served from cache vs freshly encoded — their ratio is the
/// fraction of tokenization work the cache is absorbing.
pub fn l1_cache_counters() -> (u64, u64, u64, u64) {
    (
        L1_HITS.load(Ordering::Relaxed),
        L1_MISSES.load(Ordering::Relaxed),
        L1_CACHED_TOKENS.load(Ordering::Relaxed),
        L1_UNCACHED_TOKENS.load(Ordering::Relaxed),
    )
}

/// The resolved tokenizer runtime state for `/metrics`, as
/// `(backend, l1_state)` label values. `backend` is one of `"hf"`,
/// `"fast"`, `"fast_fallback_hf"` (fast was requested but fastokens could
/// not load the file — running on HF at baseline performance); `l1_state`
/// is one of `"off"`, `"active"`, `"disabled_no_specials"` (cache was
/// requested but the tokenizer declares no safely-splittable special
/// tokens, so it is inert).
pub fn tokenizer_runtime_states() -> (&'static str, &'static str) {
    let backend = match BACKEND_STATE.load(Ordering::Relaxed) {
        BACKEND_FAST => "fast",
        BACKEND_FAST_FALLBACK_HF => "fast_fallback_hf",
        _ => "hf",
    };
    let l1 = match L1_STATE.load(Ordering::Relaxed) {
        L1_ACTIVE => "active",
        L1_DISABLED_NO_SPECIALS => "disabled_no_specials",
        _ => "off",
    };
    (backend, l1)
}

/// Resolve `source` to a local tokenizer file: an existing local file (or
/// anything with a filesystem-path shape) is used as-is; otherwise `source`
/// is treated as a HuggingFace repo id and its `tokenizer.json` is
/// downloaded (once — subsequent calls hit `hf-hub`'s on-disk cache) into
/// the HF cache, honoring `HF_TOKEN` / `HF_HOME` / `HF_HUB_OFFLINE`.
/// `dynamo_tokenizers` itself has no HF-download path, so the fetch is done
/// here via `hf-hub`.
fn resolve_local(source: &str) -> Result<PathBuf> {
    if Path::new(source).is_file() || looks_like_path(source) {
        Ok(Path::new(source).to_path_buf())
    } else {
        download_tokenizer_json(source)
    }
}

/// Settle the L1 half of `opts` BEFORE any shard-count decision: when the
/// cache is requested but the tokenizer declares no safely-splittable
/// special tokens, the wrapper would be inert — zero the budget (with a
/// warn) so callers like `TokenizerShards::load` don't also collapse the
/// shard count for a cache that cannot exist. Reads the same
/// [`l1_safe_specials`] the wrap path uses, so the two can't disagree.
pub fn finalize_load_opts(source: &str, opts: TokenizerLoadOpts) -> Result<TokenizerLoadOpts> {
    if opts.l1_cache_bytes == 0 {
        return Ok(opts);
    }
    let local = resolve_local(source)?;
    if l1_safe_specials(&local).is_empty() {
        tracing::warn!(path = %local.display(),
            "tokenizer L1 cache requested but the tokenizer declares no safely-splittable \
             special tokens; cache disabled, tokenizer sharding kept as configured");
        L1_STATE.store(L1_DISABLED_NO_SPECIALS, Ordering::Relaxed);
        return Ok(TokenizerLoadOpts {
            l1_cache_bytes: 0,
            ..opts
        });
    }
    Ok(opts)
}

/// Load a tokenizer from `source`, which is either a local `tokenizer.json`
/// path or a HuggingFace repo id (see [`resolve_local`]).
///
/// Default-opts wrapper around [`load_with_opts`] — HF backend, no L1
/// cache — kept because most tests (and any future callers that don't care
/// about the encode backend) want exactly the pre-existing behavior.
pub fn load(source: &str) -> Result<Arc<Tokenizer>> {
    load_with_opts(source, TokenizerLoadOpts::default())
}

/// Load a tokenizer from `source` with an explicit backend and optional L1
/// prefix cache. See [`TokenizerLoadOpts`]. Callers that make decisions
/// based on whether the cache will really be active (shard counts, budget
/// accounting) should pass `opts` through [`finalize_load_opts`] first.
pub fn load_with_opts(source: &str, opts: TokenizerLoadOpts) -> Result<Arc<Tokenizer>> {
    let local = resolve_local(source)?;
    let path = local
        .to_str()
        .context("tokenizer path is not valid UTF-8")?;

    let inner: Tokenizer = match opts.backend {
        TokenizerBackend::Hf => {
            BACKEND_STATE.store(BACKEND_HF, Ordering::Relaxed);
            Tokenizer::from_file(path).with_context(|| format!("load tokenizer from {path}"))?
        }
        TokenizerBackend::Fast => match FastTokenizer::from_file(path) {
            Ok(fast) => {
                BACKEND_STATE.store(BACKEND_FAST, Ordering::Relaxed);
                tracing::info!(%path, "tokenizer backend: fastokens encode + HF decode");
                Tokenizer::from(Arc::new(fast))
            }
            Err(e) => {
                BACKEND_STATE.store(BACKEND_FAST_FALLBACK_HF, Ordering::Relaxed);
                tracing::warn!(%path, error = %e,
                    "fastokens could not load this tokenizer; falling back to the HF backend \
                     (encode runs at baseline speed — alert on \
                     sgl_router_tokenizer_backend{{backend=\"fast_fallback_hf\"}})");
                Tokenizer::from_file(path)
                    .with_context(|| format!("load tokenizer from {path} (HF fallback)"))?
            }
        },
    };

    if opts.l1_cache_bytes == 0 {
        return Ok(Arc::new(inner));
    }

    // L1 prefix cache: correctness depends on splitting ONLY at boundaries
    // that are atomic in the tokenizer, so the wrapper gets the
    // safety-filtered special-token strings (the `Tokenizer` trait doesn't
    // expose them; read them from the file). An empty list would make
    // `CachedTokenizer` a pure passthrough — skip the wrap so the loaded
    // instance is exactly the unwrapped backend. `finalize_load_opts`
    // normally zeroes the budget (and warns) before this point; this branch
    // is the defensive belt for direct callers.
    let specials = l1_safe_specials(&local);
    if specials.is_empty() {
        L1_STATE.store(L1_DISABLED_NO_SPECIALS, Ordering::Relaxed);
        return Ok(Arc::new(inner));
    }
    L1_STATE.store(L1_ACTIVE, Ordering::Relaxed);
    tracing::info!(%path, budget_bytes = opts.l1_cache_bytes, n_special = specials.len(),
        "tokenizer L1 prefix cache enabled (special-token-boundary caching, extend-on-hit)");
    let cached = CachedTokenizer::new(
        (*inner).clone(), // Arc<dyn traits::Tokenizer> via Deref — a refcount bump
        specials,
        opts.l1_cache_bytes,
    )
    // Extend-on-hit: a partial hit also caches the freshly-encoded suffix,
    // so each turn of a growing conversation hits deeper than the last —
    // per-turn tokenization cost stops growing with conversation length.
    // This is the multi-turn workload the cache exists for.
    .with_extend(true)
    .with_observer(
        Arc::new(|| {
            L1_HITS.fetch_add(1, Ordering::Relaxed);
        }),
        Arc::new(|| {
            L1_MISSES.fetch_add(1, Ordering::Relaxed);
        }),
    )
    .with_token_observer(Arc::new(|usage| {
        L1_CACHED_TOKENS.fetch_add(usage.cached_tokens as u64, Ordering::Relaxed);
        L1_UNCACHED_TOKENS.fetch_add(usage.uncached_tokens as u64, Ordering::Relaxed);
    }));
    Ok(Arc::new(Tokenizer::from(Arc::new(cached))))
}

/// The special-token strings a `tokenizer.json` declares that are SAFE to
/// use as L1 split boundaries, i.e. atomic under encode: `added_tokens`
/// entries with `special: true` and none of the matching modifiers that
/// break `tokenize(prefix) + tokenize(suffix) == tokenize(prefix + suffix)`
/// at a raw-text match end:
///
///   * `lstrip`/`rstrip` — the token absorbs adjacent whitespace, so the
///     raw-text occurrence isn't where the token actually ends;
///   * `single_word` — matching is conditional on word boundaries;
///   * `normalized: true` WITH a non-null normalizer — the token is matched
///     against normalized text, so raw-text positions don't correspond
///     (with a null normalizer the flag is behaviorally moot, so it's
///     allowed).
///
/// Excluding a token only costs boundary density (fewer split points →
/// lower hit rate); INCLUDING an unsafe one would corrupt the ids that are
/// forwarded to the engine as `input_ids` — so this filters conservatively
/// and warns about what it drops.
///
/// Fail-soft by design: an unreadable or non-JSON file (e.g. a tiktoken
/// `.model`) yields an empty list — the cache degrades to disabled rather
/// than failing startup, since the tokenizer load itself is the
/// authoritative validation of the file. Reads the file because the
/// `dynamo_tokenizers::traits::Tokenizer` trait does not re-expose special
/// tokens. KNOWN LIMITATION vs dynamo's `model_card.rs` (which extracts
/// from the loaded HF tokenizer after merging `tokenizer_config.json`):
/// specials declared ONLY in `tokenizer_config.json` (e.g. Qwen2-VL's
/// `<|image_pad|>`) are missed here — costing hit rate on such models, not
/// correctness.
fn l1_safe_specials(path: &Path) -> Vec<String> {
    let value: serde_json::Value = match std::fs::read(path)
        .map_err(anyhow::Error::from)
        .and_then(|b| serde_json::from_slice(&b).map_err(anyhow::Error::from))
    {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "could not read tokenizer file as JSON while extracting special tokens; \
                 L1 cache will be disabled");
            return Vec::new();
        }
    };
    let normalizer_present = value.get("normalizer").is_some_and(|n| !n.is_null());
    let Some(added) = value.get("added_tokens").and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    let flag =
        |t: &serde_json::Value, key: &str| t.get(key).and_then(|v| v.as_bool()) == Some(true);
    let mut safe = Vec::new();
    let mut excluded: Vec<String> = Vec::new();
    for t in added {
        if !flag(t, "special") {
            continue;
        }
        let Some(content) = t.get("content").and_then(|c| c.as_str()) else {
            continue;
        };
        let unsafe_match = flag(t, "lstrip")
            || flag(t, "rstrip")
            || flag(t, "single_word")
            || (flag(t, "normalized") && normalizer_present);
        if unsafe_match {
            excluded.push(content.to_owned());
        } else {
            safe.push(content.to_owned());
        }
    }
    if !excluded.is_empty() {
        let shown: Vec<&str> = excluded.iter().take(5).map(String::as_str).collect();
        tracing::warn!(path = %path.display(), n_excluded = excluded.len(), sample = ?shown,
            "special tokens with lstrip/rstrip/single_word/normalized matching excluded from \
             L1 cache boundaries (unsafe split points; costs hit rate, preserves correctness)");
    }
    safe
}

/// Treat `source` as a filesystem path (rather than a HuggingFace repo id)
/// when it has a path-like shape — an absolute/relative prefix or a `.json`
/// suffix. HF repo ids are `namespace/name` with none of these markers, so a
/// missing local file like `/models/tok.json` reports a load error instead of
/// silently attempting a (doomed) network fetch.
fn looks_like_path(source: &str) -> bool {
    source.starts_with('/')
        || source.starts_with("./")
        || source.starts_with("../")
        || source.starts_with('~')
        || source.ends_with(".json")
}

/// Download `tokenizer.json` for a HuggingFace repo id and return the cached
/// local path, adding an actionable error context. The actual fetch (blocking
/// `ureq`, `from_env` for `HF_HOME` / endpoint overrides + an explicit
/// `HF_TOKEN` read — see [`download_repo_file`]) lives in [`download_repo_file`].
fn download_tokenizer_json(repo_id: &str) -> Result<PathBuf> {
    download_repo_file(repo_id, "tokenizer.json").with_context(|| {
        format!(
            "download tokenizer.json for HuggingFace repo {repo_id:?} \
             (pass --tokenizer-path with a local tokenizer.json, or set HF_TOKEN \
             for a gated/private repo)"
        )
    })
}

/// Download `file` from a HuggingFace repo id and return the cached local path.
/// Shared by `tokenizer.json` (required) and `tokenizer_config.json` (optional).
///
/// `with_retries(3)`: `ApiBuilder`'s default is 0 retries, so a bare
/// `from_env().build()` makes every HTTP call here a one-shot — any
/// transient DNS/TLS/connection blip fails immediately. That was tolerable
/// when this ran once per model at startup; `TokenizerShards::load` (see
/// `crate::tokenizer::TokenizerShards`) now calls this up to
/// `tokenizer_shards` times (default 8) per model, so a transient failure
/// on shard 1 aborts the whole router before shards 2..N even get a chance.
/// Shards 2..N read the same file from `hf-hub`'s on-disk cache once shard 1
/// has populated it (no network for them at all), so in practice this retry
/// budget matters only for the very first call.
///
/// `HF_TOKEN`: hf-hub 0.4's `ApiBuilder::from_env()` reads `HF_HOME` /
/// `HF_ENDPOINT` but does NOT read the `HF_TOKEN` env var — it only picks up a
/// token from the `$HF_HOME/token` file. Since gated/private tokenizer repos
/// return 401 without auth, read `HF_TOKEN` ourselves and pass it via
/// `with_token`. The guard is load-bearing: `with_token` OVERWRITES the token,
/// so calling it with `None` would clobber the file-based token that `from_env`
/// already loaded.
///
/// We override ONLY when `HF_TOKEN` holds a non-blank value, trimmed first:
/// shell/secret interpolation routinely leaves a trailing newline
/// (`HF_TOKEN=$(<token.txt)`) or surrounding spaces, and an untrimmed token
/// produces a malformed `Authorization: Bearer …\n` header that HF rejects with
/// the same 401 a missing token gives — pointing the operator at the wrong fix.
/// A set-but-blank or non-UTF-8 value is almost always a broken interpolation
/// the operator INTENDED as auth, so we warn rather than silently fall back to
/// anonymous. Genuinely unset (`NotPresent`) is the legitimate no-token case
/// (token file if present, else anonymous for public repos) — left silent.
fn download_repo_file(repo_id: &str, file: &str) -> Result<PathBuf> {
    use hf_hub::api::sync::ApiBuilder;
    let mut builder = ApiBuilder::from_env().with_retries(3);
    match std::env::var("HF_TOKEN") {
        Ok(token) if !token.trim().is_empty() => {
            builder = builder.with_token(Some(token.trim().to_string()));
        }
        Ok(_) => tracing::warn!(
            "HF_TOKEN is set but empty/whitespace-only; ignoring it and falling back to \
             $HF_HOME/token or anonymous access (gated/private repos will 401)"
        ),
        Err(std::env::VarError::NotUnicode(_)) => tracing::warn!(
            "HF_TOKEN is set but not valid UTF-8; ignoring it and falling back to \
             $HF_HOME/token or anonymous access (gated/private repos will 401)"
        ),
        Err(std::env::VarError::NotPresent) => {}
    }
    let api = builder
        .build()
        .context("initialize HuggingFace Hub client")?;
    api.model(repo_id.to_string())
        .get(file)
        .with_context(|| format!("download {file} for HuggingFace repo {repo_id:?}"))
}

/// Load the `tokenizer_config.json` co-located with the tokenizer named by
/// `source` (the same value passed to [`load`]). For a local
/// `.../tokenizer.json` path this is the sibling file; for an HF repo id it is
/// downloaded from the same repo.
///
/// Returns `Ok(None)` when the model ships no `tokenizer_config.json` (rare but
/// valid) — the caller then has no chat template and routes via raw prompt text.
pub fn load_tokenizer_config(source: &str) -> Result<Option<serde_json::Value>> {
    let path = if Path::new(source).is_file() || looks_like_path(source) {
        match Path::new(source).parent() {
            Some(dir) => dir.join("tokenizer_config.json"),
            None => return Ok(None),
        }
    } else {
        // HF repo id. The download error type doesn't distinguish a genuine
        // 404 (repo ships no tokenizer_config.json — benign) from auth/network
        // failures (wrong/expired HF_TOKEN, gated repo, timeout), so warn with
        // the cause rather than asserting the benign case at debug: a swallowed
        // auth error here silently disables chat-template routing.
        match download_repo_file(source, "tokenizer_config.json") {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!(repo = %source, error = %e,
                    "could not download tokenizer_config.json; chat-template routing disabled for this model \
                     (expected if the repo ships none — otherwise check HF_TOKEN / network for a gated or private repo)");
                return Ok(None);
            }
        }
    };
    if !path.is_file() {
        return Ok(None);
    }
    let bytes = std::fs::read(&path)
        .with_context(|| format!("read tokenizer_config.json at {}", path.display()))?;
    let value = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse tokenizer_config.json at {}", path.display()))?;
    Ok(Some(value))
}

pub fn encode(t: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = t.encode(text).context("encode")?;
    Ok(enc.token_ids().to_vec())
}

/// Decode token ids to a complete UTF-8 string.
///
/// Non-streaming callers (e.g. `/v1/detokenize`) get the full result either way:
/// - `DecodeResult::Complete(s)` — the token sequence ends on a codepoint boundary.
/// - `DecodeResult::Partial(s)` — the token sequence ends mid-codepoint; `s` ends
///   in U+FFFD. We return `s` as-is so the client sees the closest-possible string.
///
/// Streaming callers should NOT use this; they should consume `DecodeResult`
/// directly and withhold the trailing U+FFFD until the next decode produces a
/// `Complete` result.
pub fn decode_complete(t: &Tokenizer, ids: &[u32], skip_special: bool) -> Result<String> {
    let res = t.decode(ids, skip_special).context("decode")?;
    Ok(match res {
        DecodeResult::Complete(s) => s,
        DecodeResult::Partial(s) => {
            tracing::debug!(
                n_tokens = ids.len(),
                trailing_bytes = s.len(),
                "decode_complete: tokenizer returned Partial for non-streaming call"
            );
            s
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The L1 cache's split boundaries come from this extraction — a fixture
    /// drift that drops the special flag would silently disable the cache,
    /// so pin the exact list. (The fixture's `<|endoftext|>` carries
    /// `normalized: true` but the file has a null normalizer, so the safety
    /// filter keeps it — also pinning that nuance.)
    #[test]
    fn safe_specials_extraction_reads_added_tokens() {
        let specials = l1_safe_specials(Path::new("tests/fixtures/tiny_bpe_tokenizer.json"));
        assert_eq!(specials, vec!["<|endoftext|>".to_string()]);
    }

    /// A tokenizer.json with no `added_tokens` yields an empty list (the
    /// caller then loads the tokenizer uncached rather than erroring).
    #[test]
    fn safe_specials_empty_without_added_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokenizer.json");
        std::fs::write(&p, "{}").unwrap();
        assert!(l1_safe_specials(&p).is_empty());
    }

    /// Unsafe matching flags exclude a special from the boundary set:
    /// lstrip/rstrip/single_word always; `normalized: true` only when the
    /// file declares a real normalizer. Including such a token would corrupt
    /// forwarded `input_ids`; excluding costs only hit rate.
    #[test]
    fn safe_specials_excludes_unsafe_matching_flags() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokenizer.json");
        std::fs::write(
            &p,
            serde_json::json!({
                "normalizer": {"type": "Lowercase"},
                "added_tokens": [
                    {"content": "<safe>", "special": true,
                     "lstrip": false, "rstrip": false, "single_word": false, "normalized": false},
                    {"content": "<mask>", "special": true, "lstrip": true},
                    {"content": "<rs>", "special": true, "rstrip": true},
                    {"content": "<sw>", "special": true, "single_word": true},
                    {"content": "<norm>", "special": true, "normalized": true},
                    {"content": "not-special", "special": false},
                ]
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(l1_safe_specials(&p), vec!["<safe>".to_string()]);
    }

    /// With a NULL normalizer, `normalized: true` is behaviorally moot and
    /// the token stays in the boundary set (this is the fixture's shape).
    #[test]
    fn safe_specials_allows_normalized_with_null_normalizer() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokenizer.json");
        std::fs::write(
            &p,
            serde_json::json!({
                "normalizer": null,
                "added_tokens": [
                    {"content": "<norm>", "special": true, "normalized": true},
                ]
            })
            .to_string(),
        )
        .unwrap();
        assert_eq!(l1_safe_specials(&p), vec!["<norm>".to_string()]);
    }

    /// A non-JSON tokenizer file (tiktoken `.model` shape) fails SOFT: empty
    /// specials → cache disabled, never a startup abort from this path.
    #[test]
    fn safe_specials_fail_soft_on_non_json() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokenizer.model");
        std::fs::write(&p, b"\x00\x01 not json").unwrap();
        assert!(l1_safe_specials(&p).is_empty());
    }

    /// `finalize_load_opts` zeroes the L1 budget when the tokenizer has no
    /// safe specials — BEFORE any caller collapses the shard count for a
    /// cache that cannot exist — and passes through untouched otherwise.
    #[test]
    fn finalize_load_opts_zeroes_budget_without_specials() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("tokenizer.json");
        std::fs::write(&p, "{}").unwrap();
        let opts = TokenizerLoadOpts {
            backend: TokenizerBackend::Hf,
            l1_cache_bytes: 1024,
        };
        let out = finalize_load_opts(p.to_str().unwrap(), opts).unwrap();
        assert_eq!(
            out.l1_cache_bytes, 0,
            "inert cache must be disabled up front"
        );

        let kept = finalize_load_opts("tests/fixtures/tiny_bpe_tokenizer.json", opts).unwrap();
        assert_eq!(kept.l1_cache_bytes, 1024, "real specials keep the budget");
    }
}
