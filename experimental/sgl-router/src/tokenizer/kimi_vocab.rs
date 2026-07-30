// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Kimi vocabulary: Baseten's BPE, chunked the way the Python reference chunks.
//!
//! The BPE, the merge table and the pre-tokenizer all come from
//! `basetenkenizer` reading a HuggingFace `tokenizer.json` — this module builds
//! none of that. What it owns is INPUT CHUNKING, and that is the whole reason it
//! exists rather than `dynamo_tokenizers::BasetenTokenizer`.
//!
//! # Why not `dynamo_tokenizers::BasetenTokenizer`
//!
//! Its `Encoder::encode_segments` calls `encode_segments_tiktoken_safe`, which
//! reproduces the reference's two input splits (400,000 characters, then 25,000
//! per whitespace/non-whitespace run) — but decides what "whitespace" means with
//! `char::is_whitespace`, the Unicode `White_Space` property. The reference uses
//! Python's `str.isspace()`, which ALSO counts U+001C..U+001F. On a >25,000-char
//! run containing one of those four, the two disagree about where runs break,
//! and a split point is a forced token boundary — so the ids differ. The router
//! forwards ids to the engine, which makes that a different prompt rather than a
//! cache miss. See [`py_is_space`] and the parity test that pins it.
//!
//! So the split happens here, with Python's definition, and the pieces go to
//! `basetenkenizer`'s NON-chunking `encode_segments` (the one that does not
//! re-split them). Pre-splitting and then calling the upstream wrapper would not
//! work: it would re-split each piece under its own definition and reintroduce
//! exactly the divergence.

use anyhow::{Context, Result};
use dynamo_tokenizers::{
    traits::{DecodeResult, Decoder, Encoder, Tokenizer},
    EncodeSegment, Encoding, TokenIdType,
};
use std::path::Path;

use super::Segment;

/// Maximum characters Python feeds to one encode call
/// (`TIKTOKEN_MAX_ENCODE_CHARS`). Characters, not bytes — Python slices strings
/// by code point. A split point here changes the token stream, so it is
/// reproduced rather than skipped.
const MAX_ENCODE_CHARS: usize = 400_000;

/// Maximum run of consecutive whitespace / non-whitespace characters Python
/// feeds to one encode call (`MAX_NO_WHITESPACES_CHARS`, working around
/// <https://github.com/openai/tiktoken/issues/195>). Same story as
/// [`MAX_ENCODE_CHARS`]: a split point that changes the token stream.
const MAX_NO_WHITESPACE_CHARS: usize = 25_000;

/// A Kimi vocabulary loaded from a `tokenizer.json`.
///
/// Cheap to share (`Arc`) and safe to encode from many threads at once.
/// `basetenkenizer` shards its merge cache internally, so unlike the HuggingFace
/// backend there is no single per-instance lock for concurrent callers to
/// contend — which is why this type needs no sharding counterpart to
/// [`super::TokenizerShards`].
pub struct KimiVocab {
    inner: basetenkenizer::Tokenizer,
}

impl KimiVocab {
    /// Load from a `tokenizer.json`, merging any specials that live only in a
    /// sibling `tokenizer_config.json`.
    ///
    /// The merge mirrors what `dynamo_tokenizers::BasetenTokenizer::from_file`
    /// does, and is reproduced because this module cannot call that loader and
    /// still reach the non-chunking encode path behind it. It matters for a
    /// vocabulary that keeps its control markers in `added_tokens_decoder`
    /// rather than in `tokenizer.json`'s `added_tokens`; without it every marker
    /// would BPE into ordinary tokens (see [`super::kimi_k3::CONTROL_MARKERS`]).
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read tokenizer.json at {}", path.display()))?;
        let mut json: serde_json::Value = serde_json::from_str(&raw)
            .with_context(|| format!("parse tokenizer.json at {}", path.display()))?;
        if let Some(parent) = path.parent() {
            merge_config_specials(&mut json, parent);
        }
        let inner = basetenkenizer::Tokenizer::from_json(json)
            .map_err(|e| anyhow::anyhow!("load {} as a Baseten vocabulary: {e}", path.display()))?;
        Ok(Self { inner })
    }

    /// How many tokens this vocabulary can produce, for the startup log.
    ///
    /// Worth logging because [`super::kimi_k3::markers_resolve`] only proves four
    /// strings are registered — it cannot tell a K3 vocabulary from a K3-adjacent
    /// one, and the router forwards this vocabulary's ids to the engine.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Tokenize rendered segments, honoring each one's provenance and applying
    /// the reference tokenizer's input splits.
    ///
    /// THE interpreter of [`Segment`]'s `allow_special`. It lives here, rather
    /// than as a loop at the call site, because a second copy is untestable in
    /// practice: the parity fixtures would drive one copy while production ran
    /// the other, and inverting the production branch would still pass every
    /// test — the exact silent failure the segment split exists to prevent.
    pub fn encode_segments(&self, segments: &[Segment]) -> Result<Vec<u32>> {
        let pieces = self.split_for_encode(segments.iter().map(|s| (s.text(), s.allows_special())));
        self.encode_pieces(pieces)
    }

    /// Apply Python's two input-size guards, preserving each piece's
    /// `allow_special`.
    ///
    /// Both splits are reproduced rather than skipped because they are
    /// observable in the output: a prompt long enough to cross either boundary
    /// tokenizes differently on each side of it, and "long prompt" is exactly
    /// the traffic cache-aware routing exists for.
    fn split_for_encode<'a, I>(&self, segments: I) -> Vec<(&'a str, bool)>
    where
        I: Iterator<Item = (&'a str, bool)>,
    {
        let mut pieces = Vec::new();
        for (text, allow_special) in segments {
            for chunk in char_chunks(text, MAX_ENCODE_CHARS) {
                for piece in split_whitespace_runs(chunk, MAX_NO_WHITESPACE_CHARS) {
                    pieces.push((piece, allow_special));
                }
            }
        }
        pieces
    }

    /// Encode `text` as ONE piece, bypassing the split entirely.
    ///
    /// Only for the parity test that pins the C0-separator behaviour: it needs
    /// "what the reference produces when it does not split" as an independent
    /// oracle, and the split is the thing under test.
    #[cfg(test)]
    pub(crate) fn encode_unsplit_for_test(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_pieces(vec![(text, false)])
    }

    /// Hand already-split pieces to the NON-chunking segmented encode.
    ///
    /// `add_special_tokens = false`: the rendered prompt already contains every
    /// marker it needs as text, so a post-processor BOS would be a token the
    /// engine never sees.
    fn encode_pieces(&self, pieces: Vec<(&str, bool)>) -> Result<Vec<u32>> {
        self.inner
            .encode_segments(pieces, false)
            .map_err(|e| anyhow::anyhow!("Baseten segmented encode: {e}"))
    }
}

impl Encoder for KimiVocab {
    /// Specials ARE recognized here, which is the opposite of what the chat
    /// encoder does for client text — deliberately. This is the generic
    /// `Tokenizer::encode` behind `/v1/tokenize` and raw-prompt routing, and it
    /// mirrors `TikTokenTokenizer.encode`, whose `allow_special_tokens`
    /// parameter DEFAULTS TO TRUE. The segment split exists because the chat
    /// path needs the other mode for part of its input; it is not a global
    /// policy, and diverging from the Python default here would break parity on
    /// the raw path instead of fixing anything.
    fn encode(&self, input: &str) -> Result<Encoding> {
        let pieces = self.split_for_encode(std::iter::once((input, true)));
        Ok(Encoding::Sp(self.encode_pieces(pieces)?))
    }

    fn encode_batch(&self, inputs: &[&str]) -> Result<Vec<Encoding>> {
        inputs.iter().map(|i| self.encode(i)).collect()
    }

    /// Implemented rather than left at its refusing default, so segmented encode
    /// through a trait object agrees with [`KimiVocab::encode_segments`] instead
    /// of failing. Both go through the same splitter.
    fn encode_segments(&self, segments: &[EncodeSegment<'_>]) -> Result<Encoding> {
        let pieces = self.split_for_encode(segments.iter().map(|s| (s.text, s.allow_special)));
        Ok(Encoding::Sp(self.encode_pieces(pieces)?))
    }
}

impl Decoder for KimiVocab {
    /// An id outside the vocabulary is DROPPED, not rejected.
    ///
    /// That is `basetenkenizer`'s behavior (a bounds-checked lookup that skips a
    /// miss) and also the HuggingFace backend's, so `/v1/detokenize` answers the
    /// same way for every model this router serves. Noted because it is a
    /// contract, not an accident: a client sending a corrupt id list gets a
    /// shorter string rather than an error.
    fn decode(&self, token_ids: &[TokenIdType], skip_special_tokens: bool) -> Result<DecodeResult> {
        self.inner
            .decode(token_ids, skip_special_tokens)
            .map(DecodeResult::from)
            .map_err(|e| anyhow::anyhow!("Baseten decode: {e}"))
    }
}

impl Tokenizer for KimiVocab {
    /// Refuses, and is never asked: the registry inserts this type through
    /// `TokenizerShards::shared`, which does not wrap it in a `CachedTokenizer`.
    /// Kept explicit so the refusal is a decision rather than an inherited
    /// default.
    fn validate_prefix_cache(&self) -> Result<()> {
        Err(anyhow::anyhow!(
            "the Kimi vocabulary is served unsharded and uncached; \
             --tokenizer-l1-cache-mb does not apply to it"
        ))
    }
}

/// Merge `tokenizer_config.json`'s `added_tokens_decoder` into `added_tokens`.
///
/// Best-effort by design: a vocabulary whose markers are already in
/// `tokenizer.json` (which `baseten/kimi-k3-tokenizer`'s is) needs nothing from
/// here, so a missing or unparsable config must not fail the load. A config
/// that IS present and broken is warned about, because otherwise its markers go
/// missing and the only symptom is `markers_resolve` reporting them absent.
fn merge_config_specials(json: &mut serde_json::Value, dir: &Path) {
    let path = dir.join("tokenizer_config.json");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return;
    };
    let config: serde_json::Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e,
                "tokenizer_config.json is present but does not parse; any control markers \
                 declared only there will be missing from the vocabulary");
            return;
        }
    };
    let Some(decoder) = config
        .get("added_tokens_decoder")
        .and_then(serde_json::Value::as_object)
    else {
        return;
    };

    if !json.get("added_tokens").is_some_and(|v| v.is_array()) {
        json["added_tokens"] = serde_json::json!([]);
    }
    let Some(added) = json
        .get_mut("added_tokens")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };

    for (id, spec) in decoder {
        let (Ok(id), Some(spec)) = (id.parse::<u32>(), spec.as_object()) else {
            continue;
        };
        if spec.get("special").and_then(serde_json::Value::as_bool) != Some(true) {
            continue;
        }
        let Some(content) = spec
            .get("content")
            .and_then(serde_json::Value::as_str)
            .filter(|c| !c.is_empty())
        else {
            continue;
        };
        if let Some(existing) = added
            .iter_mut()
            .find(|t| t.get("content").and_then(serde_json::Value::as_str) == Some(content))
        {
            existing["special"] = serde_json::Value::Bool(true);
            continue;
        }
        let mut token = serde_json::Map::from_iter([
            ("id".to_string(), serde_json::json!(id)),
            ("content".to_string(), serde_json::json!(content)),
            ("special".to_string(), serde_json::Value::Bool(true)),
        ]);
        for field in ["single_word", "lstrip", "rstrip", "normalized"] {
            token.insert(
                field.to_string(),
                serde_json::Value::Bool(
                    spec.get(field)
                        .and_then(serde_json::Value::as_bool)
                        .unwrap_or(false),
                ),
            );
        }
        added.push(serde_json::Value::Object(token));
    }
}

/// Split `text` into runs of at most `max_chars` CHARACTERS.
///
/// Python's `range(0, len(text), N)` counts code points, so a byte-based split
/// would land in a different place on any non-ASCII prompt.
fn char_chunks(text: &str, max_chars: usize) -> Vec<&str> {
    if text.is_empty() {
        // Python's `range(0, 0, N)` is empty, so an empty input encodes to
        // nothing at all rather than to one empty piece.
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    let mut count = 0;
    for (idx, _) in text.char_indices() {
        if count == max_chars {
            chunks.push(&text[start..idx]);
            start = idx;
            count = 0;
        }
        count += 1;
    }
    chunks.push(&text[start..]);
    chunks
}

/// Break `s` wherever a run of consecutive whitespace — or of consecutive
/// non-whitespace — would exceed `max_run`, reproducing
/// `_split_whitespaces_or_nonwhitespaces`.
fn split_whitespace_runs(s: &str, max_run: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut chars = s.char_indices();
    let Some((_, first)) = chars.next() else {
        // Python yields `s[0:]` even for an empty string, so the empty piece is
        // preserved (it encodes to nothing, but dropping it here would diverge
        // if that ever stopped being true).
        out.push(s);
        return out;
    };

    let mut run_is_space = py_is_space(first);
    let mut run_len = 1;
    let mut slice_start = 0;
    for (idx, c) in chars {
        let now_space = py_is_space(c);
        if run_is_space != now_space {
            run_len = 1;
            run_is_space = now_space;
        } else {
            run_len += 1;
            if run_len > max_run {
                out.push(&s[slice_start..idx]);
                slice_start = idx;
                run_len = 1;
            }
        }
    }
    out.push(&s[slice_start..]);
    out
}

/// Python's `str.isspace()`, which is NOT `char::is_whitespace`.
///
/// Python additionally treats the C0 separators U+001C..U+001F as space; Unicode
/// `White_Space` (what Rust implements) does not. Only the run-split reads this,
/// but it decides split points, and a split point is a forced token boundary —
/// so getting it wrong changes ids on long inputs. `basetenkenizer`'s own
/// chunker uses `char::is_whitespace`, which is precisely why this module does
/// the splitting instead of delegating to it.
fn py_is_space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> KimiVocab {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/tokenizer/testdata/kimi_k3_tiny_vocab/tokenizer.json");
        KimiVocab::from_file(&path).expect("load tiny vocabulary")
    }

    #[test]
    fn py_is_space_covers_c0_separators() {
        // The four Python calls whitespace and Rust does not. This is the entire
        // reason this module owns the split.
        for c in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}'] {
            assert!(py_is_space(c), "U+{:04X} is whitespace to Python", c as u32);
            assert!(
                !c.is_whitespace(),
                "U+{:04X} is NOT Unicode White_Space — if this ever flips, \
                 py_is_space can delegate",
                c as u32
            );
        }
        // Sanity: the ordinary ones agree on both sides.
        for c in [' ', '\t', '\n', '\r', '\u{0b}', '\u{0c}'] {
            assert!(py_is_space(c) && c.is_whitespace());
        }
    }

    #[test]
    fn char_chunks_counts_characters_not_bytes() {
        // Four 3-byte characters, max 2 chars: split after the second CHARACTER
        // (byte 6), not after byte 2, which would not even be a char boundary.
        let text = "の".repeat(4);
        assert_eq!(char_chunks(&text, 2), vec!["のの", "のの"]);
        assert!(char_chunks("", 2).is_empty());
        assert_eq!(char_chunks("abc", 10), vec!["abc"]);
    }

    #[test]
    fn split_whitespace_runs_breaks_only_long_runs() {
        // Alternating runs shorter than the cap are never split, however long
        // the string: the run counter resets at every polarity change.
        assert_eq!(
            split_whitespace_runs("aaaa    bbbb", 4),
            vec!["aaaa    bbbb"]
        );
        // One run over the cap splits at the cap.
        assert_eq!(split_whitespace_runs("aaaaa", 4), vec!["aaaa", "a"]);
        assert_eq!(split_whitespace_runs("", 4), vec![""]);
    }

    /// A C0 separator flips run polarity for Python, so it RESETS the run and no
    /// split happens; `char::is_whitespace` would see one long run and split.
    #[test]
    fn c0_separator_resets_the_run_the_way_python_does() {
        let text = format!("{}\u{1f}{}", "x".repeat(6), "x".repeat(6));
        // Python: runs are 6, 1, 6 — none exceeds 8, so one piece.
        assert_eq!(split_whitespace_runs(&text, 8), vec![text.as_str()]);

        // The same input under Unicode White_Space is a single 13-char run and
        // would be split. Asserted so the divergence is pinned, not assumed.
        let rust_split = {
            let mut out = Vec::new();
            let mut run = 0usize;
            let mut is_sp = text.chars().next().is_some_and(char::is_whitespace);
            let mut start = 0usize;
            for (idx, c) in text.char_indices() {
                let now = c.is_whitespace();
                if is_sp != now {
                    run = 1;
                    is_sp = now;
                } else {
                    run += 1;
                    if run > 8 {
                        out.push(&text[start..idx]);
                        start = idx;
                        run = 1;
                    }
                }
            }
            out.push(&text[start..]);
            out
        };
        assert_eq!(rust_split.len(), 2, "the upstream chunker WOULD split here");
    }

    /// The 400,000-character guard is wired, not just declared. Asserted on the
    /// split rather than on ids so it costs nothing in CI.
    #[test]
    fn the_400k_guard_splits_a_single_long_run() {
        let tk = tiny();
        let text = "x".repeat(MAX_ENCODE_CHARS + 1);
        let pieces = tk.split_for_encode(std::iter::once((text.as_str(), false)));
        // 400_001 chars of one run: the outer guard cuts at 400_000, and the inner
        // 25_000 guard then cuts the first chunk into 16 and leaves the 1-char tail.
        assert!(
            pieces.len() > 16,
            "both guards should fire, got {} pieces",
            pieces.len()
        );
        assert_eq!(
            pieces.iter().map(|(p, _)| p.chars().count()).sum::<usize>(),
            MAX_ENCODE_CHARS + 1,
            "splitting must be lossless"
        );
    }

    #[test]
    fn encode_segments_honors_the_allow_special_flag() {
        let tk = tiny();
        let marker = tk
            .encode_segments(&[Segment::marker("<|open|>")])
            .expect("encode marker");
        assert_eq!(marker.len(), 1, "a marker is one control token");
        let client = tk
            .encode_segments(&[Segment::client_text("<|open|>")])
            .expect("encode client text");
        assert!(
            client.len() > 1,
            "identical text from a client must BPE as ordinary bytes"
        );
    }

    #[test]
    fn decode_drops_ids_outside_the_vocabulary() {
        let tk = tiny();
        let hello = tk
            .encode_segments(&[Segment::client_text("hello")])
            .unwrap();
        let text = tk.decode(&hello, false).unwrap();

        let mut corrupt = hello.clone();
        corrupt.push(u32::MAX);
        assert_eq!(
            tk.decode(&corrupt, false).unwrap().as_str(),
            text.as_str(),
            "an out-of-range id is dropped, matching every other backend"
        );
    }

    /// The committed `tokenizer.json` must still describe the committed
    /// `tiktoken.model` it was generated from.
    ///
    /// Neither file is read by production code, so nothing else notices when they
    /// drift — and they drift in both directions: `gen_kimi_k3_cases.py` rewrites
    /// the rank file, `gen_kimi_k3_tiny_tokenizer_json.py` rewrites the JSON, and
    /// running one without the other leaves the id fixtures asserting against a
    /// vocabulary that no longer exists. Checked structurally rather than by
    /// re-deriving merges, so this test does not become a second copy of the
    /// generator.
    /// GPT-2's byte -> printable-codepoint map, matching `bytes_to_unicode()` in
    /// `gen_kimi_k3_tiny_tokenizer_json.py`. Reimplemented here rather than shared
    /// because the generator is Python and this is the independent side of the
    /// check.
    fn byte_to_char_table() -> [char; 256] {
        let mut table = ['\0'; 256];
        let mut direct = [false; 256];
        for b in (b'!'..=b'~').chain(0xA1..=0xAC).chain(0xAE..=0xFF) {
            table[b as usize] = b as char;
            direct[b as usize] = true;
        }
        let mut n = 0u32;
        for b in 0..256 {
            if !direct[b] {
                table[b] = char::from_u32(256 + n).expect("in range");
                n += 1;
            }
        }
        table
    }

    #[test]
    fn the_committed_tiny_vocab_matches_its_tiktoken_source() {
        use base64::Engine as _;

        let dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tokenizer/testdata/kimi_k3_tiny_vocab");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("tokenizer.json")).unwrap())
                .expect("committed tokenizer.json parses");

        // Every rank in the model file must appear in `vocab` with the same id,
        // byte-level-encoded. A mismatch means one file was regenerated alone.
        let ranks = std::fs::read_to_string(dir.join("tiktoken.model")).unwrap();
        let vocab = json["model"]["vocab"].as_object().expect("vocab object");
        let engine = base64::engine::general_purpose::STANDARD;
        let mut n = 0;
        for line in ranks.lines().filter(|l| !l.trim().is_empty()) {
            let (b64, rank) = line.split_once(' ').expect("`base64 rank` per line");
            let token = engine.decode(b64).expect("base64 token");
            let table = byte_to_char_table();
            let encoded: String = token.iter().map(|b| table[*b as usize]).collect();
            assert_eq!(
                vocab.get(&encoded).and_then(serde_json::Value::as_u64),
                Some(rank.parse::<u64>().unwrap()),
                "token {token:?} is missing or has a different id in tokenizer.json — \
                 re-run gen_kimi_k3_tiny_tokenizer_json.py"
            );
            n += 1;
        }
        assert_eq!(
            vocab.len(),
            n,
            "tokenizer.json has entries the rank file does not"
        );

        // `ignore_merges` is load-bearing: 12 of these tokens are reachable ONLY
        // as whole-pre-token matches, and flipping it changes every fixture's ids.
        assert_eq!(json["model"]["ignore_merges"], serde_json::json!(true));

        // Every special declared in the config must be an added token, or the
        // control markers silently stop resolving.
        let config: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("tokenizer_config.json")).unwrap(),
        )
        .unwrap();
        let added = json["added_tokens"].as_array().expect("added_tokens array");
        for (id, spec) in config["added_tokens_decoder"]
            .as_object()
            .expect("added_tokens_decoder")
        {
            let content = spec["content"].as_str().unwrap();
            assert!(
                added.iter().any(|t| {
                    t["content"].as_str() == Some(content)
                        && t["id"].as_u64() == Some(id.parse().unwrap())
                }),
                "special {content} (id {id}) is missing from tokenizer.json's added_tokens"
            );
        }
    }

    #[test]
    fn prefix_cache_is_refused() {
        assert!(tiny().validate_prefix_cache().is_err());
    }
}
