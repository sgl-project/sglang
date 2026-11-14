//! Incremental encoding of prompts that extend an earlier prompt, for tokenizers
//! whose BPE merges cannot cross a space that follows a non-space: the normalizer
//! turns `" "` into `"▁"` and nothing pre-tokenizes the result (Gemma 4).
//!
//! A multi-turn chat re-sends its whole history, so most of each prompt was
//! encoded on an earlier turn. The encoder keeps recent prompts with their ids
//! and byte offsets. A new prompt reuses the ids of the stored prompt it shares
//! the longest prefix with, up to the last cut inside that prefix, and encodes
//! only the text after the cut. A cut is a space whose left neighbour is neither
//! a space nor a literal `"▁"` (the two are the same after normalization); no
//! vocabulary entry spans one except the few `crossing` entries, whose
//! occurrences are never cut, so both sides encode independently and the joined
//! ids equal a whole-prompt encode.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::Hasher;
use std::sync::{Arc, Mutex};

use serde_json::Value;

/// Prompts are indexed by hashes of their prefix at multiples of this many bytes:
/// at the powers of two, which prompts sharing a system prompt reach together, and
/// at the last multiple a prompt reaches, which is usually its session's alone. A
/// lookup probes every multiple from the top down. Arbitrary granularity.
const LEVEL_BYTES: usize = 1 << 10;
/// Prefixes longer than this are not hashed further; arbitrary.
const MAX_LEVEL_BYTES: usize = 1 << 20;
/// Most recent prompts kept per bucket; bounds the prefix compares of a lookup.
const BUCKET_CAP: usize = 8;
/// Shorter prompts are cheap to encode whole and are never stored.
const MIN_PROMPT_BYTES: usize = 4096;
/// A prompt replaces the stored one it reused when it shares all but this many
/// of its bytes (the generation stub the next turn rewrites); else it is added.
const REPLACE_SLACK_BYTES: usize = 256;
/// Arbitrary cap; a vocabulary with more space-crossing entries is not admitted.
const MAX_CROSSING_PATTERNS: usize = 64;
const SPACE_MARK: char = '\u{2581}';

/// A vocabulary entry with a `"▁"` after a non-space, split at that `"▁"` into the
/// raw text before and after it (`"▁"` written as a space): a cut at a space with
/// `left` just before it and `right` just after it would split that token.
struct CrossingPattern {
    left: Vec<u8>,
    right: Vec<u8>,
}

/// Where a prompt may be cut so that encoding the two sides apart gives its ids.
struct Cutter {
    crossing: Vec<CrossingPattern>,
}

impl Cutter {
    /// The last cut before `prefix_len`, or 0 when there is none.
    fn cut_point(&self, text: &[u8], prefix_len: usize) -> usize {
        (1..prefix_len)
            .rev()
            .find(|&p| {
                text[p] == b' ' && !ends_with_normalized(text, p, b" ") && !self.crosses(text, p)
            })
            .unwrap_or(0)
    }

    fn crosses(&self, text: &[u8], p: usize) -> bool {
        self.crossing.iter().any(|c| {
            ends_with_normalized(text, p, &c.left) && starts_with_normalized(text, p + 1, &c.right)
        })
    }
}

/// UTF-8 of `SPACE_MARK`, which a literal `"▁"` in the input encodes to.
const SPACE_MARK_UTF8: &[u8] = "\u{2581}".as_bytes();

/// Whether `text[..end]` ends with `pattern` after normalization: a space in
/// `pattern` also matches a literal `"▁"`.
fn ends_with_normalized(text: &[u8], mut end: usize, pattern: &[u8]) -> bool {
    for &byte in pattern.iter().rev() {
        if byte == b' ' && text[..end].ends_with(SPACE_MARK_UTF8) {
            end -= SPACE_MARK_UTF8.len();
        } else if end > 0 && text[end - 1] == byte {
            end -= 1;
        } else {
            return false;
        }
    }
    true
}

/// Whether `text[start..]` starts with `pattern` after normalization.
fn starts_with_normalized(text: &[u8], mut start: usize, pattern: &[u8]) -> bool {
    for &byte in pattern {
        let rest = text.get(start..).unwrap_or_default();
        if byte == b' ' && rest.starts_with(SPACE_MARK_UTF8) {
            start += SPACE_MARK_UTF8.len();
        } else if rest.first() == Some(&byte) {
            start += 1;
        } else {
            return false;
        }
    }
    true
}

/// Ids with the byte span of each in the encoded text.
struct Tokens {
    ids: Vec<u32>,
    starts: Vec<u32>,
    ends: Vec<u32>,
}

/// A stored prompt; immutable, so lookups compare it outside the store lock.
struct Entry {
    text: String,
    tokens: Tokens,
}

impl Entry {
    fn bytes(&self) -> usize {
        self.text.len() + self.tokens.ids.len() * 12
    }
}

struct Slot {
    entry: Arc<Entry>,
    keys: Vec<(usize, u64)>,
    /// Buckets still holding this slot; at 0 no lookup can reach it.
    refs: usize,
    last_used: u64,
}

#[derive(Default)]
struct Store {
    slots: HashMap<u64, Slot>,
    /// (level, hash of the first `level` bytes) -> slot ids, oldest first.
    buckets: HashMap<(usize, u64), VecDeque<u64>>,
    next_id: u64,
    bytes: usize,
    clock: u64,
}

impl Store {
    fn remove(&mut self, id: u64) {
        let Some(slot) = self.slots.remove(&id) else {
            return;
        };
        for key in &slot.keys {
            if let Some(bucket) = self.buckets.get_mut(key) {
                bucket.retain(|&other| other != id);
                if bucket.is_empty() {
                    self.buckets.remove(key);
                }
            }
        }
        self.bytes -= slot.entry.bytes();
    }
}

pub(crate) struct IncrementalEncoder {
    /// Must encode without added specials: stored ids are spliced mid-prompt.
    tokenizer: dynamo_tokenizers::Tokenizer,
    cutter: Cutter,
    budget_bytes: usize,
    store: Mutex<Store>,
}

impl IncrementalEncoder {
    /// `Err` with the reason when `tokenizer_json` is not admitted (see [`admission`]).
    pub(crate) fn new(
        tokenizer: dynamo_tokenizers::Tokenizer,
        tokenizer_json: &str,
        budget_bytes: usize,
    ) -> Result<Self, String> {
        let spec: Value = std::fs::read_to_string(tokenizer_json)
            .map_err(|e| e.to_string())
            .and_then(|text| serde_json::from_str(&text).map_err(|e| e.to_string()))?;
        Ok(Self {
            tokenizer,
            cutter: Cutter {
                crossing: admission(&spec)?,
            },
            budget_bytes,
            store: Mutex::new(Store::default()),
        })
    }

    pub(crate) fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        if text.len() < MIN_PROMPT_BYTES {
            let encoding = self.tokenizer.encode(text).map_err(|e| e.to_string())?;
            return Ok(encoding.token_ids().to_vec());
        }
        let probes = prefix_keys(text);
        let reuse = self
            .candidates(&probes)
            .into_iter()
            .map(|(id, entry)| {
                let shared = common_prefix_len(entry.text.as_bytes(), text.as_bytes());
                (id, entry, shared)
            })
            .max_by_key(|(_, _, shared)| *shared)
            .and_then(|(id, entry, shared)| {
                let cut = self.cutter.cut_point(text.as_bytes(), shared);
                let stored = &entry.tokens;
                let k = stored.starts.partition_point(|&s| (s as usize) < cut);
                let clean = cut >= MIN_PROMPT_BYTES
                    && k > 0
                    && stored.ends[k - 1] as usize <= cut
                    && stored.starts.get(k).is_none_or(|&s| s as usize >= cut);
                clean.then(|| {
                    let replace = shared + REPLACE_SLACK_BYTES >= entry.text.len();
                    (entry, k, cut, id, replace)
                })
            });

        let (tokens, reused) = match reuse {
            Some((stored, k, cut, id, replace)) => {
                let tail = self.encode_with_offsets(&text[cut..], cut)?;
                let stored = &stored.tokens;
                let tokens = Tokens {
                    ids: [&stored.ids[..k], &tail.ids[..]].concat(),
                    starts: [&stored.starts[..k], &tail.starts[..]].concat(),
                    ends: [&stored.ends[..k], &tail.ends[..]].concat(),
                };
                (tokens, Some((id, replace)))
            }
            None => (self.encode_with_offsets(text, 0)?, None),
        };
        let ids = tokens.ids.clone();
        let entry = Entry {
            text: text.to_string(),
            tokens,
        };
        let top = probes.len() - 1;
        let keys = probes
            .into_iter()
            .enumerate()
            .filter(|&(i, (level, _))| level.is_power_of_two() || i == top)
            .map(|(_, key)| key)
            .collect();
        self.insert(keys, entry, reused);
        Ok(ids)
    }

    /// The tokens of `text`, their spans shifted by `base`.
    fn encode_with_offsets(&self, text: &str, base: usize) -> Result<Tokens, String> {
        let encoding = self.tokenizer.encode(text).map_err(|e| e.to_string())?;
        let dynamo_tokenizers::Encoding::Hf(hf) = &encoding else {
            return Err("tokenizer returned no offsets".into());
        };
        let (starts, ends) = hf
            .get_offsets()
            .iter()
            .map(|&(s, e)| ((base + s) as u32, (base + e) as u32))
            .unzip();
        Ok(Tokens {
            ids: hf.get_ids().to_vec(),
            starts,
            ends,
        })
    }

    /// The prompts of the deepest non-empty bucket among `probes`.
    fn candidates(&self, probes: &[(usize, u64)]) -> Vec<(u64, Arc<Entry>)> {
        let store = self.store.lock().unwrap();
        probes
            .iter()
            .rev()
            .find_map(|key| store.buckets.get(key))
            .map(|bucket| {
                bucket
                    .iter()
                    .filter_map(|id| {
                        store
                            .slots
                            .get(id)
                            .map(|slot| (*id, Arc::clone(&slot.entry)))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Store `entry` under `keys`, and touch or replace the prompt it reused.
    fn insert(&self, keys: Vec<(usize, u64)>, entry: Entry, reused: Option<(u64, bool)>) {
        let mut store = self.store.lock().unwrap();
        store.clock += 1;
        let clock = store.clock;
        match reused {
            Some((id, true)) => store.remove(id),
            Some((id, false)) => {
                if let Some(slot) = store.slots.get_mut(&id) {
                    slot.last_used = clock;
                }
            }
            None => {}
        }
        let id = store.next_id;
        store.next_id += 1;
        store.bytes += entry.bytes();
        let mut unreachable = Vec::new();
        for key in &keys {
            let bucket = store.buckets.entry(*key).or_default();
            bucket.push_back(id);
            if bucket.len() > BUCKET_CAP {
                unreachable.extend(bucket.pop_front());
            }
        }
        let refs = keys.len();
        store.slots.insert(
            id,
            Slot {
                entry: Arc::new(entry),
                keys,
                refs,
                last_used: clock,
            },
        );
        for dropped in unreachable {
            if let Some(slot) = store.slots.get_mut(&dropped) {
                slot.refs -= 1;
                if slot.refs == 0 {
                    store.remove(dropped);
                }
            }
        }
        while store.bytes > self.budget_bytes {
            let Some(oldest) = store
                .slots
                .iter()
                .min_by_key(|(_, slot)| slot.last_used)
                .map(|(&id, _)| id)
            else {
                break;
            };
            store.remove(oldest);
        }
    }
}

/// (level, hash of the first `level` bytes) for every multiple of `LEVEL_BYTES` the
/// text reaches, from a single hashing pass.
fn prefix_keys(text: &str) -> Vec<(usize, u64)> {
    let bytes = &text.as_bytes()[..text.len().min(MAX_LEVEL_BYTES)];
    let mut hasher = DefaultHasher::new();
    bytes
        .chunks_exact(LEVEL_BYTES)
        .enumerate()
        .map(|(i, chunk)| {
            hasher.write(chunk);
            ((i + 1) * LEVEL_BYTES, hasher.clone().finish())
        })
        .collect()
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    const CHUNK: usize = 64;
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + CHUNK <= n && a[i..i + CHUNK] == b[i..i + CHUNK] {
        i += CHUNK;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// The space-crossing patterns of an admitted `tokenizer.json`, or why it is not
/// admitted. Admitted: a BPE model whose only normalization is `" "` -> `"▁"`,
/// with no pre-tokenizer split left after it, no post-processor tokens, and no
/// added token containing a space. A merge can then join the two sides of a cut
/// only through a vocabulary entry with `"▁"` after a non-space.
fn admission(spec: &Value) -> Result<Vec<CrossingPattern>, String> {
    if spec["model"]["type"] != "BPE" {
        return Err("model is not BPE".into());
    }
    let normalizer = &spec["normalizer"];
    if !(normalizer["type"] == "Replace"
        && normalizer["pattern"]["String"] == " "
        && normalizer["content"] == SPACE_MARK.to_string())
    {
        return Err("normalizer is not a lone space replace".into());
    }
    let pre = &spec["pre_tokenizer"];
    if !(pre.is_null() || (pre["type"] == "Split" && pre["pattern"]["String"] == " ")) {
        return Err("pre-tokenizer splits the normalized text".into());
    }
    let post = &spec["post_processor"];
    let adds_tokens = post["single"]
        .as_array()
        .is_some_and(|items| items.iter().any(|item| item.get("Sequence").is_none()));
    if !(post.is_null() || post["type"] == "TemplateProcessing") || adds_tokens {
        return Err("post-processor adds tokens".into());
    }
    let added = spec["added_tokens"].as_array().cloned().unwrap_or_default();
    if added
        .iter()
        .any(|t| t["content"].as_str().is_some_and(|c| c.contains(' ')))
    {
        return Err("an added token contains a space".into());
    }
    let vocab = spec["model"]["vocab"]
        .as_object()
        .ok_or("model has no vocab map")?;
    let mut crossing = Vec::new();
    for token in vocab.keys() {
        let chars: Vec<char> = token.chars().collect();
        for i in 1..chars.len() {
            if chars[i] == SPACE_MARK && chars[i - 1] != SPACE_MARK {
                let raw = |part: &[char]| -> Vec<u8> {
                    part.iter()
                        .map(|&c| if c == SPACE_MARK { ' ' } else { c })
                        .collect::<String>()
                        .into_bytes()
                };
                crossing.push(CrossingPattern {
                    left: raw(&chars[..i]),
                    right: raw(&chars[i + 1..]),
                });
            }
        }
    }
    if crossing.len() > MAX_CROSSING_PATTERNS {
        return Err(format!(
            "{} vocabulary entries cross a space",
            crossing.len()
        ));
    }
    Ok(crossing)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cut never lands inside a run of spaces, where `"▁▁"` entries merge; a
    /// literal `"▁"` before a space starts such a run too.
    #[test]
    fn cut_starts_a_space_run() {
        let cutter = Cutter {
            crossing: Vec::new(),
        };
        let text = b"one two   three";
        assert_eq!(cutter.cut_point(text, text.len()), 7);
        assert_eq!(cutter.cut_point(text, 7), 3);
        let text = "one two\u{2581} three".as_bytes();
        assert_eq!(cutter.cut_point(text, text.len()), 3);
    }

    /// A cut never splits an occurrence of a crossing pattern.
    #[test]
    fn cut_avoids_crossing_patterns() {
        let cutter = Cutter {
            crossing: vec![CrossingPattern {
                left: b">".to_vec(),
                right: b"</ b".to_vec(),
            }],
        };
        assert_eq!(cutter.cut_point(b"a b <i> </ b>", 10), 3);
        assert_eq!(cutter.cut_point(b"a b x </ b>", 8), 5);
        // The pattern's inner space also matches a literal "▁".
        let text = "a b <i> </\u{2581}b>".as_bytes();
        assert_eq!(cutter.cut_point(text, text.len()), 3);
    }

    /// Gemma 4's shape is admitted with its crossing entries; a byte-level
    /// pre-tokenizer is not.
    #[test]
    fn admits_space_replace_bpe() {
        let spec = gemma_like_spec();
        let mut crossing: Vec<_> = admission(&spec)
            .unwrap()
            .into_iter()
            .map(|c| (c.left, c.right))
            .collect();
        crossing.sort();
        assert_eq!(
            crossing,
            [
                (b">".to_vec(), b"".to_vec()),
                (b">".to_vec(), b"<".to_vec())
            ]
        );
        let mut byte_level = spec.clone();
        byte_level["pre_tokenizer"] = serde_json::json!({"type": "ByteLevel"});
        assert!(admission(&byte_level).is_err());
    }

    /// Sessions sharing a long prefix, interleaved, sometimes rewriting their
    /// history, encode to the whole-prompt ids while reusing stored ids, and the
    /// store stays bounded by the live sessions.
    #[test]
    fn interleaved_sessions_match_whole_encode() {
        let dir =
            std::env::temp_dir().join(format!("processor-incremental-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("tokenizer.json");
        std::fs::write(&file, gemma_like_spec().to_string()).unwrap();
        let path = file.to_str().unwrap();
        let load = || {
            dynamo_tokenizers::Tokenizer::from_file_with_options(
                path,
                dynamo_tokenizers::TokenizerOptions {
                    add_special_tokens: false,
                },
            )
            .unwrap()
        };
        let counting = Arc::new(CountingEncoder {
            inner: load(),
            bytes: Default::default(),
        });
        let encoder = IncrementalEncoder::new(
            dynamo_tokenizers::Tokenizer::from(
                Arc::clone(&counting) as Arc<dyn dynamo_tokenizers::traits::Tokenizer>
            ),
            path,
            64 << 20,
        )
        .unwrap();
        let whole = load();
        std::fs::remove_dir_all(&dir).ok();

        let words = [
            "ab",
            "ba",
            "a",
            "b",
            ">",
            "<",
            "a>",
            "<b",
            "\u{2581}",
            "a\u{2581}",
        ];
        let mut seed = 7u64;
        let mut phrase = |n: usize| -> String {
            (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let word = words[(seed >> 33) as usize % words.len()];
                    let gap = ["", " ", " ", " ", "  ", "\n"][(seed >> 40) as usize % 6];
                    format!("{word}{gap}")
                })
                .collect()
        };
        let system = phrase(3000);
        let mut histories: Vec<String> = (0..20).map(|_| phrase(20)).collect();
        let (mut calls, mut stored_hits, mut prompt_bytes) = (0, 0, 0);
        for round in 0..10 {
            for (s, history) in histories.iter_mut().enumerate() {
                if round == 5 && s % 4 == 0 {
                    *history = phrase(20);
                }
                history.push_str(&phrase(300));
                let prompt = format!("<bos>{system}{history}<eot>");
                let had = encoder.candidates(&prefix_keys(&prompt)).len();
                let ids = encoder.encode(&prompt).unwrap();
                assert_eq!(
                    ids,
                    whole.encode(&prompt).unwrap().token_ids(),
                    "round {round} session {s}"
                );
                calls += 1;
                stored_hits += usize::from(had > 0);
                prompt_bytes += prompt.len();
            }
        }
        assert!(stored_hits >= calls - 1, "{stored_hits} of {calls}");
        let encoded = counting.bytes.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            encoded * 10 < prompt_bytes,
            "encoded {encoded} of {prompt_bytes} bytes"
        );
        let store = encoder.store.lock().unwrap();
        assert!(store.buckets.values().all(|b| b.len() <= BUCKET_CAP));
        assert!(
            store.slots.len() <= 2 * histories.len(),
            "{} slots",
            store.slots.len()
        );
    }

    /// Counts the bytes it encodes, to show how much of each prompt was reused.
    struct CountingEncoder {
        inner: dynamo_tokenizers::Tokenizer,
        bytes: std::sync::atomic::AtomicUsize,
    }

    impl dynamo_tokenizers::traits::Encoder for CountingEncoder {
        fn encode(&self, input: &str) -> dynamo_tokenizers::Result<dynamo_tokenizers::Encoding> {
            self.bytes
                .fetch_add(input.len(), std::sync::atomic::Ordering::Relaxed);
            self.inner.encode(input)
        }

        fn encode_batch(
            &self,
            inputs: &[&str],
        ) -> dynamo_tokenizers::Result<Vec<dynamo_tokenizers::Encoding>> {
            inputs.iter().map(|input| self.encode(input)).collect()
        }
    }

    impl dynamo_tokenizers::traits::Decoder for CountingEncoder {
        fn decode(
            &self,
            token_ids: &[u32],
            skip_special_tokens: bool,
        ) -> dynamo_tokenizers::Result<dynamo_tokenizers::DecodeResult> {
            self.inner.decode(token_ids, skip_special_tokens)
        }
    }

    impl dynamo_tokenizers::traits::Tokenizer for CountingEncoder {}

    /// A Gemma-shaped BPE: spaces normalized to `▁`, merges that build `▁▁` runs,
    /// words with a leading `▁`, and one entry (`>▁<`) that crosses a space.
    fn gemma_like_spec() -> Value {
        let mut vocab = serde_json::Map::new();
        let tokens = [
            "<bos>",
            "<eot>",
            "a",
            "b",
            ">",
            "<",
            "\n",
            "▁",
            "▁▁",
            "▁▁▁▁",
            "ab",
            "ba",
            "▁a",
            "▁b",
            "▁ab",
            "▁ba",
            "a>",
            "<b",
            ">▁",
            ">▁<",
        ];
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert((*token).into(), id.into());
        }
        let merges = [
            ("▁", "▁"),
            ("▁▁", "▁▁"),
            ("a", "b"),
            ("b", "a"),
            ("▁", "a"),
            ("▁", "b"),
            ("▁a", "b"),
            ("▁b", "a"),
            ("a", ">"),
            ("<", "b"),
            (">", "▁"),
            (">▁", "<"),
        ];
        serde_json::json!({
            "added_tokens": [
                {"id": 0, "content": "<bos>", "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true},
                {"id": 1, "content": "<eot>", "single_word": false, "lstrip": false,
                    "rstrip": false, "normalized": false, "special": true},
            ],
            "normalizer": {"type": "Replace", "pattern": {"String": " "}, "content": "▁"},
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "},
                "behavior": "MergedWithPrevious", "invert": false},
            "post_processor": {"type": "TemplateProcessing",
                "single": [{"Sequence": {"id": "A", "type_id": 0}}],
                "pair": [{"Sequence": {"id": "A", "type_id": 0}}, {"Sequence": {"id": "B", "type_id": 1}}],
                "special_tokens": {}},
            "model": {"type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab,
                "merges": merges.iter().map(|(a, b)| serde_json::json!([a, b])).collect::<Vec<_>>()},
        })
    }
}
