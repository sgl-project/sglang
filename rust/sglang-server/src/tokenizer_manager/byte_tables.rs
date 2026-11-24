//! Per-id byte tables for byte-level BPE tokenizers.
//!
//! A Hugging Face tokenizer whose decoder is `ByteLevel` decodes ids by joining
//! each id's bytes and decoding the result as UTF-8 with replacement
//! (`String::from_utf8_lossy`). With every id's bytes in a table, a stream
//! decodes only each new chunk's ids, and [`Utf8Tail`] carries the bytes of an
//! incomplete trailing character to the next chunk. dynamo's `DecodeStream`
//! instead re-decodes every id since its last complete text on each token: a
//! stream whose text never completes (skipped special tokens decode to nothing,
//! stray bytes end in U+FFFD) costs O(output length) per token.
//!
//! [`build`] makes the tables once at startup and checks them against the
//! tokenizer's own decode: every id, ids past the vocabulary, and
//! [`CHECK_SEQUENCES`] random sequences, with special tokens kept and skipped.
//! A tokenizer it does not cover, or any difference, yields no tables.

use std::collections::HashMap;

use sglang_processor::dynamo_tokenizers::Tokenizer;

/// Set to false to decode with `DecodeStream` for every tokenizer. Read with
/// `utils::environ::env_bool` (default on; an invalid value warns and keeps the
/// default, as Python's `EnvBool.get` does).
pub const ENV: &str = "SGLANG_DETOKENIZER_BYTE_TABLES";

/// Whether [`ENV`] allows byte-table decoding (default: true).
pub fn enabled() -> bool {
    // A bare `ENV` keeps this read visible to the Rust-server parity scan.
    crate::utils::environ::env_bool(ENV, true)
}

/// Random id sequences checked at startup, on top of every single id.
const CHECK_SEQUENCES: usize = 512;

/// Ids past the tokenizer's token count that the dense tables may cover. A
/// sparser id space gets no tables, so they never allocate far beyond the
/// vocabulary.
const MAX_ID_GAP: usize = 65_536;

/// Every id's bytes for one `skip_special_tokens` value, stored flat: id `i`'s
/// bytes are `data[ends[i - 1]..ends[i]]`.
struct Table {
    data: Vec<u8>,
    ends: Vec<usize>,
}

impl Table {
    fn new(per_id: &[Vec<u8>]) -> Self {
        let mut data = Vec::with_capacity(per_id.iter().map(Vec::len).sum());
        let mut ends = Vec::with_capacity(per_id.len());
        for bytes in per_id {
            data.extend_from_slice(bytes);
            ends.push(data.len());
        }
        Self { data, ends }
    }

    /// An id past the table has no bytes: the tokenizer's decode drops it.
    fn bytes(&self, id: i64) -> &[u8] {
        let Ok(id) = usize::try_from(id) else {
            return &[];
        };
        match self.ends.get(id) {
            Some(&end) => &self.data[if id == 0 { 0 } else { self.ends[id - 1] }..end],
            None => &[],
        }
    }
}

/// Every id's bytes, with special tokens kept and skipped.
pub struct ByteTables {
    keep: Table,
    skip: Table,
}

impl ByteTables {
    /// The bytes `id` decodes to.
    pub fn bytes(&self, skip_special_tokens: bool, id: i64) -> &[u8] {
        if skip_special_tokens {
            self.skip.bytes(id)
        } else {
            self.keep.bytes(id)
        }
    }
}

/// Byte tables for the tokenizer file `tokenizer_path` resolves to, checked
/// against `tokenizer`, which was loaded from it. `Ok` carries a summary for
/// the startup log, `Err` the reason there are no tables.
pub fn build(
    tokenizer_path: &str,
    revision: Option<&str>,
    tokenizer: &Tokenizer,
) -> Result<(ByteTables, String), String> {
    let file = sglang_processor::resolve_tokenizer_file(tokenizer_path, revision)
        .ok_or_else(|| format!("no tokenizer file found for {tokenizer_path:?}"))?;
    if !file.ends_with(".json") {
        return Err(format!("{file} is not a Hugging Face tokenizer.json"));
    }
    let text = std::fs::read_to_string(&file).map_err(|e| format!("reading {file}: {e}"))?;
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parsing {file}: {e}"))?;
    let tokens = ByteLevelTokens::parse(&json)?;
    let vocab_size = tokenizer
        .vocab_size()
        .ok_or_else(|| "the tokenizer reports no vocab size".to_string())?;
    checked_tables(&tokens, vocab_size, |ids, skip_special_tokens| {
        tokenizer
            .decode(ids, skip_special_tokens)
            .map(String::from)
            .map_err(|e| e.to_string())
    })
}

/// A byte-level BPE `tokenizer.json`'s token strings by id, and the ids of its
/// added tokens.
struct ByteLevelTokens {
    by_id: HashMap<u32, String>,
    added: Vec<u32>,
}

impl ByteLevelTokens {
    fn parse(json: &serde_json::Value) -> Result<Self, String> {
        if json["decoder"]["type"] != "ByteLevel" {
            return Err("the tokenizer decoder is not ByteLevel".into());
        }
        let model = &json["model"];
        let vocab = match (model["type"].as_str(), model["vocab"].as_object()) {
            (Some("BPE"), Some(vocab)) => vocab,
            _ => return Err("the tokenizer model is not BPE".into()),
        };
        let id = |value: &serde_json::Value| value.as_u64().and_then(|id| u32::try_from(id).ok());
        let mut by_id: HashMap<u32, String> = vocab
            .iter()
            .filter_map(|(token, value)| Some((id(value)?, token.clone())))
            .collect();
        let mut added = Vec::new();
        // An added token wins over the model's token with its id, as in the
        // tokenizer's own id-to-token lookup.
        for token in json["added_tokens"].as_array().into_iter().flatten() {
            if let (Some(id), Some(content)) = (id(&token["id"]), token["content"].as_str()) {
                by_id.insert(id, content.to_owned());
                added.push(id);
            }
        }
        Ok(Self { by_id, added })
    }
}

fn checked_tables(
    tokens: &ByteLevelTokens,
    vocab_size: usize,
    decode: impl Fn(&[u32], bool) -> Result<String, String>,
) -> Result<(ByteTables, String), String> {
    let count = tokens.by_id.len().max(vocab_size);
    let size = tokens
        .by_id
        .keys()
        .map(|&id| id as usize + 1)
        .max()
        .unwrap_or(0)
        .max(vocab_size);
    if size == 0 {
        return Err("the tokenizer has no ids".into());
    }
    if size > count.saturating_add(MAX_ID_GAP) {
        return Err(format!(
            "the token ids are sparse: they reach {} for {count} tokens",
            size - 1
        ));
    }
    let end = u32::try_from(size).map_err(|_| format!("{size} ids do not fit u32"))?;
    let char_bytes = byte_level_char_bytes();
    // An id with no token string in the file (a special token that only
    // tokenizer_config.json declares) has only its decoded text to go by. That
    // text is its bytes unless it holds U+FFFD, which may stand for bytes the
    // lossy decode replaced (content "XÃ" is bytes 58 C3 and decodes to "X�"):
    // then the bytes are unknown and there are no tables.
    let mut unfiled = Vec::new();
    let keep = (0..end)
        .map(|id| match tokens.by_id.get(&id) {
            Some(token) => Ok(token_bytes(token, &char_bytes)),
            None => {
                let text = decode(&[id], false)?;
                if text.contains(char::REPLACEMENT_CHARACTER) {
                    return Err(format!(
                        "id {id}, not in the tokenizer file, decodes to {text:?}, so its bytes are unknown"
                    ));
                }
                if !text.is_empty() {
                    unfiled.push(id);
                }
                Ok(text.into_bytes())
            }
        })
        .collect::<Result<Vec<_>, String>>()?;

    let mut tables = Vec::with_capacity(2);
    for skip in [false, true] {
        let mut table = keep.clone();
        for (id, bytes) in (0..end).zip(table.iter_mut()) {
            let text = decode(&[id], skip)?;
            if skip && text.is_empty() {
                bytes.clear();
            }
            if String::from_utf8_lossy(bytes) != text {
                return Err(format!(
                    "id {id} decodes to {text:?} (skip_special_tokens={skip})"
                ));
            }
        }
        tables.push(table);
    }
    for id in [end, end.saturating_add(1), u32::MAX] {
        for skip in [false, true] {
            let text = decode(&[id], skip)?;
            if !text.is_empty() {
                return Err(format!("id {id}, past the vocabulary, decodes to {text:?}"));
            }
        }
    }

    // Sequences weighted to the ids most likely to interact: added tokens (from
    // the file or not), and ids whose bytes are not whole characters.
    let added: Vec<u32> = tokens
        .added
        .iter()
        .copied()
        .chain(unfiled)
        .filter(|&id| id < end)
        .collect();
    let partial: Vec<u32> = (0..end)
        .filter(|&id| std::str::from_utf8(&keep[id as usize]).is_err())
        .collect();
    let pools: Vec<&[u32]> = [added.as_slice(), partial.as_slice()]
        .into_iter()
        .filter(|pool| !pool.is_empty())
        .collect();
    let mut rng = SplitMix64(0);
    for _ in 0..CHECK_SEQUENCES {
        let len = 2 + rng.below(23);
        let ids: Vec<u32> = (0..len)
            .map(|_| {
                if !pools.is_empty() && rng.below(2) == 0 {
                    let pool = pools[rng.below(pools.len())];
                    pool[rng.below(pool.len())]
                } else {
                    rng.below_u32(end)
                }
            })
            .collect();
        for (skip, table) in [false, true].into_iter().zip(&tables) {
            let bytes: Vec<u8> = ids
                .iter()
                .flat_map(|&id| table[id as usize].iter().copied())
                .collect();
            let text = decode(&ids, skip)?;
            if String::from_utf8_lossy(&bytes) != text {
                return Err(format!(
                    "ids {ids:?} decode to {text:?} (skip_special_tokens={skip})"
                ));
            }
        }
    }

    let tables = ByteTables {
        keep: Table::new(&tables[0]),
        skip: Table::new(&tables[1]),
    };
    Ok((
        tables,
        format!("checked {size} ids and {CHECK_SEQUENCES} sequences"),
    ))
}

/// The byte each character of a byte-level token stands for: the inverse of
/// GPT-2's byte-to-unicode map, which the `ByteLevel` decoder applies.
fn byte_level_char_bytes() -> HashMap<char, u8> {
    let mut bytes: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut chars: Vec<u32> = bytes.iter().map(|&b| u32::from(b)).collect();
    let mut next = 256;
    for b in 0..=u8::MAX {
        if !bytes.contains(&b) {
            bytes.push(b);
            chars.push(next);
            next += 1;
        }
    }
    chars
        .into_iter()
        .filter_map(char::from_u32)
        .zip(bytes)
        .collect()
}

/// A token's bytes as the `ByteLevel` decoder maps them: each character to its
/// byte or, when a character is outside the map (as in an added token's
/// content), the token's own UTF-8.
fn token_bytes(token: &str, char_bytes: &HashMap<char, u8>) -> Vec<u8> {
    token
        .chars()
        .map(|c| char_bytes.get(&c).copied())
        .collect::<Option<Vec<u8>>>()
        .unwrap_or_else(|| token.as_bytes().to_vec())
}

/// Deterministic, so that a failing startup check reproduces.
pub(crate) struct SplitMix64(pub(crate) u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub(crate) fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn below_u32(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }
}

/// Incremental `String::from_utf8_lossy`: bytes pushed piece by piece decode to
/// the text of the whole, while only the bytes of an incomplete trailing
/// character (at most 3) wait for the next piece.
#[derive(Default)]
pub struct Utf8Tail {
    pending: Vec<u8>,
}

impl Utf8Tail {
    /// Append the text of `bytes`, after any pending bytes, to `out`.
    pub fn push(&mut self, bytes: &[u8], out: &mut String) {
        let joined;
        let bytes = if self.pending.is_empty() {
            bytes
        } else {
            self.pending.extend_from_slice(bytes);
            joined = std::mem::take(&mut self.pending);
            joined.as_slice()
        };
        let mut chunks = bytes.utf8_chunks().peekable();
        while let Some(chunk) = chunks.next() {
            out.push_str(chunk.valid());
            let invalid = chunk.invalid();
            if invalid.is_empty() {
                continue;
            }
            let incomplete =
                matches!(std::str::from_utf8(invalid), Err(e) if e.error_len().is_none());
            if incomplete && chunks.peek().is_none() {
                self.pending.extend_from_slice(invalid);
            } else {
                out.push(char::REPLACEMENT_CHARACTER);
            }
        }
    }

    /// Whether the bytes pushed so far end inside a character.
    pub fn is_inside_char(&self) -> bool {
        !self.pending.is_empty()
    }

    /// End the stream: an incomplete trailing character becomes U+FFFD.
    pub fn finish(&mut self, out: &mut String) {
        if !self.pending.is_empty() {
            self.pending.clear();
            out.push(char::REPLACEMENT_CHARACTER);
        }
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Single-byte tokens (ids 0-255), then multi-byte ones, then added tokens.
    pub(crate) const MULTI_BYTE_TOKENS: &[&[u8]] = &[
        "é".as_bytes(),
        // The end of one "é" and the start of the next: every boundary of a
        // run of these falls inside a character.
        &[0xA9, 0xC3],
        // The two halves of U+1F600.
        &[0xF0, 0x9F],
        &[0x98, 0x80],
        " hi".as_bytes(),
        "中".as_bytes(),
        "ST".as_bytes(),
        "OP".as_bytes(),
    ];
    /// (content, special) of the added tokens, after the multi-byte tokens.
    pub(crate) const ADDED_TOKENS: &[(&str, bool)] =
        &[("<eot>", true), ("<|reserved_0|>", true), ("<tool>", false)];

    /// Id of a single byte's token.
    pub(crate) fn byte_id(byte: u8) -> u32 {
        u32::from(byte)
    }

    /// Id of `MULTI_BYTE_TOKENS[i]`.
    pub(crate) fn multi_byte_id(i: usize) -> u32 {
        256 + i as u32
    }

    /// Id of `ADDED_TOKENS[i]`.
    pub(crate) fn added_id(i: usize) -> u32 {
        (256 + MULTI_BYTE_TOKENS.len() + i) as u32
    }

    pub(crate) fn vocab_size() -> u32 {
        added_id(ADDED_TOKENS.len())
    }

    /// Write a byte-level BPE `tokenizer.json` into a fresh directory named
    /// after `name` and load it; returns the file path and the tokenizer.
    pub(crate) fn byte_level_tokenizer(name: &str) -> (String, Tokenizer) {
        let to_char: HashMap<u8, char> = byte_level_char_bytes()
            .into_iter()
            .map(|(c, b)| (b, c))
            .collect();
        let token = |bytes: &[u8]| bytes.iter().map(|b| to_char[b]).collect::<String>();
        let mut vocab = serde_json::Map::new();
        for byte in 0..=u8::MAX {
            vocab.insert(token(&[byte]), byte_id(byte).into());
        }
        for (i, bytes) in MULTI_BYTE_TOKENS.iter().enumerate() {
            vocab.insert(token(bytes), multi_byte_id(i).into());
        }
        let added: Vec<serde_json::Value> = ADDED_TOKENS
            .iter()
            .enumerate()
            .map(|(i, (content, special))| {
                serde_json::json!({
                    "id": added_id(i), "content": content, "single_word": false,
                    "lstrip": false, "rstrip": false, "normalized": false, "special": special,
                })
            })
            .collect();
        let byte_level = serde_json::json!({
            "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true,
        });
        let file = serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": added,
            "normalizer": null,
            "pre_tokenizer": byte_level,
            "post_processor": null,
            "decoder": byte_level,
            "model": {
                "type": "BPE", "dropout": null, "unk_token": null,
                "continuing_subword_prefix": null, "end_of_word_suffix": null,
                "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                "vocab": vocab, "merges": [],
            },
        });
        let dir = std::env::temp_dir().join(format!("byte-tables-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test dir");
        let path = dir.join("tokenizer.json");
        std::fs::write(&path, file.to_string()).expect("test tokenizer.json");
        let path = path.to_string_lossy().into_owned();
        let tokenizer = Tokenizer::from_file(&path).expect("test tokenizer loads");
        (path, tokenizer)
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    #[test]
    fn the_startup_check_accepts_a_byte_level_bpe_tokenizer() {
        let (path, tokenizer) = byte_level_tokenizer("accept");
        let (tables, summary) = build(&path, None, &tokenizer).expect("tables");
        assert_eq!(
            summary,
            format!(
                "checked {} ids and {CHECK_SEQUENCES} sequences",
                vocab_size()
            )
        );
        let eot = i64::from(added_id(0));
        assert_eq!(tables.bytes(false, eot), b"<eot>");
        assert_eq!(tables.bytes(true, eot), b"");
        let tool = i64::from(added_id(2));
        assert_eq!(tables.bytes(true, tool), b"<tool>", "not special");
        assert_eq!(
            tables.bytes(false, i64::from(multi_byte_id(1))),
            [0xA9, 0xC3]
        );
        assert_eq!(tables.bytes(false, i64::from(vocab_size())), b"");
        assert_eq!(tables.bytes(false, -1), b"");
    }

    #[test]
    fn a_tokenizer_without_a_byte_level_decoder_gets_no_tables() {
        let fuse = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/testdata/special_token_tokenizer.json"
        );
        let dir = std::env::temp_dir().join(format!("byte-tables-fuse-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("test dir");
        let path = dir.join("tokenizer.json");
        std::fs::copy(fuse, &path).expect("test tokenizer.json");
        let path = path.to_string_lossy().into_owned();
        let tokenizer = Tokenizer::from_file(&path).expect("tokenizer");
        let Err(reason) = build(&path, None, &tokenizer) else {
            panic!("a Fuse decoder must not get byte tables");
        };
        assert_eq!(reason, "the tokenizer decoder is not ByteLevel");
    }

    /// A decode the tables disagree with, for one id, rejects them.
    #[test]
    fn a_difference_from_the_tokenizer_decode_rejects_the_tables() {
        let (path, tokenizer) = byte_level_tokenizer("reject");
        let json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).expect("file")).expect("json");
        let tokens = ByteLevelTokens::parse(&json).expect("tokens");
        let differ = byte_id(b'x');
        let result = checked_tables(&tokens, vocab_size() as usize, |ids, skip| {
            let text = String::from(tokenizer.decode(ids, skip).map_err(|e| e.to_string())?);
            Ok(if ids == [differ] {
                format!("{text} ")
            } else {
                text
            })
        });
        let Err(reason) = result else {
            panic!("a differing decode must reject the tables");
        };
        assert_eq!(
            reason,
            "id 120 decodes to \"x \" (skip_special_tokens=false)"
        );
    }

    /// `byte_level_tokenizer(name)` with `edit` applied to its `tokenizer.json`,
    /// and `config` (when given) written as the `tokenizer_config.json` beside it.
    fn edited_tokenizer(
        name: &str,
        edit: impl FnOnce(&mut serde_json::Value),
        config: Option<serde_json::Value>,
    ) -> (String, Tokenizer) {
        let (path, _) = byte_level_tokenizer(name);
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("file")).expect("json");
        edit(&mut json);
        std::fs::write(&path, json.to_string()).expect("tokenizer.json");
        if let Some(config) = config {
            let dir = std::path::Path::new(&path).parent().expect("dir");
            std::fs::write(dir.join("tokenizer_config.json"), config.to_string())
                .expect("tokenizer_config.json");
        }
        let tokenizer = Tokenizer::from_file(&path).expect("tokenizer loads");
        (path, tokenizer)
    }

    /// A dense 8,192-id vocabulary plus a special token that only
    /// `tokenizer_config.json` declares, at id 8192.
    fn config_only_token(name: &str, content: &str) -> (String, Tokenizer) {
        let filler = |json: &mut serde_json::Value| {
            let added = json["added_tokens"].as_array().expect("added").clone();
            let vocab = json["model"]["vocab"].as_object_mut().expect("vocab");
            for token in added {
                vocab.insert(
                    token["content"].as_str().expect("content").into(),
                    token["id"].clone(),
                );
            }
            for id in vocab_size()..8192 {
                vocab.insert(format!("filler_{id:04}"), id.into());
            }
        };
        let config = serde_json::json!({
            "added_tokens_decoder": {
                "8192": {"content": content, "special": true, "normalized": false},
            },
        });
        let (path, tokenizer) = edited_tokenizer(name, filler, Some(config));
        assert_eq!(tokenizer.token_to_id(content).expect("lookup"), Some(8192));
        (path, tokenizer)
    }

    /// The bytes of a token that only `tokenizer_config.json` declares are
    /// unknown when its lone decode holds U+FFFD: "XÃ" is bytes 58 C3, so
    /// [it, A9] decodes to "Xé", but its lone decode "X�" says only 58 EF BF BD.
    #[test]
    fn a_config_only_token_with_a_lossy_decode_gets_no_tables() {
        let (path, tokenizer) = config_only_token("config-only-lossy", "XÃ");
        assert_eq!(
            String::from(tokenizer.decode(&[8192, 0xA9], false).expect("decode")),
            "Xé"
        );
        let Err(reason) = build(&path, None, &tokenizer) else {
            panic!("unknown bytes must not get tables");
        };
        assert_eq!(
            reason,
            "id 8192, not in the tokenizer file, decodes to \"X\u{fffd}\", so its bytes are unknown"
        );
    }

    /// A config-only token whose lone decode is whole text is that text, and the
    /// startup sequences include it.
    #[test]
    fn a_config_only_token_with_whole_text_keeps_the_tables() {
        let (path, tokenizer) = config_only_token("config-only-whole", "<cfg>");
        let (tables, summary) = build(&path, None, &tokenizer).expect("tables");
        assert_eq!(
            summary,
            format!("checked 8193 ids and {CHECK_SEQUENCES} sequences")
        );
        assert_eq!(tables.bytes(false, 8192), b"<cfg>");
        assert_eq!(tables.bytes(true, 8192), b"");
    }

    #[test]
    fn an_empty_vocabulary_gets_no_tables() {
        let (path, tokenizer) = edited_tokenizer(
            "empty",
            |json| {
                json["model"]["vocab"] = serde_json::json!({});
                json["added_tokens"] = serde_json::json!([]);
            },
            None,
        );
        assert_eq!(
            build(&path, None, &tokenizer).err().as_deref(),
            Some("the tokenizer has no ids")
        );
    }

    /// A sparse id space is refused before any table is allocated: one token at
    /// id 4294967294 would otherwise need 4,294,967,295 entries.
    #[test]
    fn a_sparse_vocabulary_gets_no_tables() {
        let (path, tokenizer) = edited_tokenizer(
            "sparse",
            |json| {
                json["model"]["vocab"] = serde_json::json!({ "a": 4294967294u32 });
                json["added_tokens"] = serde_json::json!([]);
            },
            None,
        );
        assert_eq!(
            build(&path, None, &tokenizer).err().as_deref(),
            Some("the token ids are sparse: they reach 4294967294 for 1 tokens")
        );
    }

    /// Pushing a byte stream in pieces gives the text of decoding it whole, for
    /// every split of streams with invalid, truncated and split characters.
    #[test]
    fn utf8_tail_matches_a_whole_lossy_decode() {
        let streams: &[&[u8]] = &[
            "aé中😀".as_bytes(),
            &[0xC0, 0x41, 0xFF, 0xC3],
            &[0xA9, 0xC3, 0xA9, 0xC3, 0xA9],
            &[0xF0, 0x9F, 0x98, 0x41, 0xE4, 0xB8],
            &[0xED, 0xA0, 0x80, 0xF4, 0x90, 0x80, 0x80],
        ];
        for stream in streams {
            for split in 0..=stream.len() {
                for second in split..=stream.len() {
                    let mut tail = Utf8Tail::default();
                    let mut out = String::new();
                    tail.push(&stream[..split], &mut out);
                    tail.push(&stream[split..second], &mut out);
                    tail.push(&stream[second..], &mut out);
                    tail.finish(&mut out);
                    assert_eq!(
                        out,
                        String::from_utf8_lossy(stream),
                        "{stream:x?} at {split}, {second}"
                    );
                }
            }
        }
    }
}
