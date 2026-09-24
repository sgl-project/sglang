//! Wire-format types for SGLang's KV cache event stream.
//!
//! SGLang's `ZmqEventPublisher` (Python:
//! `python/sglang/srt/disaggregation/kv_events.py`) encodes batches with
//! `msgspec.msgpack`. Two struct families are involved:
//!
//! * `EventBatch` (the outer payload) — declared with
//!   `array_like=True, gc=False` (no tag), so it is a msgpack **array**
//!   `[ts, events, attn_dp_rank]`.
//! * `KVCacheEvent` (each inner event variant) — declared with
//!   `omit_defaults=True, tag=True` and no `array_like`, so each event is a
//!   msgpack **map** whose `type` key carries the class name and whose other
//!   keys are field names. Optional fields left at `None` are omitted. This
//!   is the same encoding vLLM uses for its KV events.
//!
//! Older publishers emitted each event as a tagged **array**
//! `[class_name_str, field1, ...]`; that shape is rejected.
//!
//! This module deserializes those bytes into Rust types and exposes a single
//! [`decode_event_batch`] entry point.

use std::fmt;

use serde::de::{self, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::Deserialize;

/// Top-level batch payload published by SGLang.
///
/// Wire shape (`EventBatch`, `array_like`):
/// `[ts: f64, events: [...], attn_dp_rank: int_or_nil]`.
/// SGLang declares `attn_dp_rank` as a Python `Optional[int]`; we decode
/// it as `u32` since DP ranks are non-negative and bounded by the
/// publisher's `dp_size`.
#[derive(Debug, Clone, PartialEq)]
pub struct KvEventBatch {
    /// Wall-clock timestamp from the publisher (seconds since epoch).
    pub ts: f64,
    /// Ordered list of cache events in this batch.
    pub events: Vec<KvCacheEvent>,
    /// Optional DP-attention rank that produced this batch. `None` if the
    /// publisher emitted nil. The decoder also accepts an omitted field for
    /// compatibility.
    pub attn_dp_rank: Option<u32>,
}

/// A single KV cache event. The Python base class `KVCacheEvent` uses
/// `tag=True`, so each event carries its class name under the `type` key.
#[derive(Debug, Clone, PartialEq)]
pub enum KvCacheEvent {
    /// `{"type": "BlockStored", "block_hashes", "parent_block_hash",
    /// "token_ids", "block_size", "lora_id", "medium"?, ...}`. Keys the
    /// gateway does not route on (`cache_salt`, `session_id`) are ignored.
    BlockStored(BlockStored),
    /// `{"type": "BlockRemoved", "block_hashes", "medium"?}`.
    BlockRemoved(BlockRemoved),
    /// `{"type": "AllBlocksCleared"}`.
    AllBlocksCleared,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockStored {
    /// 64-bit block hashes in declaration order. Hashes can exceed `i32`
    /// range; signedness matches SGLang's Python `int`.
    pub block_hashes: Vec<i64>,
    /// Hash of the parent block, or `None` for the first block in a chain.
    pub parent_block_hash: Option<i64>,
    /// Tokens covered by this block. SGLang uses 32-bit token IDs.
    pub token_ids: Vec<u32>,
    /// Block size (tokens per block).
    pub block_size: u32,
    /// LoRA adapter ID this block is associated with, if any.
    pub lora_id: Option<i64>,
    /// Storage tier (`"GPU"`, `"CPU_PINNED"`, `"DISK"`, `"EXTERNAL"`).
    /// Optional in the Python schema (`= None` default), so it may be
    /// omitted entirely under `omit_defaults`.
    pub medium: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockRemoved {
    pub block_hashes: Vec<i64>,
    /// Same semantics as [`BlockStored::medium`].
    pub medium: Option<String>,
}

/// Maximum number of block hashes a single decoded `BlockStored` /
/// `BlockRemoved` event may carry. A misbehaving worker (or a corrupted
/// frame) could otherwise prompt a multi-gigabyte allocation in the
/// gateway. Workers are inside the trust boundary, so this is
/// defense-in-depth — but the cost of *not* capping is unbounded memory
/// amplification, so we cap.
pub(crate) const MAX_HASHES_PER_EVENT: usize = 65_536;
/// Same rationale as [`MAX_HASHES_PER_EVENT`], but for `token_ids`. A
/// 1M-token block list is already absurdly larger than any realistic
/// `BlockStored` payload — the cap exists to bound the worst case, not
/// to constrain normal operation.
pub(crate) const MAX_TOKENS_PER_EVENT: usize = 1_048_576;

/// Errors produced by [`decode_event_batch`].
#[derive(thiserror::Error, Debug)]
pub enum DecodeError {
    /// The msgpack payload was malformed or did not match the expected schema.
    #[error("failed to decode KV event batch: {0}")]
    Msgpack(#[from] rmp_serde::decode::Error),
    /// A single event's variable-length field exceeded its hard cap. We
    /// surface this as an error rather than panicking so a single bad
    /// payload only kills its batch, not the consumer task.
    #[error("KV event field {field} length {len} exceeds cap {cap}")]
    PayloadTooLarge {
        field: &'static str,
        len: usize,
        cap: usize,
    },
}

/// Sentinel string a custom visitor uses to encode a "field too large"
/// error through serde's `de::Error::custom` channel. We rewrap as the
/// typed [`DecodeError::PayloadTooLarge`] in [`decode_event_batch`].
const PAYLOAD_TOO_LARGE_TAG: &str = "kv_events::wire::PAYLOAD_TOO_LARGE";

/// Decode a single ZMQ payload frame from SGLang's `ZmqEventPublisher`.
///
/// The payload is the `payload` arg to `_pub.send_multipart((topic, seq,
/// payload))` — the topic and 8-byte big-endian sequence number are separate
/// frames and are NOT part of the msgpack input here.
///
/// Caps the per-event `block_hashes` and `token_ids` lengths
/// ([`MAX_HASHES_PER_EVENT`], [`MAX_TOKENS_PER_EVENT`]) so a misbehaving
/// worker — or a corrupted msgpack length prefix — cannot trigger an
/// unbounded allocation in the gateway.
pub fn decode_event_batch(bytes: &[u8]) -> Result<KvEventBatch, DecodeError> {
    match rmp_serde::from_slice::<KvEventBatch>(bytes) {
        Ok(b) => Ok(b),
        Err(e) => {
            // Rewrap the size-cap sentinel into the typed variant. The
            // sentinel string is set by `BoundedI64Vec` / `BoundedU32Vec`
            // below; everything else is a true msgpack decode failure.
            let s = e.to_string();
            if let Some(rest) = s.strip_prefix(PAYLOAD_TOO_LARGE_TAG) {
                // Format: "<TAG>:<field>:<len>:<cap>"
                let mut parts = rest.trim_start_matches(':').split(':');
                if let (Some(field), Some(len), Some(cap)) =
                    (parts.next(), parts.next(), parts.next())
                {
                    if let (Ok(len), Ok(cap)) = (len.parse::<usize>(), cap.parse::<usize>()) {
                        let field = match field {
                            "block_hashes" => "block_hashes",
                            "token_ids" => "token_ids",
                            // Unknown — fall through to Msgpack.
                            _ => return Err(DecodeError::Msgpack(e)),
                        };
                        return Err(DecodeError::PayloadTooLarge { field, len, cap });
                    }
                }
            }
            Err(DecodeError::Msgpack(e))
        }
    }
}

/// Newtype wrapping `Vec<i64>` whose `Deserialize` impl rejects sequences
/// announcing more than [`MAX_HASHES_PER_EVENT`] elements *before* doing
/// the per-element work. Required because `rmp-serde` pre-sizes the
/// destination `Vec` from the msgpack length prefix; a malicious or
/// corrupted prefix would otherwise prompt a multi-gigabyte allocation.
#[derive(Debug, Clone, PartialEq)]
struct BoundedI64Vec(Vec<i64>);

impl<'de> Deserialize<'de> for BoundedI64Vec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<i64>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack array of i64 values")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Vec<i64>, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(hint) = seq.size_hint() {
                    if hint > MAX_HASHES_PER_EVENT {
                        return Err(de::Error::custom(format!(
                            "{PAYLOAD_TOO_LARGE_TAG}:block_hashes:{hint}:{MAX_HASHES_PER_EVENT}"
                        )));
                    }
                }
                let mut out: Vec<i64> = match seq.size_hint() {
                    Some(h) => Vec::with_capacity(h),
                    None => Vec::new(),
                };
                while let Some(v) = seq.next_element::<i64>()? {
                    if out.len() >= MAX_HASHES_PER_EVENT {
                        return Err(de::Error::custom(format!(
                            "{PAYLOAD_TOO_LARGE_TAG}:block_hashes:{}:{MAX_HASHES_PER_EVENT}",
                            out.len() + 1
                        )));
                    }
                    out.push(v);
                }
                Ok(out)
            }
        }
        let v = deserializer.deserialize_seq(V)?;
        Ok(BoundedI64Vec(v))
    }
}

/// One element of a `token_ids` array. SGLang emits a flat `u32` per token for
/// unigram pages, but a 2-element `[t_i, t_{i+1}]` array per token for *bigram*
/// pages (`mem_cache/events.py`, `is_bigram` branch — DeepSeek-V4-class models).
/// `token_ids` is purely informational for the gateway (routing keys off the
/// engine-provided `block_hashes`), so we accept either shape and flatten the
/// ints rather than model the bigram pairing.
enum TokenCell {
    One(u32),
    Many(Vec<u32>),
}

impl<'de> Deserialize<'de> for TokenCell {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = TokenCell;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a token id (u32) or an array of token ids")
            }
            // serde's default visit_u8/u16/u32 forward to visit_u64, and
            // visit_i8/i16/i32 forward to visit_i64, so these two cover every
            // integer width msgpack might use for a scalar token id.
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<TokenCell, E> {
                Ok(TokenCell::One(v as u32))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<TokenCell, E> {
                Ok(TokenCell::One(v as u32))
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<TokenCell, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut ts: Vec<u32> = match seq.size_hint() {
                    Some(h) => Vec::with_capacity(h.min(8)),
                    None => Vec::new(),
                };
                while let Some(t) = seq.next_element::<u32>()? {
                    if ts.len() >= MAX_TOKENS_PER_EVENT {
                        return Err(de::Error::custom(format!(
                            "{PAYLOAD_TOO_LARGE_TAG}:token_ids:{}:{MAX_TOKENS_PER_EVENT}",
                            ts.len() + 1
                        )));
                    }
                    ts.push(t);
                }
                Ok(TokenCell::Many(ts))
            }
        }
        deserializer.deserialize_any(V)
    }
}

/// `BoundedI64Vec`'s `u32` twin. Same shape, different cap. Accepts both flat
/// (unigram) token ids and bigram `[t_i, t_{i+1}]` pairs via [`TokenCell`],
/// flattening the latter.
#[derive(Debug, Clone, PartialEq)]
struct BoundedU32Vec(Vec<u32>);

impl<'de> Deserialize<'de> for BoundedU32Vec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vec<u32>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack array of u32 values")
            }
            fn visit_seq<A>(self, mut seq: A) -> Result<Vec<u32>, A::Error>
            where
                A: SeqAccess<'de>,
            {
                if let Some(hint) = seq.size_hint() {
                    if hint > MAX_TOKENS_PER_EVENT {
                        return Err(de::Error::custom(format!(
                            "{PAYLOAD_TOO_LARGE_TAG}:token_ids:{hint}:{MAX_TOKENS_PER_EVENT}"
                        )));
                    }
                }
                let mut out: Vec<u32> = match seq.size_hint() {
                    Some(h) => Vec::with_capacity(h),
                    None => Vec::new(),
                };
                // Each element is either a scalar token id (unigram) or a
                // `[t_i, t_{i+1}]` pair (bigram); flatten both into `out`.
                while let Some(cell) = seq.next_element::<TokenCell>()? {
                    let push = |t: u32, out: &mut Vec<u32>| -> Result<(), A::Error> {
                        if out.len() >= MAX_TOKENS_PER_EVENT {
                            return Err(de::Error::custom(format!(
                                "{PAYLOAD_TOO_LARGE_TAG}:token_ids:{}:{MAX_TOKENS_PER_EVENT}",
                                out.len() + 1
                            )));
                        }
                        out.push(t);
                        Ok(())
                    };
                    match cell {
                        TokenCell::One(t) => push(t, &mut out)?,
                        TokenCell::Many(ts) => {
                            for t in ts {
                                push(t, &mut out)?;
                            }
                        }
                    }
                }
                Ok(out)
            }
        }
        let v = deserializer.deserialize_seq(V)?;
        Ok(BoundedU32Vec(v))
    }
}

// ---------------------------------------------------------------------------
// Custom Deserialize impls — the batch is a msgpack array; each event is a
// tagged map. Optional fields may be absent or nil.
// ---------------------------------------------------------------------------

impl<'de> Deserialize<'de> for KvEventBatch {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct BatchVisitor;

        impl<'de> Visitor<'de> for BatchVisitor {
            type Value = KvEventBatch;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a msgpack array [ts, events, attn_dp_rank?]")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<KvEventBatch, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let ts: f64 = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("ts"))?;
                let events: Vec<KvCacheEvent> = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::missing_field("events"))?;
                // attn_dp_rank may be present-as-nil, present-as-int, or
                // omitted entirely under msgspec's `omit_defaults`.
                let attn_dp_rank: Option<u32> = seq.next_element()?.unwrap_or(None);
                // Drain any extra trailing fields a future schema might add
                // (forward-compat).
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(KvEventBatch {
                    ts,
                    events,
                    attn_dp_rank,
                })
            }
        }

        deserializer.deserialize_seq(BatchVisitor)
    }
}

impl<'de> Deserialize<'de> for KvCacheEvent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        /// Map keys of a `KVCacheEvent` the gateway reads. Every other key
        /// (`cache_salt`, `session_id`, future additions) is skipped.
        enum EventField {
            Type,
            BlockHashes,
            ParentBlockHash,
            TokenIds,
            BlockSize,
            LoraId,
            Medium,
            Other,
        }

        impl<'de> Deserialize<'de> for EventField {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;
                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = EventField;
                    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                        f.write_str("a KV event field name")
                    }
                    fn visit_str<E: de::Error>(self, v: &str) -> Result<EventField, E> {
                        Ok(match v {
                            "type" => EventField::Type,
                            "block_hashes" => EventField::BlockHashes,
                            "parent_block_hash" => EventField::ParentBlockHash,
                            "token_ids" => EventField::TokenIds,
                            "block_size" => EventField::BlockSize,
                            "lora_id" => EventField::LoraId,
                            "medium" => EventField::Medium,
                            _ => EventField::Other,
                        })
                    }
                }
                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct EventVisitor;

        impl<'de> Visitor<'de> for EventVisitor {
            type Value = KvCacheEvent;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a tagged msgpack map {\"type\": class_name, ...fields}")
            }

            fn visit_map<A>(self, mut map: A) -> Result<KvCacheEvent, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tag: Option<String> = None;
                let mut block_hashes: Option<BoundedI64Vec> = None;
                let mut parent_block_hash: Option<i64> = None;
                let mut token_ids: Option<BoundedU32Vec> = None;
                let mut block_size: Option<u32> = None;
                let mut lora_id: Option<i64> = None;
                let mut medium: Option<String> = None;
                while let Some(field) = map.next_key::<EventField>()? {
                    match field {
                        EventField::Type => tag = Some(map.next_value()?),
                        EventField::BlockHashes => block_hashes = Some(map.next_value()?),
                        // Optional fields may be present as nil or omitted entirely.
                        EventField::ParentBlockHash => parent_block_hash = map.next_value()?,
                        EventField::TokenIds => token_ids = Some(map.next_value()?),
                        EventField::BlockSize => block_size = Some(map.next_value()?),
                        EventField::LoraId => lora_id = map.next_value()?,
                        EventField::Medium => medium = map.next_value()?,
                        EventField::Other => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                let tag = tag.ok_or_else(|| de::Error::missing_field("type"))?;
                match tag.as_str() {
                    "BlockStored" => Ok(KvCacheEvent::BlockStored(BlockStored {
                        block_hashes: block_hashes
                            .ok_or_else(|| de::Error::missing_field("block_hashes"))?
                            .0,
                        parent_block_hash,
                        token_ids: token_ids
                            .ok_or_else(|| de::Error::missing_field("token_ids"))?
                            .0,
                        block_size: block_size
                            .ok_or_else(|| de::Error::missing_field("block_size"))?,
                        lora_id,
                        medium,
                    })),
                    "BlockRemoved" => Ok(KvCacheEvent::BlockRemoved(BlockRemoved {
                        block_hashes: block_hashes
                            .ok_or_else(|| de::Error::missing_field("block_hashes"))?
                            .0,
                        medium,
                    })),
                    "AllBlocksCleared" => Ok(KvCacheEvent::AllBlocksCleared),
                    other => Err(de::Error::unknown_variant(
                        other,
                        &["BlockStored", "BlockRemoved", "AllBlocksCleared"],
                    )),
                }
            }
        }

        deserializer.deserialize_map(EventVisitor)
    }
}

// ---------------------------------------------------------------------------
// Tests — golden bytes are constructed via the `rmp` low-level encoder so
// they exercise the exact msgpack map layout SGLang emits, independent of
// any Rust-side serializer.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use rmp::encode as mp;

    fn write_key(buf: &mut Vec<u8>, key: &str) {
        mp::write_str(buf, key).unwrap();
    }

    fn write_opt_sint(buf: &mut Vec<u8>, value: Option<i64>) {
        match value {
            Some(v) => {
                mp::write_sint(buf, v).unwrap();
            }
            None => mp::write_nil(buf).unwrap(),
        }
    }

    fn write_i64_array(buf: &mut Vec<u8>, values: &[i64]) {
        mp::write_array_len(buf, values.len() as u32).unwrap();
        for v in values {
            mp::write_sint(buf, *v).unwrap();
        }
    }

    fn write_u32_array(buf: &mut Vec<u8>, values: &[u32]) {
        mp::write_array_len(buf, values.len() as u32).unwrap();
        for v in values {
            mp::write_uint(buf, *v as u64).unwrap();
        }
    }

    /// Encode `token_ids` the way SGLang's *bigram* pages do: a sequence of
    /// 2-element `[t_i, t_{i+1}]` arrays instead of flat ints. See
    /// `mem_cache/events.py` (`is_bigram` branch).
    fn write_bigram_token_array(buf: &mut Vec<u8>, pairs: &[(u32, u32)]) {
        mp::write_array_len(buf, pairs.len() as u32).unwrap();
        for (a, b) in pairs {
            mp::write_array_len(buf, 2).unwrap();
            mp::write_uint(buf, *a as u64).unwrap();
            mp::write_uint(buf, *b as u64).unwrap();
        }
    }

    /// Start a tagged event map with `field_count` fields after `type`.
    fn write_event_map(buf: &mut Vec<u8>, tag: &str, field_count: u32) {
        mp::write_map_len(buf, field_count + 1).unwrap();
        write_key(buf, "type");
        mp::write_str(buf, tag).unwrap();
    }

    fn block_stored_field_count(medium: Option<&str>, extra: &[(&str, &str)]) -> u32 {
        5 + u32::from(medium.is_some()) + extra.len() as u32
    }

    /// Write the `BlockStored` fields that follow `type`. `parent_block_hash`
    /// and `lora_id` have no default in the Python schema, so msgspec always
    /// emits them (nil when unset); `medium` is omitted when `None`; `extra`
    /// adds string-valued keys the gateway must ignore.
    #[allow(clippy::too_many_arguments)]
    fn write_block_stored_fields(
        buf: &mut Vec<u8>,
        block_hashes: &[i64],
        parent: Option<i64>,
        write_tokens: impl FnOnce(&mut Vec<u8>),
        block_size: u32,
        lora_id: Option<i64>,
        medium: Option<&str>,
        extra: &[(&str, &str)],
    ) {
        write_key(buf, "block_hashes");
        write_i64_array(buf, block_hashes);
        write_key(buf, "parent_block_hash");
        write_opt_sint(buf, parent);
        write_key(buf, "token_ids");
        write_tokens(buf);
        write_key(buf, "block_size");
        mp::write_uint(buf, block_size as u64).unwrap();
        write_key(buf, "lora_id");
        write_opt_sint(buf, lora_id);
        if let Some(m) = medium {
            write_key(buf, "medium");
            mp::write_str(buf, m).unwrap();
        }
        for (key, value) in extra {
            write_key(buf, key);
            mp::write_str(buf, value).unwrap();
        }
    }

    /// Build a `BlockStored` map as msgspec emits it, plus `extra` keys.
    #[allow(clippy::too_many_arguments)]
    fn build_block_stored_bytes_with_extra(
        block_hashes: &[i64],
        parent: Option<i64>,
        token_ids: &[u32],
        block_size: u32,
        lora_id: Option<i64>,
        medium: Option<&str>,
        extra: &[(&str, &str)],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        write_event_map(
            &mut buf,
            "BlockStored",
            block_stored_field_count(medium, extra),
        );
        write_block_stored_fields(
            &mut buf,
            block_hashes,
            parent,
            |b| write_u32_array(b, token_ids),
            block_size,
            lora_id,
            medium,
            extra,
        );
        buf
    }

    fn build_block_stored_bytes(
        block_hashes: &[i64],
        parent: Option<i64>,
        token_ids: &[u32],
        block_size: u32,
        lora_id: Option<i64>,
        medium: Option<&str>,
    ) -> Vec<u8> {
        build_block_stored_bytes_with_extra(
            block_hashes,
            parent,
            token_ids,
            block_size,
            lora_id,
            medium,
            &[],
        )
    }

    /// Like `build_block_stored_bytes`, but `token_ids` is the bigram
    /// list-of-pairs shape that DeepSeek-V4-class models emit.
    fn build_block_stored_bigram_bytes(
        block_hashes: &[i64],
        parent: Option<i64>,
        token_pairs: &[(u32, u32)],
        block_size: u32,
        lora_id: Option<i64>,
        medium: Option<&str>,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        write_event_map(
            &mut buf,
            "BlockStored",
            block_stored_field_count(medium, &[]),
        );
        write_block_stored_fields(
            &mut buf,
            block_hashes,
            parent,
            |b| write_bigram_token_array(b, token_pairs),
            block_size,
            lora_id,
            medium,
            &[],
        );
        buf
    }

    fn build_block_removed_bytes(block_hashes: &[i64], medium: Option<&str>) -> Vec<u8> {
        let mut buf = Vec::new();
        write_event_map(&mut buf, "BlockRemoved", 1 + u32::from(medium.is_some()));
        write_key(&mut buf, "block_hashes");
        write_i64_array(&mut buf, block_hashes);
        if let Some(m) = medium {
            write_key(&mut buf, "medium");
            mp::write_str(&mut buf, m).unwrap();
        }
        buf
    }

    fn build_all_blocks_cleared_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        write_event_map(&mut buf, "AllBlocksCleared", 0);
        buf
    }

    /// Wrap pre-encoded event bytes into a top-level KVEventBatch array
    /// `[ts, [event0_bytes, event1_bytes, ...], attn_dp_rank_or_nil]`.
    fn build_batch_bytes(
        ts: f64,
        event_bufs: &[Vec<u8>],
        attn_dp_rank: Option<u32>,
        include_dp_field: bool,
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        let total_len = if include_dp_field { 3 } else { 2 };
        mp::write_array_len(&mut buf, total_len).unwrap();
        mp::write_f64(&mut buf, ts).unwrap();
        mp::write_array_len(&mut buf, event_bufs.len() as u32).unwrap();
        for ev in event_bufs {
            buf.extend_from_slice(ev);
        }
        if include_dp_field {
            match attn_dp_rank {
                Some(v) => {
                    mp::write_uint(&mut buf, v as u64).unwrap();
                }
                None => mp::write_nil(&mut buf).unwrap(),
            }
        }
        buf
    }

    /// Regression: bigram models (e.g. DeepSeek-V4-Flash) emit `token_ids` as
    /// `[[t_i, t_{i+1}], ...]`. The decoder previously read `token_ids` as a
    /// flat `u32` array and failed the entire batch with
    /// "wrong msgpack marker FixArray(2)", silently disabling cache-aware
    /// routing. It must instead accept the bigram shape (flattening the ints).
    #[test]
    fn decodes_block_stored_with_bigram_token_ids() {
        let event = build_block_stored_bigram_bytes(
            &[111_i64],
            None,
            &[(10, 20), (20, 30)],
            2,
            None,
            Some("GPU"),
        );
        let bytes = build_batch_bytes(1.5, &[event], Some(0), true);

        let batch = decode_event_batch(&bytes).expect("decode bigram token_ids");
        assert_eq!(batch.events.len(), 1);
        match &batch.events[0] {
            KvCacheEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes, vec![111]);
                assert_eq!(b.parent_block_hash, None);
                assert_eq!(b.block_size, 2);
                assert_eq!(b.token_ids, vec![10, 20, 20, 30]);
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
    }

    #[test]
    fn decodes_block_stored_with_all_fields() {
        let event = build_block_stored_bytes(
            &[1234567890123_i64, -987654321_i64],
            Some(42),
            &[10, 20, 30, 40],
            4,
            Some(7),
            Some("GPU"),
        );
        let bytes = build_batch_bytes(123.456, &[event], Some(2), true);

        let batch = decode_event_batch(&bytes).expect("decode");
        assert_eq!(batch.ts, 123.456);
        assert_eq!(batch.attn_dp_rank, Some(2));
        assert_eq!(batch.events.len(), 1);
        match &batch.events[0] {
            KvCacheEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes, vec![1234567890123_i64, -987654321_i64]);
                assert_eq!(b.parent_block_hash, Some(42));
                assert_eq!(b.token_ids, vec![10, 20, 30, 40]);
                assert_eq!(b.block_size, 4);
                assert_eq!(b.lora_id, Some(7));
                assert_eq!(b.medium.as_deref(), Some("GPU"));
            }
            other => panic!("expected BlockStored, got {other:?}"),
        }
    }

    /// `parent_block_hash` and `lora_id` present as nil, `medium` key omitted
    /// (msgspec `omit_defaults`).
    #[test]
    fn decodes_block_stored_with_nil_and_omitted_optionals() {
        let event = build_block_stored_bytes(&[1, 2, 3], None, &[5, 6], 16, None, None);
        let bytes = build_batch_bytes(0.0, &[event], None, true);

        let batch = decode_event_batch(&bytes).expect("decode");
        match &batch.events[0] {
            KvCacheEvent::BlockStored(b) => {
                assert_eq!(b.parent_block_hash, None);
                assert_eq!(b.lora_id, None);
                assert_eq!(b.medium, None);
                assert_eq!(b.block_size, 16);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    /// Attribution keys (`cache_salt`, `session_id`) and any future key the
    /// gateway does not route on are skipped, not rejected.
    #[test]
    fn unknown_keys_are_ignored() {
        let event = build_block_stored_bytes_with_extra(
            &[10],
            Some(1),
            &[1, 2],
            2,
            None,
            Some("GPU"),
            &[
                ("cache_salt", "tenant-a"),
                ("session_id", "session-a"),
                ("future_key", "x"),
            ],
        );
        let bytes = build_batch_bytes(0.0, &[event], None, true);

        let batch = decode_event_batch(&bytes).expect("decode");
        match &batch.events[0] {
            KvCacheEvent::BlockStored(b) => {
                assert_eq!(b.block_hashes, vec![10]);
                assert_eq!(b.parent_block_hash, Some(1));
                assert_eq!(b.medium.as_deref(), Some("GPU"));
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn decodes_block_removed() {
        let event = build_block_removed_bytes(&[100, 200], Some("DISK"));
        let bytes = build_batch_bytes(1.0, &[event], Some(0), true);

        let batch = decode_event_batch(&bytes).expect("decode");
        match &batch.events[0] {
            KvCacheEvent::BlockRemoved(r) => {
                assert_eq!(r.block_hashes, vec![100, 200]);
                assert_eq!(r.medium.as_deref(), Some("DISK"));
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn medium_omitted_in_block_removed_decodes_as_none() {
        let event = build_block_removed_bytes(&[42], None);
        let bytes = build_batch_bytes(0.0, &[event], None, true);

        let batch = decode_event_batch(&bytes).expect("decode");
        match &batch.events[0] {
            KvCacheEvent::BlockRemoved(r) => {
                assert_eq!(r.block_hashes, vec![42]);
                assert_eq!(r.medium, None);
            }
            other => panic!("unexpected variant: {other:?}"),
        }
    }

    #[test]
    fn decodes_all_blocks_cleared() {
        let event = build_all_blocks_cleared_bytes();
        let bytes = build_batch_bytes(2.0, &[event], None, true);

        let batch = decode_event_batch(&bytes).expect("decode");
        assert_eq!(batch.events.len(), 1);
        assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));
    }

    #[test]
    fn decodes_mixed_batch_preserving_order() {
        let stored = build_block_stored_bytes(&[10], Some(1), &[1, 2], 2, None, Some("GPU"));
        let removed = build_block_removed_bytes(&[20], None);
        let cleared = build_all_blocks_cleared_bytes();
        let bytes = build_batch_bytes(99.0, &[stored, removed, cleared], Some(3), true);

        let batch = decode_event_batch(&bytes).expect("decode");
        assert_eq!(batch.events.len(), 3);
        assert!(matches!(batch.events[0], KvCacheEvent::BlockStored(_)));
        assert!(matches!(batch.events[1], KvCacheEvent::BlockRemoved(_)));
        assert!(matches!(batch.events[2], KvCacheEvent::AllBlocksCleared));
        assert_eq!(batch.attn_dp_rank, Some(3));
    }

    #[test]
    fn attn_dp_rank_omitted_decodes_as_none() {
        // Accept a payload that omits the optional trailing rank field.
        let event = build_all_blocks_cleared_bytes();
        let bytes = build_batch_bytes(5.0, &[event], None, /* include_dp_field */ false);

        let batch = decode_event_batch(&bytes).expect("decode");
        assert_eq!(batch.attn_dp_rank, None);
        assert_eq!(batch.events.len(), 1);
    }

    #[test]
    fn unknown_event_tag_is_rejected() {
        let mut buf = Vec::new();
        write_event_map(&mut buf, "MysteryEvent", 0);
        let bytes = build_batch_bytes(0.0, &[buf], None, true);

        let err = decode_event_batch(&bytes).expect_err("should reject unknown variant");
        let msg = format!("{err}");
        assert!(
            msg.contains("MysteryEvent") || msg.contains("unknown variant"),
            "unexpected error message: {msg}"
        );
    }

    #[test]
    fn event_without_type_is_rejected() {
        let mut buf = Vec::new();
        mp::write_map_len(&mut buf, 1).unwrap();
        write_key(&mut buf, "block_hashes");
        write_i64_array(&mut buf, &[1]);
        let bytes = build_batch_bytes(0.0, &[buf], None, true);

        let err = decode_event_batch(&bytes).expect_err("missing type must fail");
        assert!(format!("{err}").contains("type"), "unexpected error: {err}");
    }

    #[test]
    fn block_stored_without_block_hashes_is_rejected() {
        let mut buf = Vec::new();
        write_event_map(&mut buf, "BlockStored", 2);
        write_key(&mut buf, "token_ids");
        write_u32_array(&mut buf, &[1]);
        write_key(&mut buf, "block_size");
        mp::write_uint(&mut buf, 1).unwrap();
        let bytes = build_batch_bytes(0.0, &[buf], None, true);

        let err = decode_event_batch(&bytes).expect_err("missing block_hashes must fail");
        assert!(
            format!("{err}").contains("block_hashes"),
            "unexpected error: {err}"
        );
    }

    /// The pre-map publishers encoded events as tagged arrays; that shape is
    /// no longer produced and is rejected rather than half-decoded.
    #[test]
    fn legacy_array_event_is_rejected() {
        let mut buf = Vec::new();
        mp::write_array_len(&mut buf, 7).unwrap();
        mp::write_str(&mut buf, "BlockStored").unwrap();
        write_i64_array(&mut buf, &[1]);
        mp::write_nil(&mut buf).unwrap();
        write_u32_array(&mut buf, &[1, 2]);
        mp::write_uint(&mut buf, 2).unwrap();
        mp::write_nil(&mut buf).unwrap();
        mp::write_str(&mut buf, "GPU").unwrap();
        let bytes = build_batch_bytes(0.0, &[buf], None, true);

        assert!(matches!(
            decode_event_batch(&bytes),
            Err(DecodeError::Msgpack(_))
        ));
    }

    /// Golden bytes captured from the SGLang Python publisher
    /// (`msgspec.msgpack.Encoder().encode(KVEventBatch(...))`), msgspec 0.21.1,
    /// against the schema in `python/sglang/srt/disaggregation/kv_events.py`.
    /// They lock down the exact wire format the decoder consumes.
    mod msgspec_golden {
        use super::super::*;

        fn hex_to_bytes(s: &str) -> Vec<u8> {
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        }

        #[test]
        fn full_block_stored() {
            // EventBatch(ts=123.456, events=[BlockStored([1234567890123, -987654321],
            //   parent=42, tokens=[10,20,30,40], block_size=4, lora=7, medium="GPU")],
            //   attn_dp_rank=2)
            let bytes = hex_to_bytes("93cb405edd2f1a9fbe779187a474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657392cf0000011f71fb04cbd2c521974fb1706172656e745f626c6f636b5f686173682aa9746f6b656e5f696473940a141e28aa626c6f636b5f73697a6504a76c6f72615f696407a66d656469756da347505502");
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.ts, 123.456);
            assert_eq!(batch.attn_dp_rank, Some(2));
            assert_eq!(batch.events.len(), 1);
            match &batch.events[0] {
                KvCacheEvent::BlockStored(b) => {
                    assert_eq!(b.block_hashes, vec![1234567890123_i64, -987654321_i64]);
                    assert_eq!(b.parent_block_hash, Some(42));
                    assert_eq!(b.token_ids, vec![10, 20, 30, 40]);
                    assert_eq!(b.block_size, 4);
                    assert_eq!(b.lora_id, Some(7));
                    assert_eq!(b.medium.as_deref(), Some("GPU"));
                }
                other => panic!("expected BlockStored, got {other:?}"),
            }
        }

        #[test]
        fn block_stored_with_nil_optionals() {
            // ts=0.0, BlockStored([1,2,3], parent=None, tokens=[5,6], block_size=16,
            //   lora=None, medium=None -> key omitted), attn_dp_rank=None
            let bytes = hex_to_bytes("93cb00000000000000009186a474797065ab426c6f636b53746f726564ac626c6f636b5f68617368657393010203b1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f696473920506aa626c6f636b5f73697a6510a76c6f72615f6964c0c0");
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.attn_dp_rank, None);
            match &batch.events[0] {
                KvCacheEvent::BlockStored(b) => {
                    assert_eq!(b.block_hashes, vec![1, 2, 3]);
                    assert_eq!(b.parent_block_hash, None);
                    assert_eq!(b.token_ids, vec![5, 6]);
                    assert_eq!(b.block_size, 16);
                    assert_eq!(b.lora_id, None);
                    assert_eq!(b.medium, None);
                }
                other => panic!("unexpected: {other:?}"),
            }
        }

        #[test]
        fn block_removed_with_medium() {
            // ts=1.0, [BlockRemoved([100, 200], medium="DISK")], attn_dp_rank=0
            let bytes = hex_to_bytes("93cb3ff00000000000009183a474797065ac426c6f636b52656d6f766564ac626c6f636b5f6861736865739264ccc8a66d656469756da44449534b00");
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.ts, 1.0);
            assert_eq!(batch.attn_dp_rank, Some(0));
            match &batch.events[0] {
                KvCacheEvent::BlockRemoved(r) => {
                    assert_eq!(r.block_hashes, vec![100, 200]);
                    assert_eq!(r.medium.as_deref(), Some("DISK"));
                }
                other => panic!("unexpected: {other:?}"),
            }
        }

        #[test]
        fn all_blocks_cleared() {
            // ts=2.0, [AllBlocksCleared()], attn_dp_rank=None
            let bytes = hex_to_bytes(
                "93cb40000000000000009181a474797065b0416c6c426c6f636b73436c6561726564c0",
            );
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.ts, 2.0);
            assert_eq!(batch.attn_dp_rank, None);
            assert_eq!(batch.events.len(), 1);
            assert!(matches!(batch.events[0], KvCacheEvent::AllBlocksCleared));
        }

        #[test]
        fn mixed_batch() {
            // ts=99.0, [BlockStored, BlockRemoved, AllBlocksCleared], attn_dp_rank=3
            let bytes = hex_to_bytes("93cb4058c000000000009387a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573910ab1706172656e745f626c6f636b5f6861736801a9746f6b656e5f696473920102aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da347505582a474797065ac426c6f636b52656d6f766564ac626c6f636b5f686173686573911481a474797065b0416c6c426c6f636b73436c656172656403");
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.ts, 99.0);
            assert_eq!(batch.attn_dp_rank, Some(3));
            assert_eq!(batch.events.len(), 3);
            match &batch.events[0] {
                KvCacheEvent::BlockStored(b) => {
                    assert_eq!(b.block_hashes, vec![10]);
                    assert_eq!(b.parent_block_hash, Some(1));
                    assert_eq!(b.token_ids, vec![1, 2]);
                    assert_eq!(b.block_size, 2);
                    assert_eq!(b.lora_id, None);
                    assert_eq!(b.medium.as_deref(), Some("GPU"));
                }
                other => panic!("unexpected: {other:?}"),
            }
            match &batch.events[1] {
                KvCacheEvent::BlockRemoved(r) => {
                    assert_eq!(r.block_hashes, vec![20]);
                    assert_eq!(r.medium, None);
                }
                other => panic!("unexpected: {other:?}"),
            }
            assert!(matches!(batch.events[2], KvCacheEvent::AllBlocksCleared));
        }

        #[test]
        fn mixed_batch_with_attribution_keys() {
            // ts=1.0, attn_dp_rank=0: three BlockStored (the second carries
            // cache_salt and session_id, the third session_id only), one
            // BlockRemoved, one AllBlocksCleared.
            let bytes = hex_to_bytes("93cb3ff00000000000009587a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573920b0cb1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f6964739401020304aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da347505589a474797065ab426c6f636b53746f726564ac626c6f636b5f6861736865739115b1706172656e745f626c6f636b5f686173680ca9746f6b656e5f696473920506aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055aa63616368655f73616c74a874656e616e742d61aa73657373696f6e5f6964a6736573732d3188a474797065ab426c6f636b53746f726564ac626c6f636b5f686173686573911fb1706172656e745f626c6f636b5f68617368c0a9746f6b656e5f696473920708aa626c6f636b5f73697a6502a76c6f72615f6964c0a66d656469756da3475055aa73657373696f6e5f6964a6736573732d3283a474797065ac426c6f636b52656d6f766564ac626c6f636b5f686173686573910ba66d656469756da347505581a474797065b0416c6c426c6f636b73436c656172656400");
            let batch = decode_event_batch(&bytes).expect("decode msgspec golden");
            assert_eq!(batch.ts, 1.0);
            assert_eq!(batch.attn_dp_rank, Some(0));
            assert_eq!(batch.events.len(), 5);
            match &batch.events[0] {
                KvCacheEvent::BlockStored(b) => {
                    assert_eq!(b.block_hashes, vec![11, 12]);
                    assert_eq!(b.parent_block_hash, None);
                    assert_eq!(b.token_ids, vec![1, 2, 3, 4]);
                    assert_eq!(b.block_size, 2);
                    assert_eq!(b.medium.as_deref(), Some("GPU"));
                }
                other => panic!("expected BlockStored, got {other:?}"),
            }
            match &batch.events[1] {
                KvCacheEvent::BlockStored(b) => {
                    assert_eq!(b.block_hashes, vec![21]);
                    assert_eq!(b.parent_block_hash, Some(12));
                }
                other => panic!("expected BlockStored, got {other:?}"),
            }
            match &batch.events[2] {
                KvCacheEvent::BlockStored(b) => assert_eq!(b.block_hashes, vec![31]),
                other => panic!("expected BlockStored, got {other:?}"),
            }
            match &batch.events[3] {
                KvCacheEvent::BlockRemoved(r) => {
                    assert_eq!(r.block_hashes, vec![11]);
                    assert_eq!(r.medium.as_deref(), Some("GPU"));
                }
                other => panic!("expected BlockRemoved, got {other:?}"),
            }
            assert!(matches!(batch.events[4], KvCacheEvent::AllBlocksCleared));
        }
    }

    #[test]
    fn empty_payload_is_an_error() {
        let err = decode_event_batch(&[]).expect_err("empty payload should fail");
        // Just assert we surfaced a Msgpack decode error.
        assert!(matches!(err, DecodeError::Msgpack(_)));
    }

    /// A `BlockStored` event whose `block_hashes` array prefix exceeds
    /// the per-event cap must be rejected with `PayloadTooLarge` so a
    /// misbehaving worker (or a corrupted msgpack length prefix) cannot
    /// trigger an unbounded allocation in the gateway. We don't fill the
    /// whole array — the visitor refuses on the size_hint alone.
    #[test]
    fn block_stored_with_too_many_hashes_rejected() {
        let claimed = (MAX_HASHES_PER_EVENT + 1) as u32;
        let mut event = Vec::new();
        write_event_map(&mut event, "BlockStored", 1);
        write_key(&mut event, "block_hashes");
        // Oversize block_hashes prefix; only one real element. The
        // visitor's size_hint check fires before reading anything.
        mp::write_array_len(&mut event, claimed).unwrap();
        mp::write_sint(&mut event, 0).unwrap();
        // Trailing bytes are ignored — decoder errors out earlier.
        let bytes = build_batch_bytes(0.0, &[event], None, true);
        let err = decode_event_batch(&bytes).expect_err("oversize hashes should fail");
        match err {
            DecodeError::PayloadTooLarge { field, len, cap } => {
                assert_eq!(field, "block_hashes");
                assert_eq!(cap, MAX_HASHES_PER_EVENT);
                assert_eq!(len, claimed as usize);
            }
            other => panic!("expected PayloadTooLarge, got {other:?}"),
        }
    }

    /// `token_ids` cap — uses an oversize msgpack array length prefix.
    /// rmp-serde reports `size_hint` from the prefix (an `array_len` is a
    /// known length), so the visitor refuses before reading any element.
    /// We deliberately under-fill the array to keep the test cheap; the
    /// decoder rejects on the prefix alone.
    #[test]
    fn block_stored_oversize_token_ids_prefix_rejected() {
        let claimed = (MAX_TOKENS_PER_EVENT + 1) as u32;
        let mut event = Vec::new();
        write_event_map(&mut event, "BlockStored", 2);
        write_key(&mut event, "block_hashes");
        write_i64_array(&mut event, &[42_i64]);
        write_key(&mut event, "token_ids");
        // Oversize token_ids: announce huge length but only write a single
        // element. The visitor's size_hint check fires immediately.
        mp::write_array_len(&mut event, claimed).unwrap();
        mp::write_uint(&mut event, 0).unwrap();
        let bytes = build_batch_bytes(0.0, &[event], None, true);
        let err = decode_event_batch(&bytes).expect_err("oversize token prefix should fail");
        match err {
            DecodeError::PayloadTooLarge { field, len, cap } => {
                assert_eq!(field, "token_ids");
                assert_eq!(cap, MAX_TOKENS_PER_EVENT);
                assert_eq!(len, claimed as usize);
            }
            other => panic!("expected PayloadTooLarge, got {other:?}"),
        }
    }

    /// `BlockRemoved` is also covered. Uses the `block_hashes` cap.
    #[test]
    fn block_removed_with_too_many_hashes_rejected() {
        let claimed = (MAX_HASHES_PER_EVENT + 1) as u32;
        let mut event = Vec::new();
        write_event_map(&mut event, "BlockRemoved", 1);
        write_key(&mut event, "block_hashes");
        mp::write_array_len(&mut event, claimed).unwrap();
        mp::write_sint(&mut event, 0).unwrap();
        // Trailing bytes ignored — decoder errors on the size hint.
        let bytes = build_batch_bytes(0.0, &[event], None, true);
        let err = decode_event_batch(&bytes).expect_err("oversize hashes should fail");
        match err {
            DecodeError::PayloadTooLarge { field, cap, .. } => {
                assert_eq!(field, "block_hashes");
                assert_eq!(cap, MAX_HASHES_PER_EVENT);
            }
            other => panic!("expected PayloadTooLarge, got {other:?}"),
        }
    }
}
