// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Redis key schema and placement-field helpers.
//!
//! Two hash-tag families keep related keys in one cluster slot:
//!   * `{<hash>}`      — placement (`:p`) and hit (`:h`) for a block hash, so a
//!     match can read placement and bump the hit counter in one single-slot Lua.
//!   * `{w:<worker>}`  — the worker's reverse index (`:blocks`) and registry
//!     (`:meta`), so per-worker mutations stay in one slot.
//!
//! Placement is a per-block-hash HASH. Each `(worker, tier)` is its own field
//! (`worker_id \x1f tier`) whose value is that tier's **resident component set**
//! — a sorted, comma-joined list of opaque component labels, or the empty string
//! for a legacy whole-block (base-only) placement. Keeping one field per tier
//! makes a per-tier REPLACE a single `HSET` and a per-tier revoke a single
//! `HDEL`, so removing one tier never disturbs another. A single reserved field
//! [`TOKEN_COUNT_FIELD`] co-locates the block's token count (for trailing-window
//! accumulation) and is dropped with the placement HASH.

/// Field-name separator between a worker id and a tier in the placement HASH.
/// The Unit Separator control char cannot appear in a tier decimal and is not
/// expected in worker ids, so it disambiguates the composite field.
pub const WORKER_TIER_SEP: char = '\u{1f}';

/// Reserved placement field holding the block's token count. Prefixed with NUL so
/// it can never collide with a `worker_id \x1f tier` field.
pub const TOKEN_COUNT_FIELD: &str = "\u{0}sz";

/// Placement key for a block hash: HASH of `worker_id \x1f tier -> components`.
pub fn placement_key(ns: &str, hash: &str) -> String {
    format!("{ns}:{{{hash}}}:p")
}

/// Hit key for a block hash: HASH with fields `c` (count) and `ls` (last_seen ms).
pub fn hit_key(ns: &str, hash: &str) -> String {
    format!("{ns}:{{{hash}}}:h")
}

/// Reverse index for a worker: SET of block hashes it currently holds.
pub fn worker_blocks_key(ns: &str, worker_id: &str) -> String {
    format!("{ns}:{{w:{worker_id}}}:blocks")
}

/// Registry for a worker: HASH with an `addr` field (routing address) and a
/// `spec` field (serialized WorkerCacheSpec). Never expires — this build does not
/// track worker liveness.
pub fn worker_meta_key(ns: &str, worker_id: &str) -> String {
    format!("{ns}:{{w:{worker_id}}}:meta")
}

/// The placement HASH field for one `(worker, tier)` pair.
pub fn placement_field(worker: &str, tier: i32) -> String {
    format!("{worker}{WORKER_TIER_SEP}{tier}")
}

/// The `worker_id \x1f` prefix shared by all of a worker's placement fields.
pub fn worker_field_prefix(worker: &str) -> String {
    format!("{worker}{WORKER_TIER_SEP}")
}

/// Parses a placement field back into its `(worker, tier)` pair. Returns `None`
/// for the reserved [`TOKEN_COUNT_FIELD`] or any malformed field.
pub fn parse_placement_field(field: &str) -> Option<(String, i32)> {
    let (worker, tier) = field.rsplit_once(WORKER_TIER_SEP)?;
    let tier = tier.parse::<i32>().ok()?;
    Some((worker.to_string(), tier))
}

/// Encodes a component bitmask for storage as a decimal string. `0` is a legacy
/// whole-block placement (held, no component detail).
pub fn encode_mask(mask: u32) -> String {
    mask.to_string()
}

/// Decodes a stored component bitmask; a malformed value decodes to `0`.
pub fn decode_mask(value: &str) -> u32 {
    value.parse().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_share_hash_tag_across_placement_and_hit() {
        assert_eq!(placement_key("kvidx", "123"), "kvidx:{123}:p");
        assert_eq!(hit_key("kvidx", "123"), "kvidx:{123}:h");
    }

    #[test]
    fn worker_keys_share_worker_tag() {
        assert_eq!(worker_blocks_key("kvidx", "w1"), "kvidx:{w:w1}:blocks");
        assert_eq!(worker_meta_key("kvidx", "w1"), "kvidx:{w:w1}:meta");
    }

    #[test]
    fn placement_field_round_trips() {
        let field = placement_field("w1", 2);
        assert_eq!(field, format!("w1{}2", WORKER_TIER_SEP));
        assert_eq!(parse_placement_field(&field), Some(("w1".to_string(), 2)));
        assert!(field.starts_with(&worker_field_prefix("w1")));
    }

    #[test]
    fn token_count_field_is_not_a_placement_field() {
        assert_eq!(parse_placement_field(TOKEN_COUNT_FIELD), None);
    }

    #[test]
    fn worker_id_may_contain_a_colon() {
        // Worker ids can carry colons (e.g. host:port); rsplit on the separator
        // still recovers the whole worker id.
        let field = placement_field("10.0.0.1:9000", 1);
        assert_eq!(
            parse_placement_field(&field),
            Some(("10.0.0.1:9000".to_string(), 1))
        );
    }

    #[test]
    fn mask_round_trips() {
        assert_eq!(encode_mask(0), "0");
        assert_eq!(encode_mask(0b101), "5");
        assert_eq!(decode_mask("5"), 0b101);
        assert_eq!(decode_mask(""), 0); // malformed -> 0
    }
}
