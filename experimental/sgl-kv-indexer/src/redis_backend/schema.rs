// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Redis key schema and tier-bitmask helpers.
//!
//! Two hash-tag families keep related keys in one cluster slot:
//!   * `{<hash>}`      — placement (`:p`) and hit (`:h`) for a block hash, so a
//!     match can read placement and bump the hit counter in one single-slot Lua.
//!   * `{w:<worker>}`  — the worker's reverse index (`:blocks`) and registry
//!     (`:meta`), so per-worker mutations stay in one slot.
//!
//! A tier is stored as a bit in a per-`(hash, worker)` bitmask: `bit = 1 << tier`
//! (TierType HBM=1, DRAM=2, SSD=3). The bitmask is manipulated with plain integer
//! arithmetic in Lua so the scripts run unchanged on Redis, Dragonfly and Valkey.

/// Placement key for a block hash: HASH of `worker_id -> tier bitmask`.
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

/// Registry for a worker: HASH with a single `addr` field, used to populate the
/// routing address in match responses. Never expires — this build does not track
/// worker liveness.
pub fn worker_meta_key(ns: &str, worker_id: &str) -> String {
    format!("{ns}:{{w:{worker_id}}}:meta")
}

/// The bit representing a tier in a placement bitmask.
pub fn tier_bit(tier: i32) -> i64 {
    1i64 << tier
}

/// Decodes the tiers present in a placement bitmask, ascending. Only valid tiers
/// (HBM=1, DRAM=2, SSD=3) are considered.
pub fn tiers_from_mask(mask: i64) -> Vec<i32> {
    (1..=3).filter(|t| mask & (1i64 << t) != 0).collect()
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
    fn tier_bit_matches_shift() {
        assert_eq!(tier_bit(1), 2);
        assert_eq!(tier_bit(2), 4);
        assert_eq!(tier_bit(3), 8);
    }

    #[test]
    fn mask_round_trips_through_tiers() {
        assert_eq!(tiers_from_mask(0), Vec::<i32>::new());
        assert_eq!(tiers_from_mask(tier_bit(1)), vec![1]);
        assert_eq!(tiers_from_mask(tier_bit(1) | tier_bit(3)), vec![1, 3]);
        assert_eq!(
            tiers_from_mask(tier_bit(1) | tier_bit(2) | tier_bit(3)),
            vec![1, 2, 3]
        );
    }
}
