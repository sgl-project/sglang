// SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
// SPDX-License-Identifier: Apache-2.0

//! Lua scripts. Every script declares all keys it touches in `KEYS[]` and all
//! keys share one hash tag, so each script runs in a single cluster slot (legal
//! and atomic on Redis Cluster) and, on Dragonfly, runs per-slot in parallel
//! without needing `allow-undeclared-keys`. Bitmasks are manipulated with plain
//! arithmetic (no `bit`/bitop library) for cross-engine portability.
//!
//! This build tracks placement only. There is no sequence gate, no incarnation
//! or generation fencing, and no reset script: every apply is unconditional, so
//! the scripts reduce to "replace a (worker,tier) component set", "clear a
//! (worker,tier) placement", "read the placement hash", and "bump a hit
//! counter". Placement fields are `worker_id \x1f tier`; a single reserved field
//! holds the block's token count.

use std::sync::LazyLock;

use redis::Script;

/// Script code plus redis-rs' cached-SHA invocation helper.
///
/// Cluster connections use `code` with EVAL so ASK redirects do not depend on
/// the importing node already having the SHA cached.
pub struct RedisScript {
    pub code: &'static str,
    inner: Script,
}

impl RedisScript {
    fn new(code: &'static str) -> Self {
        Self {
            code,
            inner: Script::new(code),
        }
    }
}

impl std::ops::Deref for RedisScript {
    type Target = Script;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Replaces the component set for one `(worker, tier)` placement, REPLACE
/// semantics (the value overwrites whatever was there for that tier only).
///
/// KEYS: `[placement_key]`
/// ARGV: `[worker_tier_field, components, token_count_field, token_count]`
/// `token_count` is written only when non-empty. Always returns `1`; the caller
/// performs the idempotent reverse-index `SADD`.
pub static PLACEMENT_SET: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
redis.call('HSET', KEYS[1], ARGV[1], ARGV[2])
if ARGV[4] ~= '' then
  redis.call('HSET', KEYS[1], ARGV[3], ARGV[4])
end
return 1
"#,
    )
});

/// Clears one `(worker, tier)` placement.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_tier_field, worker_field_prefix, token_count_field]`
/// Returns `1` when the worker no longer holds the hash at any tier (the caller
/// then removes it from the reverse index), else `0`.
///
/// The token-count field is bookkeeping, not a placement, so it does not keep a
/// block alive: when no `worker \x1f tier` field remains the block is gone
/// everywhere and the whole placement hash (token count included) plus the
/// co-located hit key (same `{hash}` slot) are deleted. Otherwise a
/// matched-then-evicted block would leak its `:h` key forever, since the hit key
/// is created lazily on match and nothing else ever removes it.
pub static PLACEMENT_CLEAR: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
redis.call('HDEL', KEYS[1], ARGV[1])
local prefix = ARGV[2]
local plen = string.len(prefix)
local tc_field = ARGV[3]
local worker_gone = 1
local placements = 0
local fields = redis.call('HKEYS', KEYS[1])
for _, f in ipairs(fields) do
  if f ~= tc_field then
    placements = placements + 1
    if string.sub(f, 1, plen) == prefix then
      worker_gone = 0
    end
  end
end
if placements == 0 then
  redis.call('DEL', KEYS[1])
  redis.call('DEL', KEYS[2])
end
return worker_gone
"#,
    )
});

/// Reads a placement hash.
///
/// KEYS: `[placement_key]`
/// Returns `[field, value, ...]` where each field is either a
/// `worker_id \x1f tier` placement (value = component set) or the reserved
/// token-count field.
pub static MATCH_HASH: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
return redis.call('HGETALL', KEYS[1])
"#,
    )
});

/// Increments a hit counter and records the last-seen timestamp.
///
/// KEYS: `[hit_key]`
/// ARGV: `[now_ms]`
/// Called only for hashes that were actually returned in a match response, so
/// the caller has already decided the hit is real.
pub static HIT_BUMP: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
redis.call('HINCRBY', KEYS[1], 'c', 1)
redis.call('HSET', KEYS[1], 'ls', ARGV[1])
return 1
"#,
    )
});
