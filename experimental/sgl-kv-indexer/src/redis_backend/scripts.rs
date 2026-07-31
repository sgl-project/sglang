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
//! the scripts reduce to "set a tier bit", "clear a tier bit", "read the
//! placement hash", and "bump a hit counter".

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

/// Sets a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key]`
/// ARGV: `[worker_id, bit]`
/// Returns `1` if the placement changed, else `0`. The caller performs the
/// idempotent reverse-index `SADD` either way.
pub static PLACEMENT_SET: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
local cur = tonumber(redis.call('HGET', KEYS[1], ARGV[1])) or 0
local bit = tonumber(ARGV[2])
if math.floor(cur / bit) % 2 == 1 then
  return 0
end
redis.call('HSET', KEYS[1], ARGV[1], cur + bit)
return 1
"#,
    )
});

/// Clears a tier bit for a worker in a placement hash.
///
/// KEYS: `[placement_key, hit_key]`
/// ARGV: `[worker_id, bit]`
/// Returns `1` when the worker no longer holds the hash at any tier (the caller
/// then removes it from the reverse index), else `0`.
///
/// When the placement hash becomes empty the block no longer exists anywhere, so
/// the co-located hit key (same `{hash}` slot) is deleted in the same script.
/// Otherwise a matched-then-evicted block would leak its `:h` key forever, since
/// the hit key is created lazily on match and nothing else ever removes it.
pub static PLACEMENT_CLEAR: LazyLock<RedisScript> = LazyLock::new(|| {
    RedisScript::new(
        r#"
local cur = tonumber(redis.call('HGET', KEYS[1], ARGV[1]))
local worker_gone = 0
if cur == nil then
  worker_gone = 1
else
  local bit = tonumber(ARGV[2])
  local new = cur
  if math.floor(cur / bit) % 2 == 1 then new = cur - bit end
  if new == 0 then
    redis.call('HDEL', KEYS[1], ARGV[1])
    worker_gone = 1
  else
    redis.call('HSET', KEYS[1], ARGV[1], new)
  end
end
if redis.call('HLEN', KEYS[1]) == 0 then
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
/// Returns `[worker, mask, ...]`.
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
