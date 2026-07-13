# PD Flip Target Prefix Ownership Design

## Problem

The migration target uses `DecodePreallocQueue._pre_alloc()` directly after a
decode radix/HiCache prefix match.  The normal decode preallocation path then
sets `req.cache_protected_len = total_prefix_len`, but the PD flip target path
does not.  At request completion, `cache_finished_req()` therefore treats the
already-owned matched prefix as duplicate request-owned KV and frees it while
the radix tree still counts it as evictable.  The invariant becomes
`available + evictable = total + matched_prefix_len` (observed +1975/+1981).

## Required behavior

- After target preallocation, record `total_prefix_len` as the request's
  protected cache length, matching the normal decode path.
- Preserve zero-prefix and full-source fallback behavior.
- Do not change transfer ranges, HiCache restore, activation, or finish policy.

## Acceptance

- A focused unit test proves a target prefix hit updates
  `req.cache_protected_len` to the stitched total prefix length.
- A zero-prefix control remains zero.
- Full 40-request migration completes and the target worker remains healthy
  with balanced KV accounting after request completion.
