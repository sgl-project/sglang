//! Block-hash compute matching SGLang's `radix_cache` worker.
//!
//! This is the gateway-side mirror of SGLang's per-page SHA256 chaining used
//! to derive `BlockStored.block_hashes` on workers (Python:
//! `python/sglang/srt/mem_cache/radix_cache.py::hash_page` and
//! `python/sglang/srt/mem_cache/utils.py::hash_str_to_int64`).
//!
//! ### Algorithm
//!
//! For each page (chunk of `block_size` tokens, last page possibly short):
//! 1. Initialize a SHA256 hasher.
//! 2. If the request has a `RadixKey.extra_key` namespace, feed its UTF-8 byte
//!    length as a 4-byte little-endian integer followed by the UTF-8 bytes.
//! 3. If a prior page exists, feed the prior page's **full 32-byte SHA256
//!    digest** (raw bytes, not the truncated i64) into the hasher.
//! 4. Feed each token in the page as 4 little-endian unsigned bytes.
//! 5. Take the 32-byte digest as the new "prior" for the next page.
//! 6. Truncate the digest to a signed i64 by reading the first 16 hex chars
//!    (top 64 bits) and reinterpreting as signed.
//!
//! ### Why no `parent_hash: Option<i64>` argument
//!
//! SGLang's worker chains on the **full 32-byte digest** of the parent block,
//! not on the i64 truncation. An `Option<i64>` is lossy — you cannot
//! reconstruct 32 bytes of SHA256 from 64 bits — so accepting one as a
//! "starting point" would silently produce hashes that disagree with the
//! worker.
//!
//! In the gateway we only need to compute hashes for an entire request from
//! scratch (i.e. starting with no parent). That matches the Python emission
//! path where the first page of a freshly-stored node may have a parent block
//! hash for the radix tree key, but the **page-hash computation itself**
//! starts from the parent's **full hex digest** (`node.parent.hash_value[-1]`).
//! For request-side hashing on the routing path, there is no parent, so we
//! expose the "from-scratch" entry point only.
//!
//! ### Bigram mode
//!
//! EAGLE-family workers (`is_bigram = is_eagle`) hash KV blocks over
//! overlapping `(t_i, t_{i+1})` token pairs. That path is implemented as a
//! separate [`compute_block_hashes_bigram`] (below) rather than branching
//! inside the non-bigram fast path; `CacheAwareZmqPolicy::select` chooses
//! between the two from the worker-reported bigram flag.

use sha2::{Digest, Sha256};

/// Compute per-block i64 hashes for a token sequence, matching SGLang's
/// worker emission for a chain that starts with no parent block.
///
/// The returned `Vec<i64>` has `ceil(token_ids.len() / block_size)` entries,
/// each being the i64 truncation (top 64 bits, signed) of the per-page
/// SHA256 digest as defined in [`Self`-module docs](self).
///
/// # Panics
///
/// Panics if `block_size == 0`. Callers are expected to validate this once
/// up-front against the worker-published `block_size`; an invalid value is
/// a programmer/config bug, not a runtime input we should swallow.
pub fn compute_block_hashes(token_ids: &[u32], block_size: usize) -> Vec<i64> {
    compute_block_hashes_with_extra_key(token_ids, block_size, None)
}

/// Compute per-block i64 hashes with SGLang's `RadixKey.extra_key` namespace.
///
/// The namespace is folded into every page before the parent digest. This mirrors
/// `RadixKey.hash_page` and `mem_cache.utils.get_hash_str`.
pub fn compute_block_hashes_with_extra_key(
    token_ids: &[u32],
    block_size: usize,
    extra_key: Option<&str>,
) -> Vec<i64> {
    assert!(block_size > 0, "block_size must be positive");
    if token_ids.is_empty() {
        return Vec::new();
    }

    let n = token_ids.len();
    let num_blocks = n.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks);
    let mut prior: Option<[u8; 32]> = None;

    let mut start = 0;
    while start < n {
        let end = (start + block_size).min(n);
        let digest = chain_block(prior.as_ref(), &token_ids[start..end], extra_key);
        out.push(sha256_to_i64(&digest));
        prior = Some(digest);
        start = end;
    }

    out
}

/// Hash a single page, optionally chained to a parent block's full 32-byte
/// SHA256 digest. Returns the new 32-byte digest.
#[inline]
fn chain_block(
    parent_digest: Option<&[u8; 32]>,
    block_tokens: &[u32],
    extra_key: Option<&str>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_hasher_with_extra_key(&mut hasher, extra_key);
    if let Some(parent) = parent_digest {
        hasher.update(parent);
    }
    for t in block_tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.finalize().into()
}

#[inline]
fn update_hasher_with_extra_key(hasher: &mut Sha256, extra_key: Option<&str>) {
    let Some(extra_key) = extra_key else {
        return;
    };
    let encoded = extra_key.as_bytes();
    let len = u32::try_from(encoded.len()).expect("extra_key length exceeds u32");
    hasher.update(len.to_le_bytes());
    hasher.update(encoded);
}

/// Convert a full 32-byte SHA256 digest to the signed i64 truncation that
/// SGLang publishes on the wire (top 64 bits, big-endian, reinterpreted as
/// signed).
///
/// Mirrors Python's `hash_str_to_int64`:
/// ```text
/// uint64_val = int(hash_str[:16], 16)
/// return uint64_val - 2**64 if uint64_val >= 2**63 else uint64_val
/// ```
/// which is equivalent to `i64::from_be_bytes(digest[..8])`.
#[inline]
pub fn sha256_to_i64(digest: &[u8; 32]) -> i64 {
    let mut top = [0u8; 8];
    top.copy_from_slice(&digest[..8]);
    i64::from_be_bytes(top)
}

/// Bigram variant of [`compute_block_hashes`], matching SGLang's `radix_cache`
/// worker when the model runs **EAGLE speculative decoding** (`is_bigram =
/// is_eagle`). Mirrors `RadixKey.hash_page` (Python:
/// `mem_cache/radix_cache.py`) on the bigram path:
///
/// - The logical sequence is the `N-1` overlapping bigrams of `N` raw tokens,
///   so the page count is `ceil((len-1) / block_size)`. Fewer than 2 tokens
///   yields no blocks.
/// - Each page `[start, end)` (in bigram-index space) feeds **both** tokens of
///   every bigram into the SHA256 hasher — `t[j]` then `t[j+1]`, each as 4
///   little-endian bytes — vs. the unigram path's single token per unit.
/// - Pages chain on the prior page's full 32-byte digest and truncate to i64
///   exactly as the unigram path does.
///
/// Use this (instead of [`compute_block_hashes`]) when the worker advertises an
/// EAGLE speculative algorithm via `/server_info`; otherwise the router's query
/// hashes won't match the worker's stored bigram block hashes and cache-aware
/// routing silently degrades to min-load.
pub fn compute_block_hashes_bigram(token_ids: &[u32], block_size: usize) -> Vec<i64> {
    compute_block_hashes_bigram_with_extra_key(token_ids, block_size, None)
}

/// Bigram variant of [`compute_block_hashes_with_extra_key`].
pub fn compute_block_hashes_bigram_with_extra_key(
    token_ids: &[u32],
    block_size: usize,
    extra_key: Option<&str>,
) -> Vec<i64> {
    assert!(block_size > 0, "block_size must be positive");
    // N raw tokens -> N-1 overlapping bigrams; fewer than 2 tokens -> no blocks.
    let logical_len = token_ids.len().saturating_sub(1);
    if logical_len == 0 {
        return Vec::new();
    }
    let num_blocks = logical_len.div_ceil(block_size);
    let mut out = Vec::with_capacity(num_blocks);
    let mut prior: Option<[u8; 32]> = None;

    let mut start = 0;
    while start < logical_len {
        let end = (start + block_size).min(logical_len);
        let digest = chain_block_bigram(prior.as_ref(), token_ids, start, end, extra_key);
        out.push(sha256_to_i64(&digest));
        prior = Some(digest);
        start = end;
    }

    out
}

/// Hash a single bigram page: for each unit `j` in `[start, end)`, feed
/// `tokens[j]` then `tokens[j + 1]` (4 little-endian bytes each), chained on the
/// parent block's full 32-byte digest. The caller guarantees `end <= len - 1`,
/// so `tokens[j + 1]` is always in bounds. Mirrors the engine's `hash_page`
/// bigram branch.
#[inline]
fn chain_block_bigram(
    parent_digest: Option<&[u8; 32]>,
    tokens: &[u32],
    start: usize,
    end: usize,
    extra_key: Option<&str>,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    update_hasher_with_extra_key(&mut hasher, extra_key);
    if let Some(parent) = parent_digest {
        hasher.update(parent);
    }
    for j in start..end {
        hasher.update(tokens[j].to_le_bytes());
        hasher.update(tokens[j + 1].to_le_bytes());
    }
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Bigram cross-language goldens ----
    // Values produced by SGLang's REAL `RadixKey(..., is_bigram=True).hash_page`
    // + `compute_node_hash_values` chunking + `hash_str_to_int64`, run against
    // the deployed DeepSeek-V4-Flash engine. These lock byte-exact equivalence
    // with the worker's stored block hashes — the contract that makes
    // cache-aware routing actually match for EAGLE/bigram models.

    #[test]
    fn extra_key_namespaces_first_block_and_propagates() {
        assert_eq!(
            compute_block_hashes_with_extra_key(&[1, 2, 3, 4], 4, Some("salt-A")),
            vec![1848708812856982025_i64]
        );
        assert_eq!(
            compute_block_hashes_with_extra_key(&[1, 2, 3, 4], 4, Some("salt-B")),
            vec![-3770730224180675262_i64]
        );
        assert_eq!(
            compute_block_hashes_with_extra_key(&[1, 2, 3, 4, 5], 4, Some("salt-A")),
            vec![1848708812856982025_i64, -844459331919182630_i64]
        );
    }

    #[test]
    fn empty_extra_key_is_a_namespace() {
        assert_eq!(
            compute_block_hashes_with_extra_key(&[1, 2, 3, 4], 4, Some("")),
            vec![-1934027550307904538_i64]
        );
        assert_eq!(
            compute_block_hashes_with_extra_key(&[1, 2, 3, 4], 4, None),
            vec![-3488128144981237669_i64]
        );
    }

    #[test]
    fn bigram_extra_key_namespaces_first_block_and_propagates() {
        assert_eq!(
            compute_block_hashes_bigram_with_extra_key(&[10, 20, 30, 40, 50], 2, Some("salt-A"),),
            vec![-3307084293699386976_i64, -4477231418096748271_i64]
        );
    }

    #[test]
    fn bigram_golden_single_block_full() {
        // engine: chain_bigram([10,20,30,40], 4) -> [-2735951481331064195]
        assert_eq!(
            compute_block_hashes_bigram(&[10, 20, 30, 40], 4),
            vec![-2735951481331064195_i64]
        );
    }

    #[test]
    fn bigram_golden_multi_block() {
        // engine: chain_bigram([10,20,30,40,50], 2) -> [-8847804484166691499, 4989791362144317498]
        assert_eq!(
            compute_block_hashes_bigram(&[10, 20, 30, 40, 50], 2),
            vec![-8847804484166691499_i64, 4989791362144317498_i64]
        );
    }

    #[test]
    fn bigram_golden_partial_last_block() {
        // engine: chain_bigram([1,2,3,4,5,6], 4) -> [-638950109823820341, 3604587133525381017]
        assert_eq!(
            compute_block_hashes_bigram(&[1, 2, 3, 4, 5, 6], 4),
            vec![-638950109823820341_i64, 3604587133525381017_i64]
        );
    }

    #[test]
    fn bigram_golden_longer_multi_block() {
        // engine: chain_bigram([5,6,7,8,9,10,11,12,13], 4) -> [-2900568514773989563, -322435596280658912]
        assert_eq!(
            compute_block_hashes_bigram(&[5, 6, 7, 8, 9, 10, 11, 12, 13], 4),
            vec![-2900568514773989563_i64, -322435596280658912_i64]
        );
    }

    #[test]
    fn bigram_single_bigram_equals_unigram_pair() {
        // One bigram (10,20) feeds bytes 10,20 — identical to a unigram block
        // [10,20]. engine: chain_bigram([10,20], 4) -> [978178666101069530],
        // which equals the unigram block hash of [10,20].
        assert_eq!(
            compute_block_hashes_bigram(&[10, 20], 4),
            vec![978178666101069530_i64]
        );
        assert_eq!(
            compute_block_hashes_bigram(&[10, 20], 4),
            compute_block_hashes(&[10, 20], 4)
        );
    }

    #[test]
    fn bigram_fewer_than_two_tokens_yields_no_blocks() {
        // N tokens -> N-1 bigrams; <2 tokens -> 0 bigrams -> empty.
        assert!(compute_block_hashes_bigram(&[10], 4).is_empty());
        assert!(compute_block_hashes_bigram(&[], 4).is_empty());
    }

    #[test]
    fn bigram_differs_from_unigram_for_multi_token_blocks() {
        // Sanity: for >2 tokens the bigram hash must NOT equal the unigram hash
        // (different byte stream) — this is exactly why a unigram-hashing
        // router gets zero overlap against a bigram worker.
        let toks = [10u32, 20, 30, 40];
        assert_ne!(
            compute_block_hashes_bigram(&toks, 4),
            compute_block_hashes(&toks, 4)
        );
    }

    #[test]
    #[should_panic(expected = "block_size must be positive")]
    fn bigram_zero_block_size_panics() {
        let _ = compute_block_hashes_bigram(&[1, 2, 3], 0);
    }

    /// Helper for tests: derive the expected i64 from a list of tokens
    /// chained against an optional parent digest. This mirrors `chain_block`
    /// but is duplicated here so a regression in the production helper
    /// cannot also hide itself in the test oracle.
    fn oracle_block_digest(parent: Option<&[u8; 32]>, tokens: &[u32]) -> [u8; 32] {
        let mut h = Sha256::new();
        if let Some(p) = parent {
            h.update(p);
        }
        for t in tokens {
            h.update(t.to_le_bytes());
        }
        h.finalize().into()
    }

    fn oracle_i64(digest: &[u8; 32]) -> i64 {
        let mut top = [0u8; 8];
        top.copy_from_slice(&digest[..8]);
        i64::from_be_bytes(top)
    }

    #[test]
    fn empty_input_returns_empty_vec() {
        assert!(compute_block_hashes(&[], 4).is_empty());
        assert!(compute_block_hashes(&[], 1).is_empty());
    }

    #[test]
    #[should_panic(expected = "block_size must be positive")]
    fn zero_block_size_panics() {
        let _ = compute_block_hashes(&[1, 2, 3], 0);
    }

    #[test]
    fn single_full_block() {
        // Independent oracle: SHA256 of the LE bytes of [1,2,3,4], take top 8 bytes.
        let expected_digest = oracle_block_digest(None, &[1, 2, 3, 4]);
        let expected_i64 = oracle_i64(&expected_digest);

        let got = compute_block_hashes(&[1, 2, 3, 4], 4);
        assert_eq!(got, vec![expected_i64]);
    }

    #[test]
    fn partial_last_block_chains_against_first_block_digest() {
        // 5 tokens, block_size 4 → block 0 = [1,2,3,4], block 1 = [5] chained
        // against block 0's full 32-byte digest.
        let d0 = oracle_block_digest(None, &[1, 2, 3, 4]);
        let d1 = oracle_block_digest(Some(&d0), &[5]);
        let expected = vec![oracle_i64(&d0), oracle_i64(&d1)];

        let got = compute_block_hashes(&[1, 2, 3, 4, 5], 4);
        assert_eq!(got, expected);
    }

    #[test]
    fn multi_block_chain() {
        // 8 tokens, block_size 2 → 4 blocks, each chained against the
        // previous block's full 32-byte digest.
        let tokens: [u32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        let d0 = oracle_block_digest(None, &tokens[0..2]);
        let d1 = oracle_block_digest(Some(&d0), &tokens[2..4]);
        let d2 = oracle_block_digest(Some(&d1), &tokens[4..6]);
        let d3 = oracle_block_digest(Some(&d2), &tokens[6..8]);
        let expected = vec![
            oracle_i64(&d0),
            oracle_i64(&d1),
            oracle_i64(&d2),
            oracle_i64(&d3),
        ];

        let got = compute_block_hashes(&tokens, 2);
        assert_eq!(got, expected);
    }

    #[test]
    fn sha256_to_i64_handles_top_bit_set() {
        // sha256("") = e3b0c44298fc1c14 9afbf4c8996fb924 27ae41e4649b934c a495991b7852b855
        // Top 8 bytes = e3b0c44298fc1c14 → uint64 0xe3b0c44298fc1c14
        // Top bit set → signed value = uint64 - 2**64 = -2039914840885289964
        let digest: [u8; 32] = Sha256::digest(b"").into();
        assert_eq!(sha256_to_i64(&digest), -2039914840885289964_i64);
    }

    /// Cross-language goldens: values produced by a Python script that
    /// mirrors `radix_cache.hash_page` (non-bigram path) and
    /// `mem_cache.utils.hash_str_to_int64`. These are the contract with the
    /// SGLang worker and lock down algorithmic equivalence regardless of
    /// changes to the Rust-internal helpers.
    ///
    /// Reproducer (saved temporarily to `/tmp/sglang_hash_oracle.py` during
    /// development; not committed):
    /// ```python
    /// import hashlib
    /// def hash_page(prior, toks):
    ///     h = hashlib.sha256()
    ///     if prior:
    ///         h.update(bytes.fromhex(prior))
    ///     for t in toks:
    ///         h.update(int(t).to_bytes(4, "little", signed=False))
    ///     return h.hexdigest()
    /// def hash_str_to_int64(s):
    ///     v = int(s[:16], 16)
    ///     return v - 2**64 if v >= 2**63 else v
    /// def chain(tokens, bs):
    ///     out, prior = [], None
    ///     for i in range(0, len(tokens), bs):
    ///         hx = hash_page(prior, tokens[i:i+bs])
    ///         out.append(hash_str_to_int64(hx)); prior = hx
    ///     return out
    /// ```
    #[test]
    fn cross_language_golden_single_block() {
        // Python: chain([1,2,3,4], 4) -> [-3488128144981237669]
        let got = compute_block_hashes(&[1, 2, 3, 4], 4);
        assert_eq!(got, vec![-3488128144981237669_i64]);
    }

    #[test]
    fn cross_language_golden_partial_last_block() {
        // Python: chain([1,2,3,4,5], 4)
        //   -> [-3488128144981237669, -3787494577174227566]
        let got = compute_block_hashes(&[1, 2, 3, 4, 5], 4);
        assert_eq!(
            got,
            vec![-3488128144981237669_i64, -3787494577174227566_i64]
        );
    }

    #[test]
    fn cross_language_golden_multi_block() {
        // Python: chain([10,20,30,40,50,60,70,80], 2)
        //   -> [978178666101069530, -895308556211281782,
        //       -8033692805846017938, 835415944263129316]
        let got = compute_block_hashes(&[10, 20, 30, 40, 50, 60, 70, 80], 2);
        assert_eq!(
            got,
            vec![
                978178666101069530_i64,
                -895308556211281782_i64,
                -8033692805846017938_i64,
                835415944263129316_i64,
            ]
        );
    }
}
