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
//! 2. If a prior page exists, feed the prior page's **full 32-byte SHA256
//!    digest** (raw bytes, not the truncated i64) into the hasher.
//! 3. Feed each token in the page as 4 little-endian unsigned bytes.
//! 4. Take the 32-byte digest as the new "prior" for the next page.
//! 5. Truncate the digest to a signed i64 by reading the first 16 hex chars
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
//! Not supported in v1. SGLang's bigram mode interleaves overlapping
//! `(t_i, t_{i+1})` pairs into the hash. The gateway does not need this
//! today; if/when it does, add a separate `compute_block_hashes_bigram` rather
//! than complicating the non-bigram fast path.

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
        let digest = chain_block(prior.as_ref(), &token_ids[start..end]);
        out.push(sha256_to_i64(&digest));
        prior = Some(digest);
        start = end;
    }

    out
}

/// Hash a single page, optionally chained to a parent block's full 32-byte
/// SHA256 digest. Returns the new 32-byte digest.
#[inline]
fn chain_block(parent_digest: Option<&[u8; 32]>, block_tokens: &[u32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    if let Some(parent) = parent_digest {
        hasher.update(parent);
    }
    for t in block_tokens {
        hasher.update(t.to_le_bytes());
    }
    hasher.finalize().into()
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

#[cfg(test)]
mod tests {
    use super::*;

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
