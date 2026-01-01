//! Radix tree implementations for prefix matching and cache-aware routing.
//!
//! This module provides radix tree data structures that mirror SGLang's
//! scheduler memory management patterns. Two implementations are available:
//!
//! - [`StringTree`]: Character-based tree for HTTP router (text input)
//! - [`TokenTree`]: Token-based tree for gRPC router (pre-tokenized input)
//!
//! Both implementations support:
//! - Multi-tenant prefix tracking with LRU eviction
//! - Concurrent access via DashMap and RwLock
//! - Efficient prefix matching with match counts

mod common;
mod string_tree;
mod token_tree;

pub use common::{MatchResult, TenantId};
pub use string_tree::{PrefixMatchResult as StringMatchResult, Tree as StringTree};
pub use token_tree::{PrefixMatchResult as TokenMatchResult, TokenTree};

/// Trait for radix tree implementations.
///
/// This trait provides a unified interface for both string-based and token-based
/// radix trees used in cache-aware routing.
pub trait RadixTree: Send + Sync {
    /// The key type for this tree (e.g., &str or &[u32])
    type Key: ?Sized;

    /// The result type returned by prefix matching
    type MatchResult: MatchResult;

    /// Insert a key with associated tenant.
    ///
    /// If the key already exists or shares a prefix with existing keys,
    /// the tree structure is updated to track the tenant association.
    fn insert(&self, key: &Self::Key, tenant: &str);

    /// Find the longest matching prefix and return the associated tenant.
    ///
    /// Returns `None` if no prefix matches.
    fn prefix_match(&self, key: &Self::Key) -> Option<TenantId>;

    /// Find the longest matching prefix with detailed match counts.
    ///
    /// Returns match result with:
    /// - `tenant`: The tenant that owns the matched prefix
    /// - `matched_count`: Number of units (chars/tokens) matched
    /// - `input_count`: Total units in the input key
    fn prefix_match_with_counts(&self, key: &Self::Key) -> Self::MatchResult;

    /// Evict cached entries for a tenant to reduce memory usage.
    ///
    /// # Arguments
    /// * `tenant` - The tenant whose entries should be evicted
    /// * `max_units` - Maximum units (chars/tokens) to retain for this tenant
    fn evict(&self, tenant: &TenantId, max_units: usize);

    /// Get the current size (in units) for a tenant.
    fn tenant_size(&self, tenant: &TenantId) -> usize;

    /// Reset the tree to empty state.
    fn reset(&self);
}
