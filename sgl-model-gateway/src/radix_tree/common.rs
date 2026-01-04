//! Common types shared between radix tree implementations.

use std::sync::Arc;

/// Interned tenant identifier for efficient comparison and storage.
///
/// Using `Arc<str>` allows O(1) cloning and pointer-based equality checks
/// for frequently accessed tenant identifiers.
pub type TenantId = Arc<str>;

/// Trait for prefix match results.
///
/// This trait provides a common interface for accessing match results
/// from both string-based and token-based radix trees.
pub trait MatchResult {
    /// Get the tenant that owns the matched prefix.
    fn tenant(&self) -> &TenantId;

    /// Get the number of units (chars or tokens) that matched.
    fn matched_count(&self) -> usize;

    /// Get the total number of units in the input.
    fn input_count(&self) -> usize;

    /// Calculate the cache hit ratio (matched / input).
    ///
    /// Returns 0.0 if input is empty, otherwise returns a value in [0.0, 1.0].
    fn hit_ratio(&self) -> f64 {
        let input = self.input_count();
        if input == 0 {
            0.0
        } else {
            self.matched_count() as f64 / input as f64
        }
    }
}
