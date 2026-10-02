//! Tree component identity. One enum per FULL / SWA / MAMBA, used as the
//! discriminator for per-component arrays — `[ComponentPoolState;
//! NUM_COMPONENT_TYPES]` on `TreeNodePool` (per-component sentinels +
//! size aggregates) and `[ComponentNodeState; NUM_COMPONENT_TYPES]` on
//! `TreeNode` (per-component value/lock_ref/lru_data).
//!
//! Mirrors Python `sglang.srt.mem_cache.unified_cache_components.tree_component.ComponentType`.
//! Discriminants are pinned to match the OSS Python enum so per-component
//! arrays have the same indexing on both sides of the PyO3 boundary.

use pyo3::prelude::*;

/// Tree component types. Used both as a Rust-side array index
/// (`[T; NUM_COMPONENT_TYPES]` indexed by `ct as usize`) and as a Python
/// argument on per-component PyO3 wrapper APIs.
///
/// `hash` is exposed to Python so callers can use `ComponentType` as a
/// dict key (e.g., the `tracker: dict[ComponentType, int]` shape that
/// the per-component orchestrator's `drive_eviction` expects).
#[pyclass(eq, eq_int, hash, frozen)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ComponentType {
    Full = 0,
    Swa = 1,
    Mamba = 2,
}

/// Length of per-component arrays — auto-derived from the highest
/// discriminant. Adding a new "highest" variant requires bumping the
/// basis below; the exhaustive `match` in the assertion block forces
/// any new variant to add a corresponding discriminant assertion or
/// fail to compile.
pub const NUM_COMPONENT_TYPES: usize = ComponentType::Mamba as usize + 1;

const _: () = {
    // Exhaustive match (no `_` arm) — adding a new `ComponentType` variant
    // without extending this match is a compile error, which surfaces the
    // need to add the matching `assert!` below and bump the basis above.
    match ComponentType::Full {
        ComponentType::Full => {}
        ComponentType::Swa => {}
        ComponentType::Mamba => {}
    }
    assert!(ComponentType::Full as usize == 0);
    assert!(ComponentType::Swa as usize == 1);
    assert!(ComponentType::Mamba as usize == 2);
    assert!(NUM_COMPONENT_TYPES == 3);
};
