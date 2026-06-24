//! The FULL / SWA / Mamba tree-component identity.

use pyo3::prelude::*;

/// A tree component type. Different component types behave differently on
/// metadata and tree-structure management.
#[pyclass(eq, eq_int, hash, frozen)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ComponentType {
    Full = 0,
    Swa = 1,
    Mamba = 2,
}

#[pymethods]
impl ComponentType {
    /// Use the discriminant directly as a positional index into per-component
    /// arrays (e.g. `result.freed[ComponentType.Full]`) — no `int(...)` needed.
    fn __index__(&self) -> usize {
        *self as usize
    }
}

/// Length of per-component arrays, derived from the highest discriminant.
pub const NUM_COMPONENT_TYPES: usize = ComponentType::Mamba as usize + 1;

const _: () = {
    // Static assert that NUM_COMPONENT_TYPES covers all component types.
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
