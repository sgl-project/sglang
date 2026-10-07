//! Native component ownership at the Python construction boundary.

use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::{FullComponent, MambaComponent, SwaComponent, TreeComponent};
use crate::node::ChildKeyType;
use crate::python_bindings::{TreeCoreInitParamsBinding, catch_native_panic};

type NativeComponent<K> = Arc<dyn TreeComponent<K> + Send + Sync>;

enum ComponentHandle {
    Plain(NativeComponent<Vec<i64>>),
    Bigram(NativeComponent<Vec<(i64, i64)>>),
}

/// A native component instance retained by each tree constructed with it.
#[pyclass(frozen)]
pub struct TreeComponentBinding {
    inner: ComponentHandle,
}

impl TreeComponentBinding {
    pub(crate) fn from_component<C>(component: C, is_bigram: bool) -> Self
    where
        C: TreeComponent<Vec<i64>> + TreeComponent<Vec<(i64, i64)>> + Send + Sync + 'static,
    {
        Self {
            inner: if is_bigram {
                ComponentHandle::Bigram(Arc::new(component))
            } else {
                ComponentHandle::Plain(Arc::new(component))
            },
        }
    }
}

#[pymethods]
impl TreeComponentBinding {
    #[staticmethod]
    fn full(init_params: &TreeCoreInitParamsBinding, is_bigram: bool) -> Self {
        let _ = init_params;
        Self::from_component(FullComponent, is_bigram)
    }

    #[staticmethod]
    fn swa(init_params: &TreeCoreInitParamsBinding, is_bigram: bool) -> PyResult<Self> {
        if init_params.swa_sliding_window_size.unwrap_or(0) == 0 {
            return Err(PyValueError::new_err(
                "the Swa component requires positive swa_sliding_window_size",
            ));
        }
        catch_native_panic(|| {
            Ok(Self::from_component(
                SwaComponent::new(&init_params.to_cache_init_params()?),
                is_bigram,
            ))
        })
    }

    #[staticmethod]
    fn mamba(init_params: &TreeCoreInitParamsBinding, is_bigram: bool) -> PyResult<Self> {
        if init_params.mamba_cache_chunk_size.unwrap_or(0) == 0 || init_params.page_size == 0 {
            return Err(PyValueError::new_err(
                "the Mamba component requires positive mamba_cache_chunk_size and page_size",
            ));
        }
        catch_native_panic(|| {
            Ok(Self::from_component(
                MambaComponent::new(&init_params.to_cache_init_params()?),
                is_bigram,
            ))
        })
    }

    #[getter]
    fn component_type(&self) -> u8 {
        match &self.inner {
            ComponentHandle::Plain(component) => component.component_type() as u8,
            ComponentHandle::Bigram(component) => component.component_type() as u8,
        }
    }

    #[getter]
    fn is_bigram(&self) -> bool {
        matches!(self.inner, ComponentHandle::Bigram(_))
    }
}

pub(crate) trait ComponentBindingKey: ChildKeyType {
    fn component_from_binding(binding: &TreeComponentBinding) -> PyResult<NativeComponent<Self>>;
}

impl ComponentBindingKey for Vec<i64> {
    fn component_from_binding(binding: &TreeComponentBinding) -> PyResult<NativeComponent<Self>> {
        match &binding.inner {
            ComponentHandle::Plain(component) => Ok(Arc::clone(component)),
            ComponentHandle::Bigram(_) => Err(PyValueError::new_err(
                "a plain tree requires plain component handles",
            )),
        }
    }
}

impl ComponentBindingKey for Vec<(i64, i64)> {
    fn component_from_binding(binding: &TreeComponentBinding) -> PyResult<NativeComponent<Self>> {
        match &binding.inner {
            ComponentHandle::Bigram(component) => Ok(Arc::clone(component)),
            ComponentHandle::Plain(_) => Err(PyValueError::new_err(
                "a bigram tree requires bigram component handles",
            )),
        }
    }
}
