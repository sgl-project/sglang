// PyTensor newtype for converting between Python torch.Tensor and tch::Tensor.
//
// Inlined from the `pyo3-tch` crate (MIT / Apache-2.0) by Laurent Mazare
// <https://github.com/LaurentMazare/pyo3-tch> to avoid a separate dependency
// and version-compatibility issues. Original license terms apply to this file.
//
// Adapted for pyo3 0.22 API (IntoPy<PyObject> instead of IntoPyObject).

use pyo3::prelude::*;

/// Newtype wrapper around `tch::Tensor` that implements PyO3 conversion traits.
pub struct PyTensor(pub tch::Tensor);

impl std::ops::Deref for PyTensor {
    type Target = tch::Tensor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for PyTensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<tch::Tensor> for PyTensor {
    fn from(t: tch::Tensor) -> Self {
        Self(t)
    }
}

impl<'py> FromPyObject<'py> for PyTensor {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let ptr = ob.as_ptr() as *mut tch::python::CPyObject;
        let tensor = unsafe { tch::Tensor::pyobject_unpack(ptr) };
        match tensor {
            Ok(Some(t)) => Ok(PyTensor(t)),
            Ok(None) => {
                let tp = ob.get_type();
                Err(pyo3::exceptions::PyTypeError::new_err(format!(
                    "expected a torch.Tensor, got {tp}"
                )))
            }
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("{e:?}"))),
        }
    }
}

impl IntoPy<PyObject> for PyTensor {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0.pyobject_wrap() {
            Ok(ptr) => unsafe { PyObject::from_owned_ptr(py, ptr as *mut pyo3::ffi::PyObject) },
            Err(_) => py.None(),
        }
    }
}
