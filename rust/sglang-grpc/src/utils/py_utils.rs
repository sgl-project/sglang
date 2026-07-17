use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyList};
use std::collections::HashMap;

fn json_value_to_py<'py>(py: Python<'py>, v: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match v {
        serde_json::Value::Null => Ok(py.None()),
        serde_json::Value::Bool(b) => Ok(b.into_pyobject(py)?.to_owned().into_any().unbind()),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(i.into_pyobject(py)?.into_any().unbind())
            } else if let Some(f) = n.as_f64() {
                Ok(f.into_pyobject(py)?.into_any().unbind())
            } else {
                Ok(py.None())
            }
        }
        serde_json::Value::String(s) => Ok(s.into_pyobject(py)?.into_any().unbind()),
        serde_json::Value::Array(arr) => {
            let items: Vec<Py<PyAny>> = arr
                .iter()
                .map(|item| json_value_to_py(py, item))
                .collect::<PyResult<_>>()?;
            let py_list = PyList::new(py, &items)?;
            Ok(py_list.into_any().unbind())
        }
        serde_json::Value::Object(map) => {
            let py_dict = PyDict::new(py);
            for (k, val) in map {
                py_dict.set_item(k, json_value_to_py(py, val)?)?;
            }
            Ok(py_dict.into_any().unbind())
        }
    }
}

pub(crate) fn json_map_to_pydict<'py>(
    py: Python<'py>,
    map: &HashMap<String, serde_json::Value>,
) -> PyResult<Bound<'py, PyDict>> {
    let py_dict = PyDict::new(py);
    for (k, v) in map {
        py_dict.set_item(k, json_value_to_py(py, v)?)?;
    }
    Ok(py_dict)
}

pub(crate) fn py_value_to_json_value(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    match value
        .py()
        .import("json")
        .and_then(|json| json.call_method1("dumps", (value,)))
        .and_then(|json_str| json_str.extract::<String>())
    {
        Ok(serialized) => serde_json::from_str(&serialized)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string())),
        Err(_) => Ok(serde_json::Value::String(value.str()?.to_string())),
    }
}

#[cfg(test)]
mod tests;
