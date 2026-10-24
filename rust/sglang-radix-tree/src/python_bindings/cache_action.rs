//! Python encoding for the common cache actions.

use super::*;
use crate::cache_action::{
    BackupKV, FreeComponentDeviceSlot, FreeComponentHostSlot, FreeDeviceKV, FreeDeviceKVFullOnly,
    MambaEvictExcessPathStates, RebuildFullToSwaMapping, RecoverSwaWithLockedFull, SwaRebuild,
    UpdatePendingBackupOnNodeSplit,
};

impl FreeDeviceKV {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        let indices = PyList::new(py, self.indices.into_iter().map(PyTensor))?;
        (tag, indices).into_py_any(py)
    }
}

impl FreeDeviceKVFullOnly {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        let indices = PyList::new(py, self.indices.into_iter().map(PyTensor))?;
        (tag, indices).into_py_any(py)
    }
}

impl BackupKV {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        (tag, self.node_ids).into_py_any(py)
    }
}

impl UpdatePendingBackupOnNodeSplit {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        (
            tag,
            self.ack_id,
            self.old_node_id,
            self.new_node_id,
            self.new_child_node_id,
        )
            .into_py_any(py)
    }
}

impl MambaEvictExcessPathStates {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        (tag, self.tail_node_id).into_py_any(py)
    }
}

impl FreeComponentDeviceSlot {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        let indices = PyList::new(py, self.indices.into_iter().map(PyTensor))?;
        (tag, component_type_to_u8(self.component_type), indices).into_py_any(py)
    }
}

impl FreeComponentHostSlot {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        let indices = PyList::new(py, self.host_indices.into_iter().map(PyTensor))?;
        (tag, component_type_to_u8(self.component_type), indices).into_py_any(py)
    }
}

impl RebuildFullToSwaMapping {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        let full_indices = PyList::new(py, self.full_indices.into_iter().map(PyTensor))?;
        let swa_indices = PyList::new(py, self.swa_indices.into_iter().map(PyTensor))?;
        (tag, full_indices, swa_indices).into_py_any(py)
    }
}

impl RecoverSwaWithLockedFull {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        (
            tag,
            self.node_id,
            PyTensor(self.kept_full),
            PyTensor(self.incoming_full),
        )
            .into_py_any(py)
    }
}

impl SwaRebuild {
    pub(crate) fn into_py_tagged(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let tag = self.tag();
        (tag, self.node_id, PyTensor(self.source_value)).into_py_any(py)
    }
}
